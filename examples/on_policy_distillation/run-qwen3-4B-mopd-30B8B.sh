#!/bin/bash

# Two-teacher heterogeneous full-vocab MOPD (full-length stress test, stage 1.1).
# Teachers: Qwen3-30B-A3B (128-expert MoE) + Qwen3-8B (dense), independent resident
#           models, torch_dist PP=1. The 30B MoE teacher's build normalizes moe_layer_freq
#           to a list so its per-layer torch_dist _extra_state loads (see model_provider.py).
# Student:  Qwen3-4B (dense)
# REVERSE KL (alpha=1.0, mode-seeking), full-vocab KL as auxiliary loss.
# Uses 30B+8B instead of 32B+30B because two large teachers colocated OOM a 140GB card.
# max-tokens-per-gpu lowered to 8192 (vs 32B+8B's 10240) for headroom with the MoE teacher.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

NUM_GPUS=8

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

MODEL_DIR="/mnt/amed-s1/common/ckpt/gaochang"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B/
   --ref-load ${MODEL_DIR}/Qwen3-4B/
   # --save ${MODEL_DIR}/Qwen3-4B-OPD-hetero-30b8b-output/
   # --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /personal/data/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler
   --num-rollout 300
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data

   # Eval: accuracy on aime-2024 (deepscaler reward=1 => correct). resp_len must be
   # large enough for Qwen3 thinking to finish <think>...</think> AND emit \boxed{};
   # 2048 truncates mid-think (100% truncated -> acc always 0). Match train len 8192.
   --eval-interval 10
   --eval-prompt-data aime-2024 /personal/data/aime-2024.jsonl
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 8192
   --eval-max-prompt-len 2048
   --eval-temperature 0.6
   --eval-top-p 0.95
   --eval-top-k 20
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # OPD + full-vocab KL, heterogeneous teachers
   --use-opd
   --opd-type megatron
   --opd-kl-coef 1.0
   --opd-full-vocab-kl
   --opd-full-vocab-kl-alpha 1.0
   --opd-full-vocab-kl-coef 1.0
   --opd-full-vocab-kl-tile-size 128
   --opd-hetero-teacher

   # Dual heterogeneous teachers (Qwen3-30B-A3B MoE + Qwen3-8B dense)
   --megatron-config-path ${SCRIPT_DIR}/mopd_config_30B8B.yaml
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
