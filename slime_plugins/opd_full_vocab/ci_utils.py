from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu

from .loss import distributed_log_softmax


def gather_vocab_parallel_token_log_probs(
    local_log_probs: torch.Tensor,
    target_tokens: torch.Tensor,
) -> torch.Tensor:
    """Gather token log-probs from vocab-parallel local log-probs."""
    if local_log_probs.dim() != 2:
        raise RuntimeError(f"Expected local_log_probs [T, V/tp], got {tuple(local_log_probs.shape)}")
    if target_tokens.dim() != 1:
        raise RuntimeError(f"Expected target_tokens [T], got {tuple(target_tokens.shape)}")
    if local_log_probs.size(0) != target_tokens.size(0):
        raise RuntimeError(
            f"Token/log-prob length mismatch: log_probs={local_log_probs.size(0)}, tokens={target_tokens.size(0)}"
        )

    tp_group = mpu.get_tensor_model_parallel_group()
    local_vocab_size = local_log_probs.size(-1)
    vocab_start = mpu.get_tensor_model_parallel_rank() * local_vocab_size
    vocab_end = vocab_start + local_vocab_size

    target_tokens = target_tokens.to(device=local_log_probs.device, dtype=torch.long)
    in_partition = (target_tokens >= vocab_start) & (target_tokens < vocab_end)
    local_tokens = (target_tokens - vocab_start).clamp(min=0, max=local_vocab_size - 1)

    token_log_probs = local_log_probs.gather(-1, local_tokens.unsqueeze(-1)).squeeze(-1)
    token_log_probs = torch.where(in_partition, token_log_probs, torch.zeros_like(token_log_probs))
    dist.all_reduce(token_log_probs, op=dist.ReduceOp.SUM, group=tp_group)
    return token_log_probs


@torch.no_grad()
def assert_reconstructed_token_log_probs_close(
    *,
    name: str,
    hidden_states: list[torch.Tensor],
    lm_head_weight: torch.Tensor,
    reference_log_probs: list[torch.Tensor],
    tokens: list[torch.Tensor],
    response_lengths: list[int],
    rollout_temperature: float = 1.0,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    """Assert hidden+lm_head reconstructed token log-probs match callback log-probs."""
    if len(hidden_states) != len(reference_log_probs):
        raise RuntimeError(
            f"{name}: hidden/log-prob count mismatch: {len(hidden_states)} vs {len(reference_log_probs)}"
        )

    lm_head_weight = lm_head_weight.to(device=hidden_states[0].device)
    for i, (hidden, reference, sample_tokens, response_length) in enumerate(
        zip(hidden_states, reference_log_probs, tokens, response_lengths, strict=True)
    ):
        if hidden.dim() != 2:
            raise RuntimeError(f"{name}[{i}]: expected hidden [T_resp, D], got {tuple(hidden.shape)}")

        target_tokens = sample_tokens[-response_length:].to(device=hidden.device, dtype=torch.long)
        if hidden.size(0) != target_tokens.size(0):
            raise RuntimeError(
                f"{name}[{i}]: hidden/token length mismatch: {hidden.size(0)} vs {target_tokens.size(0)}"
            )

        logits = F.linear(hidden, lm_head_weight).float()
        if rollout_temperature != 1.0:
            logits = logits / rollout_temperature
        reconstructed = gather_vocab_parallel_token_log_probs(distributed_log_softmax(logits), target_tokens)
        reference = reference.to(device=hidden.device, dtype=torch.float32)

        torch.testing.assert_close(
            reconstructed,
            reference,
            rtol=rtol,
            atol=atol,
            msg=f"{name}[{i}] reconstructed full-vocab token log-probs differ from callback log_probs",
        )
