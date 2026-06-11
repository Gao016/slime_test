import aiohttp
import torch

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    top_k = getattr(args, "opd_top_k", 0)

    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if top_k > 0:
        # Top-K OPD: request teacher log-probs on the union of the student's top-K
        # token ids (across all positions), so we can later gather T_on_S — the
        # teacher's log-prob on the *student's* top-K tokens (token-aligned KL).
        # SGLang's token_ids_logprob takes one fixed id set applied to every position;
        # we build that set as the per-sample union and gather per-position in
        # post_process_rewards. See OPD paper only_stu (kl = S_logp - T_on_S).
        if sample.student_top_k_ids is not None:
            # Build the union of student top-K ids. Exclude None and the padding id 0:
            # SGLang's token_ids_logprob rejects a None entry (torch.tensor(...) raises),
            # and padded slots carry log-prob -inf so they are masked downstream anyway.
            union_ids = sorted(
                {tid for pos in sample.student_top_k_ids for tid in pos if tid is not None and tid != 0}
            )
            payload["token_ids_logprob"] = union_ids
        else:
            # student top-K missing (shouldn't happen for OPD rollout); fall back to
            # teacher's own top-K so post-processing can still run.
            payload["top_logprobs_num"] = top_k

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. When opd_top_k > 0, also extracts teacher top-K token IDs and log-probs
    5. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]
    top_k = getattr(args, "opd_top_k", 0)

    if top_k > 0:
        # Top-K OPD: extract teacher log-probs on the student's top-K tokens (T_on_S).
        # reward_func requested token_ids_logprob = per-sample union of student top-K ids.
        # SGLang returns meta_info["input_token_ids_logprobs"]:
        #   itil[t] = [[lp, token_id, None], ...] for each input position t,
        #   itil[0] = None (no log-prob for the first token), every position uses the
        #   same union id order. We build {token_id: col} once and gather, per position,
        #   the teacher log-prob for each student top-K token id.
        for sample, reward, response_length in zip(samples, raw_rewards, response_lengths, strict=False):
            meta = reward["meta_info"]
            student_top_k_ids = sample.student_top_k_ids  # list[list[int]], len == response_length

            # Teacher's per-token sampled log-prob (kept for compat / logging).
            sampled = [item[0] for item in meta["input_token_logprobs"][1:]][-response_length:]
            sample.teacher_log_probs = torch.tensor(sampled, dtype=torch.float32)

            itil = meta.get("input_token_ids_logprobs")
            if itil is None or student_top_k_ids is None:
                # Fallback: no token_ids_logprob data; leave T_on_S unset (loss will raise).
                continue

            # Drop the leading None (off-by-one), then align to the response tail.
            resp_itil = itil[1:][-response_length:]

            # Column lookup from the (position-invariant) union id order.
            # Use the first valid row's ids; they equal the requested union_ids.
            first_valid = next((row for row in resp_itil if row), None)
            if first_valid is None:
                continue
            col_of = {cell[1]: j for j, cell in enumerate(first_valid)}

            t_on_s = []
            for t, pos_ids in enumerate(student_top_k_ids):
                row = resp_itil[t]
                row_lps = [cell[0] for cell in row]  # log-probs in union order
                t_on_s.append([row_lps[col_of[tid]] if tid in col_of else float("-inf") for tid in pos_ids])

            sample.teacher_on_student_log_probs = torch.tensor(t_on_s, dtype=torch.float32)
    else:
        # Sampled-token OPD (original behavior)
        teacher_log_probs = [
            torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
            for reward in raw_rewards
        ]
        teacher_log_probs = [
            t_log_prob[-response_length:]
            for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
        ]
        for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
            sample.teacher_log_probs = t_log_probs

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
