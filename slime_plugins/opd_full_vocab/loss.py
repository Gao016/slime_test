"""Differentiable full-vocabulary KL loss for OPD (as auxiliary training loss)."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu


def distributed_log_softmax(local_logits: torch.Tensor) -> torch.Tensor:
    """Distributed log_softmax across TP ranks (no grad, fp32).

    Each TP rank holds [tile, V/tp] logits. Global normalization via
    all_reduce for max and sum_exp.
    """
    tp_group = mpu.get_tensor_model_parallel_group()

    local_max = local_logits.max(dim=-1, keepdim=True).values
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)

    shifted = local_logits - global_max
    local_sum_exp = shifted.exp().sum(dim=-1, keepdim=True)
    global_sum_exp = local_sum_exp.clone()
    dist.all_reduce(global_sum_exp, op=dist.ReduceOp.SUM, group=tp_group)

    return shifted - global_sum_exp.log()


class _VocabParallelLogSoftmax(torch.autograd.Function):
    """Distributed log_softmax with gradient support.

    Forward: all_reduce(MAX) + all_reduce(SUM_EXP) → globally normalized log_softmax.
    Backward: grad_input = grad_output - softmax * all_reduce(sum(grad_output)).
    """

    @staticmethod
    def forward(ctx, local_logits: torch.Tensor) -> torch.Tensor:
        tp_group = mpu.get_tensor_model_parallel_group()

        local_max = local_logits.max(dim=-1, keepdim=True).values
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)

        shifted = local_logits - global_max
        exp_shifted = shifted.exp()
        local_sum_exp = exp_shifted.sum(dim=-1, keepdim=True)
        global_sum_exp = local_sum_exp.clone()
        dist.all_reduce(global_sum_exp, op=dist.ReduceOp.SUM, group=tp_group)

        log_softmax = shifted - global_sum_exp.log()
        softmax = exp_shifted / global_sum_exp

        ctx.save_for_backward(softmax)
        return log_softmax

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (softmax,) = ctx.saved_tensors
        tp_group = mpu.get_tensor_model_parallel_group()

        local_sum = grad_output.sum(dim=-1, keepdim=True)
        global_sum = local_sum.clone()
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=tp_group)

        return grad_output - softmax * global_sum


def vocab_parallel_log_softmax(local_logits: torch.Tensor) -> torch.Tensor:
    """Autograd-aware distributed log_softmax for student logits with gradient."""
    return _VocabParallelLogSoftmax.apply(local_logits)


def _compute_tile_kl(s_lp: torch.Tensor, t_lp: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute local partial KL for one tile given student/teacher log-probs."""
    if alpha == 0.0:
        return (t_lp.exp() * (t_lp - s_lp)).sum(dim=-1)
    elif alpha == 1.0:
        return (s_lp.exp() * (s_lp - t_lp)).sum(dim=-1)
    else:
        m_lp = torch.logaddexp(
            s_lp + math.log(1 - alpha),
            t_lp + math.log(alpha),
        )
        return (1 - alpha) * (s_lp.exp() * (s_lp - m_lp)).sum(dim=-1) + alpha * (t_lp.exp() * (t_lp - m_lp)).sum(
            dim=-1
        )


def _build_response_indices(
    total_lengths: list[int],
    response_lengths: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Build flat index tensor for response-aligned logit positions.

    Off-by-one: position t-1 predicts token t. For response tokens at positions
    [prompt_len, ..., total_len-1], logits come from positions [prompt_len-1, ..., total_len-2].

    Returns: [T_resp_total] int64 tensor of indices into logits_flat.
    """
    resp_indices = []
    offset = 0
    for total_len, resp_len in zip(total_lengths, response_lengths, strict=True):
        start = offset + (total_len - resp_len) - 1
        resp_indices.extend(range(start, start + resp_len))
        offset += total_len
    return torch.tensor(resp_indices, device=device, dtype=torch.long)


def compute_full_vocab_kl_loss(
    student_logits: torch.Tensor,
    teacher_hs: list[torch.Tensor],
    teacher_weight: torch.Tensor,
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    tile_size: int = 128,
    alpha: float = 0.0,
) -> torch.Tensor:
    """Differentiable full-vocab KL loss.

    Uses flat tiling over response positions with per-tile index-select on student logits.
    Each TP rank computes partial KL over its V/tp shard — no all_reduce on KL values.
    Gradient flows through student log_softmax via vocab_parallel_log_softmax backward.
    """
    logits_flat = student_logits.squeeze(0)  # [T_padded, V/tp], view, has grad

    # 1. Precompute response position indices
    resp_indices = _build_response_indices(total_lengths, response_lengths, logits_flat.device)

    # 2. Cat teacher hs and loss_masks
    teacher_hs_flat = torch.cat(teacher_hs, dim=0)  # [T_resp_total, D]
    teacher_weight = teacher_weight.to(device=teacher_hs_flat.device)
    loss_mask_flat = torch.cat(loss_masks, dim=0)  # [T_resp_total]

    T_resp = resp_indices.size(0)

    # 3. Flat tile loop — only [tile_size, V/tp] allocated per tile
    kl_tiles = []
    for t in range(0, T_resp, tile_size):
        t_end = min(t + tile_size, T_resp)
        tile_idx = resp_indices[t:t_end]

        # Student: index-select creates [tile, V/tp] copy with grad
        s_logits_tile = logits_flat[tile_idx]
        s_lp = vocab_parallel_log_softmax(s_logits_tile)

        # Teacher: reconstruct from hs (no grad)
        with torch.no_grad():
            t_logits = F.linear(teacher_hs_flat[t:t_end], teacher_weight).float()
            t_lp = distributed_log_softmax(t_logits)

        # Local partial KL (V/tp terms only — DO NOT all_reduce)
        kl_tiles.append(_compute_tile_kl(s_lp, t_lp, alpha))

    # 4. Concat preserves autograd + masked mean
    kl_per_token = torch.cat(kl_tiles)
    return (kl_per_token * loss_mask_flat).sum() / loss_mask_flat.sum().clamp(min=1)


def _build_response_indices_for_subset(
    all_total_lengths: list[int],
    all_response_lengths: list[int],
    assigned_indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Build response position indices for a subset of samples using global offsets.

    Indices are GLOBAL (into the full packed logits_flat), because student
    logits come from the full microbatch regardless of teacher assignment.
    """
    offsets = [0]
    for tl in all_total_lengths:
        offsets.append(offsets[-1] + tl)

    resp_indices = []
    for idx in assigned_indices:
        total_len = all_total_lengths[idx]
        resp_len = all_response_lengths[idx]
        sample_offset = offsets[idx]
        start = sample_offset + (total_len - resp_len) - 1
        resp_indices.extend(range(start, start + resp_len))

    return torch.tensor(resp_indices, device=device, dtype=torch.long)


def compute_multi_teacher_kl_loss(
    student_logits: torch.Tensor,
    teacher_hs: list[torch.Tensor],
    teacher_ids: list[int],
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    tile_size: int = 128,
    alpha: float = 0.0,
) -> torch.Tensor:
    """Multi-teacher full-vocab KL: per-teacher outer loop with global resp_indices.

    Groups samples by teacher_id, computes KL for each group using
    the corresponding teacher's lm_head weight from TeacherHeadRegistry,
    then normalizes by total active tokens across all teachers.
    """
    from .registry import TeacherHeadRegistry

    logits_flat = student_logits.squeeze(0)
    unique_tids = sorted(set(teacher_ids))

    total_kl_sum = torch.tensor(0.0, device=logits_flat.device)
    total_mask_sum = torch.tensor(0.0, device=logits_flat.device)

    for tid in unique_tids:
        tag = f"teacher_{tid}"
        teacher_weight = TeacherHeadRegistry.get(tag)
        if teacher_weight is None:
            continue

        assigned = [i for i, t in enumerate(teacher_ids) if t == tid]

        resp_indices = _build_response_indices_for_subset(
            total_lengths, response_lengths, assigned, logits_flat.device
        )
        teacher_hs_flat = torch.cat([teacher_hs[i] for i in assigned], dim=0)
        teacher_weight_dev = teacher_weight.to(device=teacher_hs_flat.device)
        mask_flat = torch.cat([loss_masks[i] for i in assigned], dim=0)

        T_resp = resp_indices.size(0)
        kl_tiles = []

        for t in range(0, T_resp, tile_size):
            t_end = min(t + tile_size, T_resp)
            tile_idx = resp_indices[t:t_end]

            s_logits_tile = logits_flat[tile_idx]
            s_lp = vocab_parallel_log_softmax(s_logits_tile)

            with torch.no_grad():
                t_logits = F.linear(teacher_hs_flat[t:t_end], teacher_weight_dev).float()
                t_lp = distributed_log_softmax(t_logits)

            kl_tiles.append(_compute_tile_kl(s_lp, t_lp, alpha))

        kl_per_token = torch.cat(kl_tiles)
        total_kl_sum = total_kl_sum + (kl_per_token * mask_flat).sum()
        total_mask_sum = total_mask_sum + mask_flat.sum()

    return total_kl_sum / total_mask_sum.clamp(min=1)
