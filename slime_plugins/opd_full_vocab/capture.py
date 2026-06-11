from __future__ import annotations

import torch


class HiddenCapture:
    def __init__(self):
        self._queue: list[torch.Tensor] = []

    def push(self, hidden: torch.Tensor):
        self._queue.append(hidden)

    def pop(self) -> torch.Tensor:
        return self._queue.pop(0)


def get_lm_head_weight(model) -> torch.Tensor:
    """Get the output layer weight, handling weight-tied PP=1 case.

    When share_embeddings_and_output_weights=True and PP=1, output_layer.weight
    is None (skip_weight_param_allocation). Falls back to embedding weight.
    Returns a cloned [V/tp, D] tensor.
    """
    from megatron.core.utils import unwrap_model

    for module in unwrap_model(model):
        output_layer = getattr(module, "output_layer", None)
        if output_layer is None and hasattr(module, "language_model"):
            module = module.language_model
            output_layer = getattr(module, "output_layer", None)
        if output_layer is not None:
            if output_layer.weight is not None:
                return output_layer.weight.data.clone()
            if hasattr(module, "shared_embedding_or_output_weight"):
                return module.shared_embedding_or_output_weight().data.clone()
            if hasattr(module, "embedding"):
                return module.embedding.word_embeddings.weight.data.clone()
    raise RuntimeError("Cannot locate lm_head weight on this pipeline stage")


def register_capture_hook(model, args, hidden_capture: HiddenCapture):
    """Register a forward pre_hook on output_layer to capture hidden states.

    Handles SP allgather so the captured tensor is always full-sequence [1, T, D].
    Returns the hook handle (caller MUST remove it after forward completes).
    Raises if output_layer is not found on this stage.
    """
    from megatron.core import tensor_parallel
    from megatron.core.utils import unwrap_model

    for module in unwrap_model(model):
        output_layer = getattr(module, "output_layer", None)
        if output_layer is None and hasattr(module, "language_model"):
            output_layer = getattr(module.language_model, "output_layer", None)
        if output_layer is None:
            continue

        def _capture_hook(module, input):
            hs = input[0].detach()
            if getattr(args, "sequence_parallel", False):
                hs = tensor_parallel.gather_from_sequence_parallel_region(hs, tensor_parallel_output_grad=False)
            hidden_capture.push(hs)

        return output_layer.register_forward_pre_hook(_capture_hook)
    raise RuntimeError("Cannot locate output_layer for OPD hidden capture on this pipeline stage")
