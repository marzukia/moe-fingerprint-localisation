"""MoE expert activation hooks for Qwen2MoE-style architectures.

Qwen2MoE uses `mlp` (Qwen2MoeSparseMoeBlock) with:
- 60 routed experts per layer, top-4 routing
- Gate: Qwen2MoeTopKRouter with weight [num_experts, hidden_dim]
- Experts stored as concatenated weight tensors
- Shared expert: Qwen2MoeMLP
"""

import numpy as np
import torch


class ExpertActivationTracer:
    """Captures expert routing scores per token per layer."""

    def __init__(self):
        self.activations = {}  # {layer_idx: np.array(total_tokens, num_experts)}
        self._hooks = []

    def register_hooks(self, model):
        """Attach hooks to all MoE layers."""
        for idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                self.activations[idx] = None
                hook = layer.mlp.register_forward_hook(self._make_hook(idx))
                self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook(module, inputs, outputs):
            hidden_states = inputs[0]
            gate = module.gate
            gate_output = gate(hidden_states)
            if isinstance(gate_output, tuple):
                logits = gate_output[0]
            else:
                logits = gate_output

            scores = torch.softmax(logits, dim=-1)
            arr = scores.float().cpu().numpy()
            flat = arr.reshape(-1, arr.shape[-1])

            if self.activations[layer_idx] is None:
                self.activations[layer_idx] = flat
            else:
                self.activations[layer_idx] = np.concatenate(
                    [self.activations[layer_idx], flat], axis=0
                )

        return hook

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def get_distribution(self, layer_idx, expert_idx):
        """Return activation score distribution for (layer, expert)."""
        if layer_idx not in self.activations:
            return None
        arr = self.activations[layer_idx]
        if arr is None:
            return None
        return arr[:, expert_idx]
