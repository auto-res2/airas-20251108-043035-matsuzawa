# src/model.py
"""Model construction utilities: build energy-aware HuggingFace models with
HEST or TABS adapters injected into every transformer layer.
"""
from __future__ import annotations

import types
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel

# -----------------------------------------------------------------------------
# Adapter building blocks ------------------------------------------------------
# -----------------------------------------------------------------------------

class FactorisedController(nn.Module):
    """Rank-1 factorised gating controller g(h_{ℓ,t})."""

    def __init__(self, hidden_dim: int, num_layers: int, k: int):
        super().__init__()
        self.W_tok = nn.Linear(hidden_dim, k, bias=False)
        self.W_layer = nn.Parameter(torch.zeros(num_layers, k))

    def forward(self, h: torch.Tensor, layer_id: int):  # h:(B,T,d)
        s_tok = torch.sigmoid(self.W_tok(h))  # (B,T,k)
        s_layer = torch.sigmoid(self.W_layer[layer_id])  # (k,)
        gate = (s_tok * s_layer).mean(-1)  # (B,T)
        return gate  # ∈ [0,1]


class BaseSpectralAdapter(nn.Module):
    """Low-rank spectral adapter with trainable singular values."""

    def __init__(self, hidden_dim: int, rank: int, init_std: float):
        super().__init__()
        self.U = nn.Parameter(torch.randn(hidden_dim, rank) * init_std)
        self.V = nn.Parameter(torch.randn(rank, hidden_dim) * init_std)
        self.S = nn.Parameter(torch.ones(rank))

    def _spectral_forward(self, x: torch.Tensor):  # x:(B,T,d)
        proj = torch.matmul(x, self.U) * self.S  # (B,T,R)
        return torch.matmul(proj, self.V)  # (B,T,d)


class HESTAdapter(BaseSpectralAdapter):
    """Precision-aware hierarchical energy-constrained adapter."""

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        layer_id: int,
        num_layers: int,
        k: int,
        energy_fp16: float = 1.0,
        energy_int8: float = 0.5,
        tau: float = 2 / 3,
        init_std: float = 0.02,
    ):
        super().__init__(hidden_dim, rank, init_std)
        self.ctrl = FactorisedController(hidden_dim, num_layers, k)
        self.energy_fp16 = energy_fp16
        self.energy_int8 = energy_int8
        self.tau = tau
        self.layer_id = layer_id
        self.current_lambda = torch.tensor(0.0)
        self.last_energy = torch.tensor(0.0)

    def set_lambda(self, lam: torch.Tensor | float):
        self.current_lambda = lam if isinstance(lam, torch.Tensor) else torch.tensor(lam)

    def forward(self, x: torch.Tensor):  # x:(B,T,d)
        gate = self.ctrl(x, self.layer_id).unsqueeze(-1).expand(-1, -1, self.S.numel())
        logits = torch.log(gate + 1e-6).unsqueeze(-1).expand(-1, -1, -1, 3)
        gumbel = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)  # (B,T,R,3)
        m_off, m_int8, m_fp16 = gumbel.unbind(-1)
        self.last_energy = ((m_int8 * self.energy_int8) + (m_fp16 * self.energy_fp16)).mean()
        out = self._spectral_forward(x)
        out = out * (m_int8 + m_fp16)  # deactivate OFF directions
        return out


class TABSAdapter(BaseSpectralAdapter):
    """Baseline TABS adapter (token-wise spectal gating, fp16-only)."""

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        layer_id: int,
        num_layers: int,
        tau: float = 2 / 3,
        init_std: float = 0.02,
    ):
        super().__init__(hidden_dim, rank, init_std)
        self.ctrl = FactorisedController(hidden_dim, num_layers, 1)
        self.tau = tau
        self.layer_id = layer_id
        self.last_energy = torch.tensor(0.0)
        self.current_lambda = torch.tensor(0.0)

    def set_lambda(self, lam):
        # Baseline ignores λ but we store it for completeness.
        self.current_lambda = lam if isinstance(lam, torch.Tensor) else torch.tensor(lam)

    def forward(self, x: torch.Tensor):
        gate = self.ctrl(x, self.layer_id)  # (B,T)
        mask = (gate > 0.5).float().unsqueeze(-1)  # hard gate
        self.last_energy = mask.mean()  # proxy energy consumption
        return self._spectral_forward(x) * mask


# -----------------------------------------------------------------------------
# Injection helpers ------------------------------------------------------------
# -----------------------------------------------------------------------------

def _patch_layer(layer: nn.Module, adapter: nn.Module, energy_records: List[torch.Tensor]):
    """Monkey-patch *layer.forward* to add adapter output and record energy."""

    original_forward = layer.forward

    def new_forward(self, *args, **kwargs):  # noqa: D401
        hidden_states = original_forward(*args, **kwargs)
        if isinstance(hidden_states, tuple):
            main_out, *rest = hidden_states
        else:
            main_out, rest = hidden_states, []

        main_out = main_out + adapter(main_out)
        energy_records.append(adapter.last_energy.detach())

        return (main_out, *rest) if rest else main_out

    layer.forward = types.MethodType(new_forward, layer)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Energy-aware wrapper ---------------------------------------------------------
# -----------------------------------------------------------------------------

class EnergyAwareModel(PreTrainedModel):
    def __init__(self, base: PreTrainedModel, adapters: List[nn.Module], energy_records: List[torch.Tensor]):
        super().__init__(base.config)
        self.base = base
        self.adapters = nn.ModuleList(adapters)
        self._energy_records = energy_records  # shared list

    def set_lambda(self, lam):
        for adp in self.adapters:
            if hasattr(adp, "set_lambda"):
                adp.set_lambda(lam)

    @property
    def energy(self):
        if not self._energy_records:
            return torch.tensor(0.0, device=self.base.device)
        return torch.stack(self._energy_records).mean()

    def forward(self, *args, **kwargs):  # noqa: D401
        self._energy_records.clear()
        out = self.base(*args, **kwargs)
        out.energy = self.energy  # attach attribute for trainer
        return out

    def generate(self, *args, **kwargs):  # proxy to base
        return self.base.generate(*args, **kwargs)


# -----------------------------------------------------------------------------
# Public factory ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_energy_aware_model(cfg, tokenizer):
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        cache_dir=".cache/",
        torch_dtype=(torch.float16 if cfg.training.precision == "fp16" else torch.float32),
        device_map="auto",
        attn_implementation="eager",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        base.resize_token_embeddings(len(tokenizer))

    adapters: List[nn.Module] = []
    energy_records: List[torch.Tensor] = []

    # Access model layers generically (works for Llama, GPT-NeoX, etc.)
    try:
        layer_list = base.model.layers  # type: ignore[attr-defined]
    except AttributeError:  # fallback for models that use 'transformer.h'
        layer_list = base.model.transformer.h  # type: ignore[attr-defined]

    for idx, layer in enumerate(layer_list):
        if cfg.model.adapter == "HEST":
            adapter = HESTAdapter(
                hidden_dim=cfg.model.hidden_size,
                rank=cfg.model.spectral_rank_R,
                layer_id=idx,
                num_layers=cfg.model.num_layers,
                k=cfg.model.controller_width_k,
                tau=cfg.training.gumbel_temperature_tau,
                init_std=cfg.model.init_std,
            )
        else:
            adapter = TABSAdapter(
                hidden_dim=cfg.model.hidden_size,
                rank=cfg.model.spectral_rank_R,
                layer_id=idx,
                num_layers=cfg.model.num_layers,
                tau=cfg.model.get("gating_temperature_tau", 0.67),
                init_std=cfg.model.init_std,
            )
        adapters.append(adapter)
        _patch_layer(layer, adapter, energy_records)

    return EnergyAwareModel(base, adapters, energy_records)
