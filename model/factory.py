from model.fast_causal import FastCausalModel
from model.psi0_flow import Psi0FlowAdapter
from model.smolvla import SmolVLAAdapter


def build_model(cfg):
    family = getattr(cfg, "model_family", "qwen3vl_fast")
    if family in ("qwen3vl_fast", "fast_causal", "qwen"):
        return FastCausalModel(cfg)
    if family in ("psi0_flow", "flow", "psi0"):
        return Psi0FlowAdapter(cfg)
    if family in ("smolvla", "smol_vla"):
        return SmolVLAAdapter(cfg)
    raise ValueError(
        f"Unknown model_family={family}; expected qwen3vl_fast, psi0_flow, or smolvla"
    )
