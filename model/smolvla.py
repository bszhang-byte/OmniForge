from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class SmolVLAAdapter(nn.Module):
    """Adapter for Hugging Face/LeRobot's public SmolVLA checkpoint.

    Psi0_zbs samples contain 48-D EgoDex actions and one image. The released
    SmolVLA checkpoint is configured for a 6-D LeRobot action/state interface,
    so this smoke adapter explicitly uses the first six dimensions and pads
    the one-frame sample to SmolVLA's action chunk length.
    """

    def __init__(self, cfg):
        super().__init__()
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as exc:
            raise ImportError(
                "SmolVLA requires the LeRobot package (pip install 'lerobot[smolvla]')."
            ) from exc

        checkpoint = getattr(cfg, "model_name_or_path", "")
        if not checkpoint:
            raise ValueError("model_name_or_path must point to lerobot/smolvla_base")
        # The Hub checkpoint ships action statistics for SO-100, but no
        # EgoDex state statistics. Use the released action statistics when
        # available and identity statistics for the six-dimensional smoke
        # projection used by this adapter.
        stats = {
            "action": {"mean": torch.zeros(6), "std": torch.ones(6)},
            "observation.state": {"mean": torch.zeros(6), "std": torch.ones(6)},
        }
        stats_file = Path(checkpoint) / "policy_preprocessor_step_5_normalizer_processor.safetensors"
        if stats_file.exists():
            try:
                from safetensors import safe_open
                with safe_open(str(stats_file), framework="pt") as handle:
                    mean_key = "so100.buffer.action.mean"
                    std_key = "so100.buffer.action.std"
                    if mean_key in handle.keys() and std_key in handle.keys():
                        stats["action"] = {
                            "mean": handle.get_tensor(mean_key)[:6].float(),
                            "std": handle.get_tensor(std_key)[:6].float().clamp_min(1e-6),
                        }
            except Exception:
                pass
        self.policy = SmolVLAPolicy.from_pretrained(
            checkpoint,
            dataset_stats=stats,
            local_files_only=bool(getattr(cfg, "local_files_only", True)),
        )
        self.action_dim = 6
        self.state_dim = 6
        self.chunk_size = int(getattr(self.policy.config, "chunk_size", 50))
        self.camera_key = "observation.images.camera1"

    @staticmethod
    def _images(value: Any, device: torch.device) -> torch.Tensor:
        # EgoDex DataLoader output is a list of B tensors shaped [T,H,W,3].
        rows = []
        for item in value:
            while isinstance(item, (list, tuple)):
                item = item[0]
            if not torch.is_tensor(item):
                item = torch.as_tensor(item)
            while item.ndim > 3:
                item = item[-1]
            if item.ndim != 3:
                raise ValueError(f"Expected one image [H,W,C], got {tuple(item.shape)}")
            if item.shape[-1] == 3:
                item = item.permute(2, 0, 1)
            rows.append(item)
        images = torch.stack(rows).to(device=device, dtype=torch.float32)
        if images.max() > 1.5:
            images = images / 255.0
        return images

    def _make_batch(self, batch):
        device = next(self.parameters()).device
        images = self._images(batch["current_images"], device)
        state = batch["states"].to(device=device, dtype=torch.float32)
        actions = batch["actions"].to(device=device, dtype=torch.float32)
        if state.ndim == 3:
            state = state[:, -1]
        if actions.ndim == 3:
            actions = actions[:, :1]
        state = state[..., : self.state_dim]
        actions = actions[..., : self.action_dim]
        actions = actions.expand(-1, self.chunk_size, -1).contiguous()
        task = batch["instruction"]
        if isinstance(task, str):
            task = [task] * images.shape[0]
        return {
            self.camera_key: images,
            "observation.state": state,
            "action": actions,
            "task": list(task),
        }

    def compute_loss(self, batch):
        out = self.policy(self._make_batch(batch))
        if isinstance(out, dict):
            loss = out.get("loss")
            if loss is None:
                loss = out.get("losses")
        elif isinstance(out, (tuple, list)):
            loss = out[0]
        else:
            loss = out
        if loss is None or not torch.is_tensor(loss):
            raise RuntimeError(f"SmolVLA returned an unexpected loss object: {type(loss)}")
        return loss.mean() if loss.ndim else loss

    def forward(self, batch):
        return self.compute_loss(batch)
