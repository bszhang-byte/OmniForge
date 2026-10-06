"""Read one random window per episode, matching upstream EgoDex sampling."""
from pathlib import Path
import random
import json

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torchcodec.decoders import VideoDecoder

from data.dataset import TransformableDataset


def wrist_matrix(nine):
    matrix = np.eye(4, dtype=nine.dtype)
    col0 = nine[3:6] / (np.linalg.norm(nine[3:6]) + 1e-8)
    col1 = nine[6:9] - np.dot(nine[6:9], col0) * col0
    col1 = col1 / (np.linalg.norm(col1) + 1e-8)
    matrix[:3, :3] = np.stack((col0, col1, np.cross(col0, col1)), axis=1)
    matrix[:3, 3] = nine[:3]
    return matrix


def delta_actions(actions, camera):
    """Consecutive deltas in the observation camera frame; RPY + zero padding."""
    inverse = np.linalg.inv(camera)
    hands = []
    for base in (0, 24):
        wrists = np.stack([wrist_matrix(a[base:base + 9]) for a in actions])
        wrists = inverse[None] @ wrists
        xyz = np.diff(wrists[:, :3, 3], axis=0)
        rotations = wrists[1:, :3, :3] @ wrists[:-1, :3, :3].transpose(0, 2, 1)
        rpy = Rotation.from_matrix(rotations).as_euler("xyz").astype(np.float32)
        tips = actions[:, base + 9:base + 24].reshape(-1, 3)
        homogeneous = np.hstack((tips, np.ones((len(tips), 1))))
        tips = (inverse @ homogeneous.T).T[:, :3].reshape(len(actions), 15)
        hands.extend((xyz, rpy, np.zeros_like(rpy), np.diff(tips, axis=0)))
    return np.concatenate(hands, axis=1)


class EgoDexDataset(TransformableDataset):
    def __init__(self, config, transform=None):
        self.cfg = config
        self.transform = transform
        self._action_bounds = None
        if getattr(config, "stat_path", "") and Path(config.stat_path).is_file():
            stats=json.loads(Path(config.stat_path).read_text())
            key=getattr(config, "stat_action_key", "egodex")
            entry=stats.get(key, stats.get("action", stats))
            lo=entry.get("q01", entry.get("min")); hi=entry.get("q99", entry.get("max"))
            if lo is not None and hi is not None: self._action_bounds=(np.asarray(lo,np.float32),np.asarray(hi,np.float32))
        root = Path(config.root_dir)
        parts = ("part1", "part2", "part3", "part4", "part5", "extra") if config.split == "train" else (config.split,)
        self.files = sorted(p for part in parts for p in (root / part).glob("*/*.hdf5")
                            if p.with_suffix(".mp4").is_file())
        if config.max_episodes:
            self.files = self.files[:config.max_episodes]
        if not self.files:
            raise FileNotFoundError(f"No paired EgoDex hdf5/mp4 files: {root}, {parts}")

    def __len__(self):
        return len(self.files)

    def read_episode(self, episode_index, frame_index=None):
        path = self.files[episode_index]
        cfg = self.cfg
        with h5py.File(path, "r") as root:
            total_frames = len(next(iter(root["transforms"].values())))
            if total_frames < 2:
                raise ValueError(f"Episode has fewer than two frames: {path}")
            index = random.randint(0, total_frames - 2) if frame_index is None else frame_index
            if not 0 <= index <= total_frames - 2:
                raise IndexError(f"frame_index={index}, valid range 0..{total_frames - 2}: {path}")
            attrs = root.attrs
            decode = lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x)
            reverse = decode(attrs.get("llm_type", "")) == "reversible"
            key = "llm_description2" if reverse and decode(attrs["which_llm_description"]) != "1" else "llm_description"
            instruction = decode(attrs[key]).strip()
            states = root["actions_48d"][index:index + 1].astype(np.float32)
            count = cfg.chunk_size + int(cfg.use_delta_actions)
            end = min(index + count * cfg.upsample_rate, total_frames - 1)
            start = index if cfg.use_delta_actions else index + cfg.upsample_rate
            indices = list(range(start, end, cfg.upsample_rate))
            while len(indices) < count:
                indices.append(indices[-1] if indices else index + 1)
            # numpy indexing supports repeated tail indices, unlike h5py fancy indexing.
            actions = root["actions_48d"][:][indices[:count]].astype(np.float32)
            if cfg.use_delta_actions:
                actions = delta_actions(actions, root["transforms/camera"][index])
            if self._action_bounds is not None:
                lo,hi=self._action_bounds; span=np.where(np.abs(hi-lo)<1e-4*(np.abs(hi)+np.abs(lo)+1e-8),1.0,hi-lo)
                actions=np.clip((actions-lo)/span*2.0-1.0,-1.0,1.0).astype(np.float32)
        # Preserve upstream img_history_size=1 behavior: frame index-2 at rate=3.
        image_index = max(index - cfg.upsample_rate + 1, 0)
        decoder = VideoDecoder(str(path.with_suffix(".mp4")), device="cpu", dimension_order="NHWC")
        image = decoder[image_index].numpy()
        data = dict(states=states, actions=actions, current_images=[image[None]],
                    instruction=instruction, dataset_name="egodex",
                    episode_index=episode_index, frame_index=index, image_index=image_index,
                    action_indices=indices[:count], source=str(path))
        return data

    def __getitem__(self, index):
        fixed = self.cfg.frame_index if self.cfg.frame_index >= 0 else None
        sample = self.read_episode(index, fixed)
        return self.transform(sample) if self.transform else sample
