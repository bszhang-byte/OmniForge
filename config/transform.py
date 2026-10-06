from pydantic import BaseModel, Field, PrivateAttr
from typing import List, Any, Tuple
import random
import numpy as np

# 数据如何变换？DataTransform的3个组件（如下3个）
class RepackTransform(BaseModel):
    dataset_name: str = "default"
    goal_sampling: bool = False
    goal_future_offsets: List[float] = Field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0])
    goal_end_segment_prob: float = 0.5

    # When True and meta/paraphrases.json was loaded for this dataset, render the
    # task/subtask prompt from a random member of {canonical}+full+simple per sample.
    use_paraphrases: bool = False
    _para_lookup: dict | None = PrivateAttr(default=None)

    def set_paraphrases(self, table: dict | None) -> None:
        if not table:
            self._para_lookup = None
            return
        lut: dict[str, dict[str, list[str]]] = {}
        for section in ("tasks", "subtasks"):
            lut[section] = {
                str(canon).strip().lower(): [str(canon), *v.get("full", []), *v.get("simple", [])]
                for canon, v in (table.get(section) or {}).items()
            }
        self._para_lookup = lut

    def _maybe_paraphrase(self, text: str, section: str, is_training: bool) -> str:
        if not (self.use_paraphrases and is_training) or self._para_lookup is None or not text:
            return text
        cands = self._para_lookup.get(section, {}).get(str(text).strip().lower())
        return random.choice(cands) if cands else text

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data
    
class ModelTransform(BaseModel):
    def __call__(self, data: dict[str, Any], no_aug: bool = False, **kwargs) -> dict[str, Any]:
        return data

class FieldTransform(BaseModel):

    @staticmethod
    def _bounds(field_transform) -> Tuple[np.ndarray, np.ndarray]:
        los, his = [], []
        # modality_dims: dict[str, int] = {}
        for action_key in field_transform.action_transform.apply_to:
            mode = field_transform.action_transform.normalization_modes.get(action_key, "min_max")
            stats = field_transform.action_transform.normalization_statistics.get(action_key, {})
            if not stats:
                continue
            if mode in ("min_max", "bounds"):
                lo = np.array(stats["min"], dtype=np.float32)
                hi = np.array(stats["max"], dtype=np.float32)
            elif mode in ("q99", "bounds_q99"):
                lo = np.array(stats["q01"], dtype=np.float32)
                hi = np.array(stats["q99"], dtype=np.float32)
            elif mode == "mean_std":
                std = np.array(stats["std"], dtype=np.float32)
                mean = np.array(stats["mean"], dtype=np.float32)
                lo, hi = mean - std, mean + std
            else:
                raise NotImplementedError(f"Unsupported normalization mode for denormalization: {mode}")
            los.append(lo)
            his.append(hi)
            # modality_dims[action_key] = lo.shape[0]
        if not los:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        return np.concatenate(los), np.concatenate(his) #, modality_dims

    def denormalize_L1_action_err(self, L1_err, dataset_name: list[str]):
        # overwrite by subclass
        ...

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data

# 调用自config.py，这是对config中如何描述：处理不同的数据集，的包装
class DataTransform(BaseModel):
    repack: RepackTransform
    model: ModelTransform
    field: FieldTransform

    def __call__(self, data:dict[str, Any], **kwargs) -> dict[str, Any]:
        data = self.repack(data, **kwargs)
        data = self.field(data, **kwargs)
        data = self.model(data, **kwargs)
        return data





