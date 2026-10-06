"""One configuration dataclass; no model/dataset/trainer registry or dynamic dispatch."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # Experiment / optimization (names follow the upstream recipe).
    # train.py: 全局随机种子、运行阶段与输出目录
    seed: int = 7
    # train.py: 全局随机种子
    seed: int = 7
    stage: str = "fast"
    output_dir: str = str(ROOT / ".runs" / "egodex")
    # train.py / trainer.py: batch、优化器、scheduler 与 checkpoint
    train_batch_size: int = 8
    num_workers: int = 0
    gradient_accumulation_steps: int = 2
    max_training_steps: int = 1000000
    learning_rate: float = 1e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0  # upstream MODEL weight_decay overrides optimizer default
    mm_projector_lr: float = 1e-5
    vision_tower_lr: float = 1e-5
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    min_lr: float = 5e-7
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    data_parallel: str = "ddp"
    deepspeed_config: str = ""
    enable_gradient_checkpointing: bool = True
    log_freq: int = 10
    checkpointing_steps: int = 5000
    resume_from_checkpoint: str = ""
    overfit_single_batch: bool = False

    # Data / preprocessing.
    # data/factory.py / data/egodex.py: 数据集选择与采样
    dataset_family: str = "egodex"
    # data/factory.py: 数据集选择
    dataset_family: str = "egodex"
    root_dir: str = "/data1/code/dlx/psi_home/data/egodex"
    split: str = "train"
    chunk_size: int = 1
    upsample_rate: int = 3
    use_delta_actions: bool = True
    max_episodes: int = 0
    frame_index: int = -1  # >=0 enables deterministic inspection/parity
    # data/egodex.py: 动作统计与图像预处理
    stat_path: str = str(ROOT / "assets/egodex_stat_all.json")
    # data/egodex.py: 统计文件
    stat_path: str = str(ROOT / "assets/egodex_stat_all.json")
    stat_action_key: str = "egodex"
    action_norm_type: str = "bounds_q99"
    resize: tuple = (270, 480)  # H,W; torchvision bilinear, as upstream
    model_max_length: int = 8192

    # Qwen + FAST. These are asset paths; no imports from either code mirror.
    # model/factory.py: 模型分发选择
    model_family: str = "qwen3vl_fast"
    # model/factory.py: 模型选择
    model_family: str = "qwen3vl_fast"
    # model/fast_causal.py: Qwen3-VL + FAST adapter
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    cache_dir: str = "/data1/code/dlx/psi_home/cache/hf/hub"
    pretrained_checkpoint: str = str(ROOT / "fast/egodex-rel-50w-1x48-v2048-s100")
    bins: int = 2048
    tune_mm_llm: bool = True
    tune_mm_vision: bool = True
    tune_mm_mlp: bool = True
    attn_implementation: str = "sdpa"
    local_files_only: bool = True

    # Optional continuous flow-matching stage; fed directly to model.psi0.
    # model/psi0_flow.py / model/psi0_core/: flow-matching Psi0 模型
    action_dim: int = 48
    # model/psi0_flow.py: action dimension
    action_dim: int = 48
    odim: int = 48
    hidden_dim: int = 1024
    num_blocks: int = 8
    view_feature_dim: int = 2048
    pooled_projection_dim: int = 2048
    resnet_store_path: str = ""
    use_dit: bool = False
    use_film: bool = False
    combined_temb: bool = False
    final_layer_norm: bool = True
    qk_norm: str = "rms_norm"
    state_drop_prob: float = 0.0
    vlm_layer_indices: tuple = ()
    # model/smolvla.py: released LeRobot checkpoint interface
    smolvla_action_dim: int = 6
    smolvla_state_dim: int = 6

    @property
    def action_chunk_size(self):
        return self.chunk_size

    def validate(self):
        for key in ("train_batch_size", "gradient_accumulation_steps", "max_training_steps",
                    "chunk_size", "upsample_rate", "log_freq"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.stage not in ("fast", "flow"):
            raise ValueError("stage must be fast or flow")
        if self.model_family not in ("qwen3vl_fast", "fast_causal", "qwen", "psi0_flow", "flow", "psi0", "smolvla", "smol_vla"):
            raise ValueError(f"Unknown model_family: {self.model_family}")
        if self.data_parallel not in ("ddp", "deepspeed"):
            raise ValueError("data_parallel must be ddp or deepspeed")
        if self.mixed_precision not in ("no", "bf16"):
            raise ValueError("Supported mixed_precision: no, bf16")
        if self.action_norm_type not in ("bounds", "bounds_q99"):
            raise ValueError("Unsupported action_norm_type")
        if len(self.resize) != 2 or min(self.resize) <= 0:
            raise ValueError("resize must contain positive H W")
        if self.action_dim != 48 or self.odim != 48:
            raise ValueError("EgoDex adapter requires action_dim=odim=48")
        if self.num_workers < 0 or self.max_episodes < 0:
            raise ValueError("num_workers and max_episodes must be nonnegative")
        return self

    def to_dict(self):
        return asdict(self)


# Only a type alias for existing model annotations; one actual config dataclass.
LaunchConfig = Config


def _boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_config(argv=None):
    """Flat flags are canonical; selected upstream dotted spellings are aliases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Flat JSON; CLI overrides JSON")
    defaults = Config()
    train_keys = {"output_dir", "train_batch_size", "num_workers", "gradient_accumulation_steps",
                  "max_training_steps", "learning_rate", "lr_scheduler_type", "warmup_steps",
                  "max_grad_norm", "mixed_precision", "data_parallel", "deepspeed_config",
                  "enable_gradient_checkpointing", "checkpointing_steps",
                  "resume_from_checkpoint", "overfit_single_batch"}
    data_keys = {"root_dir", "chunk_size", "upsample_rate", "use_delta_actions"}
    model_keys = {"model_name_or_path", "tune_mm_llm", "tune_mm_vision", "tune_mm_mlp",
                  "mm_projector_lr", "vision_tower_lr", "weight_decay", "model_max_length"}
    for item in fields(Config):
        key, default = item.name, getattr(defaults, item.name)
        aliases = [f"--{key}", f"--{key.replace('_', '-')}"]
        group = "train" if key in train_keys else "data" if key in data_keys else "model" if key in model_keys else None
        if group:
            aliases += [f"--{group}.{key}", f"--{group}.{key.replace('_', '-')}"]
        if key in ("bins", "pretrained_checkpoint"):
            aliases += [f"--model.action_tokenizer.{key}"]
        if key in ("stat_path", "stat_action_key", "action_norm_type"):
            aliases += [f"--data.transform.field.{key}"]
        if key in ("betas", "eps"):
            aliases += [f"--train.lr_scheduler_kwargs.{key}"]
        if key == "resize":
            aliases += ["--data.transform.model.resize.size"]
        opts = dict(dest=key, default=argparse.SUPPRESS)
        if isinstance(default, bool):
            opts.update(type=_boolean, nargs="?", const=True)
        elif isinstance(default, tuple):
            opts.update(type=float if key == "betas" else int, nargs="+")
        else:
            opts["type"] = type(default)
        parser.add_argument(*dict.fromkeys(aliases), **opts)
    args = vars(parser.parse_args(argv))
    path = args.pop("config")
    supplied = json.loads(path.read_text()) if path else {}
    unknown = supplied.keys() - defaults.to_dict().keys()
    if unknown:
        parser.error(f"Unknown JSON fields: {sorted(unknown)}")
    supplied.update(args)
    for item in fields(Config):
        if item.name not in supplied:
            continue
        default, value = getattr(defaults, item.name), supplied[item.name]
        if isinstance(default, bool):
            supplied[item.name] = _boolean(value)
        elif isinstance(default, tuple):
            supplied[item.name] = tuple(value)
        elif not isinstance(value, type(default)):
            parser.error(f"{item.name}: expected {type(default).__name__}")
    return Config(**supplied).validate()
