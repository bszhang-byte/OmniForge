# OmniForge

当前包含：

- Qwen3-VL + FAST action-token 模型；
- Psi0 flow-matching 动作专家；
- EgoDex 数据集；
- 单一 `Config` dataclass，支持 JSON 和 CLI 覆盖；
- 通用 `Trainer`；
- `build_model()` 和 `build_dataset()` 工厂。

A 线代码不 import `psi` 或 `ZCode_psi`。`fast/` 和 `assets/` 是运行时所需资源目录。

**目前所有运行程序均为demo**

## 目录结构

```text
OmniForge/
├── train.py                       # torchrun 入口和组件组装
├── trainer/trainer.py             # 训练、反向传播、优化器、checkpoint
├── config/
│   ├── config.py                  # 唯一默认配置和覆盖逻辑
│   ├── transform.py               # 数据变换
│   └── configs/
│       ├── other_config.json      # Qwen/FAST 示例
│       └── psi0_flow.json         # Psi0 flow 示例
├── data/
│   ├── dataset.py                 # 数据集基类
│   ├── egodex.py                  # EgoDexDataset
│   └── factory.py                 # build_dataset()
├── model/
│   ├── factory.py                 # build_model()
│   ├── fast_causal.py             # FastCausalModel
│   ├── psi0_flow.py               # Psi0FlowAdapter
│   └── psi0_core/                 # Psi0 blocks/attention/action head
├── fast/                          # FAST tokenizer 资源
├── assets/                        # 统计文件等资源
├── tests/test_one_batch.py        # 单 batch 测试
├── run_other.sh                   # 快捷启动脚本
└── CALL_CHAIN.md                  # 逐文件调用链
```

## 总体调用链

```text
torchrun train.py
  ├─ load_config()
  │    Config 默认值 ← JSON ← CLI
  ├─ build_dataset(cfg)
  │    └─ data.factory → EgoDexDataset
  │                         └─ __getitem__() → 单样本字典
  ├─ DataLoader(dataset, sampler, batch_size)
  │    └─ 多个样本字典 → batch 字典
  ├─ build_model(cfg)
  │    ├─ qwen3vl_fast → FastCausalModel
  │    └─ psi0_flow   → Psi0FlowAdapter
  └─ Trainer
       └─ model.compute_loss(batch) → loss.backward() → optimizer.step()
```

Dataset 负责“原始文件 → 基础 batch”；模型适配器负责“基础 batch → 模型输入”；Trainer 不负责解释数据字段。

## 启动训练

```bash
cd /data1/code/dlx/OmniForge
bash run_other.sh
```

`run_other.sh` 会激活 `.venv`，设置 `PYTHONPATH`、Hugging Face 离线缓存和默认 GPU，并读取：

```text
config/configs/other_config.json
```

指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=6,7 bash run_other.sh
```

直接用 torchrun：

```bash
source .venv/bin/activate
export PYTHONPATH=.
torchrun --standalone --nproc_per_node=2 train.py \
  --config config/configs/other_config.json
```

Psi0 flow：

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --config config/configs/psi0_flow.json
```

临时改参数：

```bash
bash run_other.sh --max_training_steps 10 --train_batch_size 1 --num_workers 0
```

## 如何换模型

### 切换已有模型

在 JSON 中设置：

```json
{ "model_family": "qwen3vl_fast" }
```

可选 family：

- `qwen3vl_fast`，别名 `fast_causal`、`qwen`；
- `psi0_flow`，别名 `flow`、`psi0`。

选择逻辑位于：

```text
model/factory.py: build_model(cfg)
```

### 增加新模型

1. 新建 `model/my_model.py`。
2. 定义 `MyModel(cfg)`，并实现：

```python
def compute_loss(self, batch):
    # 将基础 batch 转成该模型的输入
    return scalar_loss
```

3. 在 `model/factory.py` 导入并注册：

```python
from model.my_model import MyModel

if cfg.model_family == "my_model":
    return MyModel(cfg)
```

4. 在 `config/config.py` 加入模型所需默认字段，在 JSON 中写：

```json
{ "model_family": "my_model" }
```

模型专用 token 化、图像处理、动作编码和 loss 都写在模型适配器中，不写在 Trainer 中。

## 如何换数据集

### 切换已有数据集

当前数据集由 `data/factory.py` 根据 `dataset_family` 创建：

```json
{
  "dataset_family": "egodex",
  "root_dir": "/data1/code/dlx/psi_home/data/egodex",
  "split": "part1"
}
```

### 增加新数据集

1. 新建 `data/my_dataset.py`。
2. 定义 `MyDataset`，实现 `__len__()` 和 `__getitem__(index)`。
3. 在 `__getitem__()` 中读取一条原始样本，返回基础字典，例如：

```python
{
    "actions": ...,         # [T, 48] 或 action_dim
    "states": ...,          # 没有状态时可省略
    "current_images": ...,  # 图像或视频帧
    "instruction": ...,     # str
}
```

4. 在 `data/factory.py` 注册：

```python
from data.my_dataset import MyDataset

if cfg.dataset_family == "my_dataset":
    return MyDataset(cfg, transform=transform)
```

5. 在 `config/config.py` 加入该数据集的字段，在 JSON 中填写 `dataset_family` 和路径。

`DataLoader` 通过 sampler 产生索引，调用 Dataset 的 `__getitem__()`，再将多个单样本字典 collate 成 batch。通常不需要为每个数据集单独实现 Loader。

## 如何修改配置

默认值和字段定义集中在：

```text
config/config.py
└── class Config
```

覆盖优先级：

```text
Config 默认值 < --config some.json < 命令行参数
```

JSON 放在 `config/configs/`。常用配置：

```json
{
  "output_dir": ".runs/egodex",
  "root_dir": "/data1/code/dlx/psi_home/data/egodex",
  "split": "part1",
  "train_batch_size": 1,
  "num_workers": 0,
  "max_training_steps": 1000,
  "learning_rate": 0.0001,
  "gradient_accumulation_steps": 1,
  "mixed_precision": "bf16",
  "checkpointing_steps": 500,
  "model_family": "qwen3vl_fast",
  "dataset_family": "egodex"
}
```

新增配置字段时：

1. 在 `Config` 中加入带类型和默认值的字段；
2. 在实际使用它的文件中读取 `cfg.field_name`；
3. 在 `config/configs/*.json` 中为具体实验填写值。

训练字段由 `trainer/trainer.py` 使用，数据字段由 `data/egodex.py` 使用，模型字段由相应的 `model/*.py` 使用。

## 接口约定

模型适配器应接受基础 batch，并返回标量 loss：

```python
loss = model.compute_loss(batch)
```

当前 FAST 约束：

- user 只包含图像和指令；
- state 不进入 VLM；
- assistant 只包含 FAST 动作 token；
- labels 只监督动作段；
- FAST 默认资源为 `fast/egodex-rel-50w-1x48-v2048-s100`。

## 验证

```bash
.venv/bin/python -m py_compile \
  config/config.py data/*.py model/*.py model/psi0_core/*.py \
  trainer/trainer.py train.py

.venv/bin/python tests/test_one_batch.py
```

更多逐文件细节见 [CALL_CHAIN.md](CALL_CHAIN.md)。


## SmolVLA 实验

本框架已接入 Hugging Face LeRobot 的公开 SmolVLA。SmolVLA 是约 450M 参数的轻量 VLA，官方提供完整实现和 `lerobot/smolvla_base` checkpoint。

### 代码和资源

```text
model/factory.py
    └── model_family=smolvla
            └── model/smolvla.py:SmolVLAAdapter

/data1/code/dlx/psi_home/cache/hf/smolvla_base
    ├── model.safetensors
    └── config.json

/data1/code/dlx/psi_home/cache/hf/smolvlm2
    └── SmolVLM2-500M-Video-Instruct backbone
```

首次使用需要安装 LeRobot 的 SmolVLA 依赖；当前服务器还安装了 `num2words`：

```bash
.venv/bin/python -m pip install num2words
```

### 输入接口

公开 SmolVLA checkpoint 的标准接口是：

```text
observation.images.camera1 : [B, 3, 256, 256]
observation.state           : [B, 6]
task                        : list[str]
action                      : [B, 50, 6]
```

当前 EgoDex 样本是 48 维动作和 48 维状态。为了验证框架链路，`SmolVLAAdapter` 做了明确的调试适配：

- 状态取前 6 维；
- 动作取前 6 维；
- 将单帧动作重复为长度 50 的 action chunk；
- EgoDex 图像转换为 `observation.images.camera1`。

这能验证模型加载、数据转换、前向和参数更新，但不等于正式的 EgoDex-SmolVLA 数据适配。正式训练应新增返回 LeRobot action-chunk 格式的数据集。

### 配置和启动

配置文件：

```text
config/configs/smolvla.json
```

单步训练：

```bash
cd /data1/code/dlx/OmniForge
source .venv/bin/activate

export PYTHONPATH=.
export HF_HOME=/data1/code/dlx/psi_home/cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=7

python train.py   --config config/configs/smolvla.json
```

关键配置：

```json
{
  "model_family": "smolvla",
  "model_name_or_path": "/data1/code/dlx/psi_home/cache/hf/smolvla_base",
  "dataset_family": "egodex",
  "train_batch_size": 1,
  "max_training_steps": 1
}
```

已经在真实 EgoDex 数据上验证：

```text
step=1 loss=0.214403
```

官方资料：

- [LeRobot SmolVLA 文档](https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx)
- [SmolVLA 模型代码](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py)
- [SmolVLA 论文](https://arxiv.org/abs/2506.01844)
