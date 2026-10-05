# TrainLattice

[English](README.md) | 简体中文

面向深度学习训练的极简基础设施。

**构建。运行。恢复。复现。**

TrainLattice 是一个轻量、透明的 PyTorch 训练运行时，用于从第一性原理学习深度学习基础设施。一份经过严格校验的 YAML 配置会被转换成一次可复现的训练：注册表中的架构（内置 `mlp`/`cnn`，或从 Python 注册的自定义架构）与数据集（内置 `synthetic`/`tensor`，或自定义数据集）、注册表中的优化器（内置 `adamw`/`adam`/`sgd`）与损失（内置 `cross_entropy`/`mse`/`l1`）、生命周期回调、结构化指标、原子化 checkpoint、显式恢复，以及自包含的运行产物。

## 特性

- 严格校验的 YAML 实验配置：未知小节与字段在创建运行目录之前即被拒绝，错误信息带点分路径
- 基于注册表的组件构建；`mlp`/`cnn` 架构注册表与 `synthetic`/`tensor` 数据集注册表保留为可见扩展点
- 可选优化器（`adamw`、`adam`、`sgd` 及各自超参）与损失（`cross_entropy`、`mse`、`l1`），走同一套注册表机制；损失声明任务形态，回归运行因此只报告损失、不报告准确率
- 通过同一套注册表支持自定义网络与数据集：import 组件模块即完成注册，实验从 Python 经 `run_experiment`/`run_from_file` 启动，自定义 callback 经 `extra_callbacks` 观察生命周期，运行 manifest 记录组件代码所在位置
- 可配置的 MLP：`input_dim`、`hidden_dims`、`num_classes`；激活固定为 GELU，dropout 固定为 0.1
- 可配置的 CNN：`input_shape`、`channels`、`num_classes`；`Conv2d -> GELU -> MaxPool` 块加线性头
- 确定性的 synthetic 训练/验证划分，以及按 epoch 感知的洗牌采样
- 文件型 `tensor` 数据集：从本地目录读取 `train.pt`/`validation.pt` 张量对；不下载任何内容
- 显式的 `Trainer` 生命周期与可序列化的 `TrainerState`
- Console 与 JSONL 结构化指标日志，损失与准确率按样本加权
- 结构化诊断日志，格式为 `[LEVEL] logger.name: message`：INFO 及以下输出到 stdout，WARNING 及以上输出到 stderr，Python warnings 也收进该格式；`--verbose` 开启 debug 级诊断
- 原子化 checkpoint（格式 v3）：包含模型、优化器、训练状态、解析后配置与 RNG 状态；每个 epoch 结束写 `last.pt`，受监控的验证指标改进时写 `best.pt`，训练异常时写诊断用的 `emergency.pt`
- 可选梯度裁剪：`trainer.max_grad_norm`
- 显式 `--resume PATH` 从精确的已完成 epoch 边界恢复，并校验配置兼容性
- 自包含的运行产物：`config.yaml`、`environment.json`、`manifest.json`、`metrics.jsonl` 与 `checkpoints/`
- 支持单进程 CPU 与单 CUDA 设备执行，全进程统一设置随机种子

## 安装

需要 Python 3.10 或更高版本。在 PowerShell 中执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

核心依赖只有 PyYAML 与 PyTorch。synthetic 数据集在内存中生成，不下载任何内容。

## 快速开始

运行示例实验：

```powershell
python -m trainlattice train configs/synthetic_mlp.yaml
```

同样的流程无需改动即可运行卷积示例 `configs/synthetic_cnn.yaml`。指标同时输出到终端和运行目录下的 `metrics.jsonl`。运行结束时 CLI 打印目录，例如 `Run finished: runs/20261004-120000_synthetic-mlp-smoke`。

诊断日志与指标流是两条相互独立的通道。诊断记录格式为 `[LEVEL] logger.name: message`：INFO 及以下与指标、结果输出共用 stdout，WARNING 及以上（包括收进该格式的 Python warnings）走 stderr，因此 stderr 可以单独留档。加上 `--verbose` 开启 debug 级诊断：

```powershell
python -m trainlattice train configs/synthetic_mlp.yaml --verbose
```

从最近一个已完成 epoch 恢复该运行：

```powershell
python -m trainlattice train configs/synthetic_mlp.yaml --resume runs/20261004-120000_synthetic-mlp-smoke
```

`--resume` 接受运行目录（使用其中的 `checkpoints/last.pt`，或目录下直接的 `last.pt`）或 checkpoint 文件路径。每次调用都会创建新的运行目录，并在 manifest 中记录恢复来源。

## 产物与恢复

每次调用在 `runs/` 下创建相互独立、防冲突的目录：

```text
runs/20261004-120000_synthetic-mlp-smoke/
  config.yaml
  manifest.json
  environment.json
  metrics.jsonl
  checkpoints/
    last.pt
    best.pt
    emergency.pt
```

- `config.yaml` — 本次运行的解析后配置快照。
- `manifest.json` — 命令、Git 提交、种子、状态、时间戳、epoch、全局步数、恢复来源，以及定义模型、数据集、优化器与损失的模块和文件（`model_source`、`dataset_source`、`optimizer_source`、`loss_source`）；在 `created`、`running`、`finished`、`failed` 状态间原子更新。
- `environment.json` — Python、PyTorch、CUDA、平台与所选设备信息。
- `metrics.jsonl` — 追加式指标记录，每行一个 JSON 对象。
- `checkpoints/last.pt` — 每个完成的 epoch 后原子写入；恢复来源。
- `checkpoints/best.pt` — 受监控的验证指标（`trainer.monitor`，默认 `val_loss`）优于历史最佳时写入，恢复后延续同一标准。
- `checkpoints/emergency.pt` — 训练被异常打断时写入的诊断快照；不可用于恢复。

恢复会还原模型、优化器、训练状态以及进程内可用的 RNG 状态（Python、Torch、CUDA，以及存在时的 NumPy），然后从下一个 epoch 边界继续，样本顺序与不中断的训练完全一致。影响恢复语义的配置必须与 checkpoint 一致；仅运行时的设置（`trainer.device`、`trainer.log_every`、`trainer.epochs` 与 `experiment.runs_dir`）允许变化。

## 配置

`configs/synthetic_mlp.yaml` 展示了完整的受支持配置面：

```yaml
experiment:
  name: synthetic-mlp-smoke
  seed: 42
  runs_dir: runs

model:
  name: mlp
  input_dim: 784
  hidden_dims: [64, 32]
  num_classes: 10

dataset:
  name: synthetic
  batch_size: 32
  num_workers: 0
  samples: 128
  val_samples: 64
  input_shape: [1, 28, 28]
  num_classes: 10

optimizer:
  lr: 0.001

trainer:
  epochs: 2
  device: auto
  validate_every: 1
  log_every: 2
```

小节与字段：

- `experiment` — `name`（默认 `experiment`）、`seed`（默认 `42`）、`runs_dir`（默认 `runs`）。
- `model` — `name` 选择已注册的架构：内置 `mlp` 或 `cnn`，或自定义架构（见[自定义组件](#自定义组件)）。`mlp` 选项为 `input_dim`（默认 `784`）、`hidden_dims`（默认 `[512, 256, 128]`）、`num_classes`（默认 `10`），隐藏层固定使用 GELU 与 0.1 dropout。`cnn` 选项为 `input_shape`（默认 `[1, 28, 28]`）、`channels`（默认 `[32, 64]`）、`num_classes`（默认 `10`）；每个通道块为 `Conv2d(kernel 3, padding 1) -> GELU -> MaxPool(2)`，尾部接线性头，空间维度必须能被总池化系数整除。
- `dataset` — `name` 选择已注册的数据集：内置 `synthetic` 或 `tensor`，或自定义数据集；`batch_size`（默认 `64`）、`num_workers`（默认 `0`）。`synthetic` 选项为 `samples`（默认 `256`）、`val_samples`（默认 `64`）、`input_shape`（默认 `[1, 28, 28]`）、`num_classes`（默认 `10`）。`tensor` 选项为 `root`（必填）、`train_file`（默认 `train.pt`）、`val_file`（默认 `validation.pt`）、`num_classes`（默认 `10`）；每个文件须包含长度一致、标签小于 `num_classes` 的 `{"features": Tensor, "labels": Tensor}` 对。
- `optimizer` — `name`（默认 `adamw`）选择已注册的优化器：`adamw` 或 `adam`，选项为 `betas`、`eps`、`weight_decay`、`amsgrad`；`sgd`，选项为 `momentum`、`dampening`、`weight_decay`、`nesterov`；`lr`（默认 `0.001`）对所有优化器生效。省略该小节时保持 AdamW 加配置的学习率。
- `loss` — `name`（默认 `cross_entropy`）选择已注册的损失：`cross_entropy`（分类；选项 `label_smoothing`、`ignore_index`、`reduction`）或 `mse`/`l1`（回归；选项 `reduction`）。回归损失使训练器只报告损失——此时 `trainer.monitor` 必须保持 `val_loss`——且数据集目标须是与模型输出同形状的 float 张量。省略该小节时保持交叉熵。
- `trainer` — `epochs`（默认 `1`）、`device`（`auto`、`cpu`、`cuda` 或 `cuda:N`）、`validate_every`（默认 `1`）、`log_every`（默认 `10`）、`monitor`（`val_loss` 或 `val_accuracy`，默认 `val_loss`，驱动 `best.pt`）、`max_grad_norm`（默认 `null` 即关闭；在优化器步进前按总范数裁剪梯度）。

强制两条模型与数据集之间的跨字段契约：输入维度必须一致——双方都声明 `input_shape` 时要求精确相等，否则比较展平后的乘积；且模型的输出宽度必须等于数据集声明的宽度。分类任务中该宽度即类别数（两侧的 `num_classes`）；回归任务中即目标宽度（标量目标在两侧都声明 `num_classes: 1`）。未知小节、字段与组件选项会以点分路径报错，例如 `optimizer.weight_decay` 或 `model.kernel_size`。

`tensor` 数据集只读取本地普通文件。先用 PyTorch 生成一对示例文件，即可在其上训练：

```python
from pathlib import Path

import torch

root = Path("data/tensor")
root.mkdir(parents=True, exist_ok=True)
generator = torch.Generator().manual_seed(0)
for name, count in (("train.pt", 256), ("validation.pt", 64)):
    torch.save(
        {
            "features": torch.randn(count, 1, 28, 28, generator=generator),
            "labels": torch.randint(0, 10, (count,), generator=generator),
        },
        root / name,
    )
```

```powershell
python -m trainlattice train configs/tensor_mlp.yaml
```

## 自定义组件

架构、数据集、优化器与损失全部通过与内置组件相同的注册表注册——import 模块即完成注册，实验从 Python 启动。最小的自定义实验：

```python
from pathlib import Path

import my_model  # 定义 Logistic 并注册 "logistic"

from trainlattice import run_from_file

result = run_from_file(Path("my.yaml"), command="python my_experiment.py")
```

`my.yaml` 像内置组件一样按名字引用：

```yaml
model:
  name: logistic       # 你注册的架构
  input_dim: 2
  num_classes: 2
dataset:
  name: two_blobs      # 你注册的数据集
  batch_size: 16
loss:
  name: cross_entropy  # 或 mse / l1，或你自己的
```

自定义组件获得完全相同的校验、契约比对、checkpoint 与恢复语义，可与内置组件任意混用，运行 manifest 记录组件代码所在位置；自定义 callback 同样经 `extra_callbacks` 扩展生命周期。完整参考——构造器签名、`validate`/`contract` 声明、任务感知的损失、callback 与错误消息目录——见 [CUSTOMIZING.zh-CN.md](CUSTOMIZING.zh-CN.md)，可运行示例随仓库发布在 `examples/`：

```powershell
python examples/custom_experiment.py      # 自定义架构 + 数据集，分类
python examples/regression_experiment.py  # 自定义数据集，配内置 mse 损失与 sgd 优化器
```

## 测试与开发

```powershell
python -m pytest
python -m trainlattice train configs/synthetic_mlp.yaml
```

测试使用确定性 synthetic 数据，离线运行。工程边界与贡献约定见 [ARCHITECTURE.md](ARCHITECTURE.md) 与 [AGENTS.md](AGENTS.md)。

## 许可

TrainLattice 以 MIT 许可发布。
