# TrainLattice

[English](README.md) | 简体中文

面向深度学习训练的极简基础设施。

**构建。运行。恢复。复现。**

TrainLattice 是一个轻量、透明的 PyTorch 训练运行时，用于从第一性原理学习深度学习基础设施。一份经过严格校验的 YAML 配置会被转换成一次可复现的训练：注册表中的架构（内置 `mlp`/`cnn`，或从 Python 注册的自定义架构）与数据集（内置 `synthetic`/`tensor`，或自定义数据集）、注册表中的优化器（内置 `adamw`/`adam`/`sgd`）、可选的学习率调度器（内置 `warmup_cosine`/`warmup_linear`/`step_decay`）与损失（内置 `cross_entropy`/`mse`/`l1`）、生命周期回调、结构化指标、原子化 checkpoint、显式恢复，以及自包含的运行产物。

## 特性

- 严格校验的 YAML 实验配置：未知小节与字段在创建运行目录之前即被拒绝，错误信息带点分路径
- 基于注册表的组件构建；`mlp`/`cnn` 架构注册表与 `synthetic`/`tensor` 数据集注册表保留为可见扩展点
- 可选优化器（`adamw`、`adam`、`sgd` 及各自超参）与损失（`cross_entropy`、`mse`、`l1`），走同一套注册表机制；损失声明任务形态，回归运行因此只报告损失、不报告准确率
- 可选学习率调度器（带 warmup 与下限的 `warmup_cosine`、`warmup_linear`，阶梯式的 `step_decay`），走同一套注册表；指标记录携带每次优化器步进实际使用的学习率，epoch 记录携带 epoch 末学习率（下一次步进将使用的值）
- 通过同一套注册表支持自定义网络与数据集：import 组件模块即完成注册，实验从 Python 经 `run_experiment`/`run_from_file` 启动，自定义 callback 经 `extra_callbacks` 观察生命周期，运行 manifest 记录组件代码所在位置
- 可配置的 MLP：`input_dim`、`hidden_dims`、`num_classes`；激活固定为 GELU，dropout 固定为 0.1
- 可配置的 CNN：`input_shape`、`channels`、`num_classes`；`Conv2d -> GELU -> MaxPool` 块加线性头
- 确定性的 synthetic 训练/验证划分，以及按 epoch 感知的洗牌采样
- 文件型 `tensor` 数据集：从本地目录读取 `train.pt`/`validation.pt` 张量对；不下载任何内容
- 显式的 `Trainer` 生命周期与可序列化的 `TrainerState`
- Console 与 JSONL 结构化指标日志，损失与准确率按样本加权，每条 batch 与 epoch 记录附学习率与吞吐指标：batch 记录携带本次优化器步进实际使用的 `lr` 与自 epoch 起累计的 `steps_per_sec`/`samples_per_sec`（含数据等待），epoch 记录携带 `lr`（epoch 末学习率，即下一次优化器步进将使用的值）、`epoch_seconds`、`data_seconds`（等待数据耗时）、`train_samples_per_sec`，CUDA 设备另附 `max_memory_mb`
- 结构化诊断日志，格式为 `[LEVEL] logger.name: message`：INFO 及以下输出到 stdout，WARNING 及以上输出到 stderr，Python warnings 也收进该格式；`--verbose` 开启 debug 级诊断
- 原子化 checkpoint（格式 v4）：包含模型、优化器、配置了调度器时的调度器状态及其解析后的步数预算、fp16 混合精度下的损失缩放器状态、训练状态、解析后配置与 RNG 状态；每个 epoch 结束写 `last.pt`，受监控的验证指标改进时写 `best.pt`，训练异常时写诊断用的 `emergency.pt`
- 可选梯度裁剪：`trainer.max_grad_norm`
- 梯度累积：`trainer.accumulate` 让多个微批共享一次优化器步进，无需增大 `batch_size` 即可获得更大的有效批
- 混合精度：`trainer.precision` 支持 `bf16`（CPU 与 CUDA）与 `fp16`（仅 CUDA），前向计算在 autocast 下执行；`fp16` 配备 GradScaler，其状态随 checkpoint 保存与恢复，因非有限梯度被缩放器跳过的窗口不更新参数、不推进调度器、不计优化器步
- 早停与协作式停止：`trainer.early_stopping`（`patience`、`min_delta`）基于受监控的验证指标判定收敛，任意回调可通过 `Trainer.request_stop` 请求停止；两种路径都先落盘当轮 checkpoint，再以记录了停止原因的 `finished` 状态结束运行
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

`configs/synthetic_mlp_efficiency.yaml` 在同一模型与数据集上组合使用效率相关控制项：`warmup_cosine` 学习率调度、梯度累积、`bf16` 混合精度与早停。

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
- `metrics.jsonl` — 追加式指标记录，每行一个 JSON 对象：每 `trainer.log_every` 个优化器步产生一条 batch 记录，每个 epoch 产生一条 epoch 记录。batch 记录携带按样本加权的 `loss` 与 `accuracy`（仅分类）、本次优化器步进实际使用的 `lr`，以及自 epoch 起累计的 `steps_per_sec`/`samples_per_sec`（含数据等待）；epoch 记录携带 `train_loss`、`train_accuracy`（仅分类）、带验证轮次的 `val_loss`/`val_accuracy`、`lr`（epoch 末学习率，即下一次优化器步进将使用的值）、`train_samples_per_sec`、`epoch_seconds`、`data_seconds`（等待数据耗时），CUDA 设备另附 `max_memory_mb`。
- `checkpoints/last.pt` — 每个完成的 epoch 后原子写入；恢复来源。
- `checkpoints/best.pt` — 受监控的验证指标（`trainer.monitor`，默认 `val_loss`）优于历史最佳时写入，恢复后延续同一标准。
- `checkpoints/emergency.pt` — 训练被异常打断时写入的诊断快照；不可用于恢复。

恢复会还原模型、优化器、训练状态、配置了调度器时的调度器状态、fp16 混合精度运行时的损失缩放器状态，以及进程内可用的 RNG 状态（Python、Torch、CUDA，以及存在时的 NumPy），然后从下一个 epoch 边界继续，样本顺序与学习率调度均与不中断的训练完全一致。格式 v3 与 v4 的 checkpoint 均可恢复；v4 相对 v3 的增量字段是 `scheduler` 与 `scaler` 状态以及调度器解析后的步数预算，因此续训延续运行开始时的调度曲线——即使 `trainer.epochs` 改变也不变（预算与新推导值不一致时记录日志说明）。未配置调度器的 checkpoint 恢复时同样不带调度器——为其新增 `scheduler` 小节属恢复敏感变更，会被拒绝。影响恢复语义的配置必须与 checkpoint 一致；仅运行时的设置（`trainer.device`、`trainer.log_every`、`trainer.epochs`、`trainer.early_stopping` 与 `experiment.runs_dir`）允许变化。

## 配置

`configs/synthetic_mlp.yaml` 是最小的冒烟测试示例；完整的受支持配置面是下方的小节与字段清单：

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
- `scheduler` — 可选；`name` 选择已注册的调度器。`warmup_cosine` 与 `warmup_linear` 接受 `warmup_steps`（默认 `0`）、`total_steps`（默认由运行推导：每 epoch 优化器步数 —— `ceil(每 epoch 批数 / trainer.accumulate)` —— × `trainer.epochs`；显式设置优先于推导值）与 `min_lr`（默认 `0.0`，绝对下限）；学习率先在 warmup 段从 0 线性升到 `optimizer.lr`（须在 `total_steps` 之前结束），随后按余弦或线性衰减到 `min_lr` 并保持。`step_decay` 接受 `step_size`（必填，≥ 1）与 `gamma`（默认 `0.1`），每 `step_size` 步将学习率乘以 `gamma`。省略该小节时保持恒定学习率。
- `loss` — `name`（默认 `cross_entropy`）选择已注册的损失：`cross_entropy`（分类；选项 `label_smoothing`、`ignore_index`、`reduction`）或 `mse`/`l1`（回归；选项 `reduction`）。无论 `reduction` 取何值，指标一律按每样本均值聚合；`sum` 保留其更大的梯度尺度。回归损失使训练器只报告损失——此时 `trainer.monitor` 必须保持 `val_loss`——且数据集目标须是与模型输出同形状的 float 张量。省略该小节时保持交叉熵。
- `trainer` — `epochs`（默认 `1`）、`device`（`auto`、`cpu`、`cuda` 或 `cuda:N`）、`validate_every`（默认 `1`）、`log_every`（默认 `10`）、`monitor`（`val_loss` 或 `val_accuracy`，默认 `val_loss`，驱动 `best.pt`）、`max_grad_norm`（默认 `null` 即关闭；在优化器步进前按总范数裁剪梯度）、`accumulate`（默认 `1`；每次优化器步进聚合的微批数 —— 每个微批损失在 backward 前乘 `1 / accumulate`，epoch 末不足整窗时补一次步进，`global_step` 按优化器步计数）、`precision`（`fp32`、`bf16` 或 `fp16`，默认 `fp32`；混合精度模式将训练与验证的前向计算置于对应 dtype 的 autocast 下 —— `fp16` 仅允许 CUDA 设备，每次 backward 经 GradScaler 缩放并在裁剪与优化器步进前先反缩放，其状态随 checkpoint 保存，`bf16` 无需缩放器）、`early_stopping`（可选映射，含 `patience`（≥ 1）与 `min_delta`（≥ 0，默认 `0.0`），默认 `null` 即关闭；`trainer.monitor` 的每次验证评估若未改善超过 `min_delta` 就计入耐心计数，超过 `patience` 次后在该 epoch 边界停止 —— 当轮 checkpoint 照常保存，运行以 `finished` 状态结束并把原因记入训练状态，无验证的轮次不计数，恢复后计数从零重数而最优指标参照随 checkpoint 延续）。

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

## 路线图

受支持的范围保持本地、单进程。以下方向正在考虑用于后续版本，刻意未排定日程：

- 训练循环的性能剖析
- `torch.compile` 加速
- 分布式训练（DDP/FSDP）
- 在分类与回归之外增加语言建模任务形态

## 测试与开发

```powershell
python -m pytest
python -m trainlattice train configs/synthetic_mlp.yaml
```

测试使用确定性 synthetic 数据，离线运行。工程边界与贡献约定见 [ARCHITECTURE.md](ARCHITECTURE.md) 与 [AGENTS.md](AGENTS.md)。

## 许可

TrainLattice 以 MIT 许可发布。
