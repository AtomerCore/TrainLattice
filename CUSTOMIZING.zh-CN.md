# 自定义 TrainLattice

[English](CUSTOMIZING.md) | 简体中文

运行时构建的每一样东西都是注册组件。架构、数据集、优化器与损失全部活在注册表里，任何 import 了 TrainLattice 并调用 `register` 的 Python 模块都在扩展它们——没有插件加载机制、没有入口点发现、也没有"注册"本身的配置。本文档是编写自定义组件的完整参考。

## 心智模型

四个注册表即扩展点，从包顶层导出：

| 注册表 | 构建对象 | 内置 | 构造器签名 |
|---|---|---|---|
| `ARCHITECTURES` | 模型 | `mlp`、`cnn` | `(*, **options) -> nn.Module` |
| `DATASETS` | 数据集 | `synthetic`、`tensor` | `(*, train, split=None, **options) -> Dataset` |
| `OPTIMIZERS` | 优化器 | `adamw`、`adam`、`sgd` | `(params, lr=..., **options) -> Optimizer` |
| `LOSSES` | 损失 | `cross_entropy`、`mse`、`l1` | `(**options) -> nn.Module`，经 `register_loss` |

**import 模块即完成注册**。你的实验脚本 import 组件模块，然后经 `run_experiment` 或 `run_from_file` 启动；YAML 配置只按名字引用组件。内置组件使用的注册调用与你调用的是同一个——自定义组件获得完全相同的校验、契约比对、checkpoint 与恢复语义。

## 各类组件的注册

### 架构

```python
import torch.nn as nn

from trainlattice import ARCHITECTURES, ComponentContract, ConfigError


class Logistic(nn.Module):
    def __init__(self, *, input_dim: int = 2, num_classes: int = 2) -> None:
        super().__init__()
        self.network = nn.Linear(input_dim, num_classes)

    def forward(self, inputs):
        return self.network(inputs)


def _validate(path, options):
    unknown = sorted(set(options) - {"input_dim", "num_classes"})
    if unknown:
        keys = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration field(s): {keys}")


def _contract(options):
    return ComponentContract(
        input_dim=options.get("input_dim", 2),
        num_classes=options.get("num_classes", 2),
    )


ARCHITECTURES.register("logistic", Logistic, validate=_validate, contract=_contract)
```

约定：

- 构造器接受扁平关键字选项，与 YAML `model` 段（除 `name` 外）的键一一对应，返回 `nn.Module`。
- `validate(path, options)` 收到小节路径（`"model"`）与解析后的选项。在这里拒绝未知键，拼写错误才能在构建之前变成点分路径错误。
- `contract(options)` 仅从选项重述输入宽度/形状与输出宽度——运行时绝不通过探测数据来学习形状。`ComponentContract(input_dim=..., num_classes=..., input_shape=...)`：`input_dim` 是展平后的输入宽度，`input_shape` 是结构化形状（声明其一或两者；`None` 跳过该条比对），`num_classes` 是输出宽度——分类任务即类别数，回归任务即目标宽度。

### 数据集

```python
from torch.utils.data import Dataset, TensorDataset

from trainlattice import DATASETS


def build_two_blobs(
    *, train, split=None, input_dim=2, samples=128, val_samples=32, seed=None, **_
) -> Dataset:
    ...  # 每类生成一个高斯簇，返回 TensorDataset(features, labels)


DATASETS.register("two_blobs", build_two_blobs, validate=..., contract=...)
```

约定：

- 构造器**按 split 各调用一次**——`train=True, split="train"` 与 `train=False, split="validation"`——并接收运行时按 `experiment.seed` 注入的 `seed`。请从它派生各 split 的生成器，使两个 split 既确定又互不重叠随机性。
- 样本是 `(输入, 目标)` 二元组：分类损失用 `long` 目标，回归损失用**与模型输出同形状**的 float 目标。
- 应容忍未知的额外关键字参数（`**_`）——运行时会在你的 YAML 选项之上注入 `seed`。
- 数据增强属于数据集构造器的职责；运行时不做任何隐藏的变换。

### 优化器

```python
import torch

from trainlattice import OPTIMIZERS

OPTIMIZERS.register("rmsprop", torch.optim.RMSprop, validate=...)
```

约定：

- 构造器**按位置**接收模型参数，随后是学习率与选项：`constructor(params, lr=..., **options)`——与 `torch.optim` 的签名完全一致，因此优化器类可以直接注册。
- YAML：`optimizer: {name: rmsprop, lr: 0.001, alpha: 0.99}`——除 `name` 与 `lr` 外的每个键都作为选项透传。
- 优化器没有契约声明；`validate` 是唯一的可选声明。

### 损失

```python
import torch

from trainlattice import register_loss

register_loss("huber", torch.nn.SmoothL1Loss, task="regression", validate=...)
```

约定：

- 损失经 `register_loss` 注册，它额外要求声明**任务形态**：`classification` 或 `regression`。
- 任务驱动指标：分类损失报告损失与准确率；回归损失只报告损失。它同时约束 `trainer.monitor`——`val_accuracy` 需要分类损失。
- YAML：`loss: {name: huber}`，损失特有选项透传（`reduction`；`cross_entropy` 另有 `label_smoothing`、`ignore_index`）。无论 `reduction` 取何值，指标一律按每样本均值聚合；`sum` 保留其更大的梯度尺度。

## 一个完整的自定义实验

可运行版本见 `examples/`：`custom_model.py`、`custom_dataset.py`、`custom_experiment.py`、`custom.yaml`（分类），以及 `regression_experiment.py` + `regression.yaml`（自定义数据集配内置 `mse` 损失与 `sgd` 优化器）。入口脚本就是全部故事：

```python
from pathlib import Path

import custom_dataset  # import 即注册 two_blobs
import custom_model  # import 即注册 logistic

from trainlattice import run_from_file

CONFIG = Path(__file__).parent / "custom.yaml"

result = run_from_file(CONFIG, command="python my_experiment.py")
# 中断后恢复：
# run_from_file(CONFIG, resume="runs/<run>/checkpoints/last.pt", command="python my_experiment.py")
```

把真实的启动命令传给 `command=`——运行 manifest 把它记录为复现入口，连同指向各组件定义模块与文件的 `model_source`、`dataset_source`、`optimizer_source`、`loss_source` 字段。

YAML 与任何内置配置形状相同，只是名字不同：

```yaml
model:
  name: logistic
  input_dim: 2
  num_classes: 2

dataset:
  name: two_blobs
  batch_size: 16
  input_dim: 2

loss:
  name: cross_entropy
```

自定义与内置组件可以任意混用。

## 自定义 callback

训练行为通过传入 `extra_callbacks` 的 `Callback` 子类扩展：

```python
from time import perf_counter

from trainlattice import Callback, run_from_file


class EpochTimeCallback(Callback):
    def on_epoch_start(self, trainer):
        self._started = perf_counter()

    def on_epoch_end(self, trainer):
        print(f"epoch {trainer.state.epoch} 耗时 {perf_counter() - self._started:.3f}s")


run_from_file(CONFIG, command="python my_experiment.py", extra_callbacks=[EpochTimeCallback()])
```

各钩子都会收到 trainer——`on_train_start`、`on_epoch_start`、`on_batch_start`、`on_batch_end`、`on_epoch_end`、`on_train_end`、`on_exception`——callback 因此能在每个边界读到模型、优化器与 `TrainerState`。注入的 callback 在每个事件中先于指标记录之后、checkpoint 之前执行。callback 是观察者：不能中断训练（早停等控制流钩子是未来的扩展）。

## 校验与错误消息

下表所有失败都发生在运行目录创建之前：

| 错误 | 错误消息 |
|---|---|
| 组件名拼错 | `unsupported model.name 'logstic'; available: cnn, logistic, mlp` |
| 组件选项未知 | `unknown configuration field(s): model.kernel_size` |
| 选项值无效 | `optimizer.momentum must be a non-negative number` |
| 形状不匹配 | `model.input_dim (4) must match dataset.input_shape product (2)` |
| 结构化形状不匹配 | `model.input_shape ([3, 4, 4]) must match dataset.input_shape ([4])` |
| 输出宽度不匹配 | `model.num_classes (3) must match dataset.num_classes (2)` |
| monitor 与任务不匹配 | `trainer.monitor 'val_accuracy' requires a classification loss; loss.name 'mse' is regression` |

注册重名会立即报错（`architecture 'mlp' is already registered`），因此内置组件不可被覆盖。不声明 `validate` 时，未知选项会以构造错误而非点分路径消息出现；不声明 `contract` 时，该组件跳过其侧的形状比对。建议两者都声明。

## 兼容性注意

- **pickle**：`num_workers > 0` 时，Windows 与 macOS 上的 worker 进程按 spawn 方式启动——闭包或 lambda 构造的数据集会在这些平台失败。调试自定义数据集时可先用 `num_workers: 0`。
- **恢复**：影响恢复语义的配置必须与 checkpoint 一致，包括 loss 小节、优化器名与全部组件选项。恢复运行会重新 import 你的组件模块并重建相同组件；若两次运行之间修改了组件代码，恢复风险自负——比对只针对配置，不针对源码。
- **复现**：运行 manifest 记录各组件代码所在位置，但实验脚本本身也是复现入口的一部分——请与运行目录一并归档。

## 为什么这样设计

- **import 时注册而非动态加载**：YAML 固然可以指向要 import 的插件文件，但那会引入加载器、路径解析规则和第二套发现机制——而 `import` 正是 Python 程序员已有的机制。注册在脚本里保持可见，任何调试器、profiler 或读者都能跟上它。
- **声明随注册走**：`validate` 与 `contract` 挂在构造器注册的地方，组件因此把它的校验与形状语义带到一切使用场景——内置与自定义组件对运行时不可区分。
- **任务形态靠声明而非嗅探**：训练器可以在运行时按目标 dtype 猜指标，但 `register_loss` 里的声明在校验阶段就快速失败，且读起来就是意图本身。
