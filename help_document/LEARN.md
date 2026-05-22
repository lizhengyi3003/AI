# ResNeXt 图像分类项目 — 学习笔记与思考

> **项目概述**：本项目手动实现了一个轻量化 ResNeXt-101 (8×32d) 网络，在 101 类图像数据集上完成图像分类全流程。通过从零搭建网络、处理数据、训练调试到推理部署的完整实践，深入理解了 ResNeXt 的核心设计思想、分组卷积与基数的概念，以及与 ResNet 的结构差异。

---

## 目录

1. [项目简介与动机](#1-项目简介与动机)
2. [ResNeXt 核心思想](#2-resnext-核心思想)
3. [ResNeXt vs ResNet 结构对比](#3-resnext-vs-resnet-结构对比)
4. [分组卷积代码实现的关键点](#4-分组卷积代码实现的关键点)
5. [训练过程与问题解决](#5-训练过程与问题解决)
6. [学习总结与思考](#6-学习总结与思考)
7. [改进模型的具体措施](#7-改进模型的具体措施)

---

## 1. 项目简介与动机

### 1.1 ResNeXt

在深度学习计算机视觉领域，**网络深度**（层数）一直是提升模型性能的主要手段——从 AlexNet（8 层）到 VGG（16-19 层），再到 ResNet（50-152 层），网络越来越深。但是，单纯增加深度带来了**梯度消失/爆炸**、**计算量剧增**和**优化困难**等问题。

ResNet 通过**残差连接**（Residual Connection）解决了深层网络退化问题。然而，一个更深层的问题浮现出来：**除了增加深度，还有哪些维度可以扩展模型容量？**

ResNeXt 给出的答案是：**增加基数（Cardinality）**。这让我意识到一个重要的思想转变——**模型设计不只是在"深度"这一维度上堆叠，而是可以在"宽度"和"基数"多个维度上进行精细化设计**。

### 1.2 数据集

本项目使用一个 101 类图像数据集，按以下比例划分：

| 数据集 | 样本数 | 用途 |
|-------|-------|------|
| 训练集（train） | 6,941 张 | 模型参数学习 |
| 验证集（val） | 2,379 张 | 模型选择与超参数调优 |
| 测试集（test） | 1,366 张 | 最终性能评估 |
| **总计** | **10,686 张** | — |

数据集包含 101 个类别，涵盖动物（海马、大象、熊猫等）、乐器（吉他、萨克斯等）、交通工具（直升机、自行车等）、日常物品（台灯、剪刀等）等多种类别，是一个典型的中等规模图像分类任务。

---

## 2. ResNeXt 核心思想

### 2.1 什么是基数（Cardinality）？

**基数**（Cardinality）是 ResNeXt 引入的一个新的网络维度，表示**变换路径的数量**。

在传统 ResNet 中，每个残差块只有一条变换路径：

```
输入 → [卷积→BN→ReLU→卷积→BN] → 求和 → ReLU → 输出
       ↑__________一条路径___________↓
```

而在 ResNeXt 中，每个残差块包含**多条并行的变换路径**（即基数的值），这些路径的结果**聚合**在一起：

```
输入 → [分组1: 卷积→BN→ReLU] ─┐
       [分组2: 卷积→BN→ReLU] ─┼─→ 拼接/求和 → BN → ReLU → 输出
       [分组3: 卷积→BN→ReLU] ─┘
       ↑____多个变换__聚合____↓
```

> **简而言之**：基数就是"用多少条不同的小网络并行处理输入，再把结果合并起来"。

本项目使用的是 **ResNeXt-101 (8×32d)**：
- **8** = 基数（8 条并行路径）
- **32** = 每条路径的基础宽度（d）
- **32×8 = 256** = 中间通道数

### 2.2 分组卷积如何实现？

分组卷积（Grouped Convolution）是实现 ResNeXt "多条并行路径"的**高效手段**。

#### 传统卷积

标准卷积中，每个输出通道与**所有**输入通道相连：

```
输入: C_in 个通道
      ┌─ 每个输出通道与 C_in 个输入通道做 3×3 卷积
      输出: C_out 个通道
      参数量: 3×3 × C_in × C_out
```

#### 分组卷积

分组卷积将输入/输出通道分成 **G 组**，每组独立做卷积：

```
输入: C_in 个通道
  ├── 组1 (C_in/G 通道) → 3×3卷积 → 组1输出 (C_out/G 通道)
  ├── 组2 (C_in/G 通道) → 3×3卷积 → 组2输出 (C_out/G 通道)
  ├── ...
  └── 组G (C_in/G 通道) → 3×3卷积 → 组G输出 (C_out/G 通道)
  每组独立卷积，结果拼接 → C_out 个通道输出
  参数量: 3×3 × (C_in/G) × (C_out/G) × G = 3×3 × C_in × C_out / G
```

**关键理解**：分组卷积的参数量是标准卷积的 **1/G**。这意味着在相同计算预算下，我们可以使用**更宽的中间层**（更大的 mid_ch），从而提升模型容量。

### 2.3 为什么增加基数比增加深度或宽度更有效？

这是 ResNeXt 论文中最核心的发现。论文通过对比实验发现：

| 策略 | 方法 | 计算量（FLOPs） | 准确率 |
|------|------|----------------|--------|
| 增加深度 | ResNet-152（更深） | 相同 | 较低 |
| 增加宽度 | 更宽的 ResNet-200 | 相同 | 较低 |
| **增加基数** | **ResNeXt-101 (8×32d)** | **相同** | **最高** |

我个人总结的三个原因：

#### ① 参数量更高效（计算预算的"杠杆效应"）

分组卷积将参数量降为 `1/G`，这意味着我们可以用同样的计算预算，把中间通道数做得更大（G 倍）。更多的通道意味着更强的特征表达能力。

#### ② 多样化特征学习（不同组学习不同特征）

每组卷积独立学习，相当于有 G 个"专家"分别关注不同特征子空间。有些组可能学习纹理特征，有些组学习形状特征，有些组学习局部模式……最终的聚合让模型获得更丰富的特征表示。

> 这让我联想到 ensemble（集成学习）的思想——多个弱学习器组合优于单个强学习器。ResNeXt 的分组卷积相当于在**网络内部**实现了轻量级的集成。

#### ③ 更好的正则化效果

分组限制了每组卷积的感受野只覆盖 1/G 的输入通道，这天然起到了**正则化**作用，减少了过拟合风险。这也解释了为什么 ResNeXt 通常比同等计算量的宽 ResNet 有更好的泛化性能——从训练日志中也可以看到，最终训练准确率（99.9%+）远高于验证准确率（~83%），说明模型仍在过拟合，但基数带来的正则化效果减缓了这一过程。不过我在后续中采用了其他方式来解决这个问题。

---

## 3. ResNeXt vs ResNet 结构对比

### 3.1 核心差异图解

```
ResNet 瓶颈块（Bottleneck Block）:
输入 (256-d)
    │
    ├─→ 1×1 Conv (64-d) ─→ 3×3 Conv (64-d) ─→ 1×1 Conv (256-d) ─→+
    │                                                              │
    └──────────────────────── 恒等连接 ────────────────────────────┘
                                                                   ↓
                                                                ReLU 输出

ResNeXt 残差块（ResNeXt Block）:
输入 (256-d)
    │
    ├─→ 1×1 Conv (128-d)
    │         │
    │    ┌────┼────┬────┬────┬────┬────┬────┬────┐
    │    │ 组1│ 组2│ 组3│ 组4│ 组5│ 组6│ 组7│ 组8│   ← 基数=8
    │    │3×3 │3×3 │3×3 │3×3 │3×3 │3×3 │3×3 │3×3 │
    │    └────┴────┴────┴────┴────┴────┴────┴────┘
    │              ↓ (拼接)
    ├─→ 1×1 Conv (256-d) ─────────────────────────→+
    │                                                 │
    └─────────────────── 恒等连接 ────────────────────┘
                                                      ↓
                                                   ReLU 输出
```

### 3.2 详细对比表格

| 对比维度 | ResNet | ResNeXt |
|---------|--------|---------|
| **核心创新** | 残差连接（恒等映射） | 聚合残差变换（分组卷积） |
| **残差块结构** | 1×1→3×3→1×1（单路径） | 1×1→3×3(groups=G)→1×1（多路径聚合） |
| **中间层卷积** | 标准 3×3 Conv | 分组 3×3 Conv（groups=cardinality） |
| **维度扩展策略** | 增加深度（层数） | 增加基数（并行路径数） |
| **参数效率** | 基准 | 相同参数量下容量更大 |
| **计算量（同层）** | 基准 | 约 1/G（分组卷积减少） |
| **中间通道数公式** | bottleneck_channels × 4 | cardinality × base_width |
| **易优化程度** | 良好（残差设计） | 更优（内置正则化） |
| **分组数 G** | 1（无分组） | 8/16/32（通常） |
| **代表模型** | ResNet-50, ResNet-101, ResNet-152 | ResNeXt-50, ResNeXt-101 |
| **ImageNet Top-1** | ~77.37% (ResNet-101) | ~79.56% (ResNeXt-101, 8×64d) |

### 3.3 最直观的代码差异

在代码层面，ResNet 和 ResNeXt 的瓶颈块的差异**仅有一行代码**：

```python
# ResNet 瓶颈块——标准卷积
self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, 
                       stride=stride, padding=1, bias=False)

# ResNeXt 瓶颈块——分组卷积（多了 groups=cardinality）
self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3,
                       stride=stride, padding=1, 
                       groups=cardinality,  # ← 唯一的差异！
                       bias=False)
```

**一行代码的改动**，却带来了性能的显著提升——这就是 ResNeXt 设计的精妙之处。

---

## 4. 分组卷积代码实现的关键点

### 4.1 `groups` 参数详解

在 PyTorch 中，`nn.Conv2d` 的 `groups` 参数控制分组卷积的行为：

```python
torch.nn.Conv2d(
    in_channels,    # 输入通道数
    out_channels,   # 输出通道数
    kernel_size,    # 卷积核大小
    groups=1,       # 分组数（默认1=标准卷积）
    ...
)
```

**`groups` 参数的不同取值**：

| groups 值 | 行为 | 适用场景 |
|-----------|------|---------|
| `groups=1` | 标准卷积，所有输入通道连接到所有输出通道 | 普通卷积层 |
| `groups=C_in` | 深度可分离卷积（Depthwise Conv），每个输入通道独立卷积 | MobileNet 等轻量网络 |
| `groups=G`（1<G<C_in） | 分组卷积，将通道分成 G 组 | **ResNeXt** 的核心 |

### 4.2 关键约束：整除关系

使用分组卷积时，**必须满足整除关系**：

```python
# 必须满足的两个整除条件：
assert in_channels % groups == 0  # 输入通道数能被组数整除
assert out_channels % groups == 0  # 输出通道数能被组数整除
```

在 ResNeXt 块中，3×3 卷积层两边的通道数相同（都是 `mid_ch`），所以只需要满足：

```python
assert mid_ch % cardinality == 0  # 即 (cardinality × base_width) % cardinality == 0
```

因为 `mid_ch = cardinality × base_width`，这个整除条件**天然满足**——这是设计者巧妙的地方，通过公式从根本上保证了兼容性。

### 4.3 ResNeXtBlock 初始化参数详解

```python
class ResNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cardinality, base_width, stride=1):
        # 中间通道数 = 基数 × 基础宽度
        mid_ch = cardinality * base_width
        
        self.conv2 = nn.Conv2d(
            mid_ch, mid_ch,           # 输入/输出通道相同
            kernel_size=3,            # 3×3 卷积核
            stride=stride,            # 步长（控制下采样）
            padding=1,                # 保持空间尺寸
            groups=cardinality,       # 分组数 = 基数
            bias=False                # BN 后不需要偏置
        )
```

**参数数值关系**（以 Layer3 为例）：

```
in_ch = 512, out_ch = 1024
cardinality = 8, base_width = 64
mid_ch = 8 × 64 = 512

conv1: 512 → 512 (1×1, 降维? 其实保持了通道数)
conv2: 512 → 512 (3×3, groups=8, 每组处理 512/8=64 通道)
       ┌─────────────────────────────────────┐
       │ 每组输入: 64 通道                    │
       │ 每组输出: 64 通道                    │
       │ 参数量: 3×3×64×64 = 36,864          │
       │ 总参数量: 36,864 × 8 = 294,912      │
       │ 标准卷积参数量: 3×3×512×512 = 2,359,296 │
       │ 节省: 2,359,296 / 294,912 = 8 倍     │
       └─────────────────────────────────────┘
conv3: 512 → 1024 (1×1, 升维)
```

### 4.4 `_make_stage` 构建阶段的实现

```python
def _make_stage(self, in_ch, out_ch, cardinality, base_width, blocks, first_stride):
    # 阶段由多个 ResNeXtBlock 组成
    stages = []
    
    # 第一个块可能进行下采样（stride=2），同时改变通道数
    stages.append(ResNeXtBlock(in_ch, out_ch, cardinality, base_width, first_stride))
    
    # 后续块保持空间尺寸和通道数不变（stride=1）
    for _ in range(1, blocks):
        stages.append(ResNeXtBlock(out_ch, out_ch, cardinality, base_width, stride=1))
    
    return nn.Sequential(*stages)
```

**四个阶段的参数配置**：

| 阶段 | 输入→输出通道 | 基数 | base_width | mid_ch | 块数 | 首个步长 |
|------|-------------|------|-----------|--------|------|---------|
| Layer1 | 64→256 | 8 | 16 | 128 | 2 | 1 |
| Layer2 | 256→512 | 8 | 32 | 256 | 2 | 2 |
| Layer3 | 512→1024 | 8 | 64 | 512 | 2 | 2 |
| Layer4 | 1024→2048 | 8 | 128 | 1024 | 2 | 2 |

注意 `base_width` 逐层翻倍的设计——越到深层，特征越抽象，需要更大的表示容量。

### 4.5 前向传播过程

```python
def forward(self, x):
    # 1. 残差路径（shortcut）
    identity = self.shortcut(x)
    
    # 2. 主路径：1×1 → 3×3(groups) → 1×1
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    
    out = self.conv2(out)  # 分组卷积（核心）
    out = self.bn2(out)
    out = self.relu(out)
    
    out = self.conv3(out)
    out = self.bn3(out)  # 注意：第三层后不加激活
                         # 目的是让 identity 信息在相加前保持原始特征
    
    # 3. 残差连接：主路径 + shortcut
    out += identity
    
    # 4. 最终激活
    return self.relu(out)
```

---

## 5. 训练过程与问题解决

### 5.1 训练过程与结果

使用 **ResNeXt-101 (8×32d)** 轻量版本（共 8 个残差块，约 15.90M 参数）训练 80 个 epoch，关键训练过程如下：

```
# ⬇ 初始阶段（Epoch 1-10）：快速学习期
Epoch 1:  Train Loss: 3.8546, Acc: 0.2344 | Val Loss: 3.0707, Acc: 0.3581
Epoch 5:  Train Loss: 2.1489, Acc: 0.4872 | Val Loss: 2.3500, Acc: 0.4536
Epoch 10: Train Loss: 1.2890, Acc: 0.6634 | Val Loss: 1.6371, Acc: 0.6028

# ⬇ 中期（Epoch 11-40）：性能提升期
Epoch 20: Train Loss: 0.4456, Acc: 0.8659 | Val Loss: 1.2580, Acc: 0.7146
Epoch 30: Train Loss: 0.1261, Acc: 0.9669 | Val Loss: 1.1765, Acc: 0.7650
Epoch 40: Train Loss: 0.0362, Acc: 0.9934 | Val Loss: 1.0431, Acc: 0.7953

# ⬇ 后期（Epoch 41-80）：过拟合加剧期
Epoch 50: Train Loss: 0.0136, Acc: 0.9976 | Val Loss: 0.9642, Acc: 0.8150
Epoch 60: Train Loss: 0.0089, Acc: 0.9991 | Val Loss: 0.8829, Acc: 0.8188
Epoch 70: Train Loss: 0.0071, Acc: 0.9993 | Val Loss: 0.8589, Acc: 0.8272
Epoch 80: Train Loss: 0.0071, Acc: 0.9988 | Val Loss: 0.8549, Acc: 0.8285
```

**最终结果**：
- **验证最佳准确率 ≈ 83.40%**（Epoch 64）
- **训练准确率 ≈ 99.88%**（Epoch 80）
- **训练/验证差距 ≈ 17%**（过拟合明显）

### 5.2 遇到的问题和解决方法

#### 问题 1：模型初始导入路径错误（No module named 'model'）

**现象**：
```
❌ 模型初始化失败: No module named 'model'
```

**原因分析**：在目录重构后，`environment/verify_setup.py` 位于 `environment/` 子目录中。当运行 `python environment/verify_setup.py` 时，Python 会将 `environment/` 目录自动加入 `sys.path[0]`，而 `model.py` 在项目根目录，自然找不到。更糟糕的是，当运行 `python utils/predict.py` 时，Python 将 `utils/` 加入路径，导致 `from utils.utils import` 报错 `'utils' is not a package`。

**解决过程**——分两步走：

**第一步**：创建 `__init__.py` 文件，将目录标记为 Python 包：
```python
# utils/__init__.py
"""ResNeXt 项目工具函数包。"""
```

**第二步**：在脚本开头调整 sys.path：

```python
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

# 移除 Python 自动添加的脚本所在目录
if _current_dir in sys.path:
    sys.path.remove(_current_dir)

# 确保项目根目录在最前面
if _project_root in sys.path:
    sys.path.remove(_project_root)
sys.path.insert(0, _project_root)
```

**学习体会**：这个看似简单的问题让我深刻理解了 Python 的包导入机制和 `sys.path` 的搜索顺序。之前写 Python 脚本没有这个意识，因为单文件项目不会有这个问题。但随着项目规模增长，合理的包结构设计变得至关重要。**这不是一个 Bug，而是项目架构演进中必然遇到的工程问题**。

#### 问题 2：CUDA 兼容性检查和 Unicode 编码错误

**现象1**：在 Windows 系统上运行验证脚本时，部分 emoji 字符（如 📦、🔧）无法在 GBK 编码的控制台中正确显示，导致 `UnicodeEncodeError`：
```
UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f4e6'
```

**解决方法**：使用 `python -u` 参数运行（强制 Python 以 unbuffered 模式运行，输出不会被编码拦截），或者在脚本中添加以下处理：
```python
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
```

**现象2**：环境验证脚本无法检测到已安装的 CUDA，显示"未检测到系统CUDA"。

**分析**：由于本机没有安装 NVIDIA GPU 或 CUDA 驱动，这是正常现象。PyTorch 安装了 CPU 版本，所以 CUDA 不可用。脚本已经正确识别了此情况。

**学习体会**：这两个问题都很小，但让我深刻体会到了**深度学习环境的复杂性**——不仅仅是 Python 版本兼容，还有操作系统编码、GPU 驱动版本、CUDA 版本和 PyTorch 版本的匹配问题。`environment/verify_setup.py` 这个脚本就是为了系统化解决这些问题而生的。

---

#### 问题 3：训练指标计算错误

**现象**：之前的版本中，训练指标（损失、准确率）的计算使用了 `len(loader.dataset)` 作为总样本数，但这里的 `len(dataset)` 返回的是数据集的样本总数，而不是当前 epoch 实际处理的样本数。

**问题代码**（修复前）：
```python
train_loss += loss.item()
train_total += len(train_loader.dataset)  # ← 应为 1（一个 batch）
```

实际上正确的做法是累积计数器而不是使用 `len(dataset)`。修复后通过 `train_total` 和 `val_total` 变量逐 batch 累加，并在每个 epoch 结束后重置，保证了统计的准确性。

---

## 6. 学习总结与思考

### 6.1 核心收获

**① 理解了"多维度扩展"的设计哲学**

ResNeXt 教会我最重要的一课是：**模型设计是一个多维度的优化问题**。不要只盯着"增加深度"这一个维度。ResNeXt 通过增加基数，在**参数效率**、**计算成本**和**模型容量**之间找到了更好的平衡点。这启发我在解决其他问题时也应该跳出单一的思考维度。

**② 分组卷积 = 轻量级集成学习**

从直观理解上，分组卷积就像是网络内部的集成学习。每组卷积独立学习不同的特征子空间，最后聚合起来得到更丰富的特征表示。这和传统机器学习的 Bagging 思想有异曲同工之妙。

**③ 代码细节决定成败**

从 `groups` 参数的整除约束，到残差连接中第三层卷积后不加激活函数，再到 `__init__.py` 的创建——这些看似微小的代码细节，任何一个出错都会导致训练失败或性能低下。

**④ 工程能力与理论同等重要**

在这个项目中，我遇到的大多数问题不是理论层面的（理解 ResNeXt 结构），而是工程层面的（导入路径、编码问题、指标统计错误、包结构设计）。**深度学习开发 = 理论理解 × 工程能力**。两者缺一不可。

> **项目版本**：1.0.0  
> **最后更新**：2026 年 5 月 16 日  
