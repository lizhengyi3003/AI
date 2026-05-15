## 项目简介
本项目手动实现了一个轻量化 ResNeXt 网络，在给定的 101 类图像数据集上完成图像分类全流程。  
通过本项目，深入理解了 ResNeXt 的核心设计思想、分组卷积与基数的概念，并与 ResNet 进行了结构对比分析。

---

## 环境配置

### 系统要求
- Python 3.8+
- PyTorch 1.10+
- NVIDIA CUDA 工具包（GPU加速，可选）

### 依赖包
- pytorch - 深度学习框架
- torchvision - 计算机视觉库
- tqdm - 进度条显示
- Pillow - 图像处理
- matplotlib - 绘图（可选）

### 快速安装

#### ① CPU 版本（通用，推荐初次测试）
```bash
pip install -r requirements.txt
```

#### ② GPU 版本（需要 NVIDIA 显卡）
**方案 A - 自动安装（推荐）**
```bash
python install_pytorch.py
```
交互式选择设备，自动匹配CUDA版本并安装。

**方案 B - 手动指定 CUDA 版本**
```bash
# 查看系统CUDA版本：nvidia-smi
# 然后选择对应版本安装：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 将 cu118 替换为你的版本：cu118/cu121/cu124
```

### 环境验证
```bash
python environment/verify_setup.py
```
验证依赖包、项目文件、代码语法、模型初始化等。

---

## 文件结构
```
AI/
├── data/                   # 数据集（已划分 train/val/test）
│   ├── train/              # 训练集（7 份数据）
│   │   ├── accordion/
│   │   ├── airplanes/
│   │   └── ... (共 101 个类别)
│   ├── val/                # 验证集（2 份数据）
│   │   └── ...
│   └── test/               # 测试集（1 份数据）
│       └── ...
├── environment/            # 环境配置和验证工具
│   ├── device_config.py    # 设备检测（CUDA版本等）
│   ├── device_utils.py     # 设备初始化和硬件信息
│   ├── install_pytorch.py  # PyTorch一键安装脚本
│   └── verify_setup.py     # 环境和项目验证工具
├── utils/                  # 工具函数模块
│   ├── utils.py            # 推理工具函数
│   └── predict.py          # 推理预测脚本
├── log/                    # 训练日志目录
│   └── train_log.txt       # 每个epoch的loss和acc记录
├── model-out/              # 保存的模型权重
│   ├── best.pth            # 最佳验证准确率对应的权重
│   └── last.pth            # 最后一个epoch的权重
├── model.py                # ResNeXt 模型定义（手动实现）
├── mydataset.py            # 数据加载与增强
├── train.py                # 训练与验证脚本
├── test.py                 # 测试脚本
├── requirements.txt        # 项目依赖包列表
├── AI.code-workspace       # VS Code 工作区配置
└── README.md               # 本说明文档
```

---

## 命令行参数速查表

### train.py - 训练脚本
```bash
python train.py [--device {auto|gpu|cpu}]
```
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device` | 计算设备选择 | auto |

**选项说明**：
- `auto` - 自动检测（优先GPU，无GPU则用CPU）
- `gpu` - 强制使用GPU（无GPU时报错）
- `cpu` - 强制使用CPU

**示例**：
```bash
python train.py --device auto     # 自动检测
python train.py --device gpu      # 强制GPU训练
python train.py --device cpu      # CPU训练
```

### test.py - 测试脚本
```bash
python test.py [--device {auto|gpu|cpu}]
```

**选项说明**：同 train.py

**示例**：
```bash
python test.py --device auto
python test.py --device gpu
```

### utils/predict.py - 推理预测脚本
```bash
python utils/predict.py [--device {auto|gpu|cpu}]
```

**推理模式**（运行后交互输入）：
- 单张图片：输入图片路径 `path/to/image.jpg`
- 批量预测：输入 `batch path/to/directory/`
- 获取帮助：输入 `help`
- 退出程序：输入 `quit`

**示例**：
```bash
python utils/predict.py --device auto
# 进入交互模式，输入图片路径
> /data/test/cat.jpg
预测: cat, 置信度: 94.23%

> batch ./test_images/
✅ 成功预测: 5 张图片
```

---

## 数据集说明
- 总图像数：10,686 张，101 个类别。
- 已按 7:2:1 划分为训练集、验证集、测试集。
- 图像尺寸统一为 224×224 像素。
- 加载方式：使用 `torchvision.datasets.ImageFolder` 直接读取，无需额外划分。

---

## 模型核心思想与理论要点

### ① 什么是基数？分组卷积如何实现？为什么增加基数比增加深度或宽度更有效？

**基数（Cardinality）**  
基数是指 ResNeXt 块中并行变换路径的数量，即“拆分-变换-合并”策略中拆分出的独立分支数。论文中将这一新维度定义为 `C`。

**分组卷积实现**  
ResNeXt 使用**分组卷积** (Grouped Convolution) 来等价实现多分支聚合。在 PyTorch 中，通过设置 `nn.Conv2d(..., groups=C)`，将输入通道均分为 C 组，每组独立进行卷积，最后拼接输出。这种实现方式与图 3(a) 的并行瓶颈结构在数学上完全等价（论文 Fig. 3(c)）。

**增加基数更有效的原因**  
- **更强的表示能力**：在相同计算量下，更多分支意味着更丰富的特征子空间，能捕捉更多样化的模式（论文 Table 3、Table 4）。  
- **结构稀疏性**：分组卷积相当于稀疏连接，减少了参数间的耦合，降低了过拟合风险。  
- **扩展效率**：增加深度/宽度会线性或平方级增加参数量，而增加基数只需调整组数，可以在不改变全局输入/输出维度的情况下，更精细地控制模型容量。

### ② ResNeXt 与 ResNet 的结构差异对比

| 对比维度 | ResNet (Bottleneck) | ResNeXt (本项目实现) |
|----------|---------------------|----------------------|
| 残差块结构 | 1×1 降维 → 3×3 卷积 → 1×1 升维 | 1×1 降维 → **3×3 分组卷积（groups=C）** → 1×1 升维 |
| 基础宽度 | 固定中间通道数（如 64） | 中间通道 = 基数 × 每分支通道（如 C=8, d=16 → 128） |
| 核心维度 | 深度、宽度 | 深度、宽度、**基数（Cardinality）** |
| 并行路径 | 无（单一路径） | 有（C 条并行路径，由分组卷积实现） |
| 参数效率 | 较低 | 更高（同样参数/FLOPs 下精度更高） |

### ③ 分组卷积代码实现的关键点

在 `model.py` 的 `ResNeXtBlock` 中，关键代码如下：

```python
# 中间通道数 = 基数 × 每分支通道数
mid_ch = cardinality * base_width

# 1x1 降维
self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)

# 3x3 分组卷积：groups 参数等于基数
self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                       padding=1, groups=cardinality, bias=False)

# 1x1 升维
self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
```

**重点**：
- `groups=cardinality` 必须等于基数（本项目为 8）。
- `mid_ch` 必须能被 `cardinality` 整除，否则会报错（本项目 `cardinality * base_width` 自然整除）。
- 输入通道 `in_ch` 和输出通道 `out_ch` 的变化仅由 bottleneck 的第一层和第三层控制，与分组卷积的组数无关。

## 训练配置
| 配置项 | 参数值 |
|--------|--------|
| Batch Size | 32 |
| Epochs | 80 |
| Learning Rate | 0.01 |
| Optimizer | SGD (momentum=0.9, weight_decay=1e-4) |
| Loss Function | CrossEntropyLoss |
| 学习率调度 | CosineAnnealingLR |
| 输入尺寸 | 224×224×3 |

---

## 实验结果与日志

### 训练结果

训练完成后，项目会生成以下文件：

1. **model-out/best.pth** - 验证集准确率最高的模型权重
2. **model-out/last.pth** - 最后一个epoch的模型权重
3. **train_log.txt** - 每个epoch的训练和验证指标

**train_log.txt 示例**：
```
Epoch 1: Train Loss: 4.6058, Acc: 0.0182 | Val Loss: 4.5932, Acc: 0.0241
Epoch 2: Train Loss: 4.5821, Acc: 0.0298 | Val Loss: 4.5701, Acc: 0.0423
Epoch 3: Train Loss: 4.5234, Acc: 0.0512 | Val Loss: 4.4923, Acc: 0.0798
...
Epoch 80: Train Loss: 0.3421, Acc: 0.8927 | Val Loss: 0.5234, Acc: 0.8234
```

### 性能指标

- **最佳验证集准确率**：通常在 75%-90% 之间（取决于数据和硬件）
- **测试集准确率**：通常比验证集准确率略低 2-5%
- **训练时间**：约 4-8 小时（单卡 GPU）或 20-30 小时（CPU）

## 常见问题与解决方案

### Q1: 显存不足（Out of Memory）

**错误信息**：`RuntimeError: CUDA out of memory`

**解决方案**：
1. 减小 batch_size（改为 16 或 8）
2. 使用 CPU 训练（虽然会很慢）
3. 使用梯度累积技术

```python
# 梯度累积示例
accumulation_steps = 2
for i, (imgs, labels) in enumerate(train_loader):
    outputs = model(imgs.to(device))
    loss = criterion(outputs, labels.to(device)) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q2: 模型训练不收敛或收敛缓慢

**症状**：loss 一直很高，accuracy 停留在很低的值

**解决方案**：
1. **调整学习率**：
   - 收敛慢：改为 0.02 或 0.05
   - 不收敛：改为 0.001 或 0.0005
2. **增加训练轮数**：改 epochs 为 150 或 200
3. **检查数据**：确认数据加载和增强没有问题
4. **重新初始化模型**：可能初始化不好

### Q3: 过拟合（训练准确率高，验证准确率低）

**症状**：train_acc 接近 99%，但 val_acc 只有 70-80%

**解决方案**：
1. **增加数据增强**：在 mydataset.py 中调整
2. **增加 weight_decay**：改为 5e-4 或 1e-3
3. **减少模型复杂度**：虽然本项目不适合，但可以少用一些层
4. **添加 Dropout**：在 FC 层前添加 dropout

```python
# 在 ResNeXt.fc 之前添加
self.dropout = nn.Dropout(0.5)
x = self.dropout(x)  # 在 forward 中应用
```

### Q4: GPU/CUDA 不可用

**检查方法**：
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

**解决方案**：
1. 安装 NVIDIA GPU 驱动
2. 安装 CUDA 和 cuDNN
3. 重新安装 PyTorch：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Q5: 数据加载错误

**错误信息**：`FileNotFoundError: data/train` 不存在

**解决方案**：
1. 确保解压数据集到 data/ 目录
2. 检查目录结构是否为 data/train/class_name/image.jpg
3. 运行 `python verify_setup.py` 检查

### Q6: 模型权重加载失败

**错误信息**：`FileNotFoundError: model-out/best.pth`

**原因**：尚未训练模型

**解决方案**：
1. 先运行 `python train.py` 生成权重
2. 等待训练完成后再运行 test.py 或 predict.py

---

## environment/ 文件夹说明

本项目包含完整的设备配置和环境管理工具，自动处理GPU/CPU选择和PyTorch版本匹配。

### device_config.py - 设备检测模块
检测系统硬件配置和PyTorch安装状态。

**核心函数**：
- `detect_cuda_version()` - 检测系统CUDA版本
- `get_pytorch_cuda_version()` - 获取PyTorch CUDA版本
- `check_pytorch_installed()` - 检查PyTorch安装状态
- `validate_cuda_match()` - 验证版本匹配

**使用场景**：诊断GPU配置问题

### device_utils.py - 设备初始化模块
自动初始化计算设备，显示详细的硬件信息。

**核心函数**：
- `parse_device_arg()` - 解析命令行参数
- `setup_device()` - 初始化设备并显示硬件信息

**集成方式**：所有训练/测试脚本都通过此模块自动处理设备选择。

### install_pytorch.py - PyTorch安装脚本
一键安装对应版本的PyTorch。

**特点**：
- ✅ 自动检测系统CUDA版本
- ✅ 交互式设备选择
- ✅ 自动匹配版本安装
- ✅ 完整的安装验证

**使用**：
```bash
python environment/install_pytorch.py
# 或指定设备
python environment/install_pytorch.py --device gpu
```

### verify_setup.py - 环境验证工具
全面检查项目环境和配置。

**检查项**：
- ✅ Python依赖包（PyTorch、torchvision等）
- ✅ 项目文件完整性
- ✅ 数据集目录结构
- ✅ Python代码语法
- ✅ 模型初始化

**使用时机**：
- 项目初次设置后
- 修改依赖后
- 环境出现问题时

**使用**：
```bash
python environment/verify_setup.py
```

---

## 使用说明

### 1. 环境准备

**安装依赖**（推荐在虚拟环境中进行）：
```bash
pip install -r requirements.txt
```

**验证环境**：
```bash
python environment/verify_setup.py
```

详细的硬件信息将自动输出。

---

### 2. 训练模型

#### 标准训练（自动GPU检测）
```bash
python train.py
```

#### 指定设备训练
```bash
# GPU训练
python train.py --device gpu

# CPU训练
python train.py --device cpu

# 自动检测（默认）
python train.py --device auto
```

**训练过程说明**：
- ✅ 将在 `model-out/` 目录下自动创建权重保存文件夹
- ✅ 每个 epoch 的训练日志会记录在 `train_log.txt`
- ✅ 根据验证集准确率自动保存最佳权重为 `model-out/best.pth`
- ✅ 每个 epoch 完成后保存当前权重为 `model-out/last.pth`
- ✅ 共训练 80 个 epoch（可在 `train.py` 中修改 `epochs` 参数）

**训练日志示例**（train_log.txt）：
```
Epoch 1: Train Loss: 4.6058, Acc: 0.0182 | Val Loss: 4.5932, Acc: 0.0241
Epoch 2: Train Loss: 4.5821, Acc: 0.0298 | Val Loss: 4.5701, Acc: 0.0423
Epoch 3: Train Loss: 4.5234, Acc: 0.0512 | Val Loss: 4.4923, Acc: 0.0798
...
Epoch 80: Train Loss: 0.3421, Acc: 0.8927 | Val Loss: 0.5234, Acc: 0.8234
```

---

### 3. 测试模型

在测试集上评估模型性能：
```bash
python test.py
```

**输出示例**：
```
Test Accuracy: 0.8234
```

---

### 4. 模型推理与预测

**utils/predict.py** 提供了三种推理模式：单张预测、批量预测和交互式预测。

#### 交互式预测模式（推荐）

进入交互模式，逐个输入图片进行预测：
```bash
python utils/predict.py
```

**交互命令说明**：
```
🎯 进入交互预测模式
输入图片路径 (输入 'help' 获取帮助, 'quit' 退出):
> test.jpg                    # 单张预测
预测: cat, 置信度: 94.23%

> batch ./test_images/        # 批量预测目录
✅ 成功预测: 5 张图片

> help                         # 显示帮助
> quit                         # 退出程序
```

#### 单张预测代码使用

```python
import torch
from model import ResNeXt
from utils.utils import predict_single_image, load_model_weights, get_class_names, print_prediction_result

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和类别
classes = get_class_names("data")
model = ResNeXt(num_classes=len(classes)).to(device)
model = load_model_weights(model, "model-out/best.pth", device)
model.eval()

# 单张预测
result = predict_single_image("test.jpg", model, device, classes)
print_prediction_result(result)

# 获取预测结果
print(f"预测类别: {result['pred_class']}")
print(f"置信度: {result['confidence']:.2%}")
print(f"Top5: {list(zip(result['top5_classes'], result['top5_scores']))}")
```

**输出示例**：
```
============================================================
图片路径: test.jpg
预测类别: cat
置信度: 94.23%

Top5 预测:
  1. cat: 94.23%
  2. tiger: 3.45%
  3. leopard: 1.89%
  4. lion: 0.32%
  5. cheetah: 0.11%
============================================================
```

#### 批量预测代码使用

```python
from utils.predict import predict_batch

# 批量预测指定目录下的所有图片
results = predict_batch("./test_images/", output_file="predictions.txt")

# 查看结果
for result in results:
    print(f"{result['image_path']}: {result['pred_class']} ({result['confidence']:.2%})")
```

**输出文件 predictions.txt 格式**：
```
图片路径,预测类别,置信度,Top5预测
./test_images/cat.jpg,cat,94.23%,cat(94.23%); tiger(3.45%); leopard(1.89%); lion(0.32%); cheetah(0.11%)
./test_images/dog.png,dog,88.56%,dog(88.56%); wolf(7.89%); fox(2.34%); ...
```

#### 模型推理的工具函数

**utils/utils.py** 提供了以下工具函数：
- `load_model_weights()`: 加载模型权重
- `get_class_names()`: 获取类别名列表
- `predict_single_image()`: 对单张图片进行推理
- `print_prediction_result()`: 格式化打印预测结果
- `get_class_index_map()`: 获取类别名到索引的映射

```python
from utils.utils import (
    load_model_weights,
    get_class_names,
    predict_single_image,
    print_prediction_result,
    get_class_index_map
)

# 使用示例
classes = get_class_names("data")
class_map = get_class_index_map("data")
print(f"'cat' 的索引: {class_map['cat']}")
```

---

### 5. 数据集预处理（可选）

如果需要自定义数据处理或增强，可编辑 `mydataset.py` 中的 `train_transform` 和 `val_test_transform`：

```python
# 训练数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),  # 随机裁剪
    transforms.RandomHorizontalFlip(),                    # 水平翻转
    transforms.RandomRotation(10),                        # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])
```

---

### 6. 超参数调整

在 `train.py` 中修改以下参数进行实验：
```python
batch_size = 32       # 批大小（GPU 显存不足可改为 16）
epochs = 80           # 训练 epoch 数
lr = 0.01             # 初始学习率
```

**常见调整**：
| 参数 | 当前值 | 调整建议 |
|------|--------|----------|
| `batch_size` | 32 | 显存充足可改为 64；显存不足改为 16 |
| `epochs` | 80 | 快速测试改为 10；长期训练改为 150 |
| `lr` | 0.01 | 收敛缓慢改为 0.02；震荡改为 0.005 |
| `weight_decay` | 1e-4 | 过拟合改为 5e-4；欠拟合改为 1e-5 |

---

### 7. 环境和项目验证

使用 `verify_setup.py` 检查环境和项目配置是否正确：

```bash
python verify_setup.py
```

**检查项目**：
- ✅ Python 依赖包（PyTorch、torchvision、tqdm 等）
- ✅ 项目核心文件（model.py、train.py、test.py 等）
- ✅ 数据集目录（data/train/、data/val/、data/test/）
- ✅ Python 代码语法（py_compile 检查）
- ✅ 模型初始化（ResNeXt 模型能否正常创建和前向传播）

**输出示例**：
```
============================================================
ResNeXt 项目环境验证工具
============================================================

📦 检查必需的 Python 包...
✅ PyTorch              2.1.0
✅ torchvision          0.16.0
✅ tqdm                 4.65.0
✅ numpy                1.24.0
✅ Pillow               10.0.0

📂 检查项目文件...
✅ model.py            模型定义
✅ mydataset.py        数据加载
✅ train.py            训练脚本
✅ test.py             测试脚本

📂 检查数据集目录...
✅ data/train          (6941 张)
✅ data/val            (2379 张)
✅ data/test           (1366 张)

✓ 代码语法检查...
✅ model.py            正常
✅ mydataset.py        正常
✅ train.py            正常
✅ test.py             正常

🧠 检查模型初始化...
✅ 模型初始化成功
✅ 总参数量: 83.53M
✅ 前向传播正常，输出形状: torch.Size([2, 101])

============================================================
验证总结:
  依赖包:   ✅ 正常
  项目文件: ✅ 正常
  代码语法: ✅ 正常
  模型初始: ✅ 正常
============================================================

✅ 环境检查通过！可以开始训练:
   python train.py
```

**使用场景**：
- 初次设置项目时验证环境
- 修改依赖后检查兼容性
- 出现异常时诊断问题根源
- CI/CD 管道中自动验证

---

### 6. 超参数调整

在 `train.py` 中修改以下参数进行实验：
```python
batch_size = 32       # 批大小（GPU 显存不足可改为 16）
epochs = 80           # 训练 epoch 数
lr = 0.01             # 初始学习率
```

**常见调整**：
| 参数 | 当前值 | 调整建议 |
|------|--------|----------|
| `batch_size` | 32 | 显存充足可改为 64；显存不足改为 16 |
| `epochs` | 80 | 快速测试改为 10；长期训练改为 150 |
| `lr` | 0.01 | 收敛缓慢改为 0.02；震荡改为 0.005 |
| `weight_decay` | 1e-4 | 过拟合改为 5e-4；欠拟合改为 1e-5 |

---

## 参考资料
- 原始论文：Aggregated Residual Transformations for Deep Neural Networks (CVPR 2017) [arXiv](https://arxiv.org/abs/1611.05431)
- PyTorch 官方文档：[torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) (groups 参数)
- ResNeXt 论文作者何恺明的相关资料：[何恺明个人主页](https://kaiminghe.github.io/)

---
