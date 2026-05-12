# 基础项目一：基于轻量化 ResNeXt 的图像分类

## 项目简介
本项目手动实现了一个轻量化 ResNeXt 网络，在给定的 101 类图像数据集上完成图像分类全流程。  
通过本项目，深入理解了 ResNeXt 的核心设计思想、分组卷积与基数的概念，并与 ResNet 进行了结构对比分析。

---

## 环境配置
- Python 3.8+
- PyTorch 1.10+
- torchvision
- tqdm
- matplotlib

快速安装：
```bash
pip install torch torchvision tqdm matplotlib
```

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
├── model.py                # ResNeXt 模型定义（手动实现）
├── mydataset.py            # 数据加载与增强
├── train.py                # 训练与验证脚本
├── test.py                 # 测试脚本
├── utils.py                # 辅助工具函数（单图预测、权重加载等）
├── model-out/              # 保存的模型权重
│   ├── best.pth            # 最佳验证准确率对应的权重
│   └── last.pth            # 最后一个 epoch 的权重
├── train_log.txt           # 训练日志（每个 epoch 的 loss 和 acc）
├── requirements.txt        # 项目依赖包列表
└── README.md               # 本说明文档
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

**结构图示**（可在此处插入论文中 Fig. 1 的左右对比图，或自己绘制的分组卷积等效图）

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

### ④ 训练过程中遇到的问题及解决方法

> **请在此记录你实际遇到的问题和解决方法，如实填写即可，例如：**

- **问题 1**：训练初期 loss 下降缓慢，验证准确率低。  
  **解决**：调整学习率从 0.01 到 0.005，并加入余弦退火调度，训练变得更加稳定。

- **问题 2**：显存不足（OOM），无法使用 batch size 32。  
  **解决**：将 batch size 减小到 16，并开启梯度累积（每 2 步更新一次参数），等效维持了原始 batch size 的效果。

- **问题 3**：训练后期出现过拟合（训练准确率高，验证准确率停滞或下降）。  
  **解决**：增加数据增强（RandomRotation、ColorJitter），调整 weight_decay 至 5e-4，并在全连接层前加入 Dropout(0.5)。

---

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

## 实验结果
- **最佳验证集准确率**：___%（请填写）
- **测试集准确率**：___%（请填写）
- 训练曲线（loss/acc 随 epoch 变化）见 `curves.png`。

---

## 使用说明

### 1. 环境准备

**安装依赖**（推荐在虚拟环境中进行）：
```bash
pip install -r requirements.txt
```

**验证环境**：
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

### 2. 训练模型

执行以下命令开始训练：
```bash
python train.py
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

### 4. 单张图片预测

对单张图片进行分类预测。创建 `predict.py`：
```python
import torch
from model import ResNeXt
from utils import predict_single_image, load_model_weights, get_class_names, print_prediction_result

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = "data"

# 加载类别
classes = get_class_names(data_root)

# 加载模型
model = ResNeXt(num_classes=len(classes)).to(device)
model = load_model_weights(model, "model-out/best.pth", device)
model.eval()

# 预测
result = predict_single_image("path/to/your/image.jpg", model, device, classes)

# 打印结果
print_prediction_result(result)
```

运行预测：
```bash
python predict.py
```

**输出示例**：
```
============================================================
图片路径: test_image.jpg
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

## 参考资料
- 原始论文：Aggregated Residual Transformations for Deep Neural Networks (CVPR 2017)
- PyTorch 官方文档：torch.nn.Conv2d (groups 参数)
- 考核要求文档：《ACU25届深度学习方向第二批招新》
