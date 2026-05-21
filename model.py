"""ResNeXt 图像分类模型的实现模块。

该模块使用 PyTorch 实现了 ResNeXt 架构。
ResNeXt （用于深度神经网络的聚合残差变换） 通过引入分组卷积来扩展 ResNet ，
从而在保持计算效率的同时提升模型容量。

模块包含：
    - ResNeXtBlock: 基础残差块（带分组卷积的瓶颈结构）
    - ResNeXt: 完整的ResNeXt-101分类网络
"""
import torch
import torch.nn as nn


class ResNeXtBlock(nn.Module):
    """ResNeXt基础残差块（使用分组卷积的瓶颈残差块）。
    
    该块实现了ResNeXt的核心结构，包含三个卷积层和残差连接。
    前向传播公式：output = ReLU(main_path(x) + shortcut(x))
    
    主路径结构：
        1x1 Conv (降维) → 3x3 GroupedConv (分组卷积) → 1x1 Conv (升维)
    
    残差连接（shortcut）：
        - 如果步长不为1或通道数改变，使用投影短接（1x1 Conv + BN ）
        - 否则使用恒等映射（Identity）
    
    Attributes:
        conv1 (nn.Conv2d): 第一个卷积层，用于降低通道维度。
        bn1 (nn.BatchNorm2d): 第一个批标准化层。
        conv2 (nn.Conv2d): 第二个卷积层，分组卷积核心。
        bn2 (nn.BatchNorm2d): 第二个批标准化层。
        conv3 (nn.Conv2d): 第三个卷积层，用于升高通道维度。
        bn3 (nn.BatchNorm2d): 第三个批标准化层。
        relu (nn.ReLU): 激活函数。
        shortcut (nn.Module): 残差连接模块。
    """

    def __init__(self, in_ch, out_ch, cardinality, base_width, stride=1):
        """初始化ResNeXtBlock模块。
        
        Args:
            in_ch (int): 输入特征图的通道数（上一层的输出通道数）。
            out_ch (int): 输出特征图的通道数（处理后的输出通道数）。
            cardinality (int): 分组卷积的组数（分组卷积的组数）。
                通常设为8，表示将特征分成8组独立进行卷积。
            base_width (int): 每个分组的基础宽度（d），即 mid_ch = cardinality * base_width。
                不同阶段取值不同：layer1取16, layer2取32, layer3取64, layer4取128。
            stride (int, optional): 卷积步长，默认为1。取值2时进行下采样，特征图空间尺寸减半。
        
        Note:
            中间通道数计算：mid_ch = cardinality × base_width
            例如，cardinality=8, base_width=32时，mid_ch=256
        """
        super().__init__()

        # 计算分组卷积的中间通道数
        mid_ch = cardinality * base_width

        # ============ 主路径（Main Path）============
        # 第1层：1×1卷积进行降维，减少后续分组卷积的计算量
        # in_ch -> mid_ch，卷积核大小为1×1
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        # 第2层：3×3分组卷积（ResNeXt的核心）
        # 将mid_ch个通道分成cardinality组，每组独立进行3×3卷积
        # 优点：相同计算量下学习容量提升，比单一深度卷积更高效
        self.conv2 = nn.Conv2d(
            mid_ch, mid_ch, kernel_size=3, stride=stride,
            padding=1, groups=cardinality, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_ch)

        # 第3层：1×1卷积进行升维，恢复到输出通道数
        # mid_ch -> out_ch
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        # ReLU激活函数：输出 = max(0, 输入)
        # inplace=True: 直接修改输入张量，节省内存
        self.relu = nn.ReLU(inplace=True)

        # ============ 残差连接（Shortcut Path）============
        # 当步长>1或通道数改变时，需要投影短接使尺寸和通道数匹配
        if stride != 1 or in_ch != out_ch:
            # 投影短接（Projection Shortcut）：使用1×1卷积+BN调整尺寸和通道数
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            # 恒等映射：输入和输出尺寸相同时，直接返回输入
            self.shortcut = nn.Identity()

    def forward(self, x):
        """前向传播（Forward Pass）。
        
        执行顺序：
            1. 计算残差连接路径 (短接)
            2. 计算主路径：conv1-bn1-relu → conv2-bn2-relu → conv3-bn3
            3. 将两路相加，再经过ReLU激活
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, in_ch, H, W]，其中：
                B: 批量大小（batch size）
                in_ch: 输入通道数
                H: 特征图高度
                W: 特征图宽度
            
        Returns:
            torch.Tensor: 输出张量，形状为 [B, out_ch, H', W']，其中：
                out_ch: 输出通道数
                H': 高度（可能因步长而变化）
                W': 宽度（可能因步长而变化）
        """
        # 获取残差连接的输出（不经过激活）
        identity = self.shortcut(x)

        # 主路径前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  # 第一个激活

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)  # 第二个激活

        out = self.conv3(out)
        out = self.bn3(out)  # 注意：第三层后不经过激活，为的是在加法前保留原始特征信息

        # 残差相加：融合主路径和残差连接
        # 这里不对出来的结果直接激活，因为需要让 identity 的信息参与加法运算
        out += identity

        # 最后通过激活函数
        return self.relu(out)


class ResNeXt(nn.Module):
    """ResNeXt图像分类网络的完整实现。
    
    ResNeXt (用于深度神经网络的聚合残差变换)
    是ResNet的扩展。相比ResNet只使用标准卷积，ResNeXt引入了分组卷积（grouped
    convolution），在相同计算复杂度下提升模型的学习容量。
    
    网络结构：
        Stem (预处理) → Layer1 → Layer2 → Layer3 → Layer4 →
        全局平均池化 → 全连接层 (分类)
    
    参数配置（ResNeXt-101, 8×32d）：
        - Layer1: 2个块，无下采样
        - Layer2: 2个块，步长2下采样
        - Layer3: 2个块，步长2下采样
        - Layer4: 2个块，步长2下采样
    
    Attributes:
        stem (nn.Sequential): 初始预处理模块。
        layer1-4 (nn.Sequential): 四个残差阶段。
        avgpool (nn.AdaptiveAvgPool2d): 全局平均池化。
        fc (nn.Linear): 分类全连接层。
    """

    def __init__(self, num_classes=101):
        """初始化ResNeXt网络。
        
        Args:
            num_classes (int, optional): 分类类别数，默认为101。用于ImageNet数据集。
                根据实际数据集调整此参数（如CIFAR-10为10，CIFAR-100为100）。
        
        Note:
            所有conv层都不使用偏置（bias=False），因为后面紧接BN层会处理偏置。
        """
        super().__init__()

        # ============ Stem（茎部）：初始预处理模块 ============
        # 作用：对输入图像进行初步特征提取，降低空间分辨率，提升计算效率
        # 输入：[B, 3, 224, 224] (RGB图像)
        # 输出：[B, 64, 56, 56] (7×7卷积+2×池化共4倍下采样)
        self.stem = nn.Sequential(
            # 大卷积核（7×7）快速提取低级特征，步长2进行首次下采样
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 最大池化进行第二次下采样
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ============ 四个残差阶段（Layer1-4）============
        # 每个阶段逐步提升通道数，并通过步长2进行空间下采样
        # 参数解释：_make_stage(输入通道, 输出通道, 基数, 基础宽度, 块数, 首个块步长)
        # 
        # 模型容量配置：ResNeXt-50标准结构
        # blocks = [3, 4, 6, 3] 相比原来的 [2, 2, 2, 2] 增加了 30% 的卷积层
        # 作用：更大的模型容量可以学习更复杂的特征表示，适合 101 个类别的分类任务
        self.layer1 = self._make_stage(
            in_ch=64, out_ch=256, cardinality=8, base_width=16, 
            blocks=3, first_stride=1  # 从2→3个块，无下采样
        )
        self.layer2 = self._make_stage(
            in_ch=256, out_ch=512, cardinality=8, base_width=32,
            blocks=4, first_stride=2  # 从2→4个块，下采样2×
        )
        self.layer3 = self._make_stage(
            in_ch=512, out_ch=1024, cardinality=8, base_width=64,
            blocks=6, first_stride=2  # 从2→6个块，下采样2×
        )
        self.layer4 = self._make_stage(
            in_ch=1024, out_ch=2048, cardinality=8, base_width=128,
            blocks=3, first_stride=2  # 从2→3个块，下采样2×
        )

        # ============ 分类头 ============
        # 全局平均池化：将特征图空间维度平均为1×1
        # 输入：[B, 2048, 7, 7] → 输出：[B, 2048, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        # Dropout层：防止过拟合，随机丢弃20%的神经元
        # 在训练时启用（model.train()），在评估时禁用（model.eval()）
        # 作用：适度增加模型泛化能力，p=0.2 在保留足够特征信息的同时提供正则化
        self.dropout = nn.Dropout(p=0.2)

        # 全连接分类层：将2048维特征映射到类别数
        # 输入：[B, 2048] → 输出：[B, num_classes]
        self.fc = nn.Linear(2048, num_classes)

    def _make_stage(self, in_ch, out_ch, cardinality, base_width, blocks, first_stride):
        """构建单个残差阶段（Stage），由多个ResNeXtBlock堆叠而成。
        
        阶段结构：
            [ResNeXtBlock(stride=first_stride)] → [ResNeXtBlock]* → ... → [ResNeXtBlock]
            第一个块的步长可能为2（下采样），其余块步长为1（保持空间尺寸）。
        
        Args:
            in_ch (int): 该阶段的输入通道数。
            out_ch (int): 该阶段的输出通道数（所有块输出相同）。
            cardinality (int): 分组卷积的组数（通常为8）。
            base_width (int): 中间通道的基础宽度参数。
                计算公式：mid_ch = cardinality × base_width
                不同阶段值：layer1→16, layer2→32, layer3→64, layer4→128
            blocks (int): 该阶段包含的ResNeXtBlock数量。
            first_stride (int): 第一个块的步长，取值1或2。
                stride=1: 保持特征图空间尺寸
                stride=2: 特征图宽高各减半（下采样）
            
        Returns:
            nn.Sequential: 由多个ResNeXtBlock按顺序组成的模块。
        """
        layers = []
        # 第一个块：可能进行下采样（stride可能为2），通道数从in_ch变为out_ch
        layers.append(
            ResNeXtBlock(in_ch, out_ch, cardinality, base_width, stride=first_stride)
        )
        # 后续块：步长为1（保持尺寸），通道数保持为out_ch
        for _ in range(1, blocks):
            layers.append(
                ResNeXtBlock(out_ch, out_ch, cardinality, base_width, stride=1)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        """完整的前向传播过程。
        
        传播路径：
            Input → Stem → Layer1 → Layer2 → Layer3 → Layer4 →
            AvgPool → Flatten → FC → Logits
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为 [B, 3, 224, 224]，其中：
                B: 批量大小（batch size）
                3: RGB三个颜色通道
                224: 图像高度（像素）
                224: 图像宽度（像素）
            
        Returns:
            torch.Tensor: 分类logits（未经softmax），形状为 [B, num_classes]。
                每个元素代表样本属于对应类别的原始分数。
        
        Shape变化示例 (B=32, num_classes=101)：
            Input:      [32, 3, 224, 224]
            After Stem: [32, 64, 56, 56]    (2×2下采样)
            After Layer1: [32, 256, 56, 56]
            After Layer2: [32, 512, 28, 28] (2×2下采样)
            After Layer3: [32, 1024, 14, 14](2×2下采样)
            After Layer4: [32, 2048, 7, 7]  (2×2下采样)
            After AvgPool: [32, 2048, 1, 1]
            After Flatten: [32, 2048]
            Output:     [32, 101] (分类logits)
        """
        # Stem：初始特征提取
        x = self.stem(x)  # [B, 3, 224, 224] → [B, 64, 56, 56]

        # 四个残差阶段
        x = self.layer1(x)  # [B, 64, 56, 56] → [B, 256, 56, 56]
        x = self.layer2(x)  # [B, 256, 56, 56] → [B, 512, 28, 28]
        x = self.layer3(x)  # [B, 512, 28, 28] → [B, 1024, 14, 14]
        x = self.layer4(x)  # [B, 1024, 14, 14] → [B, 2048, 7, 7]

        # 全局平均池化：将空间维度平均到1×1
        x = self.avgpool(x)  # [B, 2048, 7, 7] → [B, 2048, 1, 1]

        # 展平：转换为二维张量以进行全连接
        x = torch.flatten(x, start_dim=1)  # [B, 2048, 1, 1] → [B, 2048]

        # Dropout：在训练时随机丢弃，在评估时保留所有神经元
        # 作用：防止过拟合，提升泛化能力
        x = self.dropout(x)  # [B, 2048] → [B, 2048] (或部分元素为0)

        # 分类全连接层：生成类别logits
        x = self.fc(x)  # [B, 2048] → [B, num_classes]

        return x


# ============ 模型验证 ============
if __name__ == "__main__":
    """模型自检：验证网络结构和参数数量。"""
    # 创建模型实例（101个ImageNet类别）
    model = ResNeXt(num_classes=101)

    # 创建虚拟输入张量用于测试（批量大小=2）
    dummy_input = torch.randn(2, 3, 224, 224)

    # 前向传播测试
    output = model(dummy_input)

    # 打印输出形状
    print(f"✓ Input shape:  {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")

    # 计算总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    total_params_millions = total_params / 1e6
    print(f"✓ Total parameters: {total_params_millions:.2f}M")

    # 计算可训练参数（通常与总参数相同，除非有冻结层）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters: {trainable_params/1e6:.2f}M")
