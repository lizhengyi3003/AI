import torch
import torch.nn as nn
from utils import Conv2d

class ResNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cardinality, base_width, stride=1):
        """ResNeXt 残差块
        
        Args:
            in_ch: 输入特征图的通道数
            out_ch: 输出特征图的通道数
            cardinality: 基数（分组数）
            base_width: 每个分支的通道数 d
            stride: 步长，默认为 1。下采样时设为 2，特征图尺寸会缩小一半
        """
        super().__init__()
        mid_ch = cardinality * base_width

        # 1x1 降维卷积 - kernel_size=1，调整通道数；bias=False，配合 BN 使用
        self.conv1 = Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch) # 对 mid_ch 个通道，每个通道独立做归一化

        # 3x3 分组卷积 - stride=步长（下采样），groups=基数（分组卷积），padding=1（保证尺寸）
        self.conv2 = Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)

        # 1x1 升维卷积 - kernel_size=1，升维到输出通道数；bias=False，配合 BN 使用
        self.conv3 = Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        # 残差连接 - 当尺寸或通道变化时，使用 1x1 卷积投影
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),  # 1x1 投影卷积，应用步长
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)                # 恒等映射或投影
        out = self.relu(self.bn1(self.conv1(x)))   # 降维 + BN + ReLU
        out = self.relu(self.bn2(self.conv2(out))) # 分组卷积
        out = self.bn3(self.conv3(out))            # 升维，注意 ReLU 在相加之后
        out += identity                            # 残差相加
        return self.relu(out)                      # 最后 ReLU

# ============================================================
# ResNeXt 整体网络：严格遵循 PDF 第5页结构
# ============================================================
class ResNeXt(nn.Module):
    def __init__(self, num_classes=101):
        """ResNeXt 图像分类网络
        
        Args:
            num_classes: 分类数，默认为 101
        """
        super().__init__()
        # Stage 0: stem 部分
        self.stem = nn.Sequential(
            Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 7x7 大核低级特征提取
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 3x3 最大池化
        )

        # Stage1～Stage4，参数严格按表格
        self.layer1 = self._make_stage(64,  256, 8, base_width=16, blocks=2, first_stride=1)   # Stage1: d=16
        self.layer2 = self._make_stage(256, 512, 8, base_width=32, blocks=2, first_stride=2)   # Stage2: d=32
        self.layer3 = self._make_stage(512, 1024, 8, base_width=64, blocks=2, first_stride=2)  # Stage3: d=64
        self.layer4 = self._make_stage(1024, 2048, 8, base_width=128, blocks=2, first_stride=2) # Stage4: d=128

        self.avgpool = nn.AdaptiveAvgPool2d(1)   # 全局平均池化 → 1x1
        self.fc = nn.Linear(2048, num_classes)   # 全连接分类器

    def _make_stage(self, in_ch, out_ch, cardinality, base_width, blocks, first_stride):
        """堆叠多个 ResNeXtBlock 构成一个 stage
        
        Args:
            in_ch: 输入通道数
            out_ch: 输出通道数
            cardinality: 分组基数
            base_width: 每个分支的基础宽度
            blocks: 该 stage 中包含的块数
            first_stride: 第一个块的步长
            
        Returns:
            nn.Sequential: 由多个 ResNeXtBlock 组成的序列模块
        """
        layers = []
        # 第一个块可能需要下采样
        layers.append(ResNeXtBlock(in_ch, out_ch, cardinality, base_width, stride=first_stride))
        # 后续块保持分辨率，输入输出通道相同
        for _ in range(1, blocks):
            layers.append(ResNeXtBlock(out_ch, out_ch, cardinality, base_width, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)              # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)          # [B, 2048]
        x = self.fc(x)                   # [B, num_classes]
        return x


# 快速验证模型输出尺寸（运行本文件即可测试）
if __name__ == "__main__":
    model = ResNeXt(num_classes=101)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)   # 应输出 torch.Size([2, 101])
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total/1e6:.2f}M")