import torch
import torch.nn as nn

class ResNeXtBlock(nn.Moudle):
    def __init__(self, in_ch, out_ch, cardinality=8, base_width=16, stride=1):
        """
        in_ch:        输入通道数
        out_ch:       输出通道数
        cardinality:  基数（分组数）
        base_width:   每个分支的宽度，即每分支通道数 d 
        stride:       步长，用于下采样（第一个块的 stride=2)
        """

        super().__init__()
        # 中间通道 = 基数 × 每分支通道数（必须能被基数整除）
        mid_ch = cardinality * base_width







































