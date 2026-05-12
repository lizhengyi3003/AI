"""
ResNeXt 项目辅助工具函数
包含单图预测、模型加载、类别映射等功能
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from typing import List, Dict, Tuple, Any


# ============================================================
# 参数文档辅助函数 - 集中管理所有参数说明
# ============================================================
def get_param_docs(param_name: str) -> str:
    """获取参数的详细说明文档
    
    Args:
        param_name: 参数名称
    
    Returns:
        参数的详细说明
    
    Example:
        print(get_param_docs('kernel_size'))
    """
    param_docs = {
        # Conv2d 参数
        'in_channels': '输入的通道数',
        'out_channels': '输出的通道数',
        'kernel_size': '卷积核大小（1 为 1x1 降维/升维，3 为 3x3 空间卷积）',
        'stride': '步长（1 保持分辨率，2 下采样减半）',
        'padding': '边界填充（保证输出尺寸）',
        'groups': '分组数（等于基数 cardinality 时实现分组卷积）',
        'bias': '是否使用偏置（通常与 BN 结合不用）',
        
        # BatchNorm2d 参数
        'num_features': '输入通道数（每个通道独立归一化）',
        'eps': '防止分母为零的小常数（默认 1e-5）',
        'momentum': '移动平均的动量（默认 0.1）',
        'affine': '是否学习可缩放和偏移参数 gamma 和 beta（默认 True）',
        'track_running_stats': '是否追踪运行时统计量（默认 True）',
        
        # ReLU 参数
        'inplace': '是否原地修改输入张量（节省内存但会改变原值）',
        
        # MaxPool2d 参数
        'dilation': '窗口中元素之间的间隔',
        'ceil_mode': '是否使用 ceil 而不是 floor 计算输出尺寸',
        
        # AdaptiveAvgPool2d 参数
        'output_size': '输出空间尺寸（如 1 表示 1x1，自动计算池化窗口）',
        
        # Linear 参数
        'in_features': '输入特征维度',
        'out_features': '输出特征维度',
    }
    return param_docs.get(param_name, '暂无文档')


# ============================================================
# Conv2d 包装函数（带参数文档）
# ============================================================
def Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False):
    """创建 Conv2d 层（带参数文档）
    
    Args:
        in_channels: 输入的通道数
        out_channels: 输出的通道数
        kernel_size: 卷积核大小（1 为 1x1 降维/升维，3 为 3x3 空间卷积）
        stride: 步长（1 保持分辨率，2 下采样减半）
        padding: 边界填充（保证输出尺寸）
        groups: 分组数（等于基数 cardinality 时实现分组卷积）
        bias: 是否使用偏置（通常与 BN 结合不用）
    
    Returns:
        nn.Conv2d: 卷积层
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups, bias)


# ============================================================
# 模型加载和预测函数
# ============================================================


def load_model_weights(model: nn.Module, weight_path: str, device: torch.device) -> nn.Module:
    """
    加载模型权重
    
    Args:
        model: PyTorch 模型
        weight_path: 权重文件路径（.pth 格式）
        device: 设备（cuda 或 cpu）
    
    Returns:
        加载权重后的模型
    
    Example:
        model = ResNeXt(num_classes=101).to(device)
        model = load_model_weights(model, "model-out/best.pth", device)
    """
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(f"✓ 权重加载成功: {weight_path}")
    return model


def get_class_names(data_root: str) -> List[str]:
    """
    从数据集目录获取类别名列表
    
    Args:
        data_root: 数据集根目录（包含 train/val/test 子目录）
    
    Returns:
        按字母顺序排列的类别名列表
    
    Example:
        classes = get_class_names("data")
        print(f"共有 {len(classes)} 个类别")
    """
    train_path = os.path.join(data_root, "train")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练集目录不存在: {train_path}")
    
    classes = sorted([d for d in os.listdir(train_path) 
                      if os.path.isdir(os.path.join(train_path, d))])
    
    print(f"✓ 加载类别: 共 {len(classes)} 个")
    return classes


def predict_single_image(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    classes: List[str],
    image_size: int = 224
) -> Dict[str, Any]:
    """
    对单张图片进行预测
    
    Args:
        image_path: 输入图片路径（RGB 格式）
        model: 已加载权重的 ResNeXt 模型（eval 模式）
        device: 设备（cuda 或 cpu）
        classes: 类别名列表
        image_size: 输入图片尺寸（默认 224）
    
    Returns:
        包含以下内容的字典：
        - 'pred_class': 预测类别名
        - 'pred_idx': 预测类别索引
        - 'confidence': 预测置信度（0-1）
        - 'top5_classes': Top5 预测类别（按置信度降序）
        - 'top5_scores': Top5 置信度分数
    
    Example:
        from model import ResNeXt
        from utils import predict_single_image, load_model_weights, get_class_names
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classes = get_class_names("data")
        
        model = ResNeXt(num_classes=len(classes)).to(device)
        model = load_model_weights(model, "model-out/best.pth", device)
        model.eval()
        
        result = predict_single_image("test_image.jpg", model, device, classes)
        print(f"预测: {result['pred_class']}, 置信度: {result['confidence']:.2%}")
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 图片预处理（与验证集一致）
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 加载并预处理图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = torch.as_tensor(transform(image), dtype=torch.float32).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    
    # 前向传播
    with torch.no_grad():
        outputs = model(image_tensor)  # [1, num_classes]
        probabilities = torch.softmax(outputs, dim=1)  # 获取概率分布
    
    # 获取预测结果
    pred_idx = int(outputs.argmax(dim=1).item())
    pred_class = classes[pred_idx]
    confidence = float(probabilities[0, pred_idx].item())
    
    # 获取 Top5 预测
    top5_scores, top5_indices = torch.topk(probabilities[0], k=min(5, len(classes)))
    top5_classes = [classes[int(idx.item())] for idx in top5_indices]
    top5_scores_list = top5_scores.cpu().numpy().tolist()
    
    result: Dict[str, Any] = {
        'pred_class': pred_class,
        'pred_idx': pred_idx,
        'confidence': confidence,
        'top5_classes': top5_classes,
        'top5_scores': top5_scores_list,
        'image_path': image_path
    }
    
    return result


def print_prediction_result(result: Dict[str, Any]) -> None:
    """
    格式化打印预测结果
    
    Args:
        result: predict_single_image() 返回的结果字典
    
    Example:
        result = predict_single_image("test.jpg", model, device, classes)
        print_prediction_result(result)
    """
    print("\n" + "="*60)
    print(f"图片路径: {result['image_path']}")
    print(f"预测类别: {result['pred_class']}")
    print(f"置信度: {result['confidence']:.2%}")
    print("\nTop5 预测:")
    for i, (cls, score) in enumerate(zip(result['top5_classes'], result['top5_scores']), 1):
        print(f"  {i}. {cls}: {score:.2%}")
    print("="*60 + "\n")


def get_class_index_map(data_root: str) -> Dict[str, int]:
    """
    获取类别名到索引的映射
    
    Args:
        data_root: 数据集根目录
    
    Returns:
        类别名 -> 索引 的字典
    
    Example:
        class_map = get_class_index_map("data")
        cat_idx = class_map['cat']
    """
    classes = get_class_names(data_root)
    return {cls: idx for idx, cls in enumerate(classes)}


if __name__ == "__main__":
    print("ResNeXt 工具函数测试")
    
    # 示例：获取类别
    try:
        classes = get_class_names("data")
        print(f"\n类别示例（前5个）: {classes[:5]}")
        
        # 获取类别映射
        class_map = get_class_index_map("data")
        if classes:
            print(f"类别 '{classes[0]}' 的索引: {class_map[classes[0]]}")
    
    except FileNotFoundError as e:
        print(f"提示: {e}")
        print("请确保 data/ 目录存在并包含 train/ 子目录")
