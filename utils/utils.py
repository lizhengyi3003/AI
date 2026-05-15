"""ResNeXt 项目辅助工具函数模块。

该模块提供了模型推理过程中所需的各种辅助工具函数，包括：
    1. 模型权重加载和管理
    2. 类别名称和映射获取
    3. 单张图片预测推理
    4. 预测结果格式化输出
    5. 类别索引映射

工具函数主要用于：
    - utils/predict.py: 推理预测脚本调用
    - test.py: 批量评估数据集
    - 用户自定义脚本: 集成模型推理功能

模块功能列表：
    - load_model_weights(): 加载保存的模型权重文件
    - get_class_names(): 获取数据集的类别名列表
    - predict_single_image(): 对单张图片进行预测推理
    - print_prediction_result(): 格式化打印预测结果
    - get_class_index_map(): 建立类别名与索引的映射

使用方法：
    from utils.utils import predict_single_image, load_model_weights, get_class_names
    
    # 加载模型和类别
    model = load_model_weights(model, "model-out/best.pth", device)
    classes = get_class_names("data")
    
    # 预测单张图片
    result = predict_single_image("test.jpg", model, device, classes)
    print_prediction_result(result)

注意事项：
    - 所有函数假设数据集目录结构为：data/train/class1/、data/train/class2/等
    - 图片预处理采用与验证集一致的方法（缩放、中心裁剪、标准化）
    - 使用ImageNet标准的均值和标准差进行标准化
    - 推理时需要将模型设置为eval模式
    - 建议使用torch.no_grad()上下文禁用梯度计算
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from typing import List, Dict, Tuple, Any

# ============================================================
# 模型加载和预测函数
# ============================================================


def load_model_weights(model: nn.Module, weight_path: str, device: torch.device) -> nn.Module:
    """加载保存的模型权重文件。
    
    该函数从指定路径加载PyTorch模型的权重，并验证文件是否存在。
    使用map_location参数确保权重可以加载到任意设备（GPU或CPU）。
    
    参数说明（Args）:
        model (torch.nn.Module): 初始化好的PyTorch模型对象。
            该模型的结构必须与保存权重时使用的模型结构完全一致，
            包括层数、参数形状等。
            例如：model = ResNeXt(num_classes=101)
        
        weight_path (str): 权重文件的保存路径。
            通常为.pth格式的PyTorch权重文件。
            路径支持相对路径和绝对路径。
            例如："model-out/best.pth" 或 "./weights/model.pth"
        
        device (torch.device): 目标计算设备。
            指定权重加载到的设备（GPU或CPU）。
            例如：torch.device("cuda") 或 torch.device("cpu")
    
    返回值（Returns）:
        torch.nn.Module: 加载权重后的模型对象。
            模型参数已更新为保存的权重，可直接用于推理。
    
    异常处理（Raises）:
        FileNotFoundError: 如果权重文件不存在，抛出此异常。
            提示用户检查路径是否正确、文件是否存在。
        RuntimeError: 如果权重形状与模型不匹配，PyTorch会抛出此异常。
            通常是由于模型结构不一致导致。
    
    使用示例（Example）:
        >>> import torch
        >>> from model import ResNeXt
        >>> from utils.utils import load_model_weights
        >>> 
        >>> # 设定设备
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> 
        >>> # 创建模型
        >>> model = ResNeXt(num_classes=101).to(device)
        >>> 
        >>> # 加载权重
        >>> model = load_model_weights(model, "model-out/best.pth", device)
        >>> ✓ 权重加载成功: model-out/best.pth
    
    注意事项（Note）:
        - 必须确保模型结构与权重文件对应，否则会报错
        - map_location参数自动处理设备不匹配问题（如GPU权重加载到CPU）
        - 加载成功后模型会立即更新，无需二次处理
        - 推荐在加载后立即调用 model.eval() 设置为评估模式
    """
    # ============ 第一步：验证权重文件是否存在 ============
    # 检查指定路径的权重文件是否存在
    # 这是一个防御性编程实践，避免后续加载失败时的困惑错误信息
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    
    # ============ 第二步：加载权重到模型 ============
    # torch.load()从.pth文件读取权重字典
    # map_location=device参数确保权重加载到指定设备
    # 例如：GPU权重可以直接加载到CPU或其他GPU，自动处理设备适配
    model.load_state_dict(torch.load(weight_path, map_location=device))
    
    # ============ 第三步：用户反馈 ============
    # 打印成功消息，确认权重已加载
    print(f"✓ 权重加载成功: {weight_path}")
    
    return model


def get_class_names(data_root: str) -> List[str]:
    """从数据集目录获取类别名列表。
    
    自动扫描数据集的训练集目录，读取所有子文件夹作为类别名。
    类别名按字母顺序排列，确保在不同运行中的一致性。
    
    参数说明（Args）:
        data_root (str): 数据集的根目录路径。
            期望目录结构为：
                data_root/
                ├── train/
                │   ├── class1/
                │   ├── class2/
                │   └── ...
                ├── val/
                └── test/
            
            例如："data" 或 "./datasets/ImageNet"
    
    返回值（Returns）:
        List[str]: 按字母顺序排列的类别名列表。
            每个元素是一个类别名（对应train目录下的子文件夹名）。
            例如：['accordion', 'airplanes', 'anchor', ..., 'yin_yang']
    
    异常处理（Raises）:
        FileNotFoundError: 如果data_root/train目录不存在，抛出此异常。
            通常是由于数据集路径错误或目录结构不符合要求。
    
    使用示例（Example）:
        >>> from utils.utils import get_class_names
        >>> 
        >>> # 获取数据集类别
        >>> classes = get_class_names("data")
        >>> ✓ 加载类别: 共 101 个
        >>> 
        >>> print(f"类别总数: {len(classes)}")
        >>> 类别总数: 101
        >>> print(f"前5个类别: {classes[:5]}")
        >>> 前5个类别: ['accordion', 'airplanes', 'anchor', 'ant', 'barrel']
    
    注意事项（Note）:
        - 类别名通过os.listdir()读取，不需要手动维护
        - 只会读取train/目录下的子文件夹，忽略文件
        - 返回的类别名经过排序，保证结果的确定性
        - 类别索引与列表顺序一一对应（0-indexed）
        - 多次调用会读取磁盘，如频繁使用建议缓存结果
    """
    # ============ 第一步：构建训练集目录路径 ============
    # 根据data_root和固定的"train"子目录名构建完整路径
    train_path = os.path.join(data_root, "train")
    
    # ============ 第二步：验证目录存在 ============
    # 检查训练集目录是否存在，不存在则抛出异常
    # 这个检查很重要，因为空目录会返回空列表，造成后续混淆
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练集目录不存在: {train_path}")
    
    # ============ 第三步：扫描类别子目录 ============
    # os.listdir()列出train/下的所有文件和文件夹
    # os.path.isdir()过滤出只有文件夹的项（排除.jpg等文件）
    # sorted()按字母顺序排列，确保一致性
    classes = sorted([
        d for d in os.listdir(train_path)           # 遍历train目录中的所有项
        if os.path.isdir(os.path.join(train_path, d))  # 只保留子目录
    ])
    
    # ============ 第四步：用户反馈 ============
    # 打印类别总数，方便用户验证是否正确加载
    print(f"✓ 加载类别: 共 {len(classes)} 个")
    
    return classes


def predict_single_image(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    classes: List[str],
    image_size: int = 224
) -> Dict[str, Any]:
    """对单张图片进行分类预测推理。
    
    该函数是完整的推理管道，包括图片加载、预处理、模型推理和结果处理。
    图片预处理方式与验证集一致，确保推理质量。
    
    推理流程：
        图片加载 → RGB转换 → 缩放 → 中心裁剪 → 标准化 →
        模型推理 → 概率计算 → Top5排序 → 结果返回
    
    参数说明（Args）:
        image_path (str): 输入图片的文件路径。
            支持相对路径和绝对路径。
            支持的格式：JPG、PNG、BMP、TIFF等PIL支持的格式。
            例如："test.jpg" 或 "./data/test_images/cat.png"
        
        model (torch.nn.Module): 已加载权重且处于eval模式的ResNeXt模型。
            模型必须：
                1. 已加载预训练权重（通过load_model_weights加载）
                2. 已设置为eval模式（model.eval()）
                3. 已移到指定设备（model.to(device)）
            例如：model = ResNeXt(num_classes=101).to(device)
        
        device (torch.device): 模型所在的计算设备。
            应与模型一致：torch.device("cuda") 或 torch.device("cpu")
            用于将输入张量移到同一设备进行计算。
        
        classes (List[str]): 类别名称列表。
            由get_class_names()获取的已排序类别列表。
            长度应等于模型的输出类别数。
            例如：['cat', 'dog', 'bird', ...]
        
        image_size (int, optional): 输入图片的目标尺寸。
            模型期望的输入图片尺寸（正方形）。
            默认224，对应ResNeXt标准设置。
            值越大越精确但推理越慢。
    
    返回值（Returns）:
        Dict[str, Any]: 包含详细预测结果的字典，键值对如下：
            - 'pred_class' (str): 预测的类别名
            - 'pred_idx' (int): 预测类别的索引（0到num_classes-1）
            - 'confidence' (float): 预测置信度，范围[0, 1]
            - 'top5_classes' (List[str]): Top5预测类别名列表（降序）
            - 'top5_scores' (List[float]): Top5置信度列表
            - 'image_path' (str): 输入图片的路径（便于追溯）
        
        返回示例：
            {
                'pred_class': 'cat',
                'pred_idx': 2,
                'confidence': 0.9567,
                'top5_classes': ['cat', 'tiger', 'leopard', 'lion', 'dog'],
                'top5_scores': [0.9567, 0.0321, 0.0089, 0.0018, 0.0005],
                'image_path': 'test.jpg'
            }
    
    异常处理（Raises）:
        FileNotFoundError: 如果图片文件不存在，抛出此异常。
        PIL.UnidentifiedImageError: 如果文件格式无法识别（损坏的图片）。
        RuntimeError: 如果模型前向传播出错。
    
    使用示例（Example）:
        >>> import torch
        >>> from model import ResNeXt
        >>> from utils.utils import predict_single_image, load_model_weights, get_class_names
        >>> 
        >>> # 初始化
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> classes = get_class_names("data")
        >>> model = ResNeXt(num_classes=len(classes)).to(device)
        >>> model = load_model_weights(model, "model-out/best.pth", device)
        >>> model.eval()
        >>> 
        >>> # 预测
        >>> result = predict_single_image("test_image.jpg", model, device, classes)
        >>> print(f"预测: {result['pred_class']}, 置信度: {result['confidence']:.2%}")
        >>> 预测: cat, 置信度: 95.67%
    
    注意事项（Note）:
        - 模型必须处于eval模式（禁用Dropout和BatchNorm更新）
        - 使用torch.no_grad()上下文禁用梯度计算以节省显存
        - 图片会自动转换为RGB格式（兼容灰度图等）
        - 标准化使用ImageNet官方参数，与训练时一致
        - Top5分数是softmax后的概率值，和为1
        - 推荐在batch处理时缓存模型和classes避免重复加载
    """
    # ============ 第一步：输入验证 ============
    # 检查图片文件是否存在，不存在则抛出异常
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # ============ 第二步：定义图片预处理管道 ============
    # 预处理步骤与验证集一致，确保推理与评估结果可比
    transform = transforms.Compose([
        # 将图片最小边缩放到256，保持宽高比
        # 作用：统一图片尺寸，为后续中心裁剪做准备
        transforms.Resize(256),
        
        # 从中心裁剪224×224的区域
        # 与训练集的随机裁剪不同，测试时使用确定的中心裁剪
        # 作用：保证推理的一致性和可复现性
        transforms.CenterCrop(image_size),
        
        # 将PIL图像转换为PyTorch张量（值范围[0, 1]）
        # 输出形状：[3, height, width]（C, H, W格式）
        transforms.ToTensor(),
        
        # ImageNet标准化：(x - mean) / std
        # 使用在ImageNet上统计得到的均值和标准差
        # 重要：这些参数必须与模型训练时使用的参数一致
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # R, G, B的均值
            std=[0.229, 0.224, 0.225]    # R, G, B的标准差
        )
    ])
    
    # ============ 第三步：加载并预处理图片 ============
    # 使用PIL库加载图片
    image = Image.open(image_path).convert('RGB')
    # 说明：convert('RGB')确保图片为3通道，兼容灰度图、RGBA等格式
    
    # 应用预处理变换
    image_tensor = torch.as_tensor(transform(image), dtype=torch.float32)
    # 说明：torch.as_tensor而非torch.tensor避免不必要的复制
    
    # 增加batch维度：[3, 224, 224] → [1, 3, 224, 224]
    # unsqueeze(0)在第0维添加一个维度
    # 模型期望的输入形状为[batch_size, channels, height, width]
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # ============ 第四步：模型推理 ============
    # torch.no_grad()禁用自动求导，节省显存并加快速度
    # 推理时无需计算梯度（只需前向传播）
    with torch.no_grad():
        # 前向传播得到logits（未归一化的预测分数）
        # outputs形状：[1, num_classes]
        outputs = model(image_tensor)
        
        # Softmax将logits转换为概率分布
        # dim=1表示在类别维度上应用softmax
        # 结果：[1, num_classes]，每个元素代表该类的概率
        probabilities = torch.softmax(outputs, dim=1)
    
    # ============ 第五步：提取预测结果 ============
    # 获取最高分数对应的类别索引（argmax）
    # outputs.argmax(dim=1)得到[1]形状的张量
    # .item()将单元素张量转为Python数值
    pred_idx = int(outputs.argmax(dim=1).item())
    
    # 根据索引获取类别名
    pred_class = classes[pred_idx]
    
    # 获取该类别的概率值（置信度）
    # probabilities[0, pred_idx]获取batch中第一个样本的预测类别概率
    # .item()转为Python float
    confidence = float(probabilities[0, pred_idx].item())
    
    # ============ 第六步：计算Top5预测 ============
    # torch.topk()获取最大的k个值和对应的索引
    # probabilities[0]是第一个样本的概率向量：[num_classes]
    # k=min(5, len(classes))处理类别数少于5的情况
    # 返回：(top_values, top_indices)
    top5_scores, top5_indices = torch.topk(probabilities[0], k=min(5, len(classes)))
    
    # 将索引转换为类别名
    # 逐个取出索引，转为Python int，从classes列表查询类别名
    top5_classes = [classes[int(idx.item())] for idx in top5_indices]
    
    # 将Top5分数转为Python列表
    # .cpu()移到CPU（以防GPU张量）
    # .numpy()转为numpy数组
    # .tolist()转为Python列表
    top5_scores_list = top5_scores.cpu().numpy().tolist()
    
    # ============ 第七步：组织返回结果 ============
    # 将所有结果保存在字典中，方便使用和传递
    result: Dict[str, Any] = {
        'pred_class': pred_class,        # 预测类别名
        'pred_idx': pred_idx,            # 预测类别索引
        'confidence': confidence,        # 置信度
        'top5_classes': top5_classes,    # Top5类别名
        'top5_scores': top5_scores_list, # Top5分数
        'image_path': image_path         # 图片路径（用于追溯）
    }
    
    return result


def print_prediction_result(result: Dict[str, Any]) -> None:
    """格式化打印预测结果到控制台。
    
    该函数美化predict_single_image()返回的结果字典，以易读的格式输出。
    包括图片路径、预测类别、置信度和Top5预测列表。
    
    参数说明（Args）:
        result (Dict[str, Any]): predict_single_image()函数返回的结果字典。
            必须包含以下键：
                - 'image_path': 图片路径
                - 'pred_class': 预测类别名
                - 'confidence': 置信度
                - 'top5_classes': Top5类别名列表
                - 'top5_scores': Top5置信度列表
    
    返回值（Returns）:
        None: 函数直接打印到控制台，无返回值
    
    使用示例（Example）:
        >>> from utils.utils import predict_single_image, print_prediction_result
        >>> 
        >>> result = predict_single_image("test.jpg", model, device, classes)
        >>> print_prediction_result(result)
        
        输出示例：
        ============================================================
        图片路径: test.jpg
        预测类别: cat
        置信度: 95.67%
        
        Top5 预测:
          1. cat: 95.67%
          2. tiger: 3.21%
          3. leopard: 0.89%
          4. lion: 0.18%
          5. dog: 0.05%
        ============================================================
    
    注意事项（Note）:
        - 调用前需确保result字典包含所有必要的键
        - 使用format formatting符号{:.2%}格式化百分比
        - 打印时自动添加边框和空行便于阅读
    """
    # ============ 构建输出内容 ============
    # 使用分隔线创建美观的边框
    print("\n" + "="*60)
    
    # 打印图片路径
    print(f"图片路径: {result['image_path']}")
    
    # 打印预测类别
    print(f"预测类别: {result['pred_class']}")
    
    # 打印置信度（格式化为百分比，保留2位小数）
    # {:.2%}格式符：乘以100并添加%符号
    print(f"置信度: {result['confidence']:.2%}")
    
    # 打印Top5预测列表
    print("\nTop5 预测:")
    
    # 遍历Top5类别和分数
    # enumerate()生成(索引, 值)对，start=1从1开始计数
    # zip()将两个列表并行迭代
    for i, (cls, score) in enumerate(zip(result['top5_classes'], result['top5_scores']), 1):
        # 格式：  1. cat: 95.67%
        #        2. tiger: 3.21%
        print(f"  {i}. {cls}: {score:.2%}")
    
    # 底部分隔线
    print("="*60 + "\n")


def get_class_index_map(data_root: str) -> Dict[str, int]:
    """获取类别名到索引的映射关系。
    
    该函数创建一个字典，将类别名（字符串）映射到对应的索引（整数）。
    可用于快速查询某个类别的索引值，或进行类别名和索引的相互转换。
    
    参数说明（Args）:
        data_root (str): 数据集根目录路径。
            与get_class_names()使用相同的目录结构。
            例如："data" 或 "./datasets/ImageNet"
    
    返回值（Returns）:
        Dict[str, int]: 类别名到索引的映射字典。
            键(key)：类别名（字符串），例如"cat"
            值(value)：类别索引（整数），例如0, 1, 2...
            例如：{'accordion': 0, 'airplanes': 1, ..., 'yin_yang': 100}
    
    异常处理（Raises）:
        FileNotFoundError: 如果data_root/train目录不存在（由get_class_names抛出）
    
    使用示例（Example）:
        >>> from utils.utils import get_class_index_map
        >>> 
        >>> # 获取映射
        >>> class_map = get_class_index_map("data")
        >>> 
        >>> # 查询类别索引
        >>> cat_idx = class_map['cat']
        >>> print(f"'cat' 的索引: {cat_idx}")
        >>> 'cat' 的索引: 15
        >>> 
        >>> # 检查类别是否存在
        >>> if 'dog' in class_map:
        ...     print(f"'dog' 的索引: {class_map['dog']}")
    
    注意事项（Note）:
        - 映射的顺序与get_class_names()返回的列表顺序一致
        - 可用于快速判断某个类别是否在数据集中
        - 内部调用get_class_names()，会打印"✓ 加载类别: ..."提示
        - 如频繁使用应缓存结果，避免重复调用
    """
    # ============ 第一步：获取排序后的类别列表 ============
    # 调用get_class_names()获取标准化的类别名列表
    # 该列表已按字母顺序排序，保证映射的确定性
    classes = get_class_names(data_root)
    
    # ============ 第二步：创建映射字典 ============
    # 字典推导式：遍历(索引, 类别名)对，创建{类别名: 索引}的字典
    # 例如：enumerate(['cat', 'dog', 'bird']) →
    #      [(0, 'cat'), (1, 'dog'), (2, 'bird')]
    # 转换为：{'cat': 0, 'dog': 1, 'bird': 2}
    return {cls: idx for idx, cls in enumerate(classes)}


# ============ 程序入口点 ============
if __name__ == "__main__":
    """工具函数测试和演示。
    
    当该脚本直接运行时（而非被导入为模块），执行简单的功能测试。
    这是Python的标准做法，确保模块在被其他脚本导入时不会自动执行。
    
    使用方式：
        $ python utils/utils.py
    
    测试内容：
        1. 加载并显示数据集类别
        2. 创建类别映射
        3. 输出示例类别信息
    
    用途：
        - 验证数据集结构是否正确
        - 检查类别加载是否成功
        - 演示工具函数的使用方法
    """
    print("ResNeXt 工具函数测试\n")
    
    # ============ 第一步：获取类别列表 ============
    # 尝试加载类别，测试get_class_names()函数
    try:
        classes = get_class_names("data")
        print(f"\n类别示例（前5个）: {classes[:5]}")
        
        # ============ 第二步：创建映射 ============
        # 获取类别索引映射，测试get_class_index_map()函数
        class_map = get_class_index_map("data")
        
        # 显示第一个类别的索引信息
        if classes:
            first_class = classes[0]
            idx = class_map[first_class]
            print(f"类别 '{first_class}' 的索引: {idx}")
    
    except FileNotFoundError as e:
        # 如果数据集不存在，给出友好的提示信息
        print(f"提示: {e}")
        print("请确保 data/ 目录存在并包含 train/ 子目录")
