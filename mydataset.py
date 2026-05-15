"""数据加载与预处理模块。

使用PyTorch的torchvision库提供图像数据的加载和预处理功能。为训练、验证和测试创建
单独的数据加载器，应用适当的数据增强和标准化处理。

模块功能：
    - get_dataloaders: 创建训练/验证/测试的DataLoader和类别名称
    - 训练集：应用数据增强（随机裁剪、翻转、旋转、颜色变换）
    - 验证/测试集：无增强，仅进行中心裁剪和标准化
    - 使用ImageNet标准的均值和标准差进行归一化
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_root, batch_size=32, num_workers=2):
    """创建训练、验证、测试数据加载器（DataLoader）。
    
    该函数从指定目录加载图像数据，应用不同的预处理策略：
        - 训练集：应用数据增强来防止过拟合
        - 验证/测试集：仅进行必要的标准化，无数据增强
    
    数据目录结构要求：
        data_root/
        ├── train/
        │   ├── class_1/
        │   ├── class_2/
        │   └── ...
        ├── val/
        │   ├── class_1/
        │   ├── class_2/
        │   └── ...
        └── test/
            ├── class_1/
            ├── class_2/
            └── ...
    
    Args:
        data_root (str): 数据集的根目录路径。
        batch_size (int, optional): 每批数据包含的样本数，默认为32。
            - 值越大，训练速度越快，但内存占用增加
            - 值越小，梯度更新更频繁，但速度较慢
        num_workers (int, optional): 数据加载的工作线程数，默认为2。
            - 0: 在主进程加载数据（调试时推荐）
            - >0: 使用多进程加载，加速数据读取
    
    Returns:
        tuple: 包含四个元素的元组：
            - train_loader (DataLoader): 训练数据加载器
            - val_loader (DataLoader): 验证数据加载器
            - test_loader (DataLoader): 测试数据加载器
            - classes (list): 类别名称列表（按字母顺序）
    
    Note:
        - 所有图像被缩放至224×224像素（ResNeXt模型的标准输入尺寸）
        - 使用ImageNet的官方均值[0.485, 0.456, 0.406]和标准差[0.229, 0.224, 0.225]
        - 训练集启用shuffle，验证/测试集不启用shuffle
    
    Example:
        >>> train_loader, val_loader, test_loader, classes = get_dataloaders('data', batch_size=32)
        >>> for images, labels in train_loader:
        ...     print(images.shape)  # torch.Size([32, 3, 224, 224])
        ...     break
    """
    # ============ 训练集数据变换（Data Augmentation）============
    # 目的：增加训练数据的多样性，防止模型过拟合
    # 通过随机变换生成不同视角的相同对象，提高模型的泛化能力
    train_transform = transforms.Compose([
        # 随机缩放裁剪：随机比例裁剪后缩放到224×224
        # scale=(0.8,1.0) 表示裁剪面积占原图8%-100%
        # 作用：模拟不同物体大小和位置，增强模型对尺度变化的鲁棒性
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        
        # 随机水平翻转：50%概率进行左右翻转
        # 作用：增加数据多样性，利用图像的对称性
        transforms.RandomHorizontalFlip(),
        
        # 随机旋转：±10度范围内随机旋转
        # 作用：模拟拍摄角度的变化
        transforms.RandomRotation(10),
        
        # 颜色抖动（Color Jitter）：随机调整亮度、对比度、饱和度
        # 作用：适应不同光照条件和相机设置
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        
        # 转换为PyTorch张量：将PIL图像转换为[0, 1]范围的Tensor
        # 输出形状：[3, 224, 224] (C, H, W)
        transforms.ToTensor(),
        
        # 标准化：减去均值，除以标准差
        # 使用ImageNet的官方统计值（通过大规模数据统计得出）
        # 公式：normalized_image = (image - mean) / std
        # 作用：将输入数据标准化到相似的分布，加速模型收敛
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # R, G, B的均值
            std=[0.229, 0.224, 0.225]    # R, G, B的标准差
        )
    ])

    # ============ 验证/测试集数据变换（无增强）============
    # 目的：测试时无需增强，保持数据的一致性
    # 只进行必要的预处理：缩放、裁剪、标准化
    val_test_transform = transforms.Compose([
        # 缩放：将图像最小边缩放到256像素（保持宽高比）
        # 作用：统一图像尺寸，为后续裁剪做准备
        transforms.Resize(256),
        
        # 中心裁剪：从缩放后的图像中心裁剪224×224区域
        # 与训练时的随机裁剪不同，测试时使用确定的中心裁剪
        # 作用：保证测试的确定性和可复现性
        transforms.CenterCrop(224),
        
        # 转换为PyTensor
        transforms.ToTensor(),
        
        # 使用相同的标准化参数（必须与训练集一致）
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ============ 创建数据集（Dataset）============
    # 使用ImageFolder自动从目录结构推断类别
    # 每个子文件夹对应一个类别，自动生成类别标签
    train_set = datasets.ImageFolder(
        root=f"{data_root}/train",
        transform=train_transform
    )
    val_set = datasets.ImageFolder(
        root=f"{data_root}/val",
        transform=val_test_transform
    )
    test_set = datasets.ImageFolder(
        root=f"{data_root}/test",
        transform=val_test_transform
    )

    # ============ 创建数据加载器（DataLoader）============
    # DataLoader作用：
    #   1. 从Dataset中批量采样
    #   2. 多进程加速数据读取
    #   3. 自动合并成批处理的张量
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,        # 训练集每个epoch打乱顺序（重要！提高模型泛化能力）
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,       # 验证集不打乱（保持一致的评估）
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,       # 测试集不打乱（保证可复现性）
        num_workers=num_workers
    )

    # ============ 返回结果 ============
    # train_set.classes: 类别名列表，按字母顺序排列
    # 例如：['accordion', 'airplanes', 'anchor', ...]
    return train_loader, val_loader, test_loader, train_set.classes


# ============ 数据加载器测试 ============
if __name__ == "__main__":
    """快速测试：验证DataLoader的工作状态和数据格式。"""
    # 创建DataLoader
    train_loader, val_loader, test_loader, classes = get_dataloaders("data")
    
    # 打印数据集信息
    print("="*50)
    print("📊 数据集信息")
    print("="*50)
    print(f"✓ 类别数量: {len(classes)}")
    print(f"✓ 训练批次: {len(train_loader)}")
    print(f"✓ 验证批次: {len(val_loader)}")
    print(f"✓ 测试批次: {len(test_loader)}")
    
    # 获取第一个批次并检查数据形状
    imgs, labels = next(iter(train_loader))
    print("\n" + "="*50)
    print("🖼️  批次数据信息")
    print("="*50)
    print(f"✓ 图像张量形状: {imgs.shape}")
    print(f"  - 批量大小 (B): {imgs.shape[0]}")
    print(f"  - 颜色通道 (C): {imgs.shape[1]} (RGB)")
    print(f"  - 图像高度 (H): {imgs.shape[2]} (像素)")
    print(f"  - 图像宽度 (W): {imgs.shape[3]} (像素)")
    print(f"✓ 标签张量形状: {labels.shape}")
    print(f"✓ 前5个标签: {labels[:5].tolist()}")
    print(f"✓ 前5个标签对应类别: {[classes[i] for i in labels[:5].tolist()]}")
    print("\n✓ 数据加载测试完成！")
    