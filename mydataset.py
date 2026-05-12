import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_root, batch_size=32, num_workers=2):
    """
    返回 train, val, test 三个 DataLoader 和类别名列表
    数据增强遵循通用做法，归一化使用 ImageNet 官方均值标准差
    """
    # 训练集：随机裁剪 + 水平翻转 + 颜色抖动（防止过拟合）
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),   # 随机裁剪到224
        transforms.RandomHorizontalFlip(),                    # 50%水平翻转
        transforms.RandomRotation(10),                        # 随机旋转±10度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 颜色增强
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],        # ImageNet 均值
                             std=[0.229,0.224,0.225])         # ImageNet 标准差
    ])

    # 验证/测试集：中心裁剪，无数据增强
    val_test_transform = transforms.Compose([
        transforms.Resize(256),                # 先缩放到256
        transforms.CenterCrop(224),            # 中心裁出224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    # ImageFolder 要求每个类别一个子文件夹
    train_set = datasets.ImageFolder(root=f"{data_root}/train", transform=train_transform)
    val_set   = datasets.ImageFolder(root=f"{data_root}/val",   transform=val_test_transform)
    test_set  = datasets.ImageFolder(root=f"{data_root}/test",  transform=val_test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 返回类别名列表（与文件夹名一致）
    return train_loader, val_loader, test_loader, train_set.classes


# 快速测试
if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders("data")
    print(f"Number of classes: {len(classes)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    # 打印第一张图片的尺寸和标签
    imgs, labels = next(iter(train_loader))
    print(f"Batch image shape: {imgs.shape}, labels: {labels[:5]}")