"""模型测试和评估模块。

该模块用于在测试数据集上对训练好的ResNeXt模型进行评估，计算分类准确率。
测试过程包括：
    1. 加载训练好的模型权重
    2. 从测试集逐批次读取图像数据
    3. 进行前向推理预测
    4. 统计预测准确率并输出结果

功能：
    - test(): 在测试数据集上评估模型性能，计算准确率

注意事项：
    - 需要提前运行train.py生成model-out/best.pth权重文件
    - 测试时模型处于eval模式，不更新BatchNorm统计信息
    - 使用torch.no_grad()禁用梯度计算以加速推理
"""

import sys
import os
import argparse

# ============ 调整 sys.path 以支持直接运行脚本 ============
# 当直接从项目根目录运行时，Python 会自动添加当前目录到 sys.path
# 这里确保项目根目录优先级最高，便于模块导入
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir in sys.path:
    sys.path.remove(_current_dir)
sys.path.insert(0, _current_dir)

import torch
import argparse
from model import ResNeXt
from mydataset import get_dataloaders
from environment.device_utils import parse_device_arg, setup_device


def test():
    """在测试集上评估训练好的ResNeXt模型的分类准确率。
    
    该函数执行以下步骤：
        1. 自动检测可用硬件（GPU/CPU）并选择适当的计算设备
        2. 加载测试数据集，获取标准化的图像张量和对应的类别标签
        3. 初始化ResNeXt模型并加载预训练权重
        4. 将模型设置为评估模式（禁用dropout和BatchNorm更新）
        5. 逐批次进行前向传播推理
        6. 统计预测正确的样本数
        7. 计算并打印总体准确率
    
    流程图：
        读取测试数据 → 初始化模型 → 加载权重 → 评估模式 → 
        逐批推理 → 计算预测 → 统计正确数 → 计算准确率 → 打印结果
    
    Returns:
        None: 函数直接打印测试准确率到控制台
        输出格式: "Test Accuracy: {accuracy:.4f}"
        其中accuracy为0到1之间的浮点数，表示准确率
    
    Raises:
        FileNotFoundError: 如果找不到"model-out/best.pth"权重文件
        torch.cuda.OutOfMemoryError: 如果GPU内存不足（在batch_size过大时）
    
    Example:
        >>> test()
        Test Accuracy: 0.8567
        # 表示模型在测试集上的准确率为85.67%
    
    Note:
        - 模型会自动加载到检测到的最优设备上（CUDA GPU或CPU）
        - 测试时使用torch.no_grad()上下文，禁用梯度计算以节省显存
        - 使用model.eval()确保BatchNorm层使用全局统计参数而非批次统计
        - 准确率 = 正确预测数 / 总样本数
    """
    
    # ============ 第一步：解析命令行参数 ============
    # 支持 --device 参数指定运行设备
    parser = argparse.ArgumentParser(description="ResNeXt模型测试")
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'gpu', 'cpu'],
                       help='选择计算设备: auto=自动检测, gpu=强制GPU, cpu=强制CPU')
    parser.add_argument('--exp-id', type=str, default='01',
                       help='实验编号，对应训练时使用的 exp_id (默认: 01)')
    args = parser.parse_args()
    
    # 提取 exp_id 变量
    exp_id = args.exp_id
    
    # ============ 第二步：初始化设备 ============
    # 根据参数选择并初始化计算设备，显示详细的硬件信息
    device_name = parse_device_arg(args)
    device = setup_device(device_name)
    
    # ============ 第三步：加载测试数据 ============
    # 调用get_dataloaders函数从'data/test'目录加载测试集
    # 返回值说明：
    #   _: 训练集加载器（测试时不需要，用_忽略）
    #   _: 验证集加载器（测试时不需要，用_忽略）
    #   test_loader: 测试集数据加载器，包含批量化的图像和标签
    #   classes: 类别名称列表，长度即为类别数量（例如101个类别）
    _, _, test_loader, classes = get_dataloaders("data")
    
    print(f"📊 类别数: {len(classes)}")
    print(f"📦 测试批次数: {len(test_loader)}")
    
    # ============ 第四步：模型初始化和权重加载 ============
    # 创建ResNeXt模型实例，指定类别数为实际数据集的类别数
    # 这确保全连接层的输出维度与类别数相符
    model = ResNeXt(num_classes=len(classes)).to(device)
    
    # 从保存的权重文件加载预训练的模型参数
    # f"model-out/{exp_id}/best.pth"是在训练阶段保存的最佳模型
    # map_location=device确保权重正确加载到指定设备（GPU或CPU）
    # 这避免了GPU权重加载到CPU时的兼容性问题
    model.load_state_dict(torch.load(f"model-out/{exp_id}/best.pth", map_location=device))
    print(f"✓ 成功加载权重: model-out/{exp_id}/best.pth")
    
    # ============ 第五步：设置评估模式 ============
    # model.eval() 将模型设置为评估（推理）模式
    # 这样做的重要作用：
    #   1. 禁用Dropout层：在训练时Dropout随机丢弃神经元以防止过拟合，
    #      评估时应使用所有神经元获得确定的预测
    #   2. 冻结BatchNorm参数：使用训练时积累的全局统计信息（均值和方差）
    #      而非当前批次的统计信息，确保评估的一致性
    model.eval()
    print("✓ 模型已设置为评估模式")
    
    # ============ 第六步：初始化准确率统计变量 ============
    # correct: 记录预测正确的样本总数
    # total: 记录评估的总样本数
    # 准确率 = correct / total
    correct = 0
    total = 0
    
    print("\n" + "="*50)
    print("🧪 开始测试...")
    print("="*50)
    
    # ============ 第七步：批次循环和预测 ============
    # torch.no_grad() 上下文管理器：在此块内禁用自动求导
    # 优点：
    #   1. 减少显存占用（不需要保存梯度信息）
    #   2. 加快推理速度（约10-20%）
    #   3. 防止意外的梯度累积
    with torch.no_grad():
        # 从test_loader逐个获取批次数据
        # imgs: 批量图像张量，形状为 [batch_size, 3, 224, 224]
        # labels: 批量标签张量，形状为 [batch_size]，取值为0到(num_classes-1)
        for imgs, labels in test_loader:
            # -------- 数据迁移到计算设备 --------
            # 将CPU上的图像和标签张量移动到指定设备（GPU或CPU）
            # 这是PyTorch的关键操作：GPU只能处理GPU上的张量
            imgs, labels = imgs.to(device), labels.to(device)
            
            # -------- 前向传播 --------
            # model(imgs) 执行前向传播，返回模型的输出logits
            # 输出形状：[batch_size, num_classes]
            # 每个元素是模型对应类别的原始预测分数（未经softmax）
            outputs = model(imgs)
            
            # -------- 获取预测类别 --------
            # torch.max(outputs, 1) 返回两个值：
            #   _: 最大值（logits中的最大分数）
            #   preds: 最大值所在的索引（预测的类别）
            # dim=1表示在类别维度上求最大值
            # 例如：outputs=[0.1, 0.8, 0.1]时，preds=1（第二个类别）
            _, preds = torch.max(outputs, 1)
            
            # -------- 统计准确率 --------
            # labels.size(0) 获取当前批次的样本数
            # 例如：如果batch_size=32，则labels.size(0)=32
            total += labels.size(0)
            
            # (preds == labels) 逐元素比较，返回布尔张量
            # 例如：preds=[1,2,1], labels=[1,2,0] → [True,True,False]
            # .sum().item() 统计True的个数（预测正确的样本数）
            # 例如上面的例子返回2
            correct += (preds == labels).sum().item()
    
    # ============ 第七步：计算并显示结果 ============
    # 准确率 = 正确预测数 / 总样本数
    # 例如：correct=856, total=1000 → acc=0.856 (85.6%)
    acc = correct / total
    
    print("\n" + "="*50)
    print("✅ 测试完成")
    print("="*50)
    print(f"🎯 正确预测数: {correct} / {total}")
    print(f"📈 测试准确率: {acc:.4f} ({acc*100:.2f}%)")
    print("="*50)


# ============ 程序入口点 ============
if __name__ == "__main__":
    """主程序入口。
    
    当该脚本直接运行时（而非被导入为模块），执行test()函数。
    这是Python的标准做法，确保模块在被其他脚本导入时不会自动执行。
    
    使用方式：
        $ python test.py
    """
    test()
