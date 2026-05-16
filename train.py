"""ResNeXt模型训练模块。

该模块实现了ResNeXt深度神经网络的完整训练流程。包括：
    1. 模型初始化和训练配置
    2. 数据加载和预处理
    3. 训练循环和参数更新
    4. 验证集评估和最佳模型保存
    5. 训练日志记录

训练策略：
    - 使用SGD优化器结合动量（momentum=0.9）和权重衰减（L2正则化）
    - 采用余弦退火（Cosine Annealing）学习率调度器逐步降低学习率
    - 使用交叉熵损失函数进行多分类任务
    - 每个epoch进行一次验证，保存最佳模型

模块功能：
    - train(): 执行ResNeXt模型的完整训练过程
    
注意事项：
    - 确保data/路径下有train/、val/、test/三个子目录，各包含类别子文件夹
    - GPU可用时自动使用GPU训练（速度快10-50倍）
    - 训练过程中持续监控验证准确率，自动保存最佳权重
    - 使用tqdm库显示训练进度条
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
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from model import ResNeXt
from mydataset import get_dataloaders
from environment.device_utils import parse_device_arg, setup_device


def train():
    """执行ResNeXt模型的完整训练过程，包含参数优化和模型评估。
    
    该函数实现了深度学习模型的标准训练流程：
        1. 检测硬件设备（GPU/CPU）并初始化模型
        2. 加载训练、验证数据集
        3. 配置优化器和学习率调度策略
        4. 逐个epoch进行以下操作：
           - 训练阶段：前向传播→计算损失→反向传播→参数更新
           - 验证阶段：评估模型在验证集上的性能
           - 根据验证准确率保存最佳模型权重
        5. 记录训练日志并输出最终结果
    
    训练流程图：
        初始化 → 数据加载 → [For each epoch:] →
        训练循环 → 计算指标 → 验证评估 → 保存模型 → 学习率调整
    
    Returns:
        None: 函数直接保存模型权重和训练日志到本地文件
        输出文件：
            - model-out/best.pth: 验证准确率最高的模型权重
            - model-out/last.pth: 最后一个epoch的模型权重
            - train_log.txt: 每个epoch的训练和验证指标记录
    
    Key Concepts (关键概念):
        - Epoch (轮数): 遍历整个训练集一次
        - Batch (批): 一次前向传播处理的样本组
        - Loss (损失): 衡量预测与真实标签的差异程度
        - Gradient (梯度): 损失对参数的偏导数，用于参数更新
        - Momentum (动量): 加速收敛的优化技术，考虑历史梯度
        - Learning Rate (学习率): 参数更新的步长
        - Validation (验证): 用独立数据集评估模型泛化性能
    
    Example:
        >>> train()
        📱 使用设备: cuda
        🚀 开始训练ResNeXt模型...
        Epoch 1: Train Loss: 4.1234, Acc: 0.0512 | Val Loss: 3.8901, Acc: 0.1234
        ...
        Epoch 80: Train Loss: 0.0234, Acc: 0.9876 | Val Loss: 0.3456, Acc: 0.8765
        ✅ 训练完成。最佳验证准确率: 0.8765
    
    Note:
        - 模型会自动加载到检测到的最优设备（CUDA GPU或CPU）
        - 训练时模型处于train模式，启用Dropout和BatchNorm更新
        - 验证时模型处于eval模式，禁用Dropout和BatchNorm更新
        - 使用torch.no_grad()在验证时禁用梯度计算以节省显存
        - SGD优化器配合momentum提高收敛稳定性
        - 余弦退火学习率从高→低，模拟退火加速探索→精细化过程
    """
    
    # ============ 第一步：解析命令行参数 ============
    # 支持 --device 参数指定运行设备
    parser = argparse.ArgumentParser(description="ResNeXt模型训练")
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'gpu', 'cpu'],
                       help='选择计算设备: auto=自动检测, gpu=强制GPU, cpu=强制CPU')
    args = parser.parse_args()
    
    # ============ 第二步：初始化设备 ============
    # 根据参数选择并初始化计算设备，显示详细的硬件信息
    device_name = parse_device_arg(args)
    device = setup_device(device_name)
    
    # -------- 训练超参数（Hyperparameters） --------
    # 这些参数控制训练的行为，初学者可以调整以观察效果
    data_root = "data"                # 数据集根目录路径
    batch_size = 32                   # 每批处理32个样本
                                      # - 值越大：训练越快但显存占用多，梯度平均化
                                      # - 值越小：梯度更新频繁，但训练慢且不稳定
    epochs = 80                       # 共训练80个epoch（完整遍历数据集80次）
    lr = 0.01                         # 初始学习率为0.01
                                      # - 控制每次参数更新的步长
                                      # - 过大：模型不收敛；过小：收敛太慢
    
    print("="*50)
    print("🚀 开始训练ResNeXt模型...")
    print("="*50)
    print(f"📊 配置参数: batch_size={batch_size}, epochs={epochs}, lr={lr}")
    
    # ============ 第三步：数据加载 ============
    # 从指定目录加载训练、验证、测试集
    # 该函数自动进行数据增强和标准化处理
    train_loader, val_loader, _, classes = get_dataloaders(data_root, batch_size)
    num_classes = len(classes)        # 类别总数，例如101个类别
    
    print(f"📚 类别数: {num_classes}")
    print(f"📦 训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
    
    # ============ 第四步：模型初始化 ============
    # 创建ResNeXt神经网络，指定输出类别数
    # to(device)将模型参数移动到指定设备（GPU或CPU）
    model = ResNeXt(num_classes=num_classes).to(device)
    print(f"✓ 模型初始化完成（参数数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M）")
    
    # ============ 第五步：损失函数配置 ============
    # CrossEntropyLoss = Softmax + LogLoss，标准的多分类任务损失函数
    # 作用：测量模型预测的概率分布与真实标签的差异
    # 损失值越小表示预测越准确
    criterion = nn.CrossEntropyLoss()
    
    # ============ 第六步：优化器配置 ============
    # SGD: 随机梯度下降（Stochastic Gradient Descent）
    # model.parameters(): 模型的所有可训练参数（权重和偏置）
    optimizer = optim.SGD(
        model.parameters(),           # 指定要优化的参数
        lr=lr,                        # 学习率：参数更新步长
        momentum=0.9,                 # 动量：加权过去的梯度，加速收敛
        weight_decay=1e-4             # L2正则化强度，防止过拟合
                                      # 公式：loss_new = loss + weight_decay * ||params||^2
    )
    
    # ============ 第七步：学习率调度器配置 ============
    # CosineAnnealingLR: 余弦退火，让学习率按余弦函数规律从高降低
    # 好处：早期学习率高，快速探索最优值；后期学习率低，精细化优化
    # T_max=epochs: 学习率的周期长度（这里等于总epoch数）
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # ============ 第八步：创建模型和日志保存目录 ============
    # 创建目录用于保存训练好的模型权重和训练日志
    # exist_ok=True: 如果目录已存在则不报错
    os.makedirs("model-out", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    
    # 打开文件用于记录每个epoch的训练日志
    # 模式"w": 写模式，如果文件存在则覆盖
    log_path = os.path.join("log", "train_log.txt")
    log_file = open(log_path, "w")
    
    # ============ 第九步：初始化最佳准确率跟踪 ============
    # 用于保存验证集上的最高准确率
    # 当验证准确率超过该值时，保存当前模型为最佳模型
    best_acc = 0.0
    
    print("✓ 所有配置完成，开始训练循环\n")


    # ============ 第十步：主训练循环 ============
    for epoch in range(1, epochs+1):
        print(f"\n{'='*60}")
        print(f"📍 Epoch {epoch}/{epochs}")
        print(f"{'='*60}")
        
        # ===================== 训练阶段（Training Phase） =====================
        # 在此阶段，模型不断学习：前向传播→计算损失→反向传播→参数更新
        
        # model.train() 将模型设置为训练模式
        # 重要作用：
        #   1. 启用Dropout：随机丢弃神经元以防止过拟合
        #   2. 启用BatchNorm更新：使用当前批次的统计信息更新全局参数
        model.train()
        
        # 初始化用于统计的变量
        train_loss = 0.0              # 累计训练损失（所有批次）
        train_correct = 0             # 累计预测正确的样本数
        train_total = 0               # 累计处理的总样本数
        
        # tqdm: 进度条库，显示训练进度
        # desc: 进度条描述文本
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for imgs, labels in loop:
            # -------- 1. 数据准备 --------
            # 将图像和标签从CPU移到指定设备（GPU或CPU）
            # 这是PyTorch的必须操作：设备之间的张量不能直接计算
            imgs, labels = imgs.to(device), labels.to(device)
            # imgs 形状: [batch_size, 3, 224, 224] (图像)
            # labels 形状: [batch_size] (标签，取值0到num_classes-1)
            
            # -------- 2. 梯度清零 --------
            # optimizer.zero_grad() 清空上一次迭代的梯度
            # 重要！如果不清零，梯度会累积，导致参数更新错误
            # PyTorch默认梯度是累积的，这与NumPy不同
            optimizer.zero_grad()
            
            # -------- 3. 前向传播（Forward Pass） --------
            # model(imgs) 将图像输入模型，得到分类logits（原始分数）
            # outputs 形状: [batch_size, num_classes]
            # 每个元素是模型对该类别的预测分数（未归一化）
            outputs = model(imgs)
            
            # -------- 4. 计算损失 --------
            # criterion(outputs, labels) 计算交叉熵损失
            # 损失值衡量预测有多"错误"：
            #   - 损失=0 表示完全正确预测
            #   - 损失>0 表示有错误，值越大错误越严重
            loss = criterion(outputs, labels)
            
            # -------- 5. 反向传播（Backward Pass） --------
            # loss.backward() 计算损失对所有参数的梯度
            # 使用链式法则从输出层逐层向后计算梯度
            # 时间复杂度：约等于前向传播的2-3倍
            loss.backward()
            
            # -------- 6. 参数更新（Optimization Step） --------
            # optimizer.step() 根据梯度更新模型参数
            # 参数更新公式 (SGD with momentum):
            #   v = momentum * v + lr * gradient  (速度更新)
            #   param = param - v                  (参数更新)
            # momentum项使参数更新更稳定，加速收敛
            optimizer.step()
            
            # -------- 7. 训练指标统计 --------
            # 累积当前批次的损失（乘以样本数是为了后续计算平均）
            # loss.item() 将损失张量转为Python浮点数
            # imgs.size(0) 获取当前批次的样本数
            train_loss += loss.item() * imgs.size(0)
            
            # 获取预测类别（最高分数对应的类别）
            # torch.max(outputs, dim=1) 返回最大值和对应的索引
            # 索引就是预测的类别（0到num_classes-1）
            _, preds = torch.max(outputs, 1)
            
            # 统计预测正确的样本数
            # (preds == labels) 逐元素比较，返回布尔张量
            # .sum().item() 统计True的个数
            train_correct += (preds == labels).sum().item()
            
            # 累加当前批次的样本数
            train_total += imgs.size(0)
            
            # 更新进度条显示当前批次的损失
            loop.set_postfix(loss=loss.item())
        
        # -------- 训练阶段指标计算 --------
        # 学习率调度步进：根据余弦退火公式更新学习率
        # 余弦退火：学习率 = 初始lr * (1 + cos(π * t / T_max)) / 2
        # t是当前epoch，T_max是总epoch数
        # 效果：学习率从高→低，像退火降温一样
        scheduler.step()
        
        # 平均损失 = 总损失 / 总样本数
        # 这样计算保证指标不受batch_size的影响
        train_loss /= train_total
        
        # 训练准确率 = 正确预测数 / 总样本数
        train_acc = train_correct / train_total
        
        # ===================== 验证阶段（Validation Phase） =====================
        # 在此阶段，评估模型在从未见过的验证集上的性能
        # 用于判断模型是否过拟合以及是否收敛
        
        # model.eval() 将模型设置为评估模式
        # 重要作用：
        #   1. 禁用Dropout：使用所有神经元获得确定的预测
        #   2. 冻结BatchNorm：使用全局统计参数而非批次统计
        model.eval()
        
        # 初始化验证阶段的统计变量
        val_loss = 0.0                # 累计验证损失
        val_correct = 0               # 累计预测正确的样本数
        val_total = 0                 # 累计处理的总样本数
        
        # torch.no_grad() 禁用自动求导（梯度计算）
        # 优点：
        #   1. 节省显存（不保存反向传播所需的中间变量）
        #   2. 加快推理速度（约快10-20%）
        #   3. 防止意外的梯度累积
        with torch.no_grad():
            # 遍历验证集的所有批次
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                # 数据迁移到设备
                imgs, labels = imgs.to(device), labels.to(device)
                
                # 前向传播（验证时无需反向传播）
                outputs = model(imgs)
                
                # 计算验证损失
                # 使用相同的损失函数，便于对比训练和验证性能
                loss = criterion(outputs, labels)
                
                # 累积验证损失
                val_loss += loss.item() * imgs.size(0)
                
                # 获取预测类别
                _, preds = torch.max(outputs, 1)
                
                # 统计预测正确的样本数
                val_correct += (preds == labels).sum().item()
                
                # 累加样本数
                val_total += imgs.size(0)
        
        # -------- 验证阶段指标计算 --------
        # 平均验证损失
        val_loss /= val_total
        
        # 验证准确率
        val_acc = val_correct / val_total
        
        # ============ 第十步：日志记录和模型保存 ============
        
        # -------- 构建日志信息 --------
        # 将各项指标格式化成易读的字符串
        log_msg = (
            f"Epoch {epoch}: "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )
        
        # 写入日志文件
        # "\n" 换行符，每个epoch一行
        log_file.write(log_msg + "\n")
        
        # 打印日志到控制台
        print(f"📊 {log_msg}")
        
        # -------- 模型保存策略 --------
        # 保存最佳模型（基于验证准确率）
        # 这样即使后续训练变差，也能保留最佳的模型权重
        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save 保存模型参数（权重和偏置）
            # model.state_dict() 返回模型的所有可训练参数字典
            # .pth 是PyTorch权重文件的标准扩展名
            torch.save(model.state_dict(), "model-out/best.pth")
            print(f"✨ 新的最佳模型！验证准确率: {val_acc:.4f}")
        
        # 保存最后一个epoch的模型权重
        # 作为备选方案，防止最佳模型意外丢失
        torch.save(model.state_dict(), "model-out/last.pth")
    
    # ============ 第十一步：训练完成 ============
    # 关闭日志文件，释放文件资源
    log_file.close()
    
    print("\n" + "="*60)
    print("✅ 训练完成")
    print("="*60)
    print(f"🏆 最佳验证准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"💾 最佳模型已保存: model-out/best.pth")
    print(f"💾 最后模型已保存: model-out/last.pth")
    print(f"📝 训练日志已保存: train_log.txt")
    print("="*60)


# ============ 程序入口点 ============
if __name__ == "__main__":
    """主程序入口。
    
    当该脚本直接运行时（而非被导入为模块），执行train()函数。
    这是Python的标准做法，确保模块在被其他脚本导入时不会自动执行。
    
    使用方式：
        $ python train.py
    
    脚本会输出：
        - 实时的训练进度条
        - 每个epoch的训练和验证指标
        - 最佳模型的验证准确率
        - 保存的模型文件位置
    """
    train()
