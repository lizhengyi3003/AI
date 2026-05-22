"""训练过程可视化模块。

该模块用于绘制训练过程中的性能指标曲线，包括：
    1. Loss 曲线：训练损失 vs 验证损失
    2. Accuracy 曲线：训练准确率 vs 验证准确率
    3. 合并图表：Loss 和 Accuracy 合并为双轴图表

数据来源：
    从 train_log.txt 文件中读取训练日志，格式为：
    Epoch 1: Train Loss: 3.8546, Acc: 0.2344 | Val Loss: 3.0707, Acc: 0.3581
    
模块功能：
    - parse_log(): 解析训练日志文件
    - plot_loss(): 绘制 Loss 曲线
    - plot_accuracy(): 绘制 Accuracy 曲线
    - plot_combined(): 绘制合并双轴图
    - main(): 绘制所有图表
"""

import sys
import os
import re
import argparse
from typing import List, Dict, Any

# ============ 调整 sys.path 以支持直接运行脚本 ============
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir in sys.path:
    sys.path.remove(_current_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import matplotlib  # type: ignore
matplotlib.use('Agg')  # 使用非 GUI 后端，避免显示窗口
import matplotlib.pyplot as plt  # type: ignore
from matplotlib import rcParams  # type: ignore

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 100
rcParams['savefig.dpi'] = 150


def parse_log(log_file: str = "log/training/train_log.txt") -> Dict[str, Any]:
    """从训练日志文件中提取数据。
    
    解析 train_log.txt 文件，提取每个 epoch 的训练和验证指标。
    
    日志格式示例：
        Epoch 1: Train Loss: 3.8546, Acc: 0.2344 | Val Loss: 3.0707, Acc: 0.3581
        Epoch 2: Train Loss: 3.0880, Acc: 0.3322 | Val Loss: 2.8321, Acc: 0.3808
    
    Args:
        log_file (str): 训练日志文件路径。默认为 "log/training/train_log.txt"
    
    Returns:
        Dict[str, Any]: 包含以下键的字典：
            - 'epochs' (List[int]): Epoch 编号列表
            - 'train_loss' (List[float]): 训练损失列表
            - 'val_loss' (List[float]): 验证损失列表
            - 'train_acc' (List[float]): 训练准确率列表
            - 'val_acc' (List[float]): 验证准确率列表
    
    Raises:
        FileNotFoundError: 如果日志文件不存在
        ValueError: 如果无法解析日志内容
    """
    # 检查文件是否存在
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"日志文件不存在: {log_file}")
    
    # 初始化数据存储
    data = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # 正则表达式模式
    # 匹配格式：Epoch N: Train Loss: X.XXXX, Acc: 0.XXXX | Val Loss: X.XXXX, Acc: 0.XXXX
    pattern = r'Epoch (\d+):\s+Train Loss:\s+([\d.]+),\s+Acc:\s+([\d.]+)\s+\|\s+Val Loss:\s+([\d.]+),\s+Acc:\s+([\d.]+)'
    
    # 读取并解析日志文件
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 尝试匹配日志行
            match = re.match(pattern, line.strip())
            
            if match:
                # 提取匹配的数据
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                train_acc = float(match.group(3))
                val_loss = float(match.group(4))
                val_acc = float(match.group(5))
                
                # 添加到数据字典
                data['epochs'].append(epoch)
                data['train_loss'].append(train_loss)
                data['train_acc'].append(train_acc)
                data['val_loss'].append(val_loss)
                data['val_acc'].append(val_acc)
    
    # 检查是否成功解析了数据
    if not data['epochs']:
        raise ValueError(f"无法从日志文件中提取数据: {log_file}")
    
    return data


def plot_loss(data: Dict[str, Any], output_file: str = "log/training/loss_curve.png") -> None:
    """绘制 Loss 曲线图。
    
    绘制训练损失和验证损失随 epoch 变化的曲线。
    
    Args:
        data (Dict[str, Any]): 包含训练数据的字典（由 parse_log() 返回）
        output_file (str): 输出图表文件路径。默认为 "log/training/loss_curve.png"
    
    Returns:
        None: 直接保存图表到文件
    
    Note:
        - 图表包含两条曲线：训练损失（蓝色）和验证损失（红色）
        - 使用了平滑处理以便更清晰地显示趋势
        - 包含网格、图例和标签等装饰元素
    """
    # 创建图表和坐标轴
    plt.figure(figsize=(13, 6))
    
    # 绘制训练损失曲线
    plt.plot(data['epochs'], data['train_loss'], 
             label='Train Loss', color='#2E86AB', linewidth=2.5, marker='o', markersize=4)
    
    # 绘制验证损失曲线
    plt.plot(data['epochs'], data['val_loss'], 
             label='Val Loss', color='#A23B72', linewidth=2.5, marker='s', markersize=4)
    
    # 设置图表标题和轴标签
    plt.title('Training Loss vs Validation Loss', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=11, fontweight='bold')
    plt.ylabel('Loss', fontsize=11, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(fontsize=10, loc='upper right', framealpha=0.95)
    
    # 设置 x 轴刻度
    plt.xticks(range(0, len(data['epochs']) + 1, max(1, len(data['epochs']) // 10)))
    
    # 调整布局
    plt.tight_layout(pad=1.0)
    
    # 保存图表
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Loss 曲线已保存: {output_file}")


def plot_accuracy(data: Dict[str, Any], output_file: str = "log/training/accuracy_curve.png") -> None:
    """绘制 Accuracy 曲线图。
    
    绘制训练准确率和验证准确率随 epoch 变化的曲线。
    
    Args:
        data (Dict[str, Any]): 包含训练数据的字典（由 parse_log() 返回）
        output_file (str): 输出图表文件路径。默认为 "log/training/accuracy_curve.png"
    
    Returns:
        None: 直接保存图表到文件
    
    Note:
        - 图表包含两条曲线：训练准确率（绿色）和验证准确率（橙色）
        - Y 轴范围设为 [0, 1]，便于理解
        - 包含网格、图例和标签等装饰元素
    """
    # 创建图表和坐标轴
    plt.figure(figsize=(13, 6))
    
    # 绘制训练准确率曲线
    plt.plot(data['epochs'], data['train_acc'], 
             label='Train Accuracy', color='#06A77D', linewidth=2.5, marker='o', markersize=4)
    
    # 绘制验证准确率曲线
    plt.plot(data['epochs'], data['val_acc'], 
             label='Val Accuracy', color='#F18F01', linewidth=2.5, marker='s', markersize=4)
    
    # 设置图表标题和轴标签
    plt.title('Training Accuracy vs Validation Accuracy', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=11, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=11, fontweight='bold')
    
    # 设置 Y 轴范围
    plt.ylim(0, 1)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(fontsize=10, loc='lower right', framealpha=0.95)
    
    # 设置 x 轴刻度
    plt.xticks(range(0, len(data['epochs']) + 1, max(1, len(data['epochs']) // 10)))
    
    # 调整布局
    plt.tight_layout(pad=1.0)
    
    # 保存图表
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Accuracy 曲线已保存: {output_file}")


def plot_combined(data: Dict[str, Any], output_file: str = "log/training/combined_curve.png") -> None:
    """绘制合并双轴图表。
    
    在同一张图表中绘制 Loss 和 Accuracy，使用双 Y 轴分别显示。
    这样可以在一张图中同时观察两个指标的变化趋势。
    
    Args:
        data (Dict[str, Any]): 包含训练数据的字典（由 parse_log() 返回）
        output_file (str): 输出图表文件路径。默认为 "log/training/combined_curve.png"
    
    Returns:
        None: 直接保存图表到文件
    
    Note:
        - 左 Y 轴显示 Loss，右 Y 轴显示 Accuracy
        - 使用不同的颜色区分四条曲线
        - 图表包含两个图例，分别对应两个 Y 轴
    """
    # 创建图表和主坐标轴（用于 Loss）
    fig, ax1 = plt.subplots(figsize=(15, 7))
    
    # ============ 左 Y 轴：Loss ============
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold', color='#2E86AB')
    
    # 绘制训练损失
    line1 = ax1.plot(data['epochs'], data['train_loss'], 
                     label='Train Loss', color='#2E86AB', linewidth=2.5, marker='o', markersize=4)
    
    # 绘制验证损失
    line2 = ax1.plot(data['epochs'], data['val_loss'], 
                     label='Val Loss', color='#A23B72', linewidth=2.5, marker='s', markersize=4)
    
    # 设置左 Y 轴刻度颜色
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # ============ 右 Y 轴：Accuracy ============
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold', color='#06A77D')
    
    # 绘制训练准确率
    line3 = ax2.plot(data['epochs'], data['train_acc'], 
                     label='Train Accuracy', color='#06A77D', linewidth=2.5, marker='^', markersize=4)
    
    # 绘制验证准确率
    line4 = ax2.plot(data['epochs'], data['val_acc'], 
                     label='Val Accuracy', color='#F18F01', linewidth=2.5, marker='D', markersize=4)
    
    # 设置右 Y 轴范围和刻度颜色
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelcolor='#06A77D')
    
    # ============ 图表标题和图例 ============
    plt.title('Training Progress: Loss and Accuracy', fontsize=13, fontweight='bold', pad=15)
    
    # 合并两个 Y 轴的图例
    # 获取所有线条
    lines = line1 + line2 + line3 + line4
    labels: List[str] = [str(l.get_label()) for l in lines]  # type: ignore[arg-type]
    
    # 在图表上添加合并后的图例（放在右上角，避免遮挡曲线）
    ax1.legend(lines, labels, fontsize=10, loc='upper right', framealpha=0.95)
    
    # 设置 x 轴刻度
    ax1.set_xticks(range(0, len(data['epochs']) + 1, max(1, len(data['epochs']) // 10)))
    
    # 调整布局
    fig.tight_layout(pad=1.0)
    
    # 保存图表
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 合并双轴图已保存: {output_file}")


def main():
    """主函数：绘制所有训练过程可视化图表。
    
    该函数执行以下步骤：
        1. 解析训练日志文件
        2. 绘制 Loss 曲线
        3. 绘制 Accuracy 曲线
        4. 绘制合并双轴图表
        5. 输出统计信息
    
    Returns:
        None: 直接保存图表到 log/training/ 目录
    
    Note:
        - 所有图表都以 PNG 格式保存
        - 日志文件路径：log/training/train_log.txt
        - 输出图表路径：
            - log/training/loss_curve.png
            - log/training/accuracy_curve.png
            - log/training/combined_curve.png
    """
    print("\n" + "="*60)
    print("📊 训练过程可视化")
    print("="*60)
    
    # ============ 解析命令行参数 ============
    parser = argparse.ArgumentParser(description="训练过程可视化")
    parser.add_argument('--exp-id', type=str, default='01',
                       help='实验编号，对应训练时使用的 exp_id (默认: 01)')
    args = parser.parse_args()
    
    # 提取 exp_id 变量
    exp_id = args.exp_id
    
    try:
        # ============ 确保输出目录存在 ============
        os.makedirs(f"log/{exp_id}/training", exist_ok=True)
        
        # ============ 第一步：解析日志文件 ============
        print("\n📖 正在读取训练日志...")
        data = parse_log(f"log/{exp_id}/training/train_log.txt")
        
        # 输出数据统计
        print(f"\n✅ 日志解析成功")
        print(f"   总 Epoch 数: {len(data['epochs'])}")
        print(f"   最终训练损失: {data['train_loss'][-1]:.6f}")
        print(f"   最终验证损失: {data['val_loss'][-1]:.6f}")
        print(f"   最终训练准确率: {data['train_acc'][-1]:.4f}")
        print(f"   最终验证准确率: {data['val_acc'][-1]:.4f}")
        
        # ============ 第二步：绘制 Loss 曲线 ============
        print("\n📈 正在绘制 Loss 曲线...")
        plot_loss(data, output_file=f"log/{exp_id}/training/loss_curve.png")
        
        # ============ 第三步：绘制 Accuracy 曲线 ============
        print("📊 正在绘制 Accuracy 曲线...")
        plot_accuracy(data, output_file=f"log/{exp_id}/training/accuracy_curve.png")
        
        # ============ 第四步：绘制合并双轴图 ============
        print("📊 正在绘制合并双轴图...")
        plot_combined(data, output_file=f"log/{exp_id}/training/combined_curve.png")
        
        print("\n" + "="*60)
        print("✅ 所有图表已成功生成！")
        print("="*60)
        print("\n📁 生成的文件：")
        print(f"   • log/{exp_id}/training/loss_curve.png - Loss 曲线")
        print(f"   • log/{exp_id}/training/accuracy_curve.png - Accuracy 曲线")
        print(f"   • log/{exp_id}/training/combined_curve.png - Loss + Accuracy 合并图")
        print("\n")
    
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {str(e)}")
        print("   请确保已经运行过训练脚本，生成了 log/training/train_log.txt 文件")
        return
    
    except ValueError as e:
        print(f"\n❌ 错误: {str(e)}")
        return
    
    except Exception as e:
        print(f"\n❌ 未知错误: {str(e)}")
        return


# ============ 程序入口点 ============
if __name__ == "__main__":
    """主程序入口。
    
    当该脚本直接运行时执行 main() 函数。
    这是 Python 的标准做法，确保模块在被导入为模块时不会自动执行。
    
    要求：
        - 已运行过 train.py，生成了 log/training/train_log.txt
        - 已安装 matplotlib 库
    """
    main()
