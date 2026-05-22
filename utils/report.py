"""训练结果评估和可视化报告生成模块。

该模块用于生成完整的模型评估报告，包括：
    1. 测试集准确率表格和柱状图
    2. 预测样例可视化（6-10张图片 + 真实标签 + 预测标签）
    3. 混淆矩阵热力图（所有类别的预测分布）
    4. 学习率变化曲线（从训练代码提取）
    5. 每类准确率柱状图（分类别的性能分析）
    
输出文件：
    - log/evaluation/test_accuracy_summary.txt - 测试准确率统计
    - log/evaluation/prediction_examples.png - 预测样例可视化（9宫格布局）
    - log/evaluation/confusion_matrix.png - 混淆矩阵热力图
    - log/evaluation/per_class_accuracy.png - 每类准确率柱状图
    - log/evaluation/learning_rate_schedule.png - 学习率变化曲线（如果可用）
"""

import sys
import os
import argparse
import random
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

# ============ 调整 sys.path 以支持直接运行脚本 ============
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _current_dir in sys.path:
    sys.path.remove(_current_dir)
if _project_root in sys.path:
    sys.path.remove(_project_root)
sys.path.insert(0, _project_root)

import torch
import numpy as np

import matplotlib  # type: ignore
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # type: ignore
from matplotlib import rcParams  # type: ignore

import seaborn as sns  # type: ignore

from model import ResNeXt
from mydataset import get_dataloaders
from environment.device_utils import parse_device_arg, setup_device
from utils.utils import get_class_names

# 设置中文字体和显示参数
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 100
rcParams['savefig.dpi'] = 150


def test_and_analyze(device: torch.device, classes: List[str], exp_id: str = "01") -> Tuple[float, Dict[str, Any]]:
    """在测试集上评估模型并收集详细的预测信息。
    
    Args:
        device: 计算设备
        classes: 类别名称列表
        exp_id: 实验编号，用于定位模型权重文件 (默认: "01")
        
    Returns:
        Tuple[float, Dict]: (准确率, 详细分析数据)
            详细数据包含：
                - all_preds: 所有预测标签
                - all_labels: 所有真实标签
                - all_confidences: 所有预测置信度
                - per_class_stats: 每类统计信息
    """
    print("\n" + "="*60)
    print("🧪 正在测试模型...")
    print("="*60)
    
    # 加载模型
    model_path = f"model-out/{exp_id}/best.pth"
    model = ResNeXt(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ 已加载模型权重: {model_path}")
    
    # 加载测试数据
    _, _, test_loader, _ = get_dataloaders("data")
    print(f"✓ 已加载测试集，批次数: {len(test_loader)}")
    
    # 初始化统计变量
    all_preds = []
    all_labels = []
    all_confidences = []
    per_class_stats: Dict[int, Dict[str, int]] = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    correct = 0
    total = 0
    
    # 测试循环
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            
            # 获取预测
            _, preds = torch.max(outputs, 1)
            
            # 获取置信度（softmax后的最大概率）
            confidences = torch.softmax(outputs, 1).max(1)[0]
            
            # 收集数据
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_confidences.extend(confidences.cpu().numpy().tolist())
            
            # 统计
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # 按类统计
            for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                per_class_stats[int(label)]['total'] += 1
                if int(pred) == int(label):
                    per_class_stats[int(label)]['correct'] += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    # 组织数据
    analysis_data = {
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_confidences': all_confidences,
        'per_class_stats': dict(per_class_stats),
    }
    
    print(f"✅ 测试完成")
    print(f"   总样本数: {total}")
    print(f"   正确预测: {correct}")
    print(f"   测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy, analysis_data


def save_test_accuracy_summary(accuracy: float, num_classes: int, num_test_samples: int,
                               per_class_stats: Optional[Dict[int, Dict[str, int]]] = None,
                               classes: Optional[List[str]] = None,
                               output_file: str = "log/evaluation/test_accuracy_summary.txt") -> None:
    """保存测试准确率汇总。
    
    Args:
        accuracy: 测试准确率
        num_classes: 类别总数
        num_test_samples: 测试样本总数
        per_class_stats: 每类统计信息
        classes: 类别名称列表
        output_file: 输出文件路径
    """
    print(f"\n📝 正在生成测试准确率汇总...")
    
    correct_count = int(accuracy * num_test_samples)
    error_count = num_test_samples - correct_count
    
    content = f"""{'='*60}
测试集评估报告
{'='*60}

【基本信息】
  测试样本总数: {num_test_samples}
  类别总数: {num_classes}
  数据集: 101 Caltech Objects

【测试结果】
  测试准确率: {accuracy:.4f}
  百分比: {accuracy*100:.2f}%
  正确预测: {correct_count}
  错误预测: {error_count}

【性能评价】
"""
    
    if accuracy >= 0.85:
        content += "  优秀 - 模型在测试集上表现出色"
    elif accuracy >= 0.75:
        content += "  良好 - 模型在测试集上有不错的表现"
    elif accuracy >= 0.65:
        content += "  中等 - 模型表现一般，可继续优化"
    else:
        content += "  需改进 - 模型在测试集上表现较差"
    
    content += f"\n\n生成时间: 2026-05-16\n"
    
    # ============ 每类准确率分析 ============
    if per_class_stats is not None and classes is not None:
        # 计算每类准确率
        class_acc_list = []
        for class_idx in range(num_classes):
            if class_idx in per_class_stats:
                stats = per_class_stats[class_idx]
                if stats['total'] > 0:
                    acc_val = stats['correct'] / stats['total']
                    class_acc_list.append((classes[class_idx], acc_val, stats['correct'], stats['total']))
        
        if class_acc_list:
            # 排序
            sorted_by_acc = sorted(class_acc_list, key=lambda x: x[1], reverse=True)
            
            # 统计高/中/低准确率类别数
            high = sum(1 for _, a, _, _ in class_acc_list if a >= 0.85)
            mid = sum(1 for _, a, _, _ in class_acc_list if 0.70 <= a < 0.85)
            low = sum(1 for _, a, _, _ in class_acc_list if a < 0.70)
            
            content += f"""
【每类准确率分布】
  高准确率 (≥85%):  {high} 类
  中准确率 (70-85%): {mid} 类
  低准确率 (<70%):   {low} 类

【表现最佳的 5 个类别】
"""
            for rank, (name, acc_val, corr, tot) in enumerate(sorted_by_acc[:5], 1):
                content += f"  {rank}. {name:<25s} {acc_val*100:5.1f}% ({corr}/{tot})\n"
            
            content += f"\n【表现最差的 5 个类别】\n"
            for rank, (name, acc_val, corr, tot) in enumerate(sorted_by_acc[-5:], 1):
                content += f"  {rank}. {name:<25s} {acc_val*100:5.1f}% ({corr}/{tot})\n"
    
    content += f"\n{'='*60}\n"
    
    # 保存文件
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 测试准确率汇总已保存: {output_file}")
    print(content)


def visualize_prediction_examples(classes: List[str], test_loader: torch.utils.data.DataLoader,  # type: ignore
                                 model: torch.nn.Module, device: torch.device,
                                 num_samples: int = 9, output_file: str = "log/evaluation/prediction_examples.png") -> None:
    """生成预测样例可视化（显示6-9张图片及其真实标签和预测标签）。
    
    Args:
        classes: 类别名称列表
        test_loader: 测试数据加载器
        model: 模型
        device: 计算设备
        num_samples: 显示的样本数（默认9）
        output_file: 输出文件路径
    """
    print(f"\n预测样例可视化正在生成...")
    
    # 收集所有测试样本
    all_imgs: List[torch.Tensor] = []
    all_labels: List[int] = []
    all_preds: List[int] = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs_cpu = imgs.cpu()
            for i in range(imgs.size(0)):
                # 反标准化图像
                img = imgs_cpu[i]  # [3, 224, 224]
                
                # 进行预测
                output = model(imgs[i:i+1].to(device))
                _, pred = torch.max(output, 1)
                
                all_imgs.append(img)
                all_labels.append(int(labels[i].item()))
                all_preds.append(int(pred.cpu().item()))
    
    # 随机抽取样本
    sample_count = min(num_samples, len(all_imgs))
    indices = random.sample(range(len(all_imgs)), sample_count)
    
    # 创建图表（更大的尺寸以容纳标题）
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for idx, sample_idx in enumerate(indices):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        # 反标准化并显示图像
        img = all_imgs[sample_idx].numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = np.clip(img * std + mean, 0, 1)
        
        ax.imshow(img)
        
        # 获取真实和预测标签
        true_label = classes[all_labels[sample_idx]]
        pred_label = classes[all_preds[sample_idx]]
        is_correct = all_labels[sample_idx] == all_preds[sample_idx]
        
        # 设置标题（增大字体提高可读性）
        color = 'green' if is_correct else 'red'
        status = '[OK]' if is_correct else '[X]'
        title = f"{status} {true_label} -> {pred_label}"
        ax.set_title(title, color=color, fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout(pad=0.5)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 预测样例已保存: {output_file}")


def generate_confusion_matrix(all_preds: List[int], all_labels: List[int], classes: List[str],
                             output_file: str = "log/evaluation/confusion_matrix.png", max_classes: int = 50) -> None:
    """生成混淆矩阵热力图。
    
    Args:
        all_preds: 所有预测标签
        all_labels: 所有真实标签
        classes: 类别名称列表
        output_file: 输出文件路径
        max_classes: 最多显示的类别数（太多会看不清）
    """
    print(f"\n混淆矩阵热力图生成中...")
    
    # 创建混淆矩阵
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes))
    
    for pred, label in zip(all_preds, all_labels):
        cm[label, pred] += 1
    
    # 归一化（按真实类别）
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_sum[cm_sum == 0] = 1  # 避免除以零
    cm_normalized = cm / cm_sum
    
    # 如果类别太多，只显示前 max_classes 个
    if num_classes > max_classes:
        cm_normalized = cm_normalized[:max_classes, :max_classes]
        print(f"Warning: {num_classes} classes found, showing only first {max_classes}")
    
    # 绘制热力图
    display_classes = min(num_classes, max_classes)
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',  # type: ignore
                xticklabels=False, yticklabels=False, ax=ax,
                cbar_kws={'label': 'Normalized Count'}, square=True)
    
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(f'Confusion Matrix (First {display_classes} Classes)', 
                 fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout(pad=1.0)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 混淆矩阵已保存: {output_file}")


def generate_per_class_accuracy(per_class_stats: Dict[int, Dict[str, int]], classes: List[str],
                               output_file: str = "log/evaluation/per_class_accuracy.png") -> None:
    """生成每类准确率柱状图。
    
    Args:
        per_class_stats: 每类统计信息
        classes: 类别名称列表
        output_file: 输出文件路径
    """
    print(f"\n每类准确率柱状图生成中...")
    
    # 计算每类准确率
    class_accs: List[float] = []
    class_names: List[str] = []
    
    for class_idx in range(len(classes)):
        if class_idx in per_class_stats:
            stats = per_class_stats[class_idx]
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                class_accs.append(acc)
                class_names.append(classes[class_idx])
    
    # 排序（准确率从高到低）
    sorted_pairs = sorted(zip(class_names, class_accs), key=lambda x: x[1], reverse=True)
    sorted_names, sorted_accs = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    # 绘制柱状图（高度和 DPI 根据类别数动态调整，确保文字清晰）
    total_classes = len(classes)
    fig_height = max(14, total_classes * 0.32)  # 每个类别占 0.32 英寸，101类≈32英寸
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    colors = ['#2ecc71' if acc >= 0.85 else '#f39c12' if acc >= 0.7 else '#e74c3c' 
              for acc in sorted_accs]
    
    bars = ax.barh(range(len(sorted_names)), sorted_accs, color=colors, height=0.85)
    
    ax.set_yticks(range(len(sorted_names)))
    # 字体根据类别数自动调整，小类别用大字体，大类别用适中字体
    label_fontsize = max(9, min(13, 900 / len(sorted_names)))  # 101类≈9px
    ax.set_yticklabels(sorted_names, fontsize=label_fontsize)
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class Accuracy ({total_classes} classes total)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 1)
    
    # 在柱子上显示数值（只为准确率 > 0.1 的类别标注）
    for i, (bar, acc) in enumerate(zip(bars, sorted_accs)):
        if acc > 0.1:
            ax.text(acc + 0.008, bar.get_y() + bar.get_height()/2,
                   f'{acc:.3f}', va='center', fontsize=7, fontweight='bold')
    
    # 添加参考线标注
    ax.axvline(x=0.85, color='#2ecc71', linestyle=':', alpha=0.6, linewidth=1.2)
    ax.axvline(x=0.70, color='#f39c12', linestyle=':', alpha=0.6, linewidth=1.2)
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout(pad=0.5)
    # 使用 300 DPI 确保 101 个标签和数值清晰可读
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    high_acc = sum(1 for acc in class_accs if acc >= 0.85)
    mid_acc = sum(1 for acc in class_accs if 0.7 <= acc < 0.85)
    low_acc = sum(1 for acc in class_accs if acc < 0.7)
    
    print(f"✅ 每类准确率柱状图已保存: {output_file}")
    print(f"   High accuracy (>= 85%): {high_acc}")
    print(f"   Mid accuracy (70-85%): {mid_acc}")
    print(f"   Low accuracy (< 70%): {low_acc}")


def generate_learning_rate_schedule(output_file: str = "log/evaluation/learning_rate_schedule.png") -> None:
    """生成学习率变化曲线（从train.py代码推导出的CosineAnnealingLR）。
    
    Args:
        output_file: 输出文件路径
    """
    print(f"\n学习率曲线生成中...")
    
    # CosineAnnealingLR 学习率公式
    initial_lr = 0.01
    min_lr = 0.0
    total_epochs = 80
    
    epochs = np.arange(0, total_epochs)
    lrs = min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * epochs / total_epochs)) / 2
    
    # 绘制学习率曲线
    fig, ax = plt.subplots(figsize=(13, 6))
    
    ax.plot(epochs, lrs, linewidth=2.5, color='#3498db', marker='o', 
            markersize=4, label='Learning Rate (CosineAnnealingLR)', markerfacecolor='#2980b9')
    
    ax.fill_between(epochs, 0, lrs, alpha=0.2, color='#3498db')  # type: ignore
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    ax.set_title('Learning Rate Schedule (CosineAnnealingLR)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper right')
    
    # 标注关键点
    ax.axvline(x=0, color='#27ae60', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.text(1, max(lrs)*0.9, 'Start', fontsize=9, color='#27ae60', fontweight='bold')
    
    ax.axvline(x=total_epochs-1, color='#e74c3c', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.text(total_epochs-8, max(lrs)*0.9, 'End', fontsize=9, color='#e74c3c', fontweight='bold')
    
    plt.tight_layout(pad=0.8)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 学习率曲线已保存: {output_file}")
    print(f"   Initial LR: {initial_lr}")
    print(f"   Final LR: {lrs[-1]:.6f}")
    print(f"   Total Epochs: {total_epochs}")


def main() -> None:
    """主程序入口。"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Generate model evaluation report")
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'gpu', 'cpu'],
                       help='Device selection: auto/gpu/cpu')
    parser.add_argument('--exp-id', type=str, default='01',
                       help='实验编号，对应训练时使用的 exp_id (默认: 01)')
    args = parser.parse_args()
    
    # 提取 exp_id 变量
    exp_id = args.exp_id
    
    # 初始化设备
    device_name = parse_device_arg(args)
    device = setup_device(device_name)
    
    # 获取类别和数据集信息
    classes = get_class_names("data")
    print(f"✓ Loaded {len(classes)} classes")
    
    # 加载测试数据集
    _, _, test_loader, _ = get_dataloaders("data")
    num_test_samples = sum(1 for _ in test_loader.dataset)  # type: ignore
    print(f"✓ Test samples: {num_test_samples}")
    
    # ============ 测试和分析 ============
    accuracy, analysis_data = test_and_analyze(device, classes, exp_id)
    
    # ============ 生成报告 ============
    print("\n" + "="*60)
    print("Generating evaluation report...")
    print("="*60)
    
    # 确保输出目录存在
    os.makedirs(f"log/{exp_id}/evaluation", exist_ok=True)
    
    # 1. 测试准确率汇总
    save_test_accuracy_summary(accuracy, len(classes), num_test_samples,
                               analysis_data['per_class_stats'], classes,
                               output_file=f"log/{exp_id}/evaluation/test_accuracy_summary.txt")
    
    # 2. 预测样例可视化
    model = ResNeXt(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(f"model-out/{exp_id}/best.pth", map_location=device))
    model.eval()
    visualize_prediction_examples(classes, test_loader, model, device,
                                 output_file=f"log/{exp_id}/evaluation/prediction_examples.png")
    
    # 3. 混淆矩阵
    generate_confusion_matrix(
        analysis_data['all_preds'],
        analysis_data['all_labels'],
        classes,
        output_file=f"log/{exp_id}/evaluation/confusion_matrix.png"
    )
    
    # 4. 每类准确率
    generate_per_class_accuracy(
        analysis_data['per_class_stats'],
        classes,
        output_file=f"log/{exp_id}/evaluation/per_class_accuracy.png"
    )
    
    # 5. 学习率变化曲线
    generate_learning_rate_schedule(
        output_file=f"log/{exp_id}/evaluation/learning_rate_schedule.png"
    )
    
    # ============ 总结 ============
    print("\n" + "="*60)
    print("✅ Report generation complete!")
    print("="*60)
    print("\nGenerated files:")
    print(f"   • log/{exp_id}/evaluation/test_accuracy_summary.txt")
    print(f"   • log/{exp_id}/evaluation/prediction_examples.png")
    print(f"   • log/{exp_id}/evaluation/confusion_matrix.png")
    print(f"   • log/{exp_id}/evaluation/per_class_accuracy.png")
    print(f"   • log/{exp_id}/evaluation/learning_rate_schedule.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
