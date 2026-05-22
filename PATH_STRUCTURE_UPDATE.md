# 路径结构更新完成 ✅

## 新路径结构

所有相关脚本已更新，现在支持按实验编号组织文件：

```
model-out/
├── 01/
│   ├── best.pth       # 实验01的最佳模型
│   └── last.pth       # 实验01的最后模型
├── 02/
│   ├── best.pth       # 实验02的最佳模型
│   └── last.pth       # 实验02的最后模型
└── ...

log/
├── 01/
│   ├── training/
│   │   ├── train_log.txt      # 实验01训练日志
│   │   ├── loss_curve.png
│   │   ├── accuracy_curve.png
│   │   └── combined_curve.png
│   └── evaluation/
│       ├── test_accuracy_summary.txt
│       ├── prediction_examples.png
│       ├── confusion_matrix.png
│       ├── per_class_accuracy.png
│       └── learning_rate_schedule.png
├── 02/
│   ├── training/
│   └── evaluation/
└── ...
```

## 使用方式

### 1. 训练模型

使用 `--exp-id` 参数指定实验编号（默认为 '01'）：

```bash
# 训练实验01（默认）
python train.py

# 训练实验02
python train.py --exp-id 02

# 训练实验03（指定GPU）
python train.py --exp-id 03 --device gpu

# 训练实验04（指定CPU）
python train.py --exp-id 04 --device cpu
```

**输出**：
- `model-out/{exp_id}/best.pth` - 最佳模型权重
- `model-out/{exp_id}/last.pth` - 最后一个epoch的模型权重
- `log/{exp_id}/training/train_log.txt` - 训练日志

### 2. 测试模型

```bash
# 测试实验01的模型
python test.py

# 测试实验02的模型
python test.py --exp-id 02

# 测试实验03的模型（指定GPU）
python test.py --exp-id 03 --device gpu
```

**输出**：测试准确率统计

### 3. 生成评估报告

```bash
# 为实验01生成完整评估报告
python utils/report.py

# 为实验02生成评估报告
python utils/report.py --exp-id 02

# 为实验03生成评估报告（指定GPU）
python utils/report.py --exp-id 03 --device gpu
```

**输出**：
- `log/{exp_id}/evaluation/test_accuracy_summary.txt` - 测试准确率汇总
- `log/{exp_id}/evaluation/prediction_examples.png` - 预测样例可视化
- `log/{exp_id}/evaluation/confusion_matrix.png` - 混淆矩阵
- `log/{exp_id}/evaluation/per_class_accuracy.png` - 每类准确率
- `log/{exp_id}/evaluation/learning_rate_schedule.png` - 学习率曲线

### 4. 绘制训练曲线

```bash
# 绘制实验01的训练曲线
python utils/draw.py

# 绘制实验02的训练曲线
python utils/draw.py --exp-id 02

# 绘制实验03的训练曲线
python utils/draw.py --exp-id 03
```

**输出**：
- `log/{exp_id}/training/loss_curve.png` - Loss 曲线
- `log/{exp_id}/training/accuracy_curve.png` - Accuracy 曲线
- `log/{exp_id}/training/combined_curve.png` - 合并双轴图

## 完整工作流示例

```bash
# 实验01完整工作流
python train.py --exp-id 01                    # 训练
python test.py --exp-id 01                     # 测试
python utils/draw.py --exp-id 01               # 绘制曲线
python utils/report.py --exp-id 01             # 生成报告

# 实验02完整工作流
python train.py --exp-id 02 --device gpu       # 使用GPU训练
python test.py --exp-id 02 --device gpu        # 使用GPU测试
python utils/draw.py --exp-id 02               # 绘制曲线
python utils/report.py --exp-id 02 --device gpu # 生成报告
```

## 修改的文件

✅ **train.py**
- 添加 `--exp-id` 参数（默认值 '01'）
- 修改模型保存路径：`model-out/{exp_id}/best.pth`, `model-out/{exp_id}/last.pth`
- 修改日志保存路径：`log/{exp_id}/training/train_log.txt`

✅ **test.py**
- 添加 `--exp-id` 参数（默认值 '01'）
- 修改模型加载路径：`model-out/{exp_id}/best.pth`

✅ **utils/report.py**
- 添加 `--exp-id` 参数（默认值 '01'）
- 修改模型加载路径：`model-out/{exp_id}/best.pth`
- 修改输出目录：`log/{exp_id}/evaluation/`

✅ **utils/draw.py**
- 添加 `argparse` 导入
- 添加 `--exp-id` 参数（默认值 '01'）
- 修改日志读取路径：`log/{exp_id}/training/train_log.txt`
- 修改输出目录：`log/{exp_id}/training/`

## 向后兼容性

所有脚本都设置了默认的实验编号为 '01'，这意味着：

- 如果不指定 `--exp-id` 参数，所有操作都会使用 '01' 作为实验编号
- 现有的工作流无需改动即可继续使用
- 可以逐步迁移到新的实验编号系统

## 命令行参数速查表

| 脚本 | 参数 | 说明 | 默认值 |
|-----|------|------|--------|
| train.py | --exp-id | 实验编号 | 01 |
| train.py | --device | 计算设备 | auto |
| test.py | --exp-id | 实验编号 | 01 |
| test.py | --device | 计算设备 | auto |
| utils/report.py | --exp-id | 实验编号 | 01 |
| utils/report.py | --device | 计算设备 | auto |
| utils/draw.py | --exp-id | 实验编号 | 01 |

---

**更新日期**: 2026-05-22  
**所有文件验证**: ✅ 通过
