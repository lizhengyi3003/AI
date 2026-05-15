"""ResNeXt模型推理预测模块。

该模块实现了对单张或批量图片的分类预测功能。包括：
    1. 模型加载和初始化
    2. 单张图片推理预测
    3. 批量图片处理
    4. 结果格式化输出
    5. 类别映射管理

推理流程：
    - 加载训练好的最佳模型权重
    - 输入图片自动进行预处理（缩放、裁剪、标准化）
    - 模型前向传播得到预测概率
    - 返回预测类别、置信度和Top5预测结果

模块功能：
    - predict_batch(): 批量预测一个目录下的所有图片
    - predict_single(): 预测单张图片
    - interactive_predict(): 交互式预测单张图片

使用方法：
    $ python predict.py
    # 进入交互模式，输入图片路径进行预测
    
    或者通过代码调用：
    from predict import predict_single
    result = predict_single("test_image.jpg")
    print(f"预测: {result['pred_class']}, 置信度: {result['confidence']:.2%}")

输出示例：
    预测类别: cat
    置信度: 95.67%
    
    Top5 预测:
      1. cat: 95.67%
      2. tiger: 3.21%
      3. leopard: 0.89%
      4. lion: 0.18%
      5. dog: 0.05%

注意事项：
    - 模型需提前训练并保存到 model-out/best.pth
    - 图片格式支持：JPG、PNG、BMP等PIL支持的格式
    - 图片会自动转换为RGB（兼容灰度图）
    - 推理时自动使用GPU加速（如可用）
"""

import torch
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from model import ResNeXt
from utils.utils import (
    load_model_weights,
    get_class_names,
    predict_single_image,
    print_prediction_result,
)
from environment.device_utils import parse_device_arg, setup_device


def predict_single(image_path: str, model: Optional[torch.nn.Module] = None,
                  classes: Optional[List[str]] = None) -> Dict[str, Any]:
    """对单张图片进行预测。
    
    该函数是对 utils.predict_single_image() 的便捷封装，自动处理模型加载
    和类别获取，使用户可以一行代码完成预测。
    
    Args:
        image_path (str): 输入图片路径（支持绝对路径和相对路径）。
            例如：
                - "test.jpg"
                - "./data/test/cat.png"
                - "C:\\Users\\test.jpg"
        model (torch.nn.Module, optional): 预加载的模型对象。
            如果为None，会自动创建并加载模型。
            用途：批量预测时传入相同的模型避免重复加载，提高效率。
        classes (List[str], optional): 类别名列表。
            如果为None，会自动从data/train目录读取。
            用途：批量预测时避免重复读取类别信息。
    
    Returns:
        Dict[str, Any]: 包含以下键值对的预测结果字典：
            - 'pred_class' (str): 预测的类别名
            - 'pred_idx' (int): 预测类别的索引（0到num_classes-1）
            - 'confidence' (float): 预测置信度（0到1之间）
            - 'top5_classes' (List[str]): Top5类别名列表
            - 'top5_scores' (List[float]): Top5置信度分数列表
            - 'image_path' (str): 输入图片的路径
    
    Raises:
        FileNotFoundError: 如果图片文件不存在
        FileNotFoundError: 如果模型权重文件不存在
        RuntimeError: 如果模型前向传播失败
    
    Example:
        >>> # 简单使用：自动加载模型
        >>> result = predict_single("test.jpg")
        >>> print(f"预测: {result['pred_class']}")
        >>> print(f"置信度: {result['confidence']:.2%}")
        
        >>> # 批量预测：复用同一个模型（推荐）
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> model = ResNeXt(num_classes=101).to(device)
        >>> model = load_model_weights(model, "model-out/best.pth", device)
        >>> model.eval()
        >>> classes = get_class_names("data")
        >>> 
        >>> for image_path in ["img1.jpg", "img2.jpg", "img3.jpg"]:
        ...     result = predict_single(image_path, model, classes)
        ...     print(result['pred_class'])
    
    Note:
        - 第一次调用时会自动加载模型和类别信息，耗时较长（1-2秒）
        - 推荐在循环中使用时传入model和classes参数，避免重复加载
        - 模型默认使用GPU（如可用），自动加速推理
        - 返回的置信度是softmax后的概率值
    """
    # ============ 初始化设备和模型 ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -------- 模型加载 --------
    # 如果未提供模型，则创建新模型并加载权重
    # 这样做支持两种使用方式：
    #   1. 单次预测：自动处理，无需手动加载
    #   2. 批量预测：预先加载，复用同一模型（高效）
    if model is None:
        # 创建模型实例
        # 这里假设是101个类别（ImageNet标准），实际会通过get_class_names获取
        model = ResNeXt(num_classes=101).to(device)
        
        # 加载训练好的模型权重
        # "model-out/best.pth"是训练阶段保存的最佳模型
        # map_location=device确保权重加载到正确的设备
        model = load_model_weights(model, "model-out/best.pth", device)
        
        # 设置为评估模式
        # 这是推理的必要操作：禁用Dropout和BatchNorm更新
        model.eval()
    
    # -------- 类别加载 --------
    # 如果未提供类别列表，则从数据目录读取
    if classes is None:
        # 从训练集目录自动获取类别名
        # get_class_names会扫描data/train目录的子文件夹
        classes = get_class_names("data")
    
    # ============ 执行预测 ============
    # 调用工具函数进行实际预测
    # 该函数处理：图片加载 → 预处理 → 前向传播 → 结果处理
    result = predict_single_image(image_path, model, device, classes)
    
    return result


def predict_batch(image_dir: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """批量预测目录下的所有图片。
    
    对指定目录中的所有图片进行预测，支持递归扫描子目录。
    使用同一个模型实例处理所有图片，提高效率。
    
    Args:
        image_dir (str): 包含图片的目录路径。
            支持相对路径和绝对路径。
            会递归查找该目录及其子目录中的所有图片。
        output_file (str, optional): 结果保存文件路径。
            如果指定，会将预测结果保存为文本文件。
            格式：图片名称, 预测类别, 置信度
            默认为None（仅打印，不保存）。
    
    Returns:
        List[Dict[str, Any]]: 包含所有预测结果的列表，每个元素是一个
        predict_single()返回的字典。
        
        支持的图片格式：JPG、PNG、BMP、TIFF等PIL支持的格式。
        会自动跳过无法打开的文件。
    
    Raises:
        FileNotFoundError: 如果指定的目录不存在
        ValueError: 如果目录中没有找到任何图片
    
    Example:
        >>> # 批量预测一个目录下的所有图片
        >>> results = predict_batch("test_images/")
        >>> print(f"处理了 {len(results)} 张图片")
        >>> for result in results:
        ...     print(f"{result['image_path']}: {result['pred_class']}")
        
        >>> # 批量预测并保存结果到文件
        >>> results = predict_batch("test_images/", output_file="results.txt")
        >>> # results.txt 格式:
        >>> # test_images/cat.jpg, cat, 95.67%
        >>> # test_images/dog.png, dog, 88.34%
    
    Note:
        - 会自动跳过无法识别的文件类型
        - 支持的格式：JPG、PNG、BMP、TIFF、WEBP等
        - 批量预测会复用同一模型，比单独调用predict_single快得多
        - 打印进度条显示处理进度
    """
    from tqdm import tqdm
    
    # ============ 第一步：验证目录 ============
    # 检查输入目录是否存在
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"目录不存在: {image_dir}")
    
    # ============ 第二步：扫描图片文件 ============
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    
    # 递归扫描目录中的所有图片
    # Path.rglob()方法递归查找所有匹配模式的文件
    image_files = []
    for ext in image_extensions:
        # 不区分大小写查找文件
        image_files.extend(Path(image_dir).rglob(f'*{ext}'))
        image_files.extend(Path(image_dir).rglob(f'*{ext.upper()}'))
    
    # 去重和排序（Path对象）
    image_files = sorted(set(str(p) for p in image_files))
    
    if not image_files:
        raise ValueError(f"目录中未找到任何图片: {image_dir}")
    
    print(f"\n📁 找到 {len(image_files)} 张图片")
    print(f"🔍 开始批量预测...\n")
    
    # ============ 第三步：预加载模型和类别（优化） ============
    # 预先加载一次，避免在循环中重复加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型并加载权重
    model = ResNeXt(num_classes=101).to(device)
    model = load_model_weights(model, "model-out/best.pth", device)
    model.eval()
    
    # 加载类别信息
    classes = get_class_names("data")
    
    # ============ 第四步：批量预测循环 ============
    results = []
    failed_images = []  # 记录预测失败的图片
    
    # 使用进度条显示处理进度
    with torch.no_grad():  # 禁用梯度计算以节省显存
        for image_path in tqdm(image_files, desc="预测进度"):
            try:
                # 调用预测函数
                # 传入预加载的模型和类别以提高效率
                result = predict_single_image(image_path, model, device, classes)
                results.append(result)
                
            except Exception as e:
                # 捕获预测失败的情况并记录
                # 继续处理其他图片，不中断整个流程
                print(f"⚠️  预测失败: {image_path}")
                print(f"   错误信息: {str(e)}")
                failed_images.append((image_path, str(e)))
    
    # ============ 第五步：结果保存（可选） ============
    if output_file is not None:
        # 将预测结果保存到文本文件
        # 便于后续分析和记录
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # 写入表头
                f.write("图片路径,预测类别,置信度,Top5预测\n")
                
                # 逐行写入结果
                for result in results:
                    # 构建Top5信息字符串
                    top5_info = "; ".join(
                        f"{cls}({score:.2%})"
                        for cls, score in zip(result['top5_classes'], result['top5_scores'])
                    )
                    
                    # 写入一行结果
                    line = (
                        f"{result['image_path']},"
                        f"{result['pred_class']},"
                        f"{result['confidence']:.2%},"
                        f"{top5_info}\n"
                    )
                    f.write(line)
            
            print(f"\n✅ 结果已保存到: {output_file}")
        
        except Exception as e:
            print(f"\n⚠️  保存结果文件失败: {str(e)}")
    
    # ============ 第六步：总结输出 ============
    print("\n" + "="*60)
    print("📊 批量预测完成")
    print("="*60)
    print(f"✅ 成功预测: {len(results)} 张图片")
    
    if failed_images:
        print(f"❌ 预测失败: {len(failed_images)} 张图片")
        for path, error in failed_images[:3]:  # 只显示前3个失败的
            print(f"   - {path}: {error}")
        if len(failed_images) > 3:
            print(f"   ... 还有 {len(failed_images)-3} 张失败")
    
    print("="*60 + "\n")
    
    return results


def interactive_predict(device: Optional[torch.device] = None) -> None:
    """交互式预测模式。
    
    进入交互式命令行界面，用户可以逐个输入图片路径进行预测。
    支持以下命令：
        - 输入图片路径：进行预测（例如：test.jpg 或 ./data/test.jpg）
        - 'batch': 切换到批量预测模式（输入目录路径）
        - 'quit': 退出程序
        - 'help': 显示帮助信息
    
    Args:
        device (Optional[torch.device]): 计算设备对象。
            如为None则自动检测（优先GPU）。默认值: None
    
    Returns:
        None: 交互式程序，直接打印结果到控制台
    
    Example:
        >>> interactive_predict(torch.device('cuda'))
        🎯 进入交互预测模式
        输入图片路径 (输入 'help' 获取帮助, 'quit' 退出):
        > test.jpg
        预测: cat, 置信度: 95.67%
        > dog.png
        预测: dog, 置信度: 88.34%
        > quit
        👋 再见!
    
    Note:
        - 支持相对路径和绝对路径
        - 支持拖拽文件到终端获取路径
        - 模型只加载一次，所有预测复用同一实例（高效）
    """
    # ============ 初始化 ============
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 使用设备: {device}\n")
    
    # 预加载模型和类别（避免每次预测都重新加载）
    print("⏳ 正在加载模型...")
    model = ResNeXt(num_classes=101).to(device)
    model = load_model_weights(model, "model-out/best.pth", device)
    model.eval()
    
    print("⏳ 正在加载类别...")
    classes = get_class_names("data")
    
    print("\n" + "="*60)
    print("🎯 进入交互预测模式")
    print("="*60)
    print("命令说明:")
    print("  - 输入图片路径进行单张预测 (例如: test.jpg 或 ./data/test.jpg)")
    print("  - 输入 'batch 目录路径' 批量预测 (例如: batch ./test_images/)")
    print("  - 输入 'help' 显示此帮助信息")
    print("  - 输入 'quit' 退出程序")
    print("="*60 + "\n")
    
    # ============ 交互循环 ============
    while True:
        try:
            # 获取用户输入
            user_input = input("输入命令 (图片路径/batch/help/quit): ").strip()
            
            # 处理特殊命令
            if user_input.lower() == 'quit':
                print("👋 再见!")
                break
            
            elif user_input.lower() == 'help':
                # 显示帮助信息
                print("\n" + "-"*60)
                print("📖 使用说明")
                print("-"*60)
                print("1. 单张预测: 输入图片路径")
                print("   示例: test.jpg")
                print("   示例: ./data/test/cat.png")
                print("\n2. 批量预测: 输入 'batch' + 空格 + 目录路径")
                print("   示例: batch ./test_images/")
                print("\n3. 退出: 输入 'quit'")
                print("-"*60 + "\n")
                continue
            
            elif user_input.lower().startswith('batch'):
                # 处理批量预测命令
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("⚠️  请输入目录路径，格式: batch <目录路径>\n")
                    continue
                
                batch_dir = parts[1]
                try:
                    # 调用批量预测函数
                    results = predict_batch(batch_dir)
                    
                    # 显示预测统计
                    if results:
                        print("\n前3个预测结果:")
                        for i, result in enumerate(results[:3], 1):
                            print(f"{i}. {result['pred_class']} "
                                  f"(置信度: {result['confidence']:.2%})")
                
                except (FileNotFoundError, ValueError) as e:
                    print(f"❌ 错误: {str(e)}\n")
                
                continue
            
            # 处理单张图片预测
            if not user_input:
                print("⚠️  请输入有效的路径\n")
                continue
            
            try:
                # 调用预测函数
                result = predict_single(user_input, model, classes)
                
                # 格式化显示结果
                print_prediction_result(result)
            
            except FileNotFoundError:
                print(f"❌ 文件不存在: {user_input}\n")
            
            except Exception as e:
                print(f"❌ 预测失败: {str(e)}\n")
        
        except KeyboardInterrupt:
            # 处理Ctrl+C中断
            print("\n\n👋 程序已退出")
            break


# ============ 程序入口点 ============
if __name__ == "__main__":
    """主程序入口。
    
    当该脚本直接运行时（而非被导入为模块），进入交互式预测模式。
    这是Python的标准做法，确保模块在被其他脚本导入时不会自动执行。
    
    使用方式：
        $ python predict.py                          # 自动检测设备
        $ python predict.py --device gpu            # 强制GPU
        $ python predict.py --device cpu            # 强制CPU
        # 进入交互式预测界面
    
    程序流程：
        1. 解析命令行参数选择设备
        2. 自动检测或强制使用GPU/CPU设备
        3. 加载训练好的模型权重
        4. 加载数据集类别信息
        5. 进入交互式命令行
        6. 支持单张预测、批量预测、帮助和退出
    
    交互模式命令：
        - 输入图片路径：单张预测（相对路径或绝对路径）
        - batch 目录: 批量预测指定目录下的所有图片
        - help: 显示帮助信息
        - quit: 退出程序
    """
    # ============ 解析命令行参数 ============
    parser = argparse.ArgumentParser(description="ResNeXt模型推理预测")
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'gpu', 'cpu'],
                       help='选择计算设备: auto=自动检测, gpu=强制GPU, cpu=强制CPU')
    args = parser.parse_args()
    
    # ============ 初始化设备 ============
    device_name = parse_device_arg(args)
    device = setup_device(device_name)
    
    interactive_predict(device)
