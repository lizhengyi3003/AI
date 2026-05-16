"""PyTorch设备初始化和管理工具模块。

该模块提供了方便的设备管理函数，简化模型训练和推理中的设备处理。
包括：
    1. 命令行参数解析（--device 参数）
    2. 设备初始化（CPU/GPU选择）
    3. 设备信息输出（显卡型号、内存等）
"""

import torch
import argparse


def parse_device_arg(args: argparse.Namespace) -> str:
    """从argparse参数解析--device值。
    
    从命令行解析的 args 对象中提取 device 参数，
    转换为标准的设备标识符。
    
    功能说明：
        - 从 args.device 获取用户输入
        - 标准化为小写字符串
        - 验证参数值的有效性
        - 返回标准的设备名称
    
    Args:
        args (argparse.Namespace): argparse 解析的命令行参数对象
            应包含 device 属性，其值为 'auto'|'gpu'|'cpu'
    
    Returns:
        str: 标准化的设备标识符
            - 'auto': 自动检测（优先GPU）
            - 'gpu': 强制使用GPU
            - 'cpu': 强制使用CPU
    
    Raises:
        AttributeError: args 对象不包含 device 属性
        ValueError: device 值不是预期的选项
    
    Note:
        - 函数主要用于命令行参数的标准化
        - 实际的PyTorch device对象由 setup_device() 创建
    """
    # 获取args中的device值，默认为'auto'
    device_arg = getattr(args, 'device', 'auto')
    
    # 标准化为小写
    device_name = device_arg.lower().strip()
    
    # 验证值的有效性
    valid_choices = ('auto', 'gpu', 'cuda', 'cpu')
    if device_name not in valid_choices:
        raise ValueError(
            f"无效的设备选择: {device_arg}。"
            f"必须是: {', '.join(valid_choices)}"
        )
    
    # 标准化GPU别名
    if device_name in ('gpu', 'cuda'):
        device_name = 'gpu'
    
    return device_name


def setup_device(device_choice: str = 'auto') -> torch.device:
    """初始化PyTorch设备对象并输出详细的设备信息。
    
    根据设备选择参数创建PyTorch设备对象，并打印详细的硬件信息。
    这是训练和推理脚本的必须步骤。
    
    功能说明：
        1. 解析设备选择参数
        2. 根据参数和系统情况确定最终使用的设备
        3. 打印详细的设备信息
        4. 返回PyTorch device对象
    
    支持的设备选择：
        - 'auto': 自动检测（优先GPU，如不可用则降级到CPU）
        - 'gpu': 强制使用GPU（如不可用会报错并提示）
        - 'cpu': 强制使用CPU
    
    Args:
        device_choice (str): 设备选择参数，默认为'auto'
            - 'auto': 自动检测最优设备
            - 'gpu': 强制GPU
            - 'cpu': 强制CPU
            大小写不敏感，会自动转换为小写
    
    Returns:
        torch.device: PyTorch设备对象
            可直接用于模型和张量的移动：
            - model = model.to(device)
            - tensor = tensor.to(device)
    
    Raises:
        RuntimeError: 用户强制使用GPU但系统无CUDA支持
        ValueError: device_choice 不是有效的选择
    
    Note:
        - 'auto' 模式下，如果GPU不可用会自动降级到CPU（不报错）
        - 'gpu' 模式下，如果GPU不可用会抛出异常
        - 建议在所有训练/推理脚本的开始调用此函数
        - 输出信息有助于调试和性能优化
    """
    # 标准化设备选择
    device_choice = device_choice.lower().strip()
    
    # 验证选择的有效性
    if device_choice not in ('auto', 'gpu', 'cuda', 'cpu'):
        raise ValueError(
            f"无效的设备选择: {device_choice}。"
            f"必须是: 'auto'、'gpu'、'cpu'"
        )
    
    # 标准化GPU别名
    if device_choice in ('gpu', 'cuda'):
        device_choice = 'gpu'
    
    print("\n" + "="*60)
    print("📱 设备初始化")
    print("="*60)
    
    # ============ 自动检测模式 ============
    if device_choice == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("\n✅ 自动检测: 使用 GPU 加速")
            _print_gpu_info(device)
        else:
            device = torch.device('cpu')
            print("\n📍 自动检测: 系统无GPU，降级到 CPU")
            _print_cpu_info(device)
    
    # ============ 强制GPU模式 ============
    elif device_choice == 'gpu':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "❌ 强制使用GPU但系统不支持CUDA！\n"
                "💡 解决方案:\n"
                "   1. 确认系统有NVIDIA显卡\n"
                "   2. 升级NVIDIA驱动\n"
                "   3. 重新安装GPU版PyTorch: python install_pytorch.py\n"
                "   4. 或运行: python train.py --device cpu (使用CPU替代)"
            )
        device = torch.device('cuda')
        print("\n✅ 强制选择: 使用 GPU 加速")
        _print_gpu_info(device)
    
    # ============ 强制CPU模式 ============
    else:  # device_choice == 'cpu'
        device = torch.device('cpu')
        print("\n✅ 强制选择: 使用 CPU 计算")
        _print_cpu_info(device)
    
    print()
    
    return device


def _print_gpu_info(device: torch.device):
    """打印详细的GPU信息。
    
    Args:
        device (torch.device): GPU设备对象
    """
    if not torch.cuda.is_available():
        return
    
    # GPU基本信息
    print(f"\n  📊 GPU 基本信息:")
    print(f"  设备类型: {device}")
    print(f"  设备数量: {torch.cuda.device_count()}")
    
    # 当前GPU信息
    gpu_name = torch.cuda.get_device_name(device)
    print(f"  🎮 显卡型号: {gpu_name}")
    
    # CUDA版本信息
    print(f"  🔧 CUDA 版本: {torch.version.cuda}")
    print(f"  📚 cuDNN 版本: {torch.backends.cudnn.version()}")
    
    # 显存信息
    gpu_memory_total = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
    gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
    gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
    
    print(f"  💾 显存: {gpu_memory_total:.0f} MB (总)")
    print(f"     已用: {gpu_memory_allocated:.0f} MB / 预留: {gpu_memory_reserved:.0f} MB")
    
    # 计算能力
    capability = torch.cuda.get_device_capability(device)
    print(f"  ⚡ 计算能力: {capability[0]}.{capability[1]}")


def _print_cpu_info(device: torch.device):
    """打印详细的CPU信息。
    
    Args:
        device (torch.device): CPU设备对象
    """
    import multiprocessing
    import platform
    
    print(f"\n  📊 CPU 基本信息:")
    print(f"  设备类型: {device}")
    
    # CPU型号
    processor = platform.processor()
    if processor:
        print(f"  🖥️  处理器: {processor}")
    
    # CPU核心数
    cpu_count = multiprocessing.cpu_count()
    print(f"  🔧 核心数: {cpu_count}")
    
    # Python版本
    print(f"  🐍 Python 版本: {platform.python_version()}")
    print(f"  📦 PyTorch 版本: {torch.__version__}")
    
    print(f"\n  ⚠️  CPU 计算速度较慢，建议使用 GPU 加速（如可用）")


if __name__ == "__main__":
    # ============ 测试设备初始化 ============
    print("🧪 设备工具模块测试\n")
    
    # 测试1：自动检测
    print("1️⃣  自动检测模式:")
    device_auto = setup_device('auto')
    print(f"   返回设备: {device_auto}\n")
    
    # 测试2：CPU模式
    print("2️⃣  CPU 模式:")
    device_cpu = setup_device('cpu')
    print(f"   返回设备: {device_cpu}\n")
    
    # 测试3：命令行参数解析
    print("3️⃣  命令行参数解析:")
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'gpu', 'cpu'])
    args = parser.parse_args(['--device', 'auto'])
    device_name = parse_device_arg(args)
    print(f"   解析结果: {device_name}\n")
