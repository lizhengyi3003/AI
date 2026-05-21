"""GPU/CPU设备检测与配置模块。

该模块提供了完整的PyTorch设备管理功能，包括：
    1. 系统CUDA版本检测（通过nvidia-smi）
    2. PyTorch已安装的CUDA版本检查
    3. PyTorch安装状态验证
    4. 系统CUDA与PyTorch CUDA版本匹配验证
    5. 设备选择（自动/GPU/CPU）
    6. 对应版本的PyTorch自动安装

核心功能：
    - detect_cuda_version(): 检测系统CUDA版本
    - get_pytorch_cuda_version(): 获取已安装PyTorch的CUDA版本
    - check_pytorch_installed(): 检查PyTorch是否安装及其配置
    - validate_cuda_match(): 验证CUDA版本匹配情况
    - get_device_choice(): 交互式设备选择或命令行指定
    - auto_install_pytorch(): 自动安装对应版本的PyTorch

注意事项：
    - 需要 pip、torch 等命令行工具
    - Windows 用户需要安装 nvidia-smi（NVIDIA 驱动包含）
    - Mac 用户：某些 CUDA 工具可能不可用（已处理）
    - Linux 用户：通常默认包含 nvidia-smi
"""

import subprocess
import sys
import os
import re
from typing import Optional, Dict, Tuple, Any


def detect_cuda_version() -> Optional[str]:
    """检测系统安装的CUDA版本。
    
    通过调用 nvidia-smi 命令行工具检测系统CUDA版本。
    该工具通常随NVIDIA显卡驱动一起安装。
    
    功能说明：
        - 调用 nvidia-smi 命令获取显卡和CUDA信息
        - 使用正则表达式解析CUDA版本号
        - 支持跨平台（Windows/Linux/Mac）
        - 如果nvidia-smi不存在或无显卡，返回None
    
    Returns:
        Optional[str]: CUDA版本号字符串，格式为 "XX.X" 或 "X.X"
            - 例如 "12.1"、"11.8"、"11.0"
            - 如果检测失败或无显卡返回 None
    
    Raises:
        subprocess.CalledProcessError: 命令执行异常（已捕获处理）
        FileNotFoundError: nvidia-smi 不在 PATH 中（已捕获处理）
    
    Note:
        - Windows：通常在 C:\\Program Files\\NVIDIA Corporation\\NVIDIA
          SMI\\nvidia-smi.exe
        - Linux：通常在 /usr/bin/nvidia-smi
        - Mac：如果使用 Metal Performance Shaders，nvidia-smi 不可用
        - 某些虚拟环境中可能不可用（使用 WSL 则可用）
    """
    try:
        # 调用 nvidia-smi 命令获取输出
        # nvidia-smi 是 NVIDIA 提供的系统查询工具
        # 输出包含驱动版本、CUDA版本、显卡信息等
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # 检查命令执行是否成功（返回码0表示成功）
        if result.returncode != 0:
            return None
        
        # 使用正则表达式从输出中提取CUDA版本
        # nvidia-smi 输出通常包含一行：
        # "| NVIDIA-SMI 555.55    Driver Version: 555.55       CUDA Version: 12.1     |"
        # 正则模式 r"CUDA Version:\s*(\d+\.\d+)" 匹配 CUDA Version: 后面的版本号
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
        
        if match:
            # 提取匹配的CUDA版本号（第一个捕获组）
            cuda_version = match.group(1)
            return cuda_version
        else:
            # 没有找到CUDA版本信息
            return None
            
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        # 捕捉可能的异常：
        # - TimeoutExpired: 命令执行超时（>5秒）
        # - FileNotFoundError: nvidia-smi 不在系统PATH中
        # - CalledProcessError: 命令返回非零返回码
        # 这些情况通常表示没有GPU或nvidia-smi不可用
        return None


def get_pytorch_cuda_version() -> Optional[str]:
    """获取已安装PyTorch的CUDA版本信息。
    
    检查已安装的torch库使用的CUDA版本。通过检查torch.version.cuda属性
    或torch内部标记来确定编译时的CUDA版本。
    
    功能说明：
        - 导入 torch 模块
        - 读取 torch.version.cuda 属性
        - 支持CPU版本torch（返回None）
        - 支持GPU版本torch（返回CUDA版本号）
    
    Returns:
        Optional[str]: PyTorch使用的CUDA版本号或None
            - GPU版本：返回版本号字符串，如 "12.1"、"11.8"
            - CPU版本：返回 None
            - 未安装PyTorch：返回 None
    
    Raises:
        ImportError: PyTorch 未安装（已捕获处理）
    
    Note:
        - 该函数只能检测已安装的PyTorch
        - 如果PyTorch是CPU版本（无CUDA），返回None
        - 版本号格式通常为 "X.X"（如 "12.1"）
        - PyTorch的CUDA版本应与系统驱动兼容
    """
    try:
        # 尝试导入torch库
        import torch
        
        # 检查torch.version.cuda属性
        # GPU版本的torch：torch.version.cuda 包含CUDA版本号（如"12.1"）
        # CPU版本的torch：torch.version.cuda 为 None
        pytorch_cuda = torch.version.cuda
        
        return pytorch_cuda
        
    except ImportError:
        # PyTorch 未安装或导入失败
        return None


def check_pytorch_installed() -> Dict[str, Any]:
    """检查PyTorch是否已安装及其配置信息。
    
    综合检查PyTorch的安装状态、版本号、CUDA版本等信息。
    这是快速诊断PyTorch环境的核心函数。
    
    检查内容：
        1. PyTorch是否能成功导入
        2. PyTorch版本号
        3. 是否使用GPU加速（有CUDA则支持GPU）
        4. CUDA版本号（如果可用）
        5. CUDA是否真正可用（通过torch.cuda.is_available()）
    
    返回结果：
        Dict[str, Any]: 包含以下键的字典
            - 'installed' (bool): PyTorch是否已安装
            - 'version' (str): PyTorch版本号（已安装时）
            - 'cuda_version' (str): CUDA版本号或None（已安装时）
            - 'cuda_available' (bool): CUDA是否可用（已安装时）
            - 'device' (str): 设备类型，'cuda' 或 'cpu'（已安装时）
            - 'error' (str): 错误信息（未安装时）
    
    Note:
        - 该函数不修改任何系统设置
        - 返回字典始终包含 'installed' 键
        - 已安装时包含详细的版本和设备信息
        - 未安装时仅包含错误说明
    """
    try:
        # 尝试导入torch库
        import torch
        
        # 获取PyTorch版本号
        # torch.__version__ 返回版本字符串，如 "2.0.1+cu118"
        pytorch_version = torch.__version__
        
        # 获取PyTorch编译时的CUDA版本
        pytorch_cuda = get_pytorch_cuda_version()
        
        # 检查CUDA是否真正可用（系统中是否有兼容的显卡）
        # torch.cuda.is_available() 返回True则GPU可用
        cuda_available = torch.cuda.is_available()
        
        # 确定设备类型
        device = "cuda" if cuda_available else "cpu"
        
        return {
            "installed": True,
            "version": pytorch_version,
            "cuda_version": pytorch_cuda,
            "cuda_available": cuda_available,
            "device": device
        }
        
    except ImportError as e:
        # PyTorch 未安装
        return {
            "installed": False,
            "error": f"PyTorch 未安装: {str(e)}"
        }


def validate_cuda_match() -> Dict[str, Any]:
    """验证系统CUDA版本与PyTorch CUDA版本是否匹配。
    
    比较系统安装的CUDA版本与PyTorch使用的CUDA版本，
    检查是否存在版本不匹配的问题。
    
    版本匹配规则：
        - 主版本号相同即认为兼容（如12.1与12.0兼容）
        - 例：系统CUDA 12.1，PyTorch CUDA 12.0 → 兼容
        - 例：系统CUDA 11.8，PyTorch CUDA 12.1 → 不兼容
    
    返回结果：
        Dict 包含以下键：
        - 'pytorch_installed' (bool): PyTorch是否已安装
        - 'cuda_detected' (bool): 系统是否检测到CUDA
        - 'system_cuda' (str): 系统CUDA版本或None
        - 'pytorch_cuda' (str): PyTorch CUDA版本或None
        - 'match' (bool): 版本是否匹配
        - 'warning' (str): 警告信息（不匹配时）
        - 'suggestion' (str): 修复建议（有问题时）
    
    Note:
        - PyTorch CPU版本不涉及CUDA匹配问题
        - 如果系统无GPU，此检查不适用
        - 版本不匹配可能导致性能下降或运行错误
    """
    # 检查PyTorch安装状态
    pytorch_info = check_pytorch_installed()
    
    # 检测系统CUDA版本
    system_cuda = detect_cuda_version()
    
    # 初始化返回字典
    result = {
        "pytorch_installed": pytorch_info["installed"],
        "cuda_detected": system_cuda is not None,
        "system_cuda": system_cuda,
        "pytorch_cuda": pytorch_info.get("cuda_version"),
        "match": True,
        "warning": "",
        "suggestion": ""
    }
    
    # 如果PyTorch未安装或是CPU版本，跳过匹配检查
    if not pytorch_info["installed"] or pytorch_info.get("cuda_version") is None:
        return result
    
    # 如果系统未检测到CUDA，但PyTorch是GPU版本
    if system_cuda is None and pytorch_info.get("cuda_version") is not None:
        result["match"] = False
        result["warning"] = "⚠️  系统未检测到CUDA，但PyTorch是GPU版本！"
        result["suggestion"] = "建议运行: python install_pytorch.py 重新安装PyTorch CPU版本"
        return result
    
    # 如果系统和PyTorch都有CUDA版本，检查主版本号是否匹配
    if system_cuda and pytorch_info.get("cuda_version"):
        system_major = system_cuda.split(".")[0]
        pytorch_major = pytorch_info["cuda_version"].split(".")[0]
        
        if system_major != pytorch_major:
            result["match"] = False
            result["warning"] = (
                f"⚠️  系统CUDA版本({system_cuda})与"
                f"PyTorch CUDA版本({pytorch_info['cuda_version']})不兼容！"
            )
            result["suggestion"] = "建议运行: python install_pytorch.py 重新安装对应版本的PyTorch"
    
    return result


def get_device_choice(force_device: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """获取用户的设备选择（GPU/CPU）。
    
    支持三种方式指定设备：
        1. 命令行参数：--device=gpu|cpu|auto
        2. 环境变量：PYTORCH_DEVICE=gpu|cpu|auto
        3. 交互式提示：用户选择菜单
    
    优先级（从高到低）：命令行参数 > 环境变量 > 交互式选择
    
    功能说明：
        - 检测系统CUDA版本
        - 提示用户选择设备
        - 验证选择的有效性
        - 返回选择结果和对应的CUDA版本
    
    Args:
        force_device (str, optional): 命令行强制指定的设备
            - 'gpu' 或 'cuda': 强制使用GPU
            - 'cpu': 强制使用CPU
            - 'auto': 自动检测（优先GPU）
            - None: 检查环境变量或交互式选择
            支持大小写，函数会自动转换为小写
    
    Returns:
        Tuple[str, Optional[str]]: 设备选择和对应的CUDA版本
            - 元组形式：(device_choice, cuda_version)
            - device_choice: 'cpu' 或 'cuda11.8' 或 'cuda12.1' 等
            - cuda_version: CUDA版本号或None（CPU模式时为None）
    
    Note:
        - 自动模式会优先选择GPU（如可用）
        - 如果无法使用GPU，会自动降级到CPU
        - 环境变量示例：export PYTORCH_DEVICE=gpu
    """
    # 第一优先级：命令行参数
    if force_device is not None:
        force_device = force_device.lower()
        if force_device in ('gpu', 'cuda', 'auto'):
            cuda_version = detect_cuda_version()
            if cuda_version:
                return (f"cuda{cuda_version}", cuda_version)
            else:
                if force_device == 'gpu':
                    print("⚠️  指定GPU但系统无CUDA，已降级到CPU")
                return ("cpu", None)
        elif force_device == 'cpu':
            return ("cpu", None)
    
    # 第二优先级：环境变量 PYTORCH_DEVICE
    env_device = os.environ.get("PYTORCH_DEVICE", "").lower()
    if env_device:
        if env_device in ('gpu', 'cuda', 'auto'):
            cuda_version = detect_cuda_version()
            if cuda_version:
                return (f"cuda{cuda_version}", cuda_version)
            else:
                if env_device == 'gpu':
                    print("⚠️  环境变量指定GPU但系统无CUDA，已降级到CPU")
                return ("cpu", None)
        elif env_device == 'cpu':
            return ("cpu", None)
    
    # 第三优先级：交互式选择
    print("\n" + "="*60)
    print("🔧 PyTorch 设备选择")
    print("="*60)
    
    # 检测系统CUDA版本
    cuda_version = detect_cuda_version()
    
    if cuda_version:
        print(f"\n✅ 检测到 NVIDIA CUDA: {cuda_version}")
        print(f"📱 可用设备:")
        print(f"  1. GPU (CUDA {cuda_version}) - 推荐，速度快")
        print(f"  2. CPU - 通用但较慢")
        print()
        choice = input("请选择 (1=GPU, 2=CPU，默认=GPU): ").strip().lower()
        
        if choice in ('2', 'cpu', 'c'):
            return ("cpu", None)
        else:
            return (f"cuda{cuda_version}", cuda_version)
    else:
        print("\n⚠️  未检测到 NVIDIA CUDA")
        print("📱 可用设备:")
        print(f"  1. CPU - 唯一选项")
        print()
        print("💡 如需使用GPU：")
        print("   1. 确认系统安装了NVIDIA显卡")
        print("   2. 安装或升级NVIDIA驱动")
        print("   3. 再次运行此脚本")
        input("按Enter继续...")
        return ("cpu", None)


def auto_install_pytorch(device_choice: str, cuda_version: Optional[str] = None) -> bool:
    """自动安装对应版本的PyTorch。
    
    根据指定的设备类型自动构建pip安装命令并执行安装。
    支持自动检测CUDA版本并安装对应的PyTorch版本。
    
    功能说明：
        1. 验证device_choice的有效性
        2. 确定要安装的CUDA版本（如果是GPU）
        3. 构建pip install命令
        4. 执行安装并显示进度
        5. 验证安装是否成功
    
    支持的device_choice值：
        - 'cpu': 安装CPU版本 PyTorch
        - 'cuda11.8': 安装 CUDA 11.8 版本 PyTorch
        - 'cuda12.1': 安装 CUDA 12.1 版本 PyTorch
        - 'auto': 自动检测CUDA并安装对应版本
    
    Args:
        device_choice (str): 设备选择
            - 'cpu': CPU版本
            - 'cuaXX.X': GPU版本，指定CUDA版本号
            - 'auto': 自动检测系统CUDA版本
        cuda_version (str, optional): CUDA版本号（device_choice为'auto'时使用）
            格式 "XX.X"，如 "12.1"、"11.8"
    
    Returns:
        bool: 安装是否成功
            True: PyTorch已安装且版本正确
            False: 安装失败或验证失败
    
    Raises:
        subprocess.CalledProcessError: pip命令执行失败（已捕获处理）
    
    Note:
        - 需要网络连接以下载安装包（通常1-5GB）
        - 首次安装可能需要5-15分钟
        - 安装过程中会输出详细的下载和编译进度
        - 如果pip命令不在PATH中，安装会失败
    """
    print("\n" + "="*60)
    print("📦 PyTorch 自动安装程序")
    print("="*60)
    
    # 验证device_choice
    device_choice = device_choice.lower().strip()
    
    # 处理 'auto' 选项
    if device_choice == 'auto':
        if cuda_version:
            device_choice = f"cuda{cuda_version}"
        else:
            system_cuda = detect_cuda_version()
            if system_cuda:
                device_choice = f"cuda{system_cuda}"
            else:
                device_choice = 'cpu'
    
    # 构建pip安装命令
    print(f"\n🔨 准备安装: {device_choice}")
    
    if device_choice == 'cpu':
        # CPU版本安装命令
        print("📍 安装 PyTorch CPU 版本...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--force-reinstall",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ]
    elif device_choice.startswith('cuda'):
        # GPU版本安装命令
        # 从device_choice中提取CUDA版本号（如"cuda12.1" → "cu121"）
        cuda_version_full = device_choice.replace('cuda', '')
        cuda_code = 'cu' + cuda_version_full.replace('.', '')
        
        print(f"📍 安装 PyTorch GPU 版本 (CUDA {cuda_version_full})...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--force-reinstall",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_code}"
        ]
    else:
        print(f"❌ 不支持的设备选项: {device_choice}")
        return False
    
    # 显示将执行的命令
    print(f"\n📋 执行命令: {' '.join(cmd)}")
    print("\n⏳ 正在安装（首次可能需要5-15分钟）...\n")
    
    try:
        # 执行pip install命令
        # stdout/stderr=None 将输出直接打印到控制台
        subprocess.run(cmd, check=True)
        
        print("\n✅ PyTorch 安装完成！")
        
        # 验证安装成功
        print("\n🧪 验证安装...")
        pytorch_info = check_pytorch_installed()
        
        if pytorch_info["installed"]:
            print(f"✅ PyTorch {pytorch_info['version']} 已成功安装")
            if pytorch_info.get("cuda_version"):
                print(f"✅ CUDA 版本: {pytorch_info['cuda_version']}")
                print(f"✅ GPU 支持: {'是' if pytorch_info['cuda_available'] else '否'}")
            else:
                print(f"✅ CPU 版本已安装")
            
            return True
        else:
            print(f"❌ 验证失败: {pytorch_info.get('error', '未知错误')}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 安装失败: {e}")
        print("\n💡 故障排除建议:")
        print("   1. 检查网络连接")
        print("   2. 尝试升级pip: python -m pip install --upgrade pip")
        print("   3. 检查磁盘空间（至少需要5GB）")
        print("   4. 尝试手动安装（访问 pytorch.org）")
        return False
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        return False


if __name__ == "__main__":
    # ============ 测试模块功能 ============
    # 运行此脚本时会执行以下测试：
    print("🧪 设备配置模块测试\n")
    
    # 测试1：检测系统CUDA版本
    print("1️⃣  检测系统 CUDA 版本...")
    cuda_version = detect_cuda_version()
    if cuda_version:
        print(f"   ✅ 系统 CUDA: {cuda_version}\n")
    else:
        print(f"   ⚠️  未检测到 CUDA\n")
    
    # 测试2：检查PyTorch安装状态
    print("2️⃣  检查 PyTorch 安装状态...")
    pytorch_info = check_pytorch_installed()
    if pytorch_info["installed"]:
        print(f"   ✅ PyTorch {pytorch_info['version']}")
        print(f"   ✅ CUDA: {pytorch_info.get('cuda_version', 'CPU 版本')}")
        print(f"   ✅ 设备: {pytorch_info['device']}\n")
    else:
        print(f"   ❌ {pytorch_info.get('error', '未知错误')}\n")
    
    # 测试3：验证CUDA匹配
    print("3️⃣  验证 CUDA 版本匹配...")
    match_result = validate_cuda_match()
    if match_result["pytorch_installed"]:
        if match_result["match"]:
            print(f"   ✅ CUDA 版本匹配\n")
        else:
            print(f"   ⚠️  {match_result['warning']}")
            print(f"   💡 {match_result['suggestion']}\n")
