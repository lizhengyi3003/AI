"""PyTorch 一键安装脚本。

该脚本提供了完整的PyTorch自动安装流程，包括：
    1. 系统环境检测（操作系统、CUDA版本）
    2. 安装状态检查
    3. 交互式设备选择
    4. 自动下载安装对应版本PyTorch
    5. 安装验证
"""

import sys
import os
import platform
import argparse

# ============ 调整 sys.path 以支持直接运行脚本 ============
# 问题：当直接运行 python environment/install_pytorch.py 时，Python 会将 environment/ 加入 sys.path[0]
# 解决：优先加入项目根目录，并移除/调整 environment/ 目录优先级
_current_dir = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录（environment/）
_project_root = os.path.dirname(_current_dir)  # 项目根目录

# 移除 sys.path 中的 environment/ 目录（Python 自动添加）
if _current_dir in sys.path:
    sys.path.remove(_current_dir)

# 确保项目根目录在最前面
if _project_root in sys.path:
    sys.path.remove(_project_root)
sys.path.insert(0, _project_root)

from environment.device_config import (
    detect_cuda_version,
    check_pytorch_installed,
    get_device_choice,
    auto_install_pytorch,
    validate_cuda_match
)


def print_system_info():
    """打印系统和PyTorch环境信息。
    
    输出系统操作系统、CUDA版本、PyTorch安装状态等信息，
    为用户快速了解当前环境。
    """
    print("\n" + "="*60)
    print("🖥️  系统信息")
    print("="*60)
    
    # 操作系统信息
    system = platform.system()
    version = platform.release()
    print(f"\n  操作系统: {system} {version}")
    print(f"  Python: {sys.version.split()[0]}")
    
    # CUDA信息
    cuda_version = detect_cuda_version()
    if cuda_version:
        print(f"  ✅ CUDA: {cuda_version} 检测到")
    else:
        print(f"  ⚠️  CUDA: 未检测到")
    
    # PyTorch 状态
    print("\n" + "-"*60)
    pytorch_info = check_pytorch_installed()
    if pytorch_info["installed"]:
        print(f"  PyTorch 版本: {pytorch_info['version']}")
        if pytorch_info.get("cuda_version"):
            print(f"  CUDA 支持: 是 ({pytorch_info['cuda_version']})")
            print(f"  GPU 可用: {'是' if pytorch_info['cuda_available'] else '否'}")
        else:
            print(f"  CUDA 支持: 否（CPU 版本）")
        print(f"  状态: ✅ 已安装")
    else:
        print(f"  PyTorch 版本: 未安装")
        print(f"  状态: ❌ 未安装")
    
    print()


def main():
    """主程序入口。
    
    完整的PyTorch安装流程：
        1. 解析命令行参数
        2. 打印系统信息
        3. 检查是否需要安装
        4. 获取设备选择
        5. 执行安装
        6. 验证安装结果
    """
    # ============ 解析命令行参数 ============
    parser = argparse.ArgumentParser(
        description="PyTorch 一键安装脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python install_pytorch.py                  # 交互式选择
  python install_pytorch.py --device gpu    # 强制GPU
  python install_pytorch.py --device cpu    # 强制CPU
  python install_pytorch.py --device auto   # 自动检测
        """
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["gpu", "cpu", "auto"],
        help="指定安装设备: gpu=GPU加速, cpu=通用版本, auto=自动检测"
    )
    
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="跳过已安装检查，强制重新安装"
    )
    
    args = parser.parse_args()
    
    # ============ 打印欢迎信息和系统信息 ============
    print("\n" + "="*60)
    print("🚀 PyTorch 安装助手 v1.0")
    print("="*60)
    
    print_system_info()
    
    # ============ 检查PyTorch安装状态 ============
    pytorch_info = check_pytorch_installed()
    
    if pytorch_info["installed"] and not args.skip_check:
        print("="*60)
        print("✅ PyTorch 已安装")
        print("="*60)
        print(f"\n当前版本: {pytorch_info['version']}")
        if pytorch_info.get("cuda_version"):
            print(f"CUDA 支持: {pytorch_info['cuda_version']}")
            print(f"GPU 可用: {'是' if pytorch_info['cuda_available'] else '否'}")
        else:
            print("运行模式: CPU")
        
        choice = input("\n是否重新安装? (y/N): ").lower().strip()
        if choice not in ('y', 'yes'):
            print("\n💡 后续步骤:")
            print("  1. 验证环境: python environment/verify_setup.py")
            print("  2. 开始训练: python train.py")
            print("  3. 测试模型: python test.py")
            return
    
    # ============ 验证CUDA匹配情况 ============
    match_result = validate_cuda_match()
    if match_result["pytorch_installed"] and not match_result["match"]:
        print("\n" + "="*60)
        print("⚠️  警告：CUDA 版本不匹配")
        print("="*60)
        print(f"  {match_result['warning']}")
        print(f"  {match_result['suggestion']}")
        choice = input("\n是否继续? (Y/n): ").lower().strip()
        if choice in ('n', 'no'):
            return
    
    # ============ 获取设备选择 ============
    device_choice, cuda_version = get_device_choice(args.device)
    
    # ============ 执行安装 ============
    print("\n" + "="*60)
    print("📥 开始安装 PyTorch...")
    print("="*60)
    print(f"\n📍 安装配置: {device_choice}")
    print("⏳ 请耐心等待（首次可能需要5-15分钟）...\n")
    
    success = auto_install_pytorch(device_choice, cuda_version)
    
    # ============ 安装结果处理 ============
    print("\n" + "="*60)
    if success:
        print("🎉 安装成功!")
        print("="*60)
        print("\n📚 后续步骤:")
        print("  1. 验证完整环境: python environment/verify_setup.py")
        print("  2. 开始训练模型: python train.py")
        print("  3. 测试模型性能: python test.py")
        print("  4. 图片推理预测: python utils/predict.py")
        print()
    else:
        print("❌ 安装失败")
        print("="*60)
        print("\n💡 故障排除:")
        print("  1. 检查网络连接")
        print("  2. 升级 pip: python -m pip install --upgrade pip")
        print("  3. 清理缓存: python -m pip cache purge")
        print("  4. 访问 https://pytorch.org 手动安装")
        print()


if __name__ == "__main__":
    """程序入口点。
    
    当此脚本直接运行时执行 main() 函数。
    """
    main()
