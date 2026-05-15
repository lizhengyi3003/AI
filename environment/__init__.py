"""ResNeXt 项目环境配置和工具包。

包含 GPU/CPU 设备管理、环境验证等功能模块：
    - device_config.py: 设备和 CUDA 版本检测
    - device_utils.py: 设备初始化和设置
    - install_pytorch.py: PyTorch 一键安装脚本
    - verify_setup.py: 环境和项目完整性验证脚本

使用示例：
    from environment.device_utils import setup_device, parse_device_arg
    from environment.verify_setup import check_files, check_module
"""
