"""
ResNeXt 项目环境和脚本验证工具
运行此脚本检查环境是否配置正确以及各脚本逻辑是否正常
"""

import sys
import importlib

def check_module(module_name, package_name=None):
    """检查模块是否可导入"""
    if package_name is None:
        package_name = module_name
    
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {package_name:<20} {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name:<20} (导入失败: {str(e)})")
        return False


def check_files():
    """检查必需的文件是否存在"""
    import os
    
    print("\n📂 检查项目文件...")
    files = [
        ("model.py", "模型定义"),
        ("mydataset.py", "数据加载"),
        ("train.py", "训练脚本"),
        ("test.py", "测试脚本"),
        ("utils.py", "工具函数"),
        ("requirements.txt", "依赖列表"),
        ("README.md", "说明文档"),
    ]
    
    all_ok = True
    for file, desc in files:
        if os.path.exists(file):
            print(f"✅ {file:<20} {desc}")
        else:
            print(f"❌ {file:<20} {desc} (文件不存在)")
            all_ok = False
    
    # 检查数据目录
    data_dirs = ["data/train", "data/val", "data/test"]
    print("\n📂 检查数据集目录...")
    for dir_path in data_dirs:
        if os.path.isdir(dir_path):
            num_classes = len([d for d in os.listdir(dir_path) 
                             if os.path.isdir(os.path.join(dir_path, d))])
            print(f"✅ {dir_path:<20} ({num_classes} 个类别)")
        else:
            print(f"⚠️  {dir_path:<20} (目录不存在)")
            all_ok = False
    
    return all_ok


def check_code_syntax():
    """检查主要脚本的语法"""
    import py_compile
    import os
    
    print("\n🔍 检查代码语法...")
    scripts = ["model.py", "mydataset.py", "train.py", "test.py", "utils.py"]
    
    all_ok = True
    for script in scripts:
        if os.path.exists(script):
            try:
                py_compile.compile(script, doraise=True)
                print(f"✅ {script:<20} 语法正确")
            except py_compile.PyCompileError as e:
                print(f"❌ {script:<20} 语法错误: {e}")
                all_ok = False
        else:
            print(f"⚠️  {script:<20} 文件不存在")
    
    return all_ok


def check_model_init():
    """检查模型能否正确初始化（不需要 torchvision）"""
    print("\n🧠 检查模型初始化...")
    try:
        import torch
        from model import ResNeXt
        
        model = ResNeXt(num_classes=101)
        print(f"✅ 模型初始化成功")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 总参数量: {total_params/1e6:.2f}M")
        
        # 测试前向传播
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ 前向传播正常，输出形状: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return False


def main():
    print("=" * 60)
    print("ResNeXt 项目环境验证工具")
    print("=" * 60)
    
    # 1. 检查依赖包
    print("\n📦 检查必需的 Python 包...")
    deps_ok = all([
        check_module("torch", "PyTorch"),
        check_module("torchvision", "torchvision"),
        check_module("tqdm", "tqdm"),
        check_module("numpy", "NumPy"),
        check_module("PIL", "Pillow"),
    ])
    
    if not deps_ok:
        print("\n⚠️  部分依赖缺失，请运行以下命令安装:")
        print("   pip install -r requirements.txt")
    
    # 2. 检查文件
    files_ok = check_files()
    
    # 3. 检查代码语法
    syntax_ok = check_code_syntax()
    
    # 4. 检查模型
    model_ok = check_model_init()
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结:")
    print(f"  依赖包:   {'✅ 正常' if deps_ok else '❌ 有缺失'}")
    print(f"  项目文件: {'✅ 正常' if files_ok else '❌ 有缺失'}")
    print(f"  代码语法: {'✅ 正常' if syntax_ok else '❌ 有错误'}")
    print(f"  模型初始: {'✅ 正常' if model_ok else '❌ 有错误'}")
    print("=" * 60)
    
    if all([deps_ok, files_ok, syntax_ok, model_ok]):
        print("\n✅ 环境检查通过！可以开始训练:")
        print("   python train.py")
        return 0
    else:
        print("\n❌ 环境检查未完全通过，请按上述提示修复")
        return 1


if __name__ == "__main__":
    sys.exit(main())
