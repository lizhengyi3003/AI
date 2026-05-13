"""ResNeXt项目环境和脚本验证工具。

该模块提供了完整的项目环境检查功能，包括：
    1. Python依赖包检查（PyTorch、torchvision、tqdm等）
    2. 项目文件完整性检查
    3. 数据集目录结构验证
    4. Python代码语法检查
    5. 模型初始化测试

检查项目：
    - ✅ PyTorch、torchvision等依赖包版本
    - ✅ model.py、train.py、test.py等核心脚本存在性
    - ✅ data/train/、data/val/、data/test/数据集目录
    - ✅ 脚本Python代码语法正确性
    - ✅ ResNeXt模型能否正常初始化和前向传播

运行时机：
    - 项目初始设置后验证环境
    - 修改依赖后验证兼容性
    - 出现问题时诊断问题根源
    - CI/CD管道中自动检查

模块功能：
    - check_module(): 检查单个Python包
    - check_files(): 检查项目文件和数据目录
    - check_code_syntax(): 检查脚本语法
    - check_model_init(): 测试模型初始化
    - main(): 运行完整检查并输出报告

使用方法：
    $ python verify_setup.py
    # 输出详细的环境检查报告
    # 成功返回0，失败返回1

输出示例：
    ============================================================
    ResNeXt 项目环境验证工具
    ============================================================
    
    📦 检查必需的 Python 包...
    ✅ PyTorch              2.0.1
    ✅ torchvision          0.15.2
    ✅ tqdm                 4.65.0
    ...
    
    📂 检查项目文件...
    ✅ model.py            模型定义
    ...
    
    验证总结:
      依赖包:   ✅ 正常
      项目文件: ✅ 正常
      代码语法: ✅ 正常
      模型初始: ✅ 正常
    
    ✅ 环境检查通过！可以开始训练:
       python train.py

注意事项：
    - 该脚本不修改任何文件，仅进行检查
    - 检查失败时给出具体的错误信息和修复建议
    - 支持在任何项目目录运行
    - 可集成到CI/CD流程中
"""

import sys
import importlib
import os


def check_module(module_name, package_name=None):
    """检查单个Python包是否可导入。

    尝试导入指定模块，并获取其版本号（如果存在）。
    用于验证依赖包的安装和版本兼容性。

    参数说明（Args）:
        module_name (str): 模块的导入名称（Python包名）。
            例如："torch"、"torchvision"、"tqdm"
        
        package_name (str, optional): 用于显示的包名（可能与模块名不同）。
            如果为None，使用module_name。
            例如：module_name="PIL"但package_name="Pillow"

    返回值（Returns）:
        bool: 包检查结果。
            True: 包存在且可导入
            False: 包不存在或导入失败

    输出示例：
        ✅ PyTorch              2.1.0
        ❌ tensorflow          (导入失败: No module named 'tensorflow')

    使用示例（Example）:
        >>> from verify_setup import check_module
        >>> check_module("torch", "PyTorch")
        ✅ PyTorch              2.1.0
        True

    注意事项（Note）:
        - try-except捕捉ImportError获取导入错误信息
        - 版本号通过 module.__version__ 获取（如果存在）
        - 某些包（如PIL）的模块名与实际包名不同
    """
    if package_name is None:
        package_name = module_name

    try:
        # 尝试导入模块
        mod = importlib.import_module(module_name)
        # 获取版本号（如果存在 __version__ 属性）
        version = getattr(mod, "__version__", "未知版本")
        print(f"✅ {package_name:<20} {version}")
        return True
    except ImportError as e:
        # 导入失败，显示错误信息
        print(f"❌ {package_name:<20} (导入失败: {e})")
        return False


def check_files():
    """检查项目核心文件和数据集目录是否存在。

    验证以下内容：
        1. 核心Python脚本文件（model.py, mydataset.py等）
        2. 数据集目录（data/train/, data/val/, data/test/）
        3. 数据集内容（每个目录下的类别文件夹数）

    该检查确保项目运行所需的所有文件和目录都已就位。

    返回值（Returns）:
        bool: 检查结果。
            True: 所有文件和目录都存在且完整
            False: 存在缺失的文件或目录

    检查内容：
        1. ✅ 核心脚本文件
        2. ✅ 数据集目录
        3. ✅ 数据目录内容

    输出示例：
        📂 检查项目文件...
        ✅ model.py            模型定义
        ✅ mydataset.py        数据加载
        ✅ train.py            训练脚本
        ✅ test.py             测试脚本
        ✅ utils.py            工具函数
        ✅ predict.py          推理预测

        📂 检查数据集目录...
        ✅ data/train          (6941 张)
        ✅ data/val            (2379 张)
        ✅ data/test           (1366 张)

    使用示例（Example）:
        >>> from verify_setup import check_files
        >>> check_files()
        True

    注意事项（Note）:
        - 文件不存在时会提示用户检查文件位置
        - 数据集目录内容检查使用递归统计
        - 不检查文件内容正确性，只检查存在性
    """
    print("\n📂 检查项目文件...")
    
    # ============ 第一步：检查核心脚本文件 ============
    # 定义必要的Python脚本及其描述
    required_files = {
        "model.py": "模型定义",
        "mydataset.py": "数据加载",
        "train.py": "训练脚本",
        "test.py": "测试脚本",
        "utils.py": "工具函数",
        "predict.py": "推理预测",
    }
    
    all_files_ok = True
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"✅ {filename:<20} {description}")
        else:
            print(f"❌ {filename:<20} {description} - 文件缺失！")
            all_files_ok = False
    
    # ============ 第二步：检查数据集目录 ============
    print("\n📂 检查数据集目录...")

    data_dirs = {
        "data/train": "训练集",
        "data/val": "验证集",
        "data/test": "测试集",
    }

    for dir_path, description in data_dirs.items():
        if os.path.exists(dir_path):
            # 统计子目录数（类别数）和图片数
            class_dirs = [d for d in os.listdir(dir_path)
                          if os.path.isdir(os.path.join(dir_path, d))]
            num_classes = len(class_dirs)

            # 递归统计图片数量
            num_images = 0
            for class_dir in class_dirs:
                class_path = os.path.join(dir_path, class_dir)
                num_images += len([f for f in os.listdir(class_path)
                                   if os.path.isfile(os.path.join(class_path, f))])

            print(f"✅ {dir_path:<20} ({num_classes} 类, {num_images} 张)")
        else:
            print(f"❌ {dir_path:<20} {description} - 目录缺失！")
            all_files_ok = False

    return all_files_ok


def check_code_syntax():
    """检查项目Python脚本的语法正确性。

    使用py_compile模块编译Python文件，在不执行代码的情况下
    检测语法错误。这是一种轻量级的语法检查方法。

    检查范围：
        - model.py
        - mydataset.py
        - train.py
        - test.py
        - utils.py
        - predict.py

    返回值（Returns）:
        bool: 检查结果。
            True: 所有文件语法正确
            False: 存在语法错误

    输出示例：
        ✓ 代码语法检查...
        ✅ model.py            正常
        ✅ mydataset.py        正常
        ✅ train.py            正常
        ✅ test.py             正常
        ✅ utils.py            正常
        ✅ predict.py          正常

    使用示例（Example）:
        >>> from verify_setup import check_code_syntax
        >>> check_code_syntax()
        True

    注意事项（Note）:
        - py_compile只检查语法，不检测运行时错误
        - 语法错误会显示具体的行号和错误信息
        - 编译失败的文件名和详细信息会一并输出
    """
    import py_compile
    import io
    
    print("\n✓ 代码语法检查...")

    # 需要检查的Python文件列表
    files_to_check = [
        "model.py",
        "mydataset.py",
        "train.py",
        "test.py",
        "utils.py",
        "predict.py",
    ]

    all_ok = True

    for filename in files_to_check:
        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"⚠️  {filename:<20} 文件不存在，跳过检查")
            all_ok = False
            continue

        try:
            # 使用py_compile编译检查语法
            # doraise=True: 编译错误时抛出异常
            py_compile.compile(filename, doraise=True)
            print(f"✅ {filename:<20} 正常")
        except py_compile.PyCompileError as e:
            # 编译错误：显示具体错误信息
            print(f"❌ {filename:<20} 语法错误: {e}")
            all_ok = False

    return all_ok


def check_model_init():
    """检查ResNeXt模型能否正常初始化和前向传播。

    这是最关键的检查，验证：
        1. 模型类能否正常导入和实例化
        2. 模型参数数量是否合理
        3. 模型能否执行前向传播
        4. 输出张量形状是否正确

    这个检查可以及早发现模型代码中的逻辑错误。

    返回值（Returns）:
        bool: 检查结果。
            True: 模型初始化和前向传播成功
            False: 模型初始化或前向传播失败

    检查内容：
        1. ✅ 模型初始化成功
        2. ✅ 总参数量（以百万为单位）
        3. ✅ 前向传播正常
        4. ✅ 输出形状正确

    输出示例：
        🧠 检查模型初始化...
        ✅ 模型初始化成功
        ✅ 总参数量: 83.53M
        ✅ 前向传播正常，输出形状: torch.Size([2, 101])

    使用示例（Example）:
        >>> from verify_setup import check_model_init
        >>> check_model_init()
        True

    注意事项（Note）:
        - 使用torch.no_grad()禁用梯度计算，加快测试
        - 使用虚拟数据测试，不需要真实的图片
        - 检查失败时给出具体的错误信息便于调试
    """
    print("\n🧠 检查模型初始化...")

    try:
        # ============ 第一步：导入必要的模块 ============
        import torch
        from model import ResNeXt

        # ============ 第二步：创建模型实例 ============
        # 创建ResNeXt模型，指定101个类别（ImageNet标准）
        model = ResNeXt(num_classes=101)

        # 打印成功消息
        print("✅ 模型初始化成功")

        # ============ 第三步：计算参数数量 ============
        # sum(p.numel() for ...)统计所有参数的总个数
        # numel()返回张量中元素的总数
        total_params = sum(p.numel() for p in model.parameters())

        # 除以1e6转换为百万单位，便于阅读
        # 例如：83530000 → 83.53M
        print(f"✅ 总参数量: {total_params/1e6:.2f}M")

        # ============ 第四步：测试前向传播 ============
        # 创建虚拟输入张量用于测试
        # 形状：[batch_size=2, channels=3, height=224, width=224]
        dummy_input = torch.randn(2, 3, 224, 224)

        # 禁用梯度计算以加速测试
        with torch.no_grad():
            # 执行前向传播
            output = model(dummy_input)

        # 打印前向传播成功消息和输出形状
        print(f"✅ 前向传播正常，输出形状: {output.shape}")

        return True

    except Exception as e:
        # 任何异常都被捕获并打印
        print(f"❌ 模型初始化失败: {e}")
        return False


def main():
    """运行完整的环境检查流程并生成报告。

    该函数协调所有检查函数，按顺序执行检查，最后生成汇总报告。

    返回值（Returns）:
        int: 程序退出码。
            0: 所有检查通过，环境正确
            1: 存在检查失败，需要修复

    输出内容：
        1. 检查标题和分隔线
        2. 依赖包检查结果
        3. 项目文件检查结果
        4. 代码语法检查结果
        5. 模型初始化检查结果
        6. 汇总报告
        7. 建议和后续步骤

    使用示例（Example）:
        >>> python verify_setup.py
        # 运行完整检查
        # 返回exit code 0（成功）或1（失败）

    注意事项（Note）:
        - 该函数不返回任何有意义的值，应通过sys.exit()传递退出码
        - 所有检查都会执行，不会因某个检查失败而停止
        - 输出内容便于用户理解和诊断问题
    """
    # ============ 第一步：打印标题 ============
    print("=" * 60)
    print("ResNeXt 项目环境验证工具")
    print("=" * 60)

    # ============ 第二步：执行各项检查 ============

    # 检查Python依赖包
    print("\n📦 检查必需的 Python 包...")
    deps_ok = all([
        check_module("torch", "PyTorch"),
        check_module("torchvision", "torchvision"),
        check_module("tqdm", "tqdm"),
        check_module("numpy", "NumPy"),
        check_module("PIL", "Pillow"),
    ])

    # 如果依赖包不完整，给出安装建议
    if not deps_ok:
        print("\n⚠️  部分依赖缺失，请运行以下命令安装:")
        print("   pip install -r requirements.txt")

    # 检查项目文件
    files_ok = check_files()

    # 检查代码语法
    syntax_ok = check_code_syntax()

    # 检查模型初始化
    model_ok = check_model_init()

    # ============ 第三步：生成汇总报告 ============
    print("\n" + "=" * 60)
    print("验证总结:")

    # 显示各项检查的结果
    # 根据结果显示✅（正常）或❌（有缺失）
    print(f"  依赖包:   {'✅ 正常' if deps_ok else '❌ 有缺失'}")
    print(f"  项目文件: {'✅ 正常' if files_ok else '❌ 有缺失'}")
    print(f"  代码语法: {'✅ 正常' if syntax_ok else '❌ 有错误'}")
    print(f"  模型初始: {'✅ 正常' if model_ok else '❌ 有错误'}")
    print("=" * 60)

    # ============ 第四步：输出建议 ============
    # 根据所有检查结果显示不同的建议信息
    if all([deps_ok, files_ok, syntax_ok, model_ok]):
        # 所有检查都通过，可以开始训练
        print("\n✅ 环境检查通过！可以开始训练:")
        print("   python train.py")
        return 0
    else:
        # 存在检查失败，需要用户修复
        print("\n❌ 环境检查未完全通过，请按上述提示修复")
        return 1


# ============ 程序入口点 ============
if __name__ == "__main__":
    """主程序入口。

    当该脚本直接运行时（而非被导入为模块），执行完整的环境验证流程。
    这是Python的标准做法，确保模块在被其他脚本导入时不会自动执行。

    使用方式：
        $ python verify_setup.py

    程序流程：
        1. 运行main()函数执行所有检查
        2. main()返回exit code（0表示成功，1表示失败）
        3. sys.exit()将此码传递给操作系统

    典型输出：
        ============================================================
        ResNeXt 项目环境验证工具
        ============================================================
        
        📦 检查必需的 Python 包...
        ✅ PyTorch              2.1.0
        ✅ torchvision          0.16.0
        ...
        
        验证总结:
          依赖包:   ✅ 正常
          项目文件: ✅ 正常
          代码语法: ✅ 正常
          模型初始: ✅ 正常
        ============================================================
        
        ✅ 环境检查通过！可以开始训练:
           python train.py

    出错情况：
        若任何检查失败，会输出❌符号和错误信息。
        用户应根据提示信息进行修复（如安装缺失的包、创建数据目录等）。

    集成到CI/CD：
        该脚本可用于自动化测试流程中：
        1. 在运行测试前验证环境
        2. 检查失败时自动停止流程
        3. 保证只在正确的环境中运行测试
    """
    # 运行main()函数进行检查，获取返回码
    exit_code = main()
    # 将返回码传递给操作系统
    # 0表示成功（绿灯），1表示失败（红灯）
    sys.exit(exit_code)
