#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ACT环境完整性测试脚本
测试所有依赖是否正确安装，环境是否可用

使用方法:
    python scripts/test_env.py
"""

import sys
import os

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def print_header(msg):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{msg}{Colors.END}")

def main():
    all_passed = True
    
    print("\n" + "=" * 50)
    print(f"{Colors.BOLD}    ACT 环境测试报告{Colors.END}")
    print("=" * 50)
    
    # ===== 1. Python版本 =====
    print_header("1. Python环境")
    
    python_version = sys.version.split()[0]
    if python_version.startswith('3.8'):
        print_success(f"Python版本: {python_version}")
    else:
        print_warning(f"Python版本: {python_version} (推荐3.8.x)")
        all_passed = False
    
    # ===== 2. PyTorch =====
    print_header("2. PyTorch & CUDA")
    
    try:
        import torch
        print_success(f"PyTorch版本: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print_success(f"CUDA可用: True")
            print_success(f"CUDA版本: {torch.version.cuda}")
            
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print_success(f"GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print_error("CUDA可用: False (将使用CPU，训练会很慢)")
            all_passed = False
            
    except ImportError as e:
        print_error(f"PyTorch导入失败: {e}")
        all_passed = False
    
    # ===== 3. TorchVision =====
    print_header("3. 图像处理库")
    
    try:
        import torchvision
        print_success(f"TorchVision版本: {torchvision.__version__}")
    except ImportError as e:
        print_error(f"TorchVision导入失败: {e}")
        all_passed = False
    
    try:
        import cv2
        print_success(f"OpenCV版本: {cv2.__version__}")
    except ImportError as e:
        print_error(f"OpenCV导入失败: {e}")
        all_passed = False
    
    try:
        from PIL import Image
        print_success("Pillow导入成功")
    except ImportError as e:
        print_error(f"Pillow导入失败: {e}")
        all_passed = False
    
    # ===== 4. 数据处理库 =====
    print_header("4. 数据处理库")
    
    try:
        import numpy as np
        print_success(f"NumPy版本: {np.__version__}")
    except ImportError as e:
        print_error(f"NumPy导入失败: {e}")
        all_passed = False
    
    try:
        import scipy
        print_success(f"SciPy版本: {scipy.__version__}")
    except ImportError as e:
        print_error(f"SciPy导入失败: {e}")
        all_passed = False
    
    try:
        import h5py
        print_success(f"H5Py版本: {h5py.__version__}")
    except ImportError as e:
        print_error(f"H5Py导入失败: {e}")
        all_passed = False
    
    try:
        import pandas as pd
        print_success(f"Pandas版本: {pd.__version__}")
    except ImportError as e:
        print_warning(f"Pandas导入失败: {e} (非必需)")
    
    # ===== 5. 机器人仿真库 =====
    print_header("5. 机器人仿真库")
    
    try:
        import mujoco
        print_success(f"MuJoCo版本: {mujoco.__version__}")
    except ImportError as e:
        print_error(f"MuJoCo导入失败: {e}")
        print_warning("  如使用WSL2或服务器，可能需要设置 MUJOCO_GL=egl")
        all_passed = False
    
    try:
        import dm_control
        print_success("dm_control导入成功")
    except ImportError as e:
        print_warning(f"dm_control导入失败: {e} (可选)")
    
    try:
        import pyquaternion
        print_success("pyquaternion导入成功")
    except ImportError as e:
        print_warning(f"pyquaternion导入失败: {e} (可选)")
    
    # ===== 6. 可视化库 =====
    print_header("6. 可视化库")
    
    try:
        import matplotlib
        print_success(f"Matplotlib版本: {matplotlib.__version__}")
    except ImportError as e:
        print_warning(f"Matplotlib导入失败: {e} (训练非必需)")
    
    try:
        import tensorboard
        print_success("TensorBoard导入成功")
    except ImportError as e:
        print_warning(f"TensorBoard导入失败: {e} (可选)")
    
    # ===== 7. 工具库 =====
    print_header("7. 工具库")
    
    try:
        import yaml
        print_success("PyYAML导入成功")
    except ImportError as e:
        print_error(f"PyYAML导入失败: {e}")
        all_passed = False
    
    try:
        from tqdm import tqdm
        print_success("tqdm导入成功")
    except ImportError as e:
        print_warning(f"tqdm导入失败: {e} (可选)")
    
    try:
        import einops
        print_success("einops导入成功")
    except ImportError as e:
        print_warning(f"einops导入失败: {e} (可选)")
    
    # ===== 8. DETR模块 =====
    print_header("8. DETR模块 (ACT backbone)")
    
    try:
        # 尝试导入detr相关模块
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'detr'))
        
        # 检查目录是否存在
        detr_path = os.path.join(os.path.dirname(__file__), '..', 'detr')
        if os.path.exists(detr_path):
            print_success("DETR目录存在")
            
            # 检查是否有setup.py
            setup_path = os.path.join(detr_path, 'setup.py')
            if os.path.exists(setup_path):
                print_success("DETR setup.py存在 (需执行 pip install -e .)")
            else:
                print_warning("DETR setup.py未找到")
        else:
            print_warning("DETR目录未找到，请确认已克隆完整仓库")
            all_passed = False
            
    except Exception as e:
        print_warning(f"DETR模块检查异常: {e}")
    
    # ===== 总结 =====
    print("\n" + "=" * 50)
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 所有核心测试通过！环境配置完成{Colors.END}")
        print("\n下一步:")
        print("  1. 运行 python scripts/test_single_batch.py 验证单batch训练")
        print("  2. 运行 python scripts/test_sim.py 验证仿真环境")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  部分测试未通过，请检查上述错误信息{Colors.END}")
        print("\n常见问题解决方案请参考 README.md 中的「常见问题」章节")
    print("=" * 50 + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
