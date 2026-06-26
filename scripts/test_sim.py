#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
仿真环境测试脚本
验证MuJoCo和ALOHA仿真环境是否正常工作

使用方法:
    python scripts/test_sim.py

注意:
    - 无显示器环境请先设置: export MUJOCO_GL=egl
    - Windows环境可能需要WSL2
"""

import sys
import os
import time

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

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def print_header(msg):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{msg}{Colors.END}")


def test_mujoco_basic():
    """测试MuJoCo基本功能"""
    print_header("1. MuJoCo基础功能测试")
    
    try:
        import mujoco
        print_success(f"MuJoCo版本: {mujoco.__version__}")
        
        # 创建一个简单的模型
        xml = """
        <mujoco>
          <worldbody>
            <light pos="0 0 1"/>
            <body name="box" pos="0 0 0.5">
              <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
              <joint type="free"/>
            </body>
            <geom type="plane" size="1 1 0.1" rgba="0.8 0.9 0.8 1"/>
          </worldbody>
        </mujoco>
        """
        
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        
        print_success("MuJoCo模型创建成功")
        print_success(f"模型物体数: {model.nbody}")
        print_success(f"模型自由度: {model.nv}")
        
        # 运行几步仿真
        for i in range(10):
            mujoco.mj_step(model, data)
        
        print_success("仿真步进成功")
        print_info(f"10步仿真后时间: {data.time:.3f}s")
        print_info(f"盒子位置: {data.qpos[:3]}")
        
        return True
        
    except ImportError as e:
        print_error(f"MuJoCo导入失败: {e}")
        print_warning("请检查是否已安装 mujoco==2.3.7")
        return False
    except Exception as e:
        print_error(f"MuJoCo测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mujoco_rendering():
    """测试MuJoCo渲染功能"""
    print_header("2. MuJoCo渲染测试")
    
    try:
        import mujoco
        
        # 检查渲染后端
        gl_env = os.environ.get('MUJOCO_GL', 'auto')
        print_info(f"渲染后端: MUJOCO_GL={gl_env}")
        
        # 创建简单模型
        xml = """
        <mujoco>
          <worldbody>
            <light pos="0 0 2"/>
            <camera name="top" pos="0 0 2" zaxis="0 0 1"/>
            <body name="sphere" pos="0 0 0.5">
              <geom type="sphere" size="0.2" rgba="0 0.5 1 1"/>
            </body>
            <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
          </worldbody>
        </mujoco>
        """
        
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        
        # 尝试离屏渲染
        try:
            renderer = mujoco.Renderer(model, height=240, width=320)
            renderer.update_scene(data, camera="top")
            img = renderer.render()
            
            print_success("离屏渲染成功")
            print_info(f"渲染图像shape: {img.shape}")
            print_info(f"图像数据类型: {img.dtype}")
            
            # 保存测试图像
            try:
                import cv2
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                test_img_path = os.path.join(
                    os.path.dirname(__file__), '..', 'logs', 'test_render.png'
                )
                os.makedirs(os.path.dirname(test_img_path), exist_ok=True)
                cv2.imwrite(test_img_path, img_bgr)
                print_success(f"测试图像已保存: {test_img_path}")
            except ImportError:
                print_warning("OpenCV未安装，无法保存测试图像")
            
            renderer.close()
            return True
            
        except Exception as e:
            print_warning(f"离屏渲染失败: {e}")
            print_warning("如在无显示器环境，请设置: export MUJOCO_GL=egl")
            return False
        
    except Exception as e:
        print_error(f"渲染测试异常: {e}")
        return False


def test_dm_control():
    """测试dm_control（可选）"""
    print_header("3. dm_control测试（可选）")
    
    try:
        from dm_control import suite
        print_success("dm_control导入成功")
        
        # 列出可用环境
        domains = suite.ALL_TASKS
        print_info(f"可用任务数: {len(domains)}")
        
        # 测试一个简单环境
        env = suite.load('cartpole', 'swingup')
        print_success(f"环境创建成功: cartpole swingup")
        
        # 重置环境
        time_step = env.reset()
        print_success(f"环境重置成功")
        print_info(f"观测维度: {time_step.observation.shape}")
        
        # 运行几步
        action_spec = env.action_spec()
        for i in range(10):
            action = np.random.uniform(
                action_spec.minimum, 
                action_spec.maximum, 
                action_spec.shape
            )
            time_step = env.step(action)
        
        print_success("环境步进成功")
        
        return True
        
    except ImportError as e:
        print_warning(f"dm_control未安装: {e}")
        print_info("dm_control是可选依赖，用于更丰富的仿真环境")
        return None
    except Exception as e:
        print_error(f"dm_control测试失败: {e}")
        return False


def test_aloha_sim():
    """测试ALOHA仿真环境（如果已安装）"""
    print_header("4. ALOHA Sim测试")
    
    # 尝试查找ALOHA相关代码
    aloha_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'aloha'),
        os.path.join(os.path.dirname(__file__), '..', 'mobile_aloha'),
    ]
    
    aloha_found = False
    for path in aloha_paths:
        if os.path.exists(path):
            aloha_found = True
            print_success(f"ALOHA代码目录存在: {path}")
            break
    
    if not aloha_found:
        print_warning("ALOHA Sim代码目录未找到")
        print_info("这是正常的，ALOHA环境可后续单独配置")
        print_info("参考: https://github.com/tonyzhaozh/act")
        return None
    
    # 尝试导入
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        # 尝试导入ALOHA相关模块
        # 这里只是测试，实际导入路径可能不同
        print_info("ALOHA环境导入测试（简化版）")
        print_warning("完整ALOHA环境配置请参考官方文档")
        
        return None
        
    except Exception as e:
        print_warning(f"ALOHA导入测试跳过: {e}")
        return None


def main():
    print("\n" + "=" * 50)
    print(f"{Colors.BOLD}    仿真环境测试{Colors.END}")
    print("=" * 50)
    
    results = {}
    
    # 1. MuJoCo基础测试
    results['mujoco_basic'] = test_mujoco_basic()
    
    # 2. 渲染测试
    results['mujoco_render'] = test_mujoco_rendering()
    
    # 3. dm_control测试（可选）
    results['dm_control'] = test_dm_control()
    
    # 4. ALOHA测试
    results['aloha_sim'] = test_aloha_sim()
    
    # ===== 总结 =====
    print("\n" + "=" * 50)
    print(f"{Colors.BOLD}    测试总结{Colors.END}")
    print("=" * 50)
    
    print(f"\n{'测试项':<20} {'结果':<10}")
    print("-" * 30)
    
    for name, result in results.items():
        if result is True:
            status = f"{Colors.GREEN}通过{Colors.END}"
        elif result is False:
            status = f"{Colors.RED}失败{Colors.END}"
        else:
            status = f"{Colors.YELLOW}跳过{Colors.END}"
        print(f"{name:<20} {status}")
    
    print()
    
    # 核心功能判断
    if results['mujoco_basic']:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ MuJoCo核心功能正常{Colors.END}")
        print("  可以进行物理仿真和算法开发")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ MuJoCo核心功能异常{Colors.END}")
        print("  请检查安装，参考README中的常见问题")
    
    if results['mujoco_render']:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ 渲染功能正常{Colors.END}")
        print("  可以进行可视化和视频录制")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  渲染功能不可用{Colors.END}")
        print("  训练仍可进行，但无法可视化")
        print("  无显示器环境请设置: export MUJOCO_GL=egl")
    
    print("\n" + "=" * 50 + "\n")
    
    # 返回核心测试是否通过
    return 0 if results['mujoco_basic'] else 1


if __name__ == '__main__':
    import numpy as np  # 提前导入，避免测试中报错
    sys.exit(main())
