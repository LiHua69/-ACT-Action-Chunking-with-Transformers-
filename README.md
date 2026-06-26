# ACT: Action Chunking with Transformers

《具身智能基础》期末大作业 - ACT算法复现项目

> 基于 [官方ACT仓库](https://github.com/tonyzhaozh/act) 的复现实现，在ALOHA Sim仿真环境中验证算法效果

---

## 📋 目录

- [硬件要求](#-硬件要求)
- [快速安装](#-快速安装)
- [数据集准备](#-数据集准备)
- [运行测试](#-运行测试)
- [项目结构](#-项目结构)
- [常见问题](#-常见问题)
- [团队分工](#-团队分工)

---

## 💻 硬件要求

### 最低配置
| 硬件 | 规格 | 说明 |
|------|------|------|
| GPU | NVIDIA GTX 1660 / 6GB显存 | 可运行小batch size训练 |
| CPU | 4核8线程 | 数据预处理与仿真渲染 |
| 内存 | 16GB | 数据集加载 |
| 硬盘 | 50GB可用空间 | 数据集+模型权重 |
| 系统 | Ubuntu 20.04/22.04 或 Windows 10/11 | 推荐Linux环境 |

### 推荐配置
| 硬件 | 规格 | 说明 |
|------|------|------|
| GPU | NVIDIA RTX 3060/3070 / 8-12GB显存 | 流畅训练，合理训练速度 |
| CPU | 8核16线程以上 | 加速数据预处理 |
| 内存 | 32GB | 大数据集加载无压力 |
| 硬盘 | 100GB以上SSD | 快速数据读取 |
| 系统 | Ubuntu 20.04 | 官方推荐，兼容性最好 |

### 软件依赖
- **CUDA**: 11.8（已在环境中配置）
- **Python**: 3.8.10
- **Anaconda/Miniconda**: 最新版

---

## 🚀 快速安装

### 方式一：conda一键安装（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/LiHua69/-ACT-Action-Chunking-with-Transformers-.git
cd -ACT-Action-Chunking-with-Transformers-

# 2. 创建conda环境（自动安装所有依赖）
conda env create -f environment.yml

# 3. 激活环境
conda activate act

# 4. 安装DETR模块（ACT的backbone）
cd detr
pip install -e .
cd ..

# 5. 验证安装
python scripts/test_env.py
```

### 方式二：pip安装（备选）

```bash
# 1. 创建虚拟环境
conda create -n act python=3.8.10
conda activate act

# 2. 安装PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 安装DETR模块
cd detr
pip install -e .
cd ..

# 5. 验证安装
python scripts/test_env.py
```

### 安装验证标准
- ✅ `import torch` 无报错，`torch.cuda.is_available()` 返回 True
- ✅ `import mujoco` 无报错
- ✅ `import cv2` 无报错
- ✅ `python scripts/test_env.py` 全部测试通过
- ✅ 单batch空跑测试成功（见[运行测试](#-运行测试)）

---

## 📦 数据集准备

### ALOHA Sim 仿真数据集

#### 方式一：下载官方预采集数据集

```bash
# 创建数据目录
mkdir -p data

# 下载示例数据集（约2GB）
# 官方数据集地址（选择sim_transfer_cube任务）
wget https://example.com/aloha_sim_dataset.zip -P data/

# 解压
unzip data/aloha_sim_dataset.zip -d data/
```

#### 方式二：自行采集仿真数据（推荐练习）

```bash
# 使用脚本化策略采集演示数据
python record_sim_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --dataset_dir data \
  --num_episodes 50
```

### 数据集格式说明
```
data/
└── sim_transfer_cube_scripted/
    ├── episode_0.hdf5
    ├── episode_1.hdf5
    └── ...
```

每个HDF5文件包含：
- `/observations/images/top`: 顶部视角图像序列
- `/observations/qpos`: 关节位置序列
- `/action`: 动作序列
- 其他观测数据

---

## 🧪 运行测试

### 1. 环境完整性测试

```bash
# 运行完整环境测试
python scripts/test_env.py
```

预期输出：
```
========================================
    ACT 环境测试报告
========================================

✅ Python版本: 3.8.10
✅ PyTorch版本: 2.0.1
✅ CUDA可用: True
✅ CUDA版本: 11.8
✅ GPU设备: NVIDIA GeForce RTX 3060
✅ 显存大小: 12.0 GB
✅ MuJoCo导入成功
✅ OpenCV导入成功
✅ H5Py导入成功
✅ DETR模块导入成功

========================================
🎉 所有测试通过！环境配置完成
========================================
```

### 2. 单Batch空跑测试（验证模型+数据）

```bash
# 运行单batch训练测试
python scripts/test_single_batch.py
```

预期输出：
```
========================================
    单Batch空跑测试
========================================

📦 加载数据集...
✅ 数据集加载成功，样本数: 100
✅ Batch size: 8
✅ 图像shape: torch.Size([8, 3, 480, 640])
✅ 动作shape: torch.Size([8, 100, 14])

🤖 初始化模型...
✅ 模型参数数量: 35.2M
✅ 模型初始化成功

⚡ 前向传播测试...
✅ 前向传播成功
✅ 总损失: 1.2345
✅ KL散度: 0.0123
✅ 重构损失: 1.2222

🔄 反向传播测试...
✅ 反向传播成功
✅ 参数更新成功

========================================
🎉 单Batch空跑测试通过！
========================================
```

### 3. 仿真环境测试

```bash
# 测试仿真环境是否能正常启动
python scripts/test_sim.py
```

预期：弹出仿真窗口，显示ALOHA机器人模型。

---

## 📁 项目结构

```
-ACT-Action-Chunking-with-Transformers-/
├── 📄 README.md                    # 项目说明文档（本文件）
├── 📄 environment.yml              # Conda环境配置
├── 📄 requirements.txt             # Python依赖列表
├── 📄 setup.py                     # 安装配置
│
├── 📂 detr/                        # DETR模块（backbone）
│   └── ...
│
├── 📂 act/                         # ACT核心代码
│   ├── 📄 model.py                 # 模型定义（CVAE + Transformer）
│   ├── 📄 policy.py                # 策略封装
│   ├── 📄 training.py              # 训练逻辑
│   └── 📄 utils.py                 # 工具函数
│
├── 📂 data/                        # 数据集目录
│   └── ...
│
├── 📂 scripts/                     # 测试与工具脚本
│   ├── 📄 test_env.py              # 环境测试脚本
│   ├── 📄 test_single_batch.py     # 单batch空跑测试
│   └── 📄 test_sim.py              # 仿真环境测试
│
├── 📂 configs/                     # 配置文件
│   ├── 📄 sim_transfer_cube.yaml   # 示例任务配置
│   └── ...
│
├── 📂 checkpoints/                 # 模型权重保存目录
│   └── ...
│
└── 📂 logs/                        # 训练日志
    └── ...
```

---

## ❓ 常见问题 (FAQ)

### 1. CUDA版本不匹配 / PyTorch找不到GPU

**错误信息**：`RuntimeError: CUDA error: no kernel image is available for execution on the device`

**解决方案**：
```bash
# 检查CUDA版本
nvcc --version

# 检查PyTorch CUDA版本
python -c "import torch; print(torch.version.cuda)"

# 重新安装对应版本的PyTorch
pip uninstall torch torchvision
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

**注意**：确保系统安装的CUDA驱动版本 ≥ 11.8

---

### 2. MuJoCo安装失败 / 渲染错误

**错误信息**：`ImportError: DLL load failed while importing mujoco`

**解决方案**：

**Windows用户**：
- 推荐使用WSL2 (Ubuntu 20.04)
- 或安装Visual C++ Redistributable

**Linux用户**：
```bash
# 安装依赖
sudo apt-get install libgl1-mesa-glx libglfw3 libglew2.1

# 无头模式（服务器无显示器）
export MUJOCO_GL=egl
# 或
export MUJOCO_GL=osmesa
```

**验证安装**：
```bash
python -c "import mujoco; print('MuJoCo版本:', mujoco.__version__)"
```

---

### 3. 显存不足 (OOM)

**错误信息**：`RuntimeError: CUDA out of memory`

**解决方案**：

1. **减小batch size**：
   ```yaml
   # 在config文件中修改
   batch_size: 8  # 从32减小到8或4
   ```

2. **使用梯度累积**：
   ```python
   # 每4个batch更新一次参数，等效于batch_size * 4
   gradient_accumulation_steps = 4
   ```

3. **开启混合精度训练**：
   ```python
   scaler = torch.cuda.amp.GradScaler()
   ```

4. **申请学校GPU服务器**：联系老师申请A100/V100服务器使用权限

---

### 4. 训练损失不收敛 / 震荡

**可能原因**：
- 学习率过大
- 数据归一化错误
- 损失函数实现有问题

**解决方案**：

1. **降低学习率**：
   ```yaml
   lr: 1e-5  # 从1e-4降到1e-5试试
   ```

2. **检查数据归一化**：
   - 确保动作数据归一化到 [-1, 1]
   - 图像使用ImageNet标准化

3. **从小数据集开始验证**：
   ```bash
   # 先用少量数据测试能否过拟合
   python train.py --num_episodes 5 --epochs 100
   ```

---

### 5. 推理时动作抖动严重

**原因**：没有正确实现temporal ensemble时间集成

**解决方案**：
```python
# 确保启用时间集成
policy = ACTPolicy(
    temporal_ensemble=True,
    chunk_size=100,
    action_dim=14
)
```

**其他优化**：
- 增加推理时的采样次数
- 添加动作平滑滤波
- 检查动作反归一化是否正确

---

### 6. 仿真环境无法渲染 / 黑屏

**服务器无显示器环境**：
```bash
# 使用EGL后端
export MUJOCO_GL=egl

# 或使用OSMesa（软件渲染，较慢）
export MUJOCO_GL=osmesa
```

**本地环境**：
- 确保安装了正确的显卡驱动
- 尝试更新Mesa库
- 降低渲染分辨率

---

### 7. h5py版本不兼容

**错误信息**：`OSError: Unable to open file (file signature not found)`

**解决方案**：
```bash
# 重新安装h5py
pip uninstall h5py
pip install h5py==3.8.0

# 检查HDF5文件是否损坏
python -c "import h5py; f = h5py.File('data/test.hdf5', 'r'); print(f.keys())"
```

---

### 8. Git提交冲突

**团队协作最佳实践**：

```bash
# 1. 创建个人开发分支
git checkout -b feature/memberA-env

# 2. 每日同步主分支
git checkout main
git pull origin main
git checkout feature/memberA-env
git merge main

# 3. 小步提交，每次只改一个功能
git add .
git commit -m "feat: 添加环境测试脚本"

# 4. 推送前先pull
git pull --rebase origin main
git push origin feature/memberA-env
```

**禁止**：直接push到main分支

---

## 👥 团队分工

### 成员A：环境配置与代码框架负责人

**职责**：
- ✅ Conda环境配置与依赖管理
- ✅ 官方代码部署与调试
- ✅ 代码框架规范与目录结构
- ✅ 环境问题排查与解决
- ✅ Git仓库管理与协作规范

**产出物**：
- environment.yml（环境配置文件）
- requirements.txt（依赖版本锁定）
- README.md（项目文档）
- 代码框架与目录结构
- 环境搭建教程

**答辩讲解重点**：
- 环境配置过程中的坑点与解决方案
- 代码框架设计思路
- 依赖版本选择理由

---

### 成员B：测试验证与数据负责人

**职责**：
- ✅ 环境测试脚本编写与验证
- ✅ 数据集下载与预处理测试
- ✅ 单batch空跑测试
- ✅ 仿真环境验证
- ✅ 测试报告撰写

**产出物**：
- test_env.py（环境测试脚本）
- test_single_batch.py（单batch测试）
- test_sim.py（仿真测试）
- 环境验证报告
- 数据集格式说明

**答辩讲解重点**：
- 测试用例设计思路
- 环境验证结果
- 数据格式与预处理流程

---

## 📊 环境搭建检查清单

- [ ] Anaconda/Miniconda已安装
- [ ] NVIDIA显卡驱动已安装（≥450.80.02）
- [ ] CUDA 11.8 toolkit已安装
- [ ] conda环境创建成功
- [ ] PyTorch安装成功，CUDA可用
- [ ] MuJoCo安装成功，可导入
- [ ] OpenCV安装成功
- [ ] DETR模块安装成功
- [ ] test_env.py全部测试通过
- [ ] test_single_batch.py单batch测试通过
- [ ] 数据集下载完成
- [ ] 仿真环境可启动
- [ ] README文档完整
- [ ] 两人均验证环境可用

---

## 📚 参考资源

- **官方仓库**: https://github.com/tonyzhaozh/act
- **论文**: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
- **Mobile ALOHA项目**: https://mobile-aloha.github.io/
- **MuJoCo文档**: https://mujoco.readthedocs.io/
- **PyTorch文档**: https://pytorch.org/docs/

---

## 📝 更新日志

| 日期 | 版本 | 更新内容 | 负责人 |
|------|------|---------|--------|
| 2026-04-14 | v1.0 | 初始版本，完成环境配置文档 | 成员A |
| 2026-04-14 | v1.0 | 添加测试脚本与验证流程 | 成员B |

---

**课程**：具身智能基础（2025-2026-2学期）
**团队**：2人团队
**选题**：ACT算法复现
