# 环境搭建任务分工与分步指南

> 适用：2人零基础团队，ACT算法复现项目
> 目标：环境搭建与代码运行评分项拿13-15分（满分档）

---

## 📋 分工总览

| 成员 | 角色 | 核心职责 | 产出物 |
|------|------|---------|--------|
| **成员A** | 环境架构师 | 环境配置、依赖管理、代码框架、问题排查 | environment.yml、requirements.txt、README.md、调试记录 |
| **成员B** | 测试工程师 | 测试脚本、环境验证、数据准备、测试报告 | test_env.py、test_single_batch.py、test_sim.py、验证报告 |

**分工原则**：
- 成员A负责"搭"：把环境搭起来，解决各种安装报错
- 成员B负责"测"：验证环境是否真的能用，写测试脚本
- 两人各有独立产出，答辩时各讲各的部分

---

## 👨‍💻 成员A：环境配置与代码框架负责人

### 任务清单

| 序号 | 任务 | 预计耗时 | 完成标准 |
|------|------|---------|---------|
| 1 | 安装Anaconda/Miniconda | 30分钟 | conda命令可用 |
| 2 | 配置conda镜像源（加速下载） | 15分钟 | conda install速度明显提升 |
| 3 | 编写environment.yml | 30分钟 | 所有依赖版本锁定 |
| 4 | 创建conda环境并安装依赖 | 1-2小时 | conda env create成功 |
| 5 | 安装PyTorch（CUDA版本） | 30分钟 | torch.cuda.is_available()=True |
| 6 | 安装MuJoCo仿真环境 | 1-2小时 | import mujoco无报错 |
| 7 | 部署官方ACT代码 | 30分钟 | 代码克隆完成，目录结构清晰 |
| 8 | 安装DETR模块 | 15分钟 | detr模块可导入 |
| 9 | 排查并解决所有安装报错 | 2-4小时 | import核心库无报错 |
| 10 | 编写README文档 | 1小时 | 文档完整，新人可按文档安装 |
| 11 | 导出环境配置文件 | 15分钟 | environment.yml可复现 |
| 12 | 协助成员B验证环境 | 30分钟 | 两人环境一致 |

### 分步操作指南

#### 第1步：安装Anaconda

**Windows用户**：
1. 下载 Anaconda 安装包：https://www.anaconda.com/download
2. 双击安装，一路下一步（建议安装路径不要有中文和空格）
3. 安装完成后打开「Anaconda Prompt」

**Linux用户**：
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
bash Anaconda3-2023.07-2-Linux-x86_64.sh
```

**验证安装**：
```bash
conda --version
# 输出类似: conda 23.7.4
```

#### 第2步：配置conda镜像源（加速）

```bash
# 添加清华镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --set show_channel_urls yes

# 配置pip镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 第3步：创建conda环境

```bash
# 方式一：从environment.yml创建（推荐，版本锁定）
conda env create -f environment.yml

# 方式二：手动创建
conda create -n act python=3.8.10
conda activate act
```

#### 第4步：安装PyTorch（CUDA 11.8）

```bash
# 激活环境
conda activate act

# 安装PyTorch（CUDA 11.8版本）
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 验证
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda)"
```

**⚠️ 常见问题**：
- 如果CUDA不可用，检查显卡驱动版本是否 ≥ 450.80.02
- 命令：`nvidia-smi` 查看驱动版本

#### 第5步：安装MuJoCo

```bash
# 安装mujoco
pip install mujoco==2.3.7

# 安装dm_control（可选）
pip install dm-control==1.0.14

# 验证
python -c "import mujoco; print('MuJoCo版本:', mujoco.__version__)"
```

**⚠️ Windows常见问题**：
- 报错：`DLL load failed`
- 解决方案：安装 Visual C++ Redistributable，或使用WSL2

**⚠️ 服务器无显示器**：
```bash
# 设置环境变量（加入~/.bashrc）
export MUJOCO_GL=egl
```

#### 第6步：克隆代码仓库

```bash
# 克隆仓库
git clone https://github.com/LiHua69/-ACT-Action-Chunking-with-Transformers-.git
cd -ACT-Action-Chunking-with-Transformers-

# 安装DETR模块
cd detr
pip install -e .
cd ..
```

#### 第7步：安装其他依赖

```bash
# 方式一：从requirements.txt安装
pip install -r requirements.txt

# 方式二：手动安装核心依赖
pip install numpy scipy h5py opencv-python matplotlib pyyaml tqdm einops pyquaternion
```

#### 第8步：导出环境配置

```bash
# 导出conda环境
conda env export > environment.yml

# 导出pip依赖
pip freeze > requirements.txt
```

#### 第9步：编写README文档

按照README模板填写：
- 硬件要求
- 安装步骤
- 常见问题
- 团队分工

---

## 👩‍💻 成员B：测试验证与数据负责人

### 任务清单

| 序号 | 任务 | 预计耗时 | 完成标准 |
|------|------|---------|---------|
| 1 | 学习Python基础语法 | 2-3小时 | 能看懂简单Python代码 |
| 2 | 编写环境测试脚本test_env.py | 1小时 | 能检测所有依赖是否安装 |
| 3 | 编写单batch测试脚本 | 2小时 | 能验证模型前向反向传播 |
| 4 | 编写仿真环境测试脚本 | 1小时 | 能验证MuJoCo渲染功能 |
| 5 | 运行环境测试并记录结果 | 30分钟 | 生成测试报告 |
| 6 | 下载并检查数据集 | 1-2小时 | 数据集格式正确，可读取 |
| 7 | 编写数据加载测试脚本 | 1小时 | 能验证数据预处理流程 |
| 8 | 协助成员A排查环境问题 | 1小时 | 提供测试反馈 |
| 9 | 整理环境验证报告 | 1小时 | 报告完整，有截图有数据 |
| 10 | 验证两人环境一致性 | 30分钟 | 两人测试结果相同 |

### 分步操作指南

#### 第1步：环境测试脚本编写

**目标**：写一个脚本，一键检测所有依赖是否安装成功。

**脚本功能**：
- 检测Python版本
- 检测PyTorch和CUDA
- 检测NumPy、OpenCV、H5Py等
- 检测MuJoCo
- 输出彩色测试报告

参考脚本：`scripts/test_env.py`

#### 第2步：运行环境测试

```bash
# 运行测试
python scripts/test_env.py

# 保存测试结果到文件
python scripts/test_env.py > logs/env_test_report.txt
```

**记录测试结果**：
- 截图测试通过的界面
- 记录GPU型号、显存大小
- 记录PyTorch版本、CUDA版本
- 如果有失败项，记录错误信息

#### 第3步：单Batch空跑测试

**目标**：验证模型能不能跑起来，数据能不能喂进去。

**测试内容**：
- 创建模拟数据集
- 初始化ACT模型
- 前向传播（计算损失）
- 反向传播（更新参数）
- 推理测试

参考脚本：`scripts/test_single_batch.py`

```bash
# 运行单batch测试
python scripts/test_single_batch.py
```

**记录**：
- 模型参数量
- 单步训练耗时
- 初始损失值
- 显存占用

#### 第4步：仿真环境测试

```bash
# 运行仿真测试
python scripts/test_sim.py
```

**测试内容**：
- MuJoCo基础功能
- 离屏渲染
- dm_control（可选）
- ALOHA环境（可选）

#### 第5步：数据集准备

**下载数据集**：
```bash
# 创建数据目录
mkdir -p data

# 下载示例数据集（实际地址请咨询老师）
# wget [数据集地址] -P data/

# 解压
# unzip data/dataset.zip -d data/
```

**检查数据集格式**：
```python
import h5py

# 打开一个hdf5文件
f = h5py.File('data/episode_0.hdf5', 'r')

# 查看有哪些key
print('数据集keys:', list(f.keys()))

# 查看数据shape
for key in f.keys():
    print(f'{key}: {f[key].shape}')

f.close()
```

**记录**：
- 数据集大小
- 样本数量
- 图像尺寸
- 动作维度
- 每集时长

#### 第6步：编写数据加载测试

```python
# 写一个简单的数据加载测试脚本
# 验证：
# 1. 能读取hdf5文件
# 2. 图像预处理正确
# 3. 动作归一化正确
# 4. DataLoader能正常工作
```

#### 第7步：整理测试报告

**报告内容**：
1. 测试环境信息
   - 操作系统
   - CPU型号
   - GPU型号+显存
   - 内存大小
2. 环境测试结果
   - 所有依赖版本
   - 测试通过/失败项
   - 截图
3. 单batch测试结果
   - 模型参数量
   - 训练耗时
   - 损失变化
   - 显存占用
4. 仿真测试结果
   - MuJoCo版本
   - 渲染是否正常
   - 截图
5. 数据集信息
   - 数据集来源
   - 样本数量
   - 数据格式

---

## 📅 环境搭建时间规划（3天）

### Day 1：基础环境搭建（成员A主导）

| 时间 | 任务 | 负责人 |
|------|------|--------|
| 上午 | 安装Anaconda，配置镜像源 | 成员A |
| 上午 | 创建conda环境，安装Python | 成员A |
| 下午 | 安装PyTorch，验证CUDA | 成员A |
| 下午 | 安装MuJoCo，解决报错 | 成员A |
| 晚上 | 成员B开始学习Python基础 | 成员B |

### Day 2：代码部署与测试脚本（并行）

| 时间 | 任务 | 负责人 |
|------|------|--------|
| 上午 | 克隆代码仓库，安装DETR | 成员A |
| 上午 | 编写test_env.py测试脚本 | 成员B |
| 下午 | 安装其他依赖，调试报错 | 成员A |
| 下午 | 编写test_single_batch.py | 成员B |
| 晚上 | 导出环境配置文件 | 成员A |
| 晚上 | 编写test_sim.py | 成员B |

### Day 3：验证与文档

| 时间 | 任务 | 负责人 |
|------|------|--------|
| 上午 | 成员B配置自己的环境 | 成员B |
| 上午 | 成员A编写README文档 | 成员A |
| 下午 | 运行所有测试脚本 | 两人 |
| 下午 | 记录测试结果，截图 | 成员B |
| 晚上 | 整理测试报告 | 成员B |
| 晚上 | 完善README，提交代码 | 成员A |

---

## ✅ 环境搭建完成标准（13-15分档）

### 必须满足

- [ ] **环境配置完整**：conda环境可一键创建
- [ ] **依赖版本锁定**：environment.yml + requirements.txt齐全
- [ ] **代码一键运行**：按README步骤可完成安装
- [ ] **README详细**：包含安装步骤、硬件要求、常见问题
- [ ] **支持多平台复现**：至少2人独立验证通过
- [ ] **无致命报错**：import所有核心库无报错
- [ ] **单batch空跑成功**：模型能前向+反向传播
- [ ] **测试脚本齐全**：至少3个测试脚本
- [ ] **常见问题文档化**：8个以上常见问题及解决方案
- [ ] **分工明确**：两人各有独立产出

### 加分项

- [ ] 提供Dockerfile
- [ ] 提供一键安装脚本
- [ ] 视频演示安装过程
- [ ] 测试覆盖率高
- [ ] 代码有注释

---

## 🎯 答辩准备

### 成员A答辩要点（环境部分）

1. **为什么选择这些版本？**
   - Python 3.8：兼容性最好，官方推荐
   - PyTorch 2.0：稳定，支持CUDA 11.8
   - MuJoCo 2.3.7：官方代码使用的版本

2. **环境搭建遇到了什么坑？怎么解决的？**
   - 准备2-3个真实遇到的问题
   - 说明排查过程和解决方案
   - 体现解决问题的能力

3. **如何保证环境可复现？**
   - conda env export导出完整环境
   - pip freeze锁定所有pip包版本
   - README详细步骤
   - 两人独立验证

### 成员B答辩要点（测试部分）

1. **测试脚本是怎么设计的？**
   - 为什么要测这些项
   - 测试用例怎么选的
   - 如何判断测试通过

2. **测试结果如何？**
   - 环境配置信息
   - 性能数据（耗时、显存）
   - 遇到了什么问题

3. **数据集是怎么处理的？**
   - 数据格式说明
   - 预处理流程
   - 数据增强策略

---

## 📞 问题排查流程

当遇到报错时，按以下步骤排查：

1. **读报错信息**：仔细看Error的最后一行，是什么类型的错误
2. **搜解决方案**：复制报错信息到Google/百度/GitHub Issues搜索
3. **检查版本**：确认依赖版本是否和requirements.txt一致
4. **重装试试**：有时候重装能解决玄学问题
5. **问队友**：成员A如果遇到问题，成员B可以帮忙搜索
6. **记录下来**：把问题和解决方案记下来，写进README的FAQ

---

**最后更新**：2026-04-14
**版本**：v1.0
