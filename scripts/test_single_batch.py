#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单Batch空跑测试脚本
验证模型结构、数据加载、前向传播、反向传播是否正常工作

使用方法:
    python scripts/test_single_batch.py
"""

import sys
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

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

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def print_header(msg):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{msg}{Colors.END}")


class SimpleACTDataset(torch.utils.data.Dataset):
    """
    简化版ACT数据集，用于测试
    生成模拟数据，不依赖真实数据集
    """
    def __init__(self, num_samples=100, img_height=480, img_width=640, 
                 action_dim=14, chunk_size=100):
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.action_dim = action_dim
        self.chunk_size = chunk_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 模拟图像输入 (3通道, H, W)
        image = torch.randn(3, self.img_height, self.img_width)
        
        # 模拟关节位置观测
        qpos = torch.randn(self.action_dim)
        
        # 模拟动作序列 (chunk_size, action_dim)
        action = torch.randn(self.chunk_size, self.action_dim)
        
        # 归一化到 [-1, 1]
        action = torch.tanh(action)
        
        return {
            'image': image,
            'qpos': qpos,
            'action': action
        }


class SimpleACTModel(nn.Module):
    """
    简化版ACT模型，用于测试
    包含CVAE encoder-decoder基本结构
    """
    def __init__(self, action_dim=14, chunk_size=100, hidden_dim=256, 
                 nheads=8, num_encoder_layers=4, num_decoder_layers=7):
        super().__init__()
        
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        
        # 视觉编码器 (简化版，实际使用ResNet/DETR)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden_dim)
        )
        
        # 状态编码器
        self.state_encoder = nn.Linear(action_dim, hidden_dim)
        
        # CVAE Encoder (动作 -> 潜在变量)
        self.action_encoder = nn.Sequential(
            nn.Linear(chunk_size * action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        # 动作查询嵌入
        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim))
        
        # 输出头
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # 投影层
        self.latent_proj = nn.Linear(hidden_dim // 2, hidden_dim)
    
    def encode(self, action):
        """CVAE编码器：动作 -> 潜在分布"""
        batch_size = action.shape[0]
        action_flat = action.reshape(batch_size, -1)
        h = self.action_encoder(action_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, visual_feat, state_feat, z):
        """解码器：观测 + 潜在变量 -> 动作序列"""
        batch_size = visual_feat.shape[0]
        
        # 融合观测特征
        obs_feat = visual_feat + state_feat
        
        # 潜在变量投影
        z_proj = self.latent_proj(z)
        
        # 记忆（条件）= 观测特征 + 潜在变量
        memory = (obs_feat + z_proj).unsqueeze(1)  # [B, 1, D]
        
        # 动作查询
        queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Transformer解码
        output = self.transformer_decoder(queries, memory)
        
        # 输出动作
        action_pred = self.action_head(output)
        
        return action_pred
    
    def forward(self, image, qpos, action):
        """
        前向传播（训练模式）
        返回损失字典
        """
        batch_size = image.shape[0]
        
        # 编码观测
        visual_feat = self.visual_encoder(image)
        state_feat = self.state_encoder(qpos)
        
        # 编码动作（CVAE encoder）
        mu, logvar = self.encode(action)
        z = self.reparameterize(mu, logvar)
        
        # 解码
        action_pred = self.decode(visual_feat, state_feat, z)
        
        # 计算损失
        # 1. 重构损失 (MSE)
        recon_loss = nn.functional.mse_loss(action_pred, action)
        
        # 2. KL散度损失
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # KL权重
        kl_weight = 10.0
        
        total_loss = recon_loss + kl_weight * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'action_pred': action_pred
        }
    
    def infer(self, image, qpos):
        """
        推理模式（不使用真实动作）
        """
        batch_size = image.shape[0]
        
        # 编码观测
        visual_feat = self.visual_encoder(image)
        state_feat = self.state_encoder(qpos)
        
        # 推理时使用先验（z=0）
        z = torch.zeros(batch_size, self.hidden_dim // 2, device=image.device)
        
        # 解码
        action_pred = self.decode(visual_feat, state_feat, z)
        
        return action_pred


def main():
    print("\n" + "=" * 50)
    print(f"{Colors.BOLD}    单Batch空跑测试{Colors.END}")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_info(f"使用设备: {device}")
    
    # ===== 1. 数据集测试 =====
    print_header("1. 数据集测试")
    
    try:
        dataset = SimpleACTDataset(
            num_samples=100,
            img_height=480,
            img_width=640,
            action_dim=14,
            chunk_size=100
        )
        print_success(f"数据集创建成功，样本数: {len(dataset)}")
        
        # 测试单个样本
        sample = dataset[0]
        print_success(f"图像shape: {sample['image'].shape}")
        print_success(f"关节位置shape: {sample['qpos'].shape}")
        print_success(f"动作shape: {sample['action'].shape}")
        
        # DataLoader测试
        batch_size = 8
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        print_success(f"Batch size: {batch_size}")
        print_success(f"Batch图像shape: {batch['image'].shape}")
        print_success(f"Batch动作shape: {batch['action'].shape}")
        
    except Exception as e:
        print_error(f"数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== 2. 模型初始化 =====
    print_header("2. 模型初始化")
    
    try:
        model = SimpleACTModel(
            action_dim=14,
            chunk_size=100,
            hidden_dim=256,
            nheads=8,
            num_encoder_layers=4,
            num_decoder_layers=7
        )
        model.to(device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print_success(f"模型参数总数: {total_params / 1e6:.1f}M")
        print_success(f"可训练参数: {trainable_params / 1e6:.1f}M")
        print_success("模型初始化成功")
        
    except Exception as e:
        print_error(f"模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== 3. 前向传播测试 =====
    print_header("3. 前向传播测试")
    
    try:
        image = batch['image'].to(device)
        qpos = batch['qpos'].to(device)
        action = batch['action'].to(device)
        
        start_time = time.time()
        
        with torch.no_grad():
            output = model(image, qpos, action)
        
        forward_time = time.time() - start_time
        
        print_success(f"前向传播成功")
        print_success(f"总损失: {output['total_loss'].item():.4f}")
        print_success(f"KL散度: {output['kl_loss'].item():.4f}")
        print_success(f"重构损失: {output['recon_loss'].item():.4f}")
        print_success(f"预测动作shape: {output['action_pred'].shape}")
        print_info(f"前向传播耗时: {forward_time*1000:.1f}ms")
        
    except Exception as e:
        print_error(f"前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== 4. 反向传播测试 =====
    print_header("4. 反向传播测试")
    
    try:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # 训练模式
        model.train()
        
        start_time = time.time()
        
        # 前向
        output = model(image, qpos, action)
        loss = output['total_loss']
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新
        optimizer.step()
        
        backward_time = time.time() - start_time
        
        print_success("反向传播成功")
        print_success("参数更新成功")
        print_info(f"单步训练耗时: {backward_time*1000:.1f}ms")
        
        # 检查梯度
        max_grad = max(p.grad.abs().max() for p in model.parameters() if p.grad is not None)
        print_info(f"最大梯度值: {max_grad.item():.4f}")
        
    except Exception as e:
        print_error(f"反向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== 5. 推理测试 =====
    print_header("5. 推理测试")
    
    try:
        model.eval()
        
        start_time = time.time()
        
        with torch.no_grad():
            action_pred = model.infer(image, qpos)
        
        infer_time = time.time() - start_time
        
        print_success(f"推理成功")
        print_success(f"推理输出shape: {action_pred.shape}")
        print_info(f"推理耗时: {infer_time*1000:.1f}ms")
        
        # 检查输出范围
        print_info(f"动作范围: [{action_pred.min().item():.2f}, {action_pred.max().item():.2f}]")
        
    except Exception as e:
        print_error(f"推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== 6. 显存使用统计 =====
    if torch.cuda.is_available():
        print_header("6. 显存使用统计")
        
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        print_info(f"已分配显存: {allocated:.1f} MB")
        print_info(f"已保留显存: {reserved:.1f} MB")
    
    # ===== 总结 =====
    print("\n" + "=" * 50)
    print(f"{Colors.GREEN}{Colors.BOLD}🎉 单Batch空跑测试通过！{Colors.END}")
    print("=" * 50)
    print("\n说明:")
    print("  - 本测试使用模拟数据，验证模型结构和训练流程")
    print("  - 真实数据训练请使用完整数据集和官方模型实现")
    print("  - 损失值初始较大是正常的，训练后会下降")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
