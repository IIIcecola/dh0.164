import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np


class ModularDataset(Dataset):
  def __init__(self, module_mapping):
    """
    模块化数据集：每个样本绑定所属模块标签
    param module_mapping: {模块名：样本数量}，模拟不同模块的样本
    """
    self.samples = []
    self.module_labels = []

    for module_name, sample_num in module_mapping.items():
      for _ in range(sample_num):
        self.samples.append(torch.randn(3, 256, 256))
        self.module_labels.append(module_name)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx], self.module_labels[idx]

# ===============================初始化数据集+加权采样（保留模块化采样）==========================
# 定义2个数据模块（可扩展为任意数量）

module_mapping = {
  "general_module": 100,
  "high_priority_module"：30
}

# 1. 模块采样权重（控制各模块样本被抽到的概率）
module_sample_weight = {
  "general_module": 0.3,
  "high_priority_module"：0.7
}

# 2. 模块loss权重（核心：给不同模块的样本赋予不同的loss权重）
module_loss_weight = {
  "general_module": 0.2, # 通用模块样本的loss权重（贡献小）
  "high_priority_module"：1.0 # 高优先级模块样本的loss权重（贡献大）
}

# 初始胡dataset
dataset = ModuleDataset(module_mapping)

# 生成样本级采样权重（用于WeightedRandomSmapler）
sample_weights = torch.tensor(
  [module_sample_weight[m] for m in dataset.module_labels],
  dtype=torch.float32
)

# 加权采样器
sampler = WeightedRandomSmapler(
  weights=sample_weights,
  num_samples=len(dataset),
  replacement=True
)

# 数据加载器
dataloader = DataLoader(
  dataset,
  batch_size=8,
  sampler=sampler,
  shuffle=False
)

# =====================定义模型+优化器======================
# 简单的视觉生成模型（替换为你的视觉生成模型）
class SimpleVisionModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.backbone = nn.Sequential(
      nn.Conv2d(3, 64, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 3, 3, padding=1)
    )

  def forward(self, x):
    return self.backbone(x)

# 初始化模型、优化器、损失函数
model = SimpleVisionModel()
base_lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.01)
criterion  = nn.MSELoss(reduction="none")

# ====================核心训练循环===============================
def train_with_batch_loss_weight(model, dataloader, optimizer, criterion, module_loss_weight, epochs=5):
  model.train()
  for epoch in range(epochs):
    total_loss = 0.0
    for batch_idx, (batch_data, batch_modules) in enumerate(dataloader):
      optimizer.zero_grad() # 清空梯度
      # 整个批次一次性前向传播
      batch_output = model(batch_data)
      # 计算每个样本的原始loss
      # criterion的reduction='none'，输出shape为[batch_size]（每个样本的loss）
      sample_raw_loss = criterion(batch_output, batch_data).mean(dim=[1,2,3])
      # 为每个样本赋予对应模块的loss权重
      sample_weighted_loss = []
      for loss, module in zip(sample_raw_loss, batch_modules):
        # 获取样本所属的loss权重
        weight = module_loss_weight[module]
        # 加权后的样本loss
        sample_weighted_loss.append(loss*weight)
      # 转换为张量，并计算整个batch的总loss（求和或者求平均）
      batch_total_loss = torch.stack(sample_weighted_loss).sum()
      
      # 单次反向传播+参数更新
      batch_total_loss.backword()
      optimizer.step()

      # 累计损失用于日志
      total_loss += batch_total_loss.item()

    # 打印每轮训练日志
    avg_loss = total_loss / len(dataloader)

# 启动训练
train_with_batch_loss_weight(model, dataloader, optimizer, criterion, module_loss_weight, epochs=5)














