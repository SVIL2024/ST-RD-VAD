import torch
import torch.nn as nn
import torch.nn.functional as F


def get_ano_map(feature1, feature2):
    # —— 新增：在通道维做 L2 归一化 ——
    f1_norm = F.normalize(feature1, p=2, dim=1)
    f2_norm = F.normalize(feature2, p=2, dim=1)

    # 保留原始特征用于 MSE 计算
    mse_loss = nn.MSELoss(reduction='none')
    mse = mse_loss(feature1, feature2)  # [B, C, D, H, W]
    mse = torch.mean(mse, dim=1)  # [B, D, H, W]

    # 余弦相似度（基于归一化后特征）
    cos = F.cosine_similarity(f1_norm, f2_norm, dim=1)  # [B, D, H, W]
    ano_map = 1.0 - cos  # [B, D, H, W]

    # 全图平均损失
    loss = ano_map.view(ano_map.shape[0], -1).mean(-1).mean()
    return ano_map.unsqueeze(1), loss, mse.unsqueeze(1)


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, feature1, feature2):
        # —— 新增：在通道维做 L2 归一化 ——
        f1_norm = F.normalize(feature1, p=2, dim=1)
        f2_norm = F.normalize(feature2, p=2, dim=1)

        # 计算余弦距离
        cos = F.cosine_similarity(f1_norm, f2_norm, dim=1)  # [B, D, H, W]
        ano_map = 1.0 - cos
        loss = ano_map.view(ano_map.shape[0], -1).mean(-1).mean()
        return loss