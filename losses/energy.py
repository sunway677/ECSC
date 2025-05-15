import torch
import torch.nn as nn
import torch.nn.functional as F
import math


############################################
# EnergyNet Definition
############################################
class EnergyNet(nn.Module):
    """
    Energy Network that computes energy between feature pairs.
    Uses a combination of feature encoding and interaction layers.
    """

    def __init__(self, feature_dim, hidden_dim=256):
        """
        Initialize the Energy Network.

        Args:
            feature_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
        """
        super(EnergyNet, self).__init__()

        # Feature encoder: processes each input feature separately
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Interaction layer: processes feature pairs and their interactions
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def compute_pairwise_energy(self, f1, f2):
        """
        Compute energy between two sets of features.

        Args:
            f1: First set of features [B1, D]
            f2: Second set of features [B2, D]

        Returns:
            Energy matrix [B1, B2]
        """
        B1, B2 = f1.size(0), f2.size(0)

        # Encode both sets of features
        encoded_f1 = self.encoder(f1)  # [B1, H]
        encoded_f2 = self.encoder(f2)  # [B2, H]

        # Prepare feature pairs for interaction
        encoded_f1 = encoded_f1.unsqueeze(1).expand(-1, B2, -1)  # [B1, B2, H]
        encoded_f2 = encoded_f2.unsqueeze(0).expand(B1, -1, -1)  # [B1, B2, H]

        # Compute feature interactions
        interaction = encoded_f1 * encoded_f2  # [B1, B2, H]

        # Concatenate all features
        combined = torch.cat([
            encoded_f1,  # Original features 1
            encoded_f2,  # Original features 2
            interaction  # Feature interactions
        ], dim=-1)  # [B1, B2, 3H]

        # Compute energy
        energy = self.interaction(combined)  # [B1, B2, 1]
        return energy.squeeze(-1)  # [B1, B2]

    def forward(self, f_orig, f_recon):
        """Forward pass: compute energy between original and reconstructed features"""
        return self.compute_pairwise_energy(f_orig, f_recon)


############################################
# Compression Loss
############################################
class CompressedFeatureLoss(nn.Module):
    """压缩特征的损失函数，包括稀疏性和熵"""

    def __init__(self, beta=0.1):
        super(CompressedFeatureLoss, self).__init__()
        self.beta = beta

    def compute_entropy(self, x):
        # 将特征标准化到[0,1]区间
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        # 计算直方图
        hist = torch.histc(x, bins=256, min=0, max=1)
        # 归一化直方图得到概率分布
        p = hist / hist.sum()
        # 计算熵
        entropy = -torch.sum(p * torch.log2(p + 1e-8))
        return entropy

    def forward(self, compressed_features):
        total_sparsity = 0
        total_entropy = 0
        num_features = len(compressed_features)

        for name, feature in compressed_features.items():
            # L1正则化促进稀疏性
            sparsity = torch.mean(torch.abs(feature))
            # 计算特征的熵
            entropy = self.compute_entropy(feature)

            total_sparsity += sparsity
            total_entropy += entropy

        # 平均每个特征的损失
        avg_sparsity = total_sparsity / num_features
        avg_entropy = total_entropy / num_features

        return avg_sparsity + self.beta * avg_entropy


############################################
# Contrastive Loss with Free Energy
############################################
class ContrastiveLossWithFreeEnergy(nn.Module):
    """
    Energy-based contrastive loss with free energy formulation.
    """

    def __init__(self, tau=1.0, lambda_compress=0.01, lambda_energy=0.1):
        """
        Initialize the loss function.

        Args:
            tau: Temperature parameter for energy scaling
            lambda_compress: Weight for compression loss
        """
        super(ContrastiveLossWithFreeEnergy, self).__init__()
        self.tau = tau
        self.margin = 1.0  # 默认边际值
        self.K = 5  # 近邻数量
        self.lambda_compress = lambda_compress
        self.lambda_energy = lambda_energy

        self.energy_net = None
        self.compress_criterion = CompressedFeatureLoss()
        self.recon_criterion = nn.MSELoss()

    def init_networks(self, feature_dim, device):
        """Initialize the energy network if not already initialized"""
        if self.energy_net is None:
            self.energy_net = EnergyNet(feature_dim).to(device)

    def compute_knn_idx(self, feats):
        """
        返回 feats 空间中每个样本的 KNN 索引（不含自身）
        feats: [B, D]
        return: [B, K]
        """
        # B = feats.size(0)
        f_norm = F.normalize(feats, p=2, dim=1)
        sim = torch.mm(f_norm, f_norm.t())
        # 取 top-(K+1)，第一个肯定是自己
        _, idx = torch.topk(sim, self.K + 1, dim=1)
        return idx[:, 1:]  # [B, K]

    def compute_energy(self, f_orig, f_recon):
        """Compute energy between original and reconstructed features"""
        if self.energy_net is None:
            self.init_networks(f_orig.size(-1), f_orig.device)
        return self.energy_net(f_orig, f_recon)

    def forward_energy_margin(self, f_orig, f_recon):
        """
        增强版margin-based loss，保持语义空间拓扑结构
        结合了标准对角线正例匹配和邻居结构保持
        """
        # 1) 计算标准能量矩阵
        E = self.compute_energy(f_orig, f_recon)  # [B,B]
        B = E.size(0)
        device = E.device

        # 2) 构造正例mask：只有对角元素
        pos_mask = torch.eye(B, device=device, dtype=torch.bool)  # [B,B]

        # 3) 计算标准hinge loss，只对非正例位置生效
        pos = torch.diag(E).unsqueeze(1).expand(-1, B)  # [B,B]
        losses = F.relu(self.margin + E - pos)  # [B,B]
        neg_mask = ~pos_mask  # 只选"真正负例"
        standard_loss = losses.masked_select(neg_mask).mean()

        # 4) 计算语义结构保持部分
        # 设置K值（近邻数量）
        k = min(5, B - 1)

        # 计算原始特征空间中的K近邻
        f_orig_norm = F.normalize(f_orig, p=2, dim=1)
        sim_orig = torch.matmul(f_orig_norm, f_orig_norm.T)  # 余弦相似度
        sim_orig.fill_diagonal_(-float('inf'))  # 排除自身
        _, orig_nn_idx = torch.topk(sim_orig, k=k, dim=1)  # [B,k]

        # 计算重建特征空间中的K近邻
        f_recon_norm = F.normalize(f_recon, p=2, dim=1)
        sim_recon = torch.matmul(f_recon_norm, f_recon_norm.T)
        sim_recon.fill_diagonal_(-float('inf'))
        _, recon_nn_idx = torch.topk(sim_recon, k=k, dim=1)  # [B,k]

        # 创建邻居矩阵（one-hot格式）
        orig_neighbors = torch.zeros(B, B, device=device)
        recon_neighbors = torch.zeros(B, B, device=device)

        # 填充邻居矩阵
        for i in range(B):
            orig_neighbors[i, orig_nn_idx[i]] = 1.0
            recon_neighbors[i, recon_nn_idx[i]] = 1.0

        # 计算交集和并集（使用张量运算）
        intersection = torch.sum(orig_neighbors * recon_neighbors, dim=1)  # [B]
        union = torch.sum(torch.clamp(orig_neighbors + recon_neighbors, 0, 1), dim=1)  # [B]

        # 计算每个样本的杰卡德相似度并平均
        jaccard = intersection / (union + 1e-8)  # 添加小常数避免除零
        neighbor_loss = 1-jaccard.mean()  # 取负转换为损失

        # 5) 组合损失
        # 使用0.3作为邻居结构保持损失的权重
        total_loss = standard_loss + 0.3 * neighbor_loss

        return total_loss

    def forward(self, outputs, images, f_orig, f_recon, compressed_features):
        """
        Forward pass: compute total loss including reconstruction, energy,
        and compression losses.

        Args:
            outputs: Reconstructed images
            images: Original images
            f_orig: Original features
            f_recon: Reconstructed features
            compressed_features: Dictionary of compressed features

        Returns:
            Dictionary containing all loss terms as tensors
        """
        # Reconstruction loss
        recon_loss = self.recon_criterion(outputs, images)

        # Energy loss
        energy_loss = self.forward_energy_margin(f_orig, f_recon)

        # Compression loss
        compress_loss = self.compress_criterion(compressed_features)

        # Combine all losses
        total_loss = (recon_loss +
                      self.lambda_compress * compress_loss +
                      self.lambda_energy * energy_loss)

        # Return dictionary with tensors (not converted to float)
        return {
            'total': total_loss,
            'recon': recon_loss,
            'energy': energy_loss,
            'compress': compress_loss
        }