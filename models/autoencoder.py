import torch.nn as nn
import torch.nn.functional as F
import math
from .layers import ResidualBlock, UpBlock, FeatureProcessor

class ResAutoencoder(nn.Module):
    def __init__(self, snr_db=20, bottleneck_dim=24, skip_channels=10):  # 修改为正好1024的配置
        super(ResAutoencoder, self).__init__()

        # 初始化
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 编码器块
        self.enc1 = ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, bias=False), nn.BatchNorm2d(128)
        ))
        self.enc2 = ResidualBlock(128, 256, stride=2, downsample=nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, bias=False), nn.BatchNorm2d(256)
        ))
        self.enc3 = ResidualBlock(256, 512, stride=2, downsample=nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, bias=False), nn.BatchNorm2d(512)
        ))

        # 特征处理器 - 统一处理所有特征
        self.feature_processor = FeatureProcessor(snr_db, bottleneck_channels=bottleneck_dim,
                                                  skip_channels=skip_channels)

        # 解码器块
        self.dec3 = UpBlock(512, 256)
        self.dec2 = UpBlock(256, 128)
        self.dec1 = UpBlock(128, 64)

        # 最终输出层
        self.final = nn.Sequential(nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid())

        # 存储参数
        self.bottleneck_dim = bottleneck_dim
        self.skip_channels = skip_channels

    def set_channel_snr(self, snr_db):
        """设置信道SNR"""
        self.feature_processor.set_snr(snr_db)

    def get_current_snr(self):
        """获取当前SNR"""
        return self.feature_processor.get_snr()

    def forward(self, x):
        # 编码器部分
        x0 = self.initial(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # 统一处理所有特征
        r_s0, r_s1, r_s2, r_s3, r_b, concat_feat, noisy_feat, denoised_feat = self.feature_processor(
            x0, x1, x2, x3, x3
        )

        # 解码器部分
        d3 = self.dec3(r_b) + r_s2
        d2 = self.dec2(d3) + r_s1
        d1 = self.dec1(d2) + r_s0
        out = self.final(d1)

        # 保存原始skip特征和经过信道的skip特征，用于损失计算
        original_skips = {'skip0': x0, 'skip1': x1, 'skip2': x2, 'skip3': x3}
        processed_skips = {'skip0': r_s0, 'skip1': r_s1, 'skip2': r_s2, 'skip3': r_s3}

        compressed = {
            'bottleneck': r_b,
            'skip0': r_s0,
            'skip1': r_s1,
            'skip2': r_s2,
            'skip3': r_s3,
            'concatenated': concat_feat,
            'noisy': noisy_feat,
            'denoised': denoised_feat,
            'original_skips': original_skips,
            'processed_skips': processed_skips
        }

        return out, compressed

    def get_compressed_size(self):
        """计算压缩特征的大小"""
        # 所有特征都被压缩到4x4尺寸并拼接
        # 每个跳跃连接使用skip_channels通道，瓶颈层使用bottleneck_dim通道
        total = (self.skip_channels * 4 + self.bottleneck_dim) * 4 * 4

        # 计算压缩比
        orig_size = 3 * 32 * 32
        compression_ratio = orig_size / total

        return {
            'concatenated_size': total,
            'skip_channels': self.skip_channels,
            'bottleneck_channels': self.bottleneck_dim,
            'spatial_size': 4,
            'total': total,
            'compression_ratio': compression_ratio
        }