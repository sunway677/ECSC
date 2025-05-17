import torch.nn as nn
import torch.nn.functional as F
import math
from .layers import ResidualBlock, UpBlock, FeatureProcessor

class ResAutoencoder(nn.Module):
    def __init__(self, snr_db=20, bottleneck_dim=24, skip_channels=10): # Modified to a configuration that results in exactly 1024
        super(ResAutoencoder, self).__init__()

        # Initialization
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Encoder blocks
        self.enc1 = ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, bias=False), nn.BatchNorm2d(128)
        ))
        self.enc2 = ResidualBlock(128, 256, stride=2, downsample=nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, bias=False), nn.BatchNorm2d(256)
        ))
        self.enc3 = ResidualBlock(256, 512, stride=2, downsample=nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, bias=False), nn.BatchNorm2d(512)
        ))

        # Feature processor - uniformly processes all features
        self.feature_processor = FeatureProcessor(snr_db, bottleneck_channels=bottleneck_dim,
                                                  skip_channels=skip_channels)

        # Decoder blocks
        self.dec3 = UpBlock(512, 256)
        self.dec2 = UpBlock(256, 128)
        self.dec1 = UpBlock(128, 64)

        # Final output layer
        self.final = nn.Sequential(nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid())

        # Store parameters
        self.bottleneck_dim = bottleneck_dim
        self.skip_channels = skip_channels

    def set_channel_snr(self, snr_db):
        """Set the channel SNR"""
        self.feature_processor.set_snr(snr_db)

    def get_current_snr(self):
        """Get the current SNR"""
        return self.feature_processor.get_snr()

    def forward(self, x):
        # Encoder part
        x0 = self.initial(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Uniformly process all features
        r_s0, r_s1, r_s2, r_s3, r_b, concat_feat, noisy_feat, denoised_feat = self.feature_processor(
            x0, x1, x2, x3, x3
        )

        # Decoder part
        d3 = self.dec3(r_b) + r_s2
        d2 = self.dec2(d3) + r_s1
        d1 = self.dec1(d2) + r_s0
        out = self.final(d1)

        # Store original skip features and skip features after passing through the channel, for loss calculation
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
        """Calculate the size of the compressed features"""
        # All features are compressed to a 4x4 size and concatenated
        # Each skip connection uses skip_channels channels, the bottleneck layer uses bottleneck_dim channels
        total = (self.skip_channels * 4 + self.bottleneck_dim) * 4 * 4

        # Calculate compression ratio
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
