import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelNoise(nn.Module):
    def __init__(self, snr_db=20):
        """
        :param snr_db: Signal-to-noise ratio in dB
        """
        super(ChannelNoise, self).__init__()
        self.snr_db = snr_db

    def set_snr(self, snr_db):
        """Dynamically set the SNR value."""
        self.snr_db = snr_db

    def forward(self, x):
        """
        Simulate an AWGN channel by adding Gaussian noise based on the specified SNR.
        """
        # Calculate the average signal power
        signal_power = torch.mean(x ** 2)

        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (self.snr_db / 10)

        # Calculate the noise power using the signal power and SNR
        noise_power = signal_power / snr_linear

        # Compute the noise standard deviation
        noise_std = torch.sqrt(noise_power)

        # Generate and add Gaussian noise to the input
        noise = torch.randn_like(x) * noise_std

        return x + noise


class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer for channel attention."""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        # Squeeze: Global average pooling to get channel-wise statistics
        y = self.avg_pool(x).view(b, c)
        # Excitation: Fully connected layers to compute attention weights
        y = self.fc(y).view(b, c, 1, 1)
        # Scale the input feature maps with the computed attention weights
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Standard Residual Block used in ResNet architectures."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # First convolution + batch normalization + ReLU activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolution + batch normalization
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample the identity if required to match dimensions
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the residual (skip connection) and apply ReLU
        out += identity
        out = self.relu(out)
        return out


class UpBlock(nn.Module):
    """Upsampling block for the decoder section."""

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # Upsample the feature map by a factor of 2 using nearest neighbor interpolation
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Upsample, convolve, batch normalize, and activate
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UnifiedDenoiser(nn.Module):
    """Unified denoising module, optimized for AWGN noise."""

    def __init__(self, in_channels, depth=5):
        super(UnifiedDenoiser, self).__init__()

        # Initial feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Residual denoising blocks
        layers = []
        for _ in range(depth):
            layers.append(nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64)
            ))
        self.res_blocks = nn.ModuleList(layers)

        # Output layer - predicts residual noise
        self.tail = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Feature extraction
        f = self.head(x)

        # Residual denoising process
        res = f
        for block in self.res_blocks:
            res = res + block(res) # Residual connection within the denoiser

        # Predict noise and subtract it from the input
        noise = self.tail(res)
        denoised = x - noise

        return denoised


class FeatureProcessor(nn.Module):
    """
    Feature processing module: processes each feature separately
    - Applies noise to each feature individually
    - Denoises each feature individually
    - No longer merges compressed features for transmission
    """

    def __init__(self, snr_db=20, bottleneck_channels=20, skip_channels=9):
        super(FeatureProcessor, self).__init__()

        # Channel noise module
        self.channel_noise = ChannelNoise(snr_db)

        # 1x1 convolutions to adjust channels before feature concatenation/processing
        self.s0_adjust = nn.Conv2d(64, skip_channels, kernel_size=1)
        self.s1_adjust = nn.Conv2d(128, skip_channels, kernel_size=1)
        self.s2_adjust = nn.Conv2d(256, skip_channels, kernel_size=1)
        self.s3_adjust = nn.Conv2d(512, skip_channels, kernel_size=1)
        self.b_adjust = nn.Conv2d(512, bottleneck_channels, kernel_size=1)

        # Individual denoisers - create a dedicated denoiser for each feature
        self.denoiser_s0 = UnifiedDenoiser(in_channels=skip_channels)
        self.denoiser_s1 = UnifiedDenoiser(in_channels=skip_channels)
        self.denoiser_s2 = UnifiedDenoiser(in_channels=skip_channels)
        self.denoiser_s3 = UnifiedDenoiser(in_channels=skip_channels)
        self.denoiser_b = UnifiedDenoiser(in_channels=bottleneck_channels)

        # 1x1 convolutions to restore channels after feature splitting/processing
        self.s0_restore = nn.Conv2d(skip_channels, 64, kernel_size=1)
        self.s1_restore = nn.Conv2d(skip_channels, 128, kernel_size=1)
        self.s2_restore = nn.Conv2d(skip_channels, 256, kernel_size=1)
        self.s3_restore = nn.Conv2d(skip_channels, 512, kernel_size=1)
        self.b_restore = nn.Conv2d(bottleneck_channels, 512, kernel_size=1)

        # Save channel number parameters
        self.bottleneck_channels = bottleneck_channels
        self.skip_channels = skip_channels

    def set_snr(self, snr_db):
        """Set the channel SNR"""
        self.channel_noise.set_snr(snr_db)

    def get_snr(self):
        """Get the current SNR"""
        return self.channel_noise.snr_db

    def forward(self, skip0, skip1, skip2, skip3, bottleneck):
        """
        Process all features - directly apply noise to each feature individually

        Args:
            skip0 (Tensor): First layer features [B, 64, 32, 32]
            skip1 (Tensor): Second layer features [B, 128, 16, 16]
            skip2 (Tensor): Third layer features [B, 256, 8, 8]
            skip3 (Tensor): Fourth layer features [B, 512, 4, 4]
            bottleneck (Tensor): Bottleneck features [B, 512, 4, 4]

        Returns:
            tuple: Denoised features from each layer
        """
        # Adjust channel numbers to reduce transmission overhead
        s0 = self.s0_adjust(skip0)  # [B, skip_channels, 32, 32]
        s1 = self.s1_adjust(skip1)  # [B, skip_channels, 16, 16]
        s2 = self.s2_adjust(skip2)  # [B, skip_channels, 8, 8]
        s3 = self.s3_adjust(skip3)  # [B, skip_channels, 4, 4]
        b = self.b_adjust(bottleneck)  # [B, bottleneck_channels, 4, 4]

        # Directly apply noise to each feature individually (DeepJSCC style)
        noisy_s0 = self.channel_noise(s0)
        noisy_s1 = self.channel_noise(s1)
        noisy_s2 = self.channel_noise(s2)
        noisy_s3 = self.channel_noise(s3)
        noisy_b = self.channel_noise(b)

        # Denoise each noisy feature individually
        denoised_s0 = self.denoiser_s0(noisy_s0)
        denoised_s1 = self.denoiser_s1(noisy_s1)
        denoised_s2 = self.denoiser_s2(noisy_s2)
        denoised_s3 = self.denoiser_s3(noisy_s3)
        denoised_b = self.denoiser_b(noisy_b)

        # Restore channel numbers
        r_s0 = self.s0_restore(denoised_s0)
        r_s1 = self.s1_restore(denoised_s1)
        r_s2 = self.s2_restore(denoised_s2)
        r_s3 = self.s3_restore(denoised_s3)
        r_b = self.b_restore(denoised_b)

        # For API compatibility, we still need to return a concatenated tensor.
        # However, this tensor is no longer used for transmission.
        # We can concatenate the resized features solely for loss calculation.
        s0_down = F.adaptive_avg_pool2d(s0, (4, 4))
        s1_down = F.adaptive_avg_pool2d(s1, (4, 4))
        s2_down = F.adaptive_avg_pool2d(s2, (4, 4))
        concatenated = torch.cat([s0_down, s1_down, s2_down, s3, b], dim=1)

        # For API compatibility, we also need a "noisy" and "denoised" combined version.
        # These are mainly for analysis.
        noisy_s0_down = F.adaptive_avg_pool2d(noisy_s0, (4, 4))
        noisy_s1_down = F.adaptive_avg_pool2d(noisy_s1, (4, 4))
        noisy_s2_down = F.adaptive_avg_pool2d(noisy_s2, (4, 4))
        noisy_feat = torch.cat([noisy_s0_down, noisy_s1_down, noisy_s2_down, noisy_s3, noisy_b], dim=1)

        denoised_s0_down = F.adaptive_avg_pool2d(denoised_s0, (4, 4))
        denoised_s1_down = F.adaptive_avg_pool2d(denoised_s1, (4, 4))
        denoised_s2_down = F.adaptive_avg_pool2d(denoised_s2, (4, 4))
        denoised_feat = torch.cat([denoised_s0_down, denoised_s1_down, denoised_s2_down, denoised_s3, denoised_b],
                                  dim=1)

        return r_s0, r_s1, r_s2, r_s3, r_b, concatenated, noisy_feat, denoised_feat
