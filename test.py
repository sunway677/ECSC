import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ====================
# Channel noise layer
# ====================
class ChannelNoise(nn.Module):
    def __init__(self, snr_db=20):
        super().__init__()
        self.snr_db = snr_db

    def set_snr(self, snr_db):
        self.snr_db = snr_db

    def forward(self, x):
        power = x.pow(2).mean()
        snr = 10 ** (self.snr_db / 10)
        noise_var = power / snr
        noise_std = torch.sqrt(noise_var)
        return x + torch.randn_like(x) * noise_std

# ================================
# DeepJSCC model for CIFAR-10
# bandwidth ratio = 0.3 → latent_size ≈ 0.3 × 32 × 32
# ================================
# -------------------
# Perceptual Loss 网络
# -------------------
class VGGPerceptual(nn.Module):
    def __init__(self, layer='conv3_3'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        self.selected = {
            'conv1_2': 3, 'conv2_2': 8, 'conv3_3': 15, 'conv4_3': 22
        }[layer]
        for p in vgg.parameters(): p.requires_grad_(False)
        self.vgg = vgg

    def forward(self, x, y):
        # 归一化到 ImageNet stats
        mean = torch.tensor([0.485,0.456,0.406],device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225],device=x.device).view(1,3,1,1)
        x = (x - mean) / std
        y = (y - mean) / std
        # 提取到指定层
        for i,layer in enumerate(self.vgg):
            x = layer(x); y = layer(y)
            if i == self.selected: break
        return F.mse_loss(x, y)


# -------------------
# 强化的 DeepJSCC（U-Net 风格 + Residual）
# -------------------
class DeepJSCC(nn.Module):
    def __init__(self, bandwidth_ratio=0.3, snr_db=10):
        super().__init__()
        latent = int(bandwidth_ratio * 32 * 32)

        # Encoder: U-Net 下采样
        self.enc1 = nn.Sequential(nn.Conv2d(3,64,3,1,1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(256,512,4,2,1), nn.BatchNorm2d(512), nn.ReLU())
        self.fc_z = nn.Linear(512*4*4, latent)

        # Channel
        self.channel = ChannelNoise(snr_db)

        # Decoder
        self.fc_d = nn.Linear(latent, 512*4*4)
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(512,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(512,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256,64,4,2,1),  nn.BatchNorm2d(64),  nn.ReLU())
        self.dec1 = nn.Conv2d(128,3,3,1,1)

    # def forward(self, x):
    #     # down
    #     x1 = self.enc1(x)
    #     x2 = self.enc2(x1)
    #     x3 = self.enc3(x2)
    #     x4 = self.enc4(x3)
    #     z  = self.fc_z(x4.view(x4.size(0),-1))
    #
    #     # channel
    #     z_noisy = self.channel(z)
    #
    #     # up with skip
    #     d4 = self.fc_d(z_noisy).view(-1,512,4,4)
    #     u3 = self.dec4(d4)
    #     u3 = torch.cat([u3, x3], dim=1)            # skip x3
    #     u2 = self.dec3(u3)
    #     u2 = torch.cat([u2, x2], dim=1)            # skip x2
    #     u1 = self.dec2(u2)
    #     u1 = torch.cat([u1, x1], dim=1)            # skip x1
    #     out= self.dec1(u1)
    #     return torch.sigmoid(out)

    def forward(self, x):
        # down
        x1 = self.enc1(x)  # 32→32
        x2 = self.enc2(x1)  # 32→16
        x3 = self.enc3(x2)  # 16→8
        x4 = self.enc4(x3)  # 8→4

        # bottleneck + noise
        z = self.fc_z(x4.view(x4.size(0), -1))
        zn = self.channel(z)

        # up
        d4 = self.fc_d(zn).view(-1, 512, 4, 4)
        u3 = self.dec4(d4)
        # **加噪 skip3**
        x3n = self.channel(x3)
        u3 = torch.cat([u3, x3n], dim=1)

        u2 = self.dec3(u3)
        # **加噪 skip2**
        x2n = self.channel(x2)
        u2 = torch.cat([u2, x2n], dim=1)

        u1 = self.dec2(u2)
        # **加噪 skip1**
        x1n = self.channel(x1)
        u1 = torch.cat([u1, x1n], dim=1)

        out = torch.sigmoid(self.dec1(u1))
        return out


# --------------------------------------
# 训练脚本：MSE + Perceptual + 动态 SNR
# --------------------------------------
def main():
    # 数据
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_set  = datasets.CIFAR10('./data', train=False, download=True,
                                 transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepJSCC(bandwidth_ratio=0.3, snr_db=10).to(device)
    perc  = VGGPerceptual('conv3_3').to(device)
    optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = CosineAnnealingLR(optim, T_max=200)

    for epoch in range(1,301):
        model.train()
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            # 随机 SNR 5–20 dB
            model.channel.set_snr(torch.empty(1).uniform_(5,20).item())

            out = model(imgs)
            loss_mse  = F.mse_loss(out, imgs)
            loss_perc = perc(out, imgs)
            loss = loss_mse + 0.05 * loss_perc

            optim.zero_grad(); loss.backward(); optim.step()
        sched.step()

        # 测试 PSNR
        model.eval()
        total_psnr = 0
        with torch.no_grad():
            model.channel.set_snr(20)  # 测试时用高 SNR
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                out  = model(imgs)
                mse  = F.mse_loss(out, imgs, reduction='mean')
                total_psnr += 10 * torch.log10(1/mse).item() * imgs.size(0)
        avg_psnr = total_psnr / len(test_set)
        print(f"Epoch {epoch:03d} → Test PSNR: {avg_psnr:.2f} dB")

if __name__ == "__main__":
    freeze_support()
    main()