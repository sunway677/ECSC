import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import os
import shutil
from losses.energy import ContrastiveLossWithFreeEnergy
from utils.checkpoint import save_checkpoint, load_checkpoint


def skip_mse_loss(original_skips, processed_skips, weights=None):
    """
    计算skip层特征的MSE损失

    Args:
        original_skips: 原始skip特征字典
        processed_skips: 处理后的skip特征字典
        weights: 各层的权重字典，默认为None（等权重）

    Returns:
        加权平均的skip MSE损失
    """
    if weights is None:
        # 默认权重，越靠近输出层权重越大
        weights = {
            'skip0': 0.4,  # 最靠近输出层，最重要
            'skip1': 0.3,
            'skip2': 0.2,
            'skip3': 0.1  # 最靠近输入层，最不重要
        }

    total_loss = 0.0
    for key in original_skips:
        if key in processed_skips:
            # 计算每个skip层的MSE
            layer_mse = F.mse_loss(original_skips[key], processed_skips[key])
            # 加权平均
            total_loss += weights[key] * layer_mse

    return total_loss


def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images.

    Args:
        img1, img2: Tensors of shape [B, C, H, W] with values in range [0, 1]

    Returns:
        float: Average PSNR value across the batch
    """
    # MSE loss
    mse = F.mse_loss(img1, img2, reduction='none').mean(dim=[1, 2, 3])

    # Calculate PSNR for each image in the batch
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

    # Return average PSNR
    return psnr.mean().item()


def train_step_combined(model, feature_extractor, criterion, images, device,
                        optimizer_model, optimizer_energy, use_energy_loss=True,
                        lambda_skip=0.5, use_energy_weighted_mse=False, alpha=2.0):
    """
    合并训练步骤，当能量损失开启时同时训练EnergyNet和主网络，
    当能量损失关闭时只训练主网络。

    Args:
        model: Autoencoder model
        feature_extractor: CLIP feature extractor
        criterion: Loss criterion
        images: Batch of input images
        device: Device to use
        optimizer_model: Optimizer for main model
        optimizer_energy: Optimizer for EnergyNet
        use_energy_loss: Whether to use energy loss with CLIP features
        lambda_skip: Weight for skip loss
        use_energy_weighted_mse: Whether to use energy-weighted MSE loss
        alpha: Parameter for controlling the weight range in energy-weighted MSE

    Returns:
        dict: Dictionary containing loss values
        float: PSNR value
    """
    # 获取模型输出
    outputs, compressed_features = model(images)

    # 准备计算Skip损失所需的原始特征
    original_skips = compressed_features['original_skips']
    processed_skips = compressed_features['processed_skips']

    # 计算Skip MSE损失
    skip_loss = skip_mse_loss(original_skips, processed_skips)

    # 为compress_criterion准备简化的特征字典
    compress_features_dict = {
        'bottleneck': compressed_features['bottleneck'],
        'skip0': compressed_features['skip0'],
        'skip1': compressed_features['skip1'],
        'skip2': compressed_features['skip2'],
        'skip3': compressed_features['skip3'],
        'concatenated': compressed_features['concatenated']
    }

    # 分支处理：使用能量损失与否
    if use_energy_loss and feature_extractor is not None:
        # 能量损失模式：同时训练EnergyNet和主网络

        # 提取CLIP特征
        f_orig = feature_extractor(images)
        f_recon = feature_extractor(outputs)

        # 计算能量损失 - 使用criterion中的forward_energy_margin方法
        energy_loss = criterion.forward_energy_margin(f_orig, f_recon)

        # 基础重建损失计算
        if use_energy_weighted_mse:
            # 使用基于能量加权的MSE损失
            with torch.no_grad():
                # 使用criterion中的compute_energy方法确保一致性
                E = criterion.compute_energy(f_orig, f_recon)
                e_diag = torch.diag(E)

            # 计算per-sample MSE
            per_mse = F.mse_loss(outputs, images, reduction='none').view(images.size(0), -1).mean(1)

            # 基于能量生成权重，映射到[0.8, 1.2]范围
            # 先标准化到[0,1]区间，再映射到[0.8,1.2]
            w = torch.sigmoid(alpha * (e_diag - e_diag.mean())) * 0.4 + 0.8

            # 加权重建损失
            recon_loss = (w * per_mse).mean()
        else:
            # 使用常规MSE损失
            recon_loss = nn.MSELoss()(outputs, images)

        # 计算压缩损失
        compress_loss = criterion.compress_criterion(compress_features_dict)

        # 计算总损失
        total_loss = (recon_loss +
                      criterion.lambda_energy * energy_loss +
                      criterion.lambda_compress * compress_loss +
                      lambda_skip * skip_loss)

        # 清空所有优化器的梯度
        optimizer_model.zero_grad()
        optimizer_energy.zero_grad()

        # 反向传播计算梯度
        total_loss.backward()

        # 更新所有网络的参数
        optimizer_model.step()
        optimizer_energy.step()

        # 更新损失字典
        losses = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'energy': energy_loss.item(),
            'compress': compress_loss.item(),
            'skip': skip_loss.item()
        }

    else:
        # 无能量损失模式：只训练主网络

        # 重建损失
        recon_loss = nn.MSELoss()(outputs, images)

        # 压缩损失
        if hasattr(criterion, 'compress_criterion'):
            compress_loss = criterion.compress_criterion(compress_features_dict)
        else:
            compress_loss = torch.tensor(0.0, device=device)

        # 总损失
        total_loss = recon_loss + criterion.lambda_compress * compress_loss + lambda_skip * skip_loss

        # 只清空主网络优化器的梯度
        optimizer_model.zero_grad()

        # 反向传播计算梯度
        total_loss.backward()

        # 只更新主网络参数
        optimizer_model.step()

        # 更新损失字典
        losses = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'energy': 0.0,  # 无能量损失
            'compress': compress_loss.item() if torch.is_tensor(compress_loss) else compress_loss,
            'skip': skip_loss.item()
        }

    # 计算PSNR
    psnr_value = calculate_psnr(outputs, images)

    return losses, psnr_value


def train_autoencoder_with_energy(
        model,
        feature_extractor,
        train_loader,
        test_loader,
        device,
        num_epochs=30,
        learning_rate=1e-3,
        lambda_energy=0.1,
        lambda_compress=0.01,
        lambda_skip=0.5,
        checkpoint_dir="checkpoints",
        resume=False,
        use_energy_loss=True,
        use_energy_weighted_mse=False,
        alpha=2.0
):
    """
    Main training function for the energy-based semantic autoencoder.
    使用合并的训练步骤:
    - 当能量损失开启时同时训练所有网络组件
    - 当能量损失关闭时只训练主网络

    Args:
        model: Main autoencoder model
        feature_extractor: CLIP feature extractor (can be None if use_energy_loss is False)
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to use
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizers
        lambda_energy: Weight for energy loss
        lambda_compress: Weight for compression loss
        lambda_skip: Weight for skip MSE loss
        checkpoint_dir: Directory for saving checkpoints
        resume: Whether to resume training from checkpoint
        use_energy_loss: Whether to use energy loss with CLIP features
        use_energy_weighted_mse: Whether to use energy-weighted MSE loss
        alpha: Parameter for controlling the weight range in energy-weighted MSE

    Returns:
        model: Trained model
    """
    # Initialize optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize loss criterion
    criterion = ContrastiveLossWithFreeEnergy(
        tau=1.0,
        lambda_compress=lambda_compress,
        lambda_energy=lambda_energy
    )

    # Initialize energy optimizer (only if using energy loss)
    optimizer_energy = None
    if use_energy_loss and feature_extractor is not None:
        # Initialize EnergyNet with dummy data
        dummy_images, _ = next(iter(train_loader))
        dummy_images = dummy_images.to(device)
        with torch.no_grad():
            outputs, _ = model(dummy_images)
            f_orig = feature_extractor(dummy_images)
            f_recon = feature_extractor(outputs)
            _ = criterion.compute_energy(f_orig, f_recon)

        # Initialize energy optimizer
        optimizer_energy = optim.Adam(
            criterion.energy_net.parameters(),
            lr=learning_rate
        )
    else:
        # 如果不使用能量网络，创建一个虚拟的优化器以保持API一致性
        optimizer_energy = optim.Adam([nn.Parameter(torch.zeros(1, device=device))], lr=learning_rate)

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume:
        latest_checkpoint = f"{checkpoint_dir}/latest_checkpoint.pth"
        start_epoch, _ = load_checkpoint(model, optimizer_model, latest_checkpoint)
        start_epoch += 1

    # Print compression statistics
    compressed_info = model.get_compressed_size()
    orig_size = 3 * 32 * 32
    print("\nCompression Statistics:")
    print(f"Original size: {orig_size} elements")
    print(f"Compressed size: {compressed_info['total']} elements")
    print(f"Compression ratio: {orig_size / compressed_info['total']:.2f}x")
    print(f"Details:")
    for name, size in compressed_info.items():
        if name != 'total':
            print(f"  {name}: {size} elements")
    print()

    # Print energy calculation mode
    if use_energy_loss and feature_extractor is not None:
        print("Energy calculation: Using CLIP features for energy loss")
        print("Training mode: Combined (EnergyNet and main network trained together)")
        if use_energy_weighted_mse:
            print(f"Energy-weighted MSE: ENABLED - alpha={alpha:.2f}")
        else:
            print("Energy-weighted MSE: DISABLED")
    else:
        print("Energy calculation: DISABLED - using only reconstruction and compression loss")
        print("Training mode: Main network only (EnergyNet is not updated)")
        print("Energy-weighted MSE: DISABLED (requires use_energy_loss=True)")

    # 打印Skip MSE损失信息
    print(f"Skip MSE Loss: ENABLED - weight: {lambda_skip:.2f}")
    print("Skip weights: skip0=0.4, skip1=0.3, skip2=0.2, skip3=0.1")

    # Training loop
    best_psnr = 0.0  # 跟踪最佳PSNR
    best_epoch = 0  # 记录最佳PSNR对应的epoch

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_losses = {
            'total': 0, 'recon': 0, 'energy': 0,
            'compress': 0, 'skip': 0
        }
        running_psnr = 0  # 追踪PSNR

        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 使用合并的训练步骤
        train_pbar = tqdm(train_loader, desc='Training Networks', leave=False)
        for i, (images, _) in enumerate(train_pbar):
            images = images.to(device)

            # 使用合并的训练步骤
            losses, psnr_value = train_step_combined(
                model, feature_extractor, criterion,
                images, device, optimizer_model, optimizer_energy,
                use_energy_loss, lambda_skip, use_energy_weighted_mse, alpha
            )

            # Update running losses and PSNR
            for k, v in losses.items():
                running_losses[k] += v
            running_psnr += psnr_value

            # Update progress bar with losses and PSNR
            postfix_dict = {
                k: f'{v / (i + 1):.4f}'
                for k, v in running_losses.items()
            }
            postfix_dict['psnr'] = f'{running_psnr / (i + 1):.2f}'
            train_pbar.set_postfix(postfix_dict)

        # Calculate average losses and PSNR
        avg_losses = {k: v / len(train_loader) for k, v in running_losses.items()}
        avg_psnr = running_psnr / len(train_loader)

        # 更新最佳PSNR
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch + 1

            # 保存当前checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer_model,
                epoch=epoch,
                loss=avg_losses['total'],
                checkpoint_dir=checkpoint_dir
            )

            # 将最新保存的checkpoint复制为best_psnr_checkpoint.pth
            latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
            best_checkpoint = os.path.join(checkpoint_dir, "best_psnr_checkpoint.pth")
            shutil.copyfile(latest_checkpoint, best_checkpoint)
            print(f"Best model saved as: {best_checkpoint} (PSNR: {best_psnr:.2f} dB)")

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print("Network Metrics:")
        for name, value in avg_losses.items():
            print(f"{name.capitalize()}: {value:.4f}")
        print(f"PSNR: {avg_psnr:.2f} dB")
        print(f"Best PSNR: {best_psnr:.2f} dB (Epoch {best_epoch})")

        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer_model,
            epoch=epoch,
            loss=avg_losses['total'],
            checkpoint_dir=checkpoint_dir
        )

    print(f"\nTraining completed. Best PSNR: {best_psnr:.2f} dB achieved at Epoch {best_epoch}")
    return model