import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import os
import shutil
from losses.energy import ContrastiveLossWithFreeEnergy # Assuming this is the correct path for losses
from utils.checkpoint import save_checkpoint, load_checkpoint


def skip_mse_loss(original_skips, processed_skips, weights=None):
    """
    Calculate MSE loss for skip layer features.

    Args:
        original_skips: Dictionary of original skip features.
        processed_skips: Dictionary of processed skip features.
        weights: Dictionary of weights for each layer, defaults to None (equal weights).

    Returns:
        Weighted average skip MSE loss.
    """
    if weights is None:
        # Default weights, layers closer to the output have higher weights
        weights = {
            'skip0': 0.4,  # Closest to the output layer, most important
            'skip1': 0.3,
            'skip2': 0.2,
            'skip3': 0.1   # Closest to the input layer, least important
        }

    total_loss = 0.0
    num_valid_skips = 0
    for key in original_skips:
        if key in processed_skips and key in weights: # Ensure key exists in all dicts
            # Calculate MSE for each skip layer
            layer_mse = F.mse_loss(original_skips[key], processed_skips[key])
            # Weighted average
            total_loss += weights[key] * layer_mse
            num_valid_skips += weights[key] # Sum of weights for normalization if needed, or just sum weighted losses

    # If you intend to average by the number of skip connections or sum of weights:
    # if num_valid_skips > 0:
    #     return total_loss / num_valid_skips # Or simply return total_loss if weights sum to 1 or are relative importance factors
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

    # Prevent log10 of zero or negative if MSE is zero
    # Add a small epsilon to mse if it can be zero.
    # If images are identical, PSNR is infinite, conventionally set to a high value or handled.
    # For this function, if mse is 0, sqrt(mse) is 0, 1.0/0 is inf, log10(inf) is inf.
    # torch.log10(1.0 / (torch.sqrt(mse) + 1e-8)) # Add epsilon for stability if mse can be 0

    # Calculate PSNR for each image in the batch
    psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-10)) # Add small epsilon to avoid division by zero or log of zero

    # Return average PSNR
    return psnr_val.mean().item()


def train_step_combined(model, feature_extractor, criterion, images, device,
                        optimizer_model, optimizer_energy, use_energy_loss=True,
                        lambda_skip=0.5, use_energy_weighted_mse=False, alpha=2.0):
    """
    Combined training step: trains EnergyNet and main network simultaneously when energy loss is enabled,
    otherwise trains only the main network.

    Args:
        model: Autoencoder model
        feature_extractor: CLIP feature extractor
        criterion: Loss criterion (instance of ContrastiveLossWithFreeEnergy)
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
    # Get model outputs
    outputs, compressed_features_full = model(images) # Renamed to avoid conflict

    # Prepare original features needed for calculating Skip loss
    original_skips = compressed_features_full['original_skips']
    processed_skips = compressed_features_full['processed_skips']

    # Calculate Skip MSE loss
    skip_loss = skip_mse_loss(original_skips, processed_skips)

    # Prepare a simplified feature dictionary for compress_criterion
    # Ensure all keys are present in compressed_features_full before accessing
    compress_features_dict = {
        key: compressed_features_full[key]
        for key in ['bottleneck', 'skip0', 'skip1', 'skip2', 'skip3', 'concatenated']
        if key in compressed_features_full
    }


    # Branch processing: whether to use energy loss or not
    if use_energy_loss and feature_extractor is not None and criterion.energy_net is not None:
        # Energy loss mode: train EnergyNet and main network simultaneously

        # Extract CLIP features
        f_orig = feature_extractor(images)
        f_recon = feature_extractor(outputs)

        # Calculate energy loss - using forward_energy_margin method from criterion
        energy_loss = criterion.forward_energy_margin(f_orig, f_recon)

        # Basic reconstruction loss calculation
        if use_energy_weighted_mse:
            # Use energy-weighted MSE loss
            with torch.no_grad():
                # Use compute_energy method from criterion for consistency
                E = criterion.compute_energy(f_orig, f_recon) # This requires energy_net to be initialized
                e_diag = torch.diag(E)

            # Calculate per-sample MSE
            per_mse = F.mse_loss(outputs, images, reduction='none').view(images.size(0), -1).mean(1)

            # Generate weights based on energy, mapped to [0.8, 1.2] range
            # First normalize to [0,1] range, then map to [0.8,1.2]
            # The original sigmoid mapping: torch.sigmoid(alpha * (e_diag - e_diag.mean())) * 0.4 + 0.8
            # Ensure e_diag is not all same values to avoid NaN if e_diag.mean() is subtracted from itself
            if e_diag.numel() > 1 and not torch.allclose(e_diag, e_diag.mean()):
                 w = torch.sigmoid(alpha * (e_diag - e_diag.mean()) / (e_diag.std() + 1e-8)) * 0.4 + 0.8 # Normalize by std
            else: # Handle single element or all same elements case
                 w = torch.ones_like(e_diag) * 1.0 # Neutral weight


            # Weighted reconstruction loss
            recon_loss = (w * per_mse).mean()
        else:
            # Use regular MSE loss
            recon_loss = nn.MSELoss()(outputs, images)

        # Calculate compression loss
        compress_loss = criterion.compress_criterion(compress_features_dict)

        # Calculate total loss
        total_loss = (recon_loss +
                      criterion.lambda_energy * energy_loss +
                      criterion.lambda_compress * compress_loss +
                      lambda_skip * skip_loss)

        # Clear gradients of all optimizers
        optimizer_model.zero_grad()
        if optimizer_energy: # Check if optimizer_energy is not None
             optimizer_energy.zero_grad()


        # Backpropagate to calculate gradients
        total_loss.backward()

        # Update parameters of all networks
        optimizer_model.step()
        if optimizer_energy: # Check if optimizer_energy is not None
            optimizer_energy.step()


        # Update loss dictionary
        losses = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'energy': energy_loss.item(),
            'compress': compress_loss.item(),
            'skip': skip_loss.item()
        }

    else:
        # No energy loss mode: train only the main network

        # Reconstruction loss
        recon_loss = nn.MSELoss()(outputs, images)

        # Compression loss
        if hasattr(criterion, 'compress_criterion') and callable(criterion.compress_criterion):
            compress_loss = criterion.compress_criterion(compress_features_dict)
        else:
            compress_loss = torch.tensor(0.0, device=device) # Ensure it's a tensor

        # Total loss
        lambda_compress_val = criterion.lambda_compress if hasattr(criterion, 'lambda_compress') else 0.0
        total_loss = recon_loss + lambda_compress_val * compress_loss + lambda_skip * skip_loss


        # Clear gradients of only the main network optimizer
        optimizer_model.zero_grad()

        # Backpropagate to calculate gradients
        total_loss.backward()

        # Update only main network parameters
        optimizer_model.step()

        # Update loss dictionary
        losses = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'energy': 0.0,  # No energy loss
            'compress': compress_loss.item() if torch.is_tensor(compress_loss) else compress_loss, # Handle if not tensor
            'skip': skip_loss.item()
        }

    # Calculate PSNR
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
    Uses a combined training step:
    - When energy loss is enabled, all network components are trained simultaneously.
    - When energy loss is disabled, only the main network is trained.

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
        tau=1.0, # Assuming tau is a parameter of ContrastiveLossWithFreeEnergy
        lambda_compress=lambda_compress,
        lambda_energy=lambda_energy
    )

    # Initialize energy optimizer (only if using energy loss)
    optimizer_energy = None
    if use_energy_loss and feature_extractor is not None:
        # Initialize EnergyNet within the criterion by calling compute_energy once.
        # This ensures energy_net is created before its parameters are accessed by the optimizer.
        try:
            dummy_images, _ = next(iter(train_loader))
            dummy_images = dummy_images.to(device)
            with torch.no_grad():
                outputs_dummy, _ = model(dummy_images)
                f_orig_dummy = feature_extractor(dummy_images)
                f_recon_dummy = feature_extractor(outputs_dummy)
                # This call will initialize self.energy_net inside ContrastiveLossWithFreeEnergy
                _ = criterion.compute_energy(f_orig_dummy, f_recon_dummy)

            if criterion.energy_net is not None:
                 optimizer_energy = optim.Adam(
                    criterion.energy_net.parameters(),
                    lr=learning_rate
                )
            else:
                print("Warning: criterion.energy_net is None after initialization attempt. EnergyNet won't be trained.")
                use_energy_loss = False # Disable energy loss if EnergyNet is not available
        except StopIteration:
            print("Error: Training data loader is empty. Cannot initialize EnergyNet.")
            return model # Or handle error appropriately


    # Resume from checkpoint if specified
    start_epoch = 0
    if resume:
        latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(latest_checkpoint):
            # Pass optimizer_energy if it's not None
            opt_energy_to_load = optimizer_energy if use_energy_loss and optimizer_energy is not None else None
            start_epoch, _ = load_checkpoint(model, optimizer_model, latest_checkpoint, optimizer_energy=opt_energy_to_load)
            start_epoch += 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {latest_checkpoint}. Starting from scratch.")


    # Print compression statistics
    if hasattr(model, 'get_compressed_size'):
        compressed_info = model.get_compressed_size()
        orig_size = 3 * 32 * 32 # Assuming 3 channels, 32x32 image size
        print("\nCompression Statistics:")
        print(f"Original size: {orig_size} elements")
        if 'total' in compressed_info and compressed_info['total'] > 0:
            print(f"Compressed size: {compressed_info['total']} elements")
            print(f"Compression ratio: {orig_size / compressed_info['total']:.2f}x")
            print(f"Details:")
            for name, size in compressed_info.items():
                if name != 'total':
                    print(f"  {name}: {size} elements")
        else:
            print("Compressed size information not available or invalid.")
        print()
    else:
        print("Model does not have 'get_compressed_size' method. Skipping compression stats.")


    # Print energy calculation mode
    if use_energy_loss and feature_extractor is not None and criterion.energy_net is not None:
        print("Energy calculation: Using CLIP features for energy loss")
        print("Training mode: Combined (EnergyNet and main network trained together)")
        if use_energy_weighted_mse:
            print(f"Energy-weighted MSE: ENABLED - alpha={alpha:.2f}")
        else:
            print("Energy-weighted MSE: DISABLED")
    else:
        print("Energy calculation: DISABLED - using only reconstruction and compression loss")
        print("Training mode: Main network only (EnergyNet is not updated or not available)")
        print("Energy-weighted MSE: DISABLED (requires use_energy_loss=True and EnergyNet)")

    # Print Skip MSE loss information
    print(f"Skip MSE Loss: ENABLED - weight: {lambda_skip:.2f}")
    print("Skip weights: skip0=0.4, skip1=0.3, skip2=0.2, skip3=0.1 (default)")

    # Training loop
    best_psnr = 0.0  # Track the best PSNR
    best_epoch = 0   # Record the epoch corresponding to the best PSNR

    for epoch in range(start_epoch, num_epochs):
        model.train()
        if use_energy_loss and criterion.energy_net is not None: # Set EnergyNet to train mode
            criterion.energy_net.train()

        running_losses = {
            'total': 0, 'recon': 0, 'energy': 0,
            'compress': 0, 'skip': 0
        }
        running_psnr = 0  # Track PSNR

        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Use the combined training step
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training', leave=True) # Changed leave to True
        for i, (images, _) in enumerate(train_pbar):
            images = images.to(device)

            # Use the combined training step
            losses, psnr_value = train_step_combined(
                model, feature_extractor, criterion,
                images, device, optimizer_model, optimizer_energy,
                use_energy_loss and criterion.energy_net is not None, # Pass updated use_energy_loss status
                lambda_skip, use_energy_weighted_mse, alpha
            )

            # Update running losses and PSNR
            for k, v_loss in losses.items(): # Renamed v to v_loss
                running_losses[k] += v_loss
            running_psnr += psnr_value

            # Update progress bar with losses and PSNR
            postfix_dict = {
                k_loss: f'{v_loss_val / (i + 1):.4f}' # Renamed k,v
                for k_loss, v_loss_val in running_losses.items()
            }
            postfix_dict['psnr'] = f'{running_psnr / (i + 1):.2f}'
            train_pbar.set_postfix(postfix_dict)

        # Calculate average losses and PSNR
        avg_losses = {k: v / len(train_loader) for k, v in running_losses.items()}
        avg_psnr = running_psnr / len(train_loader)

        # Update best PSNR
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch + 1

            # Save current checkpoint as the best one so far
            save_checkpoint(
                model=model,
                optimizer=optimizer_model,
                epoch=epoch,
                loss=avg_losses['total'],
                checkpoint_dir=checkpoint_dir,
                filename="best_psnr_checkpoint.pth", # Save directly as best
                optimizer_energy=optimizer_energy if use_energy_loss and criterion.energy_net is not None else None
            )
            print(f"Best model saved: best_psnr_checkpoint.pth (PSNR: {best_psnr:.2f} dB at Epoch {best_epoch})")


        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print("Network Metrics:")
        for name, value in avg_losses.items():
            print(f"{name.capitalize()}: {value:.4f}")
        print(f"PSNR: {avg_psnr:.2f} dB")
        print(f"Best PSNR: {best_psnr:.2f} dB (Epoch {best_epoch})")

        # Save latest checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer_model,
            epoch=epoch,
            loss=avg_losses['total'],
            checkpoint_dir=checkpoint_dir,
            filename="latest_checkpoint.pth", # Explicitly name latest checkpoint
            optimizer_energy=optimizer_energy if use_energy_loss and criterion.energy_net is not None else None
        )
        print(f"Latest checkpoint saved: latest_checkpoint.pth")


    print(f"\nTraining completed. Best PSNR: {best_psnr:.2f} dB achieved at Epoch {best_epoch}")
    return model
