import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import argparse
from models.autoencoder import ResAutoencoder
from models.feature_extractor import CLIPFeatureExtractor
from data.dataloader import get_dataloaders
from train import train_autoencoder_with_energy
from utils.visualization import test_and_visualize, evaluate_full_dataset
from utils.checkpoint import load_checkpoint

# CLIP model name mapping
CLIP_MODELS = {
    'ViT-B/32': 'openai/clip-vit-base-patch32',
    'ViT-B/16': 'openai/clip-vit-base-patch16',
    'ViT-L/14': 'openai/clip-vit-large-patch14'
}


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train/Test Semantic Communication Autoencoder')
    # Basic settings
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='train model or test from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='specific checkpoint file to load for testing')
    parser.add_argument('--resume', action='store_true',
                        help='resume training from latest checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='directory to save checkpoints')

    # Channel settings
    parser.add_argument('--snr_db', type=float, default=15,
                        help='Signal-to-Noise Ratio in dB for training')
    parser.add_argument('--test_snr_values', type=float, nargs='+', default=None,
                        help='List of SNR values to test (e.g., --test_snr_values 0 5 10 15 20 25)')

    # Model settings
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        choices=CLIP_MODELS.keys(),
                        help='CLIP model variant to use')

    # Energy loss settings
    parser.add_argument('--use_energy_loss', action='store_true', default=True,
                        help='use energy loss with CLIP features (default: False)')

    # 能量加权MSE参数
    parser.add_argument('--use_energy_weighted_mse', action='store_true', default=True,
                        help='use energy-weighted MSE loss (requires --use_energy_loss, default: False)')
    parser.add_argument('--energy_weight_alpha', type=float, default=2.0,
                        help='alpha parameter for energy weight mapping in range [0.8, 1.2] (default: 2.0)')

    # Loss weights
    parser.add_argument('--lambda_energy', type=float, default=0.003348570462810393,
                        help='weight for energy loss (default: 0.003348570462810393)')
    parser.add_argument('--lambda_compress', type=float, default=0.001705527596213061,
                        help='weight for compression loss (default: 0.001705527596213061)')
    parser.add_argument('--lambda_skip', type=float, default=0.5,
                        help='weight for skip loss (default: 0.5)')

    # Energy temperature parameter
    parser.add_argument('--tau', type=float, default=1.0,
                        help='temperature parameter for energy loss (default: 1.0)')

    # Testing settings
    parser.add_argument('--num_test_images', type=int, default=5,
                        help='number of images to test and visualize')
    parser.add_argument('--eval_only', action='store_true',
                        help='only run evaluation on test set')
    args = parser.parse_args()

    # 如果用户启用了能量加权MSE但未启用能量损失，则发出警告
    if args.use_energy_weighted_mse and not args.use_energy_loss:
        print("WARNING: Energy-weighted MSE requires energy loss to be enabled.")
        print("Setting --use_energy_weighted_mse=False since --use_energy_loss is not enabled.")
        args.use_energy_weighted_mse = False

    # Training configuration
    config = {
        'num_epochs': 500,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'lambda_energy': args.lambda_energy,
        'lambda_compress': args.lambda_compress,
        'lambda_skip': args.lambda_skip,
        'tau': args.tau,
        'snr_db': args.snr_db,
        'bottleneck_dim': 24,
        'skip_channels': 10,
        'use_energy_loss': args.use_energy_loss,  # 修正：使用一致的命名
        'use_energy_weighted_mse': args.use_energy_weighted_mse,
        'energy_weight_alpha': args.energy_weight_alpha
    }

    # Set device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize semantic autoencoder model
    model = ResAutoencoder(
        snr_db=config['snr_db'],
        bottleneck_dim=config['bottleneck_dim'],
        skip_channels=config['skip_channels']
    ).to(device)

    # Initialize CLIP feature extractor only if energy loss is enabled
    feature_extractor = None
    if args.use_energy_loss:
        feature_extractor = CLIPFeatureExtractor(
            device,
            model_name=CLIP_MODELS[args.clip_model]
        )
        feature_extractor.eval()
        print("Energy loss: ENABLED - Using CLIP features")

        # 打印能量加权MSE状态
        if args.use_energy_weighted_mse:
            print(f"Energy-weighted MSE: ENABLED - alpha={args.energy_weight_alpha}")
        else:
            print("Energy-weighted MSE: DISABLED")
    else:
        print("Energy loss: DISABLED - Using only reconstruction and compression loss")
        print("Energy-weighted MSE: DISABLED (requires energy loss to be enabled)")

    # Print model statistics
    print("\nModel Statistics:")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Print compression information
    compressed_info = model.get_compressed_size()
    orig_size = 3 * 32 * 32
    print(f"Original image size: {orig_size:,} elements")
    print(f"Compressed size: {compressed_info['total']:,} elements")
    print(f"Compression ratio: {orig_size / compressed_info['total']:.2f}x")
    print("Compressed feature sizes:")
    for name, size in compressed_info.items():
        if name != 'total':
            print(f"  {name}: {size:,} elements")

    # Initialize data loaders
    train_loader, test_loader = get_dataloaders(batch_size=config['batch_size'])

    if args.mode == 'test':
        # Load checkpoint for testing
        checkpoint_path = args.checkpoint_path or f"{args.checkpoint_dir}/latest_checkpoint.pth"
        print(f"\nLoading checkpoint from: {checkpoint_path}")

        dummy_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        epoch, _ = load_checkpoint(model, dummy_optimizer, checkpoint_path)
        print(f"Successfully loaded checkpoint from epoch {epoch}")

        if args.eval_only:
            # 在所有测试SNR值上进行完整评估
            if args.test_snr_values is not None:
                for snr in args.test_snr_values:
                    model.set_channel_snr(snr)
                    print(f"\nTesting with SNR = {snr} dB")
                    metrics = evaluate_full_dataset(model, test_loader, device)
                    print(f"SNR {snr}dB Results:")
                    print(f"Average PSNR: {metrics['PSNR']:.2f} dB")
                    print(f"Average SSIM: {metrics['SSIM']:.4f}")
            else:
                # 使用默认SNR进行评估
                print(f"\nTesting with default SNR = {model.get_current_snr()} dB")
                metrics = evaluate_full_dataset(model, test_loader, device)
                print(f"Results:")
                print(f"Average PSNR: {metrics['PSNR']:.2f} dB")
                print(f"Average SSIM: {metrics['SSIM']:.4f}")
        else:
            # Test model with specified SNR values
            if args.test_snr_values is not None:
                for snr in args.test_snr_values:
                    model.set_channel_snr(snr)
                    model.eval()
                    print(f"\nTesting with SNR = {snr} dB")
                    test_and_visualize(model, test_loader, device, num_images=args.num_test_images)
            else:
                # Test with default SNR
                model.eval()
                print(f"\nTesting with default SNR = {model.get_current_snr()} dB")
                test_and_visualize(model, test_loader, device, num_images=args.num_test_images)

    else:  # Training mode
        # Display training configuration
        print("\nTraining Configuration:")
        for k, v in config.items():
            print(f"{k}: {v}")

        # Train the model
        print("\nStarting training...")
        model = train_autoencoder_with_energy(
            model=model,
            feature_extractor=feature_extractor,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            lambda_energy=config['lambda_energy'],
            lambda_compress=config['lambda_compress'],
            lambda_skip=config['lambda_skip'],
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            use_energy_loss=config['use_energy_loss'],  # 修正：使用一致的命名
            use_energy_weighted_mse=config['use_energy_weighted_mse'],
            alpha=config['energy_weight_alpha']
        )

        # Evaluate trained model
        print("\nTraining completed. Running final evaluation...")
        if args.test_snr_values is not None:
            for snr in args.test_snr_values:
                model.set_channel_snr(snr)
                model.eval()
                print(f"\nFinal testing with SNR = {snr} dB")
                test_and_visualize(model, test_loader, device, num_images=args.num_test_images)
        else:
            model.eval()
            test_and_visualize(model, test_loader, device, num_images=args.num_test_images)


if __name__ == "__main__":
    main()