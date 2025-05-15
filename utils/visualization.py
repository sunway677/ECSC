import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
from datetime import datetime
import json
import os
from tqdm import tqdm  # 用于显示进度条


def calculate_metrics(original, reconstructed):
    """
    Calculate various image quality metrics.

    Args:
        original: The original image as a numpy array.
        reconstructed: The reconstructed image as a numpy array.

    Returns:
        A dictionary containing PSNR, SSIM, MSE, NMSE, and per-channel metrics.
    """
    # Ensure pixel values are within [0, 1]
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)

    # Compute basic metrics
    psnr_value = psnr(original, reconstructed, data_range=1.0)
    ssim_value = ssim(original, reconstructed, data_range=1.0, channel_axis=2)
    mse = np.mean((original - reconstructed) ** 2)
    nmse = mse / np.mean(original ** 2)

    # Calculate metrics for each color channel
    channel_metrics = {}
    for i, channel_name in enumerate(['R', 'G', 'B']):
        orig_channel = original[:, :, i]
        recon_channel = reconstructed[:, :, i]
        channel_mse = np.mean((orig_channel - recon_channel) ** 2)
        channel_psnr = psnr(orig_channel, recon_channel, data_range=1.0)
        channel_ssim = ssim(orig_channel, recon_channel, data_range=1.0)

        channel_metrics[channel_name] = {
            'MSE': channel_mse,
            'PSNR': channel_psnr,
            'SSIM': channel_ssim
        }

    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'MSE': mse,
        'NMSE': nmse,
        'channels': channel_metrics
    }


def save_metrics_to_file(metrics, compressed_info, save_dir='results'):
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics: Dictionary containing evaluation metrics.
        compressed_info: Dictionary containing compression information.
        save_dir: Directory to save the results.
    """
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate a timestamped filename for the metrics file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f'metrics_{timestamp}.json')

    result_dict = {
        'timestamp': timestamp,
        'metrics': metrics,
        'compression_info': compressed_info,
        'test_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Write the metrics and compression info to a JSON file
    with open(filename, 'w') as f:
        json.dump(result_dict, f, indent=4)

    print(f"\nMetrics saved to: {filename}")


def print_detailed_metrics(metrics, compressed_info):
    """
    Print detailed evaluation metrics.

    Args:
        metrics: Dictionary containing evaluation metrics.
        compressed_info: Dictionary containing compression information.
    """
    # Original image size (for example, for a 32x32 RGB image)
    orig_size = 3 * 32 * 32

    print("\n" + "=" * 70)
    print("SEMANTIC COMMUNICATION EVALUATION RESULTS")
    print("=" * 70)

    # Compression statistics
    print("\nCOMPRESSION STATISTICS:")
    print("-" * 50)
    print(f"Original size: {orig_size:,} elements")
    print(f"Compressed size: {compressed_info['total']:,} elements")
    print(f"Compression ratio: {orig_size / compressed_info['total']:.2f}x")
    print(f"Compression rate: {(1 - compressed_info['total'] / orig_size) * 100:.2f}%")

    print("\nFeature size breakdown:")
    for name, size in compressed_info.items():
        if name != 'total':
            print(f"  {name:10s}: {size:5.0f} elements ({size / orig_size * 100:6.2f}%)")

    # Quality metrics
    print("\nQUALITY METRICS:")
    print("-" * 50)
    print(f"Overall PSNR:    {metrics['PSNR']:.2f} dB")
    print(f"Overall SSIM:    {metrics['SSIM']:.4f}")
    print(f"Overall MSE:     {metrics['MSE']:.6f}")
    print(f"Overall NMSE:    {metrics['NMSE']:.6f}")

    # Per-channel metrics
    print("\nPER-CHANNEL METRICS:")
    print("-" * 50)
    for channel, values in metrics['channels'].items():
        print(f"\nChannel {channel}:")
        print(f"  PSNR: {values['PSNR']:.2f} dB")
        print(f"  SSIM: {values['SSIM']:.4f}")
        print(f"  MSE:  {values['MSE']:.6f}")

    print("\n" + "=" * 70)


def evaluate_full_dataset(model, data_loader, device):
    """
    在完整数据集上评估模型。

    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备

    Returns:
        dict: 包含平均PSNR和SSIM以及最佳PSNR和SSIM的字典
    """
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    # 跟踪最佳值
    best_psnr = 0.0
    best_ssim = 0.0

    print("\nEvaluating full dataset...")
    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            batch_size = images.size(0)

            # 前向传播
            outputs, _ = model(images)

            # 转换为numpy计算指标
            for i in range(batch_size):
                img_orig = images[i].cpu().numpy().transpose(1, 2, 0)
                img_recon = outputs[i].cpu().numpy().transpose(1, 2, 0)

                metrics = calculate_metrics(img_orig, img_recon)

                # 更新最佳值
                if metrics['PSNR'] > best_psnr:
                    best_psnr = metrics['PSNR']
                if metrics['SSIM'] > best_ssim:
                    best_ssim = metrics['SSIM']

                total_psnr += metrics['PSNR']
                total_ssim += metrics['SSIM']

            total_samples += batch_size

    # 计算平均值
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples

    # 返回包含平均值和最佳值的字典
    return {
        'PSNR': avg_psnr,
        'SSIM': avg_ssim,
        'best_psnr': best_psnr,
        'best_ssim': best_ssim
    }


# def test_and_visualize(model, test_loader, device, num_images=5, save_dir='results'):
#     """
#     Test the model on the entire dataset and visualize a few sample reconstruction results.
#
#     Args:
#         model: The trained model.
#         test_loader: Data loader for the test dataset.
#         device: The device (CPU or GPU) on which to perform the test.
#         num_images: Number of images to visualize.
#         save_dir: Directory to save the visualization results and metrics.
#
#     Returns:
#         Average evaluation metrics computed over the entire test dataset.
#     """
#     model.eval()
#
#     # 首先对整个数据集进行评估
#     print("\nEvaluating full dataset...")
#     full_dataset_metrics = evaluate_full_dataset(model, test_loader, device)
#
#     # 确保批次大小足够提供所需数量的图像用于可视化
#     required_batch = next(iter(test_loader))[0].shape[0]
#     if num_images > required_batch:
#         print(f"Warning: num_images ({num_images}) is larger than batch size ({required_batch}). "
#               f"Setting num_images to {required_batch}")
#         num_images = required_batch
#
#     with torch.no_grad():
#         # 获取一批图像用于可视化
#         images, _ = next(iter(test_loader))
#         images = images[:num_images].to(device)  # 只使用所需数量的图像
#         outputs, _ = model(images)
#
#     # 设置可视化布局
#     num_cols = min(5, num_images)
#     num_rows = (num_images + num_cols - 1) // num_cols  # 向上取整
#     plt.figure(figsize=(3 * num_cols, 3 * num_rows * 2))
#
#     # 处理每张要可视化的图像
#     for idx in range(num_images):
#         # 转换为numpy数组进行可视化
#         img_orig = images[idx].cpu().numpy().transpose(1, 2, 0)
#         img_recon = outputs[idx].cpu().numpy().transpose(1, 2, 0)
#
#         # 计算当前图像的质量指标（仅用于在图像标题中显示）
#         single_img_metrics = calculate_metrics(img_orig, img_recon)
#
#         # 显示原始图像
#         plt.subplot(num_rows * 2, num_cols, idx + 1)
#         plt.imshow(img_orig)
#         plt.title(f'Original {idx + 1}')
#         plt.axis('off')
#
#         # 显示重建图像及其指标
#         plt.subplot(num_rows * 2, num_cols, idx + 1 + num_cols * num_rows)
#         plt.imshow(img_recon)
#         title = (f'Reconstructed {idx + 1}\nPSNR: {single_img_metrics["PSNR"]:.2f}dB\n'
#                  f'SSIM: {single_img_metrics["SSIM"]:.4f}')
#         plt.title(title)
#         plt.axis('off')
#
#     plt.tight_layout()
#
#     # 保存可视化结果
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     plt.savefig(f'{save_dir}/reconstructions_{timestamp}.png', dpi=300, bbox_inches='tight')
#     plt.close()
#
#     # 构建平均指标（使用整个数据集的结果）
#     avg_metrics = {
#         'PSNR': float(full_dataset_metrics['PSNR']),
#         'SSIM': float(full_dataset_metrics['SSIM']),
#         'best_psnr': float(full_dataset_metrics['best_psnr']),
#         'best_ssim': float(full_dataset_metrics['best_ssim']),
#         # 保留其他指标的格式，但使用数据集平均值
#         'MSE': 0.0,  # 这个值会在print_detailed_metrics中计算但不会显示
#         'NMSE': 0.0,  # 这个值会在print_detailed_metrics中计算但不会显示
#         'channels': {
#             'R': {'MSE': 0.0, 'PSNR': 0.0, 'SSIM': 0.0},
#             'G': {'MSE': 0.0, 'PSNR': 0.0, 'SSIM': 0.0},
#             'B': {'MSE': 0.0, 'PSNR': 0.0, 'SSIM': 0.0}
#         }
#     }
#
#     # 获取模型的压缩信息
#     compressed_info = model.get_compressed_size()
#
#     # 打印详细指标摘要
#     print("\n" + "=" * 70)
#     print("SEMANTIC COMMUNICATION EVALUATION RESULTS")
#     print("=" * 70)
#
#     # 压缩统计信息
#     orig_size = 3 * 32 * 32
#     print("\nCOMPRESSION STATISTICS:")
#     print("-" * 50)
#     print(f"Original size: {orig_size:,} elements")
#     print(f"Compressed size: {compressed_info['total']:,} elements")
#     print(f"Compression ratio: {orig_size / compressed_info['total']:.2f}x")
#     print(f"Compression rate: {(1 - compressed_info['total'] / orig_size) * 100:.2f}%")
#
#     print("\nFeature size breakdown:")
#     for name, size in compressed_info.items():
#         if name != 'total':
#             print(f"  {name:10s}: {int(size):5d} elements ({size / orig_size * 100:6.2f}%)")
#
#     # 整个数据集的质量指标
#     print("\nQUALITY METRICS (FULL DATASET):")
#     print("-" * 50)
#     print(f"Dataset Avg PSNR:      {avg_metrics['PSNR']:.2f} dB")
#     print(f"Dataset Avg SSIM:      {avg_metrics['SSIM']:.4f}")
#     print(f"Dataset Best PSNR:     {avg_metrics['best_psnr']:.2f} dB")
#     print(f"Dataset Best SSIM:     {avg_metrics['best_ssim']:.4f}")
#
#     print("\n" + "=" * 70)
#
#     # 保存评估指标到文件 - 确保所有值都是JSON可序列化的
#     # 创建一个专门用于JSON序列化的字典
#     json_safe_metrics = {
#         'PSNR': float(full_dataset_metrics['PSNR']),
#         'SSIM': float(full_dataset_metrics['SSIM']),
#         'best_psnr': float(full_dataset_metrics['best_psnr']),
#         'best_ssim': float(full_dataset_metrics['best_ssim'])
#     }
#
#     # 确保compressed_info中的所有值都是原生Python类型
#     json_safe_compressed = {}
#     for key, value in compressed_info.items():
#         if isinstance(value, (np.integer, np.floating)):
#             json_safe_compressed[key] = float(value)
#         else:
#             json_safe_compressed[key] = value
#
#     result_dict = {
#         'timestamp': timestamp,
#         'metrics': json_safe_metrics,
#         'compression_info': json_safe_compressed,
#         'test_time': time.strftime('%Y-%m-%d %H:%M:%S')
#     }
#
#     metrics_filename = os.path.join(save_dir, f'metrics_{timestamp}.json')
#     with open(metrics_filename, 'w') as f:
#         json.dump(result_dict, f, indent=4)
#
#     print(f"\nMetrics saved to: {metrics_filename}")
#
#     return full_dataset_metrics

def test_and_visualize(model, test_loader, device, num_images=10, save_dir='results'):
    """
    Test the model on the entire dataset and visualize sample reconstruction results.

    Args:
        model: The trained model.
        test_loader: Data loader for the test dataset.
        device: The device (CPU or GPU) on which to perform the test.
        num_images: Number of images to visualize.
        save_dir: Directory to save the visualization results and metrics.

    Returns:
        Average evaluation metrics computed over the entire test dataset.
    """
    model.eval()

    # 首先对整个数据集进行评估
    print("\nEvaluating full dataset...")
    full_dataset_metrics = evaluate_full_dataset(model, test_loader, device)

    # 确保批次大小足够提供所需数量的图像用于可视化
    required_batch = next(iter(test_loader))[0].shape[0]
    if num_images > required_batch:
        print(f"Warning: num_images ({num_images}) is larger than batch size ({required_batch}). "
              f"Setting num_images to {required_batch}")
        num_images = required_batch

    with torch.no_grad():
        # 获取一批图像用于可视化
        images, _ = next(iter(test_loader))
        images = images[:num_images].to(device)  # 只使用所需数量的图像
        outputs, _ = model(images)

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 设置网格布局
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols  # 向上取整

    try:
        # 创建和保存原始图像的图表
        print("Generating original images plot...")
        plt.figure(figsize=(2.5 * num_cols, 2.5 * num_rows))
        for idx in range(num_images):
            img_orig = images[idx].cpu().numpy().transpose(1, 2, 0)
            plt.subplot(num_rows, num_cols, idx + 1)
            plt.imshow(img_orig)
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        orig_filename = f'{save_dir}/originals_{timestamp}.eps'
        plt.savefig(orig_filename, format='eps', bbox_inches='tight')
        plt.close()
        print(f"Original images saved to: {orig_filename}")
    except Exception as e:
        print(f"Error saving original images: {e}")

    try:
        # 创建和保存重建图像的图表
        print("Generating reconstructed images plot...")
        plt.figure(figsize=(2.5 * num_cols, 2.5 * num_rows))
        for idx in range(num_images):
            img_recon = outputs[idx].cpu().numpy().transpose(1, 2, 0)
            plt.subplot(num_rows, num_cols, idx + 1)
            plt.imshow(img_recon)
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        recon_filename = f'{save_dir}/reconstructions_{timestamp}.eps'
        plt.savefig(recon_filename, format='eps', bbox_inches='tight')
        plt.close()
        print(f"Reconstructed images saved to: {recon_filename}")
    except Exception as e:
        print(f"Error saving reconstructed images: {e}")

    # 获取模型的压缩信息
    compressed_info = model.get_compressed_size()

    # 打印详细指标摘要
    print("\n" + "=" * 70)
    print("SEMANTIC COMMUNICATION EVALUATION RESULTS")
    print("=" * 70)

    # 压缩统计信息
    orig_size = 3 * 32 * 32
    print("\nCOMPRESSION STATISTICS:")
    print("-" * 50)
    print(f"Original size: {orig_size:,} elements")
    print(f"Compressed size: {compressed_info['total']:,} elements")
    print(f"Compression ratio: {orig_size / compressed_info['total']:.2f}x")
    print(f"Compression rate: {(1 - compressed_info['total'] / orig_size) * 100:.2f}%")

    print("\nFeature size breakdown:")
    for name, size in compressed_info.items():
        if name != 'total':
            print(f"  {name:10s}: {int(size):5d} elements ({size / orig_size * 100:6.2f}%)")

    # 整个数据集的质量指标
    print("\nQUALITY METRICS (FULL DATASET):")
    print("-" * 50)
    print(f"Dataset Avg PSNR:      {full_dataset_metrics['PSNR']:.2f} dB")
    print(f"Dataset Avg SSIM:      {full_dataset_metrics['SSIM']:.4f}")
    print(f"Dataset Best PSNR:     {full_dataset_metrics['best_psnr']:.2f} dB")
    print(f"Dataset Best SSIM:     {full_dataset_metrics['best_ssim']:.4f}")

    print("\n" + "=" * 70)

    # 保存评估指标到文件
    json_safe_metrics = {
        'PSNR': float(full_dataset_metrics['PSNR']),
        'SSIM': float(full_dataset_metrics['SSIM']),
        'best_psnr': float(full_dataset_metrics['best_psnr']),
        'best_ssim': float(full_dataset_metrics['best_ssim'])
    }

    # 确保compressed_info中的所有值都是原生Python类型
    json_safe_compressed = {}
    for key, value in compressed_info.items():
        if isinstance(value, (np.integer, np.floating)):
            json_safe_compressed[key] = float(value)
        else:
            json_safe_compressed[key] = value

    result_dict = {
        'timestamp': timestamp,
        'metrics': json_safe_metrics,
        'compression_info': json_safe_compressed,
        'test_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    metrics_filename = os.path.join(save_dir, f'metrics_{timestamp}.json')
    with open(metrics_filename, 'w') as f:
        json.dump(result_dict, f, indent=4)

    print(f"\nMetrics saved to: {metrics_filename}")

    return full_dataset_metrics