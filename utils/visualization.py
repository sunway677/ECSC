import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
from datetime import datetime
import json
import os
from tqdm import tqdm  # For displaying progress bars


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
    ssim_value = ssim(original, reconstructed, data_range=1.0, channel_axis=2) # Use channel_axis for multi-channel (color) images
    mse = np.mean((original - reconstructed) ** 2)
    nmse = mse / np.mean(original ** 2)

    # Calculate metrics for each color channel
    channel_metrics = {}
    for i, channel_name in enumerate(['R', 'G', 'B']):
        orig_channel = original[:, :, i]
        recon_channel = reconstructed[:, :, i]
        channel_mse = np.mean((orig_channel - recon_channel) ** 2)
        channel_psnr = psnr(orig_channel, recon_channel, data_range=1.0)
        channel_ssim = ssim(orig_channel, recon_channel, data_range=1.0) # No channel_axis needed for single channel

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
    orig_size = 3 * 32 * 32 # Assuming 3 channels, 32x32 resolution

    print("\n" + "=" * 70)
    print("SEMANTIC COMMUNICATION EVALUATION RESULTS")
    print("=" * 70)

    # Compression statistics
    print("\nCOMPRESSION STATISTICS:")
    print("-" * 50)
    print(f"Original size: {orig_size:,} elements")
    if 'total' in compressed_info and compressed_info['total'] > 0 : # Check if total is present and positive
        print(f"Compressed size: {compressed_info['total']:,} elements")
        print(f"Compression ratio: {orig_size / compressed_info['total']:.2f}x")
        print(f"Compression rate: {(1 - compressed_info['total'] / orig_size) * 100:.2f}%")
    else:
        print("Compressed size information not available or invalid.")


    print("\nFeature size breakdown:")
    for name, size in compressed_info.items():
        if name != 'total' and orig_size > 0: # Check orig_size to prevent division by zero
            # Ensure size is a number before formatting
            if isinstance(size, (int, float)):
                 print(f"  {name:20s}: {int(size):5d} elements ({size / orig_size * 100:6.2f}%)")
            else:
                 print(f"  {name:20s}: {size}") # Print as is if not a number
        elif name != 'total':
            print(f"  {name:20s}: {size} elements (Original size is zero, percentage not calculable)")


    # Quality metrics
    print("\nQUALITY METRICS:")
    print("-" * 50)
    print(f"Overall PSNR:      {metrics.get('PSNR', 'N/A'):.2f} dB" if isinstance(metrics.get('PSNR'), (int,float)) else f"Overall PSNR:      {metrics.get('PSNR', 'N/A')}")
    print(f"Overall SSIM:      {metrics.get('SSIM', 'N/A'):.4f}" if isinstance(metrics.get('SSIM'), (int,float)) else f"Overall SSIM:      {metrics.get('SSIM', 'N/A')}")
    print(f"Overall MSE:       {metrics.get('MSE', 'N/A'):.6f}" if isinstance(metrics.get('MSE'), (int,float)) else f"Overall MSE:       {metrics.get('MSE', 'N/A')}")
    print(f"Overall NMSE:      {metrics.get('NMSE', 'N/A'):.6f}" if isinstance(metrics.get('NMSE'), (int,float)) else f"Overall NMSE:      {metrics.get('NMSE', 'N/A')}")


    # Per-channel metrics
    if 'channels' in metrics and metrics['channels']:
        print("\nPER-CHANNEL METRICS:")
        print("-" * 50)
        for channel, values in metrics['channels'].items():
            print(f"\nChannel {channel}:")
            print(f"  PSNR: {values.get('PSNR', 'N/A'):.2f} dB" if isinstance(values.get('PSNR'), (int,float)) else f"  PSNR: {values.get('PSNR', 'N/A')}")
            print(f"  SSIM: {values.get('SSIM', 'N/A'):.4f}" if isinstance(values.get('SSIM'), (int,float)) else f"  SSIM: {values.get('SSIM', 'N/A')}")
            print(f"  MSE:  {values.get('MSE', 'N/A'):.6f}" if isinstance(values.get('MSE'), (int,float)) else f"  MSE:  {values.get('MSE', 'N/A')}")
    else:
        print("\nPer-channel metrics not available.")

    print("\n" + "=" * 70)


def evaluate_full_dataset(model, data_loader, device):
    """
    Evaluate the model on the full dataset.

    Args:
        model: The trained model.
        data_loader: Data loader for the dataset.
        device: Computation device (CPU or GPU).

    Returns:
        dict: A dictionary containing average PSNR and SSIM, as well as the best PSNR and SSIM.
    """
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    # Track best values
    best_psnr = 0.0
    best_ssim = 0.0

    print("\nEvaluating full dataset...")
    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            batch_size = images.size(0)

            # Forward pass
            outputs, _ = model(images)

            # Convert to numpy to calculate metrics
            for i in range(batch_size):
                img_orig = images[i].cpu().numpy().transpose(1, 2, 0)
                img_recon = outputs[i].cpu().numpy().transpose(1, 2, 0)

                metrics = calculate_metrics(img_orig, img_recon)

                # Update best values
                if metrics['PSNR'] > best_psnr:
                    best_psnr = metrics['PSNR']
                if metrics['SSIM'] > best_ssim:
                    best_ssim = metrics['SSIM']

                total_psnr += metrics['PSNR']
                total_ssim += metrics['SSIM']

            total_samples += batch_size

    # Calculate average values
    avg_psnr = total_psnr / total_samples if total_samples > 0 else 0
    avg_ssim = total_ssim / total_samples if total_samples > 0 else 0

    # Return a dictionary containing average and best values
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
#     # First, evaluate the entire dataset
#     print("\nEvaluating full dataset...")
#     full_dataset_metrics = evaluate_full_dataset(model, test_loader, device)
#
#     # Ensure batch size is sufficient to provide the required number of images for visualization
#     required_batch = next(iter(test_loader))[0].shape[0]
#     if num_images > required_batch:
#         print(f"Warning: num_images ({num_images}) is larger than batch size ({required_batch}). "
#               f"Setting num_images to {required_batch}")
#         num_images = required_batch
#
#     with torch.no_grad():
#         # Get a batch of images for visualization
#         images, _ = next(iter(test_loader))
#         images = images[:num_images].to(device)  # Use only the required number of images
#         outputs, _ = model(images)
#
#     # Set visualization layout
#     num_cols = min(5, num_images)
#     num_rows = (num_images + num_cols - 1) // num_cols  # Round up
#     plt.figure(figsize=(3 * num_cols, 3 * num_rows * 2))
#
#     # Process each image to be visualized
#     for idx in range(num_images):
#         # Convert to numpy array for visualization
#         img_orig = images[idx].cpu().numpy().transpose(1, 2, 0)
#         img_recon = outputs[idx].cpu().numpy().transpose(1, 2, 0)
#
#         # Calculate quality metrics for the current image (only for display in the image title)
#         single_img_metrics = calculate_metrics(img_orig, img_recon)
#
#         # Display original image
#         plt.subplot(num_rows * 2, num_cols, idx + 1)
#         plt.imshow(img_orig)
#         plt.title(f'Original {idx + 1}')
#         plt.axis('off')
#
#         # Display reconstructed image and its metrics
#         plt.subplot(num_rows * 2, num_cols, idx + 1 + num_cols * num_rows)
#         plt.imshow(img_recon)
#         title = (f'Reconstructed {idx + 1}\nPSNR: {single_img_metrics["PSNR"]:.2f}dB\n'
#                  f'SSIM: {single_img_metrics["SSIM"]:.4f}')
#         plt.title(title)
#         plt.axis('off')
#
#     plt.tight_layout()
#
#     # Save visualization results
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     plt.savefig(f'{save_dir}/reconstructions_{timestamp}.png', dpi=300, bbox_inches='tight')
#     plt.close()
#
#     # Construct average metrics (using results from the entire dataset)
#     avg_metrics = {
#         'PSNR': float(full_dataset_metrics['PSNR']),
#         'SSIM': float(full_dataset_metrics['SSIM']),
#         'best_psnr': float(full_dataset_metrics['best_psnr']),
#         'best_ssim': float(full_dataset_metrics['best_ssim']),
#         # Keep the format of other metrics, but use dataset average values
#         'MSE': 0.0,  # This value will be calculated in print_detailed_metrics but not displayed
#         'NMSE': 0.0, # This value will be calculated in print_detailed_metrics but not displayed
#         'channels': {
#             'R': {'MSE': 0.0, 'PSNR': 0.0, 'SSIM': 0.0},
#             'G': {'MSE': 0.0, 'PSNR': 0.0, 'SSIM': 0.0},
#             'B': {'MSE': 0.0, 'PSNR': 0.0, 'SSIM': 0.0}
#         }
#     }
#
#     # Get compression information from the model
#     compressed_info = model.get_compressed_size()
#
#     # Print detailed metrics summary
#     print("\n" + "=" * 70)
#     print("SEMANTIC COMMUNICATION EVALUATION RESULTS")
#     print("=" * 70)
#
#     # Compression statistics
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
#     # Quality metrics for the entire dataset
#     print("\nQUALITY METRICS (FULL DATASET):")
#     print("-" * 50)
#     print(f"Dataset Avg PSNR:      {avg_metrics['PSNR']:.2f} dB")
#     print(f"Dataset Avg SSIM:      {avg_metrics['SSIM']:.4f}")
#     print(f"Dataset Best PSNR:     {avg_metrics['best_psnr']:.2f} dB")
#     print(f"Dataset Best SSIM:     {avg_metrics['best_ssim']:.4f}")
#
#     print("\n" + "=" * 70)
#
#     # Save evaluation metrics to file - ensure all values are JSON serializable
#     # Create a dictionary specifically for JSON serialization
#     json_safe_metrics = {
#         'PSNR': float(full_dataset_metrics['PSNR']),
#         'SSIM': float(full_dataset_metrics['SSIM']),
#         'best_psnr': float(full_dataset_metrics['best_psnr']),
#         'best_ssim': float(full_dataset_metrics['best_ssim'])
#     }
#
#     # Ensure all values in compressed_info are native Python types
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

    # First, evaluate the entire dataset
    print("\nEvaluating full dataset...")
    full_dataset_metrics = evaluate_full_dataset(model, test_loader, device)

    # Ensure batch size is sufficient to provide the required number of images for visualization
    # Get the first batch to check its size
    try:
        first_batch_images, _ = next(iter(test_loader))
        available_images_in_batch = first_batch_images.shape[0]
    except StopIteration:
        print("Warning: Test loader is empty. Cannot visualize images.")
        available_images_in_batch = 0 # No images to visualize

    if num_images > available_images_in_batch:
        print(f"Warning: num_images ({num_images}) is larger than available images in the first batch ({available_images_in_batch}). "
              f"Setting num_images to {available_images_in_batch}")
        num_images = available_images_in_batch
    
    if num_images == 0: # If no images can be visualized
        print("No images to visualize.")
    else:
        with torch.no_grad():
            # Get a batch of images for visualization (use the already fetched first_batch_images)
            images = first_batch_images[:num_images].to(device)  # Use only the required number of images
            outputs, _ = model(images)

        # Create save directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set grid layout
        num_cols = 5
        num_rows = (num_images + num_cols - 1) // num_cols  # Round up

        try:
            # Create and save the plot for original images
            print("Generating original images plot...")
            plt.figure(figsize=(2.5 * num_cols, 2.5 * num_rows))
            for idx in range(num_images):
                img_orig = images[idx].cpu().numpy().transpose(1, 2, 0)
                img_orig = np.clip(img_orig, 0, 1) # Clip values for correct display
                plt.subplot(num_rows, num_cols, idx + 1)
                plt.imshow(img_orig)
                plt.axis('off')
            plt.subplots_adjust(wspace=0.05, hspace=0.05) # Reduce spacing
            # plt.tight_layout() # Can sometimes conflict with subplots_adjust
            orig_filename = f'{save_dir}/originals_{timestamp}.eps'
            plt.savefig(orig_filename, format='eps', bbox_inches='tight')
            plt.close()
            print(f"Original images saved to: {orig_filename}")
        except Exception as e:
            print(f"Error saving original images: {e}")

        try:
            # Create and save the plot for reconstructed images
            print("Generating reconstructed images plot...")
            plt.figure(figsize=(2.5 * num_cols, 2.5 * num_rows))
            for idx in range(num_images):
                img_recon = outputs[idx].cpu().numpy().transpose(1, 2, 0)
                img_recon = np.clip(img_recon, 0, 1) # Clip values for correct display
                plt.subplot(num_rows, num_cols, idx + 1)
                plt.imshow(img_recon)
                plt.axis('off')
            plt.subplots_adjust(wspace=0.05, hspace=0.05) # Reduce spacing
            # plt.tight_layout()
            recon_filename = f'{save_dir}/reconstructions_{timestamp}.eps'
            plt.savefig(recon_filename, format='eps', bbox_inches='tight')
            plt.close()
            print(f"Reconstructed images saved to: {recon_filename}")
        except Exception as e:
            print(f"Error saving reconstructed images: {e}")

    # Get compression information from the model
    if hasattr(model, 'get_compressed_size') and callable(model.get_compressed_size):
        compressed_info = model.get_compressed_size()
    else:
        print("Warning: Model does not have 'get_compressed_size' method. Compression info will be empty.")
        compressed_info = {}


    # Print detailed metrics summary
    print("\n" + "=" * 70)
    print("SEMANTIC COMMUNICATION EVALUATION RESULTS")
    print("=" * 70)

    # Compression statistics
    orig_size = 3 * 32 * 32 # Assuming 3 channels, 32x32 resolution
    print("\nCOMPRESSION STATISTICS:")
    print("-" * 50)
    print(f"Original size: {orig_size:,} elements")

    if 'total' in compressed_info and isinstance(compressed_info['total'], (int, float)) and compressed_info['total'] > 0:
        print(f"Compressed size: {int(compressed_info['total']):,} elements")
        print(f"Compression ratio: {orig_size / compressed_info['total']:.2f}x")
        print(f"Compression rate: {(1 - compressed_info['total'] / orig_size) * 100:.2f}%")

        print("\nFeature size breakdown:")
        for name, size_val in compressed_info.items():
            if name != 'total' and isinstance(size_val, (int, float)) and orig_size > 0:
                print(f"  {name:20s}: {int(size_val):5d} elements ({size_val / orig_size * 100:6.2f}%)")
            elif name != 'total':
                 print(f"  {name:20s}: {size_val}")
    else:
        print("Compressed size information not available or invalid.")


    # Quality metrics for the entire dataset
    print("\nQUALITY METRICS (FULL DATASET):")
    print("-" * 50)
    print(f"Dataset Avg PSNR:      {full_dataset_metrics.get('PSNR', 'N/A'):.2f} dB" if isinstance(full_dataset_metrics.get('PSNR'), (int,float)) else f"Dataset Avg PSNR:      {full_dataset_metrics.get('PSNR', 'N/A')}")
    print(f"Dataset Avg SSIM:      {full_dataset_metrics.get('SSIM', 'N/A'):.4f}" if isinstance(full_dataset_metrics.get('SSIM'), (int,float)) else f"Dataset Avg SSIM:      {full_dataset_metrics.get('SSIM', 'N/A')}")
    print(f"Dataset Best PSNR:     {full_dataset_metrics.get('best_psnr', 'N/A'):.2f} dB" if isinstance(full_dataset_metrics.get('best_psnr'), (int,float)) else f"Dataset Best PSNR:     {full_dataset_metrics.get('best_psnr', 'N/A')}")
    print(f"Dataset Best SSIM:     {full_dataset_metrics.get('best_ssim', 'N/A'):.4f}" if isinstance(full_dataset_metrics.get('best_ssim'), (int,float)) else f"Dataset Best SSIM:     {full_dataset_metrics.get('best_ssim', 'N/A')}")

    print("\n" + "=" * 70)

    # Save evaluation metrics to file
    # Ensure all metric values are float for JSON serialization
    json_safe_metrics = {}
    for key, value in full_dataset_metrics.items():
        json_safe_metrics[key] = float(value) if isinstance(value, (int, float, np.number)) else str(value)


    # Ensure all values in compressed_info are native Python types for JSON
    json_safe_compressed = {}
    for key, value in compressed_info.items():
        if isinstance(value, (np.integer, np.floating)): # Handles numpy numeric types
            json_safe_compressed[key] = float(value)
        elif isinstance(value, (int, float)):
            json_safe_compressed[key] = value
        else:
            json_safe_compressed[key] = str(value) # Convert other types to string

    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Added microseconds for more uniqueness
    result_dict = {
        'timestamp': current_timestamp,
        'metrics': json_safe_metrics,
        'compression_info': json_safe_compressed,
        'test_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Create save_dir if it doesn't exist, to prevent error when num_images is 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    metrics_filename = os.path.join(save_dir, f'metrics_{current_timestamp}.json')
    with open(metrics_filename, 'w') as f:
        json.dump(result_dict, f, indent=4)

    print(f"\nMetrics saved to: {metrics_filename}")

    return full_dataset_metrics
