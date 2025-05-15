import os
import torch
from datetime import datetime


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    """
    Save a model checkpoint.

    Args:
        model: The model instance.
        optimizer: The optimizer instance.
        epoch: The current epoch number.
        loss: The current loss value.
        checkpoint_dir: Directory where the checkpoint will be saved.
    """
    # Ensure the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Generate a checkpoint filename with timestamp and epoch information
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(
        checkpoint_dir,
        f"checkpoint_epoch_{epoch}_{timestamp}_loss_{loss:.4f}.pth"
    )

    # Create the checkpoint dictionary containing model and optimizer states, epoch, and loss
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # Save the checkpoint to the generated filename
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

    # Also save a 'latest' checkpoint for resuming training easily
    latest_filename = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(checkpoint, latest_filename)
    print("Latest checkpoint updated")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load a model checkpoint.

    Args:
        model: The model instance.
        optimizer: The optimizer instance.
        checkpoint_path: Path to the checkpoint file.

    Returns:
        epoch: The epoch number loaded from the checkpoint.
        loss: The loss value loaded from the checkpoint.
    """
    # Check if the checkpoint file exists
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, None

    # Load the checkpoint dictionary from the file
    checkpoint = torch.load(checkpoint_path)

    # Load model and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint['epoch'], checkpoint['loss']
