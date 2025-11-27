#!/usr/bin/env python3
"""
Complete CIFAR-10 Training Script for Needle Framework
Uses ResNet9 architecture with data augmentation
"""

import sys
sys.path.append('./python')

import numpy as np
import needle as ndl
import needle.nn as nn
from apps.models import ResNet9
from apps.simple_ml import train_cifar10, evaluate_cifar10
import time
import os


def ensure_data_downloaded():
    """Check if CIFAR-10 data exists, if not download it."""
    cifar_dir = "./data/cifar-10-batches-py"

    if os.path.isdir(cifar_dir):
        print(f"CIFAR-10 data found at {cifar_dir}")
        return True

    print("CIFAR-10 data not found. Downloading...")
    try:
        import download_data
        download_data.download_cifar10("./data")
        return True
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("\nPlease run: python download_data.py --cifar10")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train ResNet9 on CIFAR-10")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training (default: 128)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--weight-decay", type=float, default=0.001,
                        help="Weight decay (default: 0.001)")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        choices=["SGD", "Adam"],
                        help="Optimizer to use (default: Adam)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use (default: cpu)")
    parser.add_argument("--data-dir", type=str, default="./data/cifar-10-batches-py",
                        help="Path to CIFAR-10 data directory")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable data augmentation")

    args = parser.parse_args()

    # Ensure data is downloaded
    if not ensure_data_downloaded():
        return

    # Set device
    if args.device == "cuda":
        device = ndl.cuda()
        if not device.enabled():
            print("CUDA not available, falling back to CPU")
            device = ndl.cpu()
    else:
        device = ndl.cpu()

    print(f"\n{'='*60}")
    print(f"Training ResNet9 on CIFAR-10")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Data augmentation: {not args.no_augmentation}")
    print(f"{'='*60}\n")

    # Create datasets
    print("Loading datasets...")

    if args.no_augmentation:
        train_transforms = None
    else:
        train_transforms = [
            ndl.data.RandomCrop(padding=4),
            ndl.data.RandomFlipHorizontal(p=0.5)
        ]

    train_dataset = ndl.data.CIFAR10Dataset(
        args.data_dir,
        train=True,
        transforms=train_transforms
    )

    test_dataset = ndl.data.CIFAR10Dataset(
        args.data_dir,
        train=False,
        transforms=None
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_dataloader = ndl.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
        dtype="float32"
    )

    test_dataloader = ndl.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
        dtype="float32"
    )

    # Create model
    print("\nInitializing ResNet9 model...")
    model = ResNet9(device=device, dtype="float32")
    print(f"Model parameters: {sum(p.numpy().size for p in model.parameters()):,}")

    # Select optimizer
    if args.optimizer == "Adam":
        optimizer = ndl.optim.Adam
    else:
        optimizer = ndl.optim.SGD

    # Loss function
    loss_fn = nn.SoftmaxLoss

    # Training
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"{'='*60}\n")

    best_test_acc = 0.0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train for one epoch
        train_acc, train_loss = train_cifar10(
            model,
            train_dataloader,
            n_epochs=1,
            optimizer=optimizer,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_fn=loss_fn
        )

        # Evaluate on test set
        test_acc, test_loss = evaluate_cifar10(
            model,
            test_dataloader,
            loss_fn=loss_fn
        )

        epoch_time = time.time() - epoch_start

        # Print results
        print(f"Epoch [{epoch+1}/{args.epochs}] ({epoch_time:.2f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"  >>> New best test accuracy: {best_test_acc*100:.2f}%")

        print()

    print(f"{'='*60}")
    print(f"Training Complete!")
    print(f"Best Test Accuracy: {best_test_acc*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
