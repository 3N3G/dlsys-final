#!/usr/bin/env python3
"""
Quick CIFAR-10 test with reduced dataset for fast iteration
"""

import sys
sys.path.append('./python')

import numpy as np
import needle as ndl
import needle.nn as nn
from apps.models import ResNet9
from apps.simple_ml import epoch_general_cifar10
import time


def main():
    print("Quick CIFAR-10 Test - Reduced Dataset")
    print("="*60)

    # Use CPU for quick testing
    device = ndl.cpu()

    # Small settings for fast testing
    batch_size = 32
    max_train_batches = 10  # Only use 10 batches per epoch
    max_test_batches = 5    # Only use 5 test batches

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Train batches per epoch: {max_train_batches}")
    print(f"Test batches: {max_test_batches}")
    print("="*60 + "\n")

    # Load datasets
    print("Loading datasets...")
    train_dataset = ndl.data.CIFAR10Dataset(
        "./data/cifar-10-batches-py",
        train=True,
        transforms=None  # No augmentation for speed
    )

    test_dataset = ndl.data.CIFAR10Dataset(
        "./data/cifar-10-batches-py",
        train=False,
        transforms=None
    )

    # Create limited dataloaders
    train_dataloader = ndl.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        device=device,
        dtype="float32"
    )

    test_dataloader = ndl.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        dtype="float32"
    )

    # Create model
    print("Initializing ResNet9...")
    model = ResNet9(device=device, dtype="float32")
    print(f"Model parameters: {sum(p.numpy().size for p in model.parameters()):,}\n")

    # Test different optimizers
    optimizers_to_test = [
        ("SGD", ndl.optim.SGD, {"lr": 0.01, "momentum": 0.9}),
        ("Adam", ndl.optim.Adam, {"lr": 0.001}),
        ("Muon", ndl.optim.Muon, {"lr": 0.02}),
        ("SOAP", ndl.optim.SOAP, {"lr": 0.001}),
    ]

    loss_fn = nn.SoftmaxLoss()

    for opt_name, opt_class, opt_kwargs in optimizers_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {opt_name} optimizer")
        print(f"{'='*60}")

        # Reinitialize model for fair comparison
        model = ResNet9(device=device, dtype="float32")
        optimizer = opt_class(model.parameters(), weight_decay=0.0, **opt_kwargs)

        # Train for 1 epoch (limited batches)
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        start_time = time.time()

        batch_count = 0
        for X, y in train_dataloader:
            if batch_count >= max_train_batches:
                break

            batch_size_actual = y.shape[0]

            # Forward
            optimizer.reset_grad()
            logits = model(X)
            loss = loss_fn(logits, y)

            # Backward
            loss.backward()
            optimizer.step()

            # Track metrics
            preds = logits.numpy().argmax(axis=1)
            true = y.numpy().astype(np.int32).reshape(-1)
            train_correct += (preds == true).sum()
            train_total += batch_size_actual
            train_loss_sum += float(loss.numpy()) * batch_size_actual

            batch_count += 1

            if batch_count % 5 == 0:
                print(f"  Batch {batch_count}/{max_train_batches}, Loss: {float(loss.numpy()):.4f}")

        train_time = time.time() - start_time
        train_acc = train_correct / train_total
        train_loss = train_loss_sum / train_total

        # Evaluate on test set (limited batches)
        model.eval()
        test_loss_sum = 0.0
        test_correct = 0
        test_total = 0

        batch_count = 0
        for X, y in test_dataloader:
            if batch_count >= max_test_batches:
                break

            batch_size_actual = y.shape[0]

            logits = model(X)
            loss = loss_fn(logits, y)

            preds = logits.numpy().argmax(axis=1)
            true = y.numpy().astype(np.int32).reshape(-1)
            test_correct += (preds == true).sum()
            test_total += batch_size_actual
            test_loss_sum += float(loss.numpy()) * batch_size_actual

            batch_count += 1

        test_acc = test_correct / test_total
        test_loss = test_loss_sum / test_total

        print(f"\n{opt_name} Results:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% ({train_time:.2f}s)")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
