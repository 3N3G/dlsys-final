"""
Needle Muon Grid Search - Compare our implementation against PyTorch baseline
"""
import sys
sys.path.append('./python')
sys.path.append('./apps')

import needle as ndl
import numpy as np
import time
import os
import pickle

# Import Needle's data loaders
from needle.data.datasets import CIFAR10Dataset
from needle.data import DataLoader

# Import Needle model (assumes you have ResNet9 in apps/models.py)
try:
    from models import ResNet9
except ImportError:
    print("Warning: Could not import ResNet9 from models.py")
    print("Make sure you have a Needle-compatible ResNet9 in apps/models.py")
    sys.exit(1)


def train_model_needle(model, dataloader, device, use_muon=False, muon_lr=0.1,
                       muon_momentum=0.9, n_epochs=5):
    """Train model using Needle"""

    if use_muon:
        # Needle Muon: Apply to 4D conv parameters (weights)
        # For Needle, we need to check parameter shapes differently
        conv_params = []
        other_params = []

        for p in model.parameters():
            if len(p.shape) == 4:  # Conv weights
                conv_params.append(p)
            else:  # BatchNorm, Linear, biases
                other_params.append(p)

        # Our Needle Muon for conv weights
        opt_muon = ndl.optim.Muon(conv_params, lr=muon_lr, momentum=muon_momentum, nesterov=True)
        # SGD for other parameters
        opt_sgd = ndl.optim.SGD(other_params, lr=0.1, momentum=0.85)
        optimizers = [opt_muon, opt_sgd]
        opt_name = f"Needle Muon+SGD (lr={muon_lr}, m={muon_momentum})"
    else:
        # Adam baseline
        optimizers = [ndl.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)]
        opt_name = "Adam"

    loss_fn = ndl.nn.SoftmaxLoss()

    start_time = time.time()
    final_acc = 0.0

    for epoch in range(n_epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (X, y) in enumerate(dataloader):
            # Reset gradients
            for opt in optimizers:
                opt.reset_grad()

            # Forward pass
            logits = model(X)
            loss = loss_fn(logits, y)

            # Backward pass
            loss.backward()

            # Optimizer step
            for opt in optimizers:
                opt.step()

            # Track metrics
            batch_size = y.shape[0]
            train_loss += loss.numpy().item() * batch_size

            # Compute accuracy
            pred = logits.numpy().argmax(axis=1)
            y_np = y.numpy().astype(np.int32)
            train_correct += (pred == y_np).sum()
            train_total += batch_size

        train_loss /= train_total
        train_acc = train_correct / train_total
        final_acc = train_acc

        if n_epochs <= 5:  # Only print for grid search
            print(f"  Epoch {epoch}: acc={train_acc:.4f}")
        else:  # Full training
            print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f}, train_loss={train_loss:.4f}")

    elapsed = time.time() - start_time
    if n_epochs > 5:
        print(f"Total time: {elapsed:.2f}s\n")

    return final_acc


def main():
    # Check device
    device = ndl.cuda() if ndl.cuda().enabled() else ndl.cpu()
    print(f"Using device: {device}")
    print(f"Using Needle backend: {ndl.backend_selection.BACKEND}")

    # Load CIFAR-10 dataset
    print("\nLoading CIFAR-10...")
    train_dataset = CIFAR10Dataset(
        "data/cifar-10-batches-py",
        train=True,
    )

    # Create dataloader with batch_size=512 to match PyTorch version
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=512,
        shuffle=True,
        device=device,
        dtype="float32"
    )

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batch size: 512")

    # Train Adam baseline
    print("\n" + "="*70)
    print("TRAINING WITH ADAM (baseline)")
    print("="*70)
    model_adam = ResNet9(device=device, dtype="float32")
    train_model_needle(model_adam, dataloader, device, use_muon=False, n_epochs=10)

    # Grid search for Needle Muon hyperparameters
    print("\n" + "="*70)
    print("GRID SEARCH: Testing Needle Muon hyperparameters (batch_size=512)")
    print("="*70)

    muon_lrs = [0.05, 0.1, 0.15, 0.2, 0.25]  # Higher LRs for larger batch size
    muon_momentums = [0.85, 0.9, 0.95]  # Higher momentums
    results = {}

    for lr in muon_lrs:
        for momentum in muon_momentums:
            print(f"\nTesting lr={lr}, momentum={momentum}")
            model = ResNet9(device=device, dtype="float32")
            acc = train_model_needle(
                model, dataloader, device,
                use_muon=True,
                muon_lr=lr,
                muon_momentum=momentum,
                n_epochs=5
            )
            results[(lr, momentum)] = acc

    # Print results sorted by accuracy
    print("\n" + "="*70)
    print("GRID SEARCH RESULTS (Needle Muon)")
    print("="*70)
    for (lr, momentum), acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"lr={lr:6.4f}, momentum={momentum:.2f} -> acc={acc:.4f}")

    best_lr, best_momentum = max(results.items(), key=lambda x: x[1])[0]
    print(f"\nBest: lr={best_lr}, momentum={best_momentum}")

    # Train with best hyperparameters
    print("\n" + "="*70)
    print(f"TRAINING WITH BEST NEEDLE MUON (lr={best_lr}, momentum={best_momentum})")
    print("="*70)
    model_muon = ResNet9(device=device, dtype="float32")
    train_model_needle(
        model_muon, dataloader, device,
        use_muon=True,
        muon_lr=best_lr,
        muon_momentum=best_momentum,
        n_epochs=10
    )

    print("\nDone!")
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print("Compare the results from this script with pytorchmuon.py to see")
    print("how our Needle Muon implementation performs vs PyTorch's bfloat16 version.")
    print("\nNote: Our implementation uses float32 for higher precision,")
    print("while PyTorch uses bfloat16 for efficiency.")


if __name__ == "__main__":
    main()
