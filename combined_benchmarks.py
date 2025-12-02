"""
Combined PyTorch and Needle Muon Grid Search Comparison
Based on airbench94 hyperparameters: Muon(lr=0.24, momentum=0.6), SGD(lr=0.053-0.67, momentum=0.85)
"""
import sys
sys.path.append('./python')
sys.path.append('./apps')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import needle as ndl
import numpy as np
import time
import os
import pickle

# ============================================================================
# PYTORCH MUON OPTIMIZER (from airbench94)
# ============================================================================
@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz iteration"""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class MuonPyTorch(torch.optim.Optimizer):
    """Muon optimizer from airbench94 (with weight normalization)"""
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                # Weight normalization (from airbench94)
                p.data.mul_(len(p.data)**0.5 / p.data.norm())
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                p.data.add_(update, alpha=-lr)

# ============================================================================
# PYTORCH DATASET
# ============================================================================
class CIFAR10Dataset(Dataset):
    def __init__(self, base_folder: str, train: bool):
        if train:
            files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            files = ["test_batch"]

        imgs = []
        labels = []

        for fname in files:
            path = os.path.join(base_folder, fname)
            with open(path, "rb") as f:
                d = pickle.load(f, encoding="latin1")
            data = d.get("data", None)
            if data is None:
                data = d.get(b"data")
            lbs = d.get("labels", None)
            if lbs is None:
                lbs = d.get(b"labels")

            data = np.asarray(data, dtype=np.float32)
            data = data.reshape(-1, 3, 32, 32) / 255.0
            imgs.append(data)
            labels.append(np.asarray(lbs, dtype=np.int64))

        self.X = np.concatenate(imgs, axis=0) if len(imgs) > 1 else imgs[0]
        self.y = np.concatenate(labels, axis=0) if len(labels) > 1 else labels[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.tensor(self.y[index], dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

# ============================================================================
# PYTORCH RESNET9
# ============================================================================
class ResNet9PyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3), nn.BatchNorm2d(16), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.fc1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        out2 = self.conv2(x)
        x = self.conv3(out2)
        x = self.conv4(x) + out2
        x = self.conv5(x)
        out6 = self.conv6(x)
        x = self.conv7(out6)
        x = self.conv8(x)
        x = x.reshape(x.shape[0], -1) + out6.reshape(out6.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ============================================================================
# PYTORCH TRAINING
# ============================================================================
def train_pytorch(model, dataloader, use_muon=False, muon_lr=0.24, muon_momentum=0.6,
                  sgd_lr=0.1, sgd_momentum=0.85, n_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if use_muon:
        # airbench94 style: Muon for conv weights (4D), SGD for everything else
        conv_params = [p for p in model.parameters() if len(p.shape) == 4]
        other_params = [p for p in model.parameters() if len(p.shape) != 4]

        opt_muon = MuonPyTorch(conv_params, lr=muon_lr, momentum=muon_momentum, nesterov=True)
        opt_sgd = torch.optim.SGD(other_params, lr=sgd_lr, momentum=sgd_momentum, nesterov=True)
        optimizers = [opt_muon, opt_sgd]
    else:
        optimizers = [torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)]

    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()
        train_correct = 0
        train_total = 0

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            for opt in optimizers:
                opt.zero_grad()

            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()

            for opt in optimizers:
                opt.step()

            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.shape[0]

        train_acc = train_correct / train_total
        if n_epochs <= 5:
            print(f"  Epoch {epoch}: acc={train_acc:.4f}")
        else:
            print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f}")

    elapsed = time.time() - start_time
    if n_epochs > 5:
        print(f"Total time: {elapsed:.2f}s")
    return train_acc

# ============================================================================
# NEEDLE TRAINING
# ============================================================================
def train_needle(model, dataloader, device, use_muon=False, muon_lr=0.24, muon_momentum=0.6,
                 sgd_lr=0.1, sgd_momentum=0.85, n_epochs=5):
    if use_muon:
        # Needle Muon for conv weights (4D), SGD for everything else
        conv_params = [p for p in model.parameters() if len(p.shape) == 4]
        other_params = [p for p in model.parameters() if len(p.shape) != 4]

        opt_muon = ndl.optim.Muon(conv_params, lr=muon_lr, momentum=muon_momentum, nesterov=True)
        opt_sgd = ndl.optim.SGD(other_params, lr=sgd_lr, momentum=sgd_momentum)
        optimizers = [opt_muon, opt_sgd]
    else:
        optimizers = [ndl.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)]

    loss_fn = ndl.nn.SoftmaxLoss()
    start_time = time.time()

    for epoch in range(n_epochs):
        train_correct = 0
        train_total = 0

        for X_np, y_np in dataloader:
            # Convert numpy arrays to Needle tensors on the correct device
            # DataLoader returns numpy arrays, need to convert to Tensors
            if isinstance(X_np, ndl.Tensor):
                X = X_np
                y = y_np
            else:
                X = ndl.Tensor(X_np, device=device, dtype="float32")
                y = ndl.Tensor(y_np, device=device, dtype="float32")

            for opt in optimizers:
                opt.reset_grad()

            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()

            for opt in optimizers:
                opt.step()

            pred = logits.numpy().argmax(axis=1)
            y_actual = y.numpy().astype(np.int32)
            train_correct += (pred == y_actual).sum()
            train_total += y.numpy().shape[0]

        train_acc = train_correct / train_total
        if n_epochs <= 5:
            print(f"  Epoch {epoch}: acc={train_acc:.4f}")
        else:
            print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f}")

    elapsed = time.time() - start_time
    if n_epochs > 5:
        print(f"Total time: {elapsed:.2f}s")
    return train_acc

# ============================================================================
# MAIN GRID SEARCH
# ============================================================================
def main():
    print("="*70)
    print("COMBINED PYTORCH AND NEEDLE MUON GRID SEARCH")
    print("="*70)
    print("\nHyperparameters based on airbench94:")
    print("  Best known: Muon(lr=0.24, momentum=0.6), SGD(lr=0.053-0.67, momentum=0.85)")
    print("  Grid search will explore ranges around these values")

    # Load CIFAR-10
    print("\nLoading CIFAR-10...")
    dataset = CIFAR10Dataset("data/cifar-10-batches-py", train=True)

    # Batch size: 512 for faster training
    batch_size = 512

    # PyTorch dataloader
    torch_dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Needle setup
    try:
        from models import ResNet9 as ResNet9Needle
        from needle.data.datasets import CIFAR10Dataset as NeedleCIFAR10
        from needle.data import DataLoader as NeedleDataLoader

        needle_device = ndl.cuda() if ndl.cuda().enabled() else ndl.cpu()
        needle_dataset = NeedleCIFAR10("data/cifar-10-batches-py", train=True)
        # Needle DataLoader doesn't take device/dtype - data is converted in training loop
        needle_dataloader = NeedleDataLoader(needle_dataset, batch_size=batch_size, shuffle=True)
        needle_available = True
        print(f"Needle device: {needle_device}")
    except Exception as e:
        print(f"Needle not available: {e}")
        needle_available = False

    # Grid search ranges (expanded around airbench94 values)
    # Muon: lr=0.24, momentum=0.6 -> explore [0.1-0.4] x [0.5-0.95]
    # SGD: lr=0.053-0.67 -> explore [0.03-0.15]
    muon_lrs = [0.1, 0.15, 0.2, 0.24, 0.3, 0.35, 0.4]
    muon_momentums = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    sgd_lrs = [0.03, 0.05, 0.075, 0.1, 0.15]

    print(f"\nGrid search space:")
    print(f"  Muon LR: {muon_lrs}")
    print(f"  Muon momentum: {muon_momentums}")
    print(f"  SGD LR: {sgd_lrs}")
    print(f"  Total combinations: {len(muon_lrs) * len(muon_momentums) * len(sgd_lrs)}")

    # ========================================================================
    # PYTORCH GRID SEARCH
    # ========================================================================
    print("\n" + "="*70)
    print("PYTORCH MUON GRID SEARCH (bfloat16)")
    print("="*70)

    pytorch_results = {}
    for muon_lr in muon_lrs:
        for muon_momentum in muon_momentums:
            for sgd_lr in sgd_lrs:
                print(f"\nPyTorch: Muon(lr={muon_lr}, m={muon_momentum}), SGD(lr={sgd_lr})")
                model = ResNet9PyTorch().cuda() if torch.cuda.is_available() else ResNet9PyTorch()
                acc = train_pytorch(model, torch_dataloader, use_muon=True,
                                   muon_lr=muon_lr, muon_momentum=muon_momentum,
                                   sgd_lr=sgd_lr, n_epochs=3)  # 3 epochs for faster grid search
                pytorch_results[(muon_lr, muon_momentum, sgd_lr)] = acc

    # ========================================================================
    # NEEDLE GRID SEARCH
    # ========================================================================
    if needle_available:
        print("\n" + "="*70)
        print("NEEDLE MUON GRID SEARCH (float32)")
        print("="*70)

        needle_results = {}
        for muon_lr in muon_lrs:
            for muon_momentum in muon_momentums:
                for sgd_lr in sgd_lrs:
                    print(f"\nNeedle: Muon(lr={muon_lr}, m={muon_momentum}), SGD(lr={sgd_lr})")
                    model = ResNet9Needle(device=needle_device, dtype="float32")
                    acc = train_needle(model, needle_dataloader, needle_device, use_muon=True,
                                      muon_lr=muon_lr, muon_momentum=muon_momentum,
                                      sgd_lr=sgd_lr, n_epochs=3)
                    needle_results[(muon_lr, muon_momentum, sgd_lr)] = acc

    # ========================================================================
    # RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("GRID SEARCH RESULTS")
    print("="*70)

    print("\nPyTorch Top 10:")
    for (muon_lr, muon_m, sgd_lr), acc in sorted(pytorch_results.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  Muon(lr={muon_lr:.2f}, m={muon_m:.2f}), SGD(lr={sgd_lr:.3f}) -> acc={acc:.4f}")

    if needle_available:
        print("\nNeedle Top 10:")
        for (muon_lr, muon_m, sgd_lr), acc in sorted(needle_results.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  Muon(lr={muon_lr:.2f}, m={muon_m:.2f}), SGD(lr={sgd_lr:.3f}) -> acc={acc:.4f}")

    best_pytorch = max(pytorch_results.items(), key=lambda x: x[1])
    print(f"\nBest PyTorch: Muon(lr={best_pytorch[0][0]}, m={best_pytorch[0][1]}), "
          f"SGD(lr={best_pytorch[0][2]:.3f}) -> acc={best_pytorch[1]:.4f}")

    if needle_available:
        best_needle = max(needle_results.items(), key=lambda x: x[1])
        print(f"Best Needle: Muon(lr={best_needle[0][0]}, m={best_needle[0][1]}), "
              f"SGD(lr={best_needle[0][2]:.3f}) -> acc={best_needle[1]:.4f}")
        print(f"\nAccuracy difference: {abs(best_pytorch[1] - best_needle[1]):.4f}")

    print("\nDone!")

if __name__ == "__main__":
    main()
