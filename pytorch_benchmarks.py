import sys
sys.path.append('./python')
sys.path.append('./apps')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os
import pickle
import numpy as np

# Force raw float32 precision globally (disable TF32)
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Muon optimizer (from airbench94)
@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """Newton-Schulz iteration"""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()  # Changed from bfloat16 to float32
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

class Muon(torch.optim.Optimizer):
    """Muon optimizer from airbench94"""
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
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

                p.data.mul_(len(p.data)**0.5 / p.data.norm())
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                p.data.add_(update, alpha=-lr)

# SOAP optimizer
class SOAP(torch.optim.Optimizer):
    """Second-Order Adaptive Preconditioning (SOAP) optimizer"""
    def __init__(self, params, lr=0.1, momentum=0, weight_decay=0, nesterov=False, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p = d_p.add(p.data, alpha=group['weight_decay'])

                param_state = self.state[p]

                if len(p.shape) >= 2:
                    # Second-order preconditioning for matrices
                    if 'H' not in param_state:
                        param_state['H'] = torch.eye(p.shape[0], device=p.device, dtype=p.dtype) * group['eps']

                    H = param_state['H']

                    # Update Hessian approximation (simplified)
                    d_p_flat = d_p.view(p.shape[0], -1)
                    grad_norm = (d_p_flat * d_p_flat).sum(dim=1, keepdim=True).sqrt()
                    H = H * 0.99 + 0.01 * torch.diag(grad_norm.squeeze())
                    param_state['H'] = H

                    # Precondition gradient
                    d_p_precond = d_p_flat / (torch.diag(H).view(-1, 1) + group['eps'])
                    d_p = d_p_precond.view(p.shape)

                if group['momentum'] != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p, alpha=1)

                    if group['nesterov']:
                        d_p = d_p.add(buf, alpha=group['momentum'])
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

        return loss

# PyTorch CIFAR10 Dataset
class CIFAR10Dataset(Dataset):
    def __init__(self, base_folder: str, train: bool):
        self.base_folder = base_folder
        self.train = train

        if train:
            files = [f"data_batch_{i}" for i in range(1, 5 + 1)]
        else:
            files = ["test_batch"]

        imgs = []
        labels = []

        def _load_pickle(fp):
            with open(fp, "rb") as f:
                return pickle.load(f, encoding="latin1")

        for fname in files:
            path = os.path.join(base_folder, fname)
            d = _load_pickle(path)
            data = d.get("data", None)
            if data is None:
                data = d.get(b"data")
            lbs = d.get("labels", None)
            if lbs is None:
                lbs = d.get(b"labels")

            data = np.asarray(data, dtype=np.float32)
            n = data.shape[0]
            data = data.reshape(n, 3, 32, 32)
            data = data / 255.0

            imgs.append(data)
            labels.append(np.asarray(lbs, dtype=np.int64))

        self.X = np.concatenate(imgs, axis=0) if len(imgs) > 1 else imgs[0]
        self.y = np.concatenate(labels, axis=0) if len(labels) > 1 else labels[0]

    def __getitem__(self, index):
        img_chw = self.X[index]
        label = int(self.y[index])
        img_tensor = torch.from_numpy(img_chw)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

    def __len__(self):
        return self.X.shape[0]


def train_model(model, dataloader, optimizer='muon', muon_lr=0.005, sgd_lr=0.2, momentum=0.95, adam_lr=0.001, adam_weight_decay=0.001, soap_lr=0.1, soap_momentum=0.9, n_epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if optimizer == 'adam':
        optimizers = [torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=adam_weight_decay)]
    elif optimizer == 'soap':
        optimizers = [SOAP(model.parameters(), lr=soap_lr, momentum=soap_momentum, weight_decay=0.001, nesterov=True)]
    else:  # muon
        # Muon for conv weights, SGD for everything else
        conv_params = [p for p in model.parameters() if len(p.shape) == 4]
        other_params = [p for p in model.parameters() if len(p.shape) != 4]

        opt_muon = Muon(conv_params, lr=muon_lr, momentum=momentum, nesterov=True)
        opt_sgd = torch.optim.SGD(other_params, lr=sgd_lr, momentum=momentum, nesterov=True)
        optimizers = [opt_muon, opt_sgd]

    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time()
    final_acc = 0.0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
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

            train_loss += loss.item() * y.shape[0]
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.shape[0]

        train_loss /= train_total
        train_acc = train_correct / train_total
        final_acc = train_acc

        print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f}, train_loss={train_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f}s\n")

    return final_acc

# ResNet9
class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        out2 = self.conv2(x)

        x = self.conv3(out2)
        x = self.conv4(x)
        x = x + out2

        x = self.conv5(x)
        out6 = self.conv6(x)

        x = self.conv7(out6)
        x = self.conv8(x)

        x_flat = x.reshape(x.shape[0], -1)
        out6_flat = out6.reshape(out6.shape[0], -1)
        x = x_flat + out6_flat

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load dataloader with larger batch size
print("Loading CIFAR10...")
dataset = CIFAR10Dataset("data/cifar-10-batches-py", train=True)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=128,
    shuffle=True,
)

# Train with Adam
print("="*70)
print("TRAINING WITH ADAM (float32 precision)")
print("adam_lr=0.001, adam_weight_decay=0.001, epochs=30")
print("="*70)
model_adam = ResNet9().cuda()
train_model(model_adam, dataloader, optimizer='adam', adam_lr=0.001, adam_weight_decay=0.001, n_epochs=30)

# Train with Muon + SGD
print("="*70)
print("TRAINING WITH MUON + SGD (float32 precision)")
print("muon_lr=0.005, sgd_lr=0.2, momentum=0.95, epochs=30")
print("="*70)
model_muon = ResNet9().cuda()
train_model(model_muon, dataloader, optimizer='muon', muon_lr=0.007, sgd_lr=0.2, momentum=0.95, n_epochs=30)

# Train with SOAP
print("="*70)
print("TRAINING WITH SOAP (float32 precision)")
print("soap_lr=0.1, soap_momentum=0.9, weight_decay=0.001, epochs=30")
print("="*70)
model_soap = ResNet9().cuda()
train_model(model_soap, dataloader, optimizer='soap', soap_lr=0.1, soap_momentum=0.9, n_epochs=30)

print("Done!")
