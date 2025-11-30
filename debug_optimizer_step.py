"""Detailed step-by-step comparison of Muon optimizer"""
import sys
sys.path.append('./python')

import needle as ndl
import numpy as np
import torch

# Set seed
np.random.seed(42)
torch.manual_seed(42)

print("=" * 70)
print("DETAILED MUON OPTIMIZER STEP COMPARISON")
print("=" * 70)

# Create identical weights
w_init = np.random.randn(20, 10).astype(np.float32)
print(f"\nInitial weight shape: {w_init.shape}")
print(f"Initial weight norm: {np.linalg.norm(w_init):.6f}")

# Needle setup
needle_w = ndl.Tensor(w_init.copy(), device=ndl.cpu(), dtype="float32", requires_grad=True)
needle_opt = ndl.optim.Muon([needle_w], lr=0.02, momentum=0.95, nesterov=False)

# PyTorch setup with float32 Newton-Schulz
torch_w = torch.nn.Parameter(torch.from_numpy(w_init.copy()))

import torch.optim._muon as muon_module
def _zeropower_via_newtonschulz_float32(grad, ns_coefficients, ns_steps, eps):
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    a, b, c = ns_coefficients
    ortho_grad = grad.clone()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    ortho_grad = ortho_grad / ortho_grad.norm().clamp(min=eps)
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad

original_ns = muon_module._zeropower_via_newtonschulz
muon_module._zeropower_via_newtonschulz = _zeropower_via_newtonschulz_float32
torch_opt = torch.optim.Muon([torch_w], lr=0.02, momentum=0.95, nesterov=False)
muon_module._zeropower_via_newtonschulz = original_ns

# Create identical data
X_np = np.random.randn(8, 10).astype(np.float32)
y_np = np.random.randint(0, 20, size=(8,)).astype(np.int64)

X_needle = ndl.Tensor(X_np, device=ndl.cpu(), dtype="float32", requires_grad=False)
y_needle = ndl.Tensor(y_np.astype(np.float32), device=ndl.cpu(), dtype="float32", requires_grad=False)

X_torch = torch.from_numpy(X_np)
y_torch = torch.from_numpy(y_np)

print("\n" + "=" * 70)
print("STEP 1: Forward and Backward")
print("=" * 70)

# Needle forward/backward
needle_opt.reset_grad()
logits_needle = X_needle @ needle_w.transpose()
loss_needle = ndl.nn.SoftmaxLoss()(logits_needle, y_needle)
loss_needle.backward()

# PyTorch forward/backward
torch_opt.zero_grad()
logits_torch = X_torch @ torch_w.T
loss_torch = torch.nn.functional.cross_entropy(logits_torch, y_torch)
loss_torch.backward()

print(f"\nNeedle loss: {loss_needle.numpy().item():.6f}")
print(f"PyTorch loss: {loss_torch.item():.6f}")

# Compare gradients
grad_needle = needle_w.grad.numpy()
grad_torch = torch_w.grad.numpy()
grad_diff = np.abs(grad_needle - grad_torch)

print(f"\nGradient comparison:")
print(f"  Needle grad norm: {np.linalg.norm(grad_needle):.6f}")
print(f"  PyTorch grad norm: {np.linalg.norm(grad_torch):.6f}")
print(f"  Max grad diff: {np.max(grad_diff):.6e}")
print(f"  Mean grad diff: {np.mean(grad_diff):.6e}")

print("\n" + "=" * 70)
print("STEP 2: Momentum Buffer Update")
print("=" * 70)

# Manually compute what should happen in Needle
g_needle = needle_w.grad.data
momentum = 0.95

# First step: momentum buffer should be initialized to 0
print(f"\nNeedle gradient shape: {g_needle.shape}")
print(f"Needle gradient norm: {np.linalg.norm(g_needle.numpy()):.6f}")

# Compute momentum update manually for Needle
# buf = momentum * buf + (1 - momentum) * grad
# On first step, buf = 0, so buf = (1 - momentum) * grad
buf_needle_expected = (1.0 - momentum) * g_needle
print(f"\nExpected Needle momentum buffer norm: {np.linalg.norm(buf_needle_expected.numpy()):.6f}")

# PyTorch momentum buffer (check what it should be)
print(f"\nPyTorch gradient norm: {np.linalg.norm(grad_torch):.6f}")
buf_torch_expected = (1.0 - momentum) * grad_torch
print(f"Expected PyTorch momentum buffer norm: {np.linalg.norm(buf_torch_expected):.6f}")

print("\n" + "=" * 70)
print("STEP 3: Newton-Schulz Orthogonalization")
print("=" * 70)

# Apply Newton-Schulz to the momentum buffer
print(f"\nApplying Newton-Schulz to momentum buffer...")

# Needle NS
needle_ns_result = needle_opt._zeropower_via_newtonschulz5(buf_needle_expected)
print(f"Needle NS output norm: {np.linalg.norm(needle_ns_result.numpy()):.6f}")
print(f"Needle NS sample values: {needle_ns_result.numpy()[0, :5]}")

# PyTorch NS (manual)
torch_ns_result = _zeropower_via_newtonschulz_float32(
    torch.from_numpy(buf_torch_expected),
    (3.4445, -4.7750, 2.0315),
    5,
    1e-7
)
print(f"PyTorch NS output norm: {torch.norm(torch_ns_result).item():.6f}")
print(f"PyTorch NS sample values: {torch_ns_result[0, :5].numpy()}")

# Compare NS outputs
ns_diff = np.abs(needle_ns_result.numpy() - torch_ns_result.numpy())
print(f"\nNewton-Schulz difference:")
print(f"  Max diff: {np.max(ns_diff):.6e}")
print(f"  Mean diff: {np.mean(ns_diff):.6e}")

print("\n" + "=" * 70)
print("STEP 4: Parameter Update")
print("=" * 70)

lr = 0.02
needle_update = lr * needle_ns_result
torch_update = lr * torch_ns_result

print(f"\nNeedle update norm: {np.linalg.norm(needle_update.numpy()):.6f}")
print(f"PyTorch update norm: {np.linalg.norm(torch_update.numpy()):.6f}")

needle_w_new_expected = w_init - needle_update.numpy()
torch_w_new_expected = w_init - torch_update.numpy()

print(f"\nExpected new weight difference:")
print(f"  Max diff: {np.max(np.abs(needle_w_new_expected - torch_w_new_expected)):.6e}")

print("\n" + "=" * 70)
print("STEP 5: Actual Optimizer Step")
print("=" * 70)

w_before_needle = needle_w.numpy().copy()
w_before_torch = torch_w.detach().numpy().copy()

# Run actual optimizer steps
needle_opt.step()
torch_opt.step()

w_after_needle = needle_w.numpy()
w_after_torch = torch_w.detach().numpy()

actual_update_needle = w_after_needle - w_before_needle
actual_update_torch = w_after_torch - w_before_torch

print(f"\nActual Needle update norm: {np.linalg.norm(actual_update_needle):.6f}")
print(f"Actual PyTorch update norm: {np.linalg.norm(actual_update_torch):.6f}")

print(f"\nExpected vs Actual (Needle):")
expected_update_needle = -needle_update.numpy()
update_diff_needle = np.abs(actual_update_needle - expected_update_needle)
print(f"  Max diff: {np.max(update_diff_needle):.6e}")
print(f"  Update matches: {np.allclose(actual_update_needle, expected_update_needle, atol=1e-6)}")

print(f"\nExpected vs Actual (PyTorch):")
expected_update_torch = -torch_update.numpy()
update_diff_torch = np.abs(actual_update_torch - expected_update_torch)
print(f"  Max diff: {np.max(update_diff_torch):.6e}")
print(f"  Update matches: {np.allclose(actual_update_torch, expected_update_torch, atol=1e-6)}")

print(f"\nFinal weight difference (Needle vs PyTorch):")
final_diff = np.abs(w_after_needle - w_after_torch)
print(f"  Max diff: {np.max(final_diff):.6e}")
print(f"  Mean diff: {np.mean(final_diff):.6e}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if np.max(grad_diff) > 1e-6:
    print("\n⚠️  Gradients differ - check loss computation")
elif np.max(ns_diff) > 1e-5:
    print("\n⚠️  Newton-Schulz outputs differ - check NS implementation")
elif np.max(update_diff_needle) > 1e-5:
    print("\n⚠️  Needle update doesn't match expected - check optimizer step")
elif np.max(update_diff_torch) > 1e-5:
    print("\n⚠️  PyTorch update doesn't match expected - check PyTorch patching")
else:
    print("\n✓ Everything matches - implementation is correct!")
