"""Debug script to compare Newton-Schulz implementations"""
import sys
sys.path.append('./python')

import needle as ndl
import numpy as np

try:
    import torch
except ImportError:
    print("PyTorch not installed")
    sys.exit(1)

# Set seed
np.random.seed(42)
torch.manual_seed(42)

# Create a simple gradient matrix
G_np = np.random.randn(10, 20).astype(np.float32)

print("=" * 60)
print("Comparing Newton-Schulz Implementations")
print("=" * 60)

# Needle version
print("\n1. Needle Newton-Schulz:")
G_needle = ndl.Tensor(G_np, device=ndl.cpu(), requires_grad=False)
dummy_param = ndl.Tensor(np.random.randn(10, 20).astype(np.float32), device=ndl.cpu(), requires_grad=True)
needle_opt = ndl.optim.Muon([dummy_param], lr=0.01)

result_needle = needle_opt._zeropower_via_newtonschulz5(G_needle)
print(f"   Input norm: {np.linalg.norm(G_np):.6f}")
print(f"   Output norm: {np.linalg.norm(result_needle.numpy()):.6f}")
print(f"   Output sample values: {result_needle.numpy()[0, :5]}")

# PyTorch version (from their source)
print("\n2. PyTorch Newton-Schulz:")
def pytorch_ns(grad, eps=1e-7, ns_steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    ortho_grad = grad.clone().bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm is at most 1
    ortho_grad = ortho_grad / ortho_grad.norm().clamp(min=eps)
    # Perform the NS iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad.float()

G_torch = torch.from_numpy(G_np)
result_torch = pytorch_ns(G_torch)
print(f"   Input norm: {torch.norm(G_torch).item():.6f}")
print(f"   Output norm: {torch.norm(result_torch).item():.6f}")
print(f"   Output sample values: {result_torch[0, :5].numpy()}")

# Compare
print("\n3. Comparison:")
diff = np.abs(result_needle.numpy() - result_torch.numpy())
print(f"   Max difference: {np.max(diff):.6e}")
print(f"   Mean difference: {np.mean(diff):.6e}")
print(f"   Relative error: {np.max(diff) / np.abs(result_torch.numpy()).max():.6e}")

# Check if bfloat16 is the issue
print("\n4. Testing float32 PyTorch (no bfloat16):")
def pytorch_ns_float32(grad, eps=1e-7, ns_steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    ortho_grad = grad.clone()  # Keep float32
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

result_torch_f32 = pytorch_ns_float32(G_torch)
print(f"   Output norm: {torch.norm(result_torch_f32).item():.6f}")
print(f"   Output sample values: {result_torch_f32[0, :5].numpy()}")

diff_f32 = np.abs(result_needle.numpy() - result_torch_f32.numpy())
print(f"   Max difference vs Needle: {np.max(diff_f32):.6e}")
print(f"   Relative error: {np.max(diff_f32) / np.abs(result_torch_f32.numpy()).max():.6e}")
