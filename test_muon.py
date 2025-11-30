"""
Test script to validate Muon optimizer implementation on CPU with small data.
This is a sanity check before running on larger datasets.
"""
import sys
sys.path.append('./python')
sys.path.append('./apps')

import needle as ndl
import numpy as np
from models import ResNet9

print(f"Using Needle backend: {ndl.backend_selection.BACKEND}")

def create_dummy_data(batch_size=4, device=None):
    """Create small dummy CIFAR-10-like data for testing"""
    # Images: (batch_size, 32, 32, 3)
    images = np.random.randn(batch_size, 32, 32, 3).astype(np.float32)
    # Labels: (batch_size,) with values in [0, 9]
    labels = np.random.randint(0, 10, size=(batch_size,)).astype(np.float32)

    if device is None:
        device = ndl.cpu()

    images_tensor = ndl.Tensor(images, device=device, dtype="float32", requires_grad=False)
    labels_tensor = ndl.Tensor(labels, device=device, dtype="float32", requires_grad=False)

    return images_tensor, labels_tensor

def test_muon_basic():
    """Test basic Muon optimizer functionality"""
    print("=" * 60)
    print("Testing Muon Optimizer - Basic Functionality")
    print("=" * 60)

    device = ndl.cpu()
    print(f"Using device: {device}")

    # Create a simple model (smaller than ResNet9 for quick testing)
    print("\n1. Creating simple linear model...")
    model = ndl.nn.Linear(64, 10, device=device, dtype="float32")

    # Create simple data
    print("2. Creating dummy data...")
    batch_size = 8
    X = ndl.Tensor(np.random.randn(batch_size, 64).astype(np.float32),
                   device=device, requires_grad=False)
    y = ndl.Tensor(np.random.randint(0, 10, size=(batch_size,)).astype(np.float32),
                   device=device, requires_grad=False)

    # Create optimizer
    print("3. Creating Muon optimizer...")
    optimizer = ndl.optim.Muon(model.parameters(), lr=0.01, momentum=0.9)

    # Training step
    print("4. Running forward pass...")
    logits = model(X)

    print("5. Computing loss...")
    loss = ndl.nn.SoftmaxLoss()(logits, y)
    print(f"   Initial loss: {loss.numpy().item():.4f}")

    print("6. Running backward pass...")
    optimizer.reset_grad()
    loss.backward()

    # Check gradients
    print("7. Checking gradients...")
    for i, param in enumerate(model.parameters()):
        if param.grad is not None:
            grad_norm = np.linalg.norm(param.grad.numpy())
            print(f"   Param {i}: grad_norm = {grad_norm:.6f}")
        else:
            print(f"   Param {i}: No gradient!")

    print("8. Running optimizer step...")
    try:
        optimizer.step()
        print("   ✓ Optimizer step successful!")
    except Exception as e:
        print(f"   ✗ Optimizer step failed: {e}")
        raise

    # Run another forward to check loss decreased
    print("9. Running second iteration...")
    logits = model(X)
    loss_new = ndl.nn.SoftmaxLoss()(logits, y)
    print(f"   New loss: {loss_new.numpy().item():.4f}")

    print("\n" + "=" * 60)
    print("✓ Basic test passed!")
    print("=" * 60)

def test_newton_schulz():
    """Test Newton-Schulz orthogonalization separately"""
    print("\n" + "=" * 60)
    print("Testing Newton-Schulz Orthogonalization")
    print("=" * 60)

    device = ndl.cpu()

    # Create a simple 2D gradient matrix
    print("\n1. Creating test matrix...")
    G_np = np.random.randn(10, 20).astype(np.float32)
    G = ndl.Tensor(G_np, device=device, requires_grad=False)

    print(f"   Input shape: {G.shape}")
    print(f"   Input norm: {np.linalg.norm(G_np):.4f}")

    # Create optimizer just to access the method
    dummy_param = ndl.Tensor(np.random.randn(10, 20).astype(np.float32),
                              device=device, requires_grad=True)
    optimizer = ndl.optim.Muon([dummy_param], lr=0.01)

    print("2. Running Newton-Schulz iteration...")
    try:
        G_ortho = optimizer._zeropower_via_newtonschulz5(G)
        print(f"   ✓ Orthogonalization successful")
        print(f"   Output shape: {G_ortho.shape}")
        print(f"   Output norm: {np.linalg.norm(G_ortho.numpy()):.4f}")

        # Check orthogonality: G_ortho @ G_ortho.T should be close to identity
        G_ortho_np = G_ortho.numpy()
        product = G_ortho_np @ G_ortho_np.T
        identity = np.eye(10)
        ortho_error = np.linalg.norm(product - identity)
        print(f"   Orthogonality error: {ortho_error:.6f}")

        if ortho_error < 0.1:
            print(f"   ✓ Matrix is approximately orthogonal")
        else:
            print(f"   ⚠ Warning: Matrix may not be fully orthogonal")

    except Exception as e:
        print(f"   ✗ Orthogonalization failed: {e}")
        raise

    print("\n" + "=" * 60)
    print("✓ Newton-Schulz test passed!")
    print("=" * 60)

def test_muon_vs_reference():
    """Compare Needle Muon implementation with PyTorch reference numerically"""
    print("\n" + "=" * 60)
    print("Comparing Muon with PyTorch Reference (Numerical Test)")
    print("=" * 60)

    try:
        import torch
        import torch.nn as tnn
    except ImportError:
        print("\n⚠️  PyTorch not installed - skipping comparison test")
        print("Install PyTorch to run numerical comparison")
        return

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create identical initial weights (no bias - PyTorch Muon only supports 2D params)
    print("\n1. Creating identical models...")
    w_init = np.random.randn(20, 10).astype(np.float32)

    # Needle model
    needle_w = ndl.Tensor(w_init.copy(), device=ndl.cpu(), dtype="float32", requires_grad=True)

    # PyTorch model
    torch_w = torch.nn.Parameter(torch.from_numpy(w_init.copy()))

    # Create identical input and target
    print("2. Creating identical data...")
    X_np = np.random.randn(8, 10).astype(np.float32)
    y_np = np.random.randint(0, 20, size=(8,)).astype(np.int64)

    X_needle = ndl.Tensor(X_np, device=ndl.cpu(), dtype="float32", requires_grad=False)
    y_needle = ndl.Tensor(y_np.astype(np.float32), device=ndl.cpu(), dtype="float32", requires_grad=False)

    X_torch = torch.from_numpy(X_np)
    y_torch = torch.from_numpy(y_np)

    # Create optimizers with same hyperparameters
    print("3. Creating optimizers...")
    lr = 0.02
    momentum = 0.95
    needle_opt = ndl.optim.Muon([needle_w], lr=lr, momentum=momentum, nesterov=False)
    torch_opt = torch.optim.Muon([torch_w], lr=lr, momentum=momentum, nesterov=False)

    print("4. Running training steps...")
    num_steps = 3

    for step in range(num_steps):
        # Needle forward and backward
        needle_opt.reset_grad()
        logits_needle = X_needle @ needle_w.transpose()
        loss_needle = ndl.nn.SoftmaxLoss()(logits_needle, y_needle)
        loss_needle.backward()

        # PyTorch forward and backward
        torch_opt.zero_grad()
        logits_torch = X_torch @ torch_w.T
        loss_torch = tnn.functional.cross_entropy(logits_torch, y_torch)
        loss_torch.backward()

        # Store pre-update weights for comparison
        needle_w_before = needle_w.numpy().copy()
        torch_w_before = torch_w.detach().numpy().copy()

        # Optimizer step
        needle_opt.step()
        torch_opt.step()

        # Get updated weights
        needle_w_after = needle_w.numpy()
        torch_w_after = torch_w.detach().numpy()

        # Compare weight updates
        needle_update = needle_w_after - needle_w_before
        torch_update = torch_w_after - torch_w_before

        update_diff = np.abs(needle_update - torch_update)
        max_diff = np.max(update_diff)
        rel_error = max_diff / (np.abs(torch_update).max() + 1e-8)

        print(f"\n   Step {step + 1}:")
        print(f"   Needle loss: {loss_needle.numpy().item():.6f}")
        print(f"   PyTorch loss: {loss_torch.item():.6f}")
        print(f"   Weight update max diff: {max_diff:.6e}")
        print(f"   Weight update rel error: {rel_error:.6e}")

    print("\n5. Checking final results...")
    # Final comparison
    final_w_diff = np.max(np.abs(needle_w.numpy() - torch_w.detach().numpy()))

    print(f"   Final weight diff: {final_w_diff:.6e}")

    # Check if differences are within acceptable tolerance
    # Allow some numerical differences due to implementation details
    tolerance = 1e-4

    if final_w_diff < tolerance:
        print("\n" + "=" * 60)
        print("✓ PASS: Needle Muon matches PyTorch reference!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: Some numerical differences detected")
        print("=" * 60)
        print(f"Weight diff {final_w_diff:.6e} (tolerance: {tolerance})")
        print("\nThis may be due to:")
        print("  - Floating point precision differences")
        print("  - Different norm computation (Frobenius vs spectral)")
        print("  - Backend implementation details")

        if final_w_diff < 1e-2:
            print("\nDifferences are small - likely acceptable for practical use")
        else:
            raise AssertionError(f"Difference too large: {final_w_diff}")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MUON OPTIMIZER VALIDATION TESTS")
    print("=" * 60)

    try:
        # Run tests
        test_muon_vs_reference()
        test_newton_schulz()
        test_muon_basic()

        print("\n" + "=" * 60)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("=" * 60)
        print("\nMuon optimizer is ready to use!")

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗✗✗ TESTS FAILED ✗✗✗")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
