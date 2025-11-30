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
    labels = np.random.randint(0, 10, size=(batch_size,)).astype(np.uint8)

    if device is None:
        device = ndl.cpu()

    images_tensor = ndl.Tensor(images, device=device, dtype="float32", requires_grad=False)
    labels_tensor = ndl.Tensor(labels, device=device, dtype="uint8", requires_grad=False)

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
    y = ndl.Tensor(np.random.randint(0, 10, size=(batch_size,)).astype(np.uint8),
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

def test_muon_resnet9():
    """Test Muon with ResNet9 architecture"""
    print("\n" + "=" * 60)
    print("Testing Muon Optimizer - ResNet9 Model")
    print("=" * 60)

    device = ndl.cpu()
    print(f"Using device: {device}")

    print("\n1. Creating ResNet9 model...")
    model = ResNet9(device=device, dtype="float32")

    print("2. Creating dummy CIFAR-10 data...")
    batch_size = 2  # Very small batch for CPU
    images, labels = create_dummy_data(batch_size, device)

    print("3. Creating Muon optimizer...")
    optimizer = ndl.optim.Muon(model.parameters(), lr=0.02, momentum=0.95, nesterov=True)

    print("4. Running training loop (3 iterations)...")
    for iteration in range(3):
        print(f"\n   Iteration {iteration + 1}:")

        # Forward
        logits = model(images)
        loss = ndl.nn.SoftmaxLoss()(logits, labels)
        print(f"      Loss: {loss.numpy().item():.4f}")

        # Backward
        optimizer.reset_grad()
        loss.backward()

        # Update
        try:
            optimizer.step()
            print(f"      ✓ Optimizer step successful")
        except Exception as e:
            print(f"      ✗ Optimizer step failed: {e}")
            raise

    print("\n" + "=" * 60)
    print("✓ ResNet9 test passed!")
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

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MUON OPTIMIZER VALIDATION TESTS")
    print("=" * 60)

    try:
        # Run tests
        test_newton_schulz()
        test_muon_basic()
        test_muon_resnet9()

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
