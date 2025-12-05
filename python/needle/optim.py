"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay
    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
          if p not in self.u:
            self.u[p] = 0
          u = self.momentum * self.u[p] + (1 - self.momentum) * (p.grad.data + self.weight_decay * p.data)
          p.data = p.data - self.lr * u
          self.u[p] = u
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
          if p not in self.m:
            self.m[p] = 0
          if p not in self.v:
            self.v[p] = 0

          grad = p.grad.data + self.weight_decay * p.data
          m = self.beta1 * self.m[p] + (1 - self.beta1) * grad
          v = self.beta2 * self.v[p] + (1 - self.beta2) * grad ** 2

          ubm = m / (1 - self.beta1 ** self.t)
          ubv = v / (1 - self.beta2 ** self.t)

          update = (p.data - self.lr * ubm / (ubv ** 0.5 + self.eps)).data
          p.data = type(p)(update, dtype=p.dtype, device=p.device)
          self.m[p] = m.data
          self.v[p] = v.data
        ### END YOUR SOLUTION


# class Muon(Optimizer):
#     """
#     Muon optimizer - Momentum with Orthogonalization and Newton steps.

#     Based on the paper: "Muon: Momentum with Orthogonalization for Neural Optimization"
#     Key idea: Combines momentum with gradient orthogonalization to improve conditioning.

#     Reference implementation combines:
#     - Momentum-based updates
#     - Gradient orthogonalization (similar to normalized gradient descent)
#     - Adaptive learning rates
#     """
#     def __init__(
#         self,
#         params,
#         lr=0.02,
#         momentum=0.95,
#         nesterov=True,
#         weight_decay=0.0,
#     ):
#         super().__init__(params)
#         self.lr = lr
#         self.momentum = momentum
#         self.nesterov = nesterov
#         self.weight_decay = weight_decay
#         self.t = 0

#         # Momentum buffer
#         self.m = {}

#     def step(self):
#         self.t += 1

#         for p in self.params:
#             if p.grad is None:
#                 continue

#             # Initialize momentum buffer
#             if p not in self.m:
#                 self.m[p] = 0

#             # Add weight decay to gradient
#             grad = p.grad.data + self.weight_decay * p.data

#             # Normalize gradient (orthogonalization step)
#             # This is a key component of Muon - gradient is normalized by its norm
#             grad_norm = ((grad ** 2).sum() ** 0.5).numpy().item()
#             grad_normalized = grad / (grad_norm + 1e-8)

#             # Momentum update with normalized gradient
#             m = self.momentum * self.m[p] + grad_normalized

#             # Nesterov momentum (look-ahead)
#             if self.nesterov:
#                 update_direction = self.momentum * m + grad_normalized
#             else:
#                 update_direction = m

#             # Apply update
#             update = (p.data - self.lr * update_direction).data
#             p.data = type(p)(update, dtype=p.dtype, device=p.device)
#             self.m[p] = m.data

class Muon(Optimizer):
    """
    Muon optimizer - Momentum with Orthogonalization via Newton-Schulz iteration.

    Based on the reference implementation from:
    https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench94_muon.py

    Key components:
    1. Weight normalization before update
    2. Newton-Schulz iteration for gradient orthogonalization
    3. Momentum-based updates
    """
    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=False,
        ns_steps=5,
        eps=1e-7,
        weight_decay=0,
    ):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps  # Newton-Schulz iteration steps
        self.eps = eps
        self.t = 0

        # Momentum buffer
        self.m = {}

    def _zeropower_via_newtonschulz5(self, G):
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

        This iteration computes an approximation to G @ (G^T @ G)^{-1/2}, which is
        the orthogonalization of G (analogous to computing UV^T from SVD G = USV^T).

        Args:
            G: Tensor of shape (m, n) representing gradient

        Returns:
            Orthogonalized gradient of same shape as G
        """
        import needle as ndl

        assert len(G.shape) == 2, "Newton-Schulz iteration requires 2D tensor"

        # Coefficients optimized for convergence
        a, b, c = (3.4445, -4.7750, 2.0315)

        # Start with normalized G
        X = G
        # Normalize by Frobenius norm to ensure top singular value <= 1
        norm = ((X ** 2).sum() ** 0.5).numpy().item()
        X = X / (norm + self.eps)

        # Transpose if needed (work with smaller matrix)
        if G.shape[0] > G.shape[1]:
            X = X.transpose()

        # Newton-Schulz iterations
        for _ in range(self.ns_steps):
            # A = X @ X^T
            A = X @ X.transpose()
            # B = b*A + c*A^2
            B = b * A + c * (A @ A)
            # X = a*X + B @ X
            X = a * X + B @ X

        # Transpose back if needed
        if G.shape[0] > G.shape[1]:
            X = X.transpose()

        return X

    def step(self):
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue

            # Get gradient
            g = p.grad.data

            # Initialize momentum buffer
            if p not in self.m:
                self.m[p] = 0 * g

            # Momentum update: buf = buf * momentum + grad * (1 - momentum)
            # This is equivalent to PyTorch's buf.lerp_(grad, 1 - momentum)
            buf = self.m[p]
            buf = self.momentum * buf + (1.0 - self.momentum) * g

            # Nesterov momentum
            if self.nesterov:
                # update = grad * (1 - momentum) + buf * momentum
                update_grad = (1.0 - self.momentum) * g + self.momentum * buf
            else:
                update_grad = buf

            # Only apply Newton-Schulz to 2D parameters (weights)
            # For biases and other 1D params, use the gradient as-is
            if len(p.shape) == 2:
                # Orthogonalize the gradient using Newton-Schulz
                update_grad_2d = update_grad.reshape((p.shape[0], -1))
                update_orthogonal = self._zeropower_via_newtonschulz5(update_grad_2d)
                update_orthogonal = update_orthogonal.reshape(p.shape)

                # Apply update (no weight normalization in PyTorch version!)
                new_w = p.data - self.lr * update_orthogonal
            else:
                # For 1D parameters (biases), just do standard update
                new_w = p.data - self.lr * update_grad

            # Write back to parameter
            p.data = type(p)(new_w, dtype=p.dtype, device=p.device)

            # Store momentum (must store .data to prevent gradient accumulation)
            self.m[p] = buf.data





# class SOAP(Optimizer):
#     """
#     SOAP optimizer - Shampoo-like Orthogonal Adaptive Preconditioning.

#     Based on Shampoo and adaptive preconditioning ideas:
#     - Maintains second-moment statistics like Adam
#     - Uses element-wise preconditioning
#     - Incorporates momentum

#     Simplified version that combines Adam-like adaptivity with momentum.
#     """
#     def __init__(
#         self,
#         params,
#         lr=0.001,
#         beta1=0.9,
#         beta2=0.999,
#         eps=1e-8,
#         weight_decay=0.0,
#         precondition_frequency=10,
#     ):
#         super().__init__(params)
#         self.lr = lr
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.eps = eps
#         self.weight_decay = weight_decay
#         self.precondition_frequency = precondition_frequency
#         self.t = 0

#         # First and second moment estimates
#         self.m = {}
#         self.v = {}
#         # Preconditioning matrix (simplified - just use diagonal)
#         self.precond = {}

#     def step(self):
#         self.t += 1

#         for p in self.params:
#             if p.grad is None:
#                 continue

#             # Initialize buffers
#             if p not in self.m:
#                 self.m[p] = 0
#             if p not in self.v:
#                 self.v[p] = 0
#             if p not in self.precond:
#                 self.precond[p] = 1.0

#             # Add weight decay
#             grad = p.grad.data + self.weight_decay * p.data

#             # Update biased first and second moment estimates
#             m = self.beta1 * self.m[p] + (1 - self.beta1) * grad
#             v = self.beta2 * self.v[p] + (1 - self.beta2) * grad ** 2

#             # Bias correction
#             m_hat = m / (1 - self.beta1 ** self.t)
#             v_hat = v / (1 - self.beta2 ** self.t)

#             # Update preconditioning (periodically)
#             if self.t % self.precondition_frequency == 0:
#                 # Simplified preconditioning: use running average of gradient magnitudes
#                 precond = v_hat ** 0.5 + self.eps
#                 self.precond[p] = precond.data

#             # Apply preconditioned update
#             # SOAP uses adaptive preconditioning based on second moments
#             precond_grad = m_hat / (self.precond[p] + self.eps)

#             # Apply update
#             update = (p.data - self.lr * precond_grad).data
#             p.data = type(p)(update, dtype=p.dtype, device=p.device)

#             # Store moments
#             self.m[p] = m.data
#             self.v[p] = v.data

class SOAP(Optimizer):
    """
    SOAP-like optimizer (Shampoo / Adafactor style) for Needle.

    Design:
    - Adam-like first-moment (m) tracking
    - Factored second-moment statistics for matrix-shaped parameters:
        * row_v: per-row squared-grad averages
        * col_v: per-column squared-grad averages
      inspired by Shampoo/Adafactor.
    - For non-matrix params (biases, LayerNorm weights, etc.), falls back
      to standard Adam-style v.

    This is not the full SOAP (which runs Adam in the eigenbasis of a
    Shampoo preconditioner), but it captures the big idea: use structured
    second-order information along rows/cols instead of purely diagonal v.
    """
    def __init__(
        self,
        params,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        # First moment
        self.m = {}
        # Diagonal second moment fallback
        self.v = {}
        # Factored second moments for matrix-shaped params
        self.row_v = {}
        self.col_v = {}

    def step(self):
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data + self.weight_decay * p.data

            # Initialize buffers
            if p not in self.m:
                self.m[p] = 0 * grad
            if p not in self.v:
                self.v[p] = 0 * grad

            m_prev = self.m[p]
            # First moment update
            m = self.beta1 * m_prev + (1.0 - self.beta1) * grad

            # Bias correction for m
            m_hat = m / (1.0 - self.beta1 ** self.t)

            # Matrix-shaped parameters: use factored second moments
            shape = grad.shape
            if len(shape) == 2:
                out_dim, in_dim = shape

                # Lazy init for row/col stats
                if p not in self.row_v:
                    # zeros with correct shapes
                    self.row_v[p] = (grad.sum(axes=1) * 0)  # (out_dim,)
                    self.col_v[p] = (grad.sum(axes=0) * 0)  # (in_dim,)

                g2 = grad ** 2

                # Update factored second moments
                row_v_prev = self.row_v[p]
                col_v_prev = self.col_v[p]

                row_v = self.beta2 * row_v_prev + (1.0 - self.beta2) * g2.sum(axes=1)
                col_v = self.beta2 * col_v_prev + (1.0 - self.beta2) * g2.sum(axes=0)

                self.row_v[p] = row_v
                self.col_v[p] = col_v

                # Bias correction for factored stats
                row_v_hat = row_v / (1.0 - self.beta2 ** self.t)
                col_v_hat = col_v / (1.0 - self.beta2 ** self.t)

                # Approximate per-element RMS with factored stats (Adafactor-style)
                # row_rms ~ sqrt(row_v / in_dim), col_rms ~ sqrt(col_v / out_dim)
                # Bias correction for factored stats
                row_v_hat = row_v / (1.0 - self.beta2 ** self.t)
                col_v_hat = col_v / (1.0 - self.beta2 ** self.t)

                # Adafactor-style RMS
                row_rms = (row_v_hat / float(in_dim)) ** 0.5   # (out_dim,)
                col_rms = (col_v_hat / float(out_dim)) ** 0.5  # (in_dim,)

                # Reshape for broadcasting, then explicitly broadcast
                row_rms = row_rms.reshape((out_dim, 1))        # (out_dim, 1)
                col_rms = col_rms.reshape((1, in_dim))        # (1, in_dim)

                row_mat = row_rms.broadcast_to((out_dim, in_dim))  # (out_dim, in_dim)
                col_mat = col_rms.broadcast_to((out_dim, in_dim))  # (out_dim, in_dim)

                denom = row_mat * col_mat + self.eps           # (out_dim, in_dim)

                # Preconditioned update
                precond_grad = m_hat / denom                   # same shape as grad

                # Preconditioned update (Adam in factored preconditioner's "shape")
                precond_grad = m_hat / denom

            else:
                # Fallback: standard Adam diagonal v
                v_prev = self.v[p]
                v = self.beta2 * v_prev + (1.0 - self.beta2) * (grad ** 2)
                self.v[p] = v

                v_hat = v / (1.0 - self.beta2 ** self.t)
                precond_grad = m_hat / (v_hat ** 0.5 + self.eps)

            # Apply update
            new_data = p.data - self.lr * precond_grad
            p.data = type(p)(new_data, dtype=p.dtype, device=p.device)

            # Store first moment
            self.m[p] = m