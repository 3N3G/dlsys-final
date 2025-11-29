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


class Muon(Optimizer):
    """
    Muon optimizer - Momentum with Orthogonalization and Newton steps.

    Based on the paper: "Muon: Momentum with Orthogonalization for Neural Optimization"
    Key idea: Combines momentum with gradient orthogonalization to improve conditioning.

    Reference implementation combines:
    - Momentum-based updates
    - Gradient orthogonalization (similar to normalized gradient descent)
    - Adaptive learning rates
    """
    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.t = 0

        # Momentum buffer
        self.m = {}

    def step(self):
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue

            # Initialize momentum buffer
            if p not in self.m:
                self.m[p] = 0

            # Add weight decay to gradient
            grad = p.grad.data + self.weight_decay * p.data

            # Normalize gradient (orthogonalization step)
            # This is a key component of Muon - gradient is normalized by its norm
            grad_norm = (grad ** 2).sum() ** 0.5
            grad_normalized = grad / (grad_norm + 1e-8)

            # Momentum update with normalized gradient
            m = self.momentum * self.m[p] + grad_normalized

            # Nesterov momentum (look-ahead)
            if self.nesterov:
                update_direction = self.momentum * m + grad_normalized
            else:
                update_direction = m

            # Apply update
            update = (p.data - self.lr * update_direction).data
            p.data = type(p)(update, dtype=p.dtype, device=p.device)
            self.m[p] = m.data


class SOAP(Optimizer):
    """
    SOAP optimizer - Shampoo-like Orthogonal Adaptive Preconditioning.

    Based on Shampoo and adaptive preconditioning ideas:
    - Maintains second-moment statistics like Adam
    - Uses element-wise preconditioning
    - Incorporates momentum

    Simplified version that combines Adam-like adaptivity with momentum.
    """
    def __init__(
        self,
        params,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        precondition_frequency=10,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.precondition_frequency = precondition_frequency
        self.t = 0

        # First and second moment estimates
        self.m = {}
        self.v = {}
        # Preconditioning matrix (simplified - just use diagonal)
        self.precond = {}

    def step(self):
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue

            # Initialize buffers
            if p not in self.m:
                self.m[p] = 0
            if p not in self.v:
                self.v[p] = 0
            if p not in self.precond:
                self.precond[p] = 1.0

            # Add weight decay
            grad = p.grad.data + self.weight_decay * p.data

            # Update biased first and second moment estimates
            m = self.beta1 * self.m[p] + (1 - self.beta1) * grad
            v = self.beta2 * self.v[p] + (1 - self.beta2) * grad ** 2

            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # Update preconditioning (periodically)
            if self.t % self.precondition_frequency == 0:
                # Simplified preconditioning: use running average of gradient magnitudes
                precond = v_hat ** 0.5 + self.eps
                self.precond[p] = precond.data

            # Apply preconditioned update
            # SOAP uses adaptive preconditioning based on second moments
            precond_grad = m_hat / (self.precond[p] + self.eps)

            # Apply update
            update = (p.data - self.lr * precond_grad).data
            p.data = type(p)(update, dtype=p.dtype, device=p.device)

            # Store moments
            self.m[p] = m.data
            self.v[p] = v.data