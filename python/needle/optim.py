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
        for p in self.params:
            if p not in self.u:
                self.u[p] = 0
            u = self.momentum * self.u[p] + (1 - self.momentum) * (p.grad.data + self.weight_decay * p.data)
            p.data = p.data - self.lr * u
            self.u[p] = u

    def clip_grad_norm(self, max_norm=0.25):
        raise NotImplementedError()


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
        self.t += 1
        for p in self.params:
            if p not in self.m:
                self.m[p] = 0
            if p not in self.v:
                self.v[p] = 0

            grad = p.grad.data + self.weight_decay * p.data
            m = self.beta1 * self.m[p] + (1 - self.beta1) * grad
            v = self.beta2 * self.v[p] + (1 - self.beta2) * grad ** 2

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            update = (p.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)).data
            p.data = type(p)(update, dtype=p.dtype, device=p.device)
            self.m[p] = m.data
            self.v[p] = v.data


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
        muon_lr=0.1,
        sgd_lr=0.001,
        momentum=0.95,
        nesterov=False,
        ns_steps=5,
        eps=1e-7,
        weight_decay=0,
        total_steps=None,
    ):
        super().__init__(params)
        self.muon_lr = muon_lr
        self.sgd_lr = sgd_lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps  # Newton-Schulz iteration steps
        self.eps = eps
        self.weight_decay = weight_decay

        self.total_steps = total_steps
        self.t = 0
        self.m = {}

    def _current_lrs(self):
        """Linear decay: lr_t = lr_0 * (1 - t / total_steps)"""
        if self.total_steps is None or self.total_steps <= 0:
            return self.muon_lr, self.sgd_lr

        decay = max(0.0, 1.0 - self.t / float(self.total_steps))
        cur_muon_lr = self.muon_lr * decay
        cur_sgd_lr = self.sgd_lr * decay
        return cur_muon_lr, cur_sgd_lr

    def _zeropower_via_newtonschulz5(self, G):
        """
        Newton-Schulz iteration for gradient orthogonalization.
        Computes G @ (G^T @ G)^{-1/2}.

        G: Tensor of shape (m, n)
        """
        import needle as ndl

        assert len(G.shape) == 2, "Newton-Schulz iteration requires 2D tensor"

        a, b, c = (3.4445, -4.7750, 2.0315)

        X = G
        norm = ((X ** 2).sum() ** 0.5).numpy().item()
        X = X / (norm + self.eps)

        if G.shape[0] > G.shape[1]:
            X = X.transpose()

        for _ in range(self.ns_steps):
            A = X @ X.transpose()
            B = b * A + c * (A @ A)
            X = a * X + B @ X

        if G.shape[0] > G.shape[1]:
            X = X.transpose()

        return X

    def step(self):
        self.t += 1
        cur_muon_lr, cur_sgd_lr = self._current_lrs()

        for p in self.params:
            if p.grad is None:
                continue

            g = p.grad.data
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            if p not in self.m:
                self.m[p] = 0 * g

            buf = self.m[p]
            buf = self.momentum * buf + (1.0 - self.momentum) * g

            if self.nesterov:
                update_grad = (1.0 - self.momentum) * g + self.momentum * buf
            else:
                update_grad = buf

            w = p.data

            if len(p.shape) >= 2:
                rows = p.shape[0]
                w_norm = ((w ** 2).sum()) ** 0.5
                w_norm_val = w_norm.numpy().item()
                scale = (rows ** 0.5) * ((w_norm_val + self.eps)**-1)
                w = w * scale

                G2d = update_grad.reshape((update_grad.shape[0], -1))
                G2d_orth = self._zeropower_via_newtonschulz5(G2d)
                update = G2d_orth.reshape(w.shape)

                new_w = w - cur_muon_lr * update
            else:
                new_w = w - cur_sgd_lr * update_grad

            p.data = type(p)(new_w, dtype=p.dtype, device=p.device)
            self.m[p] = buf.data

class SOAP(Optimizer):
    """
    SOAP optimizer: Shampoo + Adam in the preconditioner's eigenbasis,
    ported to Needle.

    This is a simplified port of the PyTorch implementation you pasted.

    - Maintains:
        * exp_avg: first moment (Adam)
        * exp_avg_sq: second moment (Adam)
        * GG[p]: list of preconditioner matrices (one per dimension)
        * Q[p]:   list of eigenbases for GG[p]
    - For each param p:
        1. Project grad into the Shampoo eigenbasis (via Q).
        2. Run Adam in that basis.
        3. Project the preconditioned update back.
        4. Apply decoupled weight decay.

    Implementation notes:
    - All preconditioner math (GG, Q, eigendecomp, tensordot) is done in NumPy.
    - Works best with relatively small layers (CIFAR/ResNet9 level).
    - We ignore `merge_dims` / `channels_last` tricks for now and assume
      `merge_dims=False`, `data_format="channels_first"`.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
        total_steps=None,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        # If shampoo_beta < 0, fall back to beta2 (same as PyTorch code)
        self.shampoo_beta = shampoo_beta if shampoo_beta >= 0 else self.beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.precondition_frequency = precondition_frequency
        self.max_precond_dim = max_precond_dim
        self.merge_dims = merge_dims
        self.precondition_1d = precondition_1d
        self.normalize_grads = normalize_grads
        self.data_format = data_format
        self.correct_bias = correct_bias

        self.total_steps = total_steps
        self.t = 0

        if self.merge_dims:
            raise NotImplementedError("merge_dims=True not yet supported in Needle SOAP.")

        self.step_count = {}
        self.exp_avg = {}
        self.exp_avg_sq = {}
        self.GG = {}
        self.Q = {}

    def _current_lr(self):
        """Linear decay: lr_t = lr_0 * (1 - t / total_steps)"""
        if self.total_steps is None or self.total_steps <= 0:
            return self.lr

        decay = max(0.0, 1.0 - self.t / float(self.total_steps))
        lr = self.lr * decay
        return lr 

    def _to_numpy(self, tensor):
        return tensor.numpy()

    def _from_numpy(self, arr, like_tensor):
        return ndl.Tensor(
            arr,
            device=like_tensor.device,
            dtype=like_tensor.dtype,
            requires_grad=False,
        )

    def _init_preconditioner(self, grad_np, p):
        """Initialize GG[p] and Q[p] for gradient ndarray."""
        GG_list = []
        if grad_np.ndim == 1:
            if not self.precondition_1d or grad_np.shape[0] > self.max_precond_dim:
                GG_list.append([])
            else:
                d = grad_np.shape[0]
                GG_list.append(np.zeros((d, d), dtype=grad_np.dtype))
        else:
            for sh in grad_np.shape:
                if sh > self.max_precond_dim:
                    GG_list.append([])
                else:
                    GG_list.append(np.zeros((sh, sh), dtype=grad_np.dtype))

        self.GG[p] = GG_list
        self.Q[p] = None

    def _compute_eigenbases(self, p):
        """Compute eigenbases Q[p] for each GG[p][k] via np.linalg.eigh."""
        GG_list = self.GG[p]
        Q_list = []
        for m in GG_list:
            if isinstance(m, list) or (isinstance(m, np.ndarray) and m.size == 0):
                Q_list.append([])
                continue
            if m.shape[0] == 0:
                Q_list.append([])
                continue

            eye = np.eye(m.shape[0], dtype=m.dtype)
            try:
                w, Q = np.linalg.eigh(m + 1e-30 * eye)
            except np.linalg.LinAlgError:
                w, Q = np.linalg.eigh(m.astype(np.float64) + 1e-30 * np.eye(m.shape[0]))
                Q = Q.astype(m.dtype)

            Q = np.flip(Q, axis=1)
            Q_list.append(Q)

        self.Q[p] = Q_list

    def _update_preconditioner(self, grad_np, p):
        """Update GG[p] using current gradient."""
        GG_list = self.GG[p]
        if grad_np.ndim == 1:
            if self.precondition_1d and grad_np.shape[0] <= self.max_precond_dim:
                g = grad_np[:, None]
                outer = g @ g.T
                beta = self.shampoo_beta
                GG_list[0] = beta * GG_list[0] + (1.0 - beta) * outer
        else:
            shape = grad_np.shape
            n_dims = len(shape)
            for idx, sh in enumerate(shape):
                if sh > self.max_precond_dim or len(GG_list[idx]) == 0:
                    continue
                axes = list(range(n_dims))
                axes.remove(idx)
                outer = np.tensordot(
                    grad_np,
                    grad_np,
                    axes=(axes, axes),
                )
                beta = self.shampoo_beta
                GG_list[idx] = beta * GG_list[idx] + (1.0 - beta) * outer

        self.GG[p] = GG_list

    def _project(self, grad_np, p):
        """Project gradient into Shampoo eigenbases Q[p]."""
        Q_list = self.Q[p]
        g = grad_np

        for Qk in Q_list:
            if isinstance(Qk, list) or (isinstance(Qk, np.ndarray) and Qk.size == 0):
                axes = list(range(g.ndim))
                if g.ndim > 1:
                    g = np.transpose(g, axes[1:] + axes[:1])
                continue

            g = np.tensordot(g, Qk, axes=([0], [0]))

        return g

    def _project_back(self, grad_np, p):
        """Project gradient back from Shampoo eigenbasis to original space."""
        Q_list = self.Q[p]
        g = grad_np

        for Qk in Q_list:
            if isinstance(Qk, list) or (isinstance(Qk, np.ndarray) and Qk.size == 0):
                axes = list(range(g.ndim))
                if g.ndim > 1:
                    g = np.transpose(g, axes[-1:] + axes[:-1])
                continue

            g = np.tensordot(g, Qk, axes=([0], [1]))

        return g

    def step(self):
        """Perform one SOAP optimization step over all parameters."""
        self.t += 1
        cur_lr = self._current_lr()

        for p in self.params:
            if p.grad is None:
                continue

            g_np = self._to_numpy(p.grad.data)

            if p not in self.step_count:
                self.step_count[p] = 0
                self.exp_avg[p] = np.zeros_like(g_np)
                self.exp_avg_sq[p] = np.zeros_like(g_np)
                self._init_preconditioner(g_np, p)
                self._update_preconditioner(g_np, p)
                self._compute_eigenbases(p)
                continue

            if self.Q[p] is None:
                self._compute_eigenbases(p)

            grad_proj = self._project(g_np, p)

            self.step_count[p] += 1
            t = self.step_count[p]

            exp_avg = self.exp_avg[p]
            exp_avg_sq = self.exp_avg_sq[p]

            beta1, beta2 = self.beta1, self.beta2

            exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad_proj
            exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (grad_proj ** 2)

            self.exp_avg[p] = exp_avg
            self.exp_avg_sq[p] = exp_avg_sq

            denom = np.sqrt(exp_avg_sq) + self.eps

            step_size = cur_lr
            if self.correct_bias:
                bias_c1 = 1.0 - beta1 ** t
                bias_c2 = 1.0 - beta2 ** t
                step_size = step_size * (bias_c2 ** 0.5) / bias_c1

            precond_proj = exp_avg / denom
            update_np = self._project_back(precond_proj, p)

            if self.normalize_grads:
                rms = np.sqrt((update_np ** 2).mean())
                if rms > 0.0:
                    update_np = update_np / (rms + 1e-30)

            update_tensor = self._from_numpy(update_np, p.data)
            new_data = p.data - step_size * update_tensor

            if self.weight_decay > 0.0:
                new_data = new_data - cur_lr * self.weight_decay * p.data

            p.data = new_data

            self._update_preconditioner(g_np, p)
            if t % self.precondition_frequency == 0:
                self._compute_eigenbases(p)




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

# class SOAP(Optimizer):
#     """
#     SOAP-like optimizer (factored second moment, Adam-style first moment).

#     Conceptual behavior:
#     - For 2D parameters (weight matrices), maintain factored second-moment
#       statistics (per-row and per-column) like Adafactor:
#         * row_v[p]: shape (out_dim,)
#         * col_v[p]: shape (in_dim,)
#       and approximate per-element second moment via outer-product-style
#       row/col RMS.
#     - For non-2D parameters (biases, LN weights, conv kernels if 4D),
#       fall back to standard Adam-style diagonal second moment v[p].

#     This is not full SOAP (which would run Adam in the eigenbasis of a
#     Shampoo preconditioner), but a memory-efficient Adam variant inspired
#     by Adafactor / Shampoo / SOAP ideas.
#     """
#     def __init__(
#         self,
#         params,
#         lr=0.001,
#         beta1=0.9,
#         beta2=0.999,
#         eps=1e-8,
#         weight_decay=0.0,
#     ):
#         super().__init__(params)
#         self.lr = lr
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.eps = eps
#         self.weight_decay = weight_decay
#         self.t = 0

#         # First moment
#         self.m = {}
#         # Diagonal second moment fallback (Adam-style)
#         self.v = {}
#         # Factored second moments for matrix-shaped params
#         self.row_v = {}
#         self.col_v = {}

#     def step(self):
#         self.t += 1

#         for p in self.params:
#             if p.grad is None:
#                 continue

#             # Get gradient as NDArray
#             g = p.grad.data

#             # L2-style weight decay (classic, not decoupled AdamW)
#             if self.weight_decay != 0.0:
#                 g = g + self.weight_decay * p.data

#             # Initialize first moment
#             if p not in self.m:
#                 self.m[p] = 0 * g

#             m_prev = self.m[p]
#             # First moment update (Adam)
#             m = self.beta1 * m_prev + (1.0 - self.beta1) * g
#             # Bias correction for m
#             m_hat = m / (1.0 - self.beta1 ** self.t)

#             shape = g.shape

#             # -------------------------
#             # 2D parameters: factored stats (Adafactor-like)
#             # -------------------------
#             if len(shape) == 2:
#                 out_dim, in_dim = shape

#                 # Lazy init for row/col stats
#                 if p not in self.row_v:
#                     # row_v[p] shape: (out_dim,)
#                     # col_v[p] shape: (in_dim,)
#                     # Use g**2 sums to get correct shapes, then zero them out.
#                     row_init = (g ** 2).sum(axes=1) * 0
#                     col_init = (g ** 2).sum(axes=0) * 0
#                     self.row_v[p] = row_init
#                     self.col_v[p] = col_init

#                 g2 = g ** 2

#                 row_v_prev = self.row_v[p]
#                 col_v_prev = self.col_v[p]

#                 # Exponential moving average of squared-grad row/col sums
#                 row_v = self.beta2 * row_v_prev + (1.0 - self.beta2) * g2.sum(axes=1)
#                 col_v = self.beta2 * col_v_prev + (1.0 - self.beta2) * g2.sum(axes=0)

#                 self.row_v[p] = row_v
#                 self.col_v[p] = col_v

#                 # Bias correction for factored stats
#                 row_v_hat = row_v / (1.0 - self.beta2 ** self.t)
#                 col_v_hat = col_v / (1.0 - self.beta2 ** self.t)

#                 # Adafactor-style RMS per row/col
#                 # (divide by dimension to approximate average per element)
#                 row_rms = (row_v_hat / float(in_dim)) ** 0.5   # (out_dim,)
#                 col_rms = (col_v_hat / float(out_dim)) ** 0.5  # (in_dim,)

#                 # Reshape to enable broadcasting to full matrix shape
#                 row_rms = row_rms.reshape((out_dim, 1))        # (out_dim, 1)
#                 col_rms = col_rms.reshape((1, in_dim))         # (1, in_dim)

#                 row_mat = row_rms.broadcast_to((out_dim, in_dim))
#                 col_mat = col_rms.broadcast_to((out_dim, in_dim))

#                 # Approximate per-element RMS
#                 denom = row_mat * col_mat + self.eps           # (out_dim, in_dim)

#                 # Preconditioned update (Adam-like in factored space)
#                 precond_grad = m_hat / denom

#             # -------------------------
#             # Non-2D parameters: fallback to Adam-style diag v
#             # -------------------------
#             else:
#                 if p not in self.v:
#                     self.v[p] = 0 * g

#                 v_prev = self.v[p]
#                 v = self.beta2 * v_prev + (1.0 - self.beta2) * (g ** 2)
#                 self.v[p] = v

#                 # Bias correction for v
#                 v_hat = v / (1.0 - self.beta2 ** self.t)

#                 precond_grad = m_hat / (v_hat ** 0.5 + self.eps)

#             # Parameter update
#             new_data = p.data - self.lr * precond_grad

#             # Write back to Tensor's underlying data
#             p.data = type(p)(new_data, dtype=p.dtype, device=p.device)

#             # Store first moment
#             self.m[p] = m
