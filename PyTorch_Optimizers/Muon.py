import torch
import torch.optim as optim

@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton–Schulz iteration to compute an approximate
    G @ (G^T G)^(-1/2), used to orthogonalize the gradient.

    G: (m, n) tensor
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Normalize so top singular value <= 1
    X = G
    X = X / (X.norm() + eps)

    # Work with smaller dimension first if tall
    if X.size(0) > X.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T          # (m, m)
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X


class Muon(optim.Optimizer):
    """
    Muon optimizer – PyTorch version mimicking your Needle implementation.

    Key differences vs the original airbench Muon:
      - Two learning rates:
          * muon_lr: for >=2D parameters (weights)
          * sgd_lr : for 1D parameters (biases, etc.)
      - Internal linear LR decay over total_steps:
          lr_t = lr_0 * max(0, 1 - t / total_steps)
      - Row-based weight normalization: scale ~ sqrt(rows) / ||W||
      - Momentum with optional Nesterov on the *gradient*.
    """
    def __init__(
        self,
        params,
        muon_lr: float = 0.1,
        sgd_lr: float  = 0.001,
        momentum: float = 0.95,
        nesterov: bool = False,
        ns_steps: int = 5,
        eps: float = 1e-7,
        weight_decay: float = 0.0,
        total_steps: int | None = None,
    ):
        if muon_lr < 0.0 or sgd_lr < 0.0:
            raise ValueError("Invalid learning rates: muon_lr / sgd_lr must be >= 0")
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: must be >= 0")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")

        defaults = dict(
            muon_lr=muon_lr,
            sgd_lr=sgd_lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.total_steps = total_steps
        self.t = 0  # global step counter (like in your Needle code)

    def _current_lrs(self, muon_lr0, sgd_lr0):
        """
        Linear decay:
            lr_t = lr_0 * (1 - t / total_steps)
        If total_steps is None, keep constant.
        """
        if self.total_steps is None or self.total_steps <= 0:
            return muon_lr0, sgd_lr0

        decay = max(0.0, 1.0 - self.t / float(self.total_steps))
        return muon_lr0 * decay, sgd_lr0 * decay

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Increment global step (mirrors Needle's self.t += 1)
        self.t += 1

        for group in self.param_groups:
            muon_lr0   = group["muon_lr"]
            sgd_lr0    = group["sgd_lr"]
            momentum   = group["momentum"]
            nesterov   = group["nesterov"]
            ns_steps   = group["ns_steps"]
            eps        = group["eps"]
            weight_decay = group["weight_decay"]

            cur_muon_lr, cur_sgd_lr = self._current_lrs(muon_lr0, sgd_lr0)

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.detach()
                if weight_decay != 0.0:
                    g = g.add(p.data, alpha=weight_decay)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                # Momentum update: buf = m * buf + (1 - m) * g
                buf.mul_(momentum).add_(g, alpha=(1.0 - momentum))

                # Nesterov option
                if nesterov:
                    update_grad = (1.0 - momentum) * g + momentum * buf
                else:
                    update_grad = buf

                W = p.data

                if W.ndim >= 2:
                    # ---- Weight normalization (row-based) ----
                    rows = W.shape[0]
                    w_norm = W.norm()
                    if w_norm > 0:
                        scale = (rows ** 0.5) / (w_norm + eps)
                        W.mul_(scale)

                    # ---- Newton–Schulz orthogonalization on gradient ----
                    G2d = update_grad.reshape(update_grad.shape[0], -1)
                    G2d_orth = zeropower_via_newtonschulz5(G2d, steps=ns_steps, eps=eps)
                    update = G2d_orth.view_as(W)

                    # Muon step
                    W.add_(update, alpha=-cur_muon_lr)
                else:
                    # 1D params (biases, LN scales, etc.) use plain SGD-style step
                    W.add_(update_grad, alpha=-cur_sgd_lr)

                state["momentum_buffer"] = buf

        return loss
