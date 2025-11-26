from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    Multi-head self attention module.
    Expects q, k, v with shape (B, H, T, D_head) and returns:
      - result: (B, H, T_q, D_head)
      - probs:  (B, H, T_q, T_k)
    """

    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, T_q, T_k, device):
        """
        Return a triangular causal mask of shape (1, 1, T_q, T_k),
        with 0 on allowed positions and -inf on masked ("future") positions.
        """
        # Mask strictly future positions: j > i
        mask_np = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, T_q, T_k), dtype=np.float32),
            k=1,
        )
        mask_nd = ndarray.array(mask_np, device=device)
        return Tensor(mask_nd, device=device, dtype="float32", requires_grad=False)

    def _softmax_last_dim(self, logit: Tensor) -> Tensor:
        """
        Numerically-stable softmax over the last dimension.
        logit: (B, H, T_q, T_k)
        """
        # Take max along last axis on raw NDArray (no grad)
        max_vals_nd = logit.realize_cached_data().max(axis=3)  # (B, H, T_q)
        max_vals = Tensor(
            max_vals_nd,
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False,
        )
        # Broadcast back to full shape
        max_vals = ops.reshape(max_vals, (*logit.shape[:-1], 1))
        max_vals = max_vals.broadcast_to(logit.shape)

        shifted = logit - max_vals
        exp_shifted = ops.exp(shifted)  # (B, H, T_q, T_k)

        denom = exp_shifted.sum(axes=3)                      # (B, H, T_q)
        denom = ops.reshape(denom, (*logit.shape[:-1], 1))   # (B, H, T_q, 1)
        denom = denom.broadcast_to(logit.shape)              # (B, H, T_q, T_k)

        return exp_shifted / denom

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        """
        q, k, v: (B, H, T_q_or_k, D_h)

        Returns:
          result: (B, H, T_q, D_h)
          probs:  (B, H, T_q, T_k)
        """
        B, H, T_q, D_h = q.shape
        _, _, T_k, _   = k.shape

        # ---------- QK^T logits ----------
        # q_exp: (B, H, T_q, 1, D_h) -> broadcast to (B, H, T_q, T_k, D_h)
        # k_exp: (B, H, 1, T_k, D_h) -> broadcast to (B, H, T_q, T_k, D_h)
        q_exp = ops.reshape(q, (B, H, T_q, 1, D_h))
        k_exp = ops.reshape(k, (B, H, 1, T_k, D_h))

        q_exp = q_exp.broadcast_to((B, H, T_q, T_k, D_h))
        k_exp = k_exp.broadcast_to((B, H, T_q, T_k, D_h))

        prod_qk = q_exp * k_exp                 # (B, H, T_q, T_k, D_h)
        logits = prod_qk.sum(axes=4)            # (B, H, T_q, T_k)

        scale = float(D_h) ** 0.5
        logits = logits / scale

        # ---------- Causal mask ----------
        if self.causal:
            mask = self.create_causal_mask(T_q, T_k, q.device)  # (1,1,T_q,T_k)
            mask = mask.broadcast_to(logits.shape)              # (B,H,T_q,T_k)
            logits = logits + mask

        # ---------- Softmax ----------
        probs = self._softmax_last_dim(logits)  # (B, H, T_q, T_k)

        # ---------- Dropout on probs ----------
        probs = self.dropout(probs)             # (B, H, T_q, T_k)

        # ---------- Attention @ V ----------
        # probs: (B, H, T_q, T_k)
        # v:     (B, H, T_k, D_h)
        # attn_exp: (B, H, T_q, T_k, 1) -> (B, H, T_q, T_k, D_h)
        # v_exp:    (B, H, 1, T_k, D_h) -> (B, H, T_q, T_k, D_h)
        attn_exp = ops.reshape(probs, (B, H, T_q, T_k, 1))
        v_exp = ops.reshape(v, (B, H, 1, T_k, D_h))

        attn_exp = attn_exp.broadcast_to((B, H, T_q, T_k, D_h))
        v_exp = v_exp.broadcast_to((B, H, T_q, T_k, D_h))

        prod_av = attn_exp * v_exp             # (B, H, T_q, T_k, D_h)
        result = prod_av.sum(axes=3)           # sum over T_k → (B, H, T_q, D_h)

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head
        inner_dim = num_head * dim_head

        # LayerNorms over feature dim
        self.prenorm_q = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(v_features, device=device, dtype=dtype)

        # Linear projections: D_in -> H * D_head
        self.q_projection = Linear(q_features, inner_dim, bias=False,
                                   device=device, dtype=dtype)
        self.k_projection = Linear(k_features, inner_dim, bias=False,
                                   device=device, dtype=dtype)
        self.v_projection = Linear(v_features, inner_dim, bias=False,
                                   device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype
        )

        self.out_projection = Linear(inner_dim, out_features, bias=False,
                                     device=device, dtype=dtype)

    def forward(self, q, k=None, v=None):
        # q: (B, T_q, q_dim)
        # k, v: (B, T_k, k_dim / v_dim)
        if k is None:
            k = q
        if v is None:
            v = q

        B, T_q, q_dim = q.shape
        _, T_k, k_dim = k.shape
        _, _, v_dim   = v.shape

        # 1) LayerNorm on the last dimension
        q_norm = self.prenorm_q(q)   # (B, T_q, q_dim)
        k_norm = self.prenorm_k(k)   # (B, T_k, k_dim)
        v_norm = self.prenorm_v(v)   # (B, T_k, v_dim)

        # 2) Apply linear projections
        # Flatten batch and time so Linear sees shape (B*T, dim)
        q_flat = q_norm.reshape((B * T_q, q_dim))
        k_flat = k_norm.reshape((B * T_k, k_dim))
        v_flat = v_norm.reshape((B * T_k, v_dim))

        q_proj_flat = self.q_projection(q_flat)    # (B*T_q, H*D)
        k_proj_flat = self.k_projection(k_flat)    # (B*T_k, H*D)
        v_proj_flat = self.v_projection(v_flat)    # (B*T_k, H*D)

        # Reshape to (B, T, H, D)
        q_proj = q_proj_flat.reshape((B, T_q, self.num_head, self.dim_head))
        k_proj = k_proj_flat.reshape((B, T_k, self.num_head, self.dim_head))
        v_proj = v_proj_flat.reshape((B, T_k, self.num_head, self.dim_head))

        # Transpose to (B, H, T, D)
        q_proj = ops.transpose(q_proj, (1, 2))   # (B, H, T_q, D)
        k_proj = ops.transpose(k_proj, (1, 2))   # (B, H, T_k, D)
        v_proj = ops.transpose(v_proj, (1, 2))   # (B, H, T_k, D)

        # 3) Multi-head attention
        x, probs = self.attn(q_proj, k_proj, v_proj)   # x: (B, H, T_q, D)
        self.probs = probs

        # Back to (B, T_q, H, D) then (B*T_q, H*D)
        x = ops.transpose(x, (1, 2))  # (B, T_q, H, D)
        x = x.reshape((B * T_q, self.num_head * self.dim_head))

        # 4) Output projection: (B*T_q, out_features) -> (B, T_q, out_features)
        x = self.out_projection(x)
        x = x.reshape((B, T_q, self.out_features))
        return x



class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.q_features = q_features
        self.hidden_size = hidden_size

        # Self-attention block (uses prenorm internally in AttentionLayer)
        self.attn = AttentionLayer(
            q_features=q_features,
            num_head=num_head,
            dim_head=dim_head,
            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype,
        )
        self.dropout_attn = Dropout(dropout)

        # MLP block: LayerNorm -> Linear1 -> ReLU -> Dropout -> Linear2 -> Dropout
        self.norm_mlp = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.relu = ReLU()
        self.dropout_mlp1 = Dropout(dropout)
        self.dropout_mlp2 = Dropout(dropout)
        ### END YOUR SOLUTION

    def forward(self, x):
        """
        x: (batch_size, seq_len, x_dim)
        returns: (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        # ---- 1) Self-attention residual: x = x + Dropout(Attention(x)) ----
        attn_out = self.attn(x)            # (B, T, x_dim)
        attn_out = self.dropout_attn(attn_out)
        x = x + attn_out                   # residual, shapes match

        # ---- 2) MLP residual: x = x + Dropout(Linear2(Dropout(ReLU(Linear1(LN(x))))) ----
        # LayerNorm over last dim
        h = self.norm_mlp(x)               # (B, T, x_dim)

        # Linear1 expects 2D, so flatten batch and time
        h_flat = h.reshape((batch_size * seq_len, x_dim))
        h1_flat = self.linear1(h_flat)     # (B*T, hidden_size)

        # Back to (B, T, hidden_size)
        h1 = h1_flat.reshape((batch_size, seq_len, self.hidden_size))
        h1 = self.relu(h1)
        h1 = self.dropout_mlp1(h1)

        # Second linear: (B*T, hidden_size) -> (B*T, x_dim)
        h1_flat = h1.reshape((batch_size * seq_len, self.hidden_size))
        h2_flat = self.linear2(h1_flat)    # (B*T, x_dim)
        h2 = h2_flat.reshape((batch_size, seq_len, x_dim))
        h2 = self.dropout_mlp2(h2)

        # Residual connection
        x = x + h2
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first
        self.embedding_size = embedding_size
        self.sequence_len = sequence_len

        ### BEGIN YOUR SOLUTION

        # Learned positional embeddings: positions 0..sequence_len-1
        # Uses your Embedding, which expects input of shape (T, B)
        self.pos_embedding = Embedding(
            num_embeddings=sequence_len,
            embedding_dim=embedding_size,
            device=device,
            dtype=dtype,
        )

        # Optional dropout on input + positional encodings
        self.input_dropout = Dropout(dropout)

        # Stack of TransformerLayers
        layers = []
        for _ in range(num_layers):
            layers.append(
                TransformerLayer(
                    q_features=embedding_size,
                    num_head=num_head,
                    dim_head=dim_head,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    causal=causal,
                    device=device,
                    dtype=dtype,
                )
            )
        self.layers = Sequential(*layers)

        ### END YOUR SOLUTION

    def forward(self, x, h=None):

        # x is either:
        #   batch_first=True:  (B, T, D)
        #   batch_first=False: (T, B, D)
        if not self.batch_first:
            # (T, B, D) -> (B, T, D)
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        B, T, D = x.shape

        # ---- Positional indices: shape (T, B) ----
        # For each time step t, all batch positions share the same index t.
        # Create a numpy array (T, B) with values 0..T-1 along rows.
        pos_np = np.arange(T, dtype=np.float32).reshape(T, 1)      # (T, 1)
        pos_np = np.repeat(pos_np, B, axis=1)                      # (T, B)

        # Wrap as NDArray and then Tensor (dtype float32 is fine; one_hot will just use the values)
        pos_nd = ndarray.array(pos_np, device=self.device)
        pos_ids = Tensor(pos_nd, device=self.device, dtype=self.dtype, requires_grad=False)  # (T, B)

        # Look up positional embeddings with your Embedding:
        # Input:  (T, B)
        # Output: (T, B, D)
        pos_emb = self.pos_embedding(pos_ids)                      # (T, B, D)

        # We currently have x as (B, T, D).  Make pos_emb match that.
        # pos_emb: (T, B, D) -> (B, T, D)
        pos_emb = ops.transpose(pos_emb, axes=(0, 1))           # (B, T, D)

        # Add positional encodings (shapes match exactly, so no auto-broadcast issues)
        x = x + pos_emb

        # Feed through the stack of TransformerLayers
        x = self.layers(x)                                         # (B, T, D)

        ### END YOUR SOLUTION

        if not self.batch_first:
            # Convert back to (T, B, D) for the caller
            x = ops.transpose(x, axes=(0, 1))

        # Transformer has no separate hidden state; return zeros_like(x) for API compatibility
        return x, init.zeros_like(x)
