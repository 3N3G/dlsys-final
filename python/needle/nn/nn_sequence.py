"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        e = ops.exp(x)
        denom = ops.add_scalar(e, 1.0)
        return ops.divide(e, denom)


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        k = 1.0 / hidden_size
        bound = float(np.sqrt(k))

        # Weight matrices
        self.W_ih = Parameter(
            init.rand(
                input_size, 
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size, 
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )

        # Biases
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size)
        h of shape (bs, hidden_size) or None
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]

        # Initialize h with zeros if not provided
        if h is None:
            h = Tensor(
                np.zeros((bs, self.hidden_size), dtype=np.float32),
                device=self.device,
            )

        # Affine transform for input and hidden
        # (bs, input_size) @ (input_size, hidden_size) -> (bs, hidden_size)
        x_term = X @ self.W_ih
        h_term = h @ self.W_hh

        if self.bias:
            bias_ih = ops.broadcast_to(self.bias_ih, x_term.shape)
            bias_hh = ops.broadcast_to(self.bias_hh, h_term.shape)
            x_term = x_term + bias_ih
            h_term = h_term + bias_hh

        pre_act = x_term + h_term

        # Nonlinearity
        if self.nonlinearity == "tanh":
            h_new = ops.tanh(pre_act)
        elif self.nonlinearity == "relu":
            h_new = ops.relu(pre_act)
        else:
            raise ValueError(f"Unsupported nonlinearity: {self.nonlinearity}")

        return h_new
        ### END YOUR SOLUTION



class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        # Build a stack of RNNCell modules
        cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = RNNCell(
                layer_input_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype,
            )
            cells.append(cell)
        self.rnn_cells: List[RNNCell] = cells
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X: (seq_len, bs, input_size)
        h0: (num_layers, bs, hidden_size) or None

        Returns:
        output: (seq_len, bs, hidden_size)  # from last layer
        h_n:   (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape

        # Initialize per-layer hidden states
        if h0 is None:
            # Start all layers at zero hidden state
            h_list = [
                Tensor(
                    np.zeros((bs, self.hidden_size), dtype=np.float32),
                    device=self.device,
                )
                for _ in range(self.num_layers)
            ]
        else:
            # h0: (num_layers, bs, hidden_size)
            # Use split along layer dimension, then reshape from (1, bs, hidden) -> (bs, hidden)
            h0_layers = ops.split(h0, axis=0)  # list of length num_layers
            h_list = []
            for h0_l in h0_layers:
                h0_l_reshaped = ops.reshape(h0_l, (bs, self.hidden_size))
                h_list.append(h0_l_reshaped)

        # Split input along time dimension: list of (1, bs, input_size)
        X_ts = ops.split(X, axis=0)

        outputs = []

        for x_t in X_ts:
            # x_t: (1, bs, input_size) -> (bs, input_size or hidden_size)
            x_t_layer = ops.reshape(
                x_t, (bs, self.input_size)
            )  # input for layer 0 at this time step

            # Propagate through layers
            for layer_idx in range(self.num_layers):
                cell = self.rnn_cells[layer_idx]
                h_prev = h_list[layer_idx]

                # cell expects (bs, *, ), returns (bs, hidden_size)
                h_new = cell(x_t_layer, h_prev)
                h_list[layer_idx] = h_new   # update hidden for this layer

                # Output of this layer is input to the next layer
                x_t_layer = h_new

            # x_t_layer is now the last layer's hidden state h_t^(L)
            outputs.append(x_t_layer)

        # Stack over time: (seq_len, bs, hidden_size)
        output = ops.stack(outputs, axis=0)

        # Final hidden states for all layers: list of (bs, hidden_size) -> (num_layers, bs, hidden_size)
        h_n = ops.stack(h_list, axis=0)

        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype

        k = 1.0 / hidden_size
        bound = float(np.sqrt(k))

        # Input-to-hidden weights: (input_size, 4*hidden_size)
        self.W_ih = Parameter(
            init.rand(
                input_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )

        # Hidden-to-hidden weights: (hidden_size, 4*hidden_size)
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )

        if bias:
            # Biases: (4*hidden_size,)
            self.bias_ih = Parameter(
                init.rand(
                    4 * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    4 * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]

        # Unpack / initialize hidden and cell state
        if h is None:
            h_prev = Tensor(
                np.zeros((bs, self.hidden_size), dtype=np.float32),
                device=self.device,
            )
            c_prev = Tensor(
                np.zeros((bs, self.hidden_size), dtype=np.float32),
                device=self.device,
            )
        else:
            h_prev, c_prev = h

        # Linear projections
        # X: (bs, input_size)
        # W_ih: (input_size, 4*hidden_size) -> (bs, 4*hidden_size)
        x_term = X @ self.W_ih

        # h_prev: (bs, hidden_size)
        # W_hh: (hidden_size, 4*hidden_size) -> (bs, 4*hidden_size)
        h_term = h_prev @ self.W_hh

        gates = x_term + h_term

        if self.bias:
            # Broadcast biases to (bs, 4*hidden_size)
            bias_ih = ops.broadcast_to(self.bias_ih, gates.shape)
            bias_hh = ops.broadcast_to(self.bias_hh, gates.shape)
            gates = gates + bias_ih + bias_hh

        # gates: (bs, 4*hidden_size) -> (bs, 4, hidden_size)
        gates_reshaped = ops.reshape(gates, (bs, 4, self.hidden_size))
        # Split along the "4" dimension into 4 tensors of shape (bs, 1, hidden_size)
        i_gate_t, f_gate_t, g_gate_t, o_gate_t = ops.split(gates_reshaped, axis=1)

        # Reshape each to (bs, hidden_size)
        i_lin = ops.reshape(i_gate_t, (bs, self.hidden_size))
        f_lin = ops.reshape(f_gate_t, (bs, self.hidden_size))
        g_lin = ops.reshape(g_gate_t, (bs, self.hidden_size))
        o_lin = ops.reshape(o_gate_t, (bs, self.hidden_size))

        # Nonlinearities
        def sigmoid(x):
            e = ops.exp(x)
            denom = ops.add_scalar(e, 1.0)
            return ops.divide(e, denom)

        i = sigmoid(i_lin)
        f = sigmoid(f_lin)
        g = ops.tanh(g_lin)
        o = sigmoid(o_lin)

        # Cell and hidden updates
        c_new = f * c_prev + i * g
        h_new = o * ops.tanh(c_new)

        return h_new, c_new
        ### END YOUR SOLUTION



class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype

        cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = LSTMCell(
                layer_input_size,
                hidden_size,
                bias=bias,
                device=device,
                dtype=dtype,
            )
            cells.append(cell)
        self.lstm_cells: List[LSTMCell] = cells
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape

        # Initialize per-layer hidden & cell states
        if h is None:
            h_list = []
            c_list = []
            for _ in range(self.num_layers):
                h0_layer = Tensor(
                    np.zeros((bs, self.hidden_size), dtype=np.float32),
                    device=self.device,
                )
                c0_layer = Tensor(
                    np.zeros((bs, self.hidden_size), dtype=np.float32),
                    device=self.device,
                )
                h_list.append(h0_layer)
                c_list.append(c0_layer)
        else:
            h0, c0 = h  # each: (num_layers, bs, hidden_size)
            h0_layers = ops.split(h0, axis=0)
            c0_layers = ops.split(c0, axis=0)
            h_list = []
            c_list = []
            for l in range(self.num_layers):
                h_l = ops.reshape(h0_layers[l], (bs, self.hidden_size))
                c_l = ops.reshape(c0_layers[l], (bs, self.hidden_size))
                h_list.append(h_l)
                c_list.append(c_l)

        # Split sequence over time
        X_ts = ops.split(X, axis=0)  # list of (1, bs, input_size)

        outputs = []

        for x_t in X_ts:
            # x_t: (1, bs, input_size) -> (bs, input_size) for layer 0
            x_t_layer = ops.reshape(x_t, (bs, self.input_size))

            for layer_idx in range(self.num_layers):
                cell = self.lstm_cells[layer_idx]
                h_prev = h_list[layer_idx]
                c_prev = c_list[layer_idx]

                h_new, c_new = cell(x_t_layer, (h_prev, c_prev))

                h_list[layer_idx] = h_new
                c_list[layer_idx] = c_new

                # Hidden of this layer is input to next layer
                x_t_layer = h_new

            # Last layer hidden is the output at this time
            outputs.append(x_t_layer)

        # Stack over time: (seq_len, bs, hidden_size)
        output = ops.stack(outputs, axis=0)

        # Final hidden and cell states for all layers:
        # lists of (bs, hidden_size) -> (num_layers, bs, hidden_size)
        h_n = ops.stack(h_list, axis=0)
        c_n = ops.stack(c_list, axis=0)

        return output, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = Parameter(
            init.randn(
                num_embeddings,
                embedding_dim,
                mean=0.0,
                std=1.0,
                device=device,
                dtype=dtype,
            )
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        # x: (T, B)
        T, B = x.shape
        N = T * B

        # Flatten indices to (N,)
        x_flat = ops.reshape(x, (N,))

        one_hot = init.one_hot(
            self.num_embeddings,
            x_flat,
            device=x.device,
            dtype=self.weight.dtype,
        )  # (N, num_embeddings)

        # (N, embedding_dim)
        emb_flat = one_hot @ self.weight
        emb = ops.reshape(emb_flat, (T, B, self.embedding_dim))
        return emb
        ### END YOUR SOLUTION