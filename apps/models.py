import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

class ConvGroup(nn.Module):
    def __init__(self, in_channels, out_channels, device=None, dtype="float32"):
        super().__init__()
        self.conv1 = nn.Conv(in_channels, out_channels, 3, 1, device=device, dtype=dtype)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1   = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)

        self.conv2 = nn.Conv(out_channels, out_channels, 3, 1, device=device, dtype=dtype)
        self.bn2   = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)

        self.act   = nn.GELU()

    def forward(self, x):
        # x: (N, H, W, C)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        return x

class CifarNetNeedle(nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()

        # 1) Whitening conv: Conv2d(3, 24, kernel_size=2, stride=1)
        self.whiten = nn.Conv(3, 24, 2, 1, device=device, dtype=dtype)
        self.whiten_act = nn.GELU()

        # 2) Conv groups
        self.group1 = ConvGroup(24,  64, device=device, dtype=dtype)
        self.group2 = ConvGroup(64, 256, device=device, dtype=dtype)
        self.group3 = ConvGroup(256, 256, device=device, dtype=dtype)

        # 3) Final pooling: MaxPool2d(kernel_size=3, stride=3)
        self.final_pool = nn.MaxPool2d(kernel_size=3, stride=3)

        # 4) Head: Linear(256, 10)
        # In the PyTorch version head is bias=False, we can mirror that
        self.head = nn.Linear(256, 10, bias=True, device=device, dtype=dtype)

        self.device = device
        self.dtype  = dtype

    def forward(self, x):
        # x: (N, 32, 32, 3), from CIFAR10Dataset
        x = self.whiten(x)
        x = self.whiten_act(x)

        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)

        x = self.final_pool(x)     # (N, H', W', 256)

        # Flatten spatial dims
        N = x.shape[0]
        x = x.reshape((N, -1))     # (N, 256 * H' * W')

        # In the PyTorch CifarNet, after all the pooling, H' and W' are 1,
        # so this is just (N, 256). Even if your Needle conv shapes differ
        # slightly, flattening still works with a linear layer.
        # If you want to be strict, you can verify H' and W' by printing x.shape.

        # If you find x.shape[1] != 256, then change:
        #   self.head = nn.Linear(256, 10, ...)
        # to:
        #   self.head = nn.Linear(x.shape[1], 10, ...)
        # and reinitialize.

        logits = self.head(x)
        return logits



class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        
        # Layer 1: ConvBN(3, 16, 7, 4)
        self.conv1 = nn.Sequential(
            nn.Conv(3, 16, 7, 4, device=device, dtype=dtype),
            nn.BatchNorm2d(16, device=device, dtype=dtype),
            nn.ReLU()
        )
        
        # Layer 2: ConvBN(16, 32, 3, 2)
        self.conv2 = nn.Sequential(
            nn.Conv(16, 32, 3, 2, device=device, dtype=dtype),
            nn.BatchNorm2d(32, device=device, dtype=dtype),
            nn.ReLU()
        )
        
        # Layer 3: ConvBN(32, 32, 3, 1)
        self.conv3 = nn.Sequential(
            nn.Conv(32, 32, 3, 1, device=device, dtype=dtype),
            nn.BatchNorm2d(32, device=device, dtype=dtype),
            nn.ReLU()
        )
        
        # Layer 4: ConvBN(32, 32, 3, 1)
        self.conv4 = nn.Sequential(
            nn.Conv(32, 32, 3, 1, device=device, dtype=dtype),
            nn.BatchNorm2d(32, device=device, dtype=dtype),
            nn.ReLU()
        )
        
        # Layer 5: ConvBN(32, 64, 3, 2) - takes residual from layer 2
        self.conv5 = nn.Sequential(
            nn.Conv(32, 64, 3, 2, device=device, dtype=dtype),
            nn.BatchNorm2d(64, device=device, dtype=dtype),
            nn.ReLU()
        )
        
        # Layer 6: ConvBN(64, 128, 3, 2) - note: fixed from spec's "32" to "64"
        self.conv6 = nn.Sequential(
            nn.Conv(64, 128, 3, 2, device=device, dtype=dtype),
            nn.BatchNorm2d(128, device=device, dtype=dtype),
            nn.ReLU()
        )
        
        # Layer 7: ConvBN(128, 128, 3, 1)
        self.conv7 = nn.Sequential(
            nn.Conv(128, 128, 3, 1, device=device, dtype=dtype),
            nn.BatchNorm2d(128, device=device, dtype=dtype),
            nn.ReLU()
        )
        
        # Layer 8: ConvBN(128, 128, 3, 1)
        self.conv8 = nn.Sequential(
            nn.Conv(128, 128, 3, 1, device=device, dtype=dtype),
            nn.BatchNorm2d(128, device=device, dtype=dtype),
            nn.ReLU()
        )
        
        # Layer 9: Linear(128, 128) - takes residual from layer 6
        self.fc1 = nn.Linear(128, 128, device=device, dtype=dtype)
        
        # Layer 10: ReLU
        self.relu = nn.ReLU()
        
        # Layer 11: Linear(128, 10)
        self.fc2 = nn.Linear(128, 10, device=device, dtype=dtype)
        
        self.device = device
        self.dtype  = dtype
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        # Layers 1-2
        x = self.conv1(x)
        out2 = self.conv2(x)
        
        # Layers 3-4
        x = self.conv3(out2)
        x = self.conv4(x)
        
        # Add residual from layer 2 before layer 5
        x = x + out2
        
        # Layer 5
        x = self.conv5(x)
        
        # Layer 6 (save for residual)
        out6 = self.conv6(x)
        
        # Layers 7-8
        x = self.conv7(out6)
        x = self.conv8(x)
        
        # Flatten: (N, C, H, W) -> (N, C*H*W)
        # Use dynamic reshaping
        x_flat = x.reshape((x.shape[0], -1))
        out6_flat = out6.reshape((out6.shape[0], -1))
        
        # Add residual from layer 6 before layer 9
        x = x_flat + out6_flat
        
        # Layers 9-11
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
        ### END YOUR SOLUTION

class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_model_type = seq_model
        self.seq_len = seq_len
        self.device = device
        self.dtype = dtype

        # Embedding: maps word IDs -> embedding vectors
        self.embedding = nn.Embedding(
            num_embeddings=output_size,
            embedding_dim=embedding_size,
            device=device,
            dtype=dtype,
        )

        # Sequence model: RNN or LSTM
        if seq_model == "rnn":
            self.seq_model = nn.RNN(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=True,
                nonlinearity="tanh",
                device=device,
                dtype=dtype,
            )
        elif seq_model == "lstm":
            self.seq_model = nn.LSTM(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=True,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown seq_model: {seq_model}")

        # Linear layer: hidden -> vocab logits
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
            device=device,
            dtype=dtype,
        )
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        # Embedding: (seq_len, bs) -> (seq_len, bs, embedding_size)
        emb = self.embedding(x)

        # Sequence model
        seq_out, h_new = self.seq_model(emb, h)  # (seq_len, bs, hidden_size)
        seq_len, bs, _ = seq_out.shape

        # Flatten time and batch: (seq_len*bs, hidden_size)
        seq_out_flat = ndl.ops.reshape(seq_out, (seq_len * bs, self.hidden_size))

        # Project to logits: (seq_len*bs, output_size)
        out = self.linear(seq_out_flat)

        return out, h_new
        ### END YOUR SOLUTION



if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
