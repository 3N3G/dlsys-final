"""hw1/apps/simple_ml.py"""
from tqdm import tqdm
import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # training / eval mode
    if opt is None:
        model.eval()
    else:
        model.train()

    # infer device / dtype from model parameters (more reliable than model.device)
    params = [p for p in model.parameters()]
    param_device = params[0].device if len(params) > 0 else None
    param_dtype = params[0].dtype if len(params) > 0 else "float32"

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    
    for X, y in dataloader:
        X = ndl.Tensor(X.numpy(), device=model.device, dtype="float32")
        y = ndl.Tensor(y.numpy(), device=model.device, dtype="float32")
        
        # Convert X to Tensor if needed
        if not isinstance(X, ndl.Tensor):
            X = ndl.Tensor(
                np.array(X, dtype=np.float32),
                device=param_device,
                dtype="float32",
            )
        # Convert y to Tensor of integer labels if needed
        if not isinstance(y, ndl.Tensor):
            y = ndl.Tensor(
                np.array(y, dtype=np.int32),
                device=param_device,
                dtype="int32",
            )

        batch_size = y.shape[0]
        total_examples += batch_size

        # Forward pass
        logits = model(X)
        loss = loss_fn(logits, y)

        # Accuracy (computed on CPU via numpy)
        preds = logits.numpy().argmax(axis=1)
        true = y.numpy().astype(np.int32).reshape(-1)
        batch_correct = (preds == true).sum()

        total_correct += int(batch_correct)
        total_loss += float(loss.numpy()) * batch_size

        # Backward & step if training
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()

    avg_acc = total_correct / total_examples
    avg_loss = total_loss / total_examples
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def train_cifar10(
    model,
    dataloader,
    n_epochs=1,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    optimizer_kwargs=None,
    loss_fn=nn.SoftmaxLoss,
    return_history=False,
):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float) - used for optimizers that accept `lr`
        weight_decay: weight decay (float)
        optimizer_kwargs: optional dict of extra keyword arguments passed to
            the optimizer constructor. This is where you can pass things like
            muon_lr, sgd_lr, momentum, etc. If None, defaults to
            {"lr": lr, "weight_decay": weight_decay}.
        loss_fn: nn.Module class
        return_history: if True, return per-epoch histories instead of just
            the final (acc, loss).

    Returns:
        If return_history is False (default):
            avg_acc, avg_loss from last epoch of training
        If return_history is True:
            (train_acc_hist, train_loss_hist, test_acc_hist, test_loss_hist)
            each a list of length n_epochs.
    """
    np.random.seed(4)

    # Build optimizer kwargs
    if optimizer_kwargs is None:
        optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay}
    else:
        # copy so we don't mutate caller's dict
        optimizer_kwargs = dict(optimizer_kwargs)
        # supply a default weight_decay if caller didn't override
        optimizer_kwargs.setdefault("weight_decay", weight_decay)

    # Instantiate optimizer and loss module
    opt = optimizer(model.parameters(), **optimizer_kwargs)
    loss_module = loss_fn()

    # Histories
    train_acc_hist = []
    train_loss_hist = []
    test_acc_hist = []
    test_loss_hist = []

    for epoch in range(n_epochs):
        train_acc, train_loss = epoch_general_cifar10(
            dataloader, model, loss_fn=loss_module, opt=opt
        )
        test_acc, test_loss = evaluate_cifar10(model, dataloader)

        train_acc_hist.append(train_acc)
        train_loss_hist.append(train_loss)
        test_acc_hist.append(test_acc)
        test_loss_hist.append(test_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"train_acc={train_acc:.4f}, train_loss={train_loss:.4f} | "
            f"test_acc={test_acc:.4f}, test_loss={test_loss:.4f}"
        )

    if return_history:
        return train_acc_hist, train_loss_hist, test_acc_hist, test_loss_hist
    else:
        return train_acc_hist[-1], train_loss_hist[-1]


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_module = loss_fn()
    avg_acc, avg_loss = epoch_general_cifar10(
        dataloader, model, loss_fn=loss_module, opt=None
    )
    return avg_acc, avg_loss
    ### END YOUR SOLUTION



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset (token-weighted)
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    training = opt is not None

    total_loss = 0.0         # token-weighted sum of losses
    total_tokens = 0         # total number of target tokens
    total_correct = 0
    total_examples = 0

    nbatch, batch_size = data.shape

    # We restart hidden state for each epoch (no carry across BPTT segments)
    for i in range(0, nbatch - 1, seq_len):
        # x: (seq_len', bs), y: (seq_len'*bs,)
        x, y = ndl.data.datasets.get_batch(data, i, seq_len, device=device, dtype=dtype)

        if training:
            opt.reset_grad()

        # Forward through language model
        out, _ = model(x)  # out: (N, vocab_size), N = seq_len'*bs
        loss = loss_fn(out, y)

        if training:
            loss.backward()

            # Optional simple gradient clipping by global norm
            if clip is not None:
                total_norm_sq = 0.0
                grads = []
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    g_arr = p.grad.cached_data
                    g_np = g_arr.numpy()
                    grads.append((p, g_arr, g_np))
                    total_norm_sq += np.sum(g_np ** 2)
                total_norm = math.sqrt(total_norm_sq) if total_norm_sq > 0 else 0.0

                if total_norm > clip and total_norm > 0:
                    scale = clip / total_norm
                    for _, g_arr, g_np in grads:
                        g_np *= scale
                        # write back scaled gradients
                        g_arr[:] = g_np

            opt.step()

        # Loss accounting: weight by number of tokens in this batch
        loss_val = float(loss.numpy())
        num_tokens = y.shape[0]
        total_loss += loss_val * num_tokens
        total_tokens += num_tokens

        # Accuracy
        logits = out.numpy()                     # (N, vocab_size)
        preds = logits.argmax(axis=1)           # (N,)
        targets = y.numpy().astype(np.int64)    # (N,)

        total_correct += (preds == targets).sum()
        total_examples += targets.shape[0]

    avg_loss = total_loss / total_tokens
    avg_acc = total_correct / total_examples

    return avg_acc, avg_loss
    ### END YOUR SOLUTION



def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_module = loss_fn()

    last_acc, last_loss = 0.0, 0.0
    for _ in range(n_epochs):
        last_acc, last_loss = epoch_general_ptb(
            data,
            model,
            seq_len=seq_len,
            loss_fn=loss_module,
            opt=opt,
            clip=clip,
            device=device,
            dtype=dtype,
        )

    return last_acc, last_loss
    ### END YOUR SOLUTION



def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_module = loss_fn()
    avg_acc, avg_loss = epoch_general_ptb(
        data,
        model,
        seq_len=seq_len,
        loss_fn=loss_module,
        opt=None,
        clip=None,
        device=device,
        dtype=dtype,
    )
    return avg_acc, avg_loss
    ### END YOUR SOLUTION



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
