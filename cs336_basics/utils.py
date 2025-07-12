from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, BinaryIO

import torch
from torch import Tensor


def softmax(logits, dim):
    eps = 1e-8
    max_logits = torch.max(logits, dim=dim, keepdim=True).values
    norm_logits = logits - max_logits
    exp_logits = torch.exp(norm_logits)
    sum_logits = torch.sum(exp_logits, dim=dim, keepdim=True)
    return exp_logits / (sum_logits + eps)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    数学推导
    sum_1~m -log(softmax(o_i)[x_i+1]) = sum_1~m (-logexp(o_i[x_i+1]-c)+log(sum_1~vocab_size exp(o_i[a]-c)))
    = sum_1~m (c + log(sum_1~vocab_size exp(o_i[a]-c)) - o_i[x_i+1])


    """
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values  # shape is (batch_size seq_len 1)
    norm_logits = inputs - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(norm_logits), dim=-1))  # shape is (batch_size seq_len)

    target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    loss_per_example = max_logits.squeeze(-1) + log_sum_exp - target_logits

    return torch.mean(loss_per_example)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # Filter parameters with gradients
    parameters_with_grad = [p for p in parameters if p.grad is not None]

    if len(parameters_with_grad) == 0:
        return

    # Calculate total L2 norm of all gradients
    total_norm = torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in parameters_with_grad))

    # Calculate clipping coefficient
    clip_coef = max_l2_norm / (total_norm + 1e-6)  # Add small value to avoid division by zero

    # If total norm exceeds max_norm, scale down all gradients
    if clip_coef < 1.0:
        for p in parameters_with_grad:
            p.grad.mul_(clip_coef)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object
            to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    if isinstance(out, str | os.PathLike):
        with open(out, "wb") as f:
            torch.save(checkpoint, f)
    else:
        torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    if isinstance(src, str | os.PathLike):
        with open(src, "rb") as f:
            checkpoint = torch.load(f)
    else:
        checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]
