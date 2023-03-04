from typing import Callable

import torch
from omegaconf import DictConfig

from src.train_models.utils import to_device


def train_simclr_fn(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    scheduler,
    device: str,
):
    def train_step(view_0, view_1):
        view_0 = to_device(view_0, device)
        view_1 = to_device(view_1, device)
        optimizer.zero_grad()
        z0 = model(view_0)
        z1 = model(view_1)
        loss = loss_fn(z0, z1)
        loss.backward()
        optimizer.step()
        scheduler.step()

        return loss.item()

    return train_step


def mini_batch_simclr(
    dataloader: torch.utils.data.DataLoader,
    step_fn: Callable,
    params: DictConfig,
):
    avg_loss = 0
    for (x0, x1), _, _ in dataloader:
        loss = step_fn(x0, x1)
        avg_loss += loss
    return avg_loss / len(dataloader)


__all__ = ["train_simclr_fn", "mini_batch_simclr"]
