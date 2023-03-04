import types
from typing import Any

import lightly
import torch
from dagster import In, Out, op
from omegaconf import DictConfig

from src.train_models.models import SimCLR


@op(ins={"params": In(DictConfig)})
def get_criterion(params) -> torch.nn.Module:
    """Method to get the required criterion for training SimCLR"""
    return getattr(
        lightly.loss, params.model.criterion
    )()


@op(ins={"params": In(DictConfig)})
def build_model(
    params: DictConfig,
) -> torch.nn.Module:
    """Method to build SimCLR model"""
    return SimCLR(params.model.backbone)


@op(
    ins={
        "params": In(DictConfig),
        "model_params": In(types.GeneratorType),
    }
)
def build_optimizer(
    params: DictConfig,
    model_params: types.GeneratorType,
) -> torch.optim.Optimizer:
    """Method to build optimizer"""
    lr = (
        params.optimizer.start_lr
        * (params.training.batch_size)
        / 256
    )
    return getattr(
        torch.optim,
        params.optimizer.optimizer_name,
    )(
        params=model_params,
        lr=lr,
        momentum=params.optimizer.momentum,
        weight_decay=params.optimizer.weight_decay,
    )


@op(
    ins={
        "params": In(DictConfig),
        "optimizer": In(torch.optim.Optimizer),
    }
)
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    params: DictConfig,
):
    """Method to build cosine annealing scheduler"""
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, params.epoch
    )
    return scheduler


@op(
    ins={"params": In(DictConfig)},
    out={
        "model": Out(torch.nn.Module),
        "optimizer": Out(torch.optim.Optimizer),
        "scheduler": Out(Any),
    },
)
def compile_model(params):
    """Method to compile model, scheduler and optimizer"""
    model = build_model(params)
    optimizer = build_optimizer(
        params=params,
        model_params=model.parameters(),
    )
    scheduler = build_scheduler(optimizer, params)
    return model, optimizer, scheduler


# __all__ = ["compile_model"]
