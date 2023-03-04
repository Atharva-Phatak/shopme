import os
from typing import List

import lightly
import torch
from dagster import In, Out, op
from omegaconf import DictConfig, OmegaConf


@op(config_schema={"path": str})
def get_params(context) -> DictConfig:
    """Get parameters for training from config file"""
    return OmegaConf.load(
        context.op_config["path"]
    )


@op(ins={"params": In(DictConfig)})
def simclr_collate_fn(
    params: DictConfig,
) -> lightly.data.ImageCollateFunction:
    """Method to return collate function required by SimCLR"""
    return lightly.data.SimCLRCollateFunction(
        input_size=params.input_size,
        vf_prob=0.5,
        rr_prob=0.5,
    )


@op(ins={"params": In(DictConfig)})
def build_paths(params: DictConfig):
    """Get training files from the local dir"""
    train_files = os.listdir(
        f"{params.data.path}/"
    )
    return train_files


@op(
    ins={
        "params": In(DictConfig),
        "train_files": In(List[str]),
    }
)
def build_train_dl(
    train_files: List[str], params: DictConfig
) -> torch.utils.data.DataLoader:
    """Method to create training dataloader"""
    # create a dataloader for training
    ds = lightly.data.LightlyDataset(
        input_dir=f"{params.data.path}/",
        filenames=train_files,
    )
    collate_fn = simclr_collate_fn(params)
    dataloader_train_simsiam = (
        torch.utils.data.DataLoader(
            ds,
            batch_size=params.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=params.num_workers,
        )
    )
    return dataloader_train_simsiam


@op(
    ins={"params": In(DictConfig)},
    out={
        "train_dl": Out(
            torch.utils.data.DataLoader
        )
    },
)
def build_dataloaders(params):
    """Method to create dataloader from a given config"""
    train_paths = build_paths(params)
    train_dl = build_train_dl(train_paths, params)
    return train_dl


# __all__ = ["build_dataloaders"]
