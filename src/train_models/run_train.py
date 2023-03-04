from typing import Any

import mlflow
import torch
from dagster import In, config_mapping, job, op
from omegaconf import DictConfig, OmegaConf

from src.train_models.data_ops import (
    build_dataloaders,
    get_params,
)
from src.train_models.model_ops import (
    compile_model,
    get_criterion,
)
from src.train_models.train_utils import (
    mini_batch_simclr,
    train_simclr_fn,
)
from src.train_models.utils import (
    seed_everything,
    set_mlflow,
    to_device,
)


@op(ins={"params": In(DictConfig)})
def set_env_vars(params):
    """Method to set proper environment variables and setup experiment"""
    seed_everything(params.seed)
    set_mlflow()


@config_mapping(config_schema={"path": str})
def simplified_config(val):
    """Config mapping required by dagster job"""
    return {
        "ops": {
            "get_params": {
                "config": {"path": val["path"]}
            }
        }
    }


@op(
    ins={
        "params": In(DictConfig),
        "model": In(torch.nn.Module),
        "criterion": In(torch.nn.Module),
        "optimizer": In(torch.optim.Optimizer),
        "scheduler": In(Any),
        "dataloader": In(
            torch.utils.data.DataLoader
        ),
    }
)
def trainer(
    context,
    params: DictConfig,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    dataloader: torch.utils.data.DataLoader,
):
    """Method to train the model and log everything using mlflow"""
    with mlflow.start_run(
        run_name=params.run_name
    ):
        # log hyperparams
        mlflow.log_params(
            OmegaConf.to_container(params)
        )
        # Move model to device
        model = to_device(
            x=model, device=params.device
        )
        # Get train function
        step = train_simclr_fn(
            model=model,
            optimizer=optimizer,
            loss_fn=criterion,
            device=params.device,
            scheduler=scheduler,
        )
        # Training loop
        for epoch in range(params.epoch):
            loss = mini_batch_simclr(
                dataloader=dataloader,
                step_fn=step,
                params=params,
            )
            context.log.info(
                f"Epoch : {epoch} || Loss : {loss}"
            )
            mlflow.log_metric(
                "loss", loss, step=epoch
            )
            # log model
            mlflow.pytorch.log_model(
                model,
                artifact_path=f"simclr-{params.model.backbone}",
                registered_model_name=f"simclr-{epoch}-{params.model.backbone}",
            )


@job(config=simplified_config)
def run_training():
    """Job to run training from the specified config"""
    params = get_params()
    # Set env variables
    set_env_vars(params)
    train_dl = build_dataloaders(params)
    # Define criterion
    criterion = get_criterion(params)
    # Prepare model
    model, optimizer, scheduler = compile_model(
        params
    )
    trainer(
        params=params,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=train_dl,
    )


if __name__ == "__main__":
    run_training.execute_in_process(
        run_config={
            "path": "../params/ssl.yaml",
        }
    )
