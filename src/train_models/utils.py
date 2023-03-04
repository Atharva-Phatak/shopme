import os
import random

import mlflow
import numpy as np
import torch
from dotenv import load_dotenv


load_dotenv("../params/.env")


def set_mlflow():
    if (
        mlflow.get_experiment_by_name(
            os.getenv("MLFLOW_EXPERIMENT_NAME")
        )
        is None
    ):
        mlflow.create_experiment(
            os.getenv("MLFLOW_EXPERIMENT_NAME"),
            artifact_location=os.getenv(
                "MLFLOW_ARTIFACT_LOCATION"
            ),
        )
        mlflow.set_experiment(
            os.getenv("MLFLOW_EXPERIMENT_NAME")
        )


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def to_device(x, device):
    return x.to(device)


__all__ = [
    "set_mlflow",
    "seed_everything",
    "to_device",
]
