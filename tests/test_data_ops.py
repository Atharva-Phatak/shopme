from src.train_models.data_ops import (
    get_params,
    build_dataloaders,
    build_paths,
    build_train_dl,
)
from dagster import (
    build_op_context,
    job,
    config_mapping,
    op,
    In,
)
from omegaconf import DictConfig, OmegaConf


def test_job():
    context = build_op_context(
        config={"path": "./configs/ssl.yaml"}
    )
    op = get_params(context)
    assert isinstance(op, DictConfig)
