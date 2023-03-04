from src.train_models.model_ops import (
    get_criterion,
    build_model,
    build_optimizer,
)
from src.train_models.data_ops import get_params
from src.train_models.train_utils import train_simclr_fn
import torch
from dagster import build_op_context


def get_dummy_model():
    return torch.nn.Linear(5, 1)


def test_criterion():
    context = build_op_context(
        config={"path": "./configs/ssl.yaml"}
    )
    params = get_params(context)
    assert (
        isinstance(
            get_criterion(params), torch.nn.Module
        )
        is True
    )


def test_build_model():
    context = build_op_context(
        config={"path": "./configs/ssl.yaml"}
    )
    params = get_params(context)
    model = build_model(params)
    assert (
        isinstance(model, torch.nn.Module) is True
    )
    rand_input = torch.randn(
        1, 3, params.input_size, params.input_size
    )
    op = model(rand_input)
    assert isinstance(op, torch.Tensor) is True


def test_build_optimizer():
    context = build_op_context(
        config={"path": "./configs/ssl.yaml"}
    )
    params = get_params(context)
    model = get_dummy_model()
    optimizer = build_optimizer(
        params, model.parameters()
    )
    assert (
        isinstance(
            optimizer, torch.optim.Optimizer
        )
        is True
    )

def test_forward_pass():
    context = build_op_context(
        config={"path": "./configs/ssl.yaml"}
    )
    model, optimizer, scheduler = compile_model(
        params
    )
