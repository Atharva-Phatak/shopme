from typing import List

import bentoml
import onnx
import torch
import torchvision
from dagster import In, op
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


@op(ins={"params": In(DictConfig)})
def onnx_to_bento(params):
    """Convert the onnx model to bento-model"""
    onnx_model = onnx.load(params.onnx_model)
    signatures = {
        "run": {"batchable": True},
    }
    bentoml.onnx.save_model(
        params.bento_embedding_model,
        onnx_model,
        signatures=signatures,
    )


@op(
    ins={
        "knn_model": In(NearestNeighbors),
        "params": In(DictConfig),
        "filenames": In(List[str]),
    }
)
def sklearn_to_bento(
    knn_model, params, filenames
):
    """Method onvert sklearn to bento-model"""
    signature = {
        "kneighbors": {"batchable": False}
    }
    bentoml.sklearn.save_model(
        params.knn_model,
        knn_model,
        custom_objects={
            "product_filenames": filenames
        },
        signatures=signature,
    )


@op(
    ins={
        "model": In(torch.nn.Module),
        "params": In(DictConfig),
    }
)
def prepare_onnx_model(model, params):
    """Method to convert torch model to onnx"""
    model.eval()
    rand_input = torch.randn(3, 256, 256)
    rand_input = rand_input.unsqueeze_(0)
    model = model.to("cpu")
    torch.onnx.export(
        model.backbone,
        rand_input,
        params.onnx_model,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=[
            "input"
        ],  # the model's input names
        output_names=[
            "output"
        ],  # the model's output names
        dynamic_axes={
            "input": {
                0: "batch_size"
            },  # variable length axes
            "output": {0: "batch_size"},
        },
    )
