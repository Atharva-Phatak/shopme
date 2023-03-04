import os
from typing import List

import numpy as np
import torch
from dagster import (
    In,
    Out,
    config_mapping,
    job,
    op,
)
from omegaconf import DictConfig, OmegaConf

from converter import (
    prepare_onnx_model,
    sklearn_to_bento,
    torch_to_bento,
)
from data import build_dl


@op(config_schema={"path": str})
def get_inference_params(path):
    """Load yaml file for inference"""
    return OmegaConf.load(path)


@op(ins={"params": In(DictConfig)})
def load_embedding_model(params):
    """Load the trained PyTorch model"""
    model = torch.load(params.artifact_path)
    return model


@op(
    ins={
        "model": In(torch.nn.Module),
        "dl": In(torch.utils.data.DataLoader),
        "params": In(DictConfig),
    },
    out={
        "embeddings": Out(np.ndarray),
        "filenames": Out(List[str]),
    },
)
def create_embeddings(model, dl, params):
    """Generate embedding for collecte data"""
    embeddings = []
    filenames = []
    model.to(params.device)
    model.eval()
    with torch.no_grad():
        for img, _, fnames in dl:
            img = img.to(params.device)
            emb = model.backbone(img).flatten(
                start_dim=1
            )
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = embeddings.cpu().numpy()
    embeddings = normalize(embeddings)
    return embeddings, filenames


@op(ins={"product_embeddings": In(np.ndarray)})
def fit_knn(product_embeddings):
    """Fit NearestNeighbors on the generated embeddings"""
    nbrs = NearestNeighbors(n_neighbors=5).fit(
        product_embeddings
    )
    return nbrs


@op(ins={"params": In(DictConfig)})
def get_product_images(params):
    return os.listdir(params.data_path)


@op(
    ins={"params": In(DictConfig)},
    out={
        "knn_model": Out(NearestNeighbors),
        "model": Out(torch.nn.Module),
        "filenames": Out(List[str]),
    },
)
def fit_models(params) -> tuple:
    """Fit the models"""
    model = load_embedding_model(params)
    file_paths = get_product_images(params)
    dl = build_dl(file_paths, params)
    (
        product_embeddings,
        filenames,
    ) = create_embeddings(model, dl, params)
    knn_model = fit_knn(product_embeddings)
    return knn_model, model, filenames


@config_mapping(config_schema={"path": str})
def simplified_config(val):
    return {
        "ops": {
            "configure_models": {
                "config": {"path": val["path"]}
            }
        }
    }


@job(config=simplified_config)
def post_training_prep():
    params = get_inference_params()
    (
        knn_model,
        torch_model,
        filenames,
    ) = fit_models(params)
    prepare_onnx_model(torch_model, params)
    onnx_to_bento(params)
    sklearn_to_bento(knn_model, params, filenames)


if __name__ == "__main__":
    post_training_prep.execute_in_process(
        run_config={
            "path": "../configs/inference_params.yaml"
        }
    )
