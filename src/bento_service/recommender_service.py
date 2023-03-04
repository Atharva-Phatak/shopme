import albumentations as A
import numpy as np
from bentoml.io import JSON, Image
from sklearn.preprocessing import normalize

from constructs import ServiceOutput


# creating runners for bentoapp
embedder = bentoml.onnx.get(
    "bento-product-embedder:latest"
).to_runner()
nearest_neighbors = bentoml.sklearn.get(
    "fashion-knn:latest"
)
files = nearest_neighbors.custom_objects[
    "product_filenames"
]
knn_runner = nearest_neighbors.to_runner()

# Creating bento service
svc = bentoml.Service(
    "fashion-recommender-service",
    runners=[embedder, knn_runner],
)

# Output specification
output_spec = JSON(pydantic_model=ServiceOutput)


# albumentations augmentation required by the model
albumentations_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def process_numpy_image(img: np.array):
    """Add batch size to image"""
    img = np.moveaxis(img, -1, 0)
    img = np.expand_dims(img, 0)
    return img


# bentoml app service
@svc.api(input=Image(), output=output_spec)
async def predict(
    query_image: Image,
) -> output_spec:
    """Return nearest neighbors and the corresponding distances"""
    query_image = np.array(query_image)
    query_image = albumentations_transform(
        image=query_image
    )["image"]
    query_image = process_numpy_image(query_image)
    query_embedding = await embedder.async_run(
        query_image
    )
    query_embedding = query_embedding.flatten()
    query_embedding = query_embedding.reshape(
        1, -1
    )
    query_embedding = normalize(query_embedding)
    (
        distances,
        indices,
    ) = await knn_runner.kneighbors.async_run(
        query_embedding
    )
    recommendations = [
        files[indx] for indx in indices[0]
    ]
    return {
        "distances": distances,
        "recommendations": recommendations,
    }
