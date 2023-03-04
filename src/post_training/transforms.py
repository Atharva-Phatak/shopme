import lightly
import torchvision
from dagster import op
from omegaconf import DictConfig


@op(ins={"params": In(DictConfig)})
def test_transforms(
    params,
) -> torchvision.transforms.Compose:
    """Return transforms as required by the model."""
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (
                    params.input_size,
                    params.input_size,
                )
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=lightly.data.collate.imagenet_normalize[
                    "mean"
                ],
                std=lightly.data.collate.imagenet_normalize[
                    "std"
                ],
            ),
        ]
    )
