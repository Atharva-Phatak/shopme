import lightly
from dagster import In, op
from torchvision import transforms


@op(
    ins={
        "files": In(List[str]),
        "params": In(DictConfig),
    }
)
def build_dl(files, params):
    """Method to build the testing dataloader"""
    transforms = test_transforms(params)
    ds = lightly.data.LightlyDataset(
        input_dir=f"{params.data_path}/",
        filenames=files,
        transform=transforms,
    )
    dataloader_test = torch.utils.data.DataLoader(
        ds,
        batch_size=params.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=params.num_workers,
    )
    return dataloader_test
