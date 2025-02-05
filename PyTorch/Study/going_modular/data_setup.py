"""
This code creates "zaebatelskie" dataloaders
"""


import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKAS = os.cpu_count()

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int=NUM_WORKAS
):
    train_data = datasets.ImageFolder(train_dir, transform)
    test_data = datasets.ImageFolder(test_dir, transform)
    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader, class_names
