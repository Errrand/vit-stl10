from pathlib import Path

import torch
from torch.utils.data import DataLoader,random_split,Subset
from torchvision import datasets,transforms

root=Path(__file__).resolve().parents[1]/"data"/"STL_10"
root.mkdir(parents=True,exist_ok=True)

STL10_MEAN=(0.4467,0.4398,0.4066)
STL10_STD=(0.2603,0.2566,0.2713)

def get_transforms():
    train_transform=transforms.Compose([
        transforms.RandomResizedCrop(
            size=96,
            scale=(0.8,1.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN,STL10_STD)
    ])
    val_transform=transforms.Compose([
        transforms.Resize((96,96)),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN,STL10_STD)

    ])
    test_transform=transforms.Compose([
        transforms.Resize((96,96)),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN,STL10_STD)
    ])
    return train_transform,val_transform,test_transform
def get_dataloaders(batch_size=64,num_workers=4,train_size=4500,val_size=500):
    train_transform,val_transform,test_transform=get_transforms()
    pin_memory=torch.cuda.is_available()
    full_train_dataset=datasets.STL10(
        root=str(root),
        split="train",
        download=True,
        transform=train_transform
    )
    full_val_dataset=datasets.STL10(
        root=str(root),
        split="train",
        download=False,
        transform=val_transform
    )
    test_dataset=datasets.STL10(
        root=str(root),
        split="test",
        download=True,
        transform=test_transform
    )
    train_indices,val_indices=random_split(
        range(len(full_train_dataset)),
        [train_size,val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_dataset=Subset(full_train_dataset,train_indices.indices)
    val_dataset=Subset(full_val_dataset,val_indices.indices)

    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,
        drop_last=True
    )
    val_loader=DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,
        drop_last=False
    )
    test_loader=DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,
        drop_last=False
    )
    return train_loader,val_loader,test_loader

