"""
Author: Hemin Qadir
Date: 15.01.2024
Task: data loader function  

"""

from monai import data
from .data_split_fold import datafold_read


def get_loader(batch_size, data_dir, json_list, fold, train_transform, val_transform):
    data_dir = data_dir
    datalist_json = json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)

    print(train_files)

    train_ds = data.Dataset(data=train_files, transform=train_transform)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, val_loader