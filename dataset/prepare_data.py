from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import numpy as np
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    SaveImaged,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    Invertd,
)


def data_loader_and_transforms(train_files, val_files, spatial_size, pix_dim, a_min, a_max, do_cache=True):

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], 
                a_min=a_min, 
                a_max=a_max, 
                b_min=0.0, 
                b_max=1.0, 
                clip=True,
                ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=pix_dim, mode=("bilinear", "nearest")),
            # RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=spatial_size,
            #     pos=1,
            #     neg=1,
            #     num_samples=4,
            #     allow_smaller=True,
            #     image_key="image",
            #     image_threshold=0,
            # ),
            Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
            # user can also add other random transforms
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=(96, 96, 96),
                rotate_range=(0, 0, np.pi/15),
                scale_range=(0.1, 0.1, 0.1)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], 
                a_min=a_min, 
                a_max=a_max, 
                b_min=0.0, 
                b_max=1.0, 
                clip=True,
                ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=pix_dim, mode=("bilinear", "nearest")),
            Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
        ]
    )

    

    if do_cache:
        # use batch_size=2 to load images and use RandCropByPosNegLabeld
        # to generate 2 x 4 images for network training
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)


    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    

    return train_loader, val_loader
