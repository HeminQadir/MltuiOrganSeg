"""
Author: Hemin Qadir
Date: 15.01.2024
Task: this is to lunch model training

"""
import os
from pathlib import Path
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Activations,
)
   
from monai import transforms
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR, UNet
from functools import partial
import torch

from datasets.data_loader import get_loader
from utilities.utility import trainer 
from utilities.helper import count_parameters

# Path to save trained models 
root_dir = "/home/jacobo/MultiOrganSeg/trained_models"

if not os.path.exists(root_dir):
    # Create the directory if it doesn't exist
    os.makedirs(root_dir)
    print(f"Directory '{root_dir}' created successfully.")
else:
    print(f"Directory '{root_dir}' already exists.")

# Path to the train dataset 
data_dir = "/media/jacobo/NewDrive/Hemin_Collection/BraTS2021/"

# Path to the JSON file for a list of training samples 
json_list = "/home/jacobo/MultiOrganSeg/training_data.json" #"/media/jacobo/NewDrive/Hemin_Collection/BraTS2021/brats21_folds.json"

# Model hyper-parameters and input size 
roi = (128, 128, 128)
batch_size = 1
sw_batch_size = 4
fold = 1
infer_overlap = 0.5
max_epochs = 100
val_every = 1
in_channels = 1
out_channels = 3
use_checkpoint = True

# Augmentation methods that will be applied on the training subset
train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.AddChanneld(keys=["image"]),    # This is needed if in_channels = 1, label does not need this becuase it is 3 classes in case of one class we needed it.
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi[0], roi[1], roi[2]],
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi[0], roi[1], roi[2]],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )

# Augmentation methods that will be applied on the validation subset
val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.AddChanneld(keys=["image"]), # This is needed if in_channels = 1, label does not need this becuase it is 3 classes in case of one class we needed it. 
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )


# Split the taining dataset to train and validation subsets and load the data with pytroch dataloader function 
train_loader, val_loader = get_loader(batch_size, data_dir, json_list, fold, train_transform, val_transform)

# Set the enviroment for the cuda 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Check for GPU and set it if available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Help for SwinUNETR
# class monai.networks.nets.SwinUNETR(img_size, in_channels, out_channels, 
#     depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), 
#     feature_size=24, norm_name='instance', 
#     drop_rate=0.0, attn_drop_rate=0.0, 
#     dropout_path_rate=0.0, normalize=True, 
#     use_checkpoint=False, spatial_dims=3, 
#     downsample='merging', use_v2=False)



SWIN_UNETR = True
UNET = False


# Create Swin UNETR model for the 3-class brain tumor semantic segmentation
if SWIN_UNETR:
    model = SwinUNETR(
        img_size=roi,       
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=use_checkpoint,
    ).to(device)


if UNET:
    model = UNet(
        spatial_dims=3,  # To create a 3D UNet
        in_channels=in_channels,   # In this example, we have 4 sequences 
        out_channels=out_channels,  # In this example, we have 3 classes 
        channels=(4, 6, 16), #, 512),  
        strides=(2, 2),  
        kernel_size=3,    #(2, 2, 2, 2) # this arg can be a single number or a sequence 
        up_kernel_size=3, #this arg can be a single number or a sequence 
        num_res_units=2, 
        act='PRELU', 
        norm='INSTANCE', 
        dropout=0.0, 
        bias=True, 
        adn_ordering='NDA'
    ).to(device)


print(model)
num_params = count_parameters(model)
print("Total number of model parameters: {} M".format(num_params))
print("")


# Check for the last checkpoint in the model directory 
root = Path(root_dir)
model_path = root / 'model.pt'
if model_path.exists():    
    state = torch.load(str(model_path))
    start_epoch = state['epoch']
    model.load_state_dict(state['state_dict'])
    print('The model is restored at epoch {}'.format(start_epoch))
else:
    start_epoch = 0 

print("Strating epoch is: {}".format(start_epoch))


# Define the optimizer and loss function
torch.backends.cudnn.benchmark = True
dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)
dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
model_inferer = partial(
    sliding_window_inference,
    roi_size=[roi[0], roi[1], roi[2]],
    sw_batch_size=sw_batch_size,
    predictor=model,
    overlap=infer_overlap,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


# Exexute training 
(   val_acc_max,
    dices_tc,
    dices_wt,
    dices_et,
    dices_avg,
    loss_epochs,
    trains_epoch,
) = trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_func=dice_loss,
    acc_func=dice_acc,
    scheduler=scheduler,
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    max_epochs=max_epochs,
    val_every=val_every,
    post_sigmoid=post_sigmoid,
    post_pred=post_pred,
    device=device,
    batch_size=batch_size,
    save_dir=root_dir,
)