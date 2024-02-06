"""
Author: Hemin Qadir
Date: 25.01.2024
Task: this is to lunch model training
"""

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    Compose,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
import torch
import os
import glob
from utilities.helper import count_parameters
from utilities.utils import trainer 
from dataset.prepare_data import data_loader_and_transforms
from pathlib import Path


# Path to save trained models 
root_dir = "/home/heminq/spleen_segmentation"

model_dir = os.path.join(root_dir, "models")

if not os.path.exists(model_dir):
    # Create the directory if it doesn't exist
    os.makedirs(model_dir)
    print(f"Directory '{model_dir}' created successfully.")
else:
    print(f"Directory '{model_dir}' already exists.")


max_epochs = 600
val_interval = 10
pix_dim = (1.5, 1.5, 2.0)    #pixdim=(1.5, 1.5, 1.0)
a_min = -200
a_max =  200
spatial_size = (96, 96, 96)
cache =  True

# Path to the train dataset 
data_dir = "/home/heminq/Decath_Spleen/Task09_Spleen"
train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files, val_files = data_dicts[:-9], data_dicts[-9:]


# For reproducibility
set_determinism(seed=0)


train_loader, val_loader = data_loader_and_transforms(train_files, val_files, spatial_size, pix_dim, a_min, a_max, do_cache=cache)

# Set the enviroment for the cuda 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Check for GPU and set it if available 
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


print("*"*10)
print(model)
num_params = count_parameters(model)
print("Total number of model parameters: {} M".format(num_params))
print("*"*10)


# Check for the last checkpoint in the model directory 
root = Path(model_dir)
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
loss_function = loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = optimizer = torch.optim.Adam(model.parameters(), 1e-4) #torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

dice_metric = DiceMetric(include_background=False, reduction="mean")

post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])


trainer(model, 
        train_loader, 
        val_loader, 
        optimizer, 
        loss_function, 
        start_epoch, 
        max_epochs, 
        post_pred, 
        post_label, 
        dice_metric, 
        val_interval, 
        model_dir, 
        device)

print("Training is done")