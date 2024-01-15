import os
import matplotlib.pyplot as plt
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from functools import partial
import torch


from datasets.data_loader import get_loader
from utilities.utility import trainer 


# Path to save trained models 
root_dir = "/home/jacobo/MultiOrganSeg/trained_models"

# Path to the train dataset 
data_dir = "/media/jacobo/NewDrive/Hemin_Collection/BraTS2021"

# Path to the JSON file for a list of training samples 
json_list = "/media/jacobo/NewDrive/Hemin_Collection/BraTS2021/brats21_folds.json"


# Model hyper-parameters 
roi = (128, 128, 128)
batch_size = 1
sw_batch_size = 4
fold = 1
infer_overlap = 0.5
max_epochs = 2
val_every = 1


# Split the taining dataset to train and validation subsets and load the data with pytroch dataloader function 
train_loader, val_loader = get_loader(batch_size, data_dir, json_list, fold, roi)

# Set the enviroment for the cuda 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Check for GPU and set it if available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Swin UNETR model for the 3-class brain tumor semantic segmentation
model = SwinUNETR(
    img_size=roi,       
    in_channels=4,
    out_channels=3,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True,
).to(device)



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

start_epoch = 0 

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

### Plot the loss and Dice metric
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
plt.xlabel("epoch")
plt.plot(trains_epoch, loss_epochs, color="red")
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
plt.xlabel("epoch")
plt.plot(trains_epoch, dices_avg, color="green")
plt.show()
plt.figure("train", (18, 6))
plt.subplot(1, 3, 1)
plt.title("Val Mean Dice TC")
plt.xlabel("epoch")
plt.plot(trains_epoch, dices_tc, color="blue")
plt.subplot(1, 3, 2)
plt.title("Val Mean Dice WT")
plt.xlabel("epoch")
plt.plot(trains_epoch, dices_wt, color="brown")
plt.subplot(1, 3, 3)
plt.title("Val Mean Dice ET")
plt.xlabel("epoch")
plt.plot(trains_epoch, dices_et, color="purple")
plt.show()