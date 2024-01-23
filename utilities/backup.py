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
from monai.networks.nets import SwinUNETR
from functools import partial
import torch


from datasets.data_loader import get_loader
from utilities.utility import trainer 


# Path to save trained models 
root_dir = "/home/jacobo/MultiOrganSeg/trained_models"

if not os.path.exists(root_dir):
    # Create the directory if it doesn't exist
    os.makedirs(root_dir)
    print(f"Directory '{root_dir}' created successfully.")
else:
    print(f"Directory '{root_dir}' already exists.")

# Path to the train dataset 
data_dir = "/media/jacobo/NewDrive/Hemin_Collection/BraTS2021"

# Path to the JSON file for a list of training samples 
json_list = "/media/jacobo/NewDrive/Hemin_Collection/BraTS2021/brats21_folds.json"


# Model hyper-parameters and input size 
roi = (128, 128, 128)
batch_size = 1
sw_batch_size = 4
fold = 1
infer_overlap = 0.5
max_epochs = 100
val_every = 1

# Augmentation methods that will be applied on the training subset
train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
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
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )


# Split the taining dataset to train and validation subsets and load the data with pytroch dataloader function 
train_loader, val_loader = get_loader(batch_size, data_dir, json_list, fold, train_transform, val_transform)

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

























"""
Author: Hemin Qadir
Date: 15.01.2024
Task: Here you can find a set of utilities functions

"""

import time
import torch
import os
import numpy as np
from monai.data import decollate_batch
from torch.utils.tensorboard import SummaryWriter
from .helper import AverageMeter, save_checkpoint

def train_epoch(
        model,
        writer, 
        global_step,
        loader, 
        optimizer, 
        epoch,
        loss_func,
        device="cpu",
        batch_size=1, 
        max_epochs=100,
):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        global_step += 1
        run_loss.update(loss.item(), n=batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )


        writer.add_scalar("train/loss", scalar_value=run_loss.avg, global_step=global_step)
        #writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

        start_time = time.time()
    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
    device="cpu",
    max_epochs=100,
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    return run_acc.avg



# Define Trainer 
def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    max_epochs=10,
    val_every=5,
    post_sigmoid=None,
    post_pred=None,
    device="cpu",
    batch_size=1,
    save_dir="./"
):
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []

    name = "multiorgans"
    writer = SummaryWriter(log_dir=os.path.join("logs", name))

    global_step = 0
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            writer,
            global_step,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
            device=device,
            batch_size=batch_size, 
            max_epochs=max_epochs,
        )
        
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
                device=device,
                max_epochs=max_epochs,
            )

            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            dice_et = val_acc[2]
            val_avg_acc = np.mean(val_acc)
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", Dice_Avg:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            writer.add_scalar("test/dices_tc", scalar_value=dices_tc, global_step=epoch)

            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model,
                    epoch,
                    best_acc=val_acc_max,
                    dir_add=save_dir,
                )
            scheduler.step()


    writer.close()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )