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
from .helper import AverageMeter, save_checkpoint, normalize_3d_scan


def train_epoch(
        model,
        writer, 
        scheduler,
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
        writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)

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

    return data, target, val_output_convert, run_acc.avg



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
            scheduler,
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
            inputs, targets, outputs, val_acc = val_epoch(
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


            print(outputs[0].shape)
            inputs = normalize_3d_scan(inputs)

            print(inputs.shape)
            print(targets.shape)

            writer.add_scalar("test/Average Accuracy:", scalar_value=torch.tensor(val_avg_acc), global_step=epoch)

            
            for Idx in range(100,105):
                #writer.add_image('Predicted class 1: {}'.format(Idx), outputs[0][0][:][:][Idx], global_step=global_step, dataformats='HW')
                writer.add_image('Predicted class 1: {}'.format(Idx), outputs[0][0][:][:][Idx], global_step=global_step, dataformats='HW')
                writer.add_image('Predicted class 2: {}'.format(Idx), outputs[0][1][:][:][Idx], global_step=global_step, dataformats='HW')
                writer.add_image('Predicted class 3: {}'.format(Idx), outputs[0][2][:][:][Idx], global_step=global_step, dataformats='HW')
                
                print(inputs[0][0][:][:][Idx].max())
                print(inputs[0][0][:][:][Idx].min())

                writer.add_image('Input {}'.format(Idx), inputs[0][0][:][:][Idx], global_step=epoch, dataformats='WH')

                writer.add_image('GT class 1 {}'.format(Idx), targets[0][0][:][:][Idx], global_step=epoch, dataformats='HW')
                writer.add_image('GT class 2 {}'.format(Idx), targets[0][1][:][:][Idx], global_step=epoch, dataformats='HW')
                writer.add_image('GT class 3 {}'.format(Idx), targets[0][2][:][:][Idx], global_step=epoch, dataformats='HW')

                #writer.add_images('Ground Truth Class 1: {}'.format(Idx), targets[0][0][:,:,Idx], global_step=epoch)

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