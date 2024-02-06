
import torch 
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from .helper import save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import os 
import matplotlib.pyplot as plt


def validation(model, writer, epoch, val_loader, device, post_pred, post_label, dice_metric):
    model.eval()
    with torch.no_grad():
        for j, val_data in enumerate(val_loader):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            
            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)

            #val_labels = val_labels != 0 # This is very important 

            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

            val_outputs_ = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels_ = [post_label(i) for i in decollate_batch(val_labels)]

            #print("#"*50)
            #print(val_outputs_[0].shape)
            #print("#"*50)
            #print(val_labels_[0].shape)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs_, y=val_labels_)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            writer.add_image('Prediction {}'.format(j), torch.argmax(val_outputs, dim=1)[0, :, :, 80], global_step=epoch, dataformats='HW')

            writer.add_image('Input Scan {}'.format(j), val_inputs[0, 0, :, :, 80], global_step=epoch, dataformats='WH')

            writer.add_image('Ground Truth {}'.format(j), val_labels[0, 0, :, :, 80], global_step=epoch, dataformats='HW')

    return metric


def trainer(model, train_loader, val_loader, optimizer, loss_function, start_epoch, max_epochs, post_pred, post_label, dice_metric, val_interval, root_dir, device):
    best_metric = -1
    best_metric_epoch = -1
    # metric_values = []
    epoch_loss_values = []
    global_step = 0

    name = "spleen"
    
    writer = SummaryWriter(log_dir=os.path.join("logs", name))

    for epoch in range(start_epoch, max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            
            inputs = batch_data["image"].to(device)

            labels = batch_data["label"].to(device)

            # label = label != 0 # This is very important 

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step+=1 
            
            print(f"step/epoch {global_step}/{epoch}, " f"train_loss: {loss.item():.4f}")
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        
        writer.add_scalar("train/loss", scalar_value=torch.tensor(epoch_loss), global_step=global_step)
        #writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)

        if (epoch + 1) % val_interval == 0:
            metric =  validation(model, writer, epoch, val_loader, device, post_pred, post_label, dice_metric)

            # metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1

                save_checkpoint(model, 
                                epoch, 
                                best_acc = best_metric, 
                                dir_add = root_dir)
                #torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

            writer.add_scalar("test/Average Accuracy:", scalar_value=torch.tensor(metric), global_step=epoch+1)