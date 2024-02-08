"""
Author: Hemin Qadir
Date: 15.01.2024
Task: helper functions for several tasks  

"""

import os
import numpy as np  
import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add="root_dir"):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def normalize_3d_scan(scan):
    min_val = torch.min(scan)
    max_val = torch.max(scan)
    normalized_scan = (scan - min_val) / (max_val - min_val)
    return normalized_scan


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000
