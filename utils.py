import os
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm

import torch
import torch.nn as nn


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.clone().detach()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
