import os
import numpy as np
import errno
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch
from datetime import datetime

'''
    TensorBoard Data will be stored in './runs' path
'''


class Logger:

    def __init__(self, model_name):
        self.model_name = model_name
        now = datetime.now().strftime("%d/%m %H:%M")
        
        comment = f"{self.model_name} {now}"
        
        logdir = f'runs/{comment}'

        self.writer = SummaryWriter(log_dir=logdir,comment=comment)
        
    def log_scalar(self, episode, scalar, name):
        self.writer.add_scalar(name,scalar,episode)

    def log_image(self, image_name, image, episode = 0):
        self.writer.add_image(image_name, image, global_step = episode)

    def close(self):
        self.writer.close()
