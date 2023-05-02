import torch
import torch.nn as nn
import numpy as np

class InitialConv(nn.Module):
    def __init__(self, conv_kernel, pool_kernel, input_channel, output_channel):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, conv_kernel, stride = 2, padding = 4),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(pool_kernel, 2,padding = 1)
        )

    def forward(self,x):
        x = self.initial_conv(x)
        return x
        

    
