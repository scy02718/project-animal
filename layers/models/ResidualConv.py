import torch
import torch.nn as nn

class ResidualConv(nn.Module):
    def __init__(self, kernel, input_channel, output_channel, pool = False, dropout = 0.0):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel, stride = 2 if pool else 1, padding = 1 if pool else "same"),
            nn.Dropout(dropout),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(input_channel, output_channel, kernel, stride = 1, padding = "same"),
            nn.BatchNorm2d(output_channel)
        )

        self.pool = pool
        self.relu = nn.ReLU(inplace = True)
        self.optional_pool = nn.MaxPool2d(kernel,2,1)

    def forward(self, x):
        conv_out = self.conv_net(x)
        if self.pool:
            x = self.optional_pool(x)
        x = x + self.relu(conv_out)

        return x


a = torch.rand(1,3,8,8)
b = ResidualConv(3,3,3,pool = True)
print(a.size())
a = b(a)
print(a.size())
