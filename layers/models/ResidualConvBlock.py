from ResidualConv import *

class ResidualConvBlock(nn.Module):
    def __init__(self, kernel, num_layers, input_channel, pool):
        super().__init__()
        self.first_conv = ResidualConv(kernel, input_channel, input_channel, pool)
        self.res_conv = nn.ModuleList([ResidualConv(kernel, input_channel, input_channel, False) for _ in range(num_layers-1)])

    def forward(self, x):
        x = self.first_conv(x)
        for l in self.res_conv:
            x = l(x)

        return x

