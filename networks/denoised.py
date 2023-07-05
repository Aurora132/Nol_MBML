import torch
import torch.nn as nn


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        return self.conv(x)


class Denoise(nn.Module):
    def __init__(self, n_channels=64):
        super(Denoise, self).__init__()
        self.n_channels = n_channels

        self.down1 = Down(1, n_channels)
        self.down2 = Down(n_channels, n_channels)
        self.down3 = Down(n_channels, n_channels)
        self.down4 = Down(n_channels, n_channels)
        self.down5 = Down(n_channels, n_channels)
        self.down6 = Down(n_channels, n_channels)
        self.down7 = Down(n_channels, n_channels)
        self.down8 = Down(n_channels, n_channels)
        self.down9 = Down(n_channels, n_channels)
        self.down10 = Down(n_channels, n_channels)
        self.down11 = Down(n_channels, n_channels)
        self.down12 = Down(n_channels, n_channels)
        self.outc = OutConv(n_channels, 1)



    def forward(self, x):

        x_first = x
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.down6(x)
        x = self.down7(x)
        x = self.down8(x)
        x = self.down9(x)
        x = self.down10(x)
        x = self.down11(x)
        x = self.down12(x)
        logits = self.outc(x)
        return logits + x_first



if __name__ == "__main__":
    import numpy as np
    x = torch.from_numpy(np.zeros((10, 1, 24, 24), dtype=np.float32))  
    print(x.shape)
    net = Denoise(n_channels=64)
    y = net(x)
    print(y.shape)
