import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=24, Ablation_att=False):
        super(CoordAtt, self).__init__()
        self.flag = Ablation_att
        if not self.flag:
            self.pool_h_Avg = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w_Avg = nn.AdaptiveAvgPool2d((1, None))
            self.pool_h_Max = nn.AdaptiveMaxPool2d((None, 1))
            self.pool_w_Max = nn.AdaptiveMaxPool2d((1, None))

            self.fc_h = nn.Sequential(nn.Conv2d(2 * inp, inp // 16, 1, bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(inp // 16, inp, 1, bias=False))

            self.fc_w = nn.Sequential(nn.Conv2d(2 * inp, inp // 16, 1, bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(inp // 16, inp, 1, bias=False))

            mip = max(8, inp // reduction)

            self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
            self.bn1 = nn.BatchNorm2d(mip)
            self.act = h_swish()

            self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
            self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if not self.flag:
            identity = x

            n, c, h, w = x.size()
            x_h_Avg = self.pool_h_Avg(x)
            x_w_Avg = self.pool_w_Avg(x).permute(0, 1, 3, 2)
            x_h_Max = self.pool_h_Max(x)
            x_w_Max = self.pool_w_Max(x).permute(0, 1, 3, 2)

            x_h = torch.concat((x_h_Max, x_h_Avg), dim=1)  # n,c,h,1
            x_h = self.fc_h(x_h)
            x_w = torch.concat((x_w_Max, x_w_Avg), dim=1)  # n,c,w,1
            x_w = self.fc_h(x_w)

            y = torch.cat([x_h, x_w], dim=2)  # n,c,(h+w),1
            y = self.conv1(y)
            y = self.bn1(y)
            y = self.act(y)

            x_h, x_w = torch.split(y, [h, w], dim=2)
            x_w = x_w.permute(0, 1, 3, 2)

            a_h = self.conv_h(x_h).sigmoid()
            a_w = self.conv_w(x_w).sigmoid()

            out = identity * a_w * a_h

        else:
            out = x
        return out

