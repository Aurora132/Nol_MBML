"""
original code from facebook research:
https://github.com/facebookresearch/ConvNeXt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.unet_parts import Up, Up_Final, OutConv
from networks.modules import CoordAtt


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.ReLU()
        # self.act = HELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + x
        return x


class Incorporation(nn.Module):
    """incorporate two modalities"""

    def __init__(self, input_channel, output_channel, Ablation_mix=False):
        super(Incorporation, self).__init__()
        self.flag = Ablation_mix
        if not self.flag:
            self.layer_conv = nn.Conv3d(input_channel, output_channel, (2, 3, 3), stride=1, padding=(0, 1, 1),
                                        bias=True)
            self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

            self.fc_init_100 = nn.Sequential(
                nn.Conv2d(2 * input_channel, input_channel // 16, 1, stride=1, bias=False),
                nn.BatchNorm2d(input_channel // 16),
                nn.ReLU()
            )
            self.fc_init_140 = nn.Sequential(
                nn.Conv2d(2 * input_channel, input_channel // 16, 1, stride=1, bias=False),
                nn.BatchNorm2d(input_channel // 16),
                nn.ReLU()
            )
            self.fc_100 = nn.Conv2d(input_channel // 16, input_channel, 1, stride=1, bias=False)
            self.fc_140 = nn.Conv2d(input_channel // 16, input_channel, 1, stride=1, bias=False)
            self.fc_cor_100 = nn.Sequential(
                nn.Conv2d(3 * input_channel, 3 * input_channel // 16, 3, stride=1, bias=False, padding=1),
                nn.BatchNorm2d(3 * input_channel // 16),
                nn.ReLU(),
                nn.Conv2d(3 * input_channel // 16, input_channel, 1, stride=1, bias=False),
                nn.ReLU()
            )
            self.fc_cor_140 = nn.Sequential(
                nn.Conv2d(3 * input_channel, 3 * input_channel // 16, 3, stride=1, bias=False, padding=1),
                nn.BatchNorm2d(3 * input_channel // 16),
                nn.ReLU(),
                nn.Conv2d(3 * input_channel // 16, input_channel, 1, stride=1, bias=False),
                nn.ReLU()
            )
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x_100, x_140):

        if not self.flag:
            n_batch, n_channel, n_width, n_height = x_100.shape
            x_100_max = self.max_pool(x_100)
            x_100_avg = self.avg_pool(x_100)
            x_140_max = self.max_pool(x_140)
            x_140_avg = self.avg_pool(x_140)
            x_weight_100 = torch.concat((x_100_max, x_100_avg), dim=1)
            x_weight_140 = torch.concat((x_140_max, x_140_avg), dim=1)
            x_weight_100 = self.fc_init_100(x_weight_100)
            x_weight_140 = self.fc_init_140(x_weight_140)

            x_weight_100 = self.fc_100(x_weight_100)
            x_weight_140 = self.fc_140(x_weight_140)
            x_weight = torch.concat((x_weight_100, x_weight_140), dim=1)
            x_weight = x_weight.view(n_batch, 2, n_channel, 1, 1)
            x_weight = self.softmax(x_weight)
            x_weight = x_weight.view(n_batch, 2 * n_channel, 1, 1)

            x_mix = torch.concat((x_100, x_140), dim=1)
            x_mix_conv = torch.stack((x_100, x_140), dim=2)
            x_mix_conv = self.layer_conv(x_mix_conv)
            x_mix_conv = torch.squeeze(x_mix_conv, dim=2)
            x_mix_conv = nn.functional.leaky_relu(x_mix_conv, negative_slope=0.1)
            x_cor_100 = torch.concat((x_100, x_mix_conv), dim=1)
            x_cor_140 = torch.concat((x_140, x_mix_conv), dim=1)
            x_cor_100 = self.fc_cor_100(x_cor_100)
            x_cor_140 = self.fc_cor_140(x_cor_140)
            x_mix_conv = torch.concat((x_cor_100, x_cor_140), dim=1)
            x_mix_conv = x_weight * x_mix_conv
            xx = torch.mul(x_mix, x_mix_conv)
        else:
            xx = torch.concat((x_100, x_140), dim=1)
        return xx


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans: int = 1, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1., Ablation_mix=False, Ablation_att=False):
        super().__init__()
        self.num_classes = num_classes
        self.downsample_layers_100 = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        self.downsample_layers_140 = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_100 = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                                 nn.BatchNorm2d(dims[0]))
        stem_140 = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                                 nn.BatchNorm2d(dims[0]))
        self.downsample_layers_100.append(stem_100)
        self.downsample_layers_140.append(stem_140)

        for i in range(3):
            downsample_layer_100 = nn.Sequential(nn.BatchNorm2d(dims[i]),
                                                 nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            downsample_layer_140 = nn.Sequential(nn.BatchNorm2d(dims[i]),
                                                 nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers_100.append(downsample_layer_100)
            self.downsample_layers_140.append(downsample_layer_140)

        self.stages_100 = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        self.stages_140 = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks

        # attention
        self.CoordAttention_100 = nn.ModuleList()
        self.CoordAttention_140 = nn.ModuleList()
        for i in range(4):
            coordAttention_100 = CoordAtt(dims[i], Ablation_att=Ablation_att)
            self.CoordAttention_100.append(coordAttention_100)
            coordAttention_140 = CoordAtt(dims[i], Ablation_att=Ablation_att)
            self.CoordAttention_140.append(coordAttention_140)

        for i in range(4):
            stage_100 = nn.Sequential(
                *[Block(dim=dims[i], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            stage_140 = nn.Sequential(
                *[Block(dim=dims[i], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages_100.append(stage_100)
            self.stages_140.append(stage_140)

        self.norm = nn.BatchNorm2d(2 * dims[-1])  # final norm layer
        self.apply(self._init_weights)
        self.Incorporation = nn.ModuleList()
        for i in range(4):
            if i != 3:
                Incorporation_layer = Incorporation(dims[i], dims[i + 1], Ablation_mix=Ablation_mix)
            else:
                Incorporation_layer = Incorporation(dims[i], dims[i] * 2, Ablation_mix=Ablation_mix)
            self.Incorporation.append(Incorporation_layer)
        self.up1 = Up(dims[3] * 2, dims[3], dims[3])
        self.up2 = Up(dims[3] * 2, dims[2], dims[3])
        self.up3 = Up(dims[3], dims[1], dims[2])
        self.up4 = Up_Final(dims[2], dims[0], dims[1])
        self.outc = OutConv(dims[0], self.num_classes)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            # nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        xx_skip = []
        x_100 = x[:, 0, :, :].unsqueeze(dim=1)
        x_140 = x[:, 1, :, :].unsqueeze(dim=1)
        for i in range(4):
            x_100 = self.downsample_layers_100[i](x_100)
            x_100 = self.stages_100[i](x_100)
            x_100 = self.CoordAttention_100[i](x_100)
            x_140 = self.downsample_layers_140[i](x_140)
            x_140 = self.stages_140[i](x_140)
            x_140 = self.CoordAttention_140[i](x_140)
            xx = self.Incorporation[i](x_100, x_140)
            xx_skip.append(xx)
        x = self.norm(xx_skip[-1])

        return x, xx_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, xx_skip = self.forward_features(x)
        x = self.up1(x, xx_skip[2])
        x = self.up2(x, xx_skip[1])
        x = self.up3(x, xx_skip[0])
        x = self.up4(x)
        logits = self.outc(x)
        return logits


def convnext_tiny(num_classes: int, Ablation_mix=False, Ablation_att=False):
    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[48, 96, 192, 384],
                     num_classes=num_classes,
                     Ablation_mix=Ablation_mix,
                     Ablation_att=Ablation_att)
    return model


def convnext_small(num_classes: int, Ablation_mix=False, Ablation_att=False):
    # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes,
                     Ablation_mix=Ablation_mix,
                     Ablation_att=Ablation_att)
    return model


def convnext_base(num_classes: int, Ablation_mix=False, Ablation_att=False):
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes,
                     Ablation_mix=Ablation_mix,
                     Ablation_att=Ablation_att)
    return model


def convnext_large(num_classes: int, Ablation_mix=False, Ablation_att=False):
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes,
                     Ablation_mix=Ablation_mix,
                     Ablation_att=Ablation_att)
    return model


def convnext_xlarge(num_classes: int, Ablation_mix=False, Ablation_att=False):
    # https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes,
                     Ablation_mix=Ablation_mix,
                     Ablation_att=Ablation_att)
    return model


if __name__ == "__main__":
    import numpy as np

    x = torch.from_numpy(np.zeros((10, 2, 32, 32), dtype=np.float32))
    print(x.shape)
    net = convnext_tiny(num_classes=5, Ablation_mix=False, Ablation_att=False)
    y = net(x)
    print(y.shape)
