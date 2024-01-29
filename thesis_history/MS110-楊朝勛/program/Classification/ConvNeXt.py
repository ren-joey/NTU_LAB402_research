import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from other_blocks import SE, GCT, BAM, CBAM, LayerNorm, SE_GCT, BAM_GCT, CBAM_GCT

class GRN(nn.Module):
    # gamma, beta: learnable affine transform parameters
    # X: input of shape (N, D, H, W, C)
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros((dim)))
        self.beta = nn.Parameter(torch.zeros((dim)))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, D, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, depth_number: int, pos_in_depth: int, drop_path=0.5, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.depth_number = depth_number
        self.pos_in_depth = pos_in_depth
        self.se_gct = SE_GCT(channel=dim, reduction=2)
        self.dim = dim

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (N, D, H, W, C) -> (N, C, D, H, W)

        if self.pos_in_depth == 3:
            x = self.se_gct(x)

        x = input + x
        return x

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
    def __init__(
            self, in_chans=1, num_classes=2, 
            depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
            layer_scale_init_value=1e-6, head_init_scale=1., len_of_clinical_features=150
        ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value, depth_number=i+1, pos_in_depth=j+1) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer

        # voi size
        self.v_fc1 = nn.Linear(3, 16)
        self.v_fc2 = nn.Linear(16, dims[-1])
        self.v_fc3 = nn.Linear(dims[-1], dims[-1])

        # final
        len_of_features = dims[-1] + len_of_clinical_features
        self.fc1 = nn.Linear(len_of_features, len_of_features // 2) # 214 = dim[-1] + dim(clinical_info) which is 150
        self.fc2 = nn.Linear(len_of_features // 2, 2)
        self.bn1 = nn.BatchNorm1d(len_of_features // 2)
        self.softmax = nn.Softmax()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02) # 正態分佈
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-3, -2, -1])) # global average pooling, (N, C, D, H, W) -> (N, C)

    def forward(self, x, clinical_info, voi_size):
        x_feature = self.forward_features(x)
        voi_feature = self.v_fc1(voi_size)
        voi_feature = self.v_fc2(voi_feature)
        voi_feature = self.v_fc3(voi_feature)

        b, n = x_feature.size()
        _sum = (x_feature * voi_feature).sum(dim=(-1))
        x = x_feature - (_sum / n).view(b, 1).expand_as(x_feature)

        x = torch.cat((x, clinical_info), 1)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        x = self.softmax(x)

        return x

def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    # original dims = [96, 192, 384, 768]
    # best dims = [8, 16, 32, 64]
    model = ConvNeXt(depths=[1, 1, 3, 1], dims=[8, 16, 32, 64], **kwargs)
    return model