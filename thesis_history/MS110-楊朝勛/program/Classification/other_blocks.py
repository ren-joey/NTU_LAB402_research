import torch
from torch import nn
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class GCT(nn.Module):
    def __init__(self, channel, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, channel, 1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channel, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel, 1, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3, 4), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        else:
            print('Unknown mode!')
            sys.exit()

        # gate = 1. + torch.tanh(embedding * norm + self.beta)
        gate = 1. + torch.sigmoid(embedding * norm + self.beta)

        return x * gate

class BAM(nn.Module):
    def __init__(self, gate_channel, reduction=2, dilation_val=4, num_layers=1):
        super(BAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channel_att = nn.Sequential(
            nn.Linear(gate_channel, gate_channel // reduction),
            nn.ReLU(),
            nn.Linear(gate_channel // reduction, gate_channel)
        )

        # input (_, _, D, H, W) -> output (_, _, D, H, W) for any Conv3D
        self.spatial_att = nn.Sequential(
            nn.Conv3d(gate_channel, gate_channel // reduction, kernel_size=1),
            LayerNorm(gate_channel // reduction, data_format="channels_first"),
            nn.GELU(),
            nn.Conv3d(gate_channel // reduction, gate_channel // reduction, kernel_size=3, padding=dilation_val, dilation=dilation_val),
            LayerNorm(gate_channel // reduction, data_format="channels_first"),
            nn.GELU(),
            nn.Conv3d(gate_channel // reduction, gate_channel // reduction, kernel_size=3, padding=dilation_val, dilation=dilation_val),
            LayerNorm(gate_channel // reduction, data_format="channels_first"),
            nn.GELU(),
            nn.Conv3d(gate_channel//reduction, 1, kernel_size=1)
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # compute channel attention
        channel_part = self.avg_pool(x).view(b, c)
        channel_part = self.channel_att(channel_part).view(b, c, 1, 1, 1).expand_as(x)
        # compute spatial attention
        spatial_part = self.spatial_att(x).expand_as(x)
        # add together
        att = 1 + F.sigmoid(channel_part + spatial_part)
        return att * x

class CBAM(nn.Module):
    def __init__(self, gate_channel, reduction=2):
        super().__init__()
        # channel attention
        self.pools = [
            nn.AdaptiveAvgPool3d(1),
            nn.AdaptiveMaxPool3d(1)
        ]
        self.mlp = nn.Sequential(
            nn.Linear(gate_channel, gate_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channel // reduction, gate_channel)
        )
        self.sigmoid = nn.Sigmoid()

        # spatial attention
        kernel_size = 7
        self.conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1) // 2),
            LayerNorm(1, data_format="channels_first"),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = x
        b, c, _, _, _ = x.size()

        channel_part = None
        for pool in self.pools:
            y = pool(x).view(b, c)
            y = self.mlp(y)
            channel_part = channel_part + y if channel_part is not None else y 
        channel_part = self.sigmoid(channel_part).view(b, c, 1, 1, 1)
        x = x * channel_part.expand_as(x)

        spatial_part = torch.cat( (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1 )
        spatial_part = self.conv(spatial_part)
        x = x * spatial_part.expand_as(x)

        return x + res

class SE_GCT(nn.Module):
    def __init__(self, channel, reduction=2, epsilon=1e-5, mode='l2', after_relu=False):
        super().__init__()
        # Squeeze and excitation
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )
        # GCT
        self.alpha = nn.Parameter(torch.ones(1, channel, 1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channel, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel, 1, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        # compute SE attention
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        se_gate = self.fc(y).view(b, c, 1, 1, 1).expand_as(x)

        # compute GCT attention
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3, 4), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        else:
            print('Unknown mode!')
            sys.exit()
        gct_gate = embedding * norm + self.beta

        # add together
        att = 1 + F.sigmoid(se_gate + gct_gate)
        return att * x

class BAM_GCT(nn.Module):
    def __init__(self, channel, reduction=2, dilation_val=4, num_layers=1, epsilon=1e-5, mode='l2', after_relu=False):
        super().__init__()
        # Squeeze and excitation
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )
        # GCT
        self.alpha = nn.Parameter(torch.ones(1, channel, 1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channel, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel, 1, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

        # input (_, _, D, H, W) -> output (_, _, D, H, W) for any Conv3D
        self.spatial_att = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1),
            LayerNorm(channel // reduction, data_format="channels_first"),
            nn.GELU(),
            nn.Conv3d(channel // reduction, channel // reduction, kernel_size=3, padding=dilation_val, dilation=dilation_val),
            LayerNorm(channel // reduction, data_format="channels_first"),
            nn.GELU(),
            nn.Conv3d(channel // reduction, channel // reduction, kernel_size=3, padding=dilation_val, dilation=dilation_val),
            LayerNorm(channel // reduction, data_format="channels_first"),
            nn.GELU(),
            nn.Conv3d(channel//reduction, 1, kernel_size=1)
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # compute channel attention

        # compute SE attention
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1).expand_as(x)

        # compute GCT attention
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3, 4), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        else:
            print('Unknown mode!')
            sys.exit()
        gate = embedding * norm + self.beta

        # add together
        channel_part = y + gate

        # compute spatial attention
        spatial_part = self.spatial_att(x).expand_as(x)
        # add together
        att = 1 + F.sigmoid(channel_part + spatial_part)
        return att * x

class CBAM_GCT(nn.Module):
    def __init__(self, channel, reduction=2, epsilon=1e-5, mode='l2', after_relu=False):
        super().__init__()
        # channel attention
        self.pools = [
            nn.AdaptiveAvgPool3d(1),
            nn.AdaptiveMaxPool3d(1)
        ]
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel)
        )
        self.sigmoid = nn.Sigmoid()

        # GCT
        self.alpha = nn.Parameter(torch.ones(1, channel, 1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channel, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel, 1, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

        # spatial attention
        kernel_size = 7
        self.conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1) // 2),
            LayerNorm(1, data_format="channels_first"),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = x
        b, c, _, _, _ = x.size()

        channel_part = None
        for pool in self.pools:
            y = pool(x).view(b, c)
            y = self.mlp(y)
            channel_part = channel_part + y if channel_part is not None else y 
        channel_part = channel_part.view(b, c, 1, 1, 1)

        # compute GCT attention
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3, 4), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        else:
            print('Unknown mode!')
            sys.exit()
        gate = embedding * norm + self.beta

        # add together
        channel_part = channel_part + gate
        channel_part = F.sigmoid(channel_part)
        x = x * channel_part.expand_as(x)

        spatial_part = torch.cat( (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1 )
        spatial_part = self.conv(spatial_part)
        x = x * spatial_part.expand_as(x)

        return x + res

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x