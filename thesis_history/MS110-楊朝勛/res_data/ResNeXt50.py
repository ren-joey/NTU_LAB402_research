import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

class ResNeXt_Block(nn.Module):
    def __init__(self, first_channels, in_channels, out_channels, depths):
        super(ResNeXt_Block, self).__init__()
        self.out_channels = out_channels
        self.depths = depths
        self.conv1_v1 = nn.Conv3d(first_channels, in_channels, kernel_size=1, bias=False)
        self.conv1_v2 = nn.Conv3d(out_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=2, bias=False)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

        self.downsample = nn.Sequential(
            nn.Conv3d(first_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        
    def forward(self, x):
        # get information from previous cardinality block
        identity = self.downsample(x)

        for depth in range(self.depths):
            if depth == 0:
                x = self.conv1_v1(x)
            else:
                x = self.conv1_v2(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = x + identity
            x = self.relu(x)
        return x

class ResNeXt50(nn.Module):
    def __init__(self, in_chans=1, depths=[3, 4, 6, 3], num_classes=2, len_of_clinical_features=150):
        super(ResNeXt50, self).__init__()
        # Before ResBlock
        self.conv1 = nn.Conv3d(in_chans, 2, kernel_size = 7, stride = 2, padding=3, bias=False)
        self.bn3d = nn.BatchNorm3d(2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size = 3, stride = 2)

        self.layer1 = ResNeXt_Block(first_channels=2, in_channels=2, out_channels=8, depths=depths[0])
        self.layer2 = ResNeXt_Block(first_channels=8, in_channels=4, out_channels=16, depths=depths[1])
        self.layer3 = ResNeXt_Block(first_channels=16, in_channels=8, out_channels=32, depths=depths[2])
        self.layer4 = ResNeXt_Block(first_channels=32, in_channels=16, out_channels=64, depths=depths[3])

        self.avgpool = nn.AdaptiveAvgPool3d((1))

        # voi size
        self.v_fc1 = nn.Linear(3, 16)
        self.v_fc2 = nn.Linear(16, 64)
        self.v_fc3 = nn.Linear(64, 64)

        # final
        len_of_features = 64 + len_of_clinical_features
        self.fc1 = nn.Linear(len_of_features, len_of_features // 2) # 214 = dim[-1] + dim(clinical_info) which is 150
        self.fc2 = nn.Linear(len_of_features // 2, 2)
        self.bn1d = nn.BatchNorm1d(len_of_features // 2)
        self.softmax = nn.Softmax()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d)):
            trunc_normal_(m.weight, std=.02) # 正態分佈
        if isinstance(m, (nn.Linear)):
            trunc_normal_(m.weight, std=.02) # 正態分佈
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x, clinical_info, voi_size):
        x = self.conv1(x)
        x = self.bn3d(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        voi_feature = self.v_fc1(voi_size)
        voi_feature = self.v_fc2(voi_feature)
        voi_feature = self.v_fc3(voi_feature)

        b, c, _, _, _ = x.size()
        x = self.avgpool(x).view(b, c)

        b, n = x.size()
        _sum = (x * voi_feature).sum(dim=(-1))
        x = x - (_sum / n).view(b, 1).expand_as(x)

        x = torch.cat((x, clinical_info), 1)
        x = F.leaky_relu(self.bn1d(self.fc1(x)))
        x = self.fc2(x)
        x = self.softmax(x)

        return x