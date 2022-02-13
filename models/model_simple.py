# coding:utf-8
"""
    Created by cheng star at 2022/1/15 22:00
    @email : xxcheng0708@163.com
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchlibrosa.augmentation import SpecAugmentation
from models.non_local import NONLocalBlock1D
from torchlibrosa.augmentation import SpecAugmentation


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsampling=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsampling = downsampling
        self.stride = (1, 1)
        if self.downsampling:
            self.stride = (1, 2)

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 3), stride=self.stride,
                               padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1), stride=self.stride)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.bn3(self.conv3(x))

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = self.relu2(out)
        return out


class AudioEmbeddingModel(nn.Module):
    def __init__(self, input_dimension=64, out_dimension=128, model_name=None):
        super(AudioEmbeddingModel, self).__init__()
        self.input_dimension = input_dimension
        self.out_dimension = out_dimension

        self.spec_aug = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

        self.conv0 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        if self.input_dimension == 128:
            self.conv5 = nn.Conv2d(256, 256, kernel_size=(16, 1), stride=(16, 1))
        elif self.input_dimension == 64:
            self.conv5 = nn.Conv2d(256, 256, kernel_size=(8, 1), stride=(8, 1))
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)

        self.block1 = ResidualBlock(256, 512, downsampling=True)
        self.non_local1 = NONLocalBlock1D(in_channels=512, inter_channels=512, sub_sample=True, bn_layer=True)

        self.block2 = ResidualBlock(512, 1024, downsampling=True)

        self.max_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc1 = nn.Linear(1024, 1024)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(1024, self.out_dimension)

    def forward(self, x):
        batch_size = x.size(0)

        if self.training:
            x = x.transpose(2, 3)
            x = self.spec_aug(x)
            x = x.transpose(2, 3)

        out = self.relu0(self.bn0(self.conv0(x)))
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(x)))
        out = self.relu3(self.bn3(self.conv3(x)))
        out = self.relu4(self.bn4(self.conv4(x)))
        out = self.relu5(self.bn5(self.conv5(x)))

        out = self.block1(out)

        out = out.squeeze(dim=2)
        out = self.non_local1(out)
        out = out.unsqueeze(dim=2)

        out = self.block2(out)

        out = self.max_pool(out) + self.avg_pool(out)
        out = out.view(batch_size, -1)

        out = self.dropout1(self.relu6(self.fc1(out)))
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    from torchsummary import summary

    net = AudioEmbeddingModel(input_dimension=64, out_dimension=128)
    x = torch.randn(4, 1, 64, 1001)
    out = net(x)
    print(out.shape)

    summary(net, input_size=(4, 1, 64, 1001), batch_dim=0)
