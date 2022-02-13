# coding:utf-8
"""
    Created by cheng star at 2022/2/13 13:36
    @email : xxcheng0708@163.com
"""
import torch
import torchvision
from torch import nn
from torchlibrosa.augmentation import SpecAugmentation
from torchvision.models.resnet import BasicBlock, ResNet, Bottleneck


class AudioEmbeddingModel(nn.Module):
    def __init__(self, input_dimension=64, out_dimension=128, in_channels=1, model_name="resnet18", pretrained=False):
        super(AudioEmbeddingModel, self).__init__()
        self.input_dimension = input_dimension
        self.out_dimension = out_dimension
        self.in_channels = in_channels
        self.model_name = model_name
        self.pretrained = pretrained

        if self.model_name == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=self.pretrained)
            if self.in_channels != 3:
                self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                             bias=False)
                nn.init.kaiming_normal_(self.model.conv1.weight, mode="fan_in", nonlinearity="relu")
            self.model.fc = nn.Linear(in_features=512, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")

        if self.model_name == "resnet34":
            self.model = torchvision.models.resnet34(pretrained=self.pretrained)
            if self.in_channels != 3:
                self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                             bias=False)
                nn.init.kaiming_normal_(self.model.conv1.weight, mode="fan_in", nonlinearity="relu")
            self.model.fc = nn.Linear(in_features=512, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")

        if self.model_name == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=self.pretrained)
            if self.in_channels != 3:
                self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                             bias=False)
                nn.init.kaiming_normal_(self.model.conv1.weight, mode="fan_in", nonlinearity="relu")
            self.model.fc = nn.Linear(in_features=2048, out_features=self.out_dimension)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")

        self.spec_aug = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

    def forward(self, x):
        if self.training:
            x = x.transpose(2, 3)
            x = self.spec_aug(x)
            x = x.transpose(2, 3)

        out = self.model(x)
        return out


if __name__ == '__main__':
    from torchsummary import summary

    net = AudioEmbeddingModel(input_dimension=64, out_dimension=128, in_channels=1, model_name="resnet18",
                              pretrained=False)
    print(net)

    x = torch.randn(4, 1, 64, 128)
    out = net(x)
    print(out.shape)

    summary(net, input_size=(4, 1, 64, 128), batch_size=0)
