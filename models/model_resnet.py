# coding:utf-8
"""
    Created by cheng star at 2022/1/15 21:08
    @email : xxcheng0708@163.com
"""
import torch
from torch import nn
from torchlibrosa.augmentation import SpecAugmentation
from torchvision.models.resnet import BasicBlock, ResNet, Bottleneck


class AudioEmbeddingModel(ResNet):
    def __init__(self, input_dimension=64, out_dimension=128, in_channels=1, model_name="resnet18"):
        self.input_dimension = input_dimension
        self.out_dimension = out_dimension
        self.in_channels = in_channels
        self.model_name = model_name

        if self.model_name == "resnet18":
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif self.model_name == "resnet34":
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif self.model_name == "resnet50":
            block = Bottleneck
            layers = [3, 4, 6, 3]
        super(AudioEmbeddingModel, self).__init__(block, layers)

        self.spec_aug = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

        if self.model_name == "resnet18":
            if self.in_channels != 3:
                self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                       bias=False)
            self.fc = nn.Linear(in_features=512, out_features=self.out_dimension)
        elif self.model_name == "resnet34":
            if self.in_channels != 3:
                self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                       bias=False)
            self.fc = nn.Linear(in_features=512, out_features=self.out_dimension)
        elif self.model_name == "resnet50":
            if self.in_channels != 3:
                self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                       bias=False)
            self.fc = nn.Linear(in_features=2048, out_features=self.out_dimension)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        if self.training:
            x = x.transpose(2, 3)
            x = self.spec_aug(x)
            x = x.transpose(2, 3)

        out = self._forward_impl(x)
        return out


if __name__ == '__main__':
    from torchsummary import summary

    net = AudioEmbeddingModel(input_dimension=64, out_dimension=128, in_channels=1, model_name="resnet18")
    print(net)

    x = torch.randn(4, 1, 64, 128)
    out = net(x)
    print(out.shape)

    summary(net, input_size=(4, 1, 64, 1001), batch_size=0)
