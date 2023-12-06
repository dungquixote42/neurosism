import torch
from torch import Tensor
import torch.nn as nn

# from torch.nn.common_types import _size_any_t
import torchvision.transforms as tf


class UNet(nn.Module):
    def __init__(self, inputChannelCount: int = 1, outputChannelCount: int = 64, kernelSize=(1, 1, 89)):
        super(UNet, self).__init__()
        self.firstOp = nn.Sequential(
            nn.Conv3d(inputChannelCount, outputChannelCount, kernelSize),
            nn.ReLU(inplace=True),
            nn.Conv3d(outputChannelCount, outputChannelCount, kernelSize),
            nn.ReLU(inplace=True),
        )

        inputChannelCount = outputChannelCount
        outputChannelCount *= 2
        self.downOp1 = UNetDown(inputChannelCount, outputChannelCount, convKernel=3)

        inputChannelCount = outputChannelCount
        outputChannelCount *= 2
        self.downOp2 = UNetDown(inputChannelCount, outputChannelCount)

        inputChannelCount = outputChannelCount
        outputChannelCount *= 2
        self.downOp3 = UNetDown(inputChannelCount, outputChannelCount)

        inputChannelCount = outputChannelCount
        outputChannelCount *= 2
        self.downOp4 = UNetDown(inputChannelCount, outputChannelCount)

        inputChannelCount = outputChannelCount
        outputChannelCount //= 2
        self.upOp1 = UNetUp(inputChannelCount, outputChannelCount)

        inputChannelCount = outputChannelCount
        outputChannelCount //= 2
        self.upOp2 = UNetUp(inputChannelCount, outputChannelCount)

        inputChannelCount = outputChannelCount
        outputChannelCount //= 2
        self.upOp3 = UNetUp(inputChannelCount, outputChannelCount)

        self.lastOp = nn.Sequential(nn.Conv3d(128, 1, (1, 42, 14)), nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        x = self.firstOp(x)
        x1 = self.downOp1(x)
        x2 = self.downOp2(x1)
        x3 = self.downOp3(x2)
        y = self.downOp4(x3)
        y = self.upOp1(y, x3)
        y = self.upOp2(y, x2)
        y = self.upOp3(y, x1)
        return self.lastOp(y)


class UNetDown(nn.Module):
    def __init__(
        self,
        inputChannelCount: int,
        outputChannelCount: int,
        convKernel=(1, 3, 3),
        poolKernel=(1, 2, 2),
        stride=(1, 2, 2),
    ):
        super().__init__()
        self.op = nn.Sequential(
            nn.MaxPool3d(poolKernel, stride),
            nn.Conv3d(inputChannelCount, outputChannelCount, convKernel),
            nn.ReLU(inplace=True),
            nn.Conv3d(outputChannelCount, outputChannelCount, convKernel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class UNetUp(nn.Module):
    def __init__(
        self, inputChannelCount: int, outputChannelCount: int, convKernel1=2, convKernel2=(1, 3, 3), scaleFactor=2
    ):
        super().__init__()
        self.op1 = nn.Sequential(
            nn.Upsample(scale_factor=scaleFactor),
            nn.Conv3d(inputChannelCount, outputChannelCount, convKernel1),
        )
        self.op2 = nn.Sequential(
            nn.Conv3d(inputChannelCount, outputChannelCount, convKernel2),
            nn.ReLU(inplace=True),
            nn.Conv3d(outputChannelCount, outputChannelCount, convKernel2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.op1(x1)
        dimension = x1.shape[-1]
        x2 = tf.CenterCrop((dimension, dimension))(x2)
        x1 = torch.cat((x1, x2), 1)
        x1 = self.op2(x1)
        return x1
