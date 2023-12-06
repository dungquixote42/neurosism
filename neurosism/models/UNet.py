import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.common_types import _size_any_t


class UNet(nn.Module):
    def __init__(
        self, inputChannelCount: int, outputChannelCount: int, kernelSize: _size_any_t, verbose: bool = False
    ):
        super(UNet, self).__init__()
        self.inputChannelCount = inputChannelCount
        self.outputChannelCount = outputChannelCount
        self.kernelSize = kernelSize
        self.verbose = verbose

        # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(inputChannelCount, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )

        # Final output layer
        self.out = nn.Conv2d(64, outputChannelCount, kernel_size=1)

    def forward(self, x):
        x1: Tensor = nn.MaxPool3d(8)(x)
        if self.verbose:
            print(x1.shape)
        x1a: Tensor = nn.Conv3d(1, 64, 3)(x1)
        if self.verbose:
            print(x1a.shape)
        x1b: Tensor = nn.ReLU(inplace=True)(x1a)
        if self.verbose:
            print(x1b.shape)

        # # Encoder
        # x1 = self.encoder(x)
        # # Decoder
        # x = self.decoder(x1)
        # # Concatenate skip connection from encoder
        # x = torch.cat([x, x1], dim=1)
        # # Output layer
        # x = self.out(x)
        # return x


class Conv3d_ReLU_Conv3d_ReLU_MaxPool3d(nn.Module):
    def __init__(self, inputChannelCount: int, outputChannelCount: int, kernelSize: tuple = (3, 3, 2), stride=2):
        super().__init__()
        self.operation = nn.Sequential(
            nn.Conv3d(inputChannelCount, outputChannelCount, kernelSize[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(outputChannelCount, outputChannelCount, kernelSize[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernelSize[2], stride),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.operation(x)


class Upsample_Conv3d_Cat_Conv3d_ReLU_Conv3d_ReLU(nn.Module):
    def __init__(self, inputChannelCount, outputChannelCount, kernelSize=(2, 3, 3), scaleFactor=2):
        super().__init__()
        self.operation1 = nn.Sequential(
            nn.Upsample(scale_factor=scaleFactor),
            nn.Conv3d(inputChannelCount, outputChannelCount, kernelSize[0], padding=1),
        )
        self.operation2 = nn.Sequential(
            nn.Conv3d(inputChannelCount, outputChannelCount, kernelSize[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(outputChannelCount, outputChannelCount, kernelSize[2]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        return


# class UNetDown(nn.Module):
#     def __init__(
#         self,
#         inputChannelCount: int,
#         outputChannelCount: int,
#         poolKernelSize: _size_any_t,
#         convKernelSize: _size_any_t,
#         verbose: bool = False,
#     ):
#         super().__init__()
#         self.inputChannelCount = inputChannelCount
#         self.outputChannelCount = outputChannelCount
#         self.poolKernelSize = poolKernelSize
#         self.convKernelSize = convKernelSize
#         self.verbose = verbose

#     def forward(self, x):
#         x1: Tensor = nn.MaxPool3d(self.poolKernelSize)(x)
#         if self.verbose:
#             print(x1.shape)
#         x2: Tensor = nn.Conv3d(self.inputChannelCount, self.outputChannelCount, self.convKernelSize)(x1)
#         if self.verbose:
#             print(x2.shape)
#         x3: Tensor = nn.ReLU(inplace=True)(x2)
#         if self.verbose:
#             print(x3.shape)
#         return x3


# class UNetUp(nn.Module):
#     def __init__(
#         self,
#         inputChannelCount: int,
#         middleChannelCount: int,
#         outputChannelCount: int,
#         convKernelSize: _size_any_t,
#         convTransposeKernelSize: _size_any_t,
#         verbose: bool = False,
#     ):
#         super().__init__()
#         self.inputChannelCount = inputChannelCount
#         self.middleChannelCount = middleChannelCount
#         self.outputChannelCount = outputChannelCount
#         self.convKernelSize = convKernelSize
#         self.convTransposeKernelSize = convTransposeKernelSize
#         self.verbose = verbose

#     def forward(self, x):
#         x1: Tensor = nn.Conv3d(self.inputChannelCount, self.middleChannelCount, self.convKernelSize)(x)
#         if self.verbose:
#             print(x1.shape)
#         x2: Tensor = nn.ReLU(inplace=True)(x1)
#         if self.verbose:
#             print(x2.shape)
#         x3: Tensor = nn.ConvTranspose3d(
#             self.middleChannelCount, self.outputChannelCount, self.convTransposeKernelSize
#         )(x2)
#         if self.verbose:
#             print(x3.shape)
#         return x3
