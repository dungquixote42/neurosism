from torch import Tensor
import torch.nn as nn
from torch.nn.common_types import _size_any_t


class UNetIn(nn.Module):
    def __init__(self, inputChannelCount: int, outputChannelCount: int, kernelSize: _size_any_t):
        super().__init__()
        self.inputChannelCount = inputChannelCount
        self.outputChannelCount = outputChannelCount
        self.kernelSize = kernelSize

    def forward(self, x: Tensor) -> Tensor:
        pass