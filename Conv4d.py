import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.common_types import _size_4_t
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
from torch.types import _int, _size, SymInt
from typing import Optional, Sequence, Union


ARGN_BATCH_SIZE_MISSING = 5
ARGN_BATCH_SIZE_PRESENT = 6

INDEX_TIME = 0
INDEX_DEPTH = 1
INDEX_HEIGHT = 2
INDEX_WIDTH = 3


def cross_correlation_4d(weight: Tensor, input: Tensor) -> Tensor:
    (wx, wy, wz, wt) = weight.shape
    (ix, iy, iz, it) = input.shape
    rx = 1 - wx + ix
    ry = 1 - wy + iy
    rz = 1 - wz + iz
    rt = 1 - wt + it
    result = torch.zeros(rx, ry, rz, rt)

    resultIndices = [(x, y, z, t) for x in range(rx) for y in range(ry) for z in range(rz) for t in range(rt)]
    weightIndices = [(x, y, z, t) for x in range(wx) for y in range(wy) for z in range(wz) for t in range(wt)]

    for rxx, ryy, rzz, rtt in resultIndices:
        sum = 0
        for wxx, wyy, wzz, wtt in weightIndices:
            sum += weight[wxx][wyy][wzz][wtt] * input[rxx + wxx][ryy + wyy][rzz + wzz][rtt + wtt]
        result[rxx][ryy][rzz][rtt] = sum
    return result


def get_output_dimension(inputDimension: int, padding: int, dilation: int, kernelSize: int, stride: int) -> int:
    outputDimension = int((inputDimension + 2 * padding - dilation * (kernelSize - 1) - 1) / stride + 1)
    return outputDimension


def _pad(
    input: Tensor,
    pad: Sequence[int],
    mode: str = ...,
    value: Optional[float] = None,
) -> Tensor:
    # TODO
    pass


class Conv4d(nn.Module):
    def __init__(
        self,
        inputChannels,
        outputChannels,
        kernelSize,
        stride: _size_4_t = 1,
        padding: Union[str, _size_4_t] = 0,
        dilation: _size_4_t = 1,
        groups: int = 1,
        bias: bool = True,
        paddingMode: str = "zeros",
    ):
        super().__init__()

        self.in_channels = inputChannels
        self.out_channels = outputChannels
        self.kernel_size = _quadruple(kernelSize)
        self.stride = _quadruple(stride)
        self.padding = padding if isinstance(padding, str) else _quadruple(padding)
        self.dilation = _quadruple(dilation)
        self.groups = groups
        self.padding_mode = paddingMode

        self.weight = nn.Parameter(torch.ones(outputChannels, inputChannels, kernelSize))
        if bias:
            self.bias = nn.Parameter(torch.zeros(outputChannels))
        else:
            self.bias = None
        # def __init__(
        #     self,
        #     in_channels: int,
        #     out_channels: int,
        #     kernel_size: _size_4_t,
        #     stride: _size_4_t = 1,
        #     padding: Union[str, _size_4_t] = 0,
        #     dilation: _size_4_t = 1,
        #     groups: int = 1,
        #     bias: bool = True,
        #     padding_mode: str = "zeros",  # TODO: refine this type
        #     device=None,
        #     dtype=None,
        # ) -> None:
        # factory_kwargs = {"device": device, "dtype": dtype}
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        # kernel_size_ = _quadruple(kernel_size)
        # stride_ = _quadruple(stride)
        # padding_ = padding if isinstance(padding, str) else _quadruple(padding)
        # dilation_ = _quadruple(dilation)
        # super(Conv4d, self).__init__()
        # super().__init__(
        #     in_channels,
        #     out_channels,
        #     kernel_size_,
        #     stride_,
        #     padding_,
        #     dilation_,
        #     False,
        #     _quadruple(0),
        #     groups,
        #     bias,
        #     padding_mode,
        #     **factory_kwargs
        # )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            # TODO
            assert False
            return self._conv4d(
                self._pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                _quadruple(0),
                self.dilation,
                self.groups,
            )
        return self._conv4d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def _conv4d(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: Union[_int, _size] = 1,
        padding: Union[Union[_int, SymInt], Sequence[Union[_int, SymInt]]] = 0,
        dilation: Union[_int, _size] = 1,
        groups: _int = 1,
    ) -> Tensor:
        shapeLength = len(input.shape)
        assert shapeLength == ARGN_BATCH_SIZE_MISSING or shapeLength == ARGN_BATCH_SIZE_PRESENT
        assert bias != None  # TODO
        assert groups == 1  # TODO

        inputTime = input.shape[-4]
        inputDepth = input.shape[-3]
        inputHeight = input.shape[-2]
        inputWidth = input.shape[-1]

        padTime = padding[INDEX_TIME]
        padDepth = padding[INDEX_DEPTH]
        padHeight = padding[INDEX_HEIGHT]
        padWidth = padding[INDEX_WIDTH]

        outputTime = get_output_dimension(
            inputTime,
            padTime,
            dilation[INDEX_TIME],
            self.kernel_size[INDEX_TIME],
            stride[INDEX_TIME],
        )
        outputDepth = get_output_dimension(
            inputDepth,
            padDepth,
            dilation[INDEX_DEPTH],
            self.kernel_size[INDEX_DEPTH],
            stride[INDEX_DEPTH],
        )
        outputHeight = get_output_dimension(
            inputHeight,
            padHeight,
            dilation[INDEX_HEIGHT],
            self.kernel_size[INDEX_HEIGHT],
            stride[INDEX_HEIGHT],
        )
        outputWidth = get_output_dimension(
            inputWidth,
            padWidth,
            dilation[INDEX_WIDTH],
            self.kernel_size[INDEX_WIDTH],
            stride[INDEX_WIDTH],
        )

        if padding != 0:
            input = F.pad(input, (padWidth, padWidth, padHeight, padHeight, padDepth, padDepth, padTime, padTime))

        output = None
        if shapeLength == ARGN_BATCH_SIZE_MISSING:
            template = torch.zeros(self.out_channels, outputTime, outputDepth, outputHeight, outputWidth)
            output = self._process_without_batch_size(template, input, weight, bias)
        elif shapeLength == ARGN_BATCH_SIZE_PRESENT:
            batchSize = input.shape[0]
            template = torch.zeros(
                batchSize,
                self.out_channels,
                outputTime,
                outputDepth,
                outputHeight,
                outputWidth,
            )
            output = self._process_with_batch_size(template, input, weight, bias, batchSize)
        else:
            assert False

        return output

    def _process_with_batch_size(
        self, template: Tensor, input: Tensor, weight: Tensor, bias: Tensor, batchSize: int
    ) -> Tensor:
        indices = [(i, j) for i in range(batchSize) for j in range(self.out_channels)]
        for i, j in indices:
            template[i][j] = bias[j]
            for k in range(0, self.in_channels):
                template[i][j] += cross_correlation_4d(weight[j][k], input[i][k])
        return template

    def _process_without_batch_size(self, template: Tensor, input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        for j in range(0, self.out_channels):
            print(template[j].shape)
            print(bias[j].shape)
            template[j] = bias[j]
            for k in range(0, self.in_channels):
                template[j] += cross_correlation_4d(weight[j][k], input[k])
        return template

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
