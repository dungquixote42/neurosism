import torch
from torch import Tensor
from torch.nn.common_types import _size_3_t
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _triple
from torch.types import _int, _size, SymInt
from typing import Optional, Sequence, Union


ARGN_BATCH_SIZE_MISSING = 4
ARGN_BATCH_SIZE_PRESENT = 5

INDEX_DEPTH = 0
INDEX_HEIGHT = 1
INDEX_WIDTH = 2


def cross_correlation_3d(weight: Tensor, input: Tensor) -> Tensor:
    (wx, wy, wz) = weight.shape
    (ix, iy, iz) = input.shape
    rx = 1 - wx + ix
    ry = 1 - wy + iy
    rz = 1 - wz + iz
    result = torch.zeros(rx, ry, rz)

    resultIndices = [(x, y, z) for x in range(rx) for y in range(ry) for z in range(rz)]
    weightIndices = [(x, y, z) for x in range(wx) for y in range(wy) for z in range(wz)]

    for rxx, ryy, rzz in resultIndices:
        sum = 0
        for wxx, wyy, wzz in weightIndices:
            sum += weight[wxx][wyy][wzz] * input[rxx + wxx][ryy + wyy][rzz + wzz]
        result[rxx][ryy][rzz] = sum
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


class MyConv3d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        # device=None,
        # dtype=None,
    ) -> None:
        super().__init__()
        # factory_kwargs = {"device": device, "dtype": dtype}
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = padding if isinstance(padding, str) else _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        # super().__init__(
        #     in_channels,
        #     out_channels,
        #     kernel_size_,
        #     stride_,
        #     padding_,
        #     dilation_,
        #     False,
        #     _triple(0),
        #     groups,
        #     bias,
        #     padding_mode,
        #     **factory_kwargs
        # )
        self.weight = torch.nn.Parameter(
            Tensor(
                out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]
            )
        )
        self.bias = torch.nn.Parameter(Tensor(out_channels)) if bias else None

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            # TODO
            assert False
            return self._conv3d(
                self._pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return self._conv3d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def _conv3d(
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

        inputDepth = input.shape[-3]
        inputHeight = input.shape[-2]
        inputWidth = input.shape[-1]

        padDepth = padding[INDEX_DEPTH]
        padHeight = padding[INDEX_HEIGHT]
        padWidth = padding[INDEX_WIDTH]

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
            input = F.pad(input, (padWidth, padWidth, padHeight, padHeight, padDepth, padDepth))

        output = None
        if shapeLength == ARGN_BATCH_SIZE_MISSING:
            template = torch.zeros(self.out_channels, outputDepth, outputHeight, outputWidth)
            output = self._process_without_batch_size(template, input, weight, bias)
            # for j in range(0, self.out_channels):
            #     output[j] = bias[j]
            #     for k in range(0, self.in_channels):
            #         output[j] += cross_correlation_3d(weight[j][k], input[k])
        elif shapeLength == ARGN_BATCH_SIZE_PRESENT:
            batchSize = input.shape[0]
            template = torch.zeros(
                batchSize,
                self.out_channels,
                outputDepth,
                outputHeight,
                outputWidth,
            )
            output = self._process_with_batch_size(template, input, weight, bias, batchSize)
            # indices = [(i, j) for i in range(batchSize) for j in range(self.out_channels)]
            # for i, j in indices:
            #     output[i][j] = bias[j]
            #     for k in range(0, self.in_channels):
            #         output[i][j] += cross_correlation_3d(weight[j][k], input[i][k])
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
                template[i][j] += cross_correlation_3d(weight[j][k], input[i][k])
        return template

    def _process_without_batch_size(self, template: Tensor, input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        for j in range(0, self.out_channels):
            template[j] = bias[j]
            print(weight.shape)
            print(bias.shape)
            for k in range(0, self.in_channels):
                template[j] += cross_correlation_3d(weight[j][k], input[k])
        return template

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
