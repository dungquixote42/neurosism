import torch
from torch import Tensor
from torch.nn.common_types import _size_3_t
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _triple
from torch.types import _int, _size, SymInt
from typing import Optional, Sequence, Union


def _cross_correlation_3d(weight: Tensor, input: Tensor) -> Tensor:
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


def _get_output_dimension(
    inputDimension: int, padding: int, dilation: int, kernelSize: int, stride: int
) -> int:
    outputDimension = int(
        (inputDimension + 2 * padding - dilation * (kernelSize - 1) - 1) / stride + 1
    )
    return outputDimension


def _pad(
    input: Tensor,
    pad: Sequence[int],
    mode: str = ...,
    value: Optional[float] = None,
) -> Tensor:
    pass


class MyConv3d(_ConvNd):
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
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return self._conv3d(
                _pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return self._conv3d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

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
        assert shapeLength == 4 or shapeLength == 5

        inputDepth = input.shape[-3]
        inputHeight = input.shape[-2]
        inputWidth = input.shape[-1]

        outputDepth = _get_output_dimension(
            inputDepth, padding[0], dilation[0], self.kernel_size[0], stride[0]
        )
        outputHeight = _get_output_dimension(
            inputHeight, padding[1], dilation[1], self.kernel_size[1], stride[1]
        )
        outputWidth = _get_output_dimension(
            inputWidth, padding[2], dilation[2], self.kernel_size[2], stride[2]
        )

        if padding != 0:
            pad0, pad1, pad2 = padding[0], padding[1], padding[2]
            input = F.pad(input, (pad2, pad2, pad1, pad1, pad0, pad0))

        output = None
        if shapeLength == 4:
            output = torch.zeros(
                self.out_channels, outputDepth, outputHeight, outputWidth
            )
            for j in range(0, self.out_channels):
                output[j] = bias[j]
                for k in range(0, self.in_channels):
                    output[j] += _cross_correlation_3d(weight[j][k], input[k])
        elif shapeLength == 5:
            batchSize = input.shape[0]
            output = torch.zeros(
                batchSize,
                self.out_channels,
                outputDepth,
                outputHeight,
                outputWidth,
            )
            indices = [
                (i, j) for i in range(batchSize) for j in range(self.out_channels)
            ]
            for i, j in indices:
                output[i][j] = bias[j]
                for k in range(0, self.in_channels):
                    output[i][j] += _cross_correlation_3d(weight[j][k], input[i][k])
        else:
            assert False

        return output

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
