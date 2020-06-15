import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Conv2d


def conv4d(data, filters, bias=None, permute_filters=True, use_half=False):
    b, c, h, w, d, t = data.size()

    # [8, 1, 25, 25, 25, 25] -> [25, 8, 1, 25, 25, 25]
    data = data.permute(
        2, 0, 1, 3, 4, 5
    ).contiguous()  # permute to avoid making contiguous inside loop

    # Same permutation is done with filters, unless already provided with permutation
    if permute_filters:
        filters = filters.permute(
            2, 0, 1, 3, 4, 5
        ).contiguous()  # permute to avoid making contiguous inside loop
    c_out = filters.size(1)
    if use_half:
        output = torch.HalfTensor(
            [h, b, c_out, w, d, t], dtype=data.dtype, requires_grad=data.requires_grad
        )
    else:
        output = torch.zeros(
            [h, b, c_out, w, d, t], dtype=data.dtype, requires_grad=data.requires_grad
        )

    kh, _, _, kw, kd, kt = filters.shape
    padding = kh // 2  # calc padding size (kernel_size - 1)/2
    padding_3d = (kw // 2, kd // 2, kt // 2)
    if use_half:
        Z = torch.zeros([padding, b, c, w, d, t], dtype=data.dtype).half()
    else:
        Z = torch.zeros([padding, b, c, w, d, t], dtype=data.dtype)

    if data.is_cuda:
        Z = Z.cuda(data.get_device())
        output = output.cuda(data.get_device())

    data_padded = torch.cat((Z, data, Z), 0)  # [29, 8, 16, 25, 25, 25]
    if bias is not None:
        bias = bias / (1 + padding * 2)
    # print('bias',bias)

    for i in range(output.size(0)):  # loop on first feature dimension
        # convolve with center channel of filter (at position=padding)
        output[i, :, :, :, :, :] = F.conv3d(
            data_padded[i + padding, :, :, :, :, :],
            filters[padding, :, :, :, :, :],
            bias=bias,
            stride=1,
            padding=padding_3d,
        )
        # convolve with upper/lower channels of filter (at postions [:padding] [padding+1:])
        for p in range(1, padding + 1):
            output[i, :, :, :, :, :] += F.conv3d(
                data_padded[i + padding - p, :, :, :, :, :],
                filters[padding - p, :, :, :, :, :],
                bias=bias,
                stride=1,
                padding=padding_3d,
            )
            output[i, :, :, :, :, :] += F.conv3d(
                data_padded[i + padding + p, :, :, :, :, :],
                filters[padding + p, :, :, :, :, :],
                bias=bias,
                stride=1,
                padding=padding_3d,
            )

    output = output.permute(1, 2, 0, 3, 4, 5).contiguous()  # [8, 16, 25, 25, 25, 25]
    return output


class Conv4d(_ConvNd):
    """Applies a 4D convolution over an input signal composed of several input
    planes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        pre_permuted_filters=True,
        bias=True,
        filters=None,
        bias_4d=None,
    ):
        # stride, dilation and groups !=1 functionality not tested
        stride = 1
        dilation = 1
        groups = 1
        # zero padding is added automatically in conv4d function to preserve tensor size
        padding = 0
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)
        super(Conv4d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _quadruple(0),
            groups,
            bias,
            "zero",
        )
        # weights will be sliced along one dimension during convolution loop
        # make the looping dimension to be the first one in the tensor,
        # so that we don't need to call contiguous() inside the loop
        self.pre_permuted_filters = pre_permuted_filters
        # self.groups=groups
        if filters is not None:
            self.weight.data = filters
        if bias_4d is not None and bias:
            self.bias.data = bias_4d

        if self.pre_permuted_filters:
            self.weight.data = self.weight.data.permute(2, 0, 1, 3, 4, 5).contiguous()
        self.use_half = False

    def forward(self, input):
        return conv4d(
            input,
            self.weight,
            bias=self.bias,
            permute_filters=not self.pre_permuted_filters,
            use_half=self.use_half,
        )  # filters pre-permuted in constructor
