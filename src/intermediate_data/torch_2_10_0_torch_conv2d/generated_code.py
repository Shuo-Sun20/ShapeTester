import torch

def call_func(inputs, bias=None, stride=1, padding=0, dilation=1, groups=1):
    input_tensor, weight_tensor = inputs[0], inputs[1]
    output = torch.conv2d(
        input=input_tensor,
        weight=weight_tensor,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )
    return output

batch_size = 2
in_channels = 4
out_channels = 6
input_height = 5
input_width = 5
kernel_height = 3
kernel_width = 3

input_tensor = torch.randn(batch_size, in_channels, input_height, input_width)
weight_tensor = torch.randn(out_channels, in_channels, kernel_height, kernel_width)
inputs = [input_tensor, weight_tensor]

example_output = call_func(inputs, bias=None, stride=1, padding=1, dilation=1, groups=1)