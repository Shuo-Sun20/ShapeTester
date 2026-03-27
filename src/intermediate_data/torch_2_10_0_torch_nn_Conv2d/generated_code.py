import torch
import torch.nn as nn

def call_func(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', inputs=None):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    output = conv_layer(inputs)
    return output

input_tensor = torch.randn(20, 16, 50, 100)
example_output = call_func(in_channels=16, out_channels=33, kernel_size=3, stride=2, inputs=input_tensor)