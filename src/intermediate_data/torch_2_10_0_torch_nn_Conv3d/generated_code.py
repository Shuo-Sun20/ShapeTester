import torch
import torch.nn as nn

def call_func(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', inputs=None):
    conv3d_layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    output = conv3d_layer(inputs)
    return output

# Generate random input tensor with shape (N, C_in, D, H, W)
inputs = torch.randn(2, 3, 16, 32, 32)

# Call the function with example parameters
example_output = call_func(in_channels=3, out_channels=6, kernel_size=(3, 5, 5), stride=(2, 1, 1), padding=(1, 2, 2), dilation=1, groups=1, bias=True, padding_mode='zeros', inputs=inputs)