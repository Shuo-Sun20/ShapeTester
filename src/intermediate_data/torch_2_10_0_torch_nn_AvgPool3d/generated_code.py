import torch

def call_func(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None, inputs=None):
    layer = torch.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)
    output = layer(inputs)
    return output

input_tensor = torch.randn(2, 3, 8, 16, 16)
example_output = call_func(kernel_size=(2, 2, 2), stride=(2, 2, 2), inputs=input_tensor)