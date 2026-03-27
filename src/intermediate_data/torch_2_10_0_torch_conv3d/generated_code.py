import torch

def call_func(inputs, stride=1, padding=0, dilation=1, groups=1):
    if len(inputs) == 3:
        input_tensor, weight_tensor, bias_tensor = inputs
    else:
        input_tensor, weight_tensor = inputs
        bias_tensor = None
    return torch.conv3d(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups)

input_tensor = torch.randn(2, 3, 10, 20, 30)
weight_tensor = torch.randn(6, 3, 4, 5, 6)
example_output = call_func([input_tensor, weight_tensor])