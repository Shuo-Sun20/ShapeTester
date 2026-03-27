import torch

def call_func(inputs, kernel_size, stride=[], padding=0, dilation=1, ceil_mode=False):
    return torch.quantized_max_pool1d(inputs, kernel_size, stride, padding, dilation, ceil_mode)

qx = torch.quantize_per_tensor(torch.rand(2, 2), 1.5, 3, torch.quint8)
example_output = call_func(qx, [2])