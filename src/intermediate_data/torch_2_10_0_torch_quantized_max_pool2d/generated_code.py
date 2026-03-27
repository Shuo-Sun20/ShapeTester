import torch

def call_func(inputs, kernel_size, stride=[], padding=0, dilation=1, ceil_mode=False):
    input_tensor = inputs[0]
    return torch.quantized_max_pool2d(input_tensor, kernel_size, stride, padding, dilation, ceil_mode)

random_tensor = torch.rand(2, 2, 2, 2)
quantized_tensor = torch.quantize_per_tensor(random_tensor, 1.5, 3, torch.quint8)
example_output = call_func([quantized_tensor], [2, 2])