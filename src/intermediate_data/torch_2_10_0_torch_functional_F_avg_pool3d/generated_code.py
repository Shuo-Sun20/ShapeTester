import torch
import torch.nn.functional as F

def call_func(inputs, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    return F.avg_pool3d(input_tensor, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

torch.manual_seed(0)
example_input = torch.randn(2, 3, 8, 8, 8)
example_output = call_func(example_input, kernel_size=2, stride=2)