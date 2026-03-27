import torch
import torch.nn.functional as F

def call_func(inputs, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    input_tensor = inputs[0]
    return F.avg_pool1d(input_tensor, kernel_size, stride, padding, ceil_mode, count_include_pad)

# Create random input tensor with shape (minibatch=2, in_channels=3, iW=10)
input_tensor = torch.randn(2, 3, 10)
example_output = call_func([input_tensor], kernel_size=3, stride=2)