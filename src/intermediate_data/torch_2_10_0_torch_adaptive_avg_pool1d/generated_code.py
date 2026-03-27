import torch
import torch.nn.functional as F

def call_func(inputs, output_size):
    return F.adaptive_avg_pool1d(inputs, output_size)

example_input = torch.randn(2, 3, 10)
example_output = call_func(example_input, 5)