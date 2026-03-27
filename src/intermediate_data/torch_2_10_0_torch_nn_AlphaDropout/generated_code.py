import torch
import torch.nn as nn

def call_func(inputs, p=0.5, inplace=False):
    module = nn.AlphaDropout(p=p, inplace=inplace)
    output = module(inputs)
    return output

example_input = torch.randn(20, 16)
example_output = call_func(example_input, p=0.2)