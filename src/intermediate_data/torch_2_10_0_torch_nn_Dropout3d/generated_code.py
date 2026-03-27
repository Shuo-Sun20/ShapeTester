import torch
import torch.nn as nn

def call_func(inputs, p=0.5, inplace=False):
    dropout = nn.Dropout3d(p=p, inplace=inplace)
    return dropout(inputs)

example_input = torch.randn(20, 16, 4, 32, 32)
example_output = call_func(example_input, p=0.2)