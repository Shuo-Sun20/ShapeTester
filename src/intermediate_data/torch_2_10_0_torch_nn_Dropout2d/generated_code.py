import torch
import torch.nn as nn

def call_func(p, inplace, inputs):
    dropout_layer = nn.Dropout2d(p=p, inplace=inplace)
    output = dropout_layer(inputs)
    return output

example_input = torch.randn(20, 16, 32, 32)
example_output = call_func(p=0.2, inplace=False, inputs=example_input)