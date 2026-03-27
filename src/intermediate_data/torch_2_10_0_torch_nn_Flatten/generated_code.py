import torch
import torch.nn as nn

def call_func(start_dim=1, end_dim=-1, inputs=None):
    flatten_layer = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
    output = flatten_layer(inputs)
    return output

example_input = torch.randn(32, 1, 5, 5)
example_output = call_func(start_dim=0, end_dim=2, inputs=example_input)