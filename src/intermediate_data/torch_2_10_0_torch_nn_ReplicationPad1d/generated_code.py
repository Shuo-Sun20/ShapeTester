import torch
import torch.nn as nn

def call_func(padding, inputs):
    pad_layer = nn.ReplicationPad1d(padding)
    return pad_layer(inputs)

# Generate a random tensor with shape (batch_size, channels, width) matching documentation examples
batch_size, channels, width = 2, 3, 10
example_input = torch.randn(batch_size, channels, width)
example_output = call_func((2, 3), example_input)