import torch
import torch.nn as nn

def call_func(padding, inputs):
    pad_layer = nn.ReflectionPad1d(padding)
    return pad_layer(inputs)

# Create a random tensor similar to the example shape (1, 2, 4)
torch.manual_seed(42)  # For reproducibility
example_input = torch.randn(1, 2, 4)
example_output = call_func(2, example_input)