import torch
import numpy as np

def call_func(inputs, out=None):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    return torch.isposinf(input=input_tensor, out=out)

np.random.seed(42)
torch.manual_seed(42)
example_input = torch.tensor([
    float('-inf'),
    float('inf'),
    np.random.randn(),
    float('inf'),
    -np.random.randn()
])
example_output = call_func(example_input)