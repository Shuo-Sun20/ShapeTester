import torch

def call_func(inputs, out=None):
    input_tensor = inputs[0]
    return torch.special.expit(input_tensor, out=out)

torch.manual_seed(42)
example_input = [torch.randn(4)]
example_output = call_func(example_input)