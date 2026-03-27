import torch

def call_func(inputs, out=None):
    input_tensor = inputs[0]
    n_tensor = inputs[1]
    return torch.special.laguerre_polynomial_l(input_tensor, n_tensor, out=out)

torch.manual_seed(42)
input_tensor = torch.randn(3, 4)
n_tensor = torch.tensor([2])
inputs = [input_tensor, n_tensor]
example_output = call_func(inputs)