import torch

def call_func(inputs, out=None):
    input_tensor = inputs[0]
    n_tensor = inputs[1]
    return torch.special.hermite_polynomial_h(input_tensor, n_tensor, out=out)

input_tensor = torch.randn(4, 4)
n_tensor = torch.tensor([3])
example_output = call_func([input_tensor, n_tensor])