import torch

def call_func(inputs):
    return torch.linalg.matrix_exp(inputs[0])

example_tensor = torch.randn(3, 3)
example_output = call_func([example_tensor])