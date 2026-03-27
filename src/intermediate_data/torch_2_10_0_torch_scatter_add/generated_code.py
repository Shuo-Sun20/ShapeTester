import torch

def call_func(inputs, dim, index):
    input_tensor, src = inputs
    return torch.scatter_add(input_tensor, dim, index, src)

# Example usage
input_tensor = torch.randn(3, 5)
src = torch.randn(3, 5)
index = torch.randint(0, 5, (3, 5))
dim = 1

example_output = call_func([input_tensor, src], dim, index)