import torch

def call_func(inputs, N=None):
    x = inputs[0]
    return torch.linalg.vander(x, N=N)

# Construct a valid input
input_tensor = torch.randint(1, 6, (4,))
example_output = call_func([input_tensor], N=3)