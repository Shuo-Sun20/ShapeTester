import torch

def call_func(inputs, out=None):
    return torch.special.ndtr(input=inputs, out=out)

# Generate a random tensor as input
torch.manual_seed(42)
example_input = torch.randn(5)

# Call the function and store the result
example_output = call_func(example_input)