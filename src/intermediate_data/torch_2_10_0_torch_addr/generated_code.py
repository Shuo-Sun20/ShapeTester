import torch

def call_func(inputs, beta=1, alpha=1, out=None):
    # Unpack inputs list into the three required tensors
    input_tensor, vec1, vec2 = inputs
    # Call torch.addr directly (it's a function, not a class)
    return torch.addr(input_tensor, vec1, vec2, beta=beta, alpha=alpha, out=out)

# Generate random tensors for valid input
vec1 = torch.randn(3)
vec2 = torch.randn(2)
input_matrix = torch.randn(3, 2)

# Call the function with inputs as a list
example_output = call_func([input_matrix, vec1, vec2])