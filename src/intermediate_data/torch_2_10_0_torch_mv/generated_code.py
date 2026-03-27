import torch

def call_func(inputs, out=None):
    # Unpack the two input tensors: matrix and vector
    input_matrix, vec = inputs
    # Directly call torch.mv function
    return torch.mv(input=input_matrix, vec=vec, out=out)

# Generate random tensors matching the documented example
mat = torch.randn(2, 3)
vec = torch.randn(3)

# Call the function and save output
example_output = call_func(inputs=[mat, vec])