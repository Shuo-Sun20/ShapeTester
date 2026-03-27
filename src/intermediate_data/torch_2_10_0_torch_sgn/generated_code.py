import torch

def call_func(inputs, out=None):
    return torch.sgn(input=inputs, out=out)

# Generate random complex tensor
real_part = torch.randn(4)
imag_part = torch.randn(4)
example_input = torch.complex(real_part, imag_part)

example_output = call_func(example_input)