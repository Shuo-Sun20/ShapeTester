import torch

def call_func(inputs):
    return torch.isreal(inputs)

# Generate random complex tensor with some zero imaginary parts
real_part = torch.randn(3, 4)
imag_part = torch.randn(3, 4)
# Set some imaginary parts to zero
imag_part[0, 0] = 0
imag_part[1, 2] = 0
imag_part[2, 3] = 0
example_input = torch.complex(real_part, imag_part)

example_output = call_func(example_input)