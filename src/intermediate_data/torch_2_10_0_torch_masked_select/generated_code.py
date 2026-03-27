import torch

def call_func(inputs, out=None):
    input_tensor = inputs[0]
    mask_tensor = inputs[1]
    return torch.masked_select(input_tensor, mask_tensor, out=out)

# Construct valid input
input_tensor = torch.randn(3, 4)
mask_tensor = input_tensor.ge(0.5)
example_output = call_func([input_tensor, mask_tensor])