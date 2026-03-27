import torch

def call_func(inputs, out=None):
    # Split the inputs list into input and values tensors
    input_tensor, values_tensor = inputs
    # Call the torch.heaviside function directly with the split tensors
    return torch.heaviside(input_tensor, values_tensor, out=out)

# Construct random input tensors
input_tensor = torch.randn(5)  # Random tensor for input
values_tensor = torch.randn(5)  # Random tensor for values
# Call the function with the inputs as a list
example_output = call_func([input_tensor, values_tensor])