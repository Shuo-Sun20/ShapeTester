import torch

def call_func(inputs, out=None):
    input_tensor = inputs[0] if isinstance(inputs, list) else inputs
    return torch.deg2rad(input=input_tensor, out=out)

# Generate random tensor for degrees
random_tensor = torch.randn(3, 4) * 180.0  # Random angles in degrees
example_output = call_func(random_tensor)