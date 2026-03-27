import torch

def call_func(padding, inputs):
    # Unwrap inputs if it's a list (for API consistency)
    if isinstance(inputs, list):
        # ZeroPad2d only accepts one input tensor
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    
    # Create ZeroPad2d instance and apply it
    pad_layer = torch.nn.ZeroPad2d(padding)
    return pad_layer(input_tensor)

# Generate a random tensor matching the input shape from documentation
# Using shape (1, 1, 3, 3) as in the example
input_tensor = torch.randn(1, 1, 3, 3)

# Call the function with padding=2 and the input tensor
example_output = call_func(padding=2, inputs=input_tensor)