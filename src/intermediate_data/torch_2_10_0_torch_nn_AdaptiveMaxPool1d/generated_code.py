import torch

def call_func(output_size, return_indices, inputs):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    pool_layer = torch.nn.AdaptiveMaxPool1d(output_size=output_size, return_indices=return_indices)
    result = pool_layer(input_tensor)
    if return_indices:
        output_tensor = result[0]
    else:
        output_tensor = result
    return output_tensor

# Example usage with random tensor
input_tensor = torch.randn(2, 3, 10)
example_output = call_func(output_size=5, return_indices=False, inputs=input_tensor)