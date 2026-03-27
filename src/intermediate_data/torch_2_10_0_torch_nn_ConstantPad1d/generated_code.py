import torch

def call_func(padding, value, inputs):
    """
    Calls torch.nn.ConstantPad1d with given parameters.
    
    Args:
        padding (int or tuple): Padding size(s) for left and right boundaries.
        value (float): Constant value to pad with.
        inputs (torch.Tensor): Input tensor to pad.
    
    Returns:
        torch.Tensor: Padded output tensor.
    """
    pad_layer = torch.nn.ConstantPad1d(padding, value)
    output = pad_layer(inputs)
    return output

# Generate random input tensor similar to documentation examples
torch.manual_seed(42)  # For reproducibility
example_input = torch.randn(1, 2, 4)  # Shape: (N, C, W_in) = (1, 2, 4)

# Call function with example parameters
example_output = call_func(padding=2, value=3.5, inputs=example_input)