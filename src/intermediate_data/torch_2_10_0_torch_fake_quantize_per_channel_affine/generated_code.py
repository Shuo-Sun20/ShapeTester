import torch
import torch.nn as nn

def call_func(inputs, scale, zero_point, axis, quant_min, quant_max):
    """
    Calls torch.fake_quantize_per_channel_affine with the provided parameters.
    
    Args:
        inputs (list): List containing input tensor [input_tensor]
        scale (Tensor): Quantization scale per channel
        zero_point (Tensor): Quantization zero_point per channel
        axis (int): Channel axis
        quant_min (int): Lower bound of quantized domain
        quant_max (int): Upper bound of quantized domain
    
    Returns:
        Tensor: Fake quantized tensor
    """
    # Extract input tensor from the inputs list
    input_tensor = inputs[0]
    
    # Call the API directly
    return torch.fake_quantize_per_channel_affine(
        input_tensor, scale, zero_point, axis, quant_min, quant_max
    )

# Construct valid inputs and call the function
# Generate random input tensor
x = torch.randn(2, 3, 4)  # Batch x Channels x Features
scales = torch.randn(3).abs() + 0.01  # Per channel scales (must be positive)
zero_points = torch.zeros(3, dtype=torch.int32)  # Per channel zero points
axis = 1  # Channel axis
quant_min = 0
quant_max = 255

# Call function and save output
example_output = call_func([x], scales, zero_points, axis, quant_min, quant_max)