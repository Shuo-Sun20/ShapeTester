import torch

def call_func(inputs, scale, zero_point):
    # Unpack the input tensors from the list
    input_tensor, weight_tensor, bias_tensor = inputs
    
    # Directly call the API with unpacked parameters
    return torch.nn.quantized.functional.linear(
        input=input_tensor,
        weight=weight_tensor,
        bias=bias_tensor,
        scale=scale,
        zero_point=zero_point
    )

# Generate random tensors for the inputs
batch_size = 4
in_features = 8
out_features = 6
extra_dims = 5

# Create random float tensors and quantize them appropriately
input_float = torch.randn(batch_size, extra_dims, in_features)
weight_float = torch.randn(out_features, in_features)
bias_float = torch.randn(out_features)

# Quantize input (torch.quint8)
input_scale = 0.1
input_zero_point = 128
input_quantized = torch.quantize_per_tensor(
    input_float, 
    scale=input_scale, 
    zero_point=input_zero_point, 
    dtype=torch.quint8
)

# Quantize weight (torch.qint8)
weight_scale = 0.05
weight_zero_point = 0
weight_quantized = torch.quantize_per_tensor(
    weight_float,
    scale=weight_scale,
    zero_point=weight_zero_point,
    dtype=torch.qint8
)

# Output quantization parameters
scale_output = 0.2
zero_point_output = 100

# Prepare inputs list
inputs_list = [input_quantized, weight_quantized, bias_float]

# Call the function and save output
example_output = call_func(
    inputs=inputs_list,
    scale=scale_output,
    zero_point=zero_point_output
)