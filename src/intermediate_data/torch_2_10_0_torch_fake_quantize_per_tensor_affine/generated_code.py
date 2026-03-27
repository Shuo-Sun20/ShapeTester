import torch

def call_func(inputs, scale, zero_point, quant_min, quant_max):
    input_tensor = inputs
    return torch.fake_quantize_per_tensor_affine(input_tensor, scale, zero_point, quant_min, quant_max)

x = torch.randn(4)
example_output = call_func(x, 0.1, 0, 0, 255)