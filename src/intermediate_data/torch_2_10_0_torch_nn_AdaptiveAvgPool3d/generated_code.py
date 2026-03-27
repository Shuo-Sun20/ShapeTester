import torch

def call_func(output_size, inputs):
    pool_layer = torch.nn.AdaptiveAvgPool3d(output_size)
    
    if isinstance(inputs, list):
        output = pool_layer(*inputs)
    else:
        output = pool_layer(inputs)
    
    return output

input_tensor = torch.randn(2, 3, 8, 16, 32)
example_output = call_func((4, 8, 12), input_tensor)