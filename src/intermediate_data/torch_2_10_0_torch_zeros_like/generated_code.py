import torch

def call_func(inputs, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    input_tensor = inputs[0]
    return torch.zeros_like(input_tensor, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)

input_tensor = torch.randn(3, 4, 5)
example_output = call_func([input_tensor])