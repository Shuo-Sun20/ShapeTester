import torch

def call_func(inputs, fill_value, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    input_tensor = inputs[0]
    return torch.full_like(input_tensor, fill_value, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)

example_input = torch.randn(2, 3)
example_output = call_func([example_input], 5.0)