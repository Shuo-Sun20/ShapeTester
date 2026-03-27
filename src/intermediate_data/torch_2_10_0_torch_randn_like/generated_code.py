import torch

def call_func(inputs, generator=None, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    return torch.randn_like(input=inputs, generator=generator, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)

example_input = torch.randn(3, 4)
example_output = call_func(example_input, dtype=torch.float64, requires_grad=True)