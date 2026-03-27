import torch

def call_func(inputs, low=0, high=None, generator=None, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    return torch.randint_like(inputs, low=low, high=high, generator=generator, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)

example_input = torch.randn(3, 4, 5)
example_output = call_func(inputs=example_input, high=100)