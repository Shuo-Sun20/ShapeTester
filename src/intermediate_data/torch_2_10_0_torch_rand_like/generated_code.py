import torch

def call_func(inputs, generator=None, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    return torch.rand_like(inputs, generator=generator, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)

example_input = torch.rand(2, 3)
example_output = call_func(example_input, requires_grad=True)