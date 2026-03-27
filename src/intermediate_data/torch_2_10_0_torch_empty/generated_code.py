import torch

def call_func(inputs, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format):
    return torch.empty(inputs, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, pin_memory=pin_memory, memory_format=memory_format)

example_output = call_func((2, 3), dtype=torch.float32)