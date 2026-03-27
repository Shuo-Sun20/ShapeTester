import torch

def call_func(pack_hook, unpack_hook, inputs):
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        output = inputs[0] * inputs[1]
    return output

def pack_hook(x):
    return x.detach()

def unpack_hook(x):
    return x

a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True) * 2
example_output = call_func(pack_hook, unpack_hook, [a, b])