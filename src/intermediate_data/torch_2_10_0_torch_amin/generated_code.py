import torch

def call_func(inputs, dim, keepdim=False, out=None):
    return torch.amin(input=inputs[0], dim=dim, keepdim=keepdim, out=out)

example_output = call_func(
    inputs=[torch.randn(4, 4)],  
    dim=1, 
    keepdim=False, 
    out=None
)