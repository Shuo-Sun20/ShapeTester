import torch

def call_func(inputs, N=None, increasing=False):
    x = inputs[0]
    return torch.vander(x, N=N, increasing=increasing)

x = torch.randn(4)
example_output = call_func(inputs=[x], N=3, increasing=True)