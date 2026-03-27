import torch

def call_func(inputs, beta=1, alpha=1, out=None):
    input_tensor, batch1, batch2 = inputs
    return torch.addbmm(input_tensor, batch1, batch2, beta=beta, alpha=alpha, out=out)

M = torch.randn(3, 5)
batch1 = torch.randn(10, 3, 4)
batch2 = torch.randn(10, 4, 5)
example_output = call_func([M, batch1, batch2])