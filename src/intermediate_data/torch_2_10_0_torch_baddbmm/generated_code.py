import torch

def call_func(inputs, out_dtype=None, beta=1, alpha=1, out=None):
    input_tensor, batch1, batch2 = inputs
    if out_dtype is not None and input_tensor.is_cuda:
        return torch.baddbmm(input_tensor, batch1, batch2, out_dtype=out_dtype, beta=beta, alpha=alpha, out=out)
    else:
        return torch.baddbmm(input_tensor, batch1, batch2, beta=beta, alpha=alpha, out=out)

batch_size, n, m, p = 10, 3, 4, 5
input_tensor = torch.randn(batch_size, n, p)
batch1 = torch.randn(batch_size, n, m)
batch2 = torch.randn(batch_size, m, p)

example_output = call_func([input_tensor, batch1, batch2])