import torch

def call_func(inputs, out_dtype=None, beta=1, alpha=1, out=None):
    input_tensor, mat1, mat2 = inputs
    if out_dtype is not None:
        return torch.addmm(input_tensor, mat1, mat2, out_dtype=out_dtype, beta=beta, alpha=alpha, out=out)
    else:
        return torch.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha, out=out)

example_output = call_func(
    inputs=[torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3)],
    out_dtype=None,
    beta=1,
    alpha=1,
    out=None
)