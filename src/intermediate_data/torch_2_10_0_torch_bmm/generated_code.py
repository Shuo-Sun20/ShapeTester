import torch

def call_func(inputs, out_dtype=None, out=None):
    input_tensor, mat2_tensor = inputs[0], inputs[1]
    if out_dtype is not None:
        return torch.bmm(input_tensor, mat2_tensor, out_dtype=out_dtype, out=out)
    else:
        if out is not None:
            return torch.bmm(input_tensor, mat2_tensor, out=out)
        else:
            return torch.bmm(input_tensor, mat2_tensor)

batch_size, n, m, p = 10, 3, 4, 5
input_tensor = torch.randn(batch_size, n, m)
mat2_tensor = torch.randn(batch_size, m, p)
inputs_list = [input_tensor, mat2_tensor]

example_output = call_func(inputs_list)