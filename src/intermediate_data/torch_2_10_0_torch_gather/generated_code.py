import torch

def call_func(inputs, dim, sparse_grad=False, out=None):
    input_tensor, index_tensor = inputs[0], inputs[1]
    output = torch.gather(input_tensor, dim, index_tensor, sparse_grad=sparse_grad, out=out)
    return output

input_tensor = torch.randn(3, 4)
index_tensor = torch.randint(0, 4, (3, 4))
example_output = call_func([input_tensor, index_tensor], 1)