import torch

def call_func(inputs, dim, index, out=None):
    return torch.index_select(input=inputs, dim=dim, index=index, out=out)

input_tensor = torch.randn(3, 4)
index_tensor = torch.tensor([0, 2])
example_output = call_func(inputs=input_tensor, dim=0, index=index_tensor)