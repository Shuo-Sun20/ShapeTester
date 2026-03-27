import torch

def call_func(inputs, dim, index):
    input_tensor = inputs[0]
    src_tensor = inputs[1]
    return torch.scatter(input=input_tensor, dim=dim, index=index, src=src_tensor)

input_tensor = torch.randn(3, 4)
src_tensor = torch.randn(3, 4)
index_tensor = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2], [1, 2, 0, 1]])
dim = 0
example_output = call_func([input_tensor, src_tensor], dim, index_tensor)