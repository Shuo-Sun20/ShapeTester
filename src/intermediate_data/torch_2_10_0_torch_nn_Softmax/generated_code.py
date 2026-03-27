import torch

def call_func(dim, inputs):
    m = torch.nn.Softmax(dim=dim)
    return m(inputs)

dim = 1
inputs = torch.randn(2, 3)
example_output = call_func(dim, inputs)