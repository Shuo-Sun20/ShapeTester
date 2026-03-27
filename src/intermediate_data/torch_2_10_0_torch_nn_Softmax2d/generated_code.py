import torch

def call_func(inputs):
    m = torch.nn.Softmax2d()
    return m(inputs)

example_input = torch.randn(2, 3, 12, 13)
example_output = call_func(example_input)