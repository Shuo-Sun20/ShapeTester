import torch

def call_func(dim, inputs):
    log_softmax_layer = torch.nn.LogSoftmax(dim=dim)
    return log_softmax_layer(inputs)

example_input = torch.randn(2, 3)
example_output = call_func(dim=1, inputs=example_input)