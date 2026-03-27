import torch

def call_func(threshold, value, inplace, inputs):
    threshold_layer = torch.nn.Threshold(threshold, value, inplace)
    output = threshold_layer(inputs)
    return output

example_input = torch.randn(3, 4)
example_output = call_func(0.5, 0.1, False, example_input)