import torch

def call_func(inputs, threshold, value):
    return torch.threshold_(inputs[0], threshold, value)

example_inputs = [torch.randn(3, 4)]
example_threshold = 0.5
example_value = 0.0
example_output = call_func(example_inputs, example_threshold, example_value)