import torch

def call_func(inputs, out=None):
    return torch.special.xlog1py(inputs[0], inputs[1], out=out)

torch.manual_seed(42)
example_inputs = [torch.randn(3, 4), torch.randn(3, 4)]
example_output = call_func(example_inputs)