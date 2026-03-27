import torch

def call_func(p=0.5, inplace=False, inputs=None):
    module = torch.nn.FeatureAlphaDropout(p=p, inplace=inplace)
    output = module(inputs)
    return output

torch.manual_seed(42)
inputs = torch.randn(20, 16, 4, 32, 32)
example_output = call_func(p=0.2, inplace=False, inputs=inputs)