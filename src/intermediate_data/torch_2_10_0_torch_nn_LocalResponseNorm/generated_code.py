import torch
import torch.nn as nn

def call_func(size, alpha, beta, k, inputs):
    lrn = nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
    return lrn(inputs)

example_input = torch.randn(32, 5, 24, 24)
example_output = call_func(size=2, alpha=0.0001, beta=0.75, k=1, inputs=example_input)