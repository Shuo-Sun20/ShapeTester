import torch
import torch.nn as nn

def call_func(inputs, p=2, eps=1e-6, keepdim=False):
    pdist = nn.PairwiseDistance(p=p, eps=eps, keepdim=keepdim)
    return pdist(inputs[0], inputs[1])

input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
example_output = call_func([input1, input2])