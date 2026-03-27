import torch
import torch.nn as nn

def call_func(dim=1, eps=1e-8, inputs=None):
    if inputs is None:
        raise ValueError("Inputs cannot be None")
    
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
        raise ValueError("Inputs must be a list or tuple containing exactly two tensors")
    
    x1, x2 = inputs
    cos_sim = nn.CosineSimilarity(dim=dim, eps=eps)
    output = cos_sim(x1, x2)
    return output

torch.manual_seed(42)
input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
example_output = call_func(dim=1, eps=1e-8, inputs=[input1, input2])