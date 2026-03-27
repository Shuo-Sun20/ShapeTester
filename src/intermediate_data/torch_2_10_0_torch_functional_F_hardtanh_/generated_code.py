import torch
import torch.nn.functional as F

def call_func(inputs, min_val=-1.0, max_val=1.0):
    return F.hardtanh_(inputs, min_val, max_val)

input_tensor = torch.randn(3, 4)
example_output = call_func(input_tensor)