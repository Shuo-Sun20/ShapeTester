import torch

def call_func(inputs, weights=None, minlength=0):
    if weights is None:
        return torch.bincount(inputs, minlength=minlength)
    else:
        return torch.bincount(inputs, weights=weights, minlength=minlength)

input_tensor = torch.randint(0, 10, (20,), dtype=torch.int64)
example_output = call_func(input_tensor, minlength=10)