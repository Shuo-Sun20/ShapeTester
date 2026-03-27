import torch

def call_func(inputs, dim=0, out=None):
    input_tensor, indices = inputs
    return torch.take_along_dim(input=input_tensor, indices=indices, dim=dim, out=out)

torch.manual_seed(42)
example_inputs = [torch.randn(3, 4), torch.randint(0, 4, (3, 4))]
example_output = call_func(example_inputs, dim=1)