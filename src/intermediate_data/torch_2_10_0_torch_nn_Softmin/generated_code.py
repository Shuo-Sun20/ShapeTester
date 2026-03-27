import torch

def call_func(inputs,dim=None):
    softmin = torch.nn.Softmin(dim=dim)
    output = softmin(inputs)
    return output

input_tensor = torch.randn(2, 3)
example_output = call_func(inputs=input_tensor, dim=1)
print(example_output)