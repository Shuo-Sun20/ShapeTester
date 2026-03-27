import torch

def call_func(p=0.5, inplace=False, inputs=None):
    dropout = torch.nn.Dropout1d(p=p, inplace=inplace)
    return dropout(inputs)

input_tensor = torch.randn(20, 16, 32)
example_output = call_func(p=0.2, inputs=input_tensor)