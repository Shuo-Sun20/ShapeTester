import torch

def call_func(kernel_size, stride, inputs, padding=0):
    unpool = torch.nn.MaxUnpool2d(kernel_size, stride, padding)
    return unpool(*inputs)

torch.manual_seed(0)
input_tensor = torch.randn(1, 1, 4, 4)
pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
pooled_output, indices = pool(input_tensor)
example_output = call_func(2, 2, [pooled_output, indices], 0)