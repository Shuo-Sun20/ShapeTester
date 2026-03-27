import torch

def call_func(inputs, dim=1, eps=1e-8):
    x1, x2 = inputs
    return torch.cosine_similarity(x1, x2, dim=dim, eps=eps)

# Generate random input tensors
torch.manual_seed(42)
input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)

# Call the function and save output
example_output = call_func([input1, input2])