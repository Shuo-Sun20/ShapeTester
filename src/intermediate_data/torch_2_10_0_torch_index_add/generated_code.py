import torch

def call_func(inputs, dim, index, alpha=1, out=None):
    input_tensor, source_tensor = inputs
    return torch.index_add(input_tensor, dim, index, source_tensor, alpha=alpha, out=out)

# Generate random tensors
torch.manual_seed(42)
input_tensor = torch.randn(5, 3)
source_tensor = torch.randn(3, 3)
index = torch.tensor([0, 2, 4])
dim = 0
alpha = 1.0

# Call function
example_output = call_func([input_tensor, source_tensor], dim, index, alpha)