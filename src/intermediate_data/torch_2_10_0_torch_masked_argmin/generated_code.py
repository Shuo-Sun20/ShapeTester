import torch

def call_func(inputs, dim, keepdim=False, dtype=None, mask=None):
    return torch.masked.argmin(inputs, dim, keepdim=keepdim, dtype=dtype, mask=mask)

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate random input tensor
input_tensor = torch.randn(3, 4, 5)

# Generate random boolean mask tensor (broadcastable to input shape)
mask_tensor = torch.randint(0, 2, (3, 4, 5), dtype=torch.bool)

# Call the function with example parameters
example_output = call_func(
    inputs=input_tensor,
    dim=1,
    keepdim=True,
    dtype=torch.int64,
    mask=mask_tensor
)