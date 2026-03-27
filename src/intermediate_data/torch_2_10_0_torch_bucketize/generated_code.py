import torch

def call_func(inputs, out_int32=False, right=False, out=None):
    input_tensor, boundaries = inputs
    return torch.bucketize(input_tensor, boundaries, out_int32=out_int32, right=right, out=out)

# Generate random tensors for demonstration
torch.manual_seed(42)
boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
input_tensor = torch.randn(3, 2) * 10

# Call the function
example_output = call_func([input_tensor, boundaries])