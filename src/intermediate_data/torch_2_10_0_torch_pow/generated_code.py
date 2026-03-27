import torch

def call_func(inputs, exponent=None, out=None):
    if isinstance(inputs, list) and len(inputs) == 2:
        # Case where both base and exponent are tensors
        return torch.pow(inputs[0], inputs[1], out=out)
    elif isinstance(inputs, torch.Tensor):
        # Case where base is tensor, exponent is scalar
        return torch.pow(inputs, exponent, out=out)
    elif isinstance(inputs, (int, float)) and isinstance(exponent, torch.Tensor):
        # Case where base is scalar, exponent is tensor
        return torch.pow(inputs, exponent, out=out)
    else:
        raise ValueError("Invalid input combination")

# Generate random tensors for testing
torch.manual_seed(42)
base_tensor = torch.randn(3, 2)
exp_tensor = torch.randint(1, 4, (3, 2)).float()
scalar_base = 2.5

# Test case 1: Tensor base with scalar exponent
example_output = call_func(base_tensor, exponent=2)

# Test case 2: Tensor base with tensor exponent  
example_output = call_func([base_tensor, exp_tensor])

# Test case 3: Scalar base with tensor exponent
example_output = call_func(scalar_base, exponent=exp_tensor)