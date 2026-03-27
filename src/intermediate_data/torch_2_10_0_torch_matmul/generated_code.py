import torch

def call_func(inputs, out=None):
    input_tensor, other_tensor = inputs
    return torch.matmul(input_tensor, other_tensor, out=out)

# tensor1 = torch.randn(10, 3, 4)
# tensor2 = torch.randn(4, 5)
# example_output = call_func([tensor1, tensor2])

x = torch.randn(4, 3)
y = torch.randn(3, 5)
valid_test_case = {"inputs": [x, y], "out": y}
try:
    print(call_func(**valid_test_case).shape)
except UserWarning:
    print("has warning")
