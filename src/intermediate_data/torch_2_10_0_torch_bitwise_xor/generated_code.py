import torch

def call_func(inputs, out=None):
    input_tensor, other_tensor = inputs
    return torch.bitwise_xor(input_tensor, other_tensor, out=out)

tensor1 = torch.tensor([1, 2, 3], dtype=torch.int32)
tensor2 = torch.tensor([1, 3, 7], dtype=torch.int32)
example_output = call_func([tensor1, tensor2])