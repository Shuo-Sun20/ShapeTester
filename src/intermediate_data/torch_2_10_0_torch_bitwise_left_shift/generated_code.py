import torch

def call_func(inputs, out=None):
    return torch.bitwise_left_shift(inputs[0], inputs[1], out=out)

example_input1 = torch.randint(-128, 127, (3,), dtype=torch.int8)
example_input2 = torch.tensor([1, 0, 3], dtype=torch.int8)
example_output = call_func([example_input1, example_input2])