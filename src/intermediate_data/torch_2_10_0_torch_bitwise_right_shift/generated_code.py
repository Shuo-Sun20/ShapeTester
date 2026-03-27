import torch
import random

def call_func(inputs, out=None):
    input_tensor = inputs[0]
    other_tensor = inputs[1]
    return torch.bitwise_right_shift(input=input_tensor, other=other_tensor, out=out)

random.seed(42)
torch.manual_seed(42)
input_tensor = torch.randint(-128, 127, (3,), dtype=torch.int8)
other_tensor = torch.randint(0, 7, (3,), dtype=torch.int8)
example_output = call_func([input_tensor, other_tensor])