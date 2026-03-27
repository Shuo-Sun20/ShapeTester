import torch

def call_func(inputs):
    return torch.argwhere(inputs)

torch.manual_seed(42)
example_input = torch.randint(0, 2, (3, 4), dtype=torch.float32)
example_output = call_func(example_input)