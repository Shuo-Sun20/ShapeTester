import torch

def call_func(inputs, assume_unique=False, invert=False):
    elements, test_elements = inputs
    return torch.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)

elements = torch.randint(0, 10, (3, 4))
test_elements = torch.tensor([2, 5, 8])
example_output = call_func([elements, test_elements])