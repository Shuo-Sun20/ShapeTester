import torch

def call_func(inputs: torch.Tensor) -> torch.Tensor:
    return torch.ravel(inputs)

example_output = call_func(torch.randn(3, 4))