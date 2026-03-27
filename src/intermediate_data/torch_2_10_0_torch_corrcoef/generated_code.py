import torch

def call_func(inputs: torch.Tensor) -> torch.Tensor:
    return torch.corrcoef(inputs)

x = torch.randn(3, 5)
example_output = call_func(x)