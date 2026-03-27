import torch

def call_func(inputs: list[torch.Tensor], dim: int | tuple[int, ...] | None = None) -> torch.Tensor:
    input_tensor = inputs[0]
    return torch.squeeze(input_tensor, dim)

example_output = call_func(inputs=[torch.randn(2, 1, 3, 1, 4)], dim=(1, 3))