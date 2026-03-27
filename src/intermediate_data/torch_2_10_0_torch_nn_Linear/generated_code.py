import torch

def call_func(in_features: int, out_features: int, bias: bool, inputs: torch.Tensor) -> torch.Tensor:
    linear_layer = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
    return linear_layer(inputs)

example_output = call_func(in_features=20, out_features=30, bias=True, inputs=torch.randn(128, 20))