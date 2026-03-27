import torch

def call_func(
    num_features: int,
    inputs: torch.Tensor,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = False,
    track_running_stats: bool = False
) -> torch.Tensor:
    instance_norm_layer = torch.nn.InstanceNorm1d(
        num_features=num_features,
        eps=eps,
        momentum=momentum,
        affine=affine,
        track_running_stats=track_running_stats
    )
    output = instance_norm_layer(inputs)
    return output

example_input = torch.randn(20, 100, 40)
example_output = call_func(num_features=100, inputs=example_input)