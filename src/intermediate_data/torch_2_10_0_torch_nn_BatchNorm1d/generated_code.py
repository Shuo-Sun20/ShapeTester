import torch

def call_func(num_features, inputs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
    instance = torch.nn.BatchNorm1d(
        num_features=num_features,
        eps=eps,
        momentum=momentum,
        affine=affine,
        track_running_stats=track_running_stats
    )
    output = instance(inputs)
    return output

# Generate random input tensor for example usage
example_input = torch.randn(32, 64, 100)  # (N, C, L) format
example_output = call_func(
    num_features=64,
    inputs=example_input,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)