import torch

def call_func(inputs, correction=1, fweights=None, aweights=None):
    return torch.cov(
        input=inputs,
        correction=correction,
        fweights=fweights,
        aweights=aweights
    )

torch.manual_seed(42)
example_inputs = torch.randn(3, 5)
example_fweights = torch.randint(1, 10, (5,))
example_aweights = torch.rand(5)
example_output = call_func(
    inputs=example_inputs,
    correction=1,
    fweights=example_fweights,
    aweights=example_aweights
)