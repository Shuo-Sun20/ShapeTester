import torch
import torch.nn as nn

def call_func(
    inputs: list[torch.Tensor],
    distance_function: callable = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Call the TripletMarginWithDistanceLoss API.
    
    Args:
        inputs: List containing three tensors [anchor, positive, negative]
        distance_function: Optional distance function callable
        margin: Non-negative margin value
        swap: Whether to use distance swapping
        reduction: Reduction method - 'none', 'mean', or 'sum'
        
    Returns:
        torch.Tensor: The computed loss value
    """
    if len(inputs) != 3:
        raise ValueError("inputs must contain exactly three tensors: [anchor, positive, negative]")
    
    anchor, positive, negative = inputs
    loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=distance_function,
        margin=margin,
        swap=swap,
        reduction=reduction
    )
    return loss_fn(anchor, positive, negative)

# Generate random tensors for testing
batch_size = 4
embedding_dim = 16
anchor = torch.randn(batch_size, embedding_dim)
positive = torch.randn(batch_size, embedding_dim)
negative = torch.randn(batch_size, embedding_dim)

# Call the function with default parameters
example_output = call_func(
    inputs=[anchor, positive, negative],
    distance_function=None,
    margin=1.0,
    swap=False,
    reduction="mean"
)