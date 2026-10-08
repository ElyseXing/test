"""Rate-based conversion from normalized EEG features to spike trains."""

import torch


def rate_encode(
    features: torch.Tensor,
    n_steps: int = 32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample Bernoulli spikes; output shape is [time, batch, features]."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    probabilities = torch.sigmoid(features).clamp_(0.0, 1.0)
    random_values = torch.rand(
        (n_steps, *probabilities.shape),
        dtype=probabilities.dtype,
        device=probabilities.device,
        generator=generator,
    )
    return (random_values < probabilities.unsqueeze(0)).to(probabilities.dtype)
