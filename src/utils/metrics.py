import torch


def relative_l2_error(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return the mean relative L2 error over a batch.

    Expected shape is [batch, n_grid], but any trailing dimensions are flattened.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got {prediction.shape} and {target.shape}"
        )

    if prediction.dim() == 1:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)

    prediction_flat = prediction.reshape(prediction.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    numerator = torch.linalg.vector_norm(prediction_flat - target_flat, dim=1)
    denominator = torch.linalg.vector_norm(target_flat, dim=1).clamp_min(eps)
    return (numerator / denominator).mean()
