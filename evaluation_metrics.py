import numpy as np
from sklearn.metrics import mean_pinball_loss
from numpy.typing import ArrayLike


def crps_score_sklearn(y_true: ArrayLike, y_pred_quantiles: ArrayLike, quantiles: list[float]) -> float:
    """
    Calculates the Continuous Ranked Probability Score (CRPS) by averaging
    the pinball loss across all predicted quantiles.

    This is useful for when the actual target is multiple quantiles. This is
    instead of a normal regression or multi-classification problem.

    Like when someone is needs to know 'when will this event happen?'
        - when will a payment be made?
        - what day will this invoice be pay?
        - etc, etc, etc.

    To solve problems like this the best thing to output is a CDF.
        - this let's you say:
            - '50% chance it will be paid in 10 days' and a 90% chance it will be paid in 20 days'

    Args:
        y_true: The true values.
        y_pred_quantiles: The predicted quantiles.
        quantiles: The quantiles to use.

    Returns:
        The CRPS score.
    """
    y_pred_quantiles = np.array(y_pred_quantiles)
    all_quantile_losses = []
    for i, q in enumerate(quantiles):
        all_quantile_losses.append(
            mean_pinball_loss(y_true, y_pred_quantiles[:, i], alpha=q)
        )

    return np.mean(all_quantile_losses)  # type: ignore[misc]
