"""Forecast accuracy measures."""
import numpy as np


def wape(actual, forecast):
    """Weighted absolute percentage error: total absolute error as a share of total demand."""
    return float(np.abs(actual - forecast).sum() / actual.sum() * 100)


def bias(actual, forecast):
    """Total over-forecast as a share of total demand. Positive means forecasts were too high."""
    return float((forecast - actual).sum() / actual.sum() * 100)


def pinball(actual, forecast, q):
    """Mean pinball loss of a quantile forecast."""
    gap = actual - forecast
    return float(np.mean(np.maximum(q * gap, (q - 1) * gap)))


def coverage(actual, forecast):
    """Share of cases where demand did not exceed the quantile forecast."""
    return float((actual <= forecast).mean() * 100)
