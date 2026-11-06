"""Turns forecasts into order-up-to targets and compares replenishment policies on the same weeks and items."""
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import forecast as fc
from . import inventory as inv
from .data import ROOT

TARGETS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
POLICIES = {
    "textbook": "Moving average + demand variability",
    "smoothing": "Smoothing + forecast error",
    "lgbm_error": "LightGBM + forecast error",
    "lgbm_quantile": "LightGBM quantiles",
}
PROCESSED = ROOT / "data" / "processed"


@dataclass
class Backtest:
    """Item x origin matrices for one store. Test weeks are the last `test_weeks` weeks of the panel."""

    panel: object
    horizon: int
    actual: np.ndarray
    available: np.ndarray
    models: dict
    baselines: dict
    flagged: np.ndarray
    test_weeks: int = 52

    @property
    def first_test_week(self):
        return self.panel.units.shape[1] - self.test_weeks

    def evaluated_origins(self):
        """Origins whose orders affect the test. An order for week t is decided with data through week t-1."""
        n_weeks = self.panel.units.shape[1]
        return np.arange(self.first_test_week - 1, n_weeks - self.horizon)


def load_backtest(panel, horizon=2, path=None, test_weeks=52):
    suffix = "" if horizon == 2 else f"_h{horizon}"
    path = Path(path) if path else PROCESSED / f"forecasts_{panel.store}{suffix}.csv.gz"
    table = pd.read_csv(path)
    shape = panel.units.shape
    items, origins = table["item_index"].to_numpy(), table["origin"].to_numpy()

    def matrix(values):
        out = np.full(shape, np.nan)
        out[items, origins] = values
        return out

    models = {c: matrix(table[c].to_numpy()) for c in table.columns if c == "mean" or c.startswith("q")}
    flagged = np.zeros(shape, dtype=bool)
    flagged[items, origins] = table["flagged"].to_numpy(dtype=bool)
    available = np.zeros(shape, dtype=bool)
    available[items, origins] = True
    return Backtest(panel, horizon, matrix(table["actual"].to_numpy()), available, models, fc.baselines(panel, horizon), flagged, test_weeks)


def demand_std(panel, weeks=26):
    """Standard deviation of weekly demand by item and origin, from listed weeks only."""
    observed = pd.DataFrame(np.where(panel.listed, panel.units, np.nan)).T
    return observed.rolling(weeks, min_periods=4).std().T.to_numpy()


def error_sigma(bt, forecast):
    """Forecast error by item and origin, falling back on a Poisson-like spread for new items."""
    actual = cumulative_actual(bt.panel, bt.horizon)
    sigma = fc.rolling_error(forecast, actual, bt.horizon)
    fallback = np.sqrt(np.clip(forecast, 1, None))
    return np.where(np.isnan(sigma), fallback, sigma)


def cumulative_actual(panel, horizon):
    """Demand over the weeks after each origin, for every origin (not only those in the LightGBM backtest)."""
    units = pd.DataFrame(panel.units).T
    return units.rolling(horizon).sum().shift(-horizon).T.to_numpy()


def targets_for(bt, policy, target):
    """Order-up-to level for each week: the level for week t comes from the forecast made at origin t-1."""
    z = NormalDist().inv_cdf(target)
    if policy == "textbook":
        level = fc_level = bt.baselines["moving_average"]
        std = demand_std(bt.panel)
        std = np.where(np.isnan(std), np.sqrt(np.clip(fc_level, 1, None)), std)
        by_origin = inv.normal_target(level, std * np.sqrt(bt.horizon), z)
    elif policy == "smoothing":
        level = bt.baselines["smoothing"]
        by_origin = inv.normal_target(level, error_sigma(bt, level), z)
    elif policy == "lgbm_error":
        level = bt.models["mean"]
        by_origin = inv.normal_target(level, error_sigma(bt, level), z)
    elif policy == "lgbm_quantile":
        by_origin = np.ceil(bt.models[f"q{int(round(target * 100))}"])
    else:
        raise ValueError(f"Unknown policy: {policy}")
    # every policy plans the same items at the same origins
    by_origin = np.where(bt.available, by_origin, np.nan)
    start = bt.first_test_week
    order_up_to = np.full((bt.panel.units.shape[0], bt.test_weeks), np.nan)
    width = min(bt.test_weeks, by_origin.shape[1] - (start - 1))
    order_up_to[:, :width] = by_origin[:, start - 1:start - 1 + width]
    return order_up_to


def run_policy(bt, order_up_to, lead_time=None):
    """Simulate over the test weeks."""
    lead_time = lead_time or bt.horizon - 1
    demand = bt.panel.units[:, bt.first_test_week:]
    return inv.simulate(order_up_to, demand, lead_time=lead_time)


def prices_in_test(bt):
    return bt.panel.price[:, bt.first_test_week:]


def segment_targets(bt, policy, segments, plan):
    """Order-up-to levels when each segment has its own service target. `plan` maps a segment to a target."""
    chosen = pd.Series(segments).map(plan).to_numpy()
    out = np.full((bt.panel.units.shape[0], bt.test_weeks), np.nan)
    for target in set(plan.values()):
        rows = chosen == target
        out[rows] = targets_for(bt, policy, target)[rows]
    return out


def demand_with_suspected_stockouts(panel, flags, window=8):
    """Sales with each suspected stockout week replaced by the item's rate over the preceding weeks."""
    observed = pd.DataFrame(np.where(panel.listed, panel.units, np.nan)).T
    rate = observed.shift(1).rolling(window, min_periods=window).mean().T.to_numpy()
    return np.where(flags, np.round(rate), panel.units)


def frontier(bt, targets=TARGETS, policies=POLICIES, **costs):
    """Fill rate and stock for every policy at every service target."""
    rows = []
    for policy in policies:
        for target in targets:
            result = run_policy(bt, targets_for(bt, policy, target))
            rows.append({"policy": policy, "target_pct": target * 100, **inv.summarise(result, prices_in_test(bt), **costs)})
    return pd.DataFrame(rows)


def item_measures(result, price, cost_ratio=0.7, skip=2):
    """Per-item units sold, units lost and average stock at cost: the pieces that a bootstrap resamples."""
    filled = pd.DataFrame(price).T.ffill().bfill().T.to_numpy()
    sold = result["sold"][:, skip:].sum(axis=1)
    lost = result["lost"][:, skip:].sum(axis=1)
    stock = (result["on_hand"][:, skip:] * filled[:, skip:] * cost_ratio).mean(axis=1)
    return sold, lost, stock


def bootstrap_saving(bt, reference, challenger, fill, targets=TARGETS, draws=500, seed=7):
    """Saving in stock at a fill rate from the challenger against the reference, resampling items with replacement."""
    measures = {}
    for policy in (reference, challenger):
        measures[policy] = [item_measures(run_policy(bt, targets_for(bt, policy, t)), prices_in_test(bt)) for t in targets]
    rng = np.random.default_rng(seed)
    n_items = bt.panel.units.shape[0]

    def stock_at(policy, weights):
        fills, stocks = [], []
        for sold, lost, stock in measures[policy]:
            fills.append((sold * weights).sum() / ((sold + lost) * weights).sum() * 100)
            stocks.append((stock * weights).sum())
        order = np.argsort(fills)
        fills, stocks = np.array(fills)[order], np.array(stocks)[order]
        if fill < fills[0] or fill > fills[-1]:
            return np.nan
        return np.interp(fill, fills, stocks)

    point = 1 - stock_at(challenger, np.ones(n_items)) / stock_at(reference, np.ones(n_items))
    draws_out = []
    for _ in range(draws):
        weights = np.bincount(rng.integers(0, n_items, n_items), minlength=n_items)
        draws_out.append(1 - stock_at(challenger, weights) / stock_at(reference, weights))
    draws_out = np.array(draws_out)
    low, high = np.nanpercentile(draws_out, [2.5, 97.5])
    return {"fill_pct": fill, "saving_pct": float(point * 100), "low_pct": float(low * 100), "high_pct": float(high * 100)}
