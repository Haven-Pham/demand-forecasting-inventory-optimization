"""Forecasts of cumulative demand over the protection interval, with a rolling-origin backtest.

An origin is the last week whose sales are known. A forecast made at origin t covers weeks t+1 to t+horizon.
"""
import numpy as np
import pandas as pd

from .data import EVENT_TYPES, suspected_stockouts

QUANTILES = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
CATEGORICAL = ["dept", "cat"]
# Chosen on 13 origins before the test window. Larger settings improved the 2-week WAPE by about 0.2 points at twice the time.
LGBM_PARAMS = dict(learning_rate=0.05, num_leaves=31, max_bin=63, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                   bagging_freq=1, lambda_l2=1.0, seed=42, deterministic=True, force_col_wise=True, verbosity=-1)
N_TREES = 200
TRAIN_ORIGINS = 156


def _trailing(values, listed, origin, weeks):
    """Mean of the last `weeks` listed observations up to the origin, by item. NaN when there are none."""
    start = max(0, origin - weeks + 1)
    window = values[:, start:origin + 1]
    mask = listed[:, start:origin + 1]
    count = mask.sum(axis=1)
    total = np.where(mask, window, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def features_at(panel, origin, horizon):
    """Features known at the origin, for items listed through the whole horizon.

    Prices and events inside the horizon are used because a retailer plans them in advance. Nothing else comes
    from after the origin.
    """
    units, price, listed = panel.units, panel.price, panel.listed
    future = np.arange(origin + 1, origin + 1 + horizon)
    in_range = future[-1] < units.shape[1]
    known_future = future[future < units.shape[1]]
    rows = listed[:, known_future].all(axis=1) & listed[:, origin]
    idx = np.flatnonzero(rows)
    frame = {"item_index": idx, "origin": origin}
    for lag in range(4):
        frame[f"lag_{lag + 1}"] = np.where(listed[idx, origin - lag], units[idx, origin - lag], np.nan) if origin - lag >= 0 else np.nan
    for weeks in (4, 8, 13, 26, 52):
        frame[f"mean_{weeks}"] = _trailing(units, listed, origin, weeks)[idx]
    start = max(0, origin - 12)
    recent = np.where(listed[idx, start:origin + 1], units[idx, start:origin + 1], np.nan)
    # `idx` items are listed at the origin, so each row has at least one observation
    frame["std_13"] = np.nanstd(recent, axis=1)
    frame["zero_share_13"] = np.where(np.isnan(recent), 0, recent == 0).sum(axis=1) / (~np.isnan(recent)).sum(axis=1)
    frame["age"] = listed[idx, :origin + 1].sum(axis=1)

    last_year = future - 52
    if last_year[0] >= 0:
        same_weeks = listed[idx][:, last_year].all(axis=1)
        frame["last_year"] = np.where(same_weeks, units[idx][:, last_year].sum(axis=1), np.nan)
    else:
        frame["last_year"] = np.nan

    planned = price[idx][:, known_future].mean(axis=1)
    usual = _trailing(price, listed, origin, 13)[idx]
    frame["price"] = planned
    frame["price_vs_usual"] = planned / usual
    weeks = panel.weeks
    for event_type in EVENT_TYPES + ["snap_days"]:
        frame[f"future_{event_type}"] = weeks[event_type].to_numpy()[known_future].sum()
    frame["week_of_year"] = weeks["week_end"].iloc[known_future[0]].isocalendar()[1]
    frame["dept"] = panel.items["dept_id"].astype("category").cat.codes.to_numpy()[idx]
    frame["cat"] = panel.items["cat_id"].astype("category").cat.codes.to_numpy()[idx]

    frame = pd.DataFrame(frame)
    frame["actual"] = units[idx][:, future].sum(axis=1) if in_range else np.nan
    return frame


def feature_columns(frame):
    return [c for c in frame.columns if c not in ("item_index", "origin", "actual", "flagged")]


def simple_smoothing(units, listed, alpha=0.2):
    """Exponentially smoothed weekly rate, by item and origin. Uses listed weeks only."""
    level = np.full(units.shape, np.nan)
    current = np.full(units.shape[0], np.nan)
    for t in range(units.shape[1]):
        seen = listed[:, t]
        current = np.where(seen & np.isnan(current), units[:, t], current)
        current = np.where(seen, alpha * units[:, t] + (1 - alpha) * current, current)
        level[:, t] = current
    return level


def croston_sba(units, listed, alpha=0.1):
    """Syntetos-Boylan approximation of Croston's method: weekly rate for intermittent demand, by item and origin."""
    size = np.full(units.shape[0], np.nan)
    interval = np.full(units.shape[0], np.nan)
    since = np.ones(units.shape[0])
    rate = np.full(units.shape, np.nan)
    for t in range(units.shape[1]):
        sale = listed[:, t] & (units[:, t] > 0)
        first = sale & np.isnan(size)
        later = sale & ~first
        size[first], interval[first] = units[first, t], since[first]
        size[later] += alpha * (units[later, t] - size[later])
        interval[later] += alpha * (since[later] - interval[later])
        since = np.where(sale, 1.0, since + listed[:, t])
        rate[:, t] = (1 - alpha / 2) * size / interval
    rate[np.isnan(rate) & listed] = 0.0
    return rate


def baselines(panel, horizon):
    """Item x origin matrices of cumulative forecasts over the horizon, from each simple method."""
    units, listed = panel.units, panel.listed
    observed = pd.DataFrame(np.where(listed, units, np.nan)).T
    moving = observed.rolling(8, min_periods=1).mean().T.to_numpy()
    smooth = simple_smoothing(units, listed)
    sba = croston_sba(units, listed)
    seasonal = np.full(units.shape, np.nan)
    for t in range(52 - horizon, units.shape[1]):
        first = t + 1 - 52
        if first >= 0:
            seasonal[:, t] = np.where(listed[:, first:first + horizon], units[:, first:first + horizon], np.nan).sum(axis=1)
    return {"moving_average": moving * horizon, "smoothing": smooth * horizon, "croston_sba": sba * horizon, "seasonal_naive": seasonal}


def rolling_error(forecast, actual, horizon, weeks=26):
    """Root mean squared error of each item's earlier forecasts, using only forecasts already resolved at each origin."""
    error = (actual - forecast) ** 2
    frame = pd.DataFrame(error).T
    rmse = np.sqrt(frame.rolling(weeks, min_periods=4).mean()).shift(horizon).T.to_numpy()
    return rmse


def _train(frame, objective, alpha=None):
    import lightgbm as lgb

    columns = feature_columns(frame)
    params = {**LGBM_PARAMS, "objective": objective}
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.2
    if alpha is not None:
        params["alpha"] = alpha
    data = lgb.Dataset(frame[columns], label=frame["actual"], categorical_feature=CATEGORICAL, free_raw_data=False)
    return lgb.train(params, data, num_boost_round=N_TREES)


def backtest(panel, test_weeks=52, warmup_weeks=30, refit_every=13, horizon=2, quantiles=QUANTILES, log=print):
    """Rolling-origin backtest of the LightGBM models.

    Models are refitted every `refit_every` weeks on origins whose target was already known at the refit, and
    predict each following origin. The warm-up origins exist only to provide forecast errors before the test.
    """
    n_weeks = panel.units.shape[1]
    first_test_origin = n_weeks - test_weeks - 1
    origins = list(range(first_test_origin - warmup_weeks, n_weeks - horizon))
    flagged = suspected_stockouts(panel)

    cache = {}

    def frame_at(origin):
        if origin not in cache:
            cache[origin] = features_at(panel, origin, horizon)
            future = np.arange(origin + 1, min(origin + 1 + horizon, n_weeks))
            cache[origin]["flagged"] = flagged[cache[origin]["item_index"].to_numpy()][:, future].any(axis=1)
        return cache[origin]

    results = []
    models = {}
    for position, origin in enumerate(origins):
        if position % refit_every == 0:
            last_known = origin - horizon
            train_origins = range(max(60, last_known - TRAIN_ORIGINS + 1), last_known + 1)
            train = pd.concat([frame_at(o) for o in train_origins], ignore_index=True)
            train = train[~train["flagged"]]
            log(f"origin {origin}: fitting on {len(train):,} rows")
            models = {"mean": _train(train, "tweedie")}
            for q in quantiles:
                models[f"q{int(q * 100)}"] = _train(train, "quantile", q)
        frame = frame_at(origin)
        columns = feature_columns(frame)
        out = frame[["item_index", "origin", "actual", "flagged"]].copy()
        for name, model in models.items():
            out[name] = np.clip(model.predict(frame[columns]), 0, None)
        # independent quantile models can cross; sorting keeps them a valid distribution
        names = [f"q{int(q * 100)}" for q in quantiles]
        out[names] = np.sort(out[names].to_numpy(), axis=1)
        results.append(out)
    return pd.concat(results, ignore_index=True)
