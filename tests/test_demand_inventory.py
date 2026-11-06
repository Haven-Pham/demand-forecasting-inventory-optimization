import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from demand_inventory import data, forecast, inventory, metrics, policies  # noqa: E402


def make_panel(n_items=6, n_weeks=70, seed=1):
    rng = np.random.default_rng(seed)
    units = rng.poisson(5, (n_items, n_weeks)).astype(float)
    price = np.full((n_items, n_weeks), 4.0)
    price[0, :10] = np.nan  # listed late
    units[0, :10] = 0
    weeks = pd.DataFrame({"wm_yr_wk": range(n_weeks), "week_end": pd.date_range("2014-01-03", periods=n_weeks, freq="7D"),
                          "snap_days": 0, "Religious": 0, "National": 0, "Cultural": 0, "Sporting": 0})
    items = pd.DataFrame({"item_id": [f"I{i}" for i in range(n_items)], "dept_id": "D", "cat_id": "C"})
    return data.Panel("TEST", items, weeks, units, price)


def test_simulation_conserves_stock():
    rng = np.random.default_rng(0)
    demand = rng.poisson(4, (5, 30)).astype(float)
    target = rng.integers(0, 15, (5, 30)).astype(float)
    out = inventory.simulate(target, demand)
    start = target[:, 0]
    assert np.allclose(start + out["received"].sum(axis=1), out["sold"].sum(axis=1) + out["on_hand"][:, -1])
    assert np.allclose(out["sold"] + out["lost"], demand)
    assert (out["on_hand"] >= 0).all() and (out["lost"] >= 0).all()


def test_order_arrives_after_the_lead_time():
    demand = np.array([[5.0, 0, 0, 0]])
    target = np.array([[5.0, 5, 5, 5]])
    out = inventory.simulate(target, demand, lead_time=1)
    # Week 0 sells the starting stock. The order placed at the start of week 1 arrives at the start of week 2.
    assert out["ordered"][0].tolist() == [0, 5, 0, 0]
    assert out["received"][0].tolist() == [0, 0, 5, 0]


def test_demand_above_stock_is_lost():
    out = inventory.simulate(np.array([[3.0, 0]]), np.array([[5.0, 1.0]]))
    assert out["sold"][0].tolist() == [3, 0]
    assert out["lost"][0].tolist() == [2, 1]


def test_lead_time_must_be_positive():
    with pytest.raises(ValueError):
        inventory.simulate(np.ones((1, 2)), np.ones((1, 2)), lead_time=0)


def test_features_do_not_see_the_future():
    panel = make_panel()
    origin, horizon = 60, 2
    before = forecast.features_at(panel, origin, horizon)
    changed = make_panel()
    changed.units[:, origin + 1:] = 999
    after = forecast.features_at(changed, origin, horizon)
    columns = forecast.feature_columns(before)
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert (after["actual"] == 999 * horizon).all()


def test_target_is_cumulative_demand():
    panel = make_panel()
    frame = forecast.features_at(panel, 50, 2)
    row = frame[frame["item_index"] == 3].iloc[0]
    assert row["actual"] == panel.units[3, 51] + panel.units[3, 52]


def test_items_not_listed_through_the_horizon_are_excluded():
    panel = make_panel()
    panel.price[2, 52] = np.nan
    assert 2 not in forecast.features_at(panel, 50, 2)["item_index"].to_numpy()


def test_croston_rate_for_regular_demand():
    units = np.full((1, 40), 6.0)
    rate = forecast.croston_sba(units, np.ones_like(units, dtype=bool), alpha=0.1)
    assert rate[0, -1] == pytest.approx(6 * 0.95)


def test_croston_rate_for_sales_every_second_week():
    units = np.tile([0.0, 8.0], (1, 100))
    rate = forecast.croston_sba(units, np.ones_like(units, dtype=bool), alpha=0.1)
    assert rate[0, -1] == pytest.approx(8 / 2 * 0.95, rel=0.02)


def test_seasonal_naive_uses_the_same_weeks_last_year():
    panel = make_panel()
    seasonal = forecast.baselines(panel, 2)["seasonal_naive"]
    assert seasonal[3, 60] == panel.units[3, 9] + panel.units[3, 10]


def test_rolling_error_only_uses_resolved_forecasts():
    fc, actual = np.zeros((1, 12)), np.ones((1, 12))
    actual[0, 10:] = 100  # large errors at the end must not affect the origin at which they are unresolved
    rmse = forecast.rolling_error(fc, actual, horizon=2, weeks=4)
    assert rmse[0, 11] == pytest.approx(1.0)


def test_suspected_stockout_flags_an_unlikely_zero():
    panel = make_panel(n_items=1, n_weeks=20)
    panel.price[:] = 4.0
    panel.units[0, :] = 30
    panel.units[0, 15] = 0
    flags = data.suspected_stockouts(panel)
    assert flags[0, 15] and flags[0].sum() == 1


def test_slow_seller_zero_is_not_flagged():
    panel = make_panel(n_items=1, n_weeks=20)
    panel.price[:] = 4.0
    panel.units[0, :] = 1
    panel.units[0, 15] = 0
    assert not data.suspected_stockouts(panel).any()


def test_abc_xyz_classes():
    units = np.array([[100.0] * 10, [10.0, 0, 0, 0, 0] * 2, [1.0] * 10])
    price = np.ones_like(units)
    abc, xyz, _ = inventory.abc_xyz(units, price, np.ones_like(units, dtype=bool))
    assert abc[0] == "A" and abc[2] == "C"
    assert xyz[0] == "X" and xyz[1] == "Z"


def test_summary_measures():
    result = {"sold": np.array([[0, 0, 8.0, 10.0]]), "lost": np.array([[0, 0, 2.0, 0.0]]),
              "on_hand": np.array([[0, 0, 4.0, 6.0]])}
    s = inventory.summarise(result, np.full((1, 4), 10.0), cost_ratio=0.5, margin_ratio=0.5, holding_rate=0.52, skip=2)
    assert s["unit_fill_pct"] == pytest.approx(90)
    assert s["average_stock_at_cost"] == pytest.approx(25)
    assert s["lost_margin"] == pytest.approx(10)
    assert s["holding_cost"] == pytest.approx(25 * 0.52 * 2 / 52)


def test_frontier_interpolates_and_refuses_to_extrapolate():
    frontier = pd.DataFrame({"unit_fill_pct": [90.0, 100.0], "average_stock_at_cost": [10.0, 20.0]})
    assert inventory.frontier_at(frontier, 95) == 15
    assert np.isnan(inventory.frontier_at(frontier, 80))


def test_backtest_runs_and_never_trains_on_unknown_targets(monkeypatch):
    panel = make_panel(n_items=40, n_weeks=120)
    monkeypatch.setattr(forecast, "N_TREES", 3)
    seen = []
    original = forecast._train

    def spy(frame, objective, alpha=None):
        seen.append(frame["origin"].max())
        return original(frame, objective, alpha)

    monkeypatch.setattr(forecast, "_train", spy)
    results = forecast.backtest(panel, test_weeks=10, warmup_weeks=6, refit_every=5, quantiles=(0.5, 0.9), log=lambda _: None)
    assert results["origin"].max() == 117
    first_origin_of_each_refit = sorted(results["origin"].unique())[::5]
    # one mean model and two quantile models at each refit, each trained only on targets resolved by then
    assert seen[::3] == [origin - 2 for origin in first_origin_of_each_refit]
    assert (results["q90"] >= results["q50"]).all()


def test_new_item_without_history_falls_into_c_and_z():
    units = np.array([[5.0, 6.0], [0.0, 0.0]])
    price = np.array([[1.0, 1.0], [np.nan, np.nan]])
    abc, xyz, _ = inventory.abc_xyz(units, price, ~np.isnan(price))
    assert abc[1] == "C" and xyz[1] == "Z"


def test_suspected_stockout_week_is_filled_with_the_prior_rate():
    panel = make_panel(n_items=1, n_weeks=20)
    panel.price[:] = 4.0
    panel.units[0, :] = 30
    panel.units[0, 15] = 0
    adjusted = policies.demand_with_suspected_stockouts(panel, data.suspected_stockouts(panel))
    assert adjusted[0, 15] == 30 and adjusted[0, 14] == 30


def test_accuracy_measures():
    actual, fc = np.array([10.0, 10.0]), np.array([12.0, 6.0])
    assert metrics.wape(actual, fc) == 30
    assert metrics.bias(actual, fc) == -10
    assert metrics.pinball(np.array([10.0]), np.array([8.0]), 0.9) == pytest.approx(1.8)
    assert metrics.coverage(actual, np.array([10.0, 9.0])) == 50
