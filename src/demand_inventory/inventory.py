"""Periodic-review (R, S) simulation with lost sales, and cost and segment calculations."""
import numpy as np
import pandas as pd


def simulate(order_up_to, demand, lead_time=1):
    """Simulate weekly ordering up to a target, item by item. Matrices are items x weeks.

    At the start of week t: receive what is due, then order up to order_up_to[:, t]. That order arrives at the start
    of week t + lead_time, so a target protects weeks t to t + lead_time. Demand that finds no stock is lost.
    """
    if lead_time < 1:
        raise ValueError("lead_time must be at least 1 week")
    n_items, n_weeks = demand.shape
    target = np.nan_to_num(order_up_to, nan=0.0)
    on_hand = target[:, 0].copy()
    pipeline = np.zeros((n_items, lead_time))
    out = {name: np.zeros((n_items, n_weeks)) for name in ("sold", "lost", "on_hand", "ordered", "received")}
    for t in range(n_weeks):
        arriving = pipeline[:, 0].copy()
        on_hand += arriving
        pipeline[:, :-1] = pipeline[:, 1:]
        pipeline[:, -1] = 0.0
        order = np.maximum(target[:, t] - on_hand - pipeline.sum(axis=1), 0.0)
        pipeline[:, -1] = order
        sold = np.minimum(on_hand, demand[:, t])
        on_hand -= sold
        out["sold"][:, t], out["lost"][:, t] = sold, demand[:, t] - sold
        out["on_hand"][:, t], out["ordered"][:, t], out["received"][:, t] = on_hand, order, arriving
    return out


def normal_target(forecast, sigma, z):
    """Order-up-to level from a point forecast and a normal safety stock, rounded up to whole units."""
    return np.ceil(np.clip(forecast + z * sigma, 0, None))


def summarise(result, price, cost_ratio=0.7, holding_rate=0.25, margin_ratio=0.3, skip=2):
    """Service and money over the simulation, after a short start-up. Unlisted weeks carry the item's last price."""
    filled = pd.DataFrame(price).T.ffill().bfill().T.to_numpy()
    sold, lost, on_hand = (result[k][:, skip:] for k in ("sold", "lost", "on_hand"))
    p = filled[:, skip:]
    demand = sold + lost
    weeks = sold.shape[1]
    stock_cost = (on_hand * p * cost_ratio).sum(axis=0)
    average_stock = stock_cost.mean()
    holding = average_stock * holding_rate * weeks / 52
    lost_margin = (lost * p).sum() * margin_ratio
    return {
        "unit_fill_pct": sold.sum() / demand.sum() * 100,
        "revenue_fill_pct": (sold * p).sum() / (demand * p).sum() * 100,
        "average_stock_at_cost": average_stock,
        "weeks_of_supply": on_hand.sum(axis=0).mean() / (demand.sum() / weeks),
        "stockout_item_weeks_pct": ((lost > 0).sum() / max((demand > 0).sum(), 1)) * 100,
        "lost_margin": lost_margin,
        "holding_cost": holding,
        "total_cost": lost_margin + holding,
    }


def abc_xyz(units, price, listed, abc_cuts=(0.8, 0.95), xyz_cuts=(0.5, 1.0)):
    """ABC by revenue share and XYZ by coefficient of variation of weekly units, from the given history."""
    observed = np.where(listed, units, np.nan)
    revenue = np.nansum(observed * price, axis=1)
    order = np.argsort(-revenue)
    share_before = np.concatenate(([0.0], np.cumsum(revenue[order])[:-1])) / revenue.sum()
    abc = np.empty(len(revenue), dtype="U1")
    abc[order] = np.select([share_before < abc_cuts[0], share_before < abc_cuts[1]], ["A", "B"], default="C")
    # items with no history in the window have a NaN variation and fall into Z
    seen = (~np.isnan(observed)).any(axis=1)
    cv = np.full(len(revenue), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv[seen] = np.nanstd(observed[seen], axis=1) / np.nanmean(observed[seen], axis=1)
    xyz = np.select([cv <= xyz_cuts[0], cv <= xyz_cuts[1]], ["X", "Y"], default="Z")
    return abc, xyz, cv


def frontier_at(frontier, fill, x="unit_fill_pct", y="average_stock_at_cost"):
    """Stock needed for a given fill rate, interpolated along one policy's trade-off curve."""
    points = frontier.sort_values(x)
    if fill < points[x].min() or fill > points[x].max():
        return np.nan
    return float(np.interp(fill, points[x], points[y]))
