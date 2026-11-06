"""Downloads the M5 files and turns one store's daily sales into a weekly panel of items by weeks."""
import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
ZENODO_URL = "https://zenodo.org/api/records/12636070/files/m5-forecasting-accuracy.zip/content"
ZENODO_MD5 = "86f57416a314197f40a17cc6fc60cbb4"
EVENT_TYPES = ["Religious", "National", "Cultural", "Sporting"]
NEEDED_FILES = ["calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv"]


@dataclass
class Panel:
    """Weekly data for one store. Matrices are items x weeks; a price of NaN means the item is not listed."""

    store: str
    items: pd.DataFrame
    weeks: pd.DataFrame
    units: np.ndarray
    price: np.ndarray

    @property
    def listed(self):
        return ~np.isnan(self.price)


def download(raw_dir=RAW_DIR):
    """Download and verify the archive, and keep only the files the project uses."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if all((raw_dir / name).exists() for name in NEEDED_FILES):
        return raw_dir
    archive = raw_dir / "m5.zip"
    request = urllib.request.Request(ZENODO_URL, headers={"User-Agent": "Mozilla/5.0 (demand-forecasting-inventory-optimization)"})
    with urllib.request.urlopen(request, timeout=300) as response:
        archive.write_bytes(response.read())
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != ZENODO_MD5:
        archive.unlink()
        raise ValueError(f"Checksum mismatch for the M5 archive: {digest}")
    with zipfile.ZipFile(archive) as bundle:
        for name in NEEDED_FILES:
            bundle.extract(name, raw_dir)
    archive.unlink()
    return raw_dir


def weekly_calendar(calendar, state):
    """One row per full Walmart week (Saturday to Friday), with the events and SNAP days it contains."""
    calendar = calendar.copy()
    for event_type in EVENT_TYPES:
        calendar[event_type] = ((calendar["event_type_1"] == event_type) | (calendar["event_type_2"] == event_type)).astype(int)
    calendar["snap_days"] = calendar[f"snap_{state}"]
    weeks = calendar.groupby("wm_yr_wk", as_index=False, sort=True).agg(
        week_end=("date", "max"), days=("date", "size"), snap_days=("snap_days", "sum"),
        **{event_type: (event_type, "sum") for event_type in EVENT_TYPES})
    return weeks


def load_panel(store, raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir)
    calendar = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    sales = sales[sales["store_id"] == store].reset_index(drop=True)
    if sales.empty:
        raise ValueError(f"Unknown store: {store}")
    day_columns = [c for c in sales.columns if c.startswith("d_")]
    calendar = calendar[calendar["d"].isin(day_columns)].reset_index(drop=True)

    weeks = weekly_calendar(calendar, store.split("_")[0])
    weeks = weeks[weeks["days"] == 7].drop(columns="days").reset_index(drop=True)
    position = pd.Series(np.arange(len(weeks)), index=weeks["wm_yr_wk"])
    day_week = calendar.set_index("d")["wm_yr_wk"].reindex(day_columns).map(position).to_numpy()

    daily = sales[day_columns].to_numpy(dtype=float)
    units = np.zeros((len(sales), len(weeks)))
    keep = ~np.isnan(day_week)
    np.add.at(units.T, day_week[keep].astype(int), daily[:, keep].T)

    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    prices = prices[prices["store_id"] == store]
    table = prices.pivot(index="item_id", columns="wm_yr_wk", values="sell_price")
    price = table.reindex(index=sales["item_id"], columns=weeks["wm_yr_wk"]).to_numpy(dtype=float)

    items = sales[["item_id", "dept_id", "cat_id"]].copy()
    return Panel(store, items, weeks, units, price)


def suspected_stockouts(panel, threshold=0.001, window=8):
    """Weeks with no sales although the previous weeks' rate makes that very unlikely under a Poisson model.

    Recorded sales are not demand: an empty shelf also shows as zero. There is no stock data to confirm this,
    so these weeks are a flag, not a fact.
    """
    observed = np.where(panel.listed, panel.units, np.nan)
    rate = pd.DataFrame(observed).T.shift(1).rolling(window, min_periods=window).mean().T.to_numpy()
    with np.errstate(invalid="ignore"):
        return panel.listed & (panel.units == 0) & (np.exp(-rate) < threshold)
