"""Runs the rolling-origin backtest for one or more stores and saves the forecasts to data/processed.

    python scripts/run_backtest.py CA_1 TX_2 WI_3
    python scripts/run_backtest.py CA_1 --horizon=3

The horizon is the lead time plus the one-week review period. A store is skipped if its results already exist,
unless --force is given.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from demand_inventory.data import load_panel  # noqa: E402
from demand_inventory.forecast import backtest  # noqa: E402

args = sys.argv[1:]
force = "--force" in args
horizon = next((int(a.split("=")[1]) for a in args if a.startswith("--horizon=")), 2)
suffix = "" if horizon == 2 else f"_h{horizon}"
stores = [a for a in args if not a.startswith("--")] or ["CA_1", "TX_2", "WI_3"]
target = ROOT / "data" / "processed"
target.mkdir(parents=True, exist_ok=True)

for store in stores:
    path = target / f"forecasts_{store}{suffix}.csv.gz"
    if path.exists() and not force:
        print(f"{store}: already done")
        continue
    started = time.time()
    results = backtest(load_panel(store), horizon=horizon, log=lambda message: print(f"{store} {message}", flush=True))
    results.round(3).to_csv(path, index=False)
    print(f"{store}: {len(results):,} forecasts in {(time.time() - started) / 60:.1f} min", flush=True)
