# Data

This project uses the [M5 Forecasting Accuracy dataset](https://zenodo.org/records/12636070) (DOI 10.5281/zenodo.12636070): daily unit sales
for 3,049 Walmart products in 10 US stores, with weekly prices and an event calendar, from January 2011 to May 2016.

`python scripts/download_data.py` downloads the archive, verifies its MD5 checksum and extracts the three files used here into `data/raw/`:

| File | Content |
|---|---|
| `sales_train_evaluation.csv` | Daily unit sales by item and store |
| `sell_prices.csv` | Weekly selling price by item and store. No row means the item was not listed |
| `calendar.csv` | Dates, Walmart weeks, events and SNAP days |

`python scripts/run_backtest.py` writes item-level forecasts to `data/processed/`.

## Why the data are not in this repository

The Zenodo record lists a CC BY 4.0 license, but it is a copy "taken over from Kaggle". I could not find a primary statement of redistribution terms
from the competition organisers, so this repository stores neither the raw files nor item-level forecasts. It holds code and aggregated results. One
figure in the notebook shows the sales of a single example item to illustrate the simulation.

## Reference

Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of
Forecasting*, 38(4), 1346–1364. https://doi.org/10.1016/j.ijforecast.2021.10.009
