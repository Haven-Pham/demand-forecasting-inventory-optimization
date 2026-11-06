"""Executes the analysis notebook in place and refreshes its outputs, figures and tables.

    python scripts/run_notebook.py

Run scripts/run_backtest.py first: the notebook reads the forecasts in data/processed.
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "demand_forecasting_inventory_optimization.ipynb"

nb = nbformat.read(NOTEBOOK, as_version=4)
NotebookClient(nb, timeout=3600, kernel_name="python3",
               resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
for cell in nb.cells:
    cell.metadata.pop("execution", None)
nbformat.write(nb, NOTEBOOK)
print(f"Executed {NOTEBOOK.name}")
