"""Builds the Excel results workbook from the CSV tables in reports/tables.

    python scripts/build_excel_report.py

Run scripts/run_notebook.py first so the tables are current.
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"

CONFIG = {
    "file": "Demand_Forecast_Inventory_Results.xlsx",
    "title": "How Much Is a Better Forecast Worth?",
    "subtitle": "Demand forecasting and inventory policy for 3,049 products at one store (CA_1), tested over 52 weeks ending May 2016.",
    "findings": [
        "LightGBM has a two-week WAPE of 30.6%, against 32.0% for exponential smoothing and 33.7% for an 8-week moving average.",
        "At a 95% fill rate, LightGBM with error-based safety stock needs 10.7% less stock than the textbook policy (95% interval 8.8-12.3%). The saving is 8-11% across three stores.",
        "Under the assumed costs, recovered margin is worth about six times the holding cost saved: about $13,500 against $2,150 a year at this store.",
        "Calibrated quantile models do not beat simple error-based safety stock: they need 2-3% more stock at a 95% fill rate across the three stores.",
        "All four ABC service-tier plans tested needed 2-11% more stock than a uniform target with the same revenue fill.",
        "A second week of lead time adds 47-50% more stock at a 95% fill rate, over four times the saving from the better forecast.",
    ],
    "sheets": {
        "Data profile": ("data_profile.csv", "the data and its gaps"),
        "Forecast accuracy": ("forecast_accuracy.csv", "WAPE and bias by method"),
        "Accuracy by group": ("forecast_accuracy_by_group.csv", "WAPE by category, ABC and XYZ class, and planned price change"),
        "Quantile calibration": ("quantile_calibration.csv", "share of demand covered by each quantile model"),
        "Policy frontier": ("policy_frontier.csv", "fill rate, stock and cost for each policy and service target"),
        "Stock needed": ("stock_needed_at_fill.csv", "stock required by each policy at three fill rates"),
        "Stock savings": ("stock_savings.csv", "savings between policies, with 95% bootstrap intervals"),
        "Value of the forecast": ("value_of_the_forecast.csv", "holding less stock against losing fewer sales"),
        "Stock by segment": ("stock_by_segment.csv", "stock and revenue share by ABC-XYZ segment"),
        "ABC service tiers": ("abc_service_tiers.csv", "tiered service targets against one uniform target"),
        "Lead time cost": ("lead_time_cost.csv", "stock needed with one-week and two-week lead times"),
        "Two-week lead frontier": ("policy_frontier_two_week_lead_time.csv", "the policy frontier with a two-week lead time"),
        "Store robustness": ("store_robustness.csv", "accuracy and savings in three stores"),
        "Stockout robustness": ("stockout_robustness.csv", "savings with suspected stockout weeks filled in"),
        "Cost sensitivity": ("cost_sensitivity.csv", "best target and cost across margin and holding rates"),
    },
    "figure": "03_stock_service_tradeoff.png",
    "source": "M5 Walmart sales data (Zenodo record 12636070). Margin 30%, holding rate 25% a year and all replenishment settings are assumptions.",
}

HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(bold=True, color="FFFFFF")
TITLE_FONT = Font(bold=True, size=16, color="1F4E79")


def number_format(column, series):
    name = column.lower()
    if not pd.api.types.is_numeric_dtype(series):
        return None
    if "pct" in name or name.startswith(("wape", "low_", "high_")):
        return '0.0"%"'
    if any(key in name for key in ("stock", "cost", "margin", "usd")):
        return '"$"#,##0'
    if pd.api.types.is_integer_dtype(series) or series.abs().max() >= 1000:
        return "#,##0"
    return "0.00"


def add_table_sheet(wb, title, frame):
    ws = wb.create_sheet(title[:31])
    ws.append(list(frame.columns))
    for row in frame.itertuples(index=False):
        ws.append([None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row])
    for cell in ws[1]:
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 48
    for idx, column in enumerate(frame.columns, start=1):
        fmt = number_format(column, frame[column])
        letter = get_column_letter(idx)
        if fmt:
            for cell in ws[letter][1:]:
                cell.number_format = fmt
        longest = max((len(str(v)) for v in frame[column].head(200)), default=8)
        ws.column_dimensions[letter].width = min(max(longest + 2, 14), 44)
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions


def main():
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = CONFIG["title"]
    ws["A1"].font = TITLE_FONT
    ws["A2"] = CONFIG["subtitle"]
    ws["A2"].font = Font(italic=True, color="555555")
    row = 4
    ws.cell(row, 1, "Key findings").font = Font(bold=True, size=12)
    for finding in CONFIG["findings"]:
        row += 1
        ws.cell(row, 1, f"• {finding}").alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row].height = 32
    row += 2
    ws.cell(row, 1, "Sheets in this workbook").font = Font(bold=True, size=12)
    for sheet, (_, description) in CONFIG["sheets"].items():
        row += 1
        ws.cell(row, 1, f"{sheet}: {description}")
    row += 2
    ws.cell(row, 1, "Source: " + CONFIG["source"]).font = Font(color="555555")
    ws.column_dimensions["A"].width = 120
    if (FIGURES / CONFIG["figure"]).exists():
        image = Image(str(FIGURES / CONFIG["figure"]))
        scale = 620 / image.width
        image.width, image.height = image.width * scale, image.height * scale
        ws.add_image(image, f"A{row + 2}")

    for sheet, (csv_name, _) in CONFIG["sheets"].items():
        add_table_sheet(wb, sheet, pd.read_csv(TABLES / csv_name))

    out = ROOT / "reports" / CONFIG["file"]
    wb.save(out)
    print(f"Wrote {out.relative_to(ROOT)} with {len(wb.sheetnames)} sheets")


if __name__ == "__main__":
    main()
