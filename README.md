# Private Equity Investment Committee Dashboard

A Streamlit dashboard for evaluating a private equity acquisition through LBO returns, valuation, debt paydown, downside sensitivity, and an investment committee memo.

## What It Does

- Builds a five-year operating forecast from historical revenue, EBITDA, capex, and working capital
- Calculates entry enterprise value, opening debt, sponsor equity, debt paydown, cash buildup, exit value, MOIC, and IRR
- Models D&A, EBIT, EBT-based cash taxes, working capital investment or release, scheduled amortization, and optional cash sweep
- Compares entry valuation against DCF-implied value and size-adjusted public comparable EV/EBITDA multiples
- Runs sensitivity analysis across revenue growth and exit multiple assumptions
- Compares bank case, management case, and downside case outputs
- Tracks credit metrics including Debt / EBITDA, Net Debt / EBITDA, and interest coverage
- Decomposes return creation through an equity value bridge across EBITDA growth, multiple movement, debt paydown, and cash buildup
- Produces a PE-style investment committee memo with recommendation, thesis, risks, and value creation levers
- Supports CSV upload for custom target company financials

## Why This Project Matters

Private equity recruiting rewards candidates who can connect financial modeling to investment judgment. This project shows LBO mechanics, valuation thinking, downside analysis, and investment memo communication in one workflow.

## Tech Stack

- Python
- Streamlit
- Pandas
- Standard-library tests with `unittest`

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app runs with sample financials for `Atlas Specialty Packaging` by default.

## Input Data Format

Upload a CSV with:

```csv
Year,Revenue,EBITDA,Capex,Net_Working_Capital
2023,82.0,13.1,3.3,10.8
2024,91.5,15.6,3.7,12.0
2025,103.0,18.2,4.1,13.6
```

## Validate

```bash
python scripts/validate.py
```

## Portfolio Talking Points

- Built an LBO model that calculates sponsor equity, scheduled amortization, cash sweep, cash buildup, exit equity value, MOIC, and IRR
- Added valuation triangulation using DCF and size-adjusted public comparable EV/EBITDA multiples
- Designed downside, bank, and management case analysis for PE-style investment risk assessment
- Added credit metrics and a return attribution bridge to explain where equity value creation comes from
- Generated an investment committee memo that connects model output to a Buy / Watchlist / Pass recommendation

## Author

Dhruv Harlalka

MBA Finance, Middlesex University Dubai
