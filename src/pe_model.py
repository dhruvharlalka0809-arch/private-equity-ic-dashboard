from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DealAssumptions:
    entry_multiple: float = 8.5
    exit_multiple: float = 8.5
    debt_multiple: float = 4.5
    revenue_growth: float = 0.085
    target_ebitda_margin: float = 0.205
    tax_rate: float = 0.25
    cash_interest_rate: float = 0.085
    capex_pct_revenue: float = 0.04
    nwc_pct_revenue: float = 0.13
    wacc: float = 0.11
    terminal_growth: float = 0.025
    hold_years: int = 5


@dataclass(frozen=True)
class ReturnsSummary:
    entry_ev: float
    sponsor_equity: float
    opening_debt: float
    exit_ev: float
    exit_equity: float
    ending_debt: float
    moic: float
    irr: float
    dcf_ev: float
    recommendation: str
    recommendation_reason: str


def load_historical_financials(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_historical_financials(df)


def normalize_historical_financials(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Year", "Revenue", "EBITDA", "Capex", "Net_Working_Capital"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    output = df.copy()
    for column in ["Year", "Revenue", "EBITDA", "Capex", "Net_Working_Capital"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["Year", "Revenue", "EBITDA"]).sort_values("Year")
    output["EBITDA_Margin"] = divide_series(output["EBITDA"], output["Revenue"])
    output["Revenue_Growth"] = output["Revenue"].pct_change().fillna(0)
    return output.reset_index(drop=True)


def build_projection(historical: pd.DataFrame, assumptions: DealAssumptions) -> pd.DataFrame:
    normalized = normalize_historical_financials(historical)
    base = normalized.iloc[-1]
    base_margin = float(base["EBITDA_Margin"])
    base_year = int(base["Year"])

    rows = []
    previous_revenue = float(base["Revenue"])
    previous_nwc = float(base["Net_Working_Capital"])

    for year_index in range(1, assumptions.hold_years + 1):
        year = base_year + year_index
        revenue = previous_revenue * (1 + assumptions.revenue_growth)
        margin_step = (assumptions.target_ebitda_margin - base_margin) / assumptions.hold_years
        ebitda_margin = base_margin + (margin_step * year_index)
        ebitda = revenue * ebitda_margin
        capex = revenue * assumptions.capex_pct_revenue
        nwc = revenue * assumptions.nwc_pct_revenue
        nwc_investment = max(0.0, nwc - previous_nwc)
        cash_taxes = max(0.0, ebitda * assumptions.tax_rate)
        unlevered_fcf = ebitda - cash_taxes - capex - nwc_investment
        rows.append(
            {
                "Year": year,
                "Revenue": revenue,
                "Revenue_Growth": assumptions.revenue_growth,
                "EBITDA_Margin": ebitda_margin,
                "EBITDA": ebitda,
                "Capex": capex,
                "Net_Working_Capital": nwc,
                "NWC_Investment": nwc_investment,
                "Cash_Taxes": cash_taxes,
                "Unlevered_FCF": unlevered_fcf,
            }
        )
        previous_revenue = revenue
        previous_nwc = nwc

    return pd.DataFrame(rows)


def build_lbo_model(historical: pd.DataFrame, assumptions: DealAssumptions) -> tuple[pd.DataFrame, ReturnsSummary]:
    normalized = normalize_historical_financials(historical)
    projection = build_projection(normalized, assumptions)
    entry_ebitda = float(normalized.iloc[-1]["EBITDA"])
    entry_ev = entry_ebitda * assumptions.entry_multiple
    opening_debt = entry_ebitda * assumptions.debt_multiple
    sponsor_equity = entry_ev - opening_debt
    debt_balance = opening_debt

    rows = []
    for _, row in projection.iterrows():
        interest = debt_balance * assumptions.cash_interest_rate
        cash_available_for_debt = max(0.0, float(row["Unlevered_FCF"]) - interest)
        debt_paydown = min(debt_balance, cash_available_for_debt)
        ending_debt = debt_balance - debt_paydown
        levered_fcf = float(row["Unlevered_FCF"]) - interest - debt_paydown
        rows.append(
            {
                **row.to_dict(),
                "Beginning_Debt": debt_balance,
                "Cash_Interest": interest,
                "Debt_Paydown": debt_paydown,
                "Ending_Debt": ending_debt,
                "Levered_FCF_After_Debt_Paydown": levered_fcf,
            }
        )
        debt_balance = ending_debt

    lbo = pd.DataFrame(rows)
    exit_ebitda = float(lbo.iloc[-1]["EBITDA"])
    exit_ev = exit_ebitda * assumptions.exit_multiple
    ending_debt = float(lbo.iloc[-1]["Ending_Debt"])
    exit_equity = exit_ev - ending_debt
    moic = divide(exit_equity, sponsor_equity)
    irr = calculate_irr([-sponsor_equity] + [0.0] * (assumptions.hold_years - 1) + [exit_equity])
    dcf_ev = calculate_dcf_ev(projection, assumptions)
    recommendation, reason = recommend_deal(irr=irr, moic=moic, debt_remaining=ending_debt, opening_debt=opening_debt)

    summary = ReturnsSummary(
        entry_ev=entry_ev,
        sponsor_equity=sponsor_equity,
        opening_debt=opening_debt,
        exit_ev=exit_ev,
        exit_equity=exit_equity,
        ending_debt=ending_debt,
        moic=moic,
        irr=irr,
        dcf_ev=dcf_ev,
        recommendation=recommendation,
        recommendation_reason=reason,
    )
    return lbo, summary


def calculate_dcf_ev(projection: pd.DataFrame, assumptions: DealAssumptions) -> float:
    discounted_fcf = 0.0
    for period, fcf in enumerate(projection["Unlevered_FCF"], start=1):
        discounted_fcf += float(fcf) / ((1 + assumptions.wacc) ** period)

    final_fcf = float(projection.iloc[-1]["Unlevered_FCF"])
    terminal_value = final_fcf * (1 + assumptions.terminal_growth) / (assumptions.wacc - assumptions.terminal_growth)
    discounted_terminal = terminal_value / ((1 + assumptions.wacc) ** len(projection))
    return discounted_fcf + discounted_terminal


def calculate_irr(cash_flows: list[float], iterations: int = 100) -> float:
    low = -0.95
    high = 1.5
    for _ in range(iterations):
        mid = (low + high) / 2
        value = npv(mid, cash_flows)
        if value > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def npv(rate: float, cash_flows: list[float]) -> float:
    return sum(cf / ((1 + rate) ** idx) for idx, cf in enumerate(cash_flows))


def recommend_deal(irr: float, moic: float, debt_remaining: float, opening_debt: float) -> tuple[str, str]:
    debt_reduction = 1 - divide(debt_remaining, opening_debt)
    if irr >= 0.22 and moic >= 2.3 and debt_reduction >= 0.35:
        return "Buy", "Base case clears PE-style return hurdles with meaningful debt paydown."
    if irr >= 0.16 and moic >= 1.8:
        return "Watchlist", "Returns are plausible but need stronger margin expansion, entry price discipline, or lower leverage risk."
    return "Pass", "Base case does not clear return hurdles with enough downside protection."


def build_returns_sensitivity(historical: pd.DataFrame, assumptions: DealAssumptions) -> pd.DataFrame:
    exit_multiples = [assumptions.exit_multiple - 1.0, assumptions.exit_multiple - 0.5, assumptions.exit_multiple, assumptions.exit_multiple + 0.5, assumptions.exit_multiple + 1.0]
    growth_rates = [assumptions.revenue_growth - 0.03, assumptions.revenue_growth - 0.015, assumptions.revenue_growth, assumptions.revenue_growth + 0.015, assumptions.revenue_growth + 0.03]
    rows = []
    for growth in growth_rates:
        row = {"Revenue Growth": f"{growth:.1%}"}
        for multiple in exit_multiples:
            scenario = DealAssumptions(
                entry_multiple=assumptions.entry_multiple,
                exit_multiple=multiple,
                debt_multiple=assumptions.debt_multiple,
                revenue_growth=growth,
                target_ebitda_margin=assumptions.target_ebitda_margin,
                tax_rate=assumptions.tax_rate,
                cash_interest_rate=assumptions.cash_interest_rate,
                capex_pct_revenue=assumptions.capex_pct_revenue,
                nwc_pct_revenue=assumptions.nwc_pct_revenue,
                wacc=assumptions.wacc,
                terminal_growth=assumptions.terminal_growth,
                hold_years=assumptions.hold_years,
            )
            _, summary = build_lbo_model(historical, scenario)
            row[f"{multiple:.1f}x Exit"] = summary.irr
        rows.append(row)
    return pd.DataFrame(rows)


def build_comps_valuation(comps: pd.DataFrame, entry_ebitda: float) -> pd.DataFrame:
    output = comps.copy()
    output["Implied_EV"] = output["EV_EBITDA"] * entry_ebitda
    return output.sort_values("EV_EBITDA").reset_index(drop=True)


def build_investment_memo(summary: ReturnsSummary, assumptions: DealAssumptions, company_name: str) -> str:
    return f"""### Investment Committee Memo

**Target:** {company_name}

**Recommendation:** {summary.recommendation}

**Deal thesis:** Acquire a defensible middle-market business at {assumptions.entry_multiple:.1f}x EBITDA and underwrite value creation through revenue growth, EBITDA margin expansion, and debt paydown.

**Returns case:** The base case generates a {summary.moic:.2f}x MOIC and {summary.irr:.1%} IRR over {assumptions.hold_years} years. Sponsor equity required is {format_money(summary.sponsor_equity)} against entry enterprise value of {format_money(summary.entry_ev)}.

**Value creation levers:** The model assumes {assumptions.revenue_growth:.1%} annual revenue growth and margin expansion to {assumptions.target_ebitda_margin:.1%}. Debt falls from {format_money(summary.opening_debt)} to {format_money(summary.ending_debt)} by exit.

**Valuation view:** DCF-implied enterprise value is {format_money(summary.dcf_ev)}, compared with entry enterprise value of {format_money(summary.entry_ev)}.

**Key risks:** Entry multiple discipline, margin execution, leverage tolerance, cash conversion, and exit multiple compression.

**IC view:** {summary.recommendation_reason}
"""


def format_money(value: float) -> str:
    return f"${value:,.1f}M"


def divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def divide_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, pd.NA)).fillna(0)
