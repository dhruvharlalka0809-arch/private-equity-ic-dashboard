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
    depreciation_pct_revenue: float = 0.03
    nwc_pct_revenue: float = 0.13
    minimum_cash_pct_revenue: float = 0.015
    mandatory_amortization_pct_opening_debt: float = 0.025
    cash_sweep_pct: float = 0.75
    comp_size_discount: float = 0.15
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
    ending_cash: float
    moic: float
    irr: float
    dcf_ev: float
    recommendation: str
    recommendation_reason: str


@dataclass(frozen=True)
class MemoContext:
    downside_irr: float
    downside_moic: float
    downside_net_debt_to_ebitda: float
    base_net_debt_to_ebitda: float
    base_interest_coverage: float
    bridge_convention: str = "EBITDA growth valued at entry multiple; multiple movement applied to exit EBITDA"
    interim_distribution_note: str = "No interim dividend recap or sponsor distribution is assumed; excess cash accumulates until exit."
    dcf_tax_note: str = "DCF uses unlevered FCF taxed on EBIT, while the LBO uses levered EBT after cash interest."


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
        depreciation = revenue * assumptions.depreciation_pct_revenue
        ebit = ebitda - depreciation
        nwc = revenue * assumptions.nwc_pct_revenue
        nwc_investment = nwc - previous_nwc
        cash_taxes = max(0.0, ebit * assumptions.tax_rate)
        unlevered_fcf = ebitda - cash_taxes - capex - nwc_investment
        rows.append(
            {
                "Year": year,
                "Revenue": revenue,
                "Revenue_Growth": assumptions.revenue_growth,
                "EBITDA_Margin": ebitda_margin,
                "EBITDA": ebitda,
                "Depreciation": depreciation,
                "EBIT": ebit,
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
    cash_balance = 0.0

    rows = []
    for _, row in projection.iterrows():
        interest = debt_balance * assumptions.cash_interest_rate
        ebt = float(row["EBIT"]) - interest
        cash_taxes = max(0.0, ebt * assumptions.tax_rate)
        cash_flow_before_debt_paydown = (
            float(row["EBITDA"])
            - cash_taxes
            - float(row["Capex"])
            - float(row["NWC_Investment"])
            - interest
        )
        minimum_cash = float(row["Revenue"]) * assumptions.minimum_cash_pct_revenue
        available_cash = max(0.0, cash_balance + cash_flow_before_debt_paydown - minimum_cash)
        scheduled_amortization = min(
            debt_balance,
            available_cash,
            opening_debt * assumptions.mandatory_amortization_pct_opening_debt,
        )
        cash_after_amortization = max(0.0, available_cash - scheduled_amortization)
        optional_sweep = min(
            debt_balance - scheduled_amortization,
            cash_after_amortization * assumptions.cash_sweep_pct,
        )
        debt_paydown = scheduled_amortization + optional_sweep
        ending_debt = debt_balance - debt_paydown
        ending_cash = max(0.0, cash_balance + cash_flow_before_debt_paydown - debt_paydown)
        levered_fcf_after_debt_paydown = ending_cash - cash_balance
        interest_coverage = divide(float(row["EBITDA"]), interest)
        net_debt = max(0.0, ending_debt - ending_cash)
        rows.append(
            {
                **row.to_dict(),
                "Beginning_Debt": debt_balance,
                "Beginning_Cash": cash_balance,
                "Cash_Interest": interest,
                "EBT": ebt,
                "LBO_Cash_Taxes": cash_taxes,
                "Cash_Flow_Before_Debt_Paydown": cash_flow_before_debt_paydown,
                "Minimum_Cash": minimum_cash,
                "Scheduled_Amortization": scheduled_amortization,
                "Optional_Cash_Sweep": optional_sweep,
                "Debt_Paydown": debt_paydown,
                "Ending_Debt": ending_debt,
                "Ending_Cash": ending_cash,
                "Net_Debt": net_debt,
                "Debt_to_EBITDA": divide(ending_debt, float(row["EBITDA"])),
                "Net_Debt_to_EBITDA": divide(net_debt, float(row["EBITDA"])),
                "Interest_Coverage": interest_coverage,
                "Levered_FCF_After_Debt_Paydown": levered_fcf_after_debt_paydown,
            }
        )
        debt_balance = ending_debt
        cash_balance = ending_cash

    lbo = pd.DataFrame(rows)
    exit_ebitda = float(lbo.iloc[-1]["EBITDA"])
    exit_ev = exit_ebitda * assumptions.exit_multiple
    ending_debt = float(lbo.iloc[-1]["Ending_Debt"])
    ending_cash = float(lbo.iloc[-1]["Ending_Cash"])
    exit_equity = exit_ev - ending_debt + ending_cash
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
        ending_cash=ending_cash,
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
                depreciation_pct_revenue=assumptions.depreciation_pct_revenue,
                nwc_pct_revenue=assumptions.nwc_pct_revenue,
                minimum_cash_pct_revenue=assumptions.minimum_cash_pct_revenue,
                mandatory_amortization_pct_opening_debt=assumptions.mandatory_amortization_pct_opening_debt,
                cash_sweep_pct=assumptions.cash_sweep_pct,
                comp_size_discount=assumptions.comp_size_discount,
                wacc=assumptions.wacc,
                terminal_growth=assumptions.terminal_growth,
                hold_years=assumptions.hold_years,
            )
            _, summary = build_lbo_model(historical, scenario)
            row[f"{multiple:.1f}x Exit"] = summary.irr
        rows.append(row)
    return pd.DataFrame(rows)


def build_comps_valuation(comps: pd.DataFrame, entry_ebitda: float, size_discount: float = 0.15) -> pd.DataFrame:
    output = comps.copy()
    output["Implied_EV"] = output["EV_EBITDA"] * entry_ebitda
    output["Size_Adjusted_EV_EBITDA"] = output["EV_EBITDA"] * (1 - size_discount)
    output["Size_Adjusted_Implied_EV"] = output["Size_Adjusted_EV_EBITDA"] * entry_ebitda
    return output.sort_values("EV_EBITDA").reset_index(drop=True)


def build_scenario_summary(historical: pd.DataFrame, assumptions: DealAssumptions) -> pd.DataFrame:
    scenarios = [
        ("Bank case", assumptions.revenue_growth - 0.025, assumptions.target_ebitda_margin - 0.015, assumptions.exit_multiple - 0.5),
        ("Management case", assumptions.revenue_growth, assumptions.target_ebitda_margin, assumptions.exit_multiple),
        ("Downside case", max(0.0, assumptions.revenue_growth - 0.055), max(0.10, assumptions.target_ebitda_margin - 0.035), max(5.0, assumptions.exit_multiple - 1.25)),
    ]
    rows = []
    for name, growth, margin, exit_multiple in scenarios:
        scenario = replace_assumptions(
            assumptions,
            revenue_growth=growth,
            target_ebitda_margin=margin,
            exit_multiple=exit_multiple,
        )
        _, summary = build_lbo_model(historical, scenario)
        rows.append(
            {
                "Scenario": name,
                "Revenue Growth": growth,
                "Exit EBITDA Margin": margin,
                "Exit Multiple": exit_multiple,
                "IRR": summary.irr,
                "MOIC": summary.moic,
                "Ending Net Debt": max(0.0, summary.ending_debt - summary.ending_cash),
                "Recommendation": summary.recommendation,
            }
        )
    return pd.DataFrame(rows)


def build_value_creation_bridge(historical: pd.DataFrame, lbo: pd.DataFrame, summary: ReturnsSummary, assumptions: DealAssumptions) -> pd.DataFrame:
    normalized = normalize_historical_financials(historical)
    entry_ebitda = float(normalized.iloc[-1]["EBITDA"])
    exit_ebitda = float(lbo.iloc[-1]["EBITDA"])
    ebitda_growth_value = (exit_ebitda - entry_ebitda) * assumptions.entry_multiple
    multiple_expansion_value = exit_ebitda * (assumptions.exit_multiple - assumptions.entry_multiple)
    debt_paydown_value = summary.opening_debt - summary.ending_debt
    cash_buildup_value = summary.ending_cash
    modeled_exit_equity = (
        summary.sponsor_equity
        + ebitda_growth_value
        + multiple_expansion_value
        + debt_paydown_value
        + cash_buildup_value
    )
    unexplained = summary.exit_equity - modeled_exit_equity
    rows = [
        ("Sponsor equity at entry", summary.sponsor_equity),
        ("EBITDA growth", ebitda_growth_value),
        ("Multiple expansion / contraction", multiple_expansion_value),
        ("Debt paydown", debt_paydown_value),
        ("Cash buildup", cash_buildup_value),
    ]
    if abs(unexplained) > 0.01:
        rows.append(("Other / rounding", unexplained))
    rows.append(("Exit equity value", summary.exit_equity))
    bridge = pd.DataFrame(rows, columns=["Bridge Item", "Equity Value Contribution"])
    bridge["Convention"] = "EBITDA growth valued at entry multiple"
    bridge.loc[bridge["Bridge Item"] == "Multiple expansion / contraction", "Convention"] = "Applied to exit EBITDA"
    bridge.loc[bridge["Bridge Item"].isin(["Debt paydown", "Cash buildup"]), "Convention"] = "Balance sheet value creation"
    bridge.loc[bridge["Bridge Item"].isin(["Sponsor equity at entry", "Exit equity value"]), "Convention"] = "Equity value"
    return bridge


def build_memo_context(historical: pd.DataFrame, assumptions: DealAssumptions, lbo: pd.DataFrame) -> MemoContext:
    downside_assumptions = replace_assumptions(
        assumptions,
        exit_multiple=max(5.0, assumptions.exit_multiple - 1.0),
        revenue_growth=max(0.0, assumptions.revenue_growth - 0.035),
        target_ebitda_margin=max(0.10, assumptions.target_ebitda_margin - 0.025),
        cash_interest_rate=assumptions.cash_interest_rate + 0.01,
    )
    downside_lbo, downside_summary = build_lbo_model(historical, downside_assumptions)
    return MemoContext(
        downside_irr=downside_summary.irr,
        downside_moic=downside_summary.moic,
        downside_net_debt_to_ebitda=float(downside_lbo.iloc[-1]["Net_Debt_to_EBITDA"]),
        base_net_debt_to_ebitda=float(lbo.iloc[-1]["Net_Debt_to_EBITDA"]),
        base_interest_coverage=float(lbo.iloc[-1]["Interest_Coverage"]),
    )


def replace_assumptions(assumptions: DealAssumptions, **updates) -> DealAssumptions:
    values = assumptions.__dict__.copy()
    values.update(updates)
    return DealAssumptions(**values)


def build_investment_memo(summary: ReturnsSummary, assumptions: DealAssumptions, company_name: str, context: MemoContext | None = None) -> str:
    downside_line = ""
    credit_line = ""
    assumptions_line = ""
    bridge_line = ""
    if context:
        downside_line = f"\n**Downside case:** Under the downside case, IRR falls to {context.downside_irr:.1%}, MOIC falls to {context.downside_moic:.2f}x, and year-five net leverage is {context.downside_net_debt_to_ebitda:.1f}x EBITDA.\n"
        credit_line = f"\n**Credit view:** Base case year-five net leverage is {context.base_net_debt_to_ebitda:.1f}x EBITDA and interest coverage is {context.base_interest_coverage:.1f}x.\n"
        assumptions_line = f"\n**Modeling notes:** {context.interim_distribution_note} {context.dcf_tax_note}\n"
        bridge_line = f"\n**Value bridge convention:** {context.bridge_convention}.\n"

    return f"""### Investment Committee Memo

**Target:** {company_name}

**Recommendation:** {summary.recommendation}

**Deal thesis:** Acquire a defensible middle-market business at {assumptions.entry_multiple:.1f}x EBITDA and underwrite value creation through revenue growth, EBITDA margin expansion, and debt paydown.

**Returns case:** The base case generates a {summary.moic:.2f}x MOIC and {summary.irr:.1%} IRR over {assumptions.hold_years} years. Sponsor equity required is {format_money(summary.sponsor_equity)} against entry enterprise value of {format_money(summary.entry_ev)}.
{downside_line}
{credit_line}

**Value creation levers:** The model assumes {assumptions.revenue_growth:.1%} annual revenue growth and margin expansion to {assumptions.target_ebitda_margin:.1%}. Debt falls from {format_money(summary.opening_debt)} to {format_money(summary.ending_debt)} and cash builds to {format_money(summary.ending_cash)} by exit.
{bridge_line}

**Valuation view:** DCF-implied enterprise value is {format_money(summary.dcf_ev)}, compared with entry enterprise value of {format_money(summary.entry_ev)}.
{assumptions_line}

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
