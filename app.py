import pandas as pd
import streamlit as st

from src.pe_model import (
    DealAssumptions,
    build_comps_valuation,
    build_investment_memo,
    build_lbo_model,
    build_projection,
    build_returns_sensitivity,
    format_money,
    load_historical_financials,
    normalize_historical_financials,
)


st.set_page_config(
    page_title="Private Equity IC Dashboard",
    page_icon=":briefcase:",
    layout="wide",
)


@st.cache_data
def load_sample_financials() -> pd.DataFrame:
    return load_historical_financials("data/historical_financials.csv")


@st.cache_data
def load_public_comps() -> pd.DataFrame:
    return pd.read_csv("data/public_comps.csv")


def metric(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


st.title("Private Equity Investment Committee Dashboard")
st.caption("LBO returns, valuation, downside risk, debt paydown, and IC memo in one deal-screening workflow.")

with st.sidebar:
    st.header("Deal Assumptions")
    company_name = st.text_input("Target company", "Atlas Specialty Packaging")
    entry_multiple = st.slider("Entry EV / EBITDA", 5.0, 13.0, 8.5, 0.1)
    exit_multiple = st.slider("Exit EV / EBITDA", 5.0, 13.0, 8.5, 0.1)
    debt_multiple = st.slider("Opening debt / EBITDA", 1.0, 6.5, 4.5, 0.1)
    revenue_growth = st.slider("Annual revenue growth", 0.00, 0.20, 0.085, 0.005)
    target_margin = st.slider("Exit EBITDA margin", 0.10, 0.35, 0.205, 0.005)
    interest_rate = st.slider("Cash interest rate", 0.04, 0.15, 0.085, 0.005)
    tax_rate = st.slider("Cash tax rate", 0.00, 0.40, 0.25, 0.01)
    st.divider()
    uploaded_file = st.file_uploader("Upload target financials", type="csv")

assumptions = DealAssumptions(
    entry_multiple=entry_multiple,
    exit_multiple=exit_multiple,
    debt_multiple=debt_multiple,
    revenue_growth=revenue_growth,
    target_ebitda_margin=target_margin,
    cash_interest_rate=interest_rate,
    tax_rate=tax_rate,
)

try:
    historical = pd.read_csv(uploaded_file) if uploaded_file else load_sample_financials()
    historical = normalize_historical_financials(historical)
except Exception as exc:
    st.error(f"Could not load financials: {exc}")
    st.stop()

projection = build_projection(historical, assumptions)
lbo, summary = build_lbo_model(historical, assumptions)
comps = build_comps_valuation(load_public_comps(), float(historical.iloc[-1]["EBITDA"]))
sensitivity = build_returns_sensitivity(historical, assumptions)

top = st.columns(5)
with top[0]:
    metric("Recommendation", summary.recommendation)
with top[1]:
    metric("Base IRR", f"{summary.irr:.1%}", "Five-year sponsor IRR")
with top[2]:
    metric("MOIC", f"{summary.moic:.2f}x", "Exit equity value divided by sponsor equity")
with top[3]:
    metric("Sponsor Equity", format_money(summary.sponsor_equity))
with top[4]:
    metric("Entry EV", format_money(summary.entry_ev))

st.divider()

overview_tab, lbo_tab, valuation_tab, risk_tab, memo_tab, data_tab = st.tabs(
    ["Deal Snapshot", "LBO Model", "Valuation", "Risk & Sensitivity", "IC Memo", "Data"]
)

with overview_tab:
    st.subheader("Investment Snapshot")
    left, right = st.columns([1.2, 1])
    with left:
        snapshot = pd.concat(
            [
                historical[["Year", "Revenue", "EBITDA", "EBITDA_Margin"]].assign(Type="Historical"),
                projection[["Year", "Revenue", "EBITDA", "EBITDA_Margin"]].assign(Type="Projected"),
            ]
        )
        st.line_chart(snapshot.set_index("Year")[["Revenue", "EBITDA"]], use_container_width=True)
    with right:
        st.markdown("#### IC Readout")
        st.write(summary.recommendation_reason)
        st.write(f"Entry valuation: **{entry_multiple:.1f}x EBITDA**")
        st.write(f"Exit valuation: **{exit_multiple:.1f}x EBITDA**")
        st.write(f"Opening leverage: **{debt_multiple:.1f}x EBITDA**")
        st.write(f"Exit EBITDA margin: **{target_margin:.1%}**")

    st.markdown("#### Historical and Projected Margin")
    margin_view = pd.concat(
        [
            historical[["Year", "EBITDA_Margin"]].assign(Period="Historical"),
            projection[["Year", "EBITDA_Margin"]].assign(Period="Projected"),
        ]
    )
    st.line_chart(margin_view.set_index("Year")["EBITDA_Margin"], use_container_width=True)

with lbo_tab:
    st.subheader("LBO Sources, Uses, and Debt Paydown")
    sources_uses = pd.DataFrame(
        {
            "Item": ["Purchase enterprise value", "Opening debt", "Sponsor equity"],
            "Amount": [summary.entry_ev, summary.opening_debt, summary.sponsor_equity],
        }
    )
    left, right = st.columns([0.8, 1.2])
    with left:
        st.dataframe(
            sources_uses,
            use_container_width=True,
            hide_index=True,
            column_config={"Amount": st.column_config.NumberColumn("Amount", format="$%.1fM")},
        )
    with right:
        st.bar_chart(lbo.set_index("Year")[["Beginning_Debt", "Debt_Paydown", "Ending_Debt"]], use_container_width=True)

    st.markdown("#### LBO Projection")
    st.dataframe(
        lbo,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Revenue_Growth": st.column_config.NumberColumn("Revenue Growth", format="%.1f%%"),
            "EBITDA_Margin": st.column_config.NumberColumn("EBITDA Margin", format="%.1f%%"),
        },
    )

with valuation_tab:
    st.subheader("Valuation View")
    val_cols = st.columns(4)
    val_cols[0].metric("Entry EV", format_money(summary.entry_ev))
    val_cols[1].metric("DCF EV", format_money(summary.dcf_ev))
    val_cols[2].metric("Median Comp Multiple", f"{comps['EV_EBITDA'].median():.1f}x")
    val_cols[3].metric("Exit EV", format_money(summary.exit_ev))

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Public Comps")
        st.dataframe(
            comps,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Revenue_Growth": st.column_config.NumberColumn("Revenue Growth", format="%.1f%%"),
                "EBITDA_Margin": st.column_config.NumberColumn("EBITDA Margin", format="%.1f%%"),
                "EV_EBITDA": st.column_config.NumberColumn("EV / EBITDA", format="%.1fx"),
                "Implied_EV": st.column_config.NumberColumn("Implied EV", format="$%.1fM"),
            },
        )
    with right:
        football = pd.DataFrame(
            {
                "Method": ["Entry price", "DCF", "Comps low", "Comps median", "Comps high", "Exit value"],
                "Enterprise Value": [
                    summary.entry_ev,
                    summary.dcf_ev,
                    comps["Implied_EV"].min(),
                    comps["Implied_EV"].median(),
                    comps["Implied_EV"].max(),
                    summary.exit_ev,
                ],
            }
        )
        st.markdown("#### Valuation Football")
        st.bar_chart(football.set_index("Method"), use_container_width=True)

with risk_tab:
    st.subheader("IRR Sensitivity")
    formatted = sensitivity.copy()
    for column in formatted.columns:
        if column != "Revenue Growth":
            formatted[column] = formatted[column].map(lambda value: f"{value:.1%}")
    st.dataframe(formatted, use_container_width=True, hide_index=True)

    downside_assumptions = DealAssumptions(
        entry_multiple=entry_multiple,
        exit_multiple=max(5.0, exit_multiple - 1.0),
        debt_multiple=debt_multiple,
        revenue_growth=max(0.0, revenue_growth - 0.035),
        target_ebitda_margin=max(0.10, target_margin - 0.025),
        cash_interest_rate=interest_rate + 0.01,
        tax_rate=tax_rate,
    )
    _, downside = build_lbo_model(historical, downside_assumptions)
    risk_cols = st.columns(3)
    risk_cols[0].metric("Downside IRR", f"{downside.irr:.1%}")
    risk_cols[1].metric("Downside MOIC", f"{downside.moic:.2f}x")
    risk_cols[2].metric("Downside Recommendation", downside.recommendation)

    st.markdown("#### Core Diligence Questions")
    st.write("- Is revenue growth supported by customer retention, pricing power, or market expansion?")
    st.write("- Are margin improvements operationally credible or just spreadsheet expansion?")
    st.write("- Can the company service debt in a higher-rate or lower-growth environment?")
    st.write("- What exit buyer universe supports the underwritten exit multiple?")

with memo_tab:
    st.subheader("Investment Committee Memo")
    memo = build_investment_memo(summary, assumptions, company_name)
    st.markdown(memo)
    st.download_button(
        "Download IC memo",
        data=memo,
        file_name="private_equity_ic_memo.md",
        mime="text/markdown",
    )

with data_tab:
    st.subheader("Input Financials")
    st.dataframe(historical, use_container_width=True, hide_index=True)
    st.subheader("Projection")
    st.dataframe(projection, use_container_width=True, hide_index=True)
