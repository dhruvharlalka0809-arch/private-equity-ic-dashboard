import unittest

import pandas as pd

from src.pe_model import (
    DealAssumptions,
    build_comps_valuation,
    build_investment_memo,
    build_lbo_model,
    build_memo_context,
    build_projection,
    build_returns_sensitivity,
    build_scenario_summary,
    build_value_creation_bridge,
    normalize_historical_financials,
)


class PrivateEquityModelTests(unittest.TestCase):
    def setUp(self):
        self.historical = pd.DataFrame(
            {
                "Year": [2023, 2024, 2025],
                "Revenue": [80.0, 90.0, 100.0],
                "EBITDA": [12.0, 15.0, 18.0],
                "Capex": [3.2, 3.6, 4.0],
                "Net_Working_Capital": [10.4, 11.7, 13.0],
            }
        )
        self.assumptions = DealAssumptions(
            entry_multiple=8.0,
            exit_multiple=8.5,
            debt_multiple=4.0,
            revenue_growth=0.08,
            target_ebitda_margin=0.22,
            depreciation_pct_revenue=0.03,
        )

    def test_projection_builds_five_year_forecast(self):
        projection = build_projection(self.historical, self.assumptions)

        self.assertEqual(len(projection), 5)
        self.assertEqual(projection.iloc[0]["Year"], 2026)
        self.assertGreater(projection.iloc[-1]["Revenue"], projection.iloc[0]["Revenue"])
        self.assertAlmostEqual(projection.iloc[-1]["EBITDA_Margin"], 0.22)
        self.assertIn("Depreciation", projection.columns)
        self.assertIn("EBIT", projection.columns)

    def test_lbo_returns_are_positive_and_reconcile(self):
        lbo, summary = build_lbo_model(self.historical, self.assumptions)

        self.assertGreater(summary.entry_ev, 0)
        self.assertAlmostEqual(summary.sponsor_equity, summary.entry_ev - summary.opening_debt)
        self.assertGreater(summary.moic, 0)
        self.assertEqual(len(lbo), self.assumptions.hold_years)
        self.assertIn("Interest_Coverage", lbo.columns)
        self.assertIn("Ending_Cash", lbo.columns)
        self.assertGreaterEqual(summary.ending_cash, 0)

    def test_cash_taxes_use_ebt_not_ebitda(self):
        lbo, _ = build_lbo_model(self.historical, self.assumptions)
        first_year = lbo.iloc[0]
        expected_tax = max(0.0, first_year["EBT"] * self.assumptions.tax_rate)

        self.assertAlmostEqual(first_year["LBO_Cash_Taxes"], expected_tax)

    def test_working_capital_can_release_cash(self):
        assumptions = DealAssumptions(revenue_growth=0.0, nwc_pct_revenue=0.08)
        projection = build_projection(self.historical, assumptions)

        self.assertLess(projection.iloc[0]["NWC_Investment"], 0)

    def test_missing_columns_raise_validation_error(self):
        with self.assertRaises(ValueError):
            normalize_historical_financials(pd.DataFrame({"Revenue": [100]}))

    def test_comps_valuation_adds_implied_ev(self):
        comps = pd.DataFrame({"Company": ["A"], "EV_EBITDA": [9.0]})
        result = build_comps_valuation(comps, entry_ebitda=20.0, size_discount=0.20)

        self.assertEqual(result.iloc[0]["Implied_EV"], 180.0)
        self.assertEqual(result.iloc[0]["Size_Adjusted_Implied_EV"], 144.0)

    def test_sensitivity_returns_matrix(self):
        sensitivity = build_returns_sensitivity(self.historical, self.assumptions)

        self.assertEqual(len(sensitivity), 5)
        self.assertIn("Revenue Growth", sensitivity.columns)

    def test_scenarios_and_value_bridge_are_available(self):
        lbo, summary = build_lbo_model(self.historical, self.assumptions)
        scenarios = build_scenario_summary(self.historical, self.assumptions)
        bridge = build_value_creation_bridge(self.historical, lbo, summary, self.assumptions)

        self.assertEqual(set(scenarios["Scenario"]), {"Bank case", "Management case", "Downside case"})
        self.assertIn("Debt paydown", set(bridge["Bridge Item"]))
        self.assertIn("Exit equity value", set(bridge["Bridge Item"]))
        self.assertIn("Convention", bridge.columns)

    def test_memo_contains_recommendation_and_returns(self):
        lbo, summary = build_lbo_model(self.historical, self.assumptions)
        context = build_memo_context(self.historical, self.assumptions, lbo)
        memo = build_investment_memo(summary, self.assumptions, "TestCo", context)

        self.assertIn("Investment Committee Memo", memo)
        self.assertIn("TestCo", memo)
        self.assertIn("MOIC", memo)
        self.assertIn("Downside case", memo)
        self.assertIn("No interim dividend recap", memo)
        self.assertIn("DCF uses unlevered FCF", memo)


if __name__ == "__main__":
    unittest.main()
