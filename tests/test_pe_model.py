import unittest

import pandas as pd

from src.pe_model import (
    DealAssumptions,
    build_comps_valuation,
    build_investment_memo,
    build_lbo_model,
    build_projection,
    build_returns_sensitivity,
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
        )

    def test_projection_builds_five_year_forecast(self):
        projection = build_projection(self.historical, self.assumptions)

        self.assertEqual(len(projection), 5)
        self.assertEqual(projection.iloc[0]["Year"], 2026)
        self.assertGreater(projection.iloc[-1]["Revenue"], projection.iloc[0]["Revenue"])
        self.assertAlmostEqual(projection.iloc[-1]["EBITDA_Margin"], 0.22)

    def test_lbo_returns_are_positive_and_reconcile(self):
        lbo, summary = build_lbo_model(self.historical, self.assumptions)

        self.assertGreater(summary.entry_ev, 0)
        self.assertAlmostEqual(summary.sponsor_equity, summary.entry_ev - summary.opening_debt)
        self.assertGreater(summary.moic, 0)
        self.assertEqual(len(lbo), self.assumptions.hold_years)

    def test_missing_columns_raise_validation_error(self):
        with self.assertRaises(ValueError):
            normalize_historical_financials(pd.DataFrame({"Revenue": [100]}))

    def test_comps_valuation_adds_implied_ev(self):
        comps = pd.DataFrame({"Company": ["A"], "EV_EBITDA": [9.0]})
        result = build_comps_valuation(comps, entry_ebitda=20.0)

        self.assertEqual(result.iloc[0]["Implied_EV"], 180.0)

    def test_sensitivity_returns_matrix(self):
        sensitivity = build_returns_sensitivity(self.historical, self.assumptions)

        self.assertEqual(len(sensitivity), 5)
        self.assertIn("Revenue Growth", sensitivity.columns)

    def test_memo_contains_recommendation_and_returns(self):
        _, summary = build_lbo_model(self.historical, self.assumptions)
        memo = build_investment_memo(summary, self.assumptions, "TestCo")

        self.assertIn("Investment Committee Memo", memo)
        self.assertIn("TestCo", memo)
        self.assertIn("MOIC", memo)


if __name__ == "__main__":
    unittest.main()
