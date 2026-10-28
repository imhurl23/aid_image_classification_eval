import unittest

from cost_accounting import reconcile
from native_eval import PRICES


class Costs(unittest.TestCase):
    def test_liquid_provider_overrides_rate(self):
        for arm in ("d1", "d1-ultra"):
            r = {
                "usage": {"input_tokens": 544, "cost": 0.00001088},
                "estimated_api_cost_usd": 0.00002176,
            }
            m = reconcile(r, arm, PRICES)
            self.assertEqual(r["recorded_cost_usd"], 0.00001088)
            self.assertEqual(m["recorded_to_provider_cost_ratio"], 1)
            self.assertEqual(m["cost_discrepancy"], 0)
            self.assertEqual(r["cost_before_reconciliation_usd"], 0.00002176)
            old = r.copy()
            reconcile(r, arm, PRICES)
            self.assertEqual(r, old)

    def test_zero_is_authoritative(self):
        r = {"usage": {"input_tokens": 544, "cost": 0}}
        self.assertEqual(reconcile(r, "d1", PRICES)["estimated_cost"], 0)
        self.assertEqual(r["recorded_to_provider_cost_ratio"], 1)

    def test_missing_invalid_cost_fallback(self):
        for cost in (None, -1, float("nan"), "0.01", True):
            r = {"usage": {"input_tokens": 544, "cost": cost}}
            m = reconcile(r, "d1", PRICES)
            self.assertEqual(r["cost_source"], "token_rate_estimate")
            self.assertEqual(m["provider_cost_available"], 0)
            self.assertNotIn("recorded_to_provider_cost_ratio", r)

    def test_reconciliation_clears_stale_provider_fields(self):
        r = {"usage": {"input_tokens": 544, "cost": 0.00001088}}
        reconcile(r, "d1", PRICES)
        del r["usage"]["cost"]
        metrics = reconcile(r, "d1", PRICES)
        self.assertEqual(r["cost_source"], "token_rate_estimate")
        self.assertEqual(metrics["provider_cost_available"], 0)
        for field in ("provider_cost_usd", "recorded_to_provider_cost_ratio", "cost_discrepancy"):
            self.assertNotIn(field, r)


if __name__ == "__main__":
    unittest.main()
