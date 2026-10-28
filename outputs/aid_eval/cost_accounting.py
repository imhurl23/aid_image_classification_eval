"""Provider-reported Liquid USD costs; token estimates remain explicitly separate."""

import math

REL_TOL = 1e-6
ABS_TOL = 1e-12


def reconcile(result, arm, prices):
    usage = result.get("usage") or (result.get("response") or {}).get("usage") or {}
    estimate = usage.get("input_tokens", 0) * prices[arm] / 1e6
    provider = usage.get("cost") if arm in ("d1", "d1-ultra") else None
    valid = (
        isinstance(provider, (int, float))
        and not isinstance(provider, bool)
        and math.isfinite(provider)
        and provider >= 0
    )
    cost = provider if valid else estimate
    old = result.get("estimated_api_cost_usd")
    if old is not None and not math.isclose(old, cost, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        result.setdefault("cost_before_reconciliation_usd", old)
    result.update(
        estimated_api_cost_usd=cost,
        recorded_cost_usd=cost,
        token_estimated_cost_usd=estimate,
        cost_source="provider_usage.cost" if valid else "token_rate_estimate",
        cost_accounting_version=1,
    )
    metrics = {
        "estimated_cost": cost,
        "recorded_cost_usd": cost,
        "provider_cost_available": int(valid),
    }
    # Reconciliation can run again after corrected usage is loaded. Do not leave
    # stale provider-only fields on records that now fall back to token rates.
    for field in ("provider_cost_usd", "recorded_to_provider_cost_ratio", "cost_discrepancy"):
        result.pop(field, None)
    if valid:
        # Both zero is an exact match; a nonzero / zero ratio is undefined.
        ratio = cost / provider if provider else (1.0 if cost == 0 else None)
        discrepancy = int(not math.isclose(cost, provider, rel_tol=REL_TOL, abs_tol=ABS_TOL))
        result.update(
            provider_cost_usd=provider,
            recorded_to_provider_cost_ratio=ratio,
            cost_discrepancy=discrepancy,
        )
        metrics.update(provider_cost_usd=provider, cost_discrepancy=discrepancy)
        if ratio is not None:
            metrics["recorded_to_provider_cost_ratio"] = ratio
    return metrics
