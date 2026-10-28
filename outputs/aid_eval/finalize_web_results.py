"""Normalize explicit provider refusals in audit/logs, retaining raw responses and billed usage."""

import json

import braintrust as bt
import web_aid_eval as w


def main():
    keys = w.keys_load()
    bt.login(api_key=keys["BRAINTRUST_API_KEY"])
    logger = bt.init_logger(project=w.PROJECT)
    rows = list(map(json.loads, (w.OUT / "predictions.jsonl").read_text().splitlines()))
    assert len(rows) == 5000
    changed = []
    for r in rows:
        response = r.get("response", {})
        if r["model"] == "luna" and any(
            a.get("type") == "refusal" for a in response.get("answers", [])
        ):
            r.update(
                status="refusal",
                error_type="ProviderRefusal",
                usage=response.get("usage", {}),
                served_model=response.get("model"),
            )
            r["estimated_api_cost_usd"] = (
                r["usage"].get("input_tokens", 0) * w.native.PRICES["luna"] / 1e6
            )
            changed.append(r)
    lookup = {(r["id"], r["model"]): r for r in rows}
    for r in changed:
        outputs = {
            m: {
                k: v
                for k, v in lookup[r["id"], m].items()
                if k not in ("response", "id", "review_log_id", "inference_log_id")
            }
            for m in w.native.MODELS
        }
        logger.update_span(
            id=r["inference_log_id"],
            output=outputs,
            metadata={"provider_refusals": ["luna"]},
            scores={"valid_output": sum(o["status"] == "ok" for o in outputs.values()) / 5},
        )
    # Update the corresponding provider spans; raw response stays intact.
    if changed:
        q = f"select id, output from project_logs('{logger.id}') where span_attributes.name = 'luna' AND output.status = 'invalid_output' limit 1000"
        childrows = w.api(keys, "POST", "/btql", json={"query": q})["data"]
        bysha = {r["input_sha256"]: r for r in changed}
        for child in childrows:
            original = child.get("output") or {}
            r = bysha.get(original.get("input_sha256"))
            if r and any(
                a.get("type") == "refusal" for a in original.get("response", {}).get("answers", [])
            ):
                logger.update_span(
                    id=child["id"],
                    output={
                        **original,
                        "status": "refusal",
                        "usage": r["usage"],
                        "estimated_api_cost_usd": r["estimated_api_cost_usd"],
                        "error_type": "ProviderRefusal",
                    },
                    error="Provider returned refusal instead of scene category",
                    metrics={
                        "prompt_tokens": r["usage"].get("input_tokens", 0),
                        "tokens": r["usage"].get("total_tokens", 0),
                        "estimated_cost": r["estimated_api_cost_usd"],
                    },
                )
    bt.flush()
    (w.OUT / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = json.loads((w.OUT / "summary.json").read_text())
    summary["refusals"] = {
        m: sum(r["model"] == m and r["status"] == "refusal" for r in rows) for m in w.native.MODELS
    }
    summary["estimated_api_cost_usd"] = sum(r["estimated_api_cost_usd"] for r in rows)
    (w.OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Recorded explicit provider refusals:", len(changed))


if __name__ == "__main__":
    main()
