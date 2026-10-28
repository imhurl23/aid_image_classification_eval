"""Idempotent cost-only repair of saved predictions and Braintrust outputs."""

import argparse
import collections
import copy
import json
import time
from pathlib import Path

import braintrust as bt
from cost_accounting import reconcile
from native_eval import CONFIG, PRICES
from web_aid_eval import api, keys_load

BASE = Path(__file__).resolve().parent
PROJECT_ID = "aa1a18b9-6a22-48a4-b10d-b7b1f79c473e"


def repair(output):
    if not isinstance(output, dict):
        return {}, False
    arm = output.get("requested_model", output.get("model"))
    if arm in ("d1", "d1-ultra") and ("usage" in output or "response" in output):
        before = copy.deepcopy(output)
        metrics = reconcile(output, arm, PRICES)
        return metrics, output != before
    changed = False
    for arm in ("d1", "d1-ultra"):
        if isinstance(output.get(arm), dict):
            _, c = repair(output[arm])
            changed |= c
    return {}, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()
    report = {
        "apply": args.apply,
        "local": {},
        "remote": {},
        "tolerance": {"relative": 1e-6, "absolute_usd": 1e-12},
    }
    for path in BASE.rglob("*.jsonl"):
        if "cost-repair" in str(path):
            continue
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        old = sum(r.get("estimated_api_cost_usd", 0) for r in rows)
        changed = sum(repair(r)[1] for r in rows)
        if changed:
            report["local"][str(path.relative_to(BASE))] = {
                "rows": changed,
                "previous_usd": old,
                "corrected_usd": sum(r.get("estimated_api_cost_usd", 0) for r in rows),
            }
            if args.apply:
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if args.remote:
        keys = keys_load()
        bt.login(api_key=keys["BRAINTRUST_API_KEY"])
        experiments = api(
            keys, "GET", "/v1/experiment", params={"project_id": PROJECT_ID, "limit": 100}
        )["objects"]
        for exp in experiments:
            if "d1" not in exp["name"]:
                continue
            reader = bt.init(
                project=CONFIG["project"], experiment=exp["name"], open=True, set_current=False
            )
            writer = (
                bt.init(
                    project=CONFIG["project"],
                    experiment=exp["name"],
                    update=True,
                    set_current=False,
                )
                if args.apply
                else None
            )
            count = 0
            for row in reader.fetch(batch_size=1000):
                output = row.get("output")
                metrics, changed = repair(output)
                if changed:
                    count += 1
                    if writer:
                        writer.update_span(
                            id=row["id"],
                            output=output,
                            **(
                                {"metrics": metrics}
                                if metrics and row.get("span_attributes", {}).get("type") == "llm"
                                else {}
                            ),
                        )
            if writer:
                writer.flush()
            report["remote"][exp["name"]] = count
            print(exp["name"], count, flush=True)
        logger = bt.init_logger(project=CONFIG["project"])
        last = ""
        count = 0
        while True:
            query = f"select id, output, span_attributes from project_logs('{PROJECT_ID}') where id > '{last}' order by id asc limit 1000"
            rows = api(keys, "POST", "/btql", json={"query": query})["data"]
            if not rows:
                break
            for row in rows:
                metrics, changed = repair(row.get("output"))
                if changed:
                    count += 1
                    if args.apply:
                        logger.update_span(
                            id=row["id"],
                            output=row["output"],
                            **(
                                {"metrics": metrics}
                                if metrics and row.get("span_attributes", {}).get("type") == "llm"
                                else {}
                            ),
                        )
            last = rows[-1]["id"]
            time.sleep(3.1)
        if args.apply:
            logger.flush()
        report["remote"]["project_logs"] = count
    path = BASE / (
        "liquid-cost-repair-applied.json" if args.apply else "liquid-cost-repair-audit.json"
    )
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
