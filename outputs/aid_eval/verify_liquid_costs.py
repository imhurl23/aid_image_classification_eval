"""Verify persisted Liquid output costs using read-only BTQL aggregates."""

import json
import time
from pathlib import Path

from repair_liquid_costs import PROJECT_ID
from web_aid_eval import api, keys_load

keys = keys_load()
experiments = api(keys, "GET", "/v1/experiment", params={"project_id": PROJECT_ID, "limit": 100})[
    "objects"
]
sources = {x["name"]: "experiment('" + x["id"] + "')" for x in experiments if "d1" in x["name"]}
sources["project_logs"] = "project_logs('" + PROJECT_ID + "')"
report = {}
for name, source in sources.items():
    q = f"select output.requested_model as model, count(*) as n, count(output.recorded_to_provider_cost_ratio) as verified_n, min(output.recorded_to_provider_cost_ratio) as min_ratio, max(output.recorded_to_provider_cost_ratio) as max_ratio, sum(output.cost_discrepancy) as discrepancies from {source} where output.requested_model in ('d1', 'd1-ultra') and output.usage.cost is not null group by output.requested_model"
    for attempt in range(5):
        try:
            rows = api(keys, "POST", "/btql", json={"query": q})["data"]
            break
        except RuntimeError as e:
            if "429" not in str(e) or attempt == 4:
                raise
            time.sleep(35)
    assert rows, (name, rows)
    for row in rows:
        assert (
            row["n"] == row["verified_n"]
            and row["min_ratio"] == row["max_ratio"] == 1
            and row["discrepancies"] == 0
        ), (name, row)
    report[name] = rows
    print(name, rows, flush=True)
    time.sleep(3.1)
Path(__file__).with_name("liquid-cost-verification.json").write_text(json.dumps(report, indent=2))
