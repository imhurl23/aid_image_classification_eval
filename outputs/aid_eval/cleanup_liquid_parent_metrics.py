"""One-time cleanup: retain cost metrics only on model inference spans."""

import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import braintrust as bt
from repair_liquid_costs import PROJECT_ID
from web_aid_eval import api, keys_load

keys = keys_load()
bt.login(api_key=keys["BRAINTRUST_API_KEY"])
exps = api(keys, "GET", "/v1/experiment", params={"project_id": PROJECT_ID, "limit": 100})[
    "objects"
]
fields = [
    "estimated_cost",
    "recorded_cost_usd",
    "provider_cost_available",
    "provider_cost_usd",
    "cost_discrepancy",
    "recorded_to_provider_cost_ratio",
]
for e in exps:
    if "d1" not in e["name"]:
        continue
    w = bt.init(
        project="aid-single-label-comparison", experiment=e["name"], update=True, set_current=False
    )
    last = ""
    n = 0
    while True:
        q = f"select id, metrics from experiment('{e['id']}') where id > '{last}' and span_attributes.type != 'llm' and metrics.provider_cost_available is not null order by id asc limit 1000"
        for attempt in range(5):
            try:
                rows = api(keys, "POST", "/btql", json={"query": q})["data"]
                break
            except RuntimeError as err:
                if "429" not in str(err):
                    raise
                time.sleep(35)
        if rows:
            events = [
                {
                    "id": r["id"],
                    "_is_merge": True,
                    "_merge_paths": [["metrics"]],
                    "metrics": {
                        k: v for k, v in (r.get("metrics") or {}).items() if k not in fields
                    },
                }
                for r in rows
            ]
            api(keys, "POST", f"/v1/experiment/{e['id']}/insert", json={"events": events})
        n += len(rows)
        if rows:
            last = rows[-1]["id"]
        time.sleep(3.1)
        if len(rows) < 1000:
            break
    w.flush()
    print(e["name"], n, flush=True)
