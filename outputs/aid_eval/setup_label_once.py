"""Replace repetitive model review assignments with one authoritative image review."""

import json

from labeling_guide import GUIDE
from native_review import review_image
from publish_model_review_rows import P, model_review_rows
from sync_image_labels import label_of
from web_aid_eval import OUT, PROJECT, REVIEW_FIELD, api, keys_load, review_rows

USER = "REVIEWER_USER_ID"
NAME = "Web images — label once"


def main():
    k = keys_load()
    masters = review_rows(k, P)
    models = model_review_rows(k)
    assert len(masters) == 1000 and len(models) == 5000
    by = {}
    for r in models:
        l = label_of(r)
        if l:
            by.setdefault(r["metadata"]["id"], set()).add(l)
    events = []
    conflicts = {}
    imports = 0
    for r in masters:
        case = r["metadata"]["id"]
        labels = by.get(case, set())
        current = label_of(r)
        if len(labels) > 1:
            conflicts[case] = sorted(labels)
        e = {
            "id": r["id"],
            "_is_merge": True,
            "_merge_paths": [["input"], ["output"]],
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Label this image once in Review → web_scene_category. Use the category guide in that review field. Choose Ambiguous rather than forcing a guess."
                            + (
                                " Previous labels disagreed; please reassess this image."
                                if len(labels) > 1
                                else ""
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": review_image(r["input"])}},
                    ],
                }
            ],
            "output": None,
            "span_attributes": {
                "name": "Label image once" + (" — needs another look" if len(labels) > 1 else "")
            },
            "metadata": {
                "~__bt_assignments": [USER],
                "~__bt_review_lists": {"__bt_default_review_list": {"status": "PENDING"}},
                "canonical_human_label": True,
                "label_rubric_version": 1,
                "prior_model_row_labels": sorted(labels),
                "needs_adjudication": len(labels) > 1,
                "superseded_review_presentation": False,
            },
            "tags": ["web-aid-v1", "label-once"]
            + (["needs-adjudication"] if len(labels) > 1 else []),
        }
        if current is None and len(labels) == 1:
            e["expected"] = {REVIEW_FIELD: next(iter(labels))}
            e["metadata"]["label_import_source"] = "Unanimous prior human model-row labels"
            imports += 1
        events.append(e)
    for r in models:
        m = r["metadata"]
        events.append(
            {
                "id": r["id"],
                "_is_merge": True,
                "_merge_paths": [["metadata", "~__bt_review_lists"]],
                "metadata": {
                    "~__bt_assignments": [u for u in m.get("~__bt_assignments", []) if u != USER],
                    "~__bt_review_lists": None,
                    "prior_review_queue_state": m.get("~__bt_review_lists"),
                    "label_from_image_review": True,
                },
            }
        )
    for i in range(0, len(events), 200):
        api(k, "POST", f"/v1/project_logs/{P}/insert", json={"events": events[i : i + 200]})
    scores = api(k, "GET", "/v1/project_score", params={"project_id": P})["objects"]
    score = next(s for s in scores if s["name"] == REVIEW_FIELD)
    config = score["config"]
    config["condition"] = {
        "when": {
            "clauses": ["metadata.batch = 'web-aid-v1' AND metadata.record_type = 'human_review'"]
        },
        "behavior": "hidden",
    }
    api(
        k,
        "PATCH",
        "/v1/project_score/" + score["id"],
        json={"config": config, "description": GUIDE},
    )
    views = api(
        k,
        "GET",
        "/v1/view",
        params={"object_type": "project", "object_id": P, "view_type": "for_review_project_log"},
    )["objects"]
    existing = next((v for v in views if v["name"] == NAME), None)
    body = {
        "object_type": "project",
        "object_id": P,
        "view_type": "for_review_project_log",
        "name": NAME,
        "description": "One image to label once. Labels propagate to all five model results. Model answers are hidden here.",
        "options": {
            "timeRangeFilter": "90d",
            "queryShape": "spans",
            "layout": "list",
            "rowHeight": "compact",
        },
        "view_data": {"search": {}},
    }
    api(k, "PATCH", "/v1/view/" + existing["id"], json=body) if existing else api(
        k, "POST", "/v1/view", json=body
    )
    # Old bookmarks also open the same deduplicated assignment queue.
    report = {
        "unique_review_images": 1000,
        "model_rows_removed_from_queue": 5000,
        "prior_labels_imported": imports,
        "conflicts_to_relabel": conflicts,
        "rubric_version": 1,
        "url": "https://www.braintrust.dev/app/eval_engineering/p/"
        + PROJECT
        + "/review?v=Web%20images%20%E2%80%94%20label%20once",
    }
    (OUT / "label-once-setup.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
