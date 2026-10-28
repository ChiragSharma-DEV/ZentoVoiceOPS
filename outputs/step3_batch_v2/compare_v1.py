"""Read-only comparison of outputs/step3_batch_v2 against v1 and its replicates."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRS = ["step3_batch", "step3_batch_rep1", "step3_batch_rep2", "step3_batch_v2"]
FIELDS = ["decision_correct", "must_include_pass", "must_not_pass", "tool_calls_correct"]


def load(name):
    out = {}
    with (ROOT / "outputs" / name / "transcripts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            out[(rec["scenario"], rec["turn"])] = rec
    return out


runs = {name: load(name) for name in DIRS}
v1 = runs["step3_batch"]
v2 = runs["step3_batch_v2"]

print("TOTALS")
for name in DIRS:
    recs = runs[name].values()
    print(
        name,
        "pass", sum(1 for r in recs if r["turn_pass"]),
        "fail", sum(1 for r in recs if not r["turn_pass"]),
        {f: sum(1 for r in recs if not r[f]) for f in FIELDS},
        "api", sum(r["api_calls"] for r in recs),
        "exceptions", sum(1 for r in recs if r["error"]),
    )

print("\nPASS/FAIL DIFFS vs v1")
for name in ["step3_batch_rep1", "step3_batch_rep2", "step3_batch_v2"]:
    diffs = [
        (k, v1[k]["turn_pass"], runs[name][k]["turn_pass"])
        for k in v1
        if v1[k]["turn_pass"] != runs[name][k]["turn_pass"]
    ]
    print(name, diffs)

print("\nCHANGED TURNS v1 -> v2 (with replies)")
for key in sorted(v1):
    if v1[key]["turn_pass"] == v2[key]["turn_pass"]:
        continue
    a = v1[key]
    b = v2[key]
    print(f"{key[0]} T{key[1]} v1_pass={a['turn_pass']} v2_pass={b['turn_pass']}")
    print("  input:", a["input"])
    print("  v1:", a["decision"], "|", a["reply"])
    print("  v2:", b["decision"], "|", b["reply"])

print("\nNEW ESCALATIONS IN v2")
for key in sorted(v1):
    if v2[key]["decision"] == "escalate" and v1[key]["decision"] != "escalate":
        print(key, "v1=", v1[key]["decision"], "sections=", v2[key]["used_sections"])
        print("  input:", v1[key]["input"])
        print("  v2:", v2[key]["reply"])

print("\nNEW POLICY-ROUTE TURNS IN v2 (api_calls >= 1)")
for key in sorted(v1):
    if v2[key]["api_calls"] >= 1 and v1[key]["api_calls"] < 1:
        print(key, "v1=", v1[key]["decision"], "v2=", v2[key]["decision"])
        print("  input:", v1[key]["input"])
        print("  sections:", v2[key]["used_sections"])

print("\nV2 FAILING TURNS")
for key in sorted(v1):
    b = v2[key]
    if b["turn_pass"]:
        continue
    print(
        key,
        "decision=", b["decision"],
        "must_include_pass=", b["must_include_pass"],
        "missing=", b["missing_facts"],
    )
