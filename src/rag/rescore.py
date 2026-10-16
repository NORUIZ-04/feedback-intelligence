"""Re-score saved v2 runs with the tolerant refusal check. No model is run.

Run from the repository root:  python -m src.rag.rescore
"""

import json
from pathlib import Path

from src.rag.scoring import is_abstain, is_exact_abstain
from src.utils.results import save_metrics

RUNS = {"Qwen 1.5B": "rag_v2_1p5b", "Phi-3.5-mini 4-bit": "rag_v2_phi35_4bit"}


def load(stem: str) -> list[dict]:
    path = Path(f"results/{stem}.json")
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]["rows"]


def is_una(row: dict) -> bool:
    return row["group"].startswith("unanswerable")


def rescore(row: dict) -> dict:
    new = dict(row)
    abstained = is_abstain(row["answer"])
    new["abstained"] = abstained
    if is_una(row):
        new["hallucinated"] = not abstained
    else:
        new["correct"] = row["overlap"] >= 0.6 and not abstained
    return new


def main() -> None:
    summary, changed_all = {}, {}
    for label, stem in RUNS.items():
        rows = load(stem)
        mismatches = sum(
            1
            for r in rows
            if not is_una(r)
            and r["correct"] != (r["overlap"] >= 0.6 and not is_exact_abstain(r["answer"]))
        )
        new_rows = [rescore(r) for r in rows]
        ans = [r for r in rows if not is_una(r)]
        una = [r for r in rows if is_una(r)]
        new_ans = [r for r in new_rows if not is_una(r)]
        new_una = [r for r in new_rows if is_una(r)]
        groups = {}
        for g in sorted({r["group"] for r in una}):
            old_g = [r for r in una if r["group"] == g]
            new_g = [r for r in new_una if r["group"] == g]
            groups[g.replace("unanswerable_", "")] = (
                len(old_g),
                sum(1 for r in old_g if r["abstained"]),
                sum(1 for r in new_g if r["abstained"]),
            )
        summary[label] = {
            "sanity_mismatches": mismatches,
            "correct": (sum(1 for r in ans if r["correct"]), sum(1 for r in new_ans if r["correct"])),
            "wrong_abstentions": (
                sum(1 for r in ans if r["abstained"]),
                sum(1 for r in new_ans if r["abstained"]),
            ),
            "refused": (
                sum(1 for r in una if r["abstained"]),
                sum(1 for r in new_una if r["abstained"]),
            ),
            "n_answerable": len(ans),
            "n_unanswerable": len(una),
            "groups": groups,
        }
        changed_all[label] = [
            {"id": o["id"], "group": o["group"], "was": o["abstained"], "now": n["abstained"],
             "answer": o["answer"]}
            for o, n in zip(rows, new_rows)
            if o["abstained"] != n["abstained"]
        ]

    print("\nSanity check (stored 'correct' vs recomputed with the OLD rule; should be 0):")
    for label, s in summary.items():
        print(f"  {label}: {s['sanity_mismatches']} mismatches")

    print("\n| Run | Correct old | Correct new | Wrong abstentions old | Wrong abstentions new | Refused old | Refused new |")
    print("|---|---|---|---|---|---|---|")
    for label, s in summary.items():
        print(
            f"| {label} | {s['correct'][0]}/{s['n_answerable']} | {s['correct'][1]}/{s['n_answerable']} | "
            f"{s['wrong_abstentions'][0]} | {s['wrong_abstentions'][1]} | "
            f"{s['refused'][0]}/{s['n_unanswerable']} | {s['refused'][1]}/{s['n_unanswerable']} |"
        )

    print("\nUnanswerable refused by group (old -> new, of n)")
    print("| Group | n | Qwen 1.5B | Phi-3.5-mini 4-bit |")
    print("|---|---|---|---|")
    names = sorted(summary["Qwen 1.5B"]["groups"])
    for g in names:
        q = summary["Qwen 1.5B"]["groups"][g]
        p = summary["Phi-3.5-mini 4-bit"]["groups"][g]
        print(f"| {g} | {q[0]} | {q[1]} -> {q[2]} | {p[1]} -> {p[2]} |")

    print("\n--- Every answer whose refusal status changed (check these by eye) ---")
    for label, items in changed_all.items():
        print(f"\n{label}: {len(items)} changed")
        for c in items:
            print(f"  {c['id']} [{c['group']}] {c['was']} -> {c['now']} | {c['answer']!r}")

    save_metrics("rag_v2_rescored", {"summary": summary, "changed": changed_all})


if __name__ == "__main__":
    main()