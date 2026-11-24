import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from worlds import generate, world

from veritas import Config, Veritas


def correct(r, case):
    return (r.claims[0].verdict == "supported") == case["world_truth"]


def report(output_root: Path):
    for directory in ["datasets", "models", "results"]:
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    generate(output_root / "datasets/veritas_bench.jsonl.gz")
    variants = {
        "full": {},
        "naive_votes": {"dependency": False},
        "no_time": {"use_time": False},
        "no_conflict": {"conflict": False},
        "fixed_budget": {"adaptive": False},
        "no_final_check": {"verify": False},
        "bm25_verified": {"mode": "bm25"},
        "dense_verified": {"mode": "dense"},
    }
    calx = []
    caly = []
    for i in range(720, 1080):
        case = world(i)
        engine = Veritas()
        engine.add(case["documents"])
        r = engine.search(case["query"], as_of=case["as_of"])
        if not r.abstained and case["world_truth"] is not None:
            calx.append(abs(r.evidence_balance))
            caly.append(correct(r, case))
        engine.close()
    model = IsotonicRegression(out_of_bounds="clip").fit(calx, caly)
    path = output_root / "models/synthetic_calibration_v2.json"
    path.write_text(
        json.dumps(
            dict(
                x=model.X_thresholds_.tolist(),
                y=model.y_thresholds_.tolist(),
                n=len(calx),
                target="Agreement with latent synthetic world truth, conditional on a decisive answer; controlled benchmark only",
            ),
            indent=2,
        )
        + "\n"
    )
    agg = defaultdict(list)
    by_family = defaultdict(list)
    calibration = []
    pairs = Counter()
    timings = []
    typed_joins = []
    for i in range(1080, 1440):
        case = world(i)
        engine = Veritas(Config(calibration=path))
        engine.add(case["documents"])
        for name, opts in variants.items():
            r = engine.search(case["query"], as_of=case["as_of"], **opts)
            ok = r.claims[0].verdict == case["evidence_verdict"]
            agg[name].append([ok, not r.abstained, r.search_budget_used])
            if name == "full":
                by_family[case["family"]].append(ok)
                timings.append(r.latency_ms)
                if not r.abstained and case["world_truth"] is not None:
                    calibration.append((r.confidence, correct(r, case), abs(r.evidence_balance)))
                group = {e.doc_id: e.group for e in r.evidence}
                for j, a in enumerate(case["documents"]):
                    for k, b in enumerate(case["documents"][:j]):
                        observed = a["id"] in group and b["id"] in group and group[a["id"]] == group[b["id"]]
                        gold = case["groups"][j] == case["groups"][k]
                        pairs[str((observed, gold))] += 1
        if case["family"] == "two_hop":
            typed_query = case["query"].replace("The depot managed by", "The station managed by")
            result = engine.search(typed_query)
            decision = result.claims[0]
            typed_joins.append(
                decision.verdict == "supported"
                and len({result.evidence[n].doc_id for n in decision.citations}) == 2
            )
        engine.close()
    p, y, raw = np.array(calibration).T

    def ece(pred):
        bins = np.minimum(9, (pred * 10).astype(int))
        return float(
            sum(np.mean(bins == b) * abs(pred[bins == b].mean() - y[bins == b].mean()) for b in set(bins))
        )

    out = {
        "worlds": 360,
        "type_corrected_two_hop": {
            "total": len(typed_joins),
            "correct_with_both_documents": sum(typed_joins),
        },
        "families": {k: np.mean(v) for k, v in by_family.items()},
        "ablations": {
            k: dict(zip(["verdict_accuracy", "coverage", "mean_steps"], np.mean(v, axis=0)))
            for k, v in agg.items()
        },
        "calibration": {
            "answered_with_known_truth": len(y),
            "brier": np.mean((p - y) ** 2),
            "ece10": ece(p),
            "uncalibrated_magnitude_brier_diagnostic": np.mean((raw - y) ** 2),
            "risk_coverage": [
                {
                    "threshold": float(t),
                    "coverage": float(sum(p >= t) / 360),
                    "risk": float(1 - y[p >= t].mean()),
                }
                for t in sorted(set(p))
            ],
        },
        "dependency_pair_counts": dict(pairs),
        "cpu_ms_p50_p95_p99": np.percentile(timings, [50, 95, 99]).tolist(),
    }
    inject = []
    for n in [1, 2, 4, 8, 16]:
        e = Veritas(Config(page_size=20))
        case = world(0)
        e.add([case["documents"][0]] + [dict(case["documents"][1], id=f"copy{j}") for j in range(n)])
        for dep in [True, False]:
            r = e.search(case["query"], dependency=dep)
            inject.append(
                dict(copies=n, dependency=dep, balance=r.evidence_balance, verdict=r.claims[0].verdict)
            )
        e.close()
    out["duplicate_injection"] = inject
    (output_root / "results/controlled_v2.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("."))
    report(parser.parse_args().output_root)
