"""Train a passage filter on official train. Validation chooses the acceptance threshold."""

import hashlib
import math
from collections import Counter

import numpy as np
from scifact import corpus, rows, write
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from veritas import Veritas
from veritas.grounding import signals
from veritas.retrieval import tokens


def main():
    docs = corpus()
    by = {d.id: d for d in docs}
    train, dev = rows("claims_train"), rows("claims_dev")
    held = {str(i) for r in dev for i in r["cited_doc_ids"]}
    held_text = {by[i].text for i in held}
    pool = [d for d in docs if d.text not in held_text]
    heldq = {r["claim"].lower().strip() for r in dev}
    df = Counter()
    for d in pool:
        df.update(set(tokens(d.text)))
    idf = {w: round(math.log1p(len(pool) / (n + 1)), 4) for w, n in df.items() if n >= 2}

    def split(d):
        return "fit" if int(hashlib.sha256(d.text.encode()).hexdigest()[:8], 16) % 5 < 4 else "validation"

    batches = {}
    for name in ["fit", "validation"]:
        eng = Veritas()
        eng.add([d for d in pool if split(d) == name])
        X, y = [], []
        for r in train:
            gold = r["evidence"]
            if not gold or r["claim"].lower().strip() in heldq or not set(gold) <= eng.docs.keys():
                continue
            ids = set(gold) | {h.doc_id for h in eng.retrieve(r["claim"], 3, "bm25")}
            for i in sorted(ids):
                gold_indices = {n for a in gold.get(i, []) for n in a["sentences"]}
                for j, s in enumerate(by[i].text.splitlines()):
                    X.append(signals(r["claim"], s, by[i].title, idf))
                    y.append(j in gold_indices)
        eng.close()
        batches[name] = (X, np.array(y))
    model = LogisticRegression(max_iter=500, random_state=17).fit(*batches["fit"])
    X, y = batches["validation"]
    p = model.predict_proba(X)[:, 1]
    # Fixed rule: maximize validation F0.5 for annotated passage identification.
    choices = []
    for t in np.arange(0.05, 0.96, 0.025):
        pick = p >= t
        tp = sum(y[pick])
        precision = tp / max(1, sum(pick))
        recall = tp / max(1, sum(y))
        f = 1.25 * precision * recall / max(1e-9, 0.25 * precision + recall)
        choices.append((f, t, precision, recall))
    f, t, prec, rec = max(choices)
    write(
        "models/rationale_gate.json",
        dict(idf=idf, coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0]), threshold=t),
    )
    write(
        "results/rationale_training.json",
        dict(
            fit_pairs=len(batches["fit"][1]),
            validation_pairs=len(y),
            validation_average_precision=average_precision_score(y, p),
            threshold=t,
            validation_precision=prec,
            validation_recall=rec,
        ),
    )


if __name__ == "__main__":
    main()
