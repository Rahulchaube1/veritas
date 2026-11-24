from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

from veritas import Config, Veritas

app = FastAPI(title="VERITAS evidence lab")
CLAIM = "The ferry carries 40 passengers."


@app.get("/")
def home():
    return FileResponse(Path(__file__).with_name("index.html"))


@lru_cache(maxsize=68)
def experiment(copies: int, limited: bool, diverse: bool):
    engine = Veritas(Config(page_size=2 if limited else 20, diverse_candidates=diverse))
    try:
        engine.add(
            [{"id": "register", "title": "Survey record", "text": CLAIM}]
            + [
                {"id": f"report-{i:02d}", "title": CLAIM, "text": "The ferry carries 41 passengers."}
                for i in range(1, copies + 1)
            ]
        )
        outputs = {}
        for name, grouped in [("grouped", True), ("ordinary", False)]:
            result = engine.search(CLAIM, max_search_steps=1, dependency=grouped)
            outputs[name] = {
                "verdict": result.claims[0].verdict,
                "balance": round(result.evidence_balance, 3),
                "examined": result.documents_examined,
                "evidence": [
                    {"document": e.doc_id, "text": e.passage, "group": e.group, "stance": e.stance}
                    for e in result.evidence
                ],
            }
        return outputs
    finally:
        engine.close()


@app.get("/experiment")
def run(copies: int = Query(8, ge=1, le=16), limited: bool = False, diverse: bool = False):
    return experiment(copies, limited, diverse)
