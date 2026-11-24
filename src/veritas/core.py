import hashlib
import json
import re
import sqlite3
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from .neural import Encoder
from .provenance import graph
from .retrieval import Index, tokens
from .schema import Claim, Config, Document, Edge, Evidence, Hit, Query, Result, Verdict
from .verification import CURRENT, DIRECTIVE, QUESTION, TIME, analyze, norm, passages, summarize, temporal


class Veritas:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        if self.config.database != ":memory:":
            Path(self.config.database).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.config.database, check_same_thread=False, timeout=10)
        self.db.execute("CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.docs = {
            d.id: d
            for (row,) in self.db.execute("SELECT payload FROM docs ORDER BY id")
            if (d := Document.model_validate_json(row))
        }
        self.index: Index | None = None
        self.neural = {}
        if self.config.model_dir:
            self.neural = {name: Encoder(self.config.model_dir / name) for name in ["dense", "nli", "rerank"]}
        self.weights = self._load(self.config.ranker)
        self.cal = self._load(self.config.calibration)
        self.gate = self._load(self.config.rationale_gate)

    @staticmethod
    def _load(path):
        return json.loads(path.read_text()) if path else None

    @classmethod
    def from_config(cls, path=None):
        if path is None:
            path = Path("configs/default.yaml")
            if not path.is_file():
                return cls(Config(database=".cache/veritas.db"))
        return cls(Config.model_validate(yaml.safe_load(Path(path).read_text()) or {}))

    def close(self):
        with self.lock:
            self.db.close()

    def add(self, documents):
        docs = [Document.model_validate(d) for d in documents]
        with self.lock:
            if len(self.docs.keys() | {d.id for d in docs}) > self.config.corpus_limit:
                raise ValueError("Corpus document limit exceeded")
            updated = {d.id: d for d in docs}
            total = sum(len(d.text.encode()) for i, d in self.docs.items() if i not in updated)
            if total + sum(len(d.text.encode()) for d in updated.values()) > self.config.corpus_byte_limit:
                raise ValueError("Corpus byte limit exceeded")
            with self.db:
                self.db.executemany(
                    "INSERT OR REPLACE INTO docs VALUES(?,?)", [(d.id, d.model_dump_json()) for d in docs]
                )
            self.docs.update({d.id: d for d in docs})
            self.index = None

    def retrieve(self, query: str, k: int = 10, mode: str = "hybrid"):
        query = Query(query=query).query
        if not 1 <= k <= 200:
            raise ValueError("k must be between 1 and 200")
        with self.lock:
            if self.index is None:
                self.index = Index(
                    sorted(self.docs.values(), key=lambda d: d.id),
                    self.neural.get("dense"),
                    self.neural.get("rerank"),
                    self.weights,
                )
            return list(self.index.search(query, k, mode))

    def search(
        self,
        query: str,
        max_search_steps: int = 3,
        as_of: date | None = None,
        follow_links: bool = False,
        **options,
    ) -> Result:
        req = Query(query=query, max_search_steps=max_search_steps, as_of=as_of, follow_links=follow_links)
        with self.lock:
            return self._search(req, **options)

    def verify_text(
        self, query: str, max_search_steps: int = 3, as_of: date | None = None, follow_links: bool = False
    ) -> Result:
        """Verify sentence-separated assertions against one shared retrieval budget."""
        req = Query(query=query, max_search_steps=max_search_steps, as_of=as_of, follow_links=follow_links)
        atoms = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])|[\n;]+", req.query) if s.strip()]
        if len(atoms) <= 1:
            return self.search(req.query, max_search_steps, as_of, follow_links)
        start = time.perf_counter()
        remaining = min(req.max_search_steps, self.config.max_steps)
        claims = []
        passages_out: list[Evidence] = []
        edges = []
        contradictions: list[int] = []
        actions = []
        bounds = set()
        with self.lock:
            for j, atom in enumerate(atoms):
                if remaining == 0:
                    claims.append(
                        Claim(
                            id=hashlib.sha256(atom.encode()).hexdigest()[:20],
                            text=atom,
                            verdict="insufficient",
                            citations=[],
                        )
                    )
                    continue
                allocation = max(1, remaining // (len(atoms) - j))
                result = self.search(atom, allocation, as_of, follow_links)
                remaining -= result.search_budget_used
                shift = len(passages_out)
                claim = result.claims[0].model_copy(
                    update={"citations": [i + shift for i in result.claims[0].citations]}
                )
                claims.append(claim)
                passages_out.extend(result.evidence)
                contradictions.extend(i + shift for i in result.contradictions)
                edges.extend(result.source_dependencies)
                actions.extend(result.actions)
                bounds.add(result.temporal_constraints.get("as_of"))
        abstained = any(c.verdict in {"insufficient", "conflict"} for c in claims)
        return Result(
            query=query,
            answer=" ".join(f"{c.verdict.capitalize()}: {c.text}." for c in claims),
            claims=claims,
            evidence=passages_out,
            source_dependencies=edges,
            contradictions=contradictions,
            temporal_constraints={"as_of": next(iter(bounds)) if len(bounds) == 1 else None},
            confidence=None,
            confidence_target="uncalibrated; inspect each assertion",
            evidence_balance=0,
            abstained=abstained,
            search_budget_used=len(actions),
            documents_examined=len({e.doc_id for e in passages_out}),
            latency_ms=(time.perf_counter() - start) * 1000,
            actions=actions,
        )

    def _search(
        self, req, mode="hybrid", dependency=True, use_time=True, conflict=True, adaptive=True, verify=True
    ):
        start = time.perf_counter()
        query = req.query
        dates = TIME.findall(query)
        when = req.as_of or (date.fromisoformat(dates[0]) if dates else date.today())
        if dates and (len(set(dates)) > 1 or req.as_of and req.as_of != date.fromisoformat(dates[0])):
            raise ValueError("Ambiguous as-of date")
        scoped = bool(req.as_of or dates or CURRENT.search(query))
        claim = TIME.sub("", query).strip(" ,").rstrip("?.")
        claim = re.sub(r"\bcurrently\b\s*", "", claim, flags=re.I).strip()
        # Controlled slot questions can be grounded by checking a full retrieved assertion.
        slot = (
            re.fullmatch(r"(?:what|who) (is|was|are|were) (.+)", claim, re.I) if query.endswith("?") else None
        )
        polar = re.fullmatch(
            r"(is|are|was|were) (.+?) (not )?(open|closed|active|available|valid)", claim, re.I
        )
        if polar:
            claim = f"{polar[2]} {polar[1].lower()} {polar[3] or ''}{polar[4]}"
        unsupported = (bool(QUESTION.match(claim)) or query.endswith("?")) and not (slot or polar)
        unsupported = unsupported or bool(DIRECTIVE.search(query)) or bool(re.search(r"[.!?]\s+\S", claim))
        hits = []
        docs: list[Document] = []
        evidence: list[Evidence] = []
        actions = []
        groups: dict[str, str] = {}
        edges: list[Edge] = []
        verdict: Verdict = "insufficient"
        margin = 0.0
        cap = min(req.max_search_steps, self.config.max_steps)
        page = self.config.page_size
        resolved = not slot
        seen = set()
        collected: dict[str, Hit] = {}
        for step in range(cap):
            if unsupported:
                break
            # Request an increasing prefix, then verify only unseen documents.
            lookup = query
            linked = []
            if req.follow_links and step % 2 == 1 and docs:
                for previous in docs:
                    for ref in previous.cites:
                        if ref in self.docs and ref not in seen and ref not in linked:
                            linked.append(ref)
                # One new retrieval round. Reference edges get first access to its page budget.
                extra = [w for w in tokens(docs[0].title) if w not in set(tokens(query))][:4]
                lookup = " ".join([query, *extra])[:1600]
            hits = self.retrieve(lookup, page * (step + 1), mode)
            if linked:
                direct = [Hit(doc_id=ref, score=0, features={"citation_hop": 1.0}) for ref in linked[:page]]
                hits = direct + [h for h in hits if h.doc_id not in linked]

            batch = [h for h in hits if h.doc_id not in seen]
            batch = batch[:page]
            collected.update({h.doc_id: h for h in batch})
            actions.append(
                f"retrieve top {page * (step + 1)}; inspect {len(batch)} new documents"
                + (f"; citation hops {len(linked[:page])}" if linked else "")
            )
            for hit in batch:
                doc = self.docs[hit.doc_id]
                docs.append(doc)
                seen.add(doc.id)
            if not resolved and slot:
                pattern = re.compile(
                    r"^"
                    + re.escape(slot[2])
                    + r"\s+"
                    + re.escape(slot[1])
                    + r"\s+[\w-]+(?:\s+[\w-]+){0,8}[.!]?$",
                    re.I,
                )
                candidates = {
                    norm(p): p
                    for d in docs
                    if temporal(d, when, scoped) in {"unspecified", "in_interval"}
                    for p, _, _ in passages(d)
                    if pattern.fullmatch(p)
                }
                if candidates:
                    claim = next(iter(candidates.values()))
                    resolved = True
            if resolved:
                done = {e.doc_id for e in evidence}
                for hit in collected.values():
                    if hit.doc_id not in done:
                        evidence.extend(
                            analyze(
                                claim,
                                self.docs[hit.doc_id],
                                hit,
                                when,
                                scoped,
                                self.neural.get("nli"),
                                use_time,
                                self.gate,
                            )
                        )
                groups, edges = graph(docs)
                verdict, margin = summarize(evidence, groups, dependency, conflict)
            pending_link = req.follow_links and any(
                ref in self.docs and ref not in seen for d in docs for ref in d.cites
            )
            if (len(hits) < page * (step + 1) and not pending_link) or len(seen) == len(self.docs):
                break
            if (
                adaptive
                and not slot
                and step >= 1
                and verdict in {"supported", "refuted"}
                and abs(margin) >= 0.65
            ):
                break
        if slot:
            # A competing value for the requested slot blocks a unique extractive answer.
            valid = (
                [
                    p
                    for d in docs
                    if temporal(d, when, scoped) in {"unspecified", "in_interval"}
                    for p, _, _ in passages(d)
                    if pattern.fullmatch(p)
                ]
                if docs
                else []
            )
            if len(set(map(norm, valid))) > 1:
                verdict = "conflict"
                for e in evidence:
                    if norm(e.passage) in set(map(norm, valid)) and norm(e.passage) != norm(claim):
                        e.stance = "contradict"
                        e.method = "competing_slot_value"
                margin = 0.0
        desired = {"supported": "support", "refuted": "contradict"}.get(verdict)
        citations = [
            i
            for i, e in enumerate(evidence)
            if desired
            and e.stance == desired
            and e.temporal in {"in_interval", "unspecified"}
            and (
                not verify
                or self.docs[e.doc_id].text[e.start : e.end] == e.passage
                and not DIRECTIVE.search(e.passage)
            )
        ]
        if desired and not citations:
            verdict = "insufficient"
        answered = verdict in {"supported", "refuted"}
        answer = (
            ("Indexed evidence supports: " if verdict == "supported" else "Indexed evidence contradicts: ")
            + claim
            if answered
            else "The indexed evidence conflicts."
            if verdict == "conflict"
            else "Insufficient eligible evidence for this query."
        )
        identifier = hashlib.sha256((claim + "\0" + str(when)).encode()).hexdigest()[:20]
        confidence = None
        if self.cal and answered:
            confidence = float(np.interp(abs(margin), self.cal["x"], self.cal["y"]))
        return Result(
            query=query,
            answer=answer,
            claims=[Claim(id=identifier, text=claim, verdict=verdict, citations=citations)],
            evidence=evidence,
            source_dependencies=edges,
            contradictions=[
                i
                for i, e in enumerate(evidence)
                if e.stance == "contradict" and e.temporal in {"unspecified", "in_interval"}
            ],
            temporal_constraints={
                "as_of": when.isoformat(),
                "scope": "explicit/current" if scoped else "unscoped",
            },
            confidence=confidence,
            confidence_target=self.cal["target"] if confidence is not None else "uncalibrated",
            evidence_balance=margin,
            abstained=not answered,
            search_budget_used=len(actions),
            documents_examined=len(docs),
            actions=actions,
            latency_ms=1000 * (time.perf_counter() - start),
        )
