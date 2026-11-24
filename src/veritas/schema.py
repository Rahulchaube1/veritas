from datetime import date
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Config(Strict):
    database: str = ":memory:"
    page_size: int = Field(4, ge=1, le=20)
    max_steps: int = Field(5, ge=1, le=10)
    corpus_limit: int = Field(10000, ge=1, le=50000)
    corpus_byte_limit: int = Field(64_000_000, ge=1000, le=1_000_000_000)
    model_dir: Path | None = None
    calibration: Path | None = None
    ranker: Path | None = None
    rationale_gate: Path | None = None
    scientific_evidence: bool = False
    diverse_candidates: bool = False

    @model_validator(mode="after")
    def validate_scientific_models(self):
        if self.scientific_evidence and (self.model_dir is None or self.rationale_gate is None):
            raise ValueError("scientific_evidence requires model_dir and rationale_gate")
        return self


class Document(Strict):
    id: str = Field(min_length=1, max_length=160)
    title: str = Field(default="", max_length=500)
    text: str = Field(min_length=1, max_length=64000)
    publisher: str = Field(default="", max_length=200)
    author: str = Field(default="", max_length=200)
    url: str | None = Field(default=None, max_length=2048)
    cites: list[str] = Field(default_factory=list, max_length=32)
    published: date | None = None
    modified: date | None = None
    event_date: date | None = None
    valid_from: date | None = None
    valid_to: date | None = None

    @model_validator(mode="before")
    @classmethod
    def supplied_html(cls, data):
        if isinstance(data, dict) and "html" in data:
            from .extraction import extract_html

            if "text" in data or not isinstance(data["html"], str):
                raise ValueError("Supply either HTML or plain text")
            data = dict(data)
            data["text"] = extract_html(data.pop("html"))
        return data

    @field_validator("url")
    @classmethod
    def web_url(cls, value):
        if value:
            p = urlsplit(value)
            if p.scheme not in {"https", "http"} or not p.hostname or p.username or p.password:
                raise ValueError("Expected HTTP(S) citation URL")
        return value

    @model_validator(mode="after")
    def intervals(self):
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("Reversed validity interval")
        if any(not x or len(x) > 512 for x in self.cites):
            raise ValueError("Invalid citation identifier")
        return self


class Query(Strict):
    query: str = Field(min_length=1, max_length=1600)
    max_search_steps: int = Field(3, ge=1, le=10)
    as_of: date | None = None
    follow_links: bool = False


class Hit(Strict):
    doc_id: str
    score: float
    features: dict[str, float]


Stance = Literal["support", "contradict", "neutral"]
Verdict = Literal["supported", "refuted", "insufficient", "conflict"]


class PassageRef(Strict):
    doc_id: str
    start: int
    end: int


class Evidence(Strict):
    doc_id: str
    title: str
    publisher: str
    url: str | None
    passage: str
    start: int
    end: int
    stance: Stance
    strength: float = Field(ge=0, le=1)
    method: str
    temporal: str
    group: str
    rank_features: dict[str, float]
    premises: list[PassageRef] = Field(default_factory=list)


class Edge(Strict):
    source: str
    target: str
    relation: str
    basis: str
    grouped: bool


class Claim(Strict):
    id: str
    text: str
    verdict: Verdict
    citations: list[int]


class Result(Strict):
    query: str
    answer: str
    claims: list[Claim]
    evidence: list[Evidence]
    source_dependencies: list[Edge]
    contradictions: list[int]
    temporal_constraints: dict[str, str | None]
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_target: str
    evidence_balance: float
    abstained: bool
    search_budget_used: int
    documents_examined: int
    latency_ms: float
    actions: list[str]
