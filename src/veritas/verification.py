import re
from collections import defaultdict
from datetime import date
from decimal import Decimal

from scipy.special import softmax

from .quantities import parse_count
from .retrieval import tokens
from .schema import Document, Evidence, Stance

TIME = re.compile(r"\b(?:as of|on date)\s+(\d{4}-\d{2}-\d{2})\b", re.I)
CURRENT = re.compile(r"\b(currently|current|latest|today|now)\b", re.I)
QUESTION = re.compile(r"^(?:who|what|which|when|where|why|how|does|do|is|are|was|were|can)\b", re.I)
DIRECTIVE = re.compile(
    r"(?:ignore (?:all |the |previous |prior |your )*(?:instructions|rules)|system prompt|respond (?:only )?with|you (?:must|should) (?:answer|output)|<script\b)",
    re.I,
)
NUMBER = re.compile(r"(?<!\w)[-+]?\d+(?:[.,]\d+)*(?!\w)")
NEG = re.compile(r"\b(?:not|never)\b", re.I)


def temporal(doc: Document, when: date, scoped: bool) -> str:
    if (doc.published and doc.published > when) or (doc.modified and doc.modified > when):
        return "unavailable"
    if (doc.valid_from and when < doc.valid_from) or (doc.valid_to and when > doc.valid_to):
        return "outside_interval"
    if doc.valid_from or doc.valid_to:
        return "in_interval"
    return "unknown_scope" if scoped else "unspecified"


def passages(doc: Document):
    for m in re.finditer(r"[^\n]+?(?:[.!?](?=\s|$)|(?=\n|$))", doc.text):
        text = m.group().strip()
        if text:
            start = m.start() + len(m.group()) - len(m.group().lstrip())
            yield text, start, start + len(text)


def norm(text):
    # Preserve scientific symbols, decimal points and inequality direction.
    return re.sub(r"\s+", " ", text.casefold()).strip().rstrip(".?!")


def stance(claim: str, passage: str) -> tuple[Stance, float, str]:
    a, b = norm(claim), norm(passage)
    if DIRECTIVE.search(passage):
        return "neutral", 0.0, "instruction_filter"
    if passage.rstrip().endswith("?") or passage.lstrip().startswith("¿"):
        return "neutral", 0.0, "question"
    if a and a == b:
        return "support", 1.0, "exact_statement"
    left, right = parse_count(claim), parse_count(passage)
    if left and right:
        if left.site != right.site or left.unit != right.unit:
            return "neutral", 0.0, "different_quantity"
        return ("support" if left.number == right.number else "contradict"), 1.0, "normalized_count"
    if NEG.search(a) or NEG.search(b):
        if bool(NEG.search(a)) != bool(NEG.search(b)) and norm(NEG.sub("", a)) == norm(NEG.sub("", b)):
            return "contradict", 1.0, "opposite_negation"
    # Same single numeric slot and identical units/context. Qualified ranges abstain.
    av, bv = NUMBER.findall(a), NUMBER.findall(b)
    if (
        len(av) == len(bv) == 1
        and not (NEG.search(a) or NEG.search(b))
        and not re.search(
            r"[<>≤≥~]|\b(at least|at most|up to|about|over|under|approximately|more than|less than|roughly|around|más de|menos de|al menos|como máximo|aproximadamente|no)\b",
            a + " " + b,
        )
        and NUMBER.sub("#", a) == NUMBER.sub("#", b)
        and av != bv
    ):
        if all(
            re.fullmatch(r"[-+]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?", v) for v in [av[0], bv[0]]
        ):
            equal = Decimal(av[0].replace(",", "")) == Decimal(bv[0].replace(",", ""))
            return ("support" if equal else "contradict"), 1.0, "numeric_slot"
    return "neutral", 0.0, "unresolved"


def analyze(claim, doc, hit, when, scoped, neural=None, use_time=True, gate=None, passage_ranker=None):
    scope = temporal(doc, when, scoped) if use_time else "unspecified"
    pieces = list(passages(doc))
    q = set(tokens(claim))
    pieces.sort(key=lambda x: len(q & set(tokens(x[0]))) / max(1, len(q | set(tokens(x[0])))), reverse=True)
    selected = pieces[: 12 if passage_ranker else 5]
    values = [stance(claim, p[0]) for p in selected]
    unknown = [i for i, x in enumerate(values) if x[2] == "unresolved" and len(selected[i][0]) <= 1600]
    passage_features = {}
    if passage_ranker and unknown:
        from .grounding import select_rationales

        unknown, passage_features = select_rationales(
            claim, doc.title, selected, unknown, gate, passage_ranker
        )
    elif gate:
        from .grounding import relevance

        unknown = [
            i for i in unknown if relevance(claim, selected[i][0], doc.title, gate) >= gate["threshold"]
        ]
    if neural and unknown:
        probs = softmax(neural.run([(selected[i][0], claim) for i in unknown]), axis=1)
        for i, row in zip(unknown, probs):
            label = neural.labels[int(row.argmax())]
            strength = float(row.max())
            if (
                label in {"entailment", "contradiction"}
                and strength >= 0.90
                and q & set(tokens(selected[i][0]))
            ):
                values[i] = ("support" if label == "entailment" else "contradict", strength, "nli")
    out: list[Evidence] = []
    # Keep both stances even if they come from the same document.
    seen = set()
    for i in sorted(range(len(values)), key=lambda j: -values[j][1]):
        label, strength, method = values[i]
        if label in seen or (label == "neutral" and out):
            continue
        seen.add(label)
        text, start, end = selected[i]
        out.append(
            Evidence(
                doc_id=doc.id,
                title=doc.title,
                publisher=doc.publisher,
                url=doc.url,
                passage=text,
                start=start,
                end=end,
                stance=label,
                strength=strength,
                method=method,
                temporal=scope,
                group=doc.id,
                rank_features={**hit.features, **passage_features.get(i, {})},
            )
        )
    return out


def summarize(evidence: list[Evidence], groups: dict[str, str], dependency=True, conflict=True):
    votes: dict[str, dict[str, float]] = defaultdict(dict)
    for e in evidence:
        e.group = groups[e.doc_id] if dependency else e.doc_id
        if e.temporal not in {"unspecified", "in_interval"} or e.stance == "neutral":
            continue
        votes[e.group][e.stance] = max(votes[e.group].get(e.stance, 0), e.strength)
    pos = sum(v.get("support", 0) for v in votes.values())
    neg = sum(v.get("contradict", 0) for v in votes.values()) if conflict else 0
    margin = (pos - neg) / (1 + pos + neg)
    if pos and neg and min(pos, neg) / max(pos, neg) >= 0.34:
        return "conflict", margin
    if pos > neg:
        return "supported", margin
    if neg > pos:
        return "refuted", margin
    return "insufficient", margin
