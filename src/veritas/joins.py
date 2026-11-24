"""Resolve a definite facility description using eligible, cited relation premises."""

import re
from dataclasses import dataclass
from datetime import date

from .quantities import KINDS, NAME, UNITS, Site, parse_count, parse_site, statement
from .schema import Document, Evidence, Hit, PassageRef
from .verification import DIRECTIVE, passages, temporal

RELATIONS = {"managed": "manages", "operated": "operates", "owned": "owns"}
DESCRIPTION = re.compile(
    rf"the (\w+) (managed|operated|owned) by ({NAME}) has ([0-9]{{1,12}}) ([\w ]{{1,30}})", re.I
)
LINK = re.compile(
    rf"({NAME}) (manages|operates|owns|does not manage|does not operate|does not own) (.+)", re.I
)


@dataclass(frozen=True)
class Request:
    kind: str
    relation: str
    owner: str
    number: int
    unit: str


def parse_request(claim: str) -> Request | None:
    m = DESCRIPTION.fullmatch(statement(claim))
    if not m:
        return None
    kind = m[1].casefold()
    unit = UNITS.get(m[5].casefold())
    if (kind not in KINDS and kind != "facility") or not unit:
        return None
    return Request(KINDS.get(kind, kind), RELATIONS[m[2].casefold()], m[3].casefold(), int(m[4]), unit)


def infer(
    request: Request, docs: list[Document], hits: dict[str, Hit], when: date, scoped: bool, use_time: bool
) -> list[Evidence]:
    bindings: dict[Site, list[Evidence]] = {}
    excluded = set()
    measurements = []
    for doc in docs:
        scope = temporal(doc, when, scoped) if use_time else "unspecified"
        if scope not in {"in_interval", "unspecified"}:
            continue
        for text, start, end in passages(doc):
            if DIRECTIVE.search(text):
                continue
            link = LINK.fullmatch(statement(text))
            site = None
            count = None
            if link and link[1].casefold() == request.owner:
                site = parse_site(link[3])
                if not site or request.kind not in {"facility", site.kind}:
                    continue
                relation = link[2].casefold()
                if relation.startswith("does not "):
                    if relation.removeprefix("does not ") + "s" == request.relation:
                        excluded.add(site)
                    continue
                if relation != request.relation:
                    continue
            else:
                count = parse_count(text)
                if not count or count.unit != request.unit:
                    continue
            item = Evidence(
                doc_id=doc.id,
                title=doc.title,
                publisher=doc.publisher,
                url=doc.url,
                passage=text,
                start=start,
                end=end,
                stance="neutral",
                strength=0,
                method="relation_premise",
                temporal=scope,
                group=doc.id,
                rank_features=hits[doc.id].features,
            )
            if site:
                bindings.setdefault(site, []).append(item)
            elif count:
                measurements.append((count, item))
    # A definite description must identify just one eligible entity. A disputed binding is unusable.
    if len(bindings) != 1:
        return []
    site, premises = next(iter(bindings.items()))
    if site in excluded:
        return []
    # One witness is sufficient. Duplicating relation witnesses must not multiply the property vote.
    witness = min(premises, key=lambda e: (e.doc_id, e.start))
    ref = PassageRef(doc_id=witness.doc_id, start=witness.start, end=witness.end)
    conclusions = []
    for count, item in measurements:
        if count.site == site:
            item.stance = "support" if count.number == request.number else "contradict"
            item.strength = 1
            item.method = "relational_count"
            item.premises = [ref]
            conclusions.append(item)
    return [witness, *conclusions] if conclusions else []
