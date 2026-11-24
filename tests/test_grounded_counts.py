import pytest
from pydantic import ValidationError

from veritas import Config, Veritas
from veritas.verification import stance


@pytest.mark.parametrize(
    "claim, source, expected",
    [
        ("The ferry does not carry 40 passengers.", "The ferry does not carry 41 passengers.", "neutral"),
        ("El hospital Mar tiene al menos 28 camas.", "El hospital Mar tiene al menos 30 camas.", "neutral"),
        ("Hospital Mar has 28 beds.", "El hospital Mar tiene 28 camas.", "support"),
        ("El hotel Costa tiene 12 habitaciones.", "Hotel Costa has 13 rooms.", "contradict"),
        ("Airport Norte has 1 runway.", "El aeropuerto Norte tiene 1 pista de aterrizaje.", "support"),
        ("Museum Azul has 14 seats.", "El museo Azul tiene 14 camas.", "neutral"),
        ("Library Río has 160 books.", "La biblioteca Rio tiene 160 libros.", "neutral"),
        ("Station Sur has 16 berths.", "La estación Sur tiene más de 16 atracaderos.", "neutral"),
        ("Station Sur has 16 berths.", "¿La estación Sur tiene 16 atracaderos?", "neutral"),
        ("Station Sur has 16 berths.", "Station Sur has 16 berths?", "neutral"),
    ],
)
def test_count_statements(claim, source, expected):
    assert stance(claim, source)[0] == expected


def check(query, texts, **options):
    engine = Veritas(Config(page_size=10))
    try:
        engine.add(
            [
                {"id": str(i), "text": t} if isinstance(t, str) else {"id": str(i), **t}
                for i, t in enumerate(texts)
            ]
        )
        return engine.search(query, **options)
    finally:
        engine.close()


CLAIM = "The hospital operated by Cedar Trust has 18 beds."
BRIDGE = "Cedar Trust operates Hospital Mar."
FACT = "El hospital Mar tiene 18 camas."


@pytest.mark.parametrize("fact, expected", [(FACT, "supported"), ("Hospital Mar has 15 beds.", "refuted")])
def test_join_preserves_both_required_spans(fact, expected):
    r = check(CLAIM, [BRIDGE, fact])
    decision = r.claims[0]
    assert decision.verdict == expected
    cited = [r.evidence[i] for i in decision.citations]
    assert [e.passage for e in cited] == [BRIDGE, fact]
    assert cited[1].premises[0].doc_id == cited[0].doc_id
    assert abs(r.evidence_balance) == 0.5


@pytest.mark.parametrize(
    "texts",
    [
        [BRIDGE],
        [FACT],
        [BRIDGE, "Hospital Mars has 18 beds."],
        [BRIDGE, "Hospital Mar has 18 seats."],
        ["Cedar Trust owns Hospital Mar.", FACT],
        [BRIDGE, FACT, "Cedar Trust operates Hospital Sur."],
        [BRIDGE, FACT, "Cedar Trust does not operate Hospital Mar."],
        ["Cedar Trust operates Station Mar.", "Station Mar has 18 beds."],
        ["Cedar Trust operates Hospital Mar?", FACT],
    ],
)
def test_missing_ambiguous_and_incompatible_premises_abstain(texts):
    result = check(CLAIM, texts)
    assert result.abstained and not result.claims[0].citations


def test_out_of_date_bridge_cannot_license_a_join():
    result = check(
        CLAIM,
        [{"text": BRIDGE, "valid_to": "2019-12-31"}, {"text": FACT, "valid_from": "2020-01-01"}],
        as_of="2025-01-01",
    )
    assert result.abstained


def test_duplicate_bridge_cannot_amplify_a_join():
    r = check(CLAIM, [BRIDGE, BRIDGE, BRIDGE, FACT])
    assert r.evidence_balance == 0.5 and len(r.claims[0].citations) == 2


def test_conflicting_measurements_survive_the_join():
    r = check(CLAIM, [BRIDGE, FACT, "Hospital Mar has 19 beds."])
    assert r.claims[0].verdict == "conflict"
    assert r.contradictions


def test_diversity_before_budget_cutoff():
    query = "The ferry carries 40 passengers."
    docs = [{"id": "primary", "title": "Survey record", "text": query}] + [
        {"id": f"copy{j}", "title": query, "text": "The ferry carries 41 passengers."} for j in range(16)
    ]
    for diverse, expected in [(False, "refuted"), (True, "conflict")]:
        e = Veritas(Config(page_size=2, diverse_candidates=diverse))
        e.add(docs)
        result = e.search(query, 1)
        assert result.claims[0].verdict == expected
        assert result.documents_examined == 2 and result.search_budget_used == 1
        e.close()


def test_scientific_profile_requires_its_models():
    with pytest.raises(ValidationError):
        Config(scientific_evidence=True)
