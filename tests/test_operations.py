import json
import sys
from pathlib import Path

import pytest

from veritas import Config, Document, Veritas
from veritas.cli import main
from veritas.extraction import extract_html
from veritas.provenance import graph


def test_cli(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("database: " + str(tmp_path / "doc.db"))
    for params in [
        ["index", "datasets/public_examples.jsonl"],
        ["retrieve", "Neptune"],
        ["search", "Neptune is the eighth and most distant planet in our solar system."],
    ]:
        monkeypatch.setattr(sys, "argv", ["veritas", "--config", str(cfg), *params])
        main()
        out = json.loads(capsys.readouterr().out)
    assert not out["abstained"]
    assert out["evidence"][0]["publisher"] == "NASA"


def test_html():
    text = extract_html(
        "<section><p>The toll is 12.</p><style>unused</style><p>Second &amp; third.</p></section>"
    )
    assert text == "The toll is 12.\nSecond & third."
    with pytest.raises(ValueError):
        extract_html("x" * 128001)


def test_link_expansion():
    e = Veritas(Config(page_size=1))
    e.add(
        [
            Document(
                id="digest",
                title="Ferry announcement",
                text="A short digest about ferry timetables.",
                cites=["record"],
            ),
            Document(id="record", title="Harbor register", text="A primary record."),
            Document(id="distractor", text="Ferry schedules change."),
        ]
    )
    r = e.search("Ferry announcement.", 2, follow_links=True)
    assert any(e.doc_id == "record" and e.rank_features.get("citation_hop") == 1 for e in r.evidence)
    assert r.search_budget_used == 2


def test_corpus_bytes():
    e = Veritas(Config(corpus_byte_limit=1000))
    e.add([Document(id="A", text="x" * 980)])
    with pytest.raises(ValueError):
        e.add([Document(id="B", text="y" * 30)])
    e.add([Document(id="A", text="y" * 500)])


def test_stale_revision():
    e = Veritas()
    e.add(
        [
            Document(
                id="x",
                text="The port is open.",
                published="2010-01-01",
                modified="2025-06-01",
                valid_from="2010-01-01",
            )
        ]
    )
    assert e.search("The port is open.", as_of="2021-01-01").abstained


def test_learned_ranking():
    e = Veritas(Config(ranker=Path("models/ranker.json")))
    e.add([Document(id="source", title="Seaport", text="The ferry is running.")])
    assert 0 < e.retrieve("Seaport")[0].features["learned_relevance"] < 1


def test_quotes():
    a = Document(
        id="a",
        text='A record includes "the eastern bank will supply seven vessels next Tuesday" in a letter.',
    )
    b = Document(
        id="b",
        text='New publication: "the eastern bank will supply seven vessels next Tuesday" in our local roundup.',
    )
    _, edges = graph([a, b])
    assert any(e.relation == "shared_quote" for e in edges)


def test_shared_decomposition_budget():
    e = Veritas()
    e.add([Document(id="a", text="The port is open. The road is closed.")])
    result = e.verify_text("The port is open. The road is closed.", 2)
    assert [c.verdict for c in result.claims] == ["supported", "supported"]
    assert result.search_budget_used == 2
    for c in result.claims:
        assert result.evidence[c.citations[0]].passage.rstrip(".") == c.text
    limited = e.verify_text("The port is open. The road is closed.", 1)
    assert limited.abstained and limited.claims[1].verdict == "insufficient"
