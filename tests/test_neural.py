from pathlib import Path

import pytest

from veritas import Config, Document, Veritas


@pytest.mark.skipif(
    not Path(".cache/models/nli/onnx/model_quantized.onnx").exists(),
    reason="pinned optional neural weights absent",
)
def test_neural_pipeline():
    e = Veritas(Config(model_dir=Path(".cache/models"), rationale_gate=Path("models/rationale_gate.json")))
    e.add(
        [
            Document(id="a", text="The cat is sleeping."),
            Document(id="b", text="The station has four entrances."),
        ]
    )
    assert e.retrieve("Un gato está durmiendo.", 1, "dense")[0].doc_id == "a"
    assert e.search("The cat is sleeping.").claims[0].verdict == "supported"
    scores = e.neural["nli"].run([("A cat is asleep.", "An animal is sleeping.")])[0]
    assert e.neural["nli"].labels[scores.argmax()] == "entailment"


@pytest.mark.skipif(
    not Path(".cache/models/scientific_nli/onnx/model_quantized.onnx").exists(),
    reason="pinned optional scientific weights absent",
)
def test_scientific_pipeline_grounds_a_paraphrase():
    engine = Veritas(
        Config(
            model_dir=Path(".cache/models"),
            rationale_gate=Path("models/rationale_gate.json"),
            scientific_evidence=True,
        )
    )
    try:
        assert "entailment" in engine.neural["nli"].labels.values()
        assert engine.config.scientific_evidence
        engine.add([Document(id="sample", title="Train schedule", text="The train departed at noon.")])
        result = engine.search("The train left at noon.")
        assert result.claims[0].verdict == "supported"
        evidence = result.evidence[result.claims[0].citations[0]]
        assert evidence.method == "nli" and "passage_relevance_semantic" in evidence.rank_features
    finally:
        engine.close()
