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
