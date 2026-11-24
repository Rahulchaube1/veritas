import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from rationale_spans import matching_sentences


def test_repeated_words_are_credited_only_at_the_annotated_position():
    text = "Dock open.\nDock open."
    assert matching_sentences(text, 0, 10, {1}) == []
    assert matching_sentences(text, 11, 21, {1}) == [1]
    assert matching_sentences(text, 11, 15, {1}) == [1]


def test_citation_boundaries_and_unicode_offsets():
    text = "Río.\n\nDeux."
    assert matching_sentences(text, 6, len(text), {2}) == [2]
    assert matching_sentences(text, 3, 7, {0, 2}) == []
    assert matching_sentences(text, 4, 5, {0, 1}) == []
    with pytest.raises(ValueError):
        matching_sentences(text, 0, 0, {0})
