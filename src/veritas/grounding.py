"""Estimate rationale relevance before accepting a neural stance."""

import math
import re
from difflib import SequenceMatcher

import numpy as np

from .retrieval import tokens


def signals(claim: str, passage: str, title: str, idf: dict[str, float]) -> list[float]:
    q = set(tokens(claim))
    s = set(tokens(passage))
    t = set(tokens(title))

    def weights(words):
        return sum(idf.get(w, 4.0) for w in sorted(words))

    qsize = max(1, weights(q))
    a = re.findall(r"\b\d+(?:\.\d+)?\b", claim)
    b = re.findall(r"\b\d+(?:\.\d+)?\b", passage)
    longest = SequenceMatcher(None, tokens(claim), tokens(passage), autojunk=False).find_longest_match().size
    return [
        weights(q & s) / qsize,
        weights(q & t) / qsize,
        weights(q & s) / max(1, weights(s)),
        len(q & s) / max(1, len(q)),
        longest / max(len(q), 1),
        float(bool(a and b and set(a) & set(b))),
        math.log1p(len(s)) / 6,
    ]


def relevance(claim, passage, title, checkpoint):
    f = signals(claim, passage, title, checkpoint["idf"])
    x = float(np.dot(f, checkpoint["coef"]) + checkpoint["intercept"])
    return 1 / (1 + math.exp(-max(-40, min(40, x))))
