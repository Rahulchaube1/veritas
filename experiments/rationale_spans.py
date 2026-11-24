"""Map the VERITAS half-open document offsets to SciFact's sentence indices."""


def matching_sentences(text: str, start: int, end: int, eligible: set[int]) -> list[int]:
    if not 0 <= start < end <= len(text):
        raise ValueError("Citation must be a nonempty span within the abstract")
    offset = 0
    # SciFact stores a list of sentences. corpus() joins those sentences with one newline.
    for i, sentence in enumerate(text.split("\n")):
        boundary = offset + len(sentence)
        if start < boundary:
            return [i] if i in eligible and end <= boundary and start >= offset else []
        offset = boundary + 1
    return []
