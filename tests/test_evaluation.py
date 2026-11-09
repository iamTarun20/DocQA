from collections import Counter

import pytest

from docqa.evaluate import load_gold, support_counts
from docqa.experiments import END, START, replace_results, run
from docqa.ingest import DEFAULT_CORPUS


def test_gold_labels_and_facts_are_backed_by_corpus() -> None:
    cases = load_gold()
    assert Counter(case.kind for case in cases) == {
        "answerable": 28,
        "unanswerable": 8,
        "injection": 4,
    }
    assert len({case.question for case in cases}) == len(cases)
    for case in cases:
        if case.kind == "answerable":
            source = (DEFAULT_CORPUS / f"{case.doc_id}.md").read_text(encoding="utf-8")
            assert case.fact.casefold() in source.casefold()


def test_lexical_support_ignores_citations_and_counts_sentences() -> None:
    assert support_counts(
        "Alpha beta gamma delta echo. [policy#c01] Foxtrot.", "alpha beta gamma"
    ) == (
        1,
        2,
    )


def test_smoke_uses_only_fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked() -> None:
        raise AssertionError("Smoke evaluation attempted to load BGE.")

    monkeypatch.setattr("docqa.experiments.BGEEmbedder", blocked)
    assert "passed (40 questions" in run(smoke=True)


def test_replaces_only_the_readme_results() -> None:
    readme = f"before\n{START}\nold\n{END}\nafter\n"
    assert replace_results(readme, "new") == f"before\n{START}\nnew\n{END}\nafter\n"
