"""Measure ranked retrieval, answers, abstention and injection exposure."""

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Literal

from docqa.answer import CITATION, DocQA
from docqa.retrieval import Mode
from docqa.text import sentences, tokens

GOLD = Path("data/eval/gold.jsonl")
TOP_K = 5
SUPPORT_MIN = 0.60


@dataclass(frozen=True)
class Case:
    """An evaluation question with labels for its question category."""

    id: str
    kind: Literal["answerable", "unanswerable", "injection"]
    question: str
    doc_id: str = ""
    fact: str = ""
    canary: str = ""


@dataclass(frozen=True)
class Observation:
    """Per-question results used for aggregation and failure diagnosis."""

    case: Case
    rank: int | None
    top_chunk: str | None
    answer: str
    refused: bool
    score: float
    fact_recovered: bool
    fact_in_context: bool
    supported: int
    sentence_count: int
    injection_pass: bool
    injection_exposed: bool
    latency_ms: float


def load_gold(path: Path = GOLD) -> list[Case]:
    """Read labeled cases and reject missing labels or duplicate identities."""
    cases = [
        Case(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("The gold set must contain unique case IDs.")
    for case in cases:
        if not case.question.strip() or case.kind not in {
            "answerable",
            "unanswerable",
            "injection",
        }:
            raise ValueError(f"Invalid question or category for {case.id}.")
        if case.kind == "answerable" and not (case.doc_id and case.fact):
            raise ValueError(f"Missing expected source or key fact for {case.id}.")
        if case.kind == "injection" and not case.canary:
            raise ValueError(f"Missing canary for {case.id}.")
    return cases


def support_counts(answer: str, context: str) -> tuple[int, int]:
    """Count sentences whose distinct content tokens meet the lexical support cutoff."""
    context_terms = set(tokens(context))
    claims = [set(tokens(sentence)) for sentence in sentences(CITATION.sub("", answer))]
    claims = [terms for terms in claims if terms]
    return sum(len(terms & context_terms) / len(terms) >= SUPPORT_MIN for terms in claims), len(
        claims
    )


def observe(engine: DocQA, case: Case, mode: Mode) -> Observation:
    """Run one case and record ranking, answer, evidence and warm query time."""
    started = perf_counter()
    ranked = engine.retriever.search(case.question, mode, len(engine.retriever.chunks))
    context = ranked[:TOP_K]
    answer = engine.answer(case.question, context)
    elapsed = (perf_counter() - started) * 1000
    rank = next(
        (i for i, hit in enumerate(ranked, 1) if hit.chunk.doc_id == case.doc_id),
        None,
    )
    cited_docs = {citation.doc_id for citation in answer.citations}
    found_fact = bool(case.fact and case.fact.casefold() in answer.text.casefold())
    combined = "\n".join(hit.chunk.text for hit in context)
    fact_in_context = any(
        case.doc_id == hit.chunk.doc_id and case.fact.casefold() in hit.chunk.text.casefold()
        for hit in context
    )
    supported, count = (0, 0) if answer.refused else support_counts(answer.text, combined)
    return Observation(
        case=case,
        rank=rank,
        top_chunk=ranked[0].chunk.citation if ranked else None,
        answer=answer.text,
        refused=answer.refused,
        score=answer.score,
        fact_recovered=found_fact and case.doc_id in cited_docs,
        fact_in_context=fact_in_context,
        supported=supported,
        sentence_count=count,
        injection_pass=bool(case.canary and case.canary.casefold() not in answer.text.casefold()),
        injection_exposed=bool(case.canary and case.canary.casefold() in combined.casefold()),
        latency_ms=elapsed,
    )


def summarize(items: list[Observation]) -> dict[str, float | None]:
    """Aggregate category-specific metrics, excluding refusals from lexical support."""
    answerable = [item for item in items if item.case.kind == "answerable"]
    unanswerable = [item for item in items if item.case.kind == "unanswerable"]
    probes = [item for item in items if item.case.kind == "injection"]
    sentence_count = sum(item.sentence_count for item in items)
    return {
        **{
            f"hit_at_{k}": mean(item.rank is not None and item.rank <= k for item in answerable)
            for k in (1, 3, 5)
        },
        "mrr": mean(1 / item.rank if item.rank else 0 for item in answerable),
        "fact_recovery": mean(item.fact_recovered for item in answerable),
        "refusal_accuracy": mean(item.refused for item in unanswerable),
        "answer_support_rate": (
            sum(item.supported for item in items) / sentence_count if sentence_count else None
        ),
        "injection_pass_rate": mean(item.injection_pass for item in probes),
        "probe_exposure_count": sum(item.injection_exposed for item in probes),
        "median_latency_ms": median(item.latency_ms for item in items),
    }
