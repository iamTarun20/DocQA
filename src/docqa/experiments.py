"""Run the benchmark matrix and regenerate checked-in evaluation outputs."""

import json
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from platform import python_version

from docqa.answer import DEFAULT_OVERLAP, DEFAULT_SIZE, DEFAULT_THRESHOLD, DocQA, OfflineProvider
from docqa.evaluate import SUPPORT_MIN, TOP_K, Case, Observation, load_gold, observe, summarize
from docqa.ingest import DEFAULT_CORPUS, load_chunks
from docqa.retrieval import BGE_MODEL, BGEEmbedder, Embedder, FakeEmbedder, Mode, Retriever

START, END = "<!-- EVAL:START -->", "<!-- EVAL:END -->"
SIZES = (DEFAULT_SIZE, 400)
THRESHOLDS = (DEFAULT_THRESHOLD, 0.75)


@dataclass(frozen=True)
class Trial:
    """Measured metrics and observations for one experiment configuration."""

    mode: Mode
    size: int
    chunk_count: int
    threshold: float
    observations: list[Observation]
    metrics: dict[str, float | None]


def _measure(cases: list[Case], embedder: Embedder, smoke: bool) -> list[Trial]:
    trials = []
    embedder.query(cases[0].question)  # Exclude cold encoder initialization.
    for size in SIZES[:1] if smoke else SIZES:
        chunks = load_chunks(DEFAULT_CORPUS, size, DEFAULT_OVERLAP)
        retriever = Retriever(chunks, embedder)
        for mode in Mode:
            for threshold in THRESHOLDS[:1] if smoke else THRESHOLDS:
                engine = DocQA(retriever, OfflineProvider(), threshold)
                items = [observe(engine, case, mode) for case in cases]
                trials.append(Trial(mode, size, len(chunks), threshold, items, summarize(items)))
    return trials


def _percentage(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _conclusion(trials: list[Trial]) -> tuple[str, Observation | None]:
    bm25, dense, hybrid = [
        trial for trial in trials if trial.size == SIZES[0] and trial.threshold == THRESHOLDS[0]
    ]
    strict = next(
        trial
        for trial in trials
        if trial.mode == Mode.HYBRID
        and trial.size == hybrid.size
        and trial.threshold == THRESHOLDS[1]
    )
    bigger = next(
        trial
        for trial in trials
        if trial.mode == Mode.HYBRID
        and trial.size == SIZES[1]
        and trial.threshold == hybrid.threshold
    )
    mrr = ", ".join(f"{trial.mode} {trial.metrics['mrr']:.3f}" for trial in (bm25, dense, hybrid))
    first = (
        f"At {hybrid.size} words and threshold {hybrid.threshold:g}, MRR was {mrr}, "
        f"and hybrid MRR was {bigger.metrics['mrr']:.3f} at {bigger.size} words."
    )
    refusals = [_percentage(trial.metrics["refusal_accuracy"]) for trial in (hybrid, strict)]
    facts = [_percentage(trial.metrics["fact_recovery"]) for trial in (hybrid, strict)]
    second = (
        f"Raising the hybrid threshold to {strict.threshold:g} at {hybrid.size} words moved "
        f"refusal accuracy from {refusals[0]} to {refusals[1]} and cited key-fact recovery "
        f"from {facts[0]} to {facts[1]}."
    )
    failures = [
        item
        for item in hybrid.observations
        if (item.case.kind == "answerable" and not item.fact_recovered)
        or (item.case.kind == "unanswerable" and not item.refused)
    ]
    currency = next(
        (
            item
            for item in failures
            if item.case.id == "u04"
            and item.answer.startswith("The standard procurement card limit")
        ),
        None,
    )
    failure = currency or next(iter(failures), None)
    if currency:
        third = (
            f"The euro-denominated card-limit question ({currency.case.id}) received the "
            f"dollar-denominated limit because the top sentence's lexical score of "
            f"{currency.score:.2f} cleared the gate despite the unmatched currency."
        )
    elif failure:
        if failure.refused:
            reason = "the evidence gate refused the question"
        elif failure.case.kind == "answerable":
            reason = "lexical selection returned text without the expected cited fact"
        else:
            reason = "lexical overlap admitted an answer to an unanswerable question"
        third = f"The baseline missed {failure.case.id} because {reason}."
    else:
        third = "The baseline had no labeled key-fact or refusal failures on these cases."
    return " ".join((first, second, third)), failure


def _table(trials: list[Trial]) -> str:
    header = (
        "| Mode | Words | Chunks | Threshold | Hit@1 | Hit@3 | Hit@5 | MRR | "
        "Cited fact | Refusal | Lexical support | Canary pass | Exposed probes | Median ms |"
    )
    rows = [header, "|" + "---|" * 14]
    for trial in trials:
        metrics = trial.metrics
        rates = [
            _percentage(metrics[key])
            for key in (
                "fact_recovery",
                "refusal_accuracy",
                "answer_support_rate",
                "injection_pass_rate",
            )
        ]
        values = [
            str(trial.mode),
            str(trial.size),
            str(trial.chunk_count),
            f"{trial.threshold:g}",
            *(_percentage(metrics[f"hit_at_{k}"]) for k in (1, 3, 5)),
            f"{metrics['mrr']:.3f}",
            *rates,
            str(metrics["probe_exposure_count"]),
            f"{metrics['median_latency_ms']:.2f}",
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def replace_results(readme: str, generated: str) -> str:
    """Replace just the marked README block, rejecting missing or ambiguous markers."""
    if (
        readme.count(START) != 1
        or readme.count(END) != 1
        or readme.index(START) > readme.index(END)
    ):
        raise ValueError("README must contain one ordered pair of evaluation markers.")
    before, rest = readme.split(START)
    _, after = rest.split(END)
    return f"{before}{START}\n{generated}\n{END}{after}"


def run(*, smoke: bool = False) -> str:
    """Benchmark local BGE or run the download-free smoke check."""
    cases = load_gold()
    counts = Counter(case.kind for case in cases)
    if len(counts) != 3:
        raise ValueError("Evaluation requires answerable, unanswerable and injection cases.")
    embedder = FakeEmbedder() if smoke else BGEEmbedder()
    trials = _measure(cases, embedder, smoke)
    if smoke:
        for trial in trials:
            metrics = trial.metrics
            if (
                not metrics["fact_recovery"]
                or not metrics["refusal_accuracy"]
                or metrics["injection_pass_rate"] != 1
                or metrics["probe_exposure_count"] != counts["injection"]
            ):
                raise ValueError(
                    f"Offline evaluation smoke test failed for {trial.mode}: {metrics}"
                )
        return (
            f"Offline evaluation smoke test passed ({len(cases)} questions × {len(trials)} modes)."
        )

    conclusion, failure = _conclusion(trials)
    document_count = len(list(DEFAULT_CORPUS.rglob("*.md")))
    generated = (
        f"{document_count} documents and {len(cases)} questions "
        f"({counts['answerable']} answerable, {counts['unanswerable']} unanswerable and "
        f"{counts['injection']} injection probes). Each answer saw up to {TOP_K} passages "
        f"with a {DEFAULT_OVERLAP}-word maximum overlap. Dense encoder "
        f"{BGE_MODEL} via FastEmbed {version('fastembed')} on Python {python_version()}.\n\n"
        f"{_table(trials)}\n\n"
        f"Lexical support uses a {SUPPORT_MIN:.0%} content-word cutoff per answer sentence.\n\n"
        f"{conclusion}\n"
    )
    readme = Path("README.md")
    updated_readme = replace_results(readme.read_text(encoding="utf-8"), generated)
    output = Path("eval")
    output.mkdir(exist_ok=True)
    payload = {
        "embedder": BGE_MODEL,
        "fastembed_version": version("fastembed"),
        "python_version": python_version(),
        "provider": "offline",
        "documents": document_count,
        "questions": dict(counts),
        "top_k": TOP_K,
        "max_overlap_words": DEFAULT_OVERLAP,
        "lexical_support_cutoff": SUPPORT_MIN,
        "runs": [
            {
                "mode": trial.mode,
                "chunk_size_words": trial.size,
                "chunks": trial.chunk_count,
                "threshold": trial.threshold,
                **trial.metrics,
            }
            for trial in trials
        ],
        "examined_failure": (
            {
                "id": failure.case.id,
                "question": failure.case.question,
                "output": failure.answer,
                "top_chunk": failure.top_chunk,
                "evidence_score": failure.score,
            }
            if failure
            else None
        ),
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("# DocQA evaluation\n\n" + generated, encoding="utf-8")
    readme.write_text(updated_readme, encoding="utf-8")
    return (
        f"Evaluated {len(cases)} questions across {len(trials)} configurations. "
        "Wrote eval/results.json, eval/report.md and README.md."
    )
