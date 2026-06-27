"""Evaluation harness for the Meridian assistant.

Runs every case in ``testset.yaml`` through the full agent and scores four dimensions:

  1. Retrieval quality (RAG/faq cases): hit@1, hit@3, MRR, recall@k at source-file granularity.
  2. Answer correctness: substring assertions + an LLM judge (correctness & groundedness)
     + RAGAS (faithfulness, response relevancy, context precision/recall).
  3. Action correctness: the right Booking API tool ran, returned the expected status, and
     (for writes) the assistant asked for confirmation BEFORE committing.
  4. Handoff correctness: handoff fired exactly when expected, to the right route.

Run:  python eval/run_eval.py            (full: judge + RAGAS)
      python eval/run_eval.py --quick    (skip RAGAS)
      python eval/run_eval.py --no-judge --no-ragas
Outputs eval/results/report.md and eval/results/report.json. Start the mock API first.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# --- RAGAS 0.4.3 imports a removed langchain-community Vertex path at module load; shim it. ---
for _path, _attrs in {
    "langchain_community.chat_models.vertexai": ["ChatVertexAI"],
    "langchain_community.llms.vertexai": ["VertexAI"],
}.items():
    if _path not in sys.modules:
        _m = types.ModuleType(_path)
        for _a in _attrs:
            setattr(_m, _a, type(_a, (), {}))
        sys.modules[_path] = _m

from meridian import booking_client  # noqa: E402
from meridian.agent.graph import Assistant  # noqa: E402
from meridian.agent.llm import get_chat_llm  # noqa: E402
from meridian.config import get_settings  # noqa: E402
from meridian.retrieval.retriever import get_retriever  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
RETRIEVAL_K = 8


@dataclass
class CaseResult:
    id: str
    intent: str
    answer: str = ""
    retrieval: dict | None = None
    keyword_ok: bool | None = None
    judge: dict | None = None
    handoff_ok: bool | None = None
    action_ok: bool | None = None
    confirmation_ok: bool | None = None
    notes: list[str] = field(default_factory=list)
    # RAGAS sample inputs (faq cases)
    ragas_sample: dict | None = None


def load_cases(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text())
    return data["cases"]


def run_turns(assistant: Assistant, case: dict) -> dict:
    """Drive the conversation; return final state + whether confirmation was requested."""
    thread = uuid.uuid4().hex[:8]
    turns = case.get("turns") or [case["query"]]
    channel = case.get("channel", "web_chat")
    saw_confirmation = False
    final: dict = {}
    for t in turns:
        final = assistant.chat(t, thread_id=thread, channel=channel)
        if final.get("awaiting_confirmation"):
            saw_confirmation = True
    final["_saw_confirmation"] = saw_confirmation
    return final


def retrieval_metrics(case: dict, retriever) -> dict:
    expected = set(case["expected_sources"])
    query = case.get("retrieval_query") or case.get("query") or (case.get("turns") or [""])[0]
    res = retriever.retrieve(query, top_k=RETRIEVAL_K, top_n=RETRIEVAL_K)
    ranked_files: list[str] = []
    for c in res.chunks:
        f = c.metadata.get("source_file", "")
        if f not in ranked_files:
            ranked_files.append(f)
    first_rank = next((i + 1 for i, f in enumerate(ranked_files) if f in expected), 0)
    found = expected & set(ranked_files)
    return {
        "expected": sorted(expected),
        "ranked": ranked_files[:5],
        "hit@1": bool(ranked_files[:1] and ranked_files[0] in expected),
        "hit@3": bool(found & set(ranked_files[:3])),
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
        "recall": len(found) / len(expected) if expected else 0.0,
        "contexts": [c.text for c in res.chunks[:4]],
        "top_score": round(res.top_score, 3),
    }


def keyword_check(case: dict, answer: str) -> bool | None:
    low = answer.lower()
    contains = case.get("answer_contains") or []
    excludes = case.get("answer_excludes") or []
    if not contains and not excludes:
        return None
    ok = all(tok.lower() in low for tok in contains)
    ok = ok and all(tok.lower() not in low for tok in excludes)
    return ok


_JUDGE_PROMPT = """You are grading a home-services assistant's answer against a reference.
Question: {q}
Reference (ground truth): {ref}
Assistant answer: {ans}

Return ONLY JSON: {{"correct": 0 or 1, "grounded": 0 or 1, "reason": "..."}}
- correct = 1 if the answer correctly answers the question and is consistent with the reference.
  Extra correct detail is fine; do NOT penalize it.
- grounded = 1 unless the answer states something that CONTRADICTS the reference or is clearly
  fabricated. Additional plausible service details that don't conflict are acceptable."""


def llm_judge(case: dict, answer: str) -> dict | None:
    ref = case.get("reference")
    if case.get("expect_handoff") or not ref:  # judge only reference-backed answer cases
        return None
    llm = get_chat_llm(0.0)
    prompt = _JUDGE_PROMPT.format(q=case.get("query") or (case.get("turns") or [""])[0],
                                  ref=ref, ans=answer)
    try:
        raw = llm.invoke(prompt).content
        start, end = raw.find("{"), raw.rfind("}")
        obj = json.loads(raw[start:end + 1])
        return {"correct": int(obj.get("correct", 0)), "grounded": int(obj.get("grounded", 0)),
                "reason": obj.get("reason", "")[:200]}
    except Exception as exc:
        return {"correct": 0, "grounded": 0, "reason": f"judge error: {exc}"}


def handoff_check(case: dict, state: dict) -> bool:
    handoff = state.get("handoff")
    want = bool(case.get("expect_handoff"))
    if bool(handoff) != want:
        return False
    if want and case.get("expected_route"):
        return case["expected_route"].lower() in (handoff.get("recommended_route", "").lower())
    return True


def action_check(case: dict, state: dict) -> tuple[bool | None, bool | None]:
    """Return (action_ok, confirmation_ok)."""
    exp = case.get("expected_action")
    if not exp:
        return None, None
    tr = state.get("tool_result") or {}
    status_ok = tr.get("status") == exp.get("status")
    confirmation_ok = None
    if case.get("expect_confirmation"):
        # the write must have been gated behind a confirmation turn
        confirmation_ok = bool(state.get("_saw_confirmation")) and bool(tr)
    return status_ok, confirmation_ok


def run_ragas(samples: list[dict]) -> dict:
    if not samples:
        return {"status": "skipped (no faq samples)"}
    try:
        from langchain_openai import OpenAIEmbeddings
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                                    LLMContextRecall, ResponseRelevancy)

        s = get_settings()
        llm = LangchainLLMWrapper(get_chat_llm(0.0))
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=s.openai_embedding_model,
                                                          api_key=s.openai_api_key))
        ds = EvaluationDataset(samples=[SingleTurnSample(**x) for x in samples])
        metrics = [Faithfulness(), ResponseRelevancy(),
                   LLMContextPrecisionWithReference(), LLMContextRecall()]
        result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb,
                          show_progress=False)
        df = result.to_pandas()
        scores = {}
        for col in df.columns:
            if df[col].dtype.kind in "fi":
                val = df[col].mean()
                scores[col] = round(float(val), 3)
        return {"status": "ok", "n_samples": len(samples), "scores": scores}
    except Exception as exc:
        return {"status": f"unavailable: {type(exc).__name__}: {str(exc)[:200]}"}


def _pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.0f}% ({num}/{den})" if den else "n/a"


def aggregate_and_report(results: list[CaseResult], ragas: dict, args) -> dict:
    def collect(pred):
        vals = [pred(r) for r in results]
        vals = [v for v in vals if v is not None]
        return sum(1 for v in vals if v), len(vals)

    retr = [r.retrieval for r in results if r.retrieval]
    summary = {
        "total_cases": len(results),
        "retrieval": {
            "n": len(retr),
            "hit@1": round(sum(x["hit@1"] for x in retr) / len(retr), 3) if retr else None,
            "hit@3": round(sum(x["hit@3"] for x in retr) / len(retr), 3) if retr else None,
            "mrr": round(sum(x["mrr"] for x in retr) / len(retr), 3) if retr else None,
            "recall@8": round(sum(x["recall"] for x in retr) / len(retr), 3) if retr else None,
        },
        "answer_keyword": collect(lambda r: r.keyword_ok),
        "judge_correct": collect(lambda r: None if not r.judge else bool(r.judge["correct"])),
        "judge_grounded": collect(lambda r: None if not r.judge else bool(r.judge["grounded"])),
        "handoff": collect(lambda r: r.handoff_ok),
        "action": collect(lambda r: r.action_ok),
        "confirmation_gate": collect(lambda r: r.confirmation_ok),
        "ragas": ragas,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full = {"summary": summary, "cases": [vars(r) for r in results]}
    (RESULTS_DIR / "report.json").write_text(json.dumps(full, indent=2, default=str))

    lines = ["# Meridian Assistant - Evaluation Report", ""]
    s = get_settings()
    lines.append(f"Model: `{s.openai_chat_model}` | Embeddings: `{s.openai_embedding_model}` | "
                 f"Reranker: `{s.reranker}` | Cases: {len(results)}")
    lines += ["", "## Summary", ""]
    rt = summary["retrieval"]
    if rt["n"]:
        lines.append(f"- Retrieval ({rt['n']} faq cases): hit@1 **{rt['hit@1']}**, "
                     f"hit@3 **{rt['hit@3']}**, MRR **{rt['mrr']}**, recall@8 **{rt['recall@8']}**")
    lines.append(f"- Answer keyword assertions: **{_pct(*summary['answer_keyword'])}**")
    lines.append(f"- LLM judge correctness: **{_pct(*summary['judge_correct'])}**, "
                 f"groundedness: **{_pct(*summary['judge_grounded'])}**")
    lines.append(f"- Action correctness: **{_pct(*summary['action'])}**, "
                 f"confirm-before-commit: **{_pct(*summary['confirmation_gate'])}**")
    lines.append(f"- Handoff routing: **{_pct(*summary['handoff'])}**")
    if isinstance(ragas, dict) and ragas.get("status") == "ok":
        sc = ", ".join(f"{k}={v}" for k, v in ragas["scores"].items())
        lines.append(f"- RAGAS ({ragas['n_samples']} samples): {sc}")
    else:
        lines.append(f"- RAGAS: {ragas.get('status') if isinstance(ragas, dict) else ragas}")

    lines += ["", "## Per-case results", "",
              "| id | intent | retr hit@3 | kw | judge c/g | action | confirm | handoff |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        retr_hit = "-" if not r.retrieval else ("Y" if r.retrieval["hit@3"] else "N")
        kw = "-" if r.keyword_ok is None else ("Y" if r.keyword_ok else "N")
        jc = "-" if not r.judge else f"{r.judge['correct']}/{r.judge['grounded']}"
        act = "-" if r.action_ok is None else ("Y" if r.action_ok else "N")
        con = "-" if r.confirmation_ok is None else ("Y" if r.confirmation_ok else "N")
        hnd = "Y" if r.handoff_ok else "N"
        lines.append(f"| {r.id} | {r.intent} | {retr_hit} | {kw} | {jc} | {act} | {con} | {hnd} |")
    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip RAGAS")
    parser.add_argument("--no-ragas", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--testset", default=str(EVAL_DIR / "testset.yaml"))
    args = parser.parse_args()

    try:
        booking_client.reset()
    except Exception as exc:
        print(f"WARNING: could not reset mock API ({exc}). Is it running on :8000?")

    cases = load_cases(Path(args.testset))
    assistant = Assistant()
    retriever = get_retriever()
    results: list[CaseResult] = []
    ragas_samples: list[dict] = []

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']} ...", flush=True)
        state = run_turns(assistant, case)
        answer = state.get("answer", "")
        cr = CaseResult(id=case["id"], intent=case.get("intent", ""), answer=answer)
        if case.get("expected_sources"):
            cr.retrieval = retrieval_metrics(case, retriever)
            if case.get("reference"):
                ragas_samples.append({
                    "user_input": case.get("query") or (case.get("turns") or [""])[0],
                    "response": answer,
                    "retrieved_contexts": cr.retrieval["contexts"],
                    "reference": case["reference"],
                })
        cr.keyword_ok = keyword_check(case, answer)
        cr.judge = None if args.no_judge else llm_judge(case, answer)
        cr.handoff_ok = handoff_check(case, state)
        cr.action_ok, cr.confirmation_ok = action_check(case, state)
        results.append(cr)

    ragas = {"status": "skipped"} if (args.quick or args.no_ragas) else run_ragas(ragas_samples)
    summary = aggregate_and_report(results, ragas, args)

    print("\n==== SUMMARY ====")
    print(json.dumps({k: v for k, v in summary.items() if k != "ragas"}, indent=2, default=str))
    print("ragas:", ragas.get("status") if isinstance(ragas, dict) else ragas)
    print(f"\nWrote {RESULTS_DIR/'report.md'} and report.json")


if __name__ == "__main__":
    main()
