# Path to Production - Meridian Assistant

Scaling this prototype to serve all 11 branches with branch-specific policies at ~8,500
interactions/month (~6k calls + ~2.5k emails). One page; grouped by harden / monitor / scale.

## Harden

- **Real Booking API + auth.** Replace the mock with the internal service behind the existing
  per-channel bearer tokens (`ivr | web_chat | email | agent`); keep `booking_client` as the only
  integration point (already a one-line base-URL swap). Add retries with backoff, timeouts,
  idempotency keys on `POST` (so a retried booking does not double-book), and circuit-breaking to
  the handoff path on outage.
- **Secrets & config.** Move keys to a secret manager (not `.env`); pin model + prompt versions;
  treat the knowledge pack as versioned content with an ingestion pipeline that re-indexes on
  change and records `doc_version` per chunk for auditable citations.
- **Prompt-injection / PII.** Strip/important-quote document content, never execute instructions
  from retrieved text, and keep tool arguments schema-validated (already enforced). Add PII
  minimization + redaction in logs, ownership checks on `GET` (the mock already supports
  `customer_id` validation), and a profanity/abuse filter before the LLM.
- **Confirmation & authorization.** Keep the confirm-before-commit gate; add server-side
  authorization for fee waivers and cancellations so the assistant can never exceed agent
  authority (mirrors the "escalate to Branch Manager" policy).
- **Determinism.** Keep ZIP eligibility, fees, and date windows in code (not the LLM); expand the
  deterministic test suite as policies grow.

## Monitor

- **Tracing & evals in CI.** Adopt LangSmith / OpenTelemetry tracing on every turn (route, tools,
  retrieval scores, latency, token cost). Run `eval/run_eval.py` in CI on each prompt/model/policy
  change with thresholds that block regressions; expand the set toward hundreds of labeled cases
  mined from real transcripts.
- **Live KPIs.** Containment (resolved without human), handoff rate by reason, action success
  rate, p50/p95 latency, cost/interaction, and **groundedness** (online faithfulness sampling).
  Alert on retrieval-confidence drift and rising handoff/abandon rates.
- **Feedback loop.** Capture thumbs-up/down + human-agent edits; route low-confidence and
  thumbs-down turns to a review queue that feeds new eval cases and content fixes.
- **Safety auditing.** Log every emergency detection and every write with full context; sample for
  false-negatives on emergencies (the highest-severity failure mode).

## Scale to 11 branches and branch-specific policies

- **Multi-tenant knowledge.** Tag every chunk with `branch` / `region` and filter retrieval by the
  caller's branch (Chroma metadata filters already support this). Model policy as
  base + per-branch override docs; the existing per-doc chunking + `version` metadata extends
  cleanly. **Fix the data gaps first**: add the missing South-region service-area doc and reconcile
  ZIP 22046, so eligibility stops defaulting to handoff there.
- **Branch routing & config.** A branch resolver (from ZIP / phone DID / channel) selects hours,
  surcharges, branch contacts, and the correct handoff destination per branch.
- **Throughput (~8,500/mo ~= a few concurrent at peak).** Run the API + agent as stateless
  services behind autoscaling; move conversation state + checkpointer to Redis/Postgres; make tool
  calls async. Cache embeddings and frequent FAQ answers. This volume is modest; reliability and
  cost control matter more than raw QPS.
- **Channel adapters.** Add telephony (ASR/TTS, barge-in, DTMF) and an email worker (threading,
  async replies) on top of the same agent core; the `channel` field already flows end-to-end.
- **Cost.** Use a smaller model for routing/judging and the larger model only for customer-facing
  generation; batch embeddings; cache. Re-run evals when changing tiers to confirm quality holds.

## Biggest risks to watch

1. **Silent grounding failures** (confident wrong answers) - mitigated by faithfulness monitoring
   + strict "answer-only-from-context" + citations.
2. **Emergency false-negatives** - keep the deterministic safety overlay and audit it continuously.
3. **Stale/branch-mismatched policy** - enforce versioned content + per-branch filters + eval gates.
