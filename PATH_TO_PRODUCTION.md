# Path to production

What I'd do to take this from a prototype to something that can run the contact center for all 11
branches, each with its own policies, at roughly 8,500 interactions a month (about 6k calls and 2.5k
emails). That volume is modest, so most of the work is reliability, safety, and per-branch
correctness rather than raw throughput.

## Harden

- Swap the mock for the real Booking API. It sits behind the per-channel bearer tokens the spec
  already defines, and `booking_client` is the only integration point, so the switch is a one-line
  base-URL change. Add timeouts, retries with backoff, and idempotency keys on `POST` so a retried
  request can't double-book, and fall back to the handoff path (circuit-break) when the API is down.
- Move secrets into a real secret manager instead of `.env`, and pin the model and prompt versions so
  behavior doesn't drift underneath you. Treat the knowledge pack as versioned content with an
  ingestion pipeline that re-indexes on change and stamps a `doc_version` on every chunk, so
  citations stay auditable.
- Tighten the boundary around the LLM. Never execute instructions found in retrieved text, keep tool
  arguments schema-validated (already the case), redact PII in logs, and enforce ownership checks on
  `GET` (the mock already takes a `customer_id`). An abuse/profanity filter in front of the model is
  cheap insurance.
- Keep the confirm-before-commit gate, and add server-side authorization for the higher-risk actions
  like fee waivers and cancellations, so the assistant can never do more than a human agent could.
- Keep eligibility, fees, and the booking window deterministic in code rather than letting the model
  decide them, and grow the unit-test suite as the policies get more complex.

## Monitor

- Trace every turn (LangSmith or OpenTelemetry): the route taken, tool calls, retrieval scores,
  latency, and token cost. Run `eval/run_eval.py` in CI on every prompt, model, or policy change,
  with thresholds that block regressions, and keep growing the labeled set from real transcripts.
- Watch the KPIs that actually matter: containment (resolved without a human), handoff rate by
  reason, action success rate, p50/p95 latency, cost per interaction, and groundedness from online
  faithfulness sampling. Alert when retrieval confidence drifts or handoff and abandon rates climb.
- Close the loop. Capture thumbs up/down and the edits human agents make, and send low-confidence or
  thumbs-down turns to a review queue that feeds new eval cases and content fixes.
- Audit safety continuously. Log every emergency detection and every write with full context, and
  sample specifically for emergency false-negatives, which are the worst failure mode here.

## Scale to 11 branches with branch-specific policies

- Make the knowledge multi-tenant. Tag each chunk with its `branch`/`region` and filter retrieval to
  the caller's branch (Chroma metadata filters already support this), then model policy as a shared
  base plus per-branch overrides. Fix the known data gaps first: add the missing South-region
  service-area document and reconcile ZIP 22046, so eligibility there stops defaulting to a handoff.
- Resolve the branch up front from the ZIP, phone number (DID), or channel, and use it to pick the
  right hours, surcharges, contacts, and handoff destination.
- For throughput, run the API and agent as stateless services behind an autoscaler, move conversation
  state and the checkpointer into Redis or Postgres, and make tool calls async. Cache embeddings and
  common FAQ answers. At ~8,500/month you're only handling a few concurrent conversations at peak, so
  reliability and cost control matter far more than QPS.
- Add channel adapters on top of the same agent core: telephony (ASR/TTS, barge-in, DTMF) and an
  email worker (threading, async replies). The `channel` field already flows end to end, so the core
  doesn't have to change.
- Control cost by using a small model for routing and judging and reserving the larger model for
  customer-facing answers, batching embeddings, and caching. Re-run the evals whenever you change
  model tiers to confirm quality holds.

## Biggest risks

- **Silent grounding failures**, where the assistant is confidently wrong. Covered by the
  answer-only-from-context rule, citations, and online faithfulness monitoring.
- **Emergency false-negatives**, the highest-severity miss. The deterministic keyword overlay stays
  in place and gets audited continuously.
- **Stale or branch-mismatched policy.** Versioned content, per-branch retrieval filters, and eval
  gates in CI keep answers tied to the right, current policy.
