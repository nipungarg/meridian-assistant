# Meridian Assistant - Evaluation Report

Model: `gpt-4.1` | Embeddings: `text-embedding-3-large` | Reranker: `cross_encoder` | Cases: 34

## Summary

- Retrieval (16 faq cases): hit@1 **1.0**, hit@3 **1.0**, MRR **1.0**, recall@8 **1.0**
- Answer keyword assertions: **100% (25/25)**
- LLM judge correctness: **94% (17/18)**, groundedness: **94% (17/18)**
- Action correctness: **100% (5/5)**, confirm-before-commit: **100% (3/3)**
- Handoff routing: **100% (34/34)**
- RAGAS (16 samples): faithfulness=0.929, answer_relevancy=0.752, llm_context_precision_with_reference=0.984, context_recall=0.969

## Per-case results

| id | intent | retr hit@3 | kw | judge c/g | action | confirm | handoff |
|---|---|---|---|---|---|---|---|
| hours_herndon_sat | faq_policy | Y | Y | 1/1 | - | - | Y |
| pricing_water_heater | faq_policy | Y | Y | 1/1 | - | - | Y |
| maintenance_plans | faq_policy | Y | Y | 1/1 | - | - | Y |
| payments_zelle | faq_policy | Y | Y | 1/1 | - | - | Y |
| after_hours_sunday | faq_policy | Y | Y | 1/1 | - | - | Y |
| cancellation_fee_window | faq_policy | Y | Y | 1/1 | - | - | Y |
| estimate_panel_upgrade | faq_policy | Y | Y | 0/0 | - | - | Y |
| preferred_tech_policy | faq_policy | Y | Y | 1/1 | - | - | Y |
| booking_advance_window | faq_policy | Y | Y | 1/1 | - | - | Y |
| financing_available | faq_policy | Y | Y | 1/1 | - | - | Y |
| warranty_labor | faq_policy | Y | Y | 1/1 | - | - | Y |
| warranty_exclusions | faq_policy | Y | Y | 1/1 | - | - | Y |
| emergency_definition | faq_policy | Y | Y | 1/1 | - | - | Y |
| ev_charger_price | faq_policy | Y | Y | 1/1 | - | - | Y |
| drain_clearing_price | faq_policy | Y | Y | 1/1 | - | - | Y |
| contact_center_sunday | faq_policy | Y | Y | 1/1 | - | - | Y |
| sa_loudoun_plumbing | service_area | - | Y | 1/1 | - | - | Y |
| sa_fairfax_hvac | service_area | - | Y | - | - | - | Y |
| sa_pg_electrical | service_area | - | Y | 1/1 | - | - | Y |
| sa_out_of_area | service_area | - | Y | - | - | - | Y |
| booking_create_rockville | booking_create | - | - | - | Y | Y | Y |
| booking_create_missing_info | booking_create | - | Y | - | - | - | Y |
| booking_create_22046_discrepancy | booking_create | - | - | - | - | - | Y |
| status_check | booking_status | - | Y | - | Y | - | Y |
| tech_eta | booking_status | - | Y | - | Y | - | Y |
| reschedule_far | booking_reschedule | - | - | - | Y | Y | Y |
| cancel_booking | booking_cancel | - | - | - | Y | Y | Y |
| emergency_leak | out_of_scope | - | Y | - | - | - | Y |
| fee_dispute | out_of_scope | - | - | - | - | - | Y |
| complaint | out_of_scope | - | - | - | - | - | Y |
| commercial | out_of_scope | - | - | - | - | - | Y |
| out_of_scope_weather | out_of_scope | - | - | - | - | - | Y |
| out_of_scope_jobs | out_of_scope | - | - | - | - | - | Y |
| smalltalk_greeting | smalltalk | - | Y | - | - | - | Y |
