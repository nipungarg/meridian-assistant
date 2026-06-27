# Booking API Specification

Specification for the internal Booking API, implemented by the mock at
`src/meridian/api/mock_booking_api.py`.

- **Base URL:** `https://api.meridian-home.internal/v1`

## Authentication

Bearer token via the `Authorization` header. Tokens are scoped per channel
(`ivr`, `web_chat`, `email`, `agent`).

## POST /bookings — Create a Booking

Creates a confirmed or pending booking. Returns a booking record with a unique `booking_id`.

### Request fields

| Field | Type | Req | Description |
|---|---|---|---|
| `customer_id` | string | Y | Meridian customer ID (`CID-XXXX`). Pass `null` and include `customer_info` if unknown. |
| `customer_info` | object | * | Required when `customer_id` is null. Fields: `name`, `phone`, `email`, `address`. |
| `service_type` | enum | Y | `hvac` / `plumbing` / `electrical` |
| `job_type` | enum | Y | `diagnostic` / `repair` / `install` / `tune_up` / `warranty_return` / `estimate` |
| `zip_code` | string | Y | 5-digit ZIP of service address. Validated against service-area coverage. |
| `preferred_date` | date | Y | ISO 8601 date. Must be within 60 days. |
| `preferred_window` | enum | Y | `morning` (7–11) / `midday` (11–2) / `afternoon` (2–6) / `first_available` |
| `preferred_tech` | string | N | Tech employee ID. System warns if unavailable. |
| `notes` | string | N | Customer-provided problem description, max 500 characters. |
| `channel` | enum | Y | `ivr` / `web_chat` / `email` / `agent` |

### Response fields

| Field | Type | Description |
|---|---|---|
| `booking_id` | string | `BK-XXXXXXXX` format |
| `status` | enum | `confirmed` / `pending_availability` / `out_of_area` |
| `assigned_branch` | string | Branch name or null |
| `appointment_window` | object | `{date, start_time, end_time}` |
| `tech_name` | string | Assigned technician name or null |
| `confirmation_sent` | boolean | True if SMS/email confirmation was dispatched |
| `channel` | enum | Channel the booking was created through (`ivr` / `web_chat` / `email` / `agent`) |

### Example

```http
POST /bookings
```

```json
{
  "customer_id": "CID-8842",
  "service_type": "hvac",
  "job_type": "diagnostic",
  "zip_code": "22032",
  "preferred_date": "2026-01-22",
  "preferred_window": "morning",
  "notes": "AC not cooling - compressor cycling on/off",
  "channel": "web_chat"
}
```

```json
{
  "booking_id": "BK-00512883",
  "status": "confirmed",
  "assigned_branch": "Falls Church (overflow: Tysons)",
  "appointment_window": { "date": "2026-01-22", "start_time": "07:00", "end_time": "11:00" },
  "tech_name": "Marcus Webb",
  "confirmation_sent": true,
  "channel": "web_chat"
}
```

## GET /bookings/{booking_id} — Look Up a Booking

Returns the full booking record. Use to answer status questions, retrieve the appointment
window, and check technician assignment.

### Parameters

| Parameter | In | Req | Description |
|---|---|---|---|
| `booking_id` | path | Y | `BK-XXXXXXXX` from a prior create call or customer record. |
| `customer_id` | query | N | If provided, the system validates ownership before returning PII fields. |

### Response fields

| Field | Type | Description |
|---|---|---|
| `booking_id` | string | `BK-XXXXXXXX` |
| `status` | enum | `confirmed` / `en_route` / `completed` / `cancelled` / `no_show` |
| `service_type` | string | `hvac` / `plumbing` / `electrical` |
| `job_type` | string | See POST /bookings |
| `appointment_window` | object | `{date, start_time, end_time}` |
| `tech_name` | string | Assigned technician name |
| `tech_eta_minutes` | integer | Non-null only when `status = en_route` |
| `notes` | string | Original booking notes |
| `invoice_total` | number | Non-null when `status = completed` |
| `channel` | enum | Channel the booking was created through (`ivr` / `web_chat` / `email` / `agent`) |

### Example

```http
GET /bookings/BK-00483921 → 200 OK
```

```json
{
  "booking_id": "BK-00483921",
  "status": "confirmed",
  "service_type": "plumbing",
  "job_type": "repair",
  "appointment_window": { "date": "2026-01-21", "start_time": "10:00", "end_time": "12:00" },
  "tech_name": "Marcus Webb",
  "tech_eta_minutes": null,
  "notes": "Kitchen sink draining slowly",
  "invoice_total": null,
  "channel": "web_chat"
}
```

## PATCH /bookings/{booking_id} — Reschedule or Cancel

Modifies an existing booking. Handles rescheduling, cancellations, and note updates. The
system enforces cancellation-policy fee logic.

### Request fields

| Field | Type | Req | Description |
|---|---|---|---|
| `booking_id` | path | Y | Booking to modify |
| `action` | enum | Y | `reschedule` / `cancel` / `update_notes` |
| `new_date` | date | * | Required when `action = reschedule` |
| `new_window` | enum | * | Required when `action = reschedule` |
| `cancel_reason` | enum | N | `customer_request` / `tech_unavailable` / `weather` / `duplicate` / `other` |
| `notes` | string | N | Updated notes (used when `action = update_notes`) |

### Response fields

| Field | Type | Description |
|---|---|---|
| `booking_id` | string | |
| `status` | enum | `rescheduled` / `cancelled` |
| `fee_applied` | number | Cancellation fee in USD (0 if waived) |
| `waiver_used` | boolean | True if the first-time no-show waiver was consumed |
| `new_appointment_window` | object | Present when `status = rescheduled` |

### Example

```http
PATCH /bookings/BK-00483921
```

```json
{ "action": "reschedule", "new_date": "2026-01-24", "new_window": "afternoon" }
```

```json
{
  "booking_id": "BK-00483921",
  "status": "rescheduled",
  "fee_applied": 0,
  "waiver_used": false,
  "new_appointment_window": { "date": "2026-01-24", "start_time": "14:00", "end_time": "18:00" }
}
```
