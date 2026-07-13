# 5. Reliability

## The failure modes unique to streaming

A streaming chat system fails differently from a stateless request system.
A request either succeeds or fails. A stream can succeed partway and then die,
leaving the client with a partial reply. That partial state creates three
problems that do not exist in non-streaming systems:

1. **Partial output.** The user received the first half of a reply, the
   connection dropped, and the session store may or may not have recorded the
   partial generation as the assistant's turn.

2. **Duplicate generation.** The client did not see the reply, retries the
   request, and the server generates again. If the first generation was recorded
   in the session store, the retry appends a second identical reply. If it was
   not recorded, the client sees an inconsistent view.

3. **Orphaned slots.** The client dropped, but the server did not detect it and
   kept generating, burning a slot for minutes.

Each of these has a concrete fix.

## Reconnect and session resumability

Design reconnect around two invariants:

**Invariant 1: the session store is the source of truth.** The gateway only
writes a completed assistant turn to the session store once generation finishes
(or once it reaches a safe checkpoint). It does not write partial output.

**Invariant 2: the client tracks the last event sequence number it received.**
Each SSE event carries a monotonic sequence number in the `id:` field (this is
standard SSE behavior; the browser's `EventSource` sends a `Last-Event-ID`
header on reconnect). The gateway uses this to resume from the right position
in the token stream.

On reconnect:

- If the generation is still in flight on the same replica, the gateway resumes
  streaming from `Last-Event-ID + 1`.
- If the generation completed, the gateway reads the reply from the session
  store and streams it from the appropriate offset.
- If the generation was orphaned (the client dropped, the slot was freed), the
  client gets a "generation was cancelled" signal and can re-submit.

This design means a brief network drop does not force a full retry. The user
gets a smooth reconnect, the generation is not duplicated, and the slot is
freed promptly either way.

## Idempotency

When the client does need to retry (the generation truly was lost), the retry
must be idempotent. The mechanism is a client-generated request id passed on
every attempt. The gateway deduplicates on this id: if a generation for this
request id is already recorded in the session store, it returns the stored
result rather than generating again.

This is the same pattern as idempotency keys in payment APIs. The window only
needs to cover the retry window (typically a few seconds to a few minutes).
A short TTL (time-to-live) on the dedup store is sufficient.

## Handling partial output in the session store

There are two schools of thought on partial output:

**Checkpoint-at-sentence-boundaries.** The gateway writes the partial reply to
the session store every time a sentence boundary is detected (a period, a
newline). If the connection drops, the session store has up to the last complete
sentence. On reconnect, the client gets a clean reply up to that point.

**Write-only-on-completion.** The gateway writes to the session store only when
generation finishes. Partial drops result in a lost generation, which the client
retries. Simpler to implement and reason about; the risk is that a very long
generation (several paragraphs, a code block) may lose the whole thing on a
late-in-generation drop.

For most chat products, write-only-on-completion is the right default. The
generations are short enough that the cost of a retry is low, and the
simplicity is worth it. Checkpoint-at-sentence-boundaries makes sense for
long-form generation products where a single reply may run for tens of seconds.

## Failure mode summary

| Failure mode | Cause | Detection | Fix |
|---|---|---|---|
| Orphaned generation | Client drops without explicit cancel | Write error on next token; heartbeat timeout | Abort generation, free slot |
| Partial output delivered | Connection drops mid-generation | `Last-Event-ID` on reconnect | Resume from last-seen event id |
| Duplicate generation on retry | Client retries without idempotency | No dedup key | Client-side request id, gateway dedup |
| Stale partial in session store | Gateway writes partial on interrupt | No completion flag | Only write to store on clean completion |
| Cache miss on reconnect | Different replica handles the retry | Cold prefill (performance, not correctness) | Accept cold prefill as fallback; prioritize the same replica |
