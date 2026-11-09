"""Alignment + coherence judging through the Anthropic Message Batches API.

Same requests as the synchronous path (`hl.judge.alignment.judge_parallel`): every request is built by
`hl.judge.client.request_params` from the same prompt bodies and settings (model, max_tokens, temperature),
and every reply is scored by the same regexes (`hl.judge.client.parse_score`). Only the transport differs:

1. all (answer x {alignment, coherence}) requests are submitted in as few batches as the limits allow
   (100,000 requests / 256 MB per batch; we stay under `max_batch_bytes`, 200 MB by default);
2. the batches are polled until they have ended, or until `timeout_s` (default 2 h) has passed since
   submission, in which case the batches that are still running are cancelled;
3. the results are fetched and mapped back by `custom_id`. A succeeded item is scored like a synchronous
   reply (no parsable score tag -> None, not retried, exactly as in the synchronous path);
4. every item without a batch result (errored, expired, cancelled, a failed submission or download, or
   the time-out) is judged synchronously with the existing worker pool (`hl.judge.client.call`, which
   returns None after the SDK's own retries).

Batch results are fetched through the client's base URL (ANTHROPIC_BASE_URL). The API's `results_url` is an
absolute URL on api.anthropic.com, and the SDK's `batches.results()` follows it as is, which bypasses a
proxy given in ANTHROPIC_BASE_URL; `_iter_results` requests the same path relative to the base URL instead.

With `state_path`, the submitted batch ids are recorded in a JSON file next to the outputs. A re-launch
after a crash (state `complete: false`, same requests) re-attaches to those batches instead of paying for
new ones.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from hl.judge import client as _client

log = logging.getLogger("hl.judge.batch")

MAX_BATCH_REQUESTS = 100_000  # API limit per batch
MAX_BATCH_BYTES = 200_000_000  # API limit is 256 MB per batch; keep a margin for the envelope


@dataclass(frozen=True)
class BatchOptions:
    timeout_s: float = 7200.0  # cancel the batches and judge the rest synchronously after this long
    poll_s: float = 30.0
    cancel_grace_s: float = 600.0  # after cancelling, wait this long for the batch to end and keep its results
    max_batch_requests: int = MAX_BATCH_REQUESTS
    max_batch_bytes: int = MAX_BATCH_BYTES


def _custom_id(i: int, kind: str) -> str:
    # ^[a-zA-Z0-9_-]{1,64}$
    return f"r{i:07d}-{kind}"


def build_requests(pairs: list[tuple[str, str]], settings) -> list[tuple[str, dict, object]]:
    """(custom_id, params, regex) for every (pair x {alignment, coherence}), in the synchronous path's order."""
    from hl.judge import alignment as al

    out = []
    for i, (q, a) in enumerate(pairs):
        transcript = f"User: {q}\nAI: {a}"
        for kind, body, regex in (("al", al.grader_body(), al.SCORE_RE), ("coh", al.coherence_body(settings), al.COHERENCE_RE)):
            prompt = body.format(transcript=transcript)
            out.append((_custom_id(i, kind),
                        _client.request_params(settings.model, prompt, settings.max_tokens, settings.temperature),
                        regex))
    return out


def chunk_requests(reqs: list[dict], max_requests: int, max_bytes: int) -> list[list[dict]]:
    """Split batch requests ({custom_id, params}) into consecutive chunks within both limits."""
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    size = 0
    for r in reqs:
        n = len(json.dumps(r).encode()) + 2
        if cur and (len(cur) >= max_requests or size + n > max_bytes):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += n
    if cur:
        chunks.append(cur)
    return chunks


def _fingerprint(reqs: list[dict]) -> str:
    h = hashlib.sha256()
    for r in reqs:
        h.update(json.dumps(r, sort_keys=True).encode())
        h.update(b"\n")
    return h.hexdigest()


def _iter_results(client, batch_id: str):
    """The batch's result entries, fetched relative to the client's base URL (see the module docstring)."""
    base = getattr(client, "base_url", None)
    url = client.messages.batches.retrieve(batch_id).results_url
    if base is not None and url and urlsplit(url).netloc and urlsplit(url).netloc != urlsplit(str(base)).netloc:
        from anthropic._decoders.jsonl import JSONLDecoder
        from anthropic.types.messages import MessageBatchIndividualResponse

        return client.get(urlsplit(url).path, cast_to=JSONLDecoder[MessageBatchIndividualResponse],
                          options={"headers": {"Accept": "application/binary"}}, stream=True)
    return client.messages.batches.results(batch_id)


def _write_state(path: Path | None, state: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def _sync_score(client, settings, params: dict, regex) -> int | None:
    prompt = params["messages"][0]["content"]
    return _client.call(client, settings.model, prompt, settings.max_tokens, settings.temperature, regex)


def judge_batch(
    pairs: list[tuple[str, str]],
    client,
    settings,
    workers: int = 16,
    options: BatchOptions | None = None,
    state_path: Path | None = None,
    sleep=time.sleep,
    clock=time.monotonic,
) -> tuple[list[tuple[int | None, int | None]], dict]:
    """Two judge calls (alignment, coherence) per (question, answer) pair via the Message Batches API.

    Returns (scores in the order of `pairs`, info) with the same None semantics as
    `hl.judge.alignment.judge_responses`. `info` summarises the run (batch ids, how many items came from
    the batch and how many were judged synchronously, token usage of the batch items, timings).
    """
    built = build_requests(pairs, settings)
    scores, info = run_batch(built, client, settings, workers=workers, options=options, state_path=state_path,
                             sleep=sleep, clock=clock)
    if not built:
        return [], info
    return [(scores[_custom_id(i, "al")], scores[_custom_id(i, "coh")]) for i in range(len(pairs))], info


def run_batch(
    built: list[tuple[str, dict, object]],
    client,
    settings,
    workers: int = 16,
    options: BatchOptions | None = None,
    state_path: Path | None = None,
    sleep=time.sleep,
    clock=time.monotonic,
) -> tuple[dict[str, int | None], dict]:
    """The transport of `judge_batch` for any list of (custom_id, request params, score regex): submit, poll,
    fetch, fall back to synchronous calls for the rest (see the module docstring). Returns ({custom_id: score or
    None}, info). `settings` needs model, max_tokens and temperature (for the synchronous fallback)."""
    o = options or BatchOptions()
    regex_of = {cid: rx for cid, _, rx in built}
    params_of = {cid: p for cid, p, _ in built}
    reqs = [{"custom_id": cid, "params": p} for cid, p, _ in built]
    fp = _fingerprint(reqs)
    scores: dict[str, int | None] = {}
    info: dict = {"n_requests": len(reqs), "options": asdict(o), "batch_ids": [], "timed_out": False,
                  "n_from_batch": 0, "n_unparsed_from_batch": 0, "batch_result_types": {},
                  "n_sync": 0, "usage": {"input_tokens": 0, "output_tokens": 0}}
    if not reqs:
        return {}, info

    # submit (or re-attach to the batches of an interrupted launch with the same requests)
    batch_ids: list[str] = []
    t_start = clock()
    elapsed_before = 0.0
    if state_path is not None and Path(state_path).exists():
        try:
            st = json.loads(Path(state_path).read_text())
            if st.get("fingerprint") == fp and not st.get("complete") and st.get("batch_ids"):
                batch_ids = list(st["batch_ids"])
                elapsed_before = float(st.get("elapsed_s", 0.0))
                log.info("re-attaching to %d batch(es) of an interrupted launch: %s", len(batch_ids), batch_ids)
        except (OSError, ValueError) as e:
            log.warning("ignoring unreadable batch state %s: %s", state_path, e)
    if not batch_ids:
        chunks = chunk_requests(reqs, o.max_batch_requests, o.max_batch_bytes)
        for k, chunk in enumerate(chunks):
            try:
                b = client.messages.batches.create(requests=chunk)
                batch_ids.append(b.id)
                log.info("submitted batch %d/%d: %s (%d requests)", k + 1, len(chunks), b.id, len(chunk))
            except Exception as e:  # these items are judged synchronously below
                log.warning("batch submission %d/%d failed (%d requests go to the synchronous path): %s",
                            k + 1, len(chunks), len(chunk), e)
    info["batch_ids"] = batch_ids
    state = {"fingerprint": fp, "batch_ids": batch_ids, "n_requests": len(reqs), "complete": False,
             "elapsed_s": elapsed_before}
    _write_state(state_path, state)

    # poll
    ended: set[str] = set()
    last_log = -1e9
    while batch_ids and len(ended) < len(batch_ids):
        elapsed = elapsed_before + clock() - t_start
        if elapsed >= o.timeout_s:
            break
        for bid in batch_ids:
            if bid in ended:
                continue
            try:
                b = client.messages.batches.retrieve(bid)
            except Exception as e:
                log.warning("retrieving batch %s failed (will retry): %s", bid, e)
                continue
            if b.processing_status == "ended":
                ended.add(bid)
                log.info("batch %s ended: %s", bid, b.request_counts)
            elif clock() - last_log > 300:
                log.info("batch %s %s: %s (%.0f s elapsed)", bid, b.processing_status, b.request_counts, elapsed)
        if clock() - last_log > 300:
            last_log = clock()
        state["elapsed_s"] = elapsed_before + clock() - t_start
        _write_state(state_path, state)
        if len(ended) < len(batch_ids):
            sleep(o.poll_s)

    # time-out: cancel what is still running, keep whatever it has finished
    pending_batches = [bid for bid in batch_ids if bid not in ended]
    if pending_batches:
        info["timed_out"] = True
        log.warning("batch time-out after %.0f s: cancelling %s; their unfinished items are judged synchronously",
                    o.timeout_s, pending_batches)
        for bid in pending_batches:
            try:
                client.messages.batches.cancel(bid)
            except Exception as e:
                log.warning("cancelling batch %s failed: %s", bid, e)
        t_cancel = clock()
        while pending_batches and clock() - t_cancel < o.cancel_grace_s:
            for bid in list(pending_batches):
                try:
                    if client.messages.batches.retrieve(bid).processing_status == "ended":
                        ended.add(bid)
                        pending_batches.remove(bid)
                except Exception as e:
                    log.warning("retrieving batch %s failed: %s", bid, e)
            if pending_batches:
                sleep(min(o.poll_s, 30.0))
    info["batch_wall_s"] = round(elapsed_before + clock() - t_start, 1)

    # results, mapped back by custom_id
    types: dict[str, int] = {}
    for bid in batch_ids:
        if bid not in ended:
            continue
        try:
            for entry in _iter_results(client, bid):
                cid = entry.custom_id
                if cid not in regex_of:
                    continue
                rtype = entry.result.type
                types[rtype] = types.get(rtype, 0) + 1
                if rtype != "succeeded":
                    continue
                msg = entry.result.message
                usage = getattr(msg, "usage", None)
                if usage is not None:
                    info["usage"]["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
                    info["usage"]["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
                sc = _client.parse_score(_client.reply_text(msg), regex_of[cid])
                scores[cid] = sc
                if sc is None:
                    info["n_unparsed_from_batch"] += 1
        except Exception as e:  # items not read here are judged synchronously
            log.warning("reading the results of batch %s failed: %s", bid, e)
    info["batch_result_types"] = types
    info["n_from_batch"] = len(scores)

    # everything else: the synchronous path
    todo = [cid for cid, _, _ in built if cid not in scores]
    info["n_sync"] = len(todo)
    if todo:
        log.info("judging %d item(s) synchronously (%d workers)", len(todo), workers)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            got = list(ex.map(lambda cid: _sync_score(client, settings, params_of[cid], regex_of[cid]), todo))
        scores.update(zip(todo, got, strict=True))
    info["n_sync_none"] = sum(1 for cid in todo if scores[cid] is None)
    info["wall_s"] = round(elapsed_before + clock() - t_start, 1)
    state.update(complete=True, elapsed_s=info["wall_s"])
    _write_state(state_path, state)
    log.info("batch judge done: %d from batches (%s), %d synchronous, %.0f s",
             info["n_from_batch"], types, info["n_sync"], info["wall_s"])
    return scores, info
