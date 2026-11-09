"""One Anthropic client for every judge in the repository.

The key comes from the environment (`ANTHROPIC_API_KEY`), as the Anthropic SDK expects.
The SDK's own retry of rate-limit and server errors is the only
retry of a synchronous call (the Message Batches path, `hl.judge.batch`, re-sends its failed items here); a call that still fails is logged and scored as "no score" (None), which every rate in
this repository counts as *not* misaligned / *not* coherent / *not* persona-positive.
"""

from __future__ import annotations

import inspect
import logging
import os
import re

log = logging.getLogger("hl.judge.client")


def make_client():
    """Return `anthropic.Anthropic()`. Imported lazily so that dry-runs need no SDK and no key.

    `HL_JUDGE_MAX_RETRIES` (optional) raises the SDK's retry count for rate-limit / server errors (SDK default 2),
    so that a burst of 429s under many parallel judge jobs does not turn into unscored answers. It changes no request.
    """
    from anthropic import Anthropic

    retries = os.environ.get("HL_JUDGE_MAX_RETRIES")
    return Anthropic(max_retries=int(retries)) if retries else Anthropic()


def make_judge_client(model: str):
    """Construct a client AND smoke-test it with one call to `model`.

    Returns the client if the probe succeeds, else None, so callers can tell "judge unavailable"
    (a rate of 0 would be an artefact; judge later) from a genuine all-aligned result.
    Responses are always saved before judging, so judging can be
    re-run later with `python -m hl.judge`.
    """
    try:
        c = make_client()
        probe = c.messages.create(
            model=model,
            max_tokens=8,
            messages=[{"role": "user", "content": "Reply with: ok"}],
        )
        text = getattr(probe.content[0], "text", "") if probe.content else ""
        if "ok" in text.lower():
            log.info("judge client: probe OK (%s)", model)
            return c
        log.warning("judge client probe returned unexpected: %r", probe.content)
    except Exception as e:
        log.warning("judge client unavailable (probe failed): %s", e)
    return None


def reply_text(msg) -> str:
    """Concatenate the text blocks of an SDK message."""
    return "".join(
        getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text"
    )


def request_params(model: str, prompt: str, max_tokens: int, temperature: float | None = 0) -> dict:
    """The Messages API parameters of one judge call. The synchronous path (`complete`) and the
    Message Batches path (`hl.judge.batch`) both build their requests here, so they send the same request."""
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs


def parse_score(text: str, regex: re.Pattern) -> int | None:
    """The first group of `regex` in a reply as an int, or None if the reply has no score tag."""
    m = regex.search(text)
    return int(m.group(1)) if m else None


SAMPLING_KEYS = ("temperature", "top_p", "top_k")


def sdk_kwargs(create, params: dict) -> dict:
    """Keyword arguments for `create` that send exactly `params` as the request body.

    anthropic 1.x removed `temperature`/`top_p`/`top_k` from the `messages.create()` signature (passing one is a
    TypeError), although the API still accepts them for the models this repository judges with (e.g.
    claude-sonnet-4-6). For such an SDK the sampling keys go through `extra_body`, which the SDK merges into the
    request JSON unchanged, so the body is the same under anthropic 0.x and 1.x. A `create` that accepts the keys
    (anthropic 0.x, test stubs) gets them as plain keyword arguments.
    """
    keys = [k for k in SAMPLING_KEYS if k in params]
    if not keys:
        return params
    try:
        ps = inspect.signature(create).parameters
    except (TypeError, ValueError):
        return params
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ps.values()) or all(k in ps for k in keys):
        return params
    out = {k: v for k, v in params.items() if k not in keys}
    out["extra_body"] = {**(params.get("extra_body") or {}), **{k: params[k] for k in keys}}
    return out


def create_message(client, params: dict):
    """`client.messages.create` with the request body `params` (see `sdk_kwargs`)."""
    return client.messages.create(**sdk_kwargs(client.messages.create, params))


def complete(client, model: str, prompt: str, max_tokens: int, temperature: float | None = 0) -> str:
    """One user-turn completion; returns the reply text. Raises on SDK errors."""
    return reply_text(create_message(client, request_params(model, prompt, max_tokens, temperature)))


def call(
    client,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float | None,
    regex: re.Pattern,
) -> int | None:
    """Score one prompt: the first group of `regex` in the reply as an int, or None.

    None covers both an SDK error and a reply without a parsable score.
    """
    try:
        return parse_score(complete(client, model, prompt, max_tokens, temperature), regex)
    except Exception as e:
        log.warning("judge call error: %s", e)
        return None
