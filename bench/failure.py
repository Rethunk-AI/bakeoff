"""Failure-reason taxonomy for the bakeoff harness (refs: Rethunk-AI/bakeoff#23).

Controlled enum of nine failure codes that replace the unstructured ``error``
string in result records.  Every failed cell in ``result.json`` carries one of
these codes in ``failure_code``; successful cells carry ``null``.

Taxonomy codes
--------------
timeout        Model did not respond within the configured timeout.
refusal        Model responded but explicitly declined (safety/topic rejection).
malformed_output  Response received but could not be parsed or scored.
oom            Out-of-memory signal during inference (VRAM/RAM exhaustion).
load_failure   Model could not be loaded at all (crash, missing file, bad quant).
capability_gap Applied post-hoc (not by classify); systematic inability across cells.
infra_error    Runner-side infrastructure failure (network, proxy, etc.).
cancelled      Cell explicitly cancelled (operator interrupt / queue CANCELLED).
unknown        Exception caught but matches no pattern above.

Detection regexes used by ``classify``
---------------------------------------
All patterns are case-insensitive (``re.IGNORECASE``).

cancelled:
    /cancelled|canceled|interrupt|KeyboardInterrupt/

timeout:
    /timed?\\s*out|timeout/
    also: httpx.ReadTimeout, httpx.TimeoutException types

oom:
    /out of memory|oom|cuda.*memory|vram|alloc.*fail/

load_failure:
    /failed to load|load.*model|no such file|cannot load|incompatible|swap.*fail/

infra_error:
    /connection refused|proxy|network|econnrefused/
    also: httpx.ConnectError type

refusal (checked via is_refusal()):
    /\\b(I (cannot|can't|won't|am unable|am not able)|I'm sorry|as an AI|I do not feel comfortable)\\b/

malformed_output:
    /malformed|could not parse|unparseable|invalid json/

Precedence order (highest → lowest):
    cancelled → timeout → oom → load_failure → infra_error → refusal
    → malformed_output → unknown

Note: ``capability_gap`` is computed post-hoc in scoring and is never returned
by ``classify``.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

FAILURE_CODES: tuple[str, ...] = (
    "timeout",
    "refusal",
    "malformed_output",
    "oom",
    "load_failure",
    "capability_gap",
    "infra_error",
    "cancelled",
    "unknown",
)

# ---------------------------------------------------------------------------
# Internal compiled patterns
# ---------------------------------------------------------------------------

_RE_CANCELLED = re.compile(r"cancelled|canceled|interrupt|KeyboardInterrupt", re.IGNORECASE)
_RE_TIMEOUT = re.compile(r"timed?\s*out|timeout", re.IGNORECASE)
_RE_OOM = re.compile(r"out of memory|oom|cuda.*memory|vram|alloc.*fail", re.IGNORECASE)
_RE_LOAD_FAILURE = re.compile(
    r"failed to load|load.*model|no such file|cannot load|incompatible|swap.*fail",
    re.IGNORECASE,
)
_RE_INFRA = re.compile(r"connection refused|proxy|network|econnrefused", re.IGNORECASE)
_RE_REFUSAL = re.compile(
    r"\b(I (cannot|can't|won't|am unable|am not able)|I'm sorry|as an AI"
    r"|I do not feel comfortable)\b",
    re.IGNORECASE,
)
_RE_MALFORMED = re.compile(r"malformed|could not parse|unparseable|invalid json", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_refusal(text: str | None) -> bool:
    """Return True when *text* contains a refusal marker.

    Used by ``classify`` and available to callers that need to check
    response text independently.
    """
    if not text:
        return False
    return bool(_RE_REFUSAL.search(text))


def classify(
    exc: BaseException | None = None,
    *,
    message: str | None = None,
    status: int | None = None,
    response_text: str | None = None,
) -> str:
    """Classify a failure into one taxonomy code.

    Parameters
    ----------
    exc:
        The caught exception (or None).
    message:
        Free-text message to match against; typically ``str(exc)`` or an
        explicit error string from the runner.  When *exc* is provided and
        *message* is None, ``str(exc)`` is used automatically.
    status:
        HTTP status code returned by the proxy (reserved for future use;
        not currently matched against any code).
    response_text:
        The raw response body, used to detect refusals.

    Returns
    -------
    str
        One of the nine taxonomy codes.  ``capability_gap`` is never returned
        here; it is assigned post-hoc during scoring aggregation.

    Precedence
    ----------
    cancelled → timeout → oom → load_failure → infra_error → refusal
    → malformed_output → unknown
    """
    # Resolve the search text: prefer explicit message, fall back to str(exc).
    text: str = message or (str(exc) if exc is not None else "")
    exc_type = type(exc) if exc is not None else None

    # 1. cancelled
    if exc_type is KeyboardInterrupt or (text and _RE_CANCELLED.search(text)):
        return "cancelled"

    # 2. timeout
    _httpx_timeout_types: tuple[type, ...] = _get_httpx_timeout_types()
    if (exc_type is not None and issubclass(exc_type, _httpx_timeout_types)) or (
        text and _RE_TIMEOUT.search(text)
    ):
        return "timeout"

    # 3. oom
    if text and _RE_OOM.search(text):
        return "oom"

    # 4. load_failure
    if text and _RE_LOAD_FAILURE.search(text):
        return "load_failure"

    # 5. infra_error
    _httpx_connect_types: tuple[type, ...] = _get_httpx_connect_types()
    if (exc_type is not None and issubclass(exc_type, _httpx_connect_types)) or (
        text and _RE_INFRA.search(text)
    ):
        return "infra_error"

    # 6. refusal (only when response_text is non-empty)
    if response_text and is_refusal(response_text):
        return "refusal"

    # 7. malformed_output
    if text and _RE_MALFORMED.search(text):
        return "malformed_output"

    # 8. unknown — something was raised/provided but matched nothing
    if exc is not None or text:
        return "unknown"

    # No information at all — still unknown
    return "unknown"


def dominant_failure_code(codes: list[str]) -> str | None:
    """Return the most frequent failure code in *codes*.

    Parameters
    ----------
    codes:
        A list of taxonomy code strings (may be empty).

    Returns
    -------
    str | None
        The most frequent code, or ``None`` when *codes* is empty.
        Ties are broken by the order in ``FAILURE_CODES`` (earlier = higher
        priority).
    """
    if not codes:
        return None

    # Count occurrences.
    counts: dict[str, int] = {}
    for c in codes:
        counts[c] = counts.get(c, 0) + 1

    max_count = max(counts.values())

    # Among codes with the maximum count, pick the one earliest in FAILURE_CODES.
    for code in FAILURE_CODES:
        if counts.get(code, 0) == max_count:
            return code

    # Fallback: return the first code with max count (handles codes not in FAILURE_CODES).
    return next(c for c, n in counts.items() if n == max_count)


# ---------------------------------------------------------------------------
# Internal helpers for optional httpx import
# ---------------------------------------------------------------------------


def _get_httpx_timeout_types() -> tuple[type, ...]:
    """Return httpx timeout exception types, or empty tuple if httpx unavailable."""
    try:
        import httpx  # noqa: PLC0415

        return (httpx.ReadTimeout, httpx.TimeoutException)
    except ImportError:
        return ()


def _get_httpx_connect_types() -> tuple[type, ...]:
    """Return httpx connect exception types, or empty tuple if httpx unavailable."""
    try:
        import httpx  # noqa: PLC0415

        return (httpx.ConnectError,)
    except ImportError:
        return ()
