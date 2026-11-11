"""HTTP client for the TrioFormer API (the only way this tier reaches a model).

Every call returns a ``Result`` that keeps two clocks apart:

* ``server`` -- what the API measured on Render (``latency_ms`` = model compute,
  ``load_ms`` = checkpoint loading during this request, ``total_ms`` = the whole
  request on the server, including any wait for the model lock);
* ``e2e_ms`` -- how long this interface waited, from sending the request to
  having the parsed response: server time + network transit both ways + Render's
  proxy + (when asleep) the free instance waking up.

Failures are classified so the UI can say something useful:

``cold_start``  Render answered with its own 502/503/504 page: the free instance
                is asleep or still booting -- retry soon.
``timeout``     no answer within the time limit: the instance is either waking up
                or overloaded (0.1 CPU) -- retry soon; the UI says which is likelier.
``outage``      no connection at all (DNS, refused, TLS) or still failing after
                the wake-up window -- the service is down.
``bad_input``   422: the API rejected the request; the message says why.
``service``     the API itself answered with an error (e.g. 503 model unavailable).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests

COLD_START_STATUSES = {502, 503, 504}
# Generous on purpose: on the free 0.1-CPU instance a 40-token GPT-2 call has been measured
# anywhere from ~11 s to ~230 s depending on how much CPU Render is granting at the time.
DEFAULT_TIMEOUT_S = 300


@dataclass
class Result:
    ok: bool
    data: dict = field(default_factory=dict)
    e2e_ms: float = 0.0
    kind: Optional[str] = None            # None | cold_start | timeout | outage | bad_input | service
    message: str = ""

    @property
    def server_ms(self) -> Optional[float]:
        return self.data.get("latency_ms")


def _is_render_page(resp: requests.Response) -> bool:
    """Render's proxy returns HTML (not our JSON) while the instance is down or waking."""
    return "application/json" not in resp.headers.get("content-type", "")


def classify(resp: Optional[requests.Response], exc: Optional[Exception]) -> tuple[str, str]:
    """Map a failed HTTP exchange to (kind, user-facing message)."""
    if exc is not None:
        if isinstance(exc, requests.Timeout):
            return "timeout", ("The service did not answer in time. Either it is waking up after "
                               "15 idle minutes, or the small free server is busy with other requests.")
        return "outage", f"Could not reach the service at all ({type(exc).__name__}). It may be down."
    assert resp is not None
    if resp.status_code in COLD_START_STATUSES and _is_render_page(resp):
        return "cold_start", (f"Render answered {resp.status_code} for the sleeping instance: "
                              "the service is starting up.")
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = resp.text[:200]
    if resp.status_code == 422:
        if isinstance(detail, list):   # pydantic validation errors
            detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail)
        return "bad_input", f"The service rejected the input: {detail}"
    return "service", f"The service answered {resp.status_code}: {detail}"


class TrioClient:
    """Thin wrapper over ``requests`` with timing and failure classification.

    ``transport`` is injectable so tests can route calls without a network.
    """

    def __init__(self, base_url: str, transport: Optional[Callable[..., requests.Response]] = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport or requests.request
        self.timeout_s = timeout_s

    def _call(self, method: str, path: str, payload: Optional[dict] = None, timeout: Optional[float] = None) -> Result:
        t0 = time.perf_counter()
        resp, exc = None, None
        try:
            resp = self.transport(method, f"{self.base_url}{path}", json=payload,
                                  timeout=timeout or self.timeout_s)
        except requests.RequestException as e:
            exc = e
        e2e = (time.perf_counter() - t0) * 1000
        if exc is None and resp is not None and resp.ok and not _is_render_page(resp):
            return Result(True, resp.json(), e2e)
        kind, msg = classify(resp, exc)
        return Result(False, {}, e2e, kind, msg)

    def post(self, path: str, payload: dict) -> Result:
        return self._call("POST", path, payload)

    def get(self, path: str, timeout: Optional[float] = None) -> Result:
        return self._call("GET", path, timeout=timeout)

    def health(self, timeout: float = 8) -> Result:
        return self.get("/healthz", timeout=timeout)

    def wait_until_awake(self, budget_s: float = 90, every_s: float = 5,
                         sleep: Callable[[float], None] = time.sleep,
                         on_try: Callable[[int, Result], Any] = lambda i, r: None) -> Result:
        """Poll /healthz until it answers or ``budget_s`` runs out.

        A cold start becomes an ``outage`` only if it never answers within the
        budget; a connection that is refused outright is an outage at once.
        """
        start = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            r = self.health()
            on_try(attempt, r)
            if r.ok or r.kind not in ("cold_start", "timeout"):
                return r
            if (time.perf_counter() - start) + every_s > budget_s:
                return Result(False, {}, r.e2e_ms, "outage",
                              f"The service did not wake up within {budget_s:.0f} s. It may be down; "
                              "try again in a few minutes.")
            sleep(every_s)
