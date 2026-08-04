"""CLI: sample playhead Q&A against live /ask and store results for Grafana.

Usage::

    cowatcher-bench-ask
    cowatcher-bench-ask --title-id "Friends Ross"
    PYTHONPATH=. python -m ai_cowatcher.bench.ask_runner --n 5

Default title name is **Friends Ross** (resolved to ingest ``title_id``,
usually ``friends_ross``). Requires MOCK_MODE=false for real Gemini unless
``--allow-mock``. QA cache stays ON; cache_source is derived from model_name.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from ai_cowatcher.bench.duration import (
    DEFAULT_TITLE_NAME,
    resolve_duration,
    resolve_title_id,
)
from ai_cowatcher.bench.metrics import record_bench_ask
from ai_cowatcher.bench.sampling import parse_cache_source, sample_questions
from ai_cowatcher.bench.store import (
    BenchResultRow,
    append_jsonl,
    ensure_bench_schema,
    insert_bench_result,
)
from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.personas.loader import (
    DEFAULT_PERSONA_ID,
    list_personas,
    normalize_companion_gender,
    resolve_persona_id,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_QUESTIONS = _REPO_ROOT / "benchmarks" / "friends_ross_questions.json"
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "benchmarks" / "results"


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Question bank must be a non-empty JSON array: {path}")
    out: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid question entry (expected object): {row!r}")
        qid = str(row.get("id") or "").strip()
        text = str(row.get("text") or "").strip()
        if not qid or not text:
            raise ValueError(f"Question needs id + text: {row!r}")
        out.append({"id": qid, "text": text, "kind": str(row.get("kind") or "")})
    return out


def _http_error_detail(response: httpx.Response) -> str:
    """Best-effort parse of FastAPI/LiteLLM error body for user-facing logs."""
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        text = (response.text or "").strip().replace("\n", " ")
        return text[:400] if text else response.reason_phrase
    if isinstance(body, dict):
        detail = body.get("detail", body)
        if isinstance(detail, str):
            return detail.replace("\n", " ")[:500]
        return str(detail).replace("\n", " ")[:500]
    return str(body)[:500]


def _is_retryable_ask_status(status_code: int, detail: str) -> bool:
    if status_code in (408, 429, 502, 503, 504):
        return True
    if status_code == 500:
        low = detail.lower()
        # Upstream Gemini capacity / rate limits often land as our 500.
        return any(
            token in low
            for token in (
                "high demand",
                "unavailable",
                "resource_exhausted",
                "rate limit",
                "rate_limit",
                "temporarily",
                "try again",
                "503",
                "429",
                "overloaded",
            )
        )
    return False


def post_ask(
    client: httpx.Client,
    *,
    base_url: str,
    title_id: str,
    question: str,
    current_ts: float,
    user_id: str,
    persona_id: str | None = None,
    companion_gender: str | None = None,
    retries: int = 4,
    retry_base_sec: float = 2.0,
) -> tuple[dict[str, Any], float]:
    """POST /ask; return (response_json, client_latency_ms).

    Retries transient Gemini/API capacity errors (503 high demand, 429, etc.).
    """
    payload: dict[str, Any] = {
        "title_id": title_id,
        "current_ts": current_ts,
        "question": question,
        "user_id": user_id,
        "persona_id": persona_id or DEFAULT_PERSONA_ID,
        "companion_gender": companion_gender or "neutral",
    }
    url = f"{base_url.rstrip('/')}/ask"
    attempts = max(1, retries + 1)
    last_error: Exception | None = None
    started = time.perf_counter()

    for attempt in range(attempts):
        try:
            response = client.post(url, json=payload)
        except httpx.TransportError as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            delay = retry_base_sec * (2**attempt)
            logger.warning(
                "ask transport error (%s); retry %s/%s in %.1fs",
                type(exc).__name__,
                attempt + 1,
                attempts - 1,
                delay,
            )
            time.sleep(delay)
            continue

        if response.is_success:
            latency_ms = (time.perf_counter() - started) * 1000.0
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Unexpected /ask response (not an object)")
            return data, latency_ms

        detail = _http_error_detail(response)
        if attempt + 1 < attempts and _is_retryable_ask_status(response.status_code, detail):
            delay = retry_base_sec * (2**attempt)
            logger.warning(
                "ask HTTP %s (retryable); retry %s/%s in %.1fs — %s",
                response.status_code,
                attempt + 1,
                attempts - 1,
                delay,
                detail[:160],
            )
            time.sleep(delay)
            last_error = httpx.HTTPStatusError(
                f"{response.status_code}: {detail}",
                request=response.request,
                response=response,
            )
            continue

        raise RuntimeError(f"HTTP {response.status_code}: {detail}") from None

    if last_error is not None:
        raise last_error
    raise RuntimeError("ask failed with no response")


def _print_summary(rows: list[BenchResultRow]) -> None:
    if not rows:
        print("No results.")
        return
    headers = ("qid", "persona", "gender", "ts", "cache", "ms", "answer")
    print()
    print(
        f"{headers[0]:<6} {headers[1]:<18} {headers[2]:<8} "
        f"{headers[3]:>8} {headers[4]:<10} {headers[5]:>8} {headers[6]}"
    )
    print("-" * 110)
    for r in rows:
        ans = (r.answer or r.error or "").replace("\n", " ")
        if len(ans) > 48:
            ans = ans[:45] + "..."
        print(
            f"{r.question_id:<6} {(r.persona_id or ''):<18} "
            f"{(r.companion_gender or ''):<8} {r.current_ts:8.1f} "
            f"{r.cache_source:<10} {r.latency_ms:8.0f} {ans}"
        )
    caches = {}
    for r in rows:
        caches[r.cache_source] = caches.get(r.cache_source, 0) + 1
    personas = {}
    for r in rows:
        personas[r.persona_id or ""] = personas.get(r.persona_id or "", 0) + 1
    latencies = [r.latency_ms for r in rows if r.status == "ok"]
    print("-" * 110)
    print(f"cache_source counts: {caches}")
    print(f"persona_id counts: {personas}")
    if latencies:
        print(
            f"latency_ms: min={min(latencies):.0f} "
            f"p50={sorted(latencies)[len(latencies) // 2]:.0f} "
            f"max={max(latencies):.0f}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cowatcher-bench-ask",
        description="Sample playhead questions against /ask and store for Grafana.",
    )
    p.add_argument(
        "--title-id",
        default=DEFAULT_TITLE_NAME,
        help=(
            "Ingested title id or display name "
            f'(default: "{DEFAULT_TITLE_NAME}" → e.g. friends_ross)'
        ),
    )
    p.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    p.add_argument("--n", type=int, default=5, help="Number of questions per run")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help="JSON question bank (default: benchmarks/friends_ross_questions.json)",
    )
    p.add_argument(
        "--allow-mock",
        action="store_true",
        help="Allow MOCK_MODE=true (not for published latency numbers)",
    )
    p.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Video duration override when API/DB/ffprobe unavailable",
    )
    p.add_argument(
        "--user-id",
        default="bench-runner",
        help="user_id sent to /ask (low-impact; memory still optional server-side)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
        help="Directory for bonus JSONL output",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout seconds per /ask",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Retries per question on Gemini capacity / 429/503-style errors",
    )
    p.add_argument(
        "--retry-base-sec",
        type=float,
        default=2.0,
        help="Base backoff seconds (doubles each retry)",
    )
    p.add_argument(
        "--pause-sec",
        type=float,
        default=1.0,
        help="Pause between questions to reduce provider rate spikes",
    )
    p.add_argument(
        "--persona-id",
        default=None,
        help=(
            "Companion persona for this run "
            f"(default: Settings.DEFAULT_PERSONA_ID / {DEFAULT_PERSONA_ID}). "
            f"Known: {', '.join(p.persona_id for p in list_personas()) or DEFAULT_PERSONA_ID}"
        ),
    )
    p.add_argument(
        "--companion-gender",
        default="neutral",
        choices=["male", "female", "neutral"],
        help="Companion / TTS gender hint (default: neutral)",
    )
    p.add_argument(
        "--all-personas",
        action="store_true",
        help=(
            "Run the same n samples once per shipped persona "
            "(for tone comparison / cache isolation)"
        ),
    )
    p.add_argument(
        "--skip-postgres",
        action="store_true",
        help="Skip Postgres insert (JSONL only; not recommended for Grafana)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    settings: Settings = get_settings()
    if settings.mock_mode and not args.allow_mock:
        print(
            "ERROR: MOCK_MODE=true. Bench is meant for real Gemini "
            "(set MOCK_MODE=false in .env / environment). "
            "Pass --allow-mock only for local smoke tests.",
            file=sys.stderr,
        )
        return 2

    if settings.mock_mode and args.allow_mock:
        logger.warning("Running with MOCK_MODE=true (--allow-mock); numbers are not real.")

    questions_path = args.questions_file or _DEFAULT_QUESTIONS
    if not questions_path.is_file():
        print(f"ERROR: questions file not found: {questions_path}", file=sys.stderr)
        return 2
    questions = load_questions(questions_path)

    run_id = uuid.uuid4().hex[:12]
    session_factory = None
    duration_sec: float
    title_ref = (args.title_id or DEFAULT_TITLE_NAME).strip()
    title_id = title_ref  # may be overridden after DB resolve

    try:
        if not args.skip_postgres or args.duration_sec is None:
            session_factory = ensure_bench_schema(settings)
            with session_factory() as session:
                title_id = resolve_title_id(session, title_ref)
                duration_sec = resolve_duration(
                    session,
                    title_id,
                    override_sec=args.duration_sec,
                )
        else:
            # No DB: slug fallback so "Friends Ross" → friends_ross for duration override path
            if " " in title_ref and args.duration_sec is not None:
                title_id = title_ref.lower().replace(" ", "_")
            title_id = title_id or title_ref
            duration_sec = resolve_duration(
                None,
                title_id,
                override_sec=args.duration_sec,
            )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    samples = sample_questions(
        questions,
        n=args.n,
        duration_sec=duration_sec,
        seed=args.seed,
    )

    default_persona = getattr(settings, "default_persona_id", None) or DEFAULT_PERSONA_ID
    if args.all_personas:
        persona_ids = [p.persona_id for p in list_personas()] or [default_persona]
    else:
        persona_ids = [
            resolve_persona_id(args.persona_id, default_id=default_persona)
        ]
    gender = normalize_companion_gender(args.companion_gender) or "neutral"

    print(
        f"run_id={run_id} title={title_ref!r} title_id={title_id} n={len(samples)} "
        f"personas={persona_ids} companion_gender={gender} "
        f"duration_sec={duration_sec:.1f} mock_mode={settings.mock_mode} "
        f"qa_cache_enabled={getattr(settings, 'qa_cache_enabled', False)}"
    )
    print(f"base_url={args.base_url} (questions: {questions_path})")

    jsonl_path = Path(args.results_dir) / f"{run_id}.jsonl"
    rows: list[BenchResultRow] = []
    ask_index = 0

    with httpx.Client(timeout=args.timeout) as client:
        for persona_id in persona_ids:
            for sample in samples:
                if ask_index > 0 and args.pause_sec > 0:
                    time.sleep(args.pause_sec)
                ask_index += 1
                qid = sample["id"]
                qtext = sample["text"]
                ts = float(sample["current_ts"])
                kind = sample.get("kind") or ""
                try:
                    data, latency_ms = post_ask(
                        client,
                        base_url=args.base_url,
                        title_id=title_id,
                        question=qtext,
                        current_ts=ts,
                        user_id=args.user_id,
                        persona_id=persona_id,
                        companion_gender=gender,
                        retries=args.retries,
                        retry_base_sec=args.retry_base_sec,
                    )
                    model_name = str(data.get("model_name") or "")
                    model_tier = str(data.get("model_tier") or "")
                    escalation = str(data.get("escalation_reason") or "")
                    cache_source = parse_cache_source(
                        model_name,
                        model_tier=model_tier,
                        escalation_reason=escalation,
                    )
                    answer = str(data.get("answer") or "")
                    skip_memory = bool(data.get("skip_memory", False))
                    used_context = cache_source in ("exact", "semantic")
                    row = BenchResultRow(
                        run_id=run_id,
                        title_id=title_id,
                        question_id=qid,
                        question=qtext,
                        current_ts=ts,
                        answer=answer,
                        latency_ms=round(latency_ms, 2),
                        cache_source=cache_source,
                        model_name=model_name,
                        model_tier=model_tier,
                        persona_id=persona_id,
                        companion_gender=gender,
                        used_context=used_context,
                        skip_memory=skip_memory,
                        kind=kind or None,
                        status="ok",
                    )
                except Exception as exc:  # noqa: BLE001 — record and continue
                    msg = str(exc).replace("\n", " ")
                    logger.error(
                        "ask failed for %s persona=%s: %s",
                        qid,
                        persona_id,
                        msg[:300],
                    )
                    row = BenchResultRow(
                        run_id=run_id,
                        title_id=title_id,
                        question_id=qid,
                        question=qtext,
                        current_ts=ts,
                        answer="",
                        latency_ms=-1.0,
                        cache_source="none",
                        model_name="",
                        model_tier="",
                        persona_id=persona_id,
                        companion_gender=gender,
                        kind=kind or None,
                        status="error",
                        error=f"{type(exc).__name__}: {msg[:800]}",
                    )

                rows.append(row)
                record_bench_ask(
                    title_id=row.title_id,
                    question_id=row.question_id,
                    cache_source=row.cache_source,
                    latency_ms=max(0.0, row.latency_ms),
                    persona_id=row.persona_id,
                )
                append_jsonl(jsonl_path, row)

                if not args.skip_postgres and session_factory is not None:
                    with session_factory() as session:
                        insert_bench_result(session, row)

    _print_summary(rows)
    print(f"\nJSONL: {jsonl_path}")
    if not args.skip_postgres:
        print(f"Postgres: bench_ask_result run_id={run_id}")
    print("Grafana: dashboard 'Ask Bench' (folder AI Co-watcher)")
    return 0 if all(r.status == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
