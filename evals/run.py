"""Run the evaluation suite.

  .venv/bin/python -m evals.run --suite dry     # 3 calls: check everything works, measure cost
  .venv/bin/python -m evals.run --suite smoke   # 20 calls (what CI runs on every pull request)
  .venv/bin/python -m evals.run --suite full    # all 100 calls

Options:
  --concurrency N     calls in parallel (default 6; each call runs in real time)
  --check-baseline    exit with an error if results regress vs evals/baseline-<suite>.json (CI uses this)
  --write-baseline    save this run's metrics as the new baseline

Needs OPENAI_API_KEY and DEEPGRAM_API_KEY in .env. Makes no phone calls and
sends no texts; it only uses those two APIs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

from evals import report
from evals.caller import LLMCaller
from evals.harness import EvalActions, run_scenario
from evals.scenarios import SCENARIO_FILE, generate, load, save, smoke_subset
from hungercue.llm.classify import make_classifier
from hungercue.audio.vad import SileroVAD, VadSegmenter, load_session
from hungercue.config import Settings
from hungercue.db.repo import Repo
from hungercue.db.session import Database
from hungercue.pipeline.call_session import Services
from hungercue.services import missing_call_keys
from hungercue.state.session_store import SessionStore, make_redis
from hungercue.stt.deepgram import DeepgramSTT
from hungercue.timeutil import now_local
from hungercue.tts.aura import AuraTTS

EVAL_DIR = Path(__file__).parent


def baseline_path(suite: str) -> Path:
    """One baseline per suite: a smoke run must be compared with a smoke run (same 20 calls)."""
    return EVAL_DIR / f"baseline-{suite}.json"


def pick(suite: str) -> list:
    if not SCENARIO_FILE.exists():
        save(generate())
    scenarios = load()
    if suite == "dry":
        # One of each caller type, so the dry run touches every scoring path.
        return [next(s for s in scenarios if s.caller_type == t) for t in ("restaurant", "shelter", "volunteer")]
    if suite == "smoke":
        return smoke_subset(scenarios)
    return scenarios


async def main(args: argparse.Namespace) -> int:
    settings = Settings()
    missing = missing_call_keys(settings)
    if missing:
        print(f"Missing in .env: {', '.join(missing)}")
        return 2
    stamp = now_local(settings.timezone).strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else EVAL_DIR / "results" / f"{args.suite}-{stamp}"
    raw = EVAL_DIR / "results" / "raw" / f"{args.suite}-{stamp}"
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    from openai import AsyncOpenAI

    db = Database(f"sqlite+aiosqlite:///{raw / 'eval.db'}")
    await db.create_tables()
    repo = Repo(db)
    store = SessionStore(make_redis(""))
    openai = AsyncOpenAI(api_key=settings.openai_api_key)
    http = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))
    agent_tts = AuraTTS(settings.deepgram_api_key, settings.tts_voice_en, settings.tts_voice_es, client=http)
    vad_session = load_session()
    call_settings = settings.model_copy(update={"recordings_dir": str(raw / "recordings")})
    classifier = make_classifier(openai, settings.llm_model)

    def services_factory(actions: EvalActions) -> Services:
        return Services(
            settings=call_settings, repo=repo, store=store, llm_client=openai, tts=agent_tts,
            stt_factory=lambda: DeepgramSTT(settings.deepgram_api_key, settings.stt_model,
                                            endpointing_ms=settings.stt_endpointing_ms),
            vad_factory=lambda: VadSegmenter(SileroVAD(session=vad_session), min_speech_ms=settings.vad_min_speech_ms,
                                             min_silence_ms=settings.vad_min_silence_ms),
            actions=actions,
            classifier=classifier,
        )

    scenarios = pick(args.suite)
    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async def one(s) -> None:
        async with sem:
            caller_tts = AuraTTS(settings.deepgram_api_key, s.conditions.voice, s.conditions.voice, client=http)
            brain = LLMCaller(openai, settings.llm_model, s, now_local(settings.timezone))
            r = await run_scenario(s, services_factory, brain, caller_tts, repo, now_local(settings.timezone))
            results.append(asdict(r))
            ok = sum(f["ok"] for f in r.fields)
            print(f"{s.id} {s.caller_type:<10} {s.language} slots {ok}/{len(r.fields)} "
                  f"done={r.completed} v2v={r.client_v2v_ms[:3]} {r.error}", flush=True)

    await asyncio.gather(*(one(s) for s in scenarios))
    await http.aclose()
    await db.close()

    results.sort(key=lambda r: r["scenario_id"])
    metrics = report.aggregate(results)
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    title = {"dry": "Eval dry run", "smoke": "Smoke eval (20 calls)", "full": "Full eval (100 calls)"}[args.suite]
    (out / "report.md").write_text(report.to_markdown(metrics, results, title))
    print(f"\nReport: {out / 'report.md'}")
    print(json.dumps({k: metrics[k] for k in ("slot_accuracy", "classification_accuracy", "task_completion_rate",
                                              "latency_ms", "barge_in", "cost_usd")}, indent=1))

    baseline = baseline_path(args.suite)
    if args.write_baseline:
        baseline.write_text(json.dumps(metrics, indent=1))
        print(f"Baseline updated: {baseline}")
    if args.check_baseline:
        if not baseline.exists():
            print(f"No baseline yet ({baseline.name}); nothing to compare against.")
            return 0
        problems = report.compare(metrics, json.loads(baseline.read_text()))
        for p in problems:
            print(f"REGRESSION: {p}")
        return 1 if problems else 0
    return 0


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=["dry", "smoke", "full"], default="dry")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--out", default="")
    p.add_argument("--check-baseline", action="store_true")
    p.add_argument("--write-baseline", action="store_true")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(asyncio.run(main(p.parse_args())))


if __name__ == "__main__":
    cli()
