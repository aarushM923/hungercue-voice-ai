"""Turning per-call results into the headline metrics, a report, and a CI verdict."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from hungercue.pipeline.latency import percentile

# CI fails if a run is worse than the saved baseline by more than these margins.
# Margins exist because an LLM-driven test has some run-to-run noise.
TOLERANCE = {
    "slot_accuracy_drop": 0.03,  # absolute: 0.95 -> below 0.92 fails
    "latency_p50_increase": 0.15,  # relative: +15%
    "latency_p95_increase": 0.20,  # relative: +20%
}


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def _slots(results: list[dict[str, Any]]) -> tuple[int, int]:
    fields = [f for r in results for f in r["fields"]]
    return sum(f["ok"] for f in fields), len(fields)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    ok, total = _slots(results)
    per_field: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        for f in r["fields"]:
            per_field[f"{r['caller_type']}.{f['field']}"].append(f["ok"])
    client = [v for r in results for v in r["client_v2v_ms"]]
    server = [t["v2v_ms"] for r in results for t in r["server_turns"] if t.get("v2v_ms") is not None]
    stages = {
        k: statistics.median(vals) if (vals := [t[k] for r in results for t in r["server_turns"] if t.get(k) is not None]) else None
        for k in ("stt_ms", "endpoint_ms", "llm_first_token_ms", "tts_first_byte_ms")
    }
    barges = [b for r in results for b in r["barge_ins"]]
    reactions = [b["reaction_ms"] for b in barges if b["reaction_ms"] is not None]

    def subset(pred: Any) -> dict[str, Any]:
        rs = [r for r in results if pred(r)]
        hit, tot = _slots(rs)
        return {"calls": len(rs), "slot_accuracy": _rate(hit, tot),
                "completion": _rate(sum(r["completed"] for r in rs), len(rs))}

    return {
        "calls": n,
        "errors": sum(bool(r["error"]) for r in results),
        "classification_accuracy": _rate(sum(r["classified_as"] == r["caller_type"] for r in results), n),
        "slot_accuracy": _rate(ok, total),
        "slots_correct": ok,
        "slots_total": total,
        "task_completion_rate": _rate(sum(r["completed"] for r in results), n),
        "latency_ms": {
            "client_p50": percentile(client, 50), "client_p95": percentile(client, 95),
            "server_p50": percentile(server, 50), "server_p95": percentile(server, 95),
            "turns": len(client), "stage_medians": stages,
        },
        "barge_in": {
            "attempts": len(barges),
            "success_rate": _rate(sum(b["success"] for b in barges), len(barges)),
            "median_reaction_ms": statistics.median(reactions) if reactions else None,
        },
        "cost_usd": {
            "agent_per_call": round(statistics.mean(r["agent_cost_usd"] for r in results), 4) if n else None,
            "eval_overhead_per_call": round(statistics.mean(r["eval_cost_usd"] for r in results), 4) if n else None,
            "total_run": round(sum(r["agent_cost_usd"] + r["eval_cost_usd"] for r in results), 2),
        },
        "per_field_accuracy": {k: _rate(sum(v), len(v)) for k, v in sorted(per_field.items())},
        "by_condition": {
            "clean_audio": subset(lambda r: r["conditions"]["noise"] == "none" and not r["conditions"]["mumble"]),
            "background_noise": subset(lambda r: r["conditions"]["noise"] != "none"),
            "mumbling": subset(lambda r: r["conditions"]["mumble"]),
            "interruptions": subset(lambda r: r["conditions"]["interrupt"]),
            "topic_change": subset(lambda r: r["conditions"]["topic_change"]),
            "spanish": subset(lambda r: r["language"] == "es"),
            "restaurant": subset(lambda r: r["caller_type"] == "restaurant"),
            "shelter": subset(lambda r: r["caller_type"] == "shelter"),
            "volunteer": subset(lambda r: r["caller_type"] == "volunteer"),
        },
    }


def compare(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Return human-readable regressions (empty list = pass)."""
    problems = []
    cur_slot, base_slot = current.get("slot_accuracy"), baseline.get("slot_accuracy")
    if cur_slot is not None and base_slot is not None and cur_slot < base_slot - TOLERANCE["slot_accuracy_drop"]:
        problems.append(f"slot accuracy {cur_slot:.1%} is below baseline {base_slot:.1%} minus {TOLERANCE['slot_accuracy_drop']:.0%}")
    for key, tol in (("client_p50", "latency_p50_increase"), ("client_p95", "latency_p95_increase")):
        cur, base = current["latency_ms"].get(key), baseline["latency_ms"].get(key)
        if cur is not None and base is not None and cur > base * (1 + TOLERANCE[tol]):
            problems.append(f"latency {key} {cur:.0f} ms exceeds baseline {base:.0f} ms by more than {TOLERANCE[tol]:.0%}")
    if current.get("calls") and current.get("errors", 0) > max(1, current["calls"] // 10):
        problems.append(f"{current['errors']} of {current['calls']} calls errored")
    return problems


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def _ms(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f} ms"


def to_markdown(m: dict[str, Any], results: list[dict[str, Any]], title: str) -> str:
    lat, b, cost = m["latency_ms"], m["barge_in"], m["cost_usd"]
    lines = [
        f"# {title}", "",
        f"{m['calls']} simulated calls through the real pipeline (Deepgram STT and TTS, GPT-4.1-mini), real-time audio.", "",
        "| Metric | Result |", "|---|---|",
        f"| Caller classification accuracy | {_pct(m['classification_accuracy'])} |",
        f"| Slot accuracy | {_pct(m['slot_accuracy'])} ({m['slots_correct']}/{m['slots_total']} fields) |",
        f"| Task completion rate | {_pct(m['task_completion_rate'])} |",
        f"| Voice-to-voice latency p50 / p95 (caller side) | {_ms(lat['client_p50'])} / {_ms(lat['client_p95'])} |",
        f"| Voice-to-voice latency p50 / p95 (server side) | {_ms(lat['server_p50'])} / {_ms(lat['server_p95'])} |",
        f"| Barge-in success rate | {_pct(b['success_rate'])} of {b['attempts']} (median reaction {_ms(b['median_reaction_ms'])}) |",
        f"| Cost per call (agent, incl. Twilio estimate) | ${cost['agent_per_call']} |",
        f"| Calls with errors | {m['errors']} |", "",
        "## Where the time goes (median per stage)", "",
        "| Stage | Median |", "|---|---|",
        *[f"| {k.replace('_ms', '').replace('_', ' ')} | {_ms(v)} |" for k, v in lat["stage_medians"].items()], "",
        "## Accuracy by condition", "",
        "| Slice | Calls | Slot accuracy | Completion |", "|---|---|---|---|",
        *[f"| {k.replace('_', ' ')} | {v['calls']} | {_pct(v['slot_accuracy'])} | {_pct(v['completion'])} |"
          for k, v in m["by_condition"].items()], "",
        "## Accuracy by field", "",
        "| Field | Accuracy |", "|---|---|",
        *[f"| {k} | {_pct(v)} |" for k, v in m["per_field_accuracy"].items()], "",
    ]
    misses = [(r["scenario_id"], f) for r in results for f in r["fields"] if not f["ok"]]
    if misses:
        lines += ["## Missed fields", "", "| Call | Field | Expected | Captured |", "|---|---|---|---|"]
        lines += [f"| {sid} | {f['field']} | {f['expected']} | {f['got']} |" for sid, f in misses[:60]]
        lines.append("")
    errors = [(r["scenario_id"], r["error"]) for r in results if r["error"]]
    if errors:
        lines += ["## Errors", "", *[f"- {sid}: {e}" for sid, e in errors], ""]
    lines.append(f"Eval overhead (simulated caller LLM and voice): ${cost['eval_overhead_per_call']} per call; whole run ${cost['total_run']}.")
    return "\n".join(lines) + "\n"
