"""ModelPath: task-aware Ollama routing with failover and benchmark logs."""
from __future__ import annotations
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

class ModelUnavailable(RuntimeError):
    pass

@dataclass(frozen=True)
class ModelSpec:
    name: str
    task_types: tuple[str, ...]
    context: int = 4096
    priority: int = 0

def classify(prompt: str) -> str:
    """Deterministic, inspectable baseline; replace only after evaluation."""
    p = prompt.lower()
    if any(k in p for k in ("json", "extract", "schema", "fields")):
        return "extraction"
    if any(k in p for k in ("python", "function", "code", "bug", "sql")):
        return "coding"
    return "reasoning"

def select_models(task: str, registry: list[ModelSpec]) -> list[ModelSpec]:
    eligible = [m for m in registry if task in m.task_types]
    fallback = [m for m in registry if m not in eligible]
    return sorted(eligible, key=lambda m: m.priority) + sorted(fallback, key=lambda m: m.priority)

def ollama_generate(model: str, prompt: str, *, base_url: str = "http://localhost:11434", timeout: float = 90) -> dict:
    base_url = base_url.rstrip("/")
    if not re.fullmatch(r"https?://[^/]+", base_url):
        raise ValueError("Invalid Ollama base URL")
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(base_url + "/api/generate", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            data = json.load(res)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise ModelUnavailable(f"{model}: {exc}") from exc
    if "response" not in data:
        raise ModelUnavailable(f"{model}: no response field")
    return data

def route(prompt: str, registry: list[ModelSpec], *, generate=ollama_generate) -> dict:
    task = classify(prompt)
    errors = []
    for model in select_models(task, registry):
        started = time.perf_counter()
        try:
            result = generate(model.name, prompt)
            answer = result["response"]
            if not isinstance(answer, str) or not answer.strip():
                raise ModelUnavailable("Empty model response")
            return {"task": task, "model": model.name, "answer": answer, "latency_ms": round((time.perf_counter()-started)*1000, 2), "attempts": len(errors)+1, "errors": errors, "usage": {k: result.get(k) for k in ("prompt_eval_count", "eval_count")}}
        except (ModelUnavailable, KeyError, TypeError) as exc:
            errors.append({"model": model.name, "reason": str(exc)})
    raise ModelUnavailable("All models failed: " + json.dumps(errors))

def load_registry(path: str | Path) -> list[ModelSpec]:
    payload = json.loads(Path(path).read_text())
    models = [ModelSpec(name=x["name"], task_types=tuple(x["task_types"]), context=int(x.get("context",4096)), priority=int(x.get("priority",0))) for x in payload]
    if not models or len({m.name for m in models}) != len(models):
        raise ValueError("Registry must contain uniquely named models")
    return models

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--registry", default="models.json")
    p.add_argument("--log", default="benchmark.jsonl")
    args = p.parse_args()
    output = route(args.prompt, load_registry(args.registry))
    print(json.dumps(output, indent=2))
    with open(args.log, "a", encoding="utf-8") as log:
        log.write(json.dumps({"prompt": args.prompt, **output}) + "\n")

if __name__ == "__main__":
    main()
