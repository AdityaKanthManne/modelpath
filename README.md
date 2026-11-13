# ModelPath — Local Multi-Model Router

Working Python CLI for deterministic task classification, Ollama model selection, fallback and structured benchmark logging. **Not yet an experimentally optimized router:** routing policy and performance claims must be measured on the deployment machine.

## Quick start
Python 3.11+, [Ollama](https://ollama.com/) installed and running.

```bash
python -m pip install -e '.[dev]'
ollama pull qwen2.5:1.5b
ollama pull qwen2.5-coder:1.5b
ollama pull llama3.2:1b
modelpath "Extract invoice fields as JSON"
pytest -q
```

Edit `models.json` to list models available locally. Requests default to http://localhost:11434. CLI writes performance observations to `benchmark.jsonl` (may contain user text; do not use for sensitive prompts).

## Behavior
- Explains routing with a deterministic baseline classifier.
- Prefers task-appropriate models, then falls back on model errors or empty replies.
- Captures model, latency, attempt count, reported token counts and errors.
- Unit tests use a fake inference backend and need no Ollama.

## Known limitations
The router does not yet measure quality against a labelled dataset, validate generated JSON, or optimize a learned policy. Hardware latency and accuracy improvements have not been measured. Do not claim 20% savings until benchmarked.
