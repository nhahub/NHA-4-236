# Contributing

Thanks for your interest in improving the Hybrid Medical Assistant. This is an
**educational** project; contributions that improve correctness, grounding,
safety, documentation, or test coverage are especially welcome. New clinical
"features" that could be mistaken for medical advice are out of scope — see the
[medical disclaimer](../LICENSE) and the safety notes in the [README](../README.md).

## Development setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements/base.txt
cp MedicalHybirdModel.env.example MedicalHybirdModel.env
python -m scripts.setup --with-ml   # download data, build the index, train the ML head
python -m scripts.healthcheck       # verify interpreter + deps (+ optional Ollama)
```

See the [README quick start](../README.md#quick-start) for the full run
instructions (Ollama, the API, and the dashboard).

## Before you open a PR

- **Run the tests** — they are fully offline (no Ollama, no network, no trained
  artifacts required; the tests that need those auto-skip):
  ```bash
  pytest -q            # ~200 offline tests
  ruff check .         # lint
  ```
- **Keep claims measured.** If you change retrieval, prompts, the ML head, or a
  model default, re-run the relevant `eval` module and update any numbers you
  touch in the README (`python -m eval` — see [Evaluation](../README.md#evaluation)).
  Every benchmark in the docs is reproducible from a documented command; keep it
  that way.
- **Update the docs** in the same PR as the code they describe (README config
  table, `DOCUMENTATION.md`, command examples).

## Commit & branch conventions

These mirror [GIT.md](GIT.md):

- Branch off `main` (one branch per coherent change); don't commit non-trivial
  work directly to `main`.
- **Subject line:** imperative mood, ≤ ~72 chars, no trailing period.
- **Body:** wrap ~72 cols; explain *why*, not just *what*; note test results when
  relevant (e.g. `Suite: 204 passed`).
- Prefer small, self-contained commits over one large mixed commit.

## Code style

- Follow the surrounding code: type hints on public functions, module/function
  docstrings, and the structured one-line-per-request logging already in place.
- Lint with `ruff check .` (config in `pyproject.toml`).
- Don't commit regenerated artifacts (FAISS index, trained weights, `data/`) —
  the [GIT.md](GIT.md) table lists what is tracked vs. rebuilt.

## Reporting issues

For a security or data-handling concern, see [SECURITY.md](SECURITY.md). For
anything else, open an issue describing what you expected, what happened, and the
steps to reproduce.
