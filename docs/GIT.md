# Git & repository guide

How this repository is organised for version control: what is tracked, what is
regenerated, how to branch and commit, and how to publish when a remote is added.

---

## 1. Current state

- **Remote:** none. This repo is local-only right now (`git remote -v` is empty).
  Pushing and pull requests happen once you add a remote — see section 7.
- **Branches:**
  - `main` — the integration branch.
  - `p0-credibility-pass` — the active working branch.
- **Line endings:** the repo stores LF. On Windows you will see
  `warning: LF will be replaced by CRLF` when staging — this is expected and
  harmless; Git normalises on checkout.

---

## 2. What is tracked vs. regenerated

Large or machine-specific artifacts are **not** committed — they are rebuilt from
code and scripts. Keep them out of Git; regenerate them after a fresh clone.

| Path | Tracked? | How to (re)generate |
|------|----------|---------------------|
| Source code (`*.py`), prompts (`llm/prompts/*.txt`) | Yes | — |
| Dependency lists (`requirements/base.txt`, `requirements/notebooks.txt`) | Yes | — |
| Docs: `README.md` + `LICENSE` at root, `models/README.md` with its package, and the guides under `docs/` (`DOCUMENTATION.md`, `GIT.md`, `PROJECT_GUIDE.md`, `MANUAL_TESTING.md`, `CONTRIBUTING.md`, `SECURITY.md`) | Yes | — |
| Internal working-notes (`PR_BODY.md`, `CHANGELOG.md`) | **No** (local-only) | Working notes; keep on disk |
| `MedicalHybirdModel.env` | **No** (secrets) | Copy from `MedicalHybirdModel.env.example` |
| `data/raw`, `data/processed`, `data/vector_store` | **No** | `python -m scripts.setup` (download + ingest) |
| `data/users` (saved profiles) | **No** | Created at runtime |
| FAISS index / BM25 corpus (`*.faiss`, `*.bin`) | **No** | `python -m rag.ingest` |
| ML artifacts (`ml_model/artifacts/`) | **No** | `python -m ml_model.symptom_classifier_train` |
| Signal/imaging weights (`models/checkpoints/*.pth`) | **No** (large binaries) | Trained in `models/training/*.ipynb`; drop the `.pth` files into `models/checkpoints/` |
| Generated sample signals (`samples/`, `*.npy`) | **No** | `python -m scripts.make_sample_signals` |

Empty tracked folders are preserved with `.gitkeep` files so the layout survives a
clone even when their contents are ignored.

### Rebuild-from-clone recipe

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements/base.txt
cp MedicalHybirdModel.env.example MedicalHybirdModel.env      # then edit if needed
python -m scripts.setup --with-ml                            # data + index + live ML
python -m scripts.healthcheck                                # verify deps + Ollama
```

The signal/imaging checkpoints (`models/checkpoints/*.pth`) are the exception:
they are not produced by `scripts.setup`. Retrain them from the notebooks in
`models/training/` (or obtain the `.pth` files) and place them in
`models/checkpoints/`.

---

## 3. Branching workflow

```bash
git switch main
git switch -c my-feature          # branch off main for each unit of work
# ... edit, then stage and commit (section 4) ...
git switch main && git merge --no-ff my-feature   # or open a PR once a remote exists
```

Keep one branch per coherent change. Do not commit directly to `main` for
non-trivial work.

---

## 4. Commit conventions

- **Subject:** imperative mood, <= ~72 chars, no trailing period.
  e.g. `Fix stale ML-train command in Docker docs`.
- **Body:** wrap ~72 cols. Explain *why*, not just *what*. Note test results when
  relevant (`Suite: 204 passed.`).
- **No authorship trailers.** Do not add `Co-Authored-By:` or
  "Generated with ..." footers — this project keeps commits free of tool/AI
  signatures.
- Prefer small, self-contained commits over one large mixed commit.

```bash
git add -A
git commit -F- <<'MSG'
Short imperative subject

Why the change was needed and what it does, in a sentence or two.
Suite: 204 passed.
MSG
```

---

## 5. Before you commit

- Run the suite: `.venv/Scripts/python.exe -m pytest -q`.
- Review the diff: `git diff --staged`.
- Confirm you are not adding a regenerated artifact (check `git status` against
  the table in section 2).
- Prompt files (`llm/prompts/*.txt`) are read fresh at runtime; code changes
  (API/dashboard) need a process restart to take effect.

---

## 6. Handy commands

```bash
git status --short                 # what changed
git log --oneline -15              # recent history
git switch -                       # jump back to previous branch
git restore --staged <path>        # unstage
git restore <path>                 # discard working-tree change (destructive)
git check-ignore -v <path>         # explain why a path is ignored
```

---

## 7. Publishing (once a remote exists)

There is no remote and no `gh` CLI configured on this machine yet, so pull
requests must be opened by hand. After creating a repo on your host:

```bash
git remote add origin <url>
git push -u origin main
git push -u origin p0-credibility-pass
```

Then open a pull request from `p0-credibility-pass` into `main` in the host's web
UI. `PR_BODY.md` (local-only) holds a ready-to-paste description.

---

## 8. The Markdown rule

Markdown is **tracked by default** — a new doc you create (a guide, a design note)
is versioned like any other file. Only a short, explicit list of internal
working-notes is kept local-only (see the ignore block in `.gitignore`:
`PR_BODY.md`, `CHANGELOG.md`). To keep a new Markdown file off the repo, add its
path to that block.
