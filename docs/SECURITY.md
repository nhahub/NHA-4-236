# Security Policy

## Scope & intended use

This project is a **local, educational** medical Q&A assistant. It is **not a
medical device** and must not be used for clinical decisions or deployed to
serve real patients. Please keep that context in mind when reporting issues.

By design the whole system runs locally: the LLM (via Ollama), the embeddings,
the vector index, and any uploaded files stay on the machine that runs it. No
data is sent to a third-party API.

## Reporting a vulnerability

If you find a security or privacy issue (for example: prompt-injection that
defeats the safety guardrails, a way to leak the system prompt, unsafe file
handling in the `/analyze/*` upload endpoints, or a path-traversal / SSRF in the
API), please report it privately rather than opening a public issue:

- Open a GitHub **security advisory** on the repository (Security → Report a
  vulnerability), or
- Email the maintainer (see the commit history / profile).

Please include steps to reproduce and the impact. We aim to acknowledge reports
within a few days. As an educational project maintained on a best-effort basis,
there is no formal SLA.

## Handling of secrets

- No credentials are required or committed. `MedicalHybirdModel.env` is
  gitignored; only `MedicalHybirdModel.env.example` (no secrets) is tracked.
- Do not commit real patient data, API keys, or model weights. See
  [GIT.md](GIT.md) for what is tracked vs. regenerated.

## Safety guardrails (context for reports)

Several protections are enforced in code, not left to the model — a report that
bypasses any of these is in scope:

- Emergency / self-harm triage short-circuits before any LLM call.
- Intent routing prevents hallucinated sources for chit-chat / identity probes.
- A retrieval confidence gate declines out-of-corpus queries instead of guessing.
- A citation-integrity pass strips invented citation markers.
- The non-diagnostic disclaimer is appended in code on every generated answer.
- The imaging/signal models are a sandbox and never influence the medical answer.

The prompt-injection resistance eval (`python -m eval.injection`) documents the
currently-covered attack classes; new bypasses are valuable bug reports.
