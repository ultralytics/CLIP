# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

Respecting these principles is critical for every PR.

**Less is more. The simplest solution is the best solution.**

The action hierarchy for every change: **Delete > Replace > Add**. The best code change is a deletion. The second best is modifying what exists. Adding new code is the last resort.

1. **Minimal**: The simplest solution that works. Do not over-engineer, over-abstract, or add code just in case. Three similar lines beat a premature abstraction. Avoid error handling for impossible states, feature flags, compatibility shims, or policy scaffolding unless they are truly required.
2. **Solve at the source**: Do not hack fixes. Solve problems at their root. If something is broken, fix or remove the broken thing. Never patch over a broken abstraction, add workarounds, or add synchronization code for state that should not be duplicated.
3. **Delete ruthlessly**: When replacing code, delete what it replaced. Remove unused imports, functions, types, files, and commented-out code. Git preserves history. Run the repo's relevant dead-code or cleanup check when available.
4. **Replace > Add**: Modify existing code over adding new code. Edit existing files, extend existing components or functions with minimal parameters, and reuse existing utilities. If creating a new file, first prove it cannot fit cleanly in an existing file.
5. **Check existing**: Search the entire repo before creating anything new. If a feature, component, helper, responder, workflow, or utility already solves a similar problem, reuse or adapt it and delete the duplicate path.
6. **Deduplicate**: Do not duplicate existing code when updating the repo. Consolidate or refactor duplicates you find when it is in scope and low risk.
7. **Zero Regression**: Do not break existing features or workflows unless the PR intentionally removes them with evidence.
8. **Production ready**: All changes must be thoroughly debugged, validated, and production ready.

**When fixing bugs, ask: "What can I delete?" before "What can I replace?" before "What should I add?"**

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Launch an independent adversarial review agent with cold context (just the PR diff and this file) to hunt for bugs, regressions, and Core Principles violations — use the Codex CLI, one fresh `codex exec` run per round. Fix, push, and repeat until a fresh run reports LGTM.
3. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never force-push, reset, or revert commits you did not author.
4. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
uv pip install -e ".[dev]"                                            # install package + pytest (never bare pip install)
uv run pytest                                                         # run all tests exactly as CI does (no coverage step in CI)
uv run pytest "tests/test_consistency.py::test_consistency[ViT-B/32]" # run one parametrized case
uvx codespell                                                         # spelling check; config in [tool.codespell] in pyproject.toml
```

- CI (`.github/workflows/ci.yml`) runs on push/PR to `main` and daily at 03:00 UTC, on a matrix of Python {3.9, 3.13} × PyTorch {2.5.0, 2.8.0} using CPU wheels (`--extra-index-url https://download.pytorch.org/whl/cpu`), and retries `uv run pytest` up to 2 times via `ultralytics/actions/retry`.
- `pyproject.toml` declares `requires-python = ">=3.7"`, but Python 3.9 is the tested floor.
- There is no local Ruff/format config: Python (Ruff + docformatter), Prettier, and codespell formatting are auto-applied to PR branches by Ultralytics Actions (`.github/workflows/format.yml`) — pull its commits rather than hand-formatting.

## Architecture

Ultralytics-maintained fork of OpenAI's CLIP: a single small package (`clip/`, ~950 lines) for zero-shot image-text inference with pretrained checkpoints. `clip/clip.py` is the entry point: the `_MODELS` registry maps 9 model names (RN50…ViT-L/14@336px) to checkpoint URLs on `openaipublic.azureedge.net` with the SHA256 embedded in the URL and verified by `_download()`; `load()` builds either a Python model via `build_model()` or a TorchScript one, in the JIT case rewriting graph constants to patch device and fp32-on-CPU; `tokenize()` returns int32 token tensors. `clip/model.py` defines the network (`ModifiedResNet`, `VisionTransformer`, `CLIP`), with `build_model()` inferring the architecture from state-dict shapes. `clip/simple_tokenizer.py` is a BPE tokenizer whose vocab `clip/bpe_simple_vocab_16e6.txt.gz` ships inside the package (`MANIFEST.in` + `include-package-data`). `hubconf.py` generates one `torch.hub` entrypoint per model with punctuation mapped to underscores (`ViT-B/32` → `ViT_B_32`).

There is no release or publish workflow and the package is not on PyPI: `version = "1.0"` in `pyproject.toml` is static, and users install straight from git (`pip install git+https://github.com/ultralytics/CLIP.git`). The only merge gating is CI plus Ultralytics Actions on PRs.

## Conventions

- Every source file starts with the `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` header — Ultralytics Actions adds it automatically; don't add or revert it manually.
- The single test file `tests/test_consistency.py` parametrizes over all 9 models and hits the live network, downloading every checkpoint (several GB) to compare JIT vs non-JIT outputs — expect long, network-dependent local runs.
- No version-bump or release process: merges to `main` are the release; distribution is directly from the git repo.
