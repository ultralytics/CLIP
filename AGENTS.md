# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

Ultralytics-maintained fork of OpenAI's [CLIP](https://github.com/openai/CLIP) (AGPL-3.0), a neural network trained on image-text pairs that predicts the most relevant caption for an image without task-specific training. The repository ships the small `clip` inference package, the pretrained-checkpoint loader, and the notebooks and model card that document zero-shot use.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

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
