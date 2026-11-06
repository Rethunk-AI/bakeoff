# Model Testing Queue

P0/P1/P2 are a **one-time seeding mechanism**, not a rolling tier system.
Models enter the testing queue in tier order. Priority numbers do not shift —
a P2 model stays P2 until explicitly promoted. Management passes to the queue
itself once testing begins.

GGUF paths are relative to `engined.models_dir` in `config.yaml` (default
`~/.local/share/engined-models/llm-bench`). Paths marked **TBD** have not yet
been confirmed available on the test host.

---

## P0 — Current

Commodity hardware, sub-15B, dense or small-active-param MoE. These form the
baseline cohort.

| Model | Weights | GGUF path | Status |
| ------- | --------- | ----------- | -------- |
| llama3.2 | 1B | `lmstudio-community/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| llama3.2 | 3B | `lmstudio-community/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| llama3.1 | 8B | `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| qwen3 | 1.7B | `lmstudio-community/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| qwen3 | 4B | `lmstudio-community/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| qwen3 | 8B | `lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| qwen3.5 | 9B | `lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| qwen3.6 | 35B-A3B | `lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| gemma3 | 4B | `lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| gemma4 | e2b | `lmstudio-community/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| gemma4 | e4b | `lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| phi3.5-mini | 3.8B | `bartowski/Phi-3.5-mini-instruct-GGUF/Phi-3.5-mini-instruct-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| phi4 | 14B | `lmstudio-community/phi-4-GGUF/phi-4-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| ministral | 3B | `lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |
| ministral | 8B | `bartowski/Ministral-8B-Instruct-2410-GGUF/Ministral-8B-Instruct-2410-Q4_K_M.gguf` | Run 2026-11-05 (unpublished) |

---

## P1 — Near

Next cycle after P0 baseline is published.

| Model | Weights | GGUF path |
| ------- | --------- | ----------- |
| qwen3 | 14B | `lmstudio-community/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf` |
| qwen2.5-coder | 7B | `lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` |
| qwen2.5-coder | 14B | `lmstudio-community/Qwen2.5-Coder-14B-Instruct-GGUF/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf` |
| gemma3 | 12B | TBD |
| gemma3 | 27B | TBD |
| gemma4 | e26b | TBD |
| DeepSeek-R1-0528-Qwen3 | 8B | `lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf` |
| llama4-scout | fp8 MoE | TBD — deferred from P0; large download |

---

## P2 — Long

Deferred: large VRAM requirement or lower relative priority.

| Model | Weights | Reason |
| ------- | --------- | -------- |
| qwen3 | 30B, 32B, 235B | Exceeds comfortable VRAM budget |
| qwen2.5-coder | 32B | Large |
| Qwopus models | — | Lower priority |
| llama4-maverick | 400B total MoE | Not commodity hardware |

---

## Mechanics

- Models are tested in P0 → P1 → P2 order.
- To run a model: copy the relevant entry into `config.yaml` `models:` block,
  confirm the GGUF is available at `engined.models_dir/<path>`, and run the
  harness.
- P0 cohort ran together on 2026-11-05 (`scored` judge, Qwen3.6-35B-A3B); `llama3.1` 8B first scored 0.026 heuristic / 1.50 judge because its GGUF template treats unset `tools` as present and answered every prompt with a function-call JSON; rerun with `chat_template_kwargs: {tools: null}` on the model entry scored 0.658 heuristic / 4.88 ± 0.33 judge (`results/run-20261105-231504.json`). Use that entry for any Llama-3.1 template. Descriptors for Qwen3-1.7B and Qwen3-4B were refused by `descriptor submit` (upstream declares no license).
- Mark a model's status as **Tested** once a result bundle has been published
  to the results repo and update its row above.
- To promote a model between tiers: edit this file and note the reason.
- TBD paths: confirm via `fd -e gguf . ~/.local/share/engined-models/llm-bench/`
  or fetch with `./run.sh fetch` before seeding into config.
