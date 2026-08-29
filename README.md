<h1 align="center">Local LLM N-vs-N Benchmark</h1>

<div align="center">

[![ci](https://github.com/Rethunk-AI/bakeoff/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Rethunk-AI/bakeoff/actions/workflows/ci.yml)
[![license](https://img.shields.io/github/license/Rethunk-AI/bakeoff)](LICENSE)
[![python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](pyproject.toml)

</div>

---

Serves GGUFs from `~/.lmstudio/models/` through a [`llama-swap`](https://github.com/mostlygeek/llama-swap) proxy in front of `llama.cpp` podman containers. Benchmarks **quality**, **latency**, and **cost** (energy) across `tasks × prompt_variants × models`. Judge modes: `pairwise_all` tournament or `scored` rubric. Emits JSON, Markdown, and a single-file HTML dashboard under `results/`.

## Quick start

```sh
./run.sh
```

Prerequisites, install, and configuration: [HUMANS.md](HUMANS.md).

## Highlights

- `llama-swap` + llama.cpp Vulkan podman image; OpenAI-compatible client
- `pairwise_all` tournament or `scored` 1–5 rubric; heuristic fallbacks
- Quality, latency, and energy-based cost metrics
- JSON, Markdown, and single-file HTML dashboard output
- Any number of models; deterministic seeded synthetic dataset
- `./run.sh fetch` pulls missing GGUFs from Hugging Face

## Documentation

| Document | Audience | Contents |
| --- | --- | --- |
| [HUMANS.md](HUMANS.md) | Operators & developers | Prerequisites, install, run, configure, troubleshoot, clean up |
| [AGENTS.md](AGENTS.md) | LLMs & contributors | Design invariants, hardware caveats, judge-mode selection, editing conventions |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributors | PR checklist, commit style |
| [config.yaml](config.yaml) | Reference | Server, models, prompts, dataset, judge, cost, output (inline comments) |

## License

Licensed under the terms in [LICENSE](LICENSE).
