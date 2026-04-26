# Tasks

## P0

- [x] Define the `provenance` JSON shape with backwards-compatible optional fields.
- [x] Capture git SHA, dirty state, branch, and config hash.
- [x] Capture Python, package, platform, podman, and llama-swap details where available.
- [x] Capture server image and benchmark seed in provenance.
- [x] Normalize configured model metadata into a top-level `model_metadata` list.
- [x] Infer safe local fields such as repo ID, filename, quantization, context length, and MoE CPU-offload setting.
- [x] Add provenance and model metadata to emitted JSON payloads.
- [x] Render concise provenance in Markdown reports.
- [x] Render concise provenance in HTML reports.
- [x] Preserve report compatibility with legacy payloads.

## P1

- [x] Add optional HuggingFace metadata enrichment for config entries in `<org>/<repo>/<file>.gguf` form.
- [x] Record HuggingFace lookup failures as metadata warnings, not benchmark failures.
- [x] Add config or CLI control for enrichment mode: off, best-effort, strict.
- [x] Mock HuggingFace calls in tests.
- [x] Document offline behavior and credential expectations.

## P2

- [ ] ~~Consider recording local GGUF file size and mtime when the file exists.~~
- [ ] ~~Consider adding a compact report footer with dependency versions.~~
