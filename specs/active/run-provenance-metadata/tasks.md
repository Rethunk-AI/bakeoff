# Tasks

## P0

- [ ] Define the `provenance` JSON shape with backwards-compatible optional fields.
- [ ] Capture git SHA, dirty state, branch, and config hash.
- [ ] Capture Python, package, platform, podman, and llama-swap details where available.
- [ ] Capture server image and benchmark seed in provenance.
- [ ] Normalize configured model metadata into a top-level `model_metadata` list.
- [ ] Infer safe local fields such as repo ID, filename, quantization, context length, and MoE CPU-offload setting.
- [ ] Add provenance and model metadata to emitted JSON payloads.
- [ ] Render concise provenance in Markdown reports.
- [ ] Render concise provenance in HTML reports.
- [ ] Preserve report compatibility with legacy payloads.

## P1

- [ ] Add optional HuggingFace metadata enrichment for config entries in `<org>/<repo>/<file>.gguf` form.
- [ ] Record HuggingFace lookup failures as metadata warnings, not benchmark failures.
- [ ] Add config or CLI control for enrichment mode: off, best-effort, strict.
- [ ] Mock HuggingFace calls in tests.
- [ ] Document offline behavior and credential expectations.

## P2

- [ ] Consider recording local GGUF file size and mtime when the file exists.
- [ ] Consider adding a compact report footer with dependency versions.
