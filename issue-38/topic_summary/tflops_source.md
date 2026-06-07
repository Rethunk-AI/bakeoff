# Summary: tflops_source

## Final state

`tflops_source` resolved as a FK column on a related `gpu_tflops` table, pointing to a seeded `tflops_sources` lookup table. The per-format tflops values are stored in `gpu_tflops`, not as flat columns on `gpu_hardware`. The `tflops_sources` table tracks provenance (manufacturer, measured, unknown) with structured contact data and URL templating.

### `gpu_tflops` (resolved)

| column | SQL type | notes |
|---|---|---|
| `gpu_hardware_id` | FK → `gpu_hardware` | PK component |
| `compute_format_id` | FK → `compute_formats` | PK component; seeded: fp16, fp32, bf16, int8, fp8, etc. |
| `tflops_value` | FLOAT | |
| `tflops_source_id` | FK → `tflops_sources` | replaces free `tflops_source` TEXT field |

PK: `(gpu_hardware_id, compute_format_id)`

### `tflops_sources` (resolved)

| column | SQL type | notes |
|---|---|---|
| `source_id` | SERIAL | PK; seed 1 = 'unknown/unverified' |
| `name` | TEXT NOT NULL | |
| `contacts` | JSONB | replaces single `contact_url`; array of `{type, value}` objects |
| `url_template` | TEXT | static URL or expression prefix; complex construction deferred |

Seed 2 = 'Rethunk measured' — added when own benchmarks ship (per 4552726518).

`peak_tflops_fp16` on `gpu_hardware` (from #18 body) is superseded by `gpu_tflops` — the per-format lookup table is the canonical store; the flat column was recognised as a duplicate and dropped (per 4534654458).

## Notable / unusual decisions

- **Lookup table replaces flat per-format columns** — rather than `tflops_fp16`, `tflops_fp32`, `tflops_bf16` as separate nullable columns, a related `gpu_tflops` table keyed by `compute_format_id` was chosen. Rationale: extensible without schema migration per new format; avoids wide sparse null columns (per 4534654458). Downstream: any query for a specific precision pivots through a join rather than a direct column read.

- **`tflops_source` as FK, not enum** — `'manufacturer'`|`'measured'` was the original proposal but @gissf1 directed it become a FK to a lookup table with contact/URL metadata, so sources are addressable entities (per 4550408728). Downstream: lets Rethunk eventually register itself as a source and link measured results to a public reference.

- **`contacts` as JSONB array** — `contact_url TEXT` was rejected because a single source may have multiple contact methods (email, phone, web form) and multiple people. JSONB `[{type, value}]` covers this for Phase 1; a normalised contacts table is Phase 2 (per 4553785030, confirmed 4554217644).

- **`url_template` simplified** — Bastion proposed `url_template` (simple tokens) + `url_script` (Lua) alongside it. @gissf1 collapsed this: one `url_template` field holds either a static URL or an expression using whatever scripting/templating language is settled for `schema_versions`; prefix inspection distinguishes the two (per 4554217644). No separate `url_script` column.

- **GPU performance metrics (tflops, bandwidth, TDP) excluded from UUID** — explicitly confirmed not identity fields (per 4550408728). Downstream: changing a GPU's measured tflops rating does not invalidate hardware records or break result linkage.

- **bf16 included for completeness** — @gissf1 noted the distinction from fp16 was subtle but accepted inclusion for completeness in `compute_formats` (per 4552726518).

- **`compute_units` dropped** — deemed implied by PCI IDs + architecture; no independent signal for the test surface (per 4552726518, following 4550408728 direction).

## Open / unresolved

- **Scripting language for `url_template` expressions** — @gissf1 directed that `url_template` can hold an expression in whatever language is chosen for `schema_versions` migration scripts, but that language itself is unresolved. Lua was Bastion's recommendation; @gissf1 raised Gonja, Pongo2, sprig, Go templates, and CEL as alternatives and asked for a clearer comparison before committing (per 4554217644). The `url_template` design depends on this resolution.
  - address: #22, https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4554217644

- **`tflops_sources.url_template` complex-substitution capability** — the settled design uses a single field with prefix-detected mode. Whether that field can handle GPU-architecture-based slug mappings, capitalisation normalisation, or cross-field construction (the cases @gissf1 raised) depends on the scripting language decision above. If the chosen language cannot handle those cases, the field design needs revisiting.
  - address: #22, https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4554217644

- **Multiple contacts / contacts table (Phase 2)** — whether `tflops_sources.contacts` JSONB is sufficient long-term or needs a normalised contacts table was explicitly deferred. JSONB accepted for Phase 1 only (per 4554217644).
  - address: #22, https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4554217644

## Cross-topic links

- **`gpu_hardware`** — `gpu_tflops` FKs to `gpu_hardware_id`; `peak_tflops_fp16` flat column on that table is superseded here.
- **`compute_formats`** — seeded lookup table; `gpu_tflops.compute_format_id` FKs to it. Topic: `compute_formats` (if separated) or subsumed under `gpu_hardware`.
- **`schema_versions` / scripting language** — `tflops_sources.url_template` expression mode is contingent on the scripting language decision made in the `schema_versions` / `uuid_migrations` topic.
