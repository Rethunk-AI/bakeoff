# Summary: open_questions (cross-cutting roll-up)

This is a synthetic roll-up, not a source topic. It indexes every unresolved decision across the corpus so a reviewer can see at a glance what is still open and exactly where to answer it. Built from the '## Open / unresolved' sections of the per-topic summaries. (#38 thread excluded from this corpus variant.)

---

## Open questions by topic

### gpu_architecture

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | `uuid_migrations.new_id` — store or derive? | Stored nullable enables O(1) repeat lookup after first resolution; pure derivation avoids stale cache but requires re-read of disk file every ingest | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534654458> |
| 2 | `uuid_migrations` PK — what replaces `(old_id, new_id)` when `new_id` is not fixed? | Composite PK needs stable `new_id`; single-column PK loses ability to cache multiple resolution paths; surrogate PK adds indirection | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534370703> |
| 3 | `tdp_w` reliability — include in schema or annotate as spec-only estimate? | TDP is a manufacturer spec that frequently diverges from actual power draw; including it without a source flag may mislead result consumers | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534370703> |
| 4 | `compute_units` rename — confirm `parallel_execution_units` or another name? | Original label was flagged as unclear; no explicit confirmation of final column name after rename was offered | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534370703> |
| 5 | `bios_notes` UUID whitelist — does the proposed initial whitelist look right? | Whitelist controls which BIOS settings are hashed into the hardware UUID; an incorrect whitelist causes false UUID matches or spurious migration records | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534654458> |
| 6 | GPU UUID field list — confirm `slot_index` exclusion and current field set? | Slot exclusion was proposed but never explicitly confirmed; an unconfirmed field list leaves the UUID computation underspecified | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534654458> |

---

### gpu_hardware

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | `bios_notes` UUID whitelist completeness — are motherboard bus-width overrides missing? | Forcing x8 for an x16 slot is a BIOS setting that affects performance; if not captured in UUID it creates false hardware identity matches | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4552726518> |
| 2 | `schema_versions.changes` JSONB vocabulary — are proposed keys (`uuid_fields_added/removed`, `tables_added/modified`, `encoding_changes`) complete? | Unrecognized change types cannot be queried or filtered; gaps leave migration auditing incomplete | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4552726518> |
| 3 | `schema_tables.uses_uuid_identity` flag — is it adequate as the discriminator for UUID-migration-eligible tables? | If the flag is insufficient, tables eligible for UUID migration may be silently omitted from the migration framework | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4552726518> |
| 4 | `uuid_migrations.verification_field` / `verification_source` necessity — are these fields accepted after Bastion's explanation? | Without confirmation, it is unclear whether these columns will be kept or dropped, affecting the migration table's final DDL | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728> |
| 5 | GPU `hardware_id` UUID field list completeness — is the extended `bios_notes` list and full GPU field set confirmed? | Thread ends without final sign-off from @gissf1; an unratified field list blocks correct UUID generation | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728> |
| 6 | `quantization_methods` seed multipliers — has @gissf1 flagged any multipliers as wrong? | Incorrect VRAM multipliers corrupt claim-query VRAM estimates and runner dispatch; seed SQL posted explicitly requesting review | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4519570310> |
| 7 | Model capability ingestion path — does `active_parameter_count_b` / `quantization` come from model card, CI artifact, or runner autodiscovery? | Source determines where validation errors surface and who is responsible for correcting them | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4513822952> |
| 8 | Result signing/verification scheme — deferred to a separate thread; which thread and when? | Standalone runner output authenticity is unverified until this is scoped | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4519072195> |

---

### interface_type

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | Seed file ships `name`-only rows vs full DDL — which columns are actually seeded? | The ratified DDL requires `bandwidth_peak_gb_s`, `description`, `interface_family`, `transfer_rate`, `lane_count`; if only `name` is seeded, foreign key consumers get incomplete records | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4570546987> |
| 2 | Hardware snapshot UUID field list (Q2) — which fields are included before @gissf1 confirms? | Agreement in principle without a confirmed field list leaves UUID generation underspecified; BIOS wait-states and memory interleaving were flagged as candidates | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533634217> |
| 3 | Disk layout for gpu_hardware / system_gpu_link (Q3) — Option A (inline) or Option B (separate per-GPU files)? | Option A is simpler and air-gap compatible; Option B is more query-efficient for shared GPU records; no final ruling | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533634217> |
| 4 | Hardware snapshot timing (Q5) — once per process start vs per run; what is the policy for network-connected GPU hot-swap? | Per-run snapshots are more accurate but costly; per-process is cheaper but misses mid-session changes; hot-swap case has no firm policy | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533634217> |
| 5 | Phase-2 migration framework — still in design; no detail or ticket in thread | Without scoping, dependents cannot plan around it | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4570546987> |

---

### model_descriptor

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | `creators.creator_id` UUID — should a source-site user ID be incorporated instead of or alongside homepage? | Homepage-primary was implemented but the source-site-ID question was not explicitly closed; inconsistent creator UUIDs across systems would break cross-host deduplication | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524573993> |
| 2 | Provisional UUID promotion — multi-provisional race (two hosts resolving same hash simultaneously) not explicitly addressed | Two uncoordinated hosts could create divergent records for the same model weights; collision handling for the race case is undefined | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524573993> |
| 3 | `models/pending/` subdirectory for stub isolation — design option or decided? | Without a decision, stub files and settled records share the same directory; naming collisions are a latent risk at scale | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4533408702> |
| 4 | Embedded GGUF metadata cross-check (runner startup validation) — deferred to P2; no concrete trigger or scope | Until scoped, runners cannot validate model file integrity at startup | #15, <https://github.com/Rethunk-AI/bakeoff/issues/15#issuecomment-4570345289> |
| 5 | URL-submit ingestion handler — deferred to P2; implementation excluded from closed scope | The agreed target architecture (URL → metadata pull → validate → store) is not implemented; manual/admin path only | #15, <https://github.com/Rethunk-AI/bakeoff/issues/15#issuecomment-4570345289> |
| 6 | `quantization_methods` seed multipliers cross-check — has Bastion's promised cross-check against llama.cpp MODELS.md been posted? | Multiplier discrepancy (`q4_k_m`: example showed 0.45, seed used 0.563) was flagged; no confirmation comment in thread | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4519570310> |

---

### model_trust

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | Trust-tier taxonomy for community/unverified models — no schema column, tier values, or enforcement rule settled | Without a taxonomy, community models from Ollama or HuggingFace are admitted with no trust signal; future trust-gating has no anchor | #14, <https://github.com/Rethunk-AI/bakeoff/issues/14#issuecomment-4503203471> |

---

### persistence_layout

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | `schema_tables.uuid_namespace` — stored in DB + backing disk file, or defined only as Go constants? | DB storage enables tooling lookup without code changes; Go constants are simpler but couple namespace to a build artifact; gissf1 said "schema_tables table as you suggested" but no implementation confirmation was given | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4572098058> |
| 2 | `schema_hops` table name vs `table_migrations` or merge into `schema_versions` — not explicitly ratified before #22 closed | Naming and structural location determine the DDL for the migration framework; blocked on #25 | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4573134793> |
| 3 | BIOS UUID field set — `memory_speed_mhz`, `memory_channels`, `memory_interleave_profile` proposed but not confirmed by @gissf1 | An unconfirmed field set means the hardware UUID computation is underspecified; results could silently diverge across implementations | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533647917> |
| 4 | `--strict-cache` default — warn vs abort on model source file drift; is the three-case policy (size change / mtime+same-size / mtime+same-content) confirmed as implemented? | Standalone vs queue-worker default split was proposed but not confirmed; implementation coverage not referenced in close-out | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524200643> |
| 5 | Prompt descriptor disk file format — deferred to scoring architecture thread; no ticket number assigned | Dependents (runner, evaluator) cannot implement prompt loading until format is decided | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524304026> |

---

### process_meta

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | Migration framework structural shape — how does `table_migrations` merge into `schema_versions`? (one row per table per version, JSONB array, or another form?) | No implementation may begin until this is answered in #25; wrong structure creates unbounded table growth or loses audit granularity | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4574254854> |
| 2 | UUID namespace migration strategy — "UUID changes to any table can simply be migrated to a new table using a new UUID namespace" flagged as open design question for #25 | Strategy affects whether existing hardware/model UUIDs are invalidated on schema bumps | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4574254854> |
| 3 | Queue-worker implementation — `run_queue` table shipped but opt-in worker loop in `runner.py` not yet implemented | Multi-runner operation is blocked until the worker loop is implemented | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4570344944> |
| 4 | Testing-queue issue status — Bastion noted it was "previously stated as 'opening now'"; no confirmation in thread | Dependent work (difficulty gating, prompt dispatch logic) cannot be tracked without a ticket | #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503669210> |

---

### run_hardware_metrics

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | `run_hardware_metrics` FK nullability — when and by whom is the NOT NULL promotion confirmed after backfill? | Nullable FKs allow rows with no hardware context; no criteria or owner defined for the promotion | #21, <https://github.com/Rethunk-AI/bakeoff/issues/21> |
| 2 | `run_model_metrics` schema — `model_id`, `task_id`, `prompt_id` remain TEXT placeholders pending FK resolution; is prompt a subset of task? | Load-bearing design question for scoring; deferred to #12 | #12, <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4480962683> |
| 3 | `gpu_event_seconds` CUDA/ROCm event API wiring — field exists, always `None`; no implementation timeline or trigger defined | CUDA kernel wall time cannot be reported until this is wired; Phase 2 deferral has no concrete trigger | #8, <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4519663607> |
| 4 | Multi-GPU / multi-system distribution — deferred to Phase 2 with no scope or trigger; forward-compatibility design noted | Future `run_gpu_usage` join table would require schema changes to `run_hardware_metrics`; no trigger criterion defined | #21, <https://github.com/Rethunk-AI/bakeoff/issues/21> |

---

### run_model_metrics

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | Task/prompt cardinality in a single run — does a run dispatch all prompts for a task or a subset? Does runner skip remaining prompts on failure? | Dependency/difficulty-gate logic was proposed but deferred; affects correctness of pass-rate calculations | #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4496938804> |
| 2 | Prompt dependency relative-reference resolution pass — two-pass scan (build difficulty map, then resolve `parent+1` expressions) was proposed but flagged as scanner complexity to be specced, not settled | Without a resolution pass spec, the scanner cannot be implemented correctly | #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4496938804> |
| 3 | `task.yaml` alias bulk-declaration syntax — `aliases:` block proposed as alternative to per-prompt YAML stubs; no schema or syntax finalized | Without a decision, intentional prompt duplication requires one stub file per alias | #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503203398> |
| 4 | Testing-queue issue — stated as "opening now" in an earlier comment but no ticket number or link appears in this thread | Dependent work (difficulty gate, runner scheduling) cannot be tracked | #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503669210> |
| 5 | Implementation sub-issues per table/component — explicitly listed as pending direction at thread close; no sub-issues opened | Build work cannot be assigned or parallelized until sub-issues exist | #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503669210> |
| 6 | `model_hash` NULLABLE vs NOT NULL — UNIQUE constraint present but nullability not resolved; @gissf1's comment implies it should be authoritative | A nullable `model_hash` on a deduplication-anchor column allows ghost records with no dedup signal | #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4495675210> |

---

### scheduling_queue

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | Model capability ingestion path — where does `active_parameter_count_b`, `quantization`, `model_source_size`, `model_hash` originate? | Capability-based claim matching is useless until the ingestion path is defined; source determines validation owner | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4512649799> |
| 2 | `quantization_methods` seed multiplier sign-off — @gissf1 asked for cross-checking against reference model file sizes (within 1% tolerance); no confirmation comment in thread | Incorrect multipliers silently corrupt VRAM estimates and runner dispatch | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524200643> |
| 3 | `BAKEOFF_MODEL_NAMESPACE` committed to `bench/constants.py` — is this commit confirmed visible and complete? | Without a committed namespace constant, UUID generation is inconsistent across environments | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524304026> |
| 4 | `--strict-cache` default (warn vs abort) — policy stated but not confirmed as implemented; no test coverage reference | Runners may silently accept stale model files in standalone mode if the policy is unimplemented | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524304026> |
| 5 | Multi-runner per host — hostname suffix scheme (`hostname`, `hostname-2`) proposed but not prioritized; concurrent startup collision risk unresolved | Two runners starting simultaneously on the same host could claim identical `runner_id`s | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4516038912> |
| 6 | Hardware-assignment scope — close-out deferred it to #22; the relationship between VRAM capability matching (settled here) and the broader hardware-assignment mechanism is undefined | Runners may over- or under-filter jobs if hardware assignment semantics differ between the two scopes | #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4570344944> |

---

### system_gpu_link

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | BIOS UUID field set — `memory_speed_mhz`, `memory_channels`, `memory_interleave_profile` proposed but not confirmed | Unconfirmed UUID field set means hardware fingerprinting is underspecified; implementations may diverge | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533647917> |
| 2 | `schema_versions` / UUID migration scripting language — YAML/Jinja2, AWK, Lua, or another? | Security and file-count explosion concerns raised; `url_template` expression mode and `record_migration_script` both depend on this choice | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553530231> |
| 3 | `tflops_sources.contact_url` multi-contact structure — single URL cannot represent multiple contacts per source | Schema cannot represent manager + engineer + PM contacts for a single source without redesign | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553530231> |
| 4 | `tflops_sources.url_template` complexity — can it handle GPU-architecture slug mappings, capitalisation transforms, and cross-field construction? | If the chosen scripting language cannot handle those cases, the field design needs revisiting | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553530231> |
| 5 | `hardware-snapshot/<uuid>.json` single-file layout (Q3) — Option A (combined system + GPU array) not explicitly confirmed; Option B (separate per-GPU files) not ratified | Ambiguity in snapshot layout means ingest code may diverge from the intended design | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533647917> |

---

### system_hardware

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | GPU UUID field set incomplete — @gissf1 noted "we probably need more fields here too" beyond PCI IDs + vram_mb + vram_type; no additional fields agreed | Incomplete UUID field set allows distinct GPU SKUs to hash to the same UUID | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728> |
| 2 | BIOS UUID whitelist additions — only `bar_size_mb` confirmed for Phase 1; @gissf1 noted other bus clock rates, multipliers, widths should be tracked | Whitelist additions require schema_version increment; open list means Phase 2 scope is undefined | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728> |
| 3 | `uuid_migrations.entity_type` — TEXT vs FK to `schema_tables(table_id)`; two inconsistent representations within the thread, never reconciled | Inconsistent representations in the thread make the final DDL ambiguous | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4568864985> |
| 4 | `schema_tables.uuid_namespace_v1` — DB storage vs Go constants; Bastion raised at end of final comment without receiving direction | Without direction, the critical seed artifact that drives UUID generation has no canonical home | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4568864985> |
| 5 | `tflops_sources.contacts` JSONB shape — JSONB array of `{type, value}` accepted for Phase 1 but awaiting confirmation from @gissf1 | Unconfirmed shape blocks implementation of the `tflops_sources` seed and ingest code | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553785030> |
| 6 | EAV approach for BIOS settings — @gissf1 proposed a lookup table for BIOS settings/hardware variants; deferred to Phase 2 with no design started | Without a Phase 2 design, duplicated large BIOS text records will proliferate in production data | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728> |
| 7 | Multi-processor systems — @gissf1 raised handling; deferred as "not urgent enough" with no design proposed | Multi-socket hosts cannot be correctly represented in `system_hardware` under the current schema | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728> |
| 8 | `power_limit_w` UUID inclusion — confirmed not-UUID for Phase 1; Phase 2 candidate with no threshold or trigger criterion defined | Without a trigger criterion, `power_limit_w` may never be promoted, even if results data shows divergence | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4559373879> |

---

### system_software

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | `system_software` deduplication threshold — Phase 2 full-column-hash dedup: UNIQUE constraint, separate hash column, or background job? | Without a decision, `system_software` rows proliferate unboundedly; storage and query cost grow proportional to run count | #19, <https://github.com/Rethunk-AI/bakeoff/issues/19> |
| 2 | `system_software_id` nullability in `run_hardware_metrics` — when and how is the NOT NULL promotion confirmed; who runs the backfill? | Nullable FK allows runs with no software context; no criteria or responsible party defined for the transition | #21, <https://github.com/Rethunk-AI/bakeoff/issues/21> |

---

### tflops_source

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | Scripting language for `url_template` expressions — Lua, Gonja, Pongo2, sprig, Go templates, or CEL? | `url_template` expression mode and `record_migration_script` both depend on this choice; without it, neither can be implemented | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4554217644> |
| 2 | `tflops_sources.url_template` complex-substitution capability — can it handle GPU-architecture slug mappings, capitalisation normalisation, cross-field construction? | If the chosen language cannot handle these cases, the field design needs revisiting before implementation | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4554217644> |
| 3 | Multiple contacts / contacts table (Phase 2) — `tflops_sources.contacts` JSONB accepted for Phase 1 only; normalised contacts table deferred without scope | Phase 2 contact normalization has no design, ticket, or trigger criterion | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4554217644> |

---

### uuids

| # | Open question (concise) | Trade-off / why it matters | Address (ticket, comment URL) |
|---|---|---|---|
| 1 | `table_migrations` final schema / merge into `schema_versions` — @gissf1 directed a merge but the exact structure was not resolved; handed off to #25 | All migration framework implementation is blocked; UUID namespace strategy is also unconfirmed in this context | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4574170103> |
| 2 | `schema_tables.uuid_namespace` — DB + backing disk file confirmed in principle but storage mechanism (column type, seeding, disk format) not fully specified before handoff to #25 | The critical namespace seed artifact has no canonical implementation spec | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4572098058> |
| 3 | GPU UUID field set completeness — @gissf1 noted more fields are needed beyond PCI IDs + vram_mb + vram_type; no "GPU UUID field list is final" confirmation issued | Without a ratified field list, distributed GPU UUID generation will be inconsistent | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533894066> |
| 4 | `bios_notes` UUID whitelist completeness — `bar_size_mb` confirmed for Phase 1; other bus clock rates/multipliers/widths noted as future candidates but no formal closure | Informal open list means Phase 1 scope is ambiguous; schema version increments cannot be planned | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728> |
| 5 | Multi-hop `uuid_migrations` chaining protocol — mechanics (how many hops, when to auto-resolve vs flag for operator) not confirmed before thread closed; first item for #25 | Without chaining protocol, migration tooling cannot handle records that span more than one schema version hop | #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4572098058> |

---

## Counts

**Total open questions: 74**

| Topic | Open questions |
|---|---|
| system_hardware | 8 |
| gpu_hardware | 8 |
| gpu_architecture | 6 |
| scheduling_queue | 6 |
| run_model_metrics | 6 |
| uuids | 5 |
| persistence_layout | 5 |
| interface_type | 5 |
| system_gpu_link | 5 |
| run_hardware_metrics | 4 |
| process_meta | 4 |
| model_descriptor | 6 |
| tflops_source | 3 |
| system_software | 2 |
| model_trust | 1 |
