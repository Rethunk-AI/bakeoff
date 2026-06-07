# Summary: run_hardware_metrics

## Final state

Schema as settled by the end of this thread (per 4495333862, ratified 4494986575, implementation confirmed 4519663607):

```
interface_type
──────────────────────────────────────────────────────
interface_type_id       SERIAL PRIMARY KEY
bandwidth_peak_gb_s     FLOAT NOT NULL
description             TEXT NOT NULL
interface_family        TEXT NULLABLE
transfer_rate           INT NULLABLE        -- PCIe: GT/s per lane
lane_count              INT NULLABLE        -- PCIe: lane width

gpu_hardware
──────────────────────────────────────────────────────
gpu_hardware_id                  SERIAL PRIMARY KEY
gpu_name                         TEXT NOT NULL
pci_device_id                    TEXT NULLABLE
pci_sub_device_id                TEXT NULLABLE
vram_total_mb                    INT NULLABLE
vram_type                        TEXT NULLABLE
memory_bus_width_bits            INT NULLABLE
memory_bandwidth_peak_gb_s       FLOAT NULLABLE
clock_memory_mhz                 INT NULLABLE
clock_graphics_boost_mhz         INT NULLABLE
peak_tflops_fp16                 FLOAT NULLABLE
tdp_watts                        INT NULLABLE
gpu_native_interface_type_id     INT NULLABLE REFERENCES interface_type

system_hardware
──────────────────────────────────────────────────────
system_hardware_id      SERIAL PRIMARY KEY
system_id               UUID NOT NULL UNIQUE
publisher_id            TEXT NOT NULL
cpu_model               TEXT NULLABLE
ram_total_gb            FLOAT NULLABLE

system_software
──────────────────────────────────────────────────────
system_software_id      SERIAL PRIMARY KEY
os                      TEXT NULLABLE
kernel_version          TEXT NULLABLE
python_version          TEXT NULLABLE
gpu_driver_version      TEXT NULLABLE
cuda_version            TEXT NULLABLE
rocm_version            TEXT NULLABLE
runner_version          TEXT NULLABLE

system_gpu_link
──────────────────────────────────────────────────────
system_hardware_id              INT NOT NULL REFERENCES system_hardware  -- PK part 1
slot_index                      INT NOT NULL                             -- PK part 2
PRIMARY KEY (system_hardware_id, slot_index)
gpu_hardware_id                 INT NOT NULL REFERENCES gpu_hardware
slot_native_interface_type_id   INT NULLABLE REFERENCES interface_type
actual_interface_type_id        INT NULLABLE REFERENCES interface_type
INDEX: (gpu_hardware_id)

run_hardware_metrics
──────────────────────────────────────────────────────
run_id                          TEXT NOT NULL
system_hardware_id              INT NULLABLE REFERENCES system_hardware  -- FK part 1 (via #21, nullable on first add)
slot_index                      INT NULLABLE                             -- FK part 2
FOREIGN KEY (system_hardware_id, slot_index) REFERENCES system_gpu_link
system_software_id              INT NULLABLE REFERENCES system_software
wall_clock_seconds              FLOAT NULLABLE
seconds_to_first_token          FLOAT NULLABLE
tokens_per_second               FLOAT NULLABLE
peak_vram_mb                    FLOAT NULLABLE
gpu_sm_utilization_pct          FLOAT NULLABLE
tflops_utilization_pct          FLOAT NULLABLE
cpu_cycles_elapsed              BIGINT NULLABLE
cpu_seconds_user                FLOAT NULLABLE
cpu_seconds_sys                 FLOAT NULLABLE
gpu_event_seconds               FLOAT NULLABLE    -- CUDA/ROCm cudaEventElapsedTime(); kernel wall time
gpu_weighted_seconds            FLOAT NULLABLE    -- wall_clock_seconds × mean(gpu_sm_utilization_pct / 100)
```

`run_model_metrics` was explicitly spun off to a new issue (#12) and is out of scope here (per 4476713600 + 4480962683).

## Notable / unusual decisions

- **Three-category schema split** — gissf1 proposed separating invariant hardware specs, per-run hardware metrics, and per-run model metrics into distinct tables (per 4450406819); Bastion formalised this into five tables (`interface_type`, `gpu_hardware`, `system_hardware`, `system_software`, `system_gpu_link`, `run_hardware_metrics`). Rationale: eliminates redundancy across runs on the same hardware; enables hardware-normalized leaderboard views without storing derived fields.

- **`system_software` extracted from `system_hardware`** — gissf1 objected to OS/kernel fields on the hardware table (per 4476713600); software is not a fixed property of hardware (same machine, different driver/kernel on different days). FK placed on `run_hardware_metrics.system_software_id`, not on `system_hardware`. Downstream: any cross-run comparison must join both hardware and software to rule out driver/kernel effects.

- **`gpu_native_interface_type_id` belongs to `gpu_hardware`, not `system_gpu_link`** — it describes the card's rated spec, a fixed property. `system_gpu_link` carries `slot_native_interface_type_id` (slot's rated max) and `actual_interface_type_id` (negotiated running state). The rename from `slot_actual_interface_type_id` to `actual_interface_type_id` was accepted because the negotiated state belongs to neither side alone (per 4474333279).

- **Limitation attribution logic uses per-axis comparison, not collapsed bandwidth** — `bandwidth_peak_gb_s` alone can mask cases where two interfaces share the same bandwidth number from different gen/width combos; `transfer_rate` and `lane_count` on `interface_type` allow per-axis (generation vs lane width) attribution of which device is constraining (per 4474333279).

- **Dual GPU timing fields instead of a boolean flag** — gissf1 directed storing both `gpu_event_seconds` and `gpu_weighted_seconds` independently rather than using a `gpu_seconds_is_direct` flag (per 4482149275). Rationale: enables post-hoc divergence queries; absence/presence of each field encodes the same information as the flag would. Divergence (`gpu_event_seconds - gpu_weighted_seconds`) surfaces scheduling and transfer overhead directly in SQL.

- **`gpu_event_seconds` / `gpu_weighted_seconds` naming** — gissf1 drove the word-order reordering (noun_descriptor_unit) and approved these exact names (per 4494986575). "weighted" was preferred over "sampled" because it describes the mathematical relationship (utilization-weighted derivation of wall time) rather than just the acquisition method.

- **`cost_usd` removed from storage** — agreed to be a derived value; computed at display time from `energy_wh × kwh_rate` or provider pricing × token counts (per 4450406819, applied 4462460688). Not stored in any table.

- **Tokens are retained as proxy despite critique** — gissf1 noted "tokens" is a vague, architecture-variable metric (per 4446310454); however `time_to_first_token` / `tokens_per_second` were retained because they remain useful latency signals even if not hardware-invariant normalization axes. `tflops_utilization_pct` serves as the hardware-invariant normalization.

- **`gpu_data_transfer_seconds` deferred** — per-inference host-to-device transfer for local models is kilobytes, subsumed by the `wall_clock_seconds - gpu_event_seconds` gap (per 4482502460). Deferred to Phase 2. Also noted: this project targets local LLMs only, not cloud-based frontier models (per 4492063588).

- **`gpu_event_seconds` always `None` at ship time** — CUDA/ROCm event API path not yet wired; field present to keep schema stable (per 4519663607). `gpu_weighted_seconds` is fully live.

- **All time fields in seconds with float precision** — gissf1 mandated unit consistency: `time_to_first_token_ms` → `seconds_to_first_token`, `cpu_time_user_ms` → `cpu_seconds_user`, `cpu_time_sys_ms` → `cpu_seconds_sys`, `gpu_wall_time_ms` → `gpu_event_seconds` (per 4476713600, confirmed 4480962683).

## Open / unresolved

- **`run_hardware_metrics` FK nullability** — issue #21 adds `system_hardware_id`, `slot_index`, `system_software_id` as nullable to avoid breaking existing rows, with a note to set NOT NULL "once backfill is confirmed clean or if starting fresh." No decision in the thread on when or by whom that NOT NULL promotion is confirmed. The acceptance criteria in #21 leaves this conditional.
  - address: #21, <https://github.com/Rethunk-AI/bakeoff/issues/21>

- **`run_model_metrics` schema** — explicitly deferred to #12; `model_id`, `task_id`, `prompt_id` remain TEXT placeholders pending FK resolution. Whether a prompt is a subset of a task and whether prompt data lives in DB or files were flagged as load-bearing open questions (per 4480962683).
  - address: #12, <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4480962683>

- **`gpu_event_seconds` CUDA/ROCm event API wiring** — field exists in schema, always `None` at end of thread (per 4519663607). No implementation timeline set; listed as deferred to Phase 2 without a concrete triggering condition.
  - address: #8, <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4519663607>

- **Multi-GPU / multi-system distribution** — deferred to Phase 2 with a forward-compatibility note: a future `run_gpu_usage` join table would reference `(system_hardware_id, slot_index)` per GPU per run, and `run_hardware_metrics` would reference `run_gpu_usage.run_id` instead of carrying the compound FK directly (per #21 body). No scope or trigger defined.
  - address: #21, <https://github.com/Rethunk-AI/bakeoff/issues/21>

## Cross-topic links

- **`system_gpu_link`** — compound FK `(system_hardware_id, slot_index)` anchors `run_hardware_metrics` to this table; schema designed in the same thread (per 4474333279, formalised in #21).
- **`system_software`** — FK `system_software_id` on `run_hardware_metrics`; table designed in this thread (per 4480962683), implementation tracked as #19 (referenced in #21 `Depends on`).
- **`interface_type`** — referenced by both `gpu_hardware.gpu_native_interface_type_id` and `system_gpu_link.slot_native_interface_type_id` / `actual_interface_type_id`.
- **`run_model_metrics`** — spun off to **#12** (run_model_metrics schema: models, tasks, prompts, and scoring); not covered in this thread.
