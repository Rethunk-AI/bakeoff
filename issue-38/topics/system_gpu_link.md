# Topic: system_gpu_link

Consolidated chat history (chronological, ascending comment-id). 18 entries. Verbatim quotes; attribution in each header.

## #8 comment 4469626687 — @gissf1 — 2026-05-17T06:39:58Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4469626687>

Sounds good.  Does it make sense to include vram memory bus width and channel count too so we can calculate theoretical peak memory bandwidth?  or is there another way to acquire the peak memory bandwidth rate (MB/s)?

It may also matter how the GPU interfaces with the rest of the system, so whether thats PCI Express Gen 1 x1 link vs a Gen 3 x16 link is a huge differnence.  And there are other interfaces than PCIe also; I doubt AGP is a contender here, but I know there are networked GPUs, CXL interfaces, SXM, OCuLink, and I suppose even USB4, or Thunderbolt.

Maybe it makes sense to have a "GPU hardware" schema for the GPU itself and then properties in the system object schema for how it is interconnected and identifying critical parts of the rest of the system?  That could also allow us to separate the specs of the system (CPU, RAM, motherboard chipset, etc) from the GPU.  the GPU will most likely be the primary performance contributor in most cases, but perhaps someone is using an RTX 4090 24GB on an old raspberry pi 2, which will likely limit the performance of the 4090 to some extent.

---

## #8 comment 4469686066 — @AlbinoGeek — 2026-05-17T07:07:49Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4469686066>

**Bastion response — 170707ZMAY26**

Both proposals are sound. Addressing each, then a schema draft.

---

**VRAM bus width and channel count — feasible, recommended**

nvidia-smi does not expose bus width as a structured query field. However, `pynvml.nvmlDeviceGetMemoryBusWidth()` returns it directly (integer, bits). Channel count is derivable: a 320-bit bus is 10 × 32-bit channels; a 256-bit bus is 8 × 32-bit channels; HBM stacks use 1024 bits per stack × N stacks. No need to store channel count separately — derive it at display time from bus width and memory type.

**Theoretical peak bandwidth formula:**

```
# GDDR (DDR factor = 2):
bandwidth_GB_s = (memory_clock_mhz × 2 × memory_bus_width_bits / 8) / 1000

# HBM (effective clock, not DDR — factor = 1):
bandwidth_GB_s = (memory_clock_mhz × memory_bus_width_bits / 8) / 1000
```

With `memory_bus_width_bits` and the existing `clock_memory_mhz` + `vram_type` fields, peak bandwidth is fully derivable at display time. No need to store it — computed and presented in the leaderboard view.

---

**GPU interface — detectable where possible, declared elsewhere**

nvidia-smi exposes PCIe link properties directly:

```
--query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max
```

`link.gen.current` vs `link.gen.max` detects PCIe slot downgrade (a Gen 4 card in a Gen 3 slot is an immediate flag — measurable bottleneck). `width.current` vs `width.max` detects mechanical lane limitation (x16 card in x4 slot).

NVLink: detectable via `nvmlDeviceGetNvLinkState()` — presence and link count.

SXM form factor: identifiable from GPU name string (H100 SXM, A100 SXM5). No separate query needed — the PCI device ID lookup already implies form factor for known GPUs.

CXL, OCuLink, USB4/Thunderbolt: not programmatically detectable from within the driver. These require user declaration in the config. Recommendation: accept an optional `gpu_interface_override` field in the runner config for non-PCIe topologies.

---

**Schema split — strong agree**

Your three-entity model maps cleanly:

| Schema | Contents | Detection |
|--------|----------|-----------|
| `gpu_hardware` | Die-level intrinsics: compute, VRAM type/size/bus width, theoretical bandwidth, TDP, clock profile | Auto via nvidia-smi + pynvml |
| `system_hardware` | CPU model, RAM capacity/speed, chipset, OS, kernel | `/proc/cpuinfo`, `psutil`, `dmidecode` (optional) |
| `gpu_system_interface` | PCIe gen/lanes (current + max), NVLink count, form factor, interface override | Auto where detectable; declared for exotic interconnects |

The RPi example is the correct motivating case. If PCIe gen or lane count are bottlenecked, wall_clock_seconds will diverge from `tflops_utilization_pct` in a predictable direction. Separating interface from die specs lets us flag that divergence as a known artifact of the measurement configuration.

---

**Proposed `gpu_hardware` additions:**

```json
{
  "memory_bus_width_bits": 320,
  "memory_bandwidth_peak_gb_s": 1008.0
}
```

`memory_bandwidth_peak_gb_s` stored (not derived at display time) because the formula requires `vram_type` to select the DDR factor — storage is simpler than re-implementing that at every display callsite.

**Proposed `gpu_system_interface` record:**

```json
{
  "pcie_gen_max": 4,
  "pcie_gen_current": 4,
  "pcie_width_max": 16,
  "pcie_width_current": 16,
  "is_pcie_slot_limited": false,
  "nvlink_count": 0,
  "form_factor": "PCIe",
  "interface_override": null
}
```

`is_pcie_slot_limited` = true when `gen_current < gen_max` OR `width_current < width_max`. Flagged in leaderboard display.

---

Will open sub-issues for the additions above if directed. If NOMAD approves the three-schema split as the target structure, recommend updating bakeoff-results#12 to reflect that structure before any further implementation work lands.

— Bastion

---

## #8 comment 4469822466 — @gissf1 — 2026-05-17T07:54:08Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4469822466>

I'm thinking gpu_system_interface and system_hardware can be merged.  gpu_hardware should remain separate, and there should be some reference between it and system_hardware.

I'm not clear on which direction they should be linked.  I tend to think system_hardware should point to the gpu_hardware record, but maybe it makes sense to have a join table to allow a many-to-many relationship?  If so, we could link multiple GPUs to a system, and multiple systems to a GPU.  Maybe let's discuss the pros and cons to various directions so I can give you some more guidance?

---

## #8 comment 4470208621 — @AlbinoGeek — 2026-05-17T09:53:11Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4470208621>

**Bastion response — 171005ZMAY26**

Merge of `gpu_system_interface` into `system_hardware` is correct. Interface data describes the relationship between a GPU and a host — it belongs in the link, not either entity. Addressing the three relationship options:

---

**Option A — `system_hardware.gpu_hardware_id` (FK, one GPU per system)**

Pros: simplest schema, simplest queries.

Cons: assumes one GPU per system. Breaks for multi-GPU workstations, dual-slot setups, server nodes with 4× or 8× GPUs. Also conflates the GPU *model* with a specific physical instance in a specific machine — `gpu_hardware` is intended as a model-level record, not an instance record. Two RTX 4090 cards in different machines should reference the same `gpu_hardware` row.

Not recommended. Adequate only if bakeoff is restricted to single-GPU consumer hardware forever.

---

**Option B — `gpu_hardware.system_hardware_id` (FK, GPU belongs to system)**

Wrong direction semantically. A GPU model is not "owned by" a system. This structure prevents the same GPU model from being referenced by multiple systems — every community member with an RTX 4090 would need a separate `gpu_hardware` row with identical specs. Defeats normalization entirely.

Not recommended.

---

**Option C — Join table `system_gpu_link` (many-to-many)**

`system_hardware` and `gpu_hardware` remain independent entity tables. `system_gpu_link` carries both the relationship and the interface fields (PCIe gen/width, NVLink count, form factor, interface override). This is where the "merged" interface data lives — it belongs on the relationship, not on either entity.

```
system_hardware       system_gpu_link              gpu_hardware
───────────────       ───────────────────────────  ────────────────
system_id (PK)  ←──  system_hardware_id (FK)  ──→ gpu_hardware_id (PK)
cpu_model             gpu_hardware_id (FK)          gpu_name
ram_total_gb          pcie_gen_current              vram_total_mb
os                    pcie_gen_max                  memory_bus_width_bits
...                   pcie_width_current            peak_tflops
                      pcie_width_max                ...
                      is_pcie_slot_limited
                      nvlink_count
                      form_factor
                      interface_override
                      slot_index  ← slot 0, slot 1 for multi-GPU
```

Pros:
- Multiple GPUs per system (multi-GPU workstation, server rack)
- Multiple systems per GPU model (community benchmarking at scale — many users with RTX 4090)
- Interface data (PCIe gen/lanes) lives on the relationship where it belongs
- `run_hardware_metrics` records reference `system_gpu_link.id` — captures both which GPU *and* which system/slot produced the run

Cons: join required for any query spanning GPU and system data. Acceptable — this is a known, bounded query pattern.

---

**Recommendation: Option C**

Two reasons the join table is load-bearing here:

1. Community benchmarking is the explicit use case. Multiple submitters with the same GPU model is not hypothetical — it is the expected steady state once the tool is in broader use. Normalization prevents storing redundant `gpu_hardware` rows per user.

2. Interface data (PCIe bottleneck detection) is only meaningful on the relationship. A standalone `system_hardware` table that embeds PCIe link data implies a system has exactly one GPU slot, which is architecturally incorrect.

Final proposed table structure:
- `gpu_hardware` — GPU intrinsics (die, VRAM, compute, thermals)
- `system_hardware` — host intrinsics (CPU, RAM, chipset, OS, `system_id`)
- `system_gpu_link` — relationship + interface (PCIe, NVLink, form factor, slot index)
- `run_hardware_metrics` references `system_gpu_link.id` + `gpu_hardware_id` — captures full context per run

Standing by for direction. Will open sub-issues to implement once schema is ratified.

— Bastion

---

## #8 comment 4470606665 — @gissf1 — 2026-05-17T12:19:59Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4470606665>

I agree that options A and B are bad for the reasons you suggested.  We will use option C, but I think it makes sense to move the fields describing the slot type into a reusable table of its own.  For example, "PCIe 3.0 16x" is a pretty well defined interface, which means we can normalize all the "full bandwidth" fields 

I think it makes sense to have an "interface_type" table with fields like:
pcie_gen_max, pcie_width_max, form_factor, description

Then we can remove those same fields from system_gpu_link and add another FK to interface_type_id.  interface_type_id does not need to be part of the unique/PK in that table, but the slot index should be.  Each card in a system should be some unique combination of system_hardware_id, gpu_hardware_id, and slot index.

The *_current fields could remain in the system_gpu_link table, or we could export them to the interface_type table and simply have "interface type" entries for all manner of degraded interfaces as well.  Having entries for degraded interfaces may be of benefit so we can provide descriptive text for those conditions that is tuned to common language (Example: "PCIe 3.0 16x slot running degraded at PCIe 2.0 8x signalling rate").

I don't know if nvlink is valuable, but perhaps we should consider how niche or practical it is before including it.

On the interface_type table, perhaps we should abstract those field names a bit to consider interfaces that are not based on a PCI Express standard?  The key bit of information being presented is the bandwidth numbers, not the PCI Express specifics.  If you prefer to keep that level of detail in a non description field, I'm open to using a more generic field to detail the interface-specific description of any interface bandwidth (so, the "2.0 8x" part of the description).  In this case, we store the bandwidth numbers in the data fields, but the description can still say things like "PCI Express 2.0 8x".  That may reduce the number of unnecessary fields and help to focus our data while remaining descriptive for user presentation.

---

## #8 comment 4470666757 — @AlbinoGeek — 2026-05-17T12:33:04Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4470666757>

**Bastion response — 171232ZMAY26**

Normalization of `interface_type` is correct. Addressing each point:

---

**`interface_type` table — generic bandwidth-first schema**

Agreed: field names should not assume PCIe. The table describes an interface *slot type*, and the only data invariant across all interface technologies is bandwidth and a human-readable description. Proposed schema:

```
interface_type
───────────────────────────────────────
interface_type_id   PK (serial/UUID)
bandwidth_peak_gb_s FLOAT NOT NULL      -- theoretical peak, both directions
description         TEXT NOT NULL       -- human-readable: "PCIe 4.0 x16", "SXM5", "NVLink 4.0 x2", "Thunderbolt 4"
interface_family    TEXT NULLABLE       -- grouping: "PCIe", "SXM", "CXL", "NVLink", "USB", "OcuLink"
```

`bandwidth_peak_gb_s` is the primary filterable metric. `description` carries the interface-specific detail (gen, width, protocol version) as a string — no further normalization needed. `interface_family` enables grouping without encoding PCIe specifics into column names.

Rows are pre-seeded for well-known interface types and pre-seeded for well-known degraded states. Not exhaustive — only states commonly observed in practice need entries.

---

**Where current (`*_current`) fields land**

Both native and actual interface are represented as `interface_type` rows. `system_gpu_link` holds two FKs:

```
system_gpu_link
───────────────────────────────────────────────────
system_hardware_id       FK → system_hardware
gpu_hardware_id          FK → gpu_hardware
slot_index               INT NOT NULL
PK: (system_hardware_id, gpu_hardware_id, slot_index)

native_interface_type_id FK → interface_type   -- card's rated max
actual_interface_type_id FK → interface_type   -- running state
is_slot_limited          BOOL GENERATED        -- native ≠ actual
```

When `native_interface_type_id ≠ actual_interface_type_id`, the leaderboard flags the slot as limited and surfaces the `description` from `actual_interface_type` as the explanation. Pre-seeded `interface_type` row for "PCIe 3.0 x16 (degraded from PCIe 4.0 x16)" carries the tuned description text you described — no runtime string construction needed at display time.

No `*_current` fields remain in `system_gpu_link`. The split is clean.

---

**NVLink — recommendation: defer, but reserve**

NVLink is narrow scope today — confined to RTX 3090/4090 with NVLink bridge (consumer), and H100/A100 SXM (datacenter). Community benchmarking is unlikely to encounter it at meaningful volume in Phase 1.

However, NVLink fundamentally changes the GPU-to-GPU topology: two physically linked GPUs can appear as a unified memory pool. This is not the same as two independent GPUs in `system_gpu_link`. It warrants a separate `gpu_gpu_link` table if we want to model it correctly.

Recommendation:
- Phase 1: do not implement `gpu_gpu_link`. No NVLink columns in any current table.
- Phase 1: the `interface_family = "NVLink"` row type is reserved in `interface_type` as a named constant, but no FK references it yet.
- Phase 2: add `gpu_gpu_link` with NVLink bandwidth and link count when a community use case materializes.

This avoids schema debt while not burning the namespace.

---

**Revised four-table structure:**

```
gpu_hardware         -- die-level intrinsics (VRAM, compute, clocks, TFLOPs)
system_hardware      -- host intrinsics (CPU, RAM, OS, system_id, publisher_id)
interface_type       -- named slot/interface types (bandwidth, description, family)
system_gpu_link      -- system↔GPU relationship + slot index + native/actual interface FK
```

`run_hardware_metrics` references `system_gpu_link.id` (compound or surrogate key). Full hardware context for any run is a three-way join: `system_gpu_link → system_hardware + gpu_hardware + interface_type`.

---

Will open sub-issues for implementation when directed. If NOMAD approves this structure, recommend updating bakeoff-results#12 to reflect the four-table target before any further schema implementation lands.

— Bastion

---

## #8 comment 4472485588 — @gissf1 — 2026-05-17T20:57:55Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4472485588>

In system_gpu_link, perhaps we should also have indexes on (system_hardware_id) and (gpu_hardware_id) since we will likely need to look up records by either of those fields independently.

Actually, rethinking system_gpu_link a bit, system_hardware_id and slot_index are the PKs for system_gpu_link.  A system only has a certain number of slots and they are sequential.  Each slot can be filled with any compatible GPU, which implies the GPU type is data about the slot.

Also, do we even need is_slot_limited?  we should be able to tell by simply looking at if the FK id's for the interface_type fields are equal; if actual != native then it must be degraded.  Maybe there is a performance reason to keep it though?

On that, I'm not sure if it makes more sense to have interface_type entries specific for degraded operation or simply concatenating it at runtime as: (actual_interface_type + "(degraded from" + native_interface_type +")").  I believe processing time for concatenating is negligible, but storing records for every possible combination of downgrade could become extensive.  I suppose a critical question would be: does bandwidth on a degraded link equal the same as the degraded link speed?  for example, if my "PCIe 3.0 16x" GPU is in a "PCIe 4.0 16x" slot, does the effective bandwidth equal that of a "PCIe 3.0 16x" slot in all scenarios?  I consider this because I could imagine a scenario where some newer generation has improved efficiency or signalling in such a way that even degraded links might be faster than their original native implementation.  Or inversely, that degraded links require some additional overhead making them actually slower than their actual link speed would imply.

Also, I think "native" of "native_interface_type" might not be specific enough - is that describing the system's native interface type, or the GPU's native interface type?  I think it should be a bit more specific.  Maybe "system_interface_type" or "gpu_interface_type"?  we should probably also add a FK field to the gpu_hardware table to declare the native interface_type for that GPU.  Obviously since there are multiple slots on a motherboard, each of which could have any of a variety of interface types, that needs to remain in the system_gpu_link table to define the value independently per slot.

run_hardware_metrics should probably reference whatever it is actually running on.  For now, it probably makes the most sense for run_hardware_metrics entries to reference the individual system_gpu_link entry since the other hardware information is accessible from there.  So it would connect the system_gpu_link id to the model used for a test run.

on NVLink, I'm not convinced that the gpu_gpu_link table is the best approach, but I agree with your phase 1 suggestion to ignore it for now.

If the system is capable of distributing a single model to run across multiple GPUs simultaneously, then we probably need a new table to define which system_gpu_link entries it was using for the test run.  This implies (in the most obvious example) a system_id and a range of slot indexes.

I suppose similarly, it's also theoretically possible to build an implementation that distributes a model across multiple physical systems too, so maybe when we encounter that, we can have a generalized "composite" system type that aggregates multiple system_gpu_link records into meta-systems and stores them in a new table for aggregate systems.

Much like the NVLink discussion, I don't think using multiple GPUs (or systems) for running a single model is a short term concern since I don't believe there is any hardware or software capable of doing that yet, so it probably makes more sense to put this in the backlog and plan this in more detail later rather than now.

I agree that we should update bakeoff-results soon, but we are still on hold there as well waiting for some report results to test UI things.  It probably makes the most sense to update the documentation there once we finalize the design planning here.

---

## #8 comment 4472501126 — @AlbinoGeek — 2026-05-17T21:04:56Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4472501126>

**Bastion response — 172104ZMAY26**

Addressing each point in order:

---

**Indexes on `system_gpu_link`**

Agreed. Since the PK is `(system_hardware_id, slot_index)` (see below), the composite PK index covers lookups by `system_hardware_id` already. `gpu_hardware_id` is data — no implicit index — and will be queried independently (e.g., "all systems running this GPU model"), so a standalone index on `gpu_hardware_id` is correct.

---

**PK revision: `(system_hardware_id, slot_index)`**

Correct. A slot is a fixed property of the system. At any point in time, a given slot holds at most one GPU — so the GPU identity is data about the slot, not part of the key. The PK from my prior schema `(system_hardware_id, gpu_hardware_id, slot_index)` was wrong: it would prevent the same GPU model from occupying two slots in the same system (two identical cards), which is a valid and common configuration.

Revised `system_gpu_link`:

```
system_gpu_link
───────────────────────────────────────────────────
system_hardware_id          FK → system_hardware
slot_index                  INT NOT NULL
PK: (system_hardware_id, slot_index)

gpu_hardware_id             FK → gpu_hardware      (indexed)
gpu_native_interface_type_id  FK → interface_type  -- GPU card's rated spec
slot_actual_interface_type_id FK → interface_type  -- what the slot is running
```

---

**`is_slot_limited` — drop it**

Derivable: `gpu_native_interface_type_id ≠ slot_actual_interface_type_id`. PostgreSQL `GENERATED ALWAYS AS` does not support subquery-based expressions, so a truly computed column is not viable here. Storing it requires a trigger. Neither is worth the complexity. Recommendation: drop the column entirely; compute the condition in queries and views. Cheap comparison on indexed FKs.

---

**Field naming: "native" ambiguity**

You are correct. "Native" is ambiguous — it could describe the system slot's native spec or the GPU card's native spec. Revised names applied above:

- `gpu_native_interface_type_id` — what the GPU card itself is rated for (GPU's spec)
- `slot_actual_interface_type_id` — what the slot is actually running at (observed state)

Clearer at a glance which side the constraint comes from.

---

**Degraded description — runtime concatenation vs stored rows**

The theoretical question is valid: does a "PCIe 3.0 x16 link in a PCIe 4.0 x16 slot" yield exactly PCIe 3.0 x16 bandwidth? In practice: yes, the slower side governs, and the bandwidth equals the lower-rated interface spec. Exceptions (signal integrity issues, driver overhead) would show up as reduced measured throughput in `run_hardware_metrics`, which is the right place for that data.

For display: recommend runtime concatenation rather than exhaustively pre-seeded degraded-state rows. The number of possible native/actual combinations is large; pre-seeding every pairing is impractical. The `description` field from each `interface_type` row is already human-readable. Presentation layer constructs: `"[slot_actual.description] (limited from [gpu_native.description])"` when native ≠ actual. `bandwidth_peak_gb_s` from `slot_actual_interface_type_id` is the authoritative number for filtering and comparison.

---

**`run_hardware_metrics` → `system_gpu_link`**

Agreed — reference the `(system_hardware_id, slot_index)` compound key (or a surrogate if added). Full hardware context for any run is then: `run_hardware_metrics → system_gpu_link → system_hardware + gpu_hardware + interface_type`.

---

**Multi-GPU runs — defer**

Agreed. When multi-GPU distribution becomes relevant, the natural extension is a `run_gpu_usage` join table: `(run_id, system_hardware_id, slot_index)` referencing `system_gpu_link`. That design is forward-compatible with the revised PK above. No changes needed now — log it as a Phase 2 schema extension.

---

**Multi-system runs — defer**

Agreed. The "composite system" concept for distributed runs across physical machines is valid but niche. No schema work in Phase 1.

---

**bakeoff-results documentation — hold**

Agreed. Will update once this design is finalized and ratified.

— Bastion

---

## #8 comment 4472607438 — @gissf1 — 2026-05-17T21:46:57Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4472607438>

I think system_gpu_link should have slot_native_interface_type describing the motherboard's slot native interface type.  The GPU's native link interface type should be a field in the gpu_hardware table.  The slot_actual_interface_type makes sense to remain in system_gpu_link, but can probably be named as actual_interface_type, since this is negotiated and synchronized by the hardware to be consistent across all related physical objects (system-slot, gpu), not just the slot.

If we are going to identify "limited from" in the interface description, maybe it also makes sense to identify which device is being constrained or contraining?  Perhaps an older GPU is slowing the motherboard slot, or perhaps an older motherboard is slowing the GPU from it's potential.  Both cases should be indicated, and especially indicating which is doing the limiting.  So things like "PCIe 3.0 x16 GPU in a PCIe 4.0 x16 motherboard slot" could be indicated as "PCIe 3.0 x16 GPU interface (limiting PCIe 4.0 x16 system slot)"; or, swapping hardware capabilities, "PCIe 3.0 x16 system slot (limiting PCIe 4.0 x16 GPU interface)".  I think it makes slightly more sense to indicate the actual type first, rather than always keeping either of the system's- or gpu's- type identified first.  There is also a condition where there is a mix of downgrading, like "PCIe 3.0 x16 GPU in a PCIe 4.0 x8 motherboard slot"; I believe this degrades to PCIe 3.0 x8 performance, and if so this description could say something like "PCIe 3.0 x8 link (limited by PCIe 4.0 x8 system slot and PCIe 3.0 x16 GPU interface)".

If there's nothing else, maybe you could provide me a dump of the resulting schema for all the tables, so I can indicate any last minute concerns.  If I think everything is good from there, I can reply with confirmation and we can proceed with the next step.

---

## #8 comment 4474333279 — @AlbinoGeek — 2026-05-18T04:25:19Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4474333279>

**Bastion response — 180502ZMAY26**

All three points addressed. Schema corrections applied, attribution logic defined, full dump follows.

---

**Field corrections — confirmed**

`gpu_native_interface_type_id` moves from `system_gpu_link` to `gpu_hardware`. This is the right home — it describes a fixed property of the GPU card, not the relationship between card and slot.

`system_gpu_link` now carries:
- `slot_native_interface_type_id` — the motherboard slot's rated spec (what that physical slot can do at maximum)
- `actual_interface_type_id` — negotiated running state, synchronized across slot and GPU by hardware at link-up

Rename from `slot_actual_interface_type_id` to `actual_interface_type_id` accepted. "Actual" is accurate — it belongs to neither side alone; it is what both sides agreed to.

---

**Limitation attribution logic**

Determining which device is constraining requires comparing both axes (generation and lane width) independently, not just `bandwidth_peak_gb_s`. Bandwidth collapses two independent variables into one number — two interfaces can share the same bandwidth but from different gen/width combinations, both degrading the other on different axes. Example: PCIe 3.0 x16 GPU in a PCIe 4.0 x8 slot — `actual` degrades to PCIe 3.0 x8; both devices constrain on one axis each.

Recommendation: add `pcie_gen` and `pcie_width` (or their generic equivalents — `transfer_rate` and `lane_count`) as nullable integer fields to `interface_type`. Non-PCIe interfaces leave these null; comparison is then on `bandwidth_peak_gb_s` only. For PCIe, both fields are populated and the comparison is per-axis.

Attribution rules for display:

| Condition | Description format |
|-----------|-------------------|
| `slot_native.bandwidth < gpu_native.bandwidth` | `"[actual] link (PCIe [slot_native] system slot limiting PCIe [gpu_native] GPU interface)"` |
| `gpu_native.bandwidth < slot_native.bandwidth` | `"[actual] link (PCIe [gpu_native] GPU interface limiting PCIe [slot_native] system slot)"` |
| Both constrain on different axes | `"[actual] link (limited by PCIe [slot_native] system slot and PCIe [gpu_native] GPU interface)"` |
| native = actual on both sides | No annotation — no degradation |

Display order: actual type first, attribution parenthetical second. Matches your stated preference.

For the mixed case (PCIe 3.0 x16 GPU in PCIe 4.0 x8 slot → actual PCIe 3.0 x8): `"PCIe 3.0 x8 link (limited by PCIe 4.0 x8 system slot and PCIe 3.0 x16 GPU interface)"` — both constraining devices named explicitly.

---

**Full schema dump**

```
interface_type
──────────────────────────────────────────────────────────
interface_type_id       SERIAL PRIMARY KEY
bandwidth_peak_gb_s     FLOAT NOT NULL
description             TEXT NOT NULL           -- "PCIe 4.0 x16", "SXM5", "Thunderbolt 4"
interface_family        TEXT NULLABLE           -- "PCIe", "SXM", "CXL", "NVLink", "USB", "OCuLink"
transfer_rate           INT NULLABLE            -- PCIe: GT/s per lane (e.g. 16 for Gen4); null for non-PCIe
lane_count              INT NULLABLE            -- PCIe: lane width (e.g. 16); null for non-PCIe
```

```
gpu_hardware
──────────────────────────────────────────────────────────
gpu_hardware_id                  SERIAL PRIMARY KEY
gpu_name                         TEXT NOT NULL
pci_device_id                    TEXT NULLABLE            -- "0x2684"
pci_sub_device_id                TEXT NULLABLE            -- board partner variant
vram_total_mb                    INT NULLABLE
vram_type                        TEXT NULLABLE            -- "GDDR6X", "HBM2e", etc.
memory_bus_width_bits            INT NULLABLE
memory_bandwidth_peak_gb_s       FLOAT NULLABLE
clock_memory_mhz                 INT NULLABLE
clock_graphics_boost_mhz         INT NULLABLE
peak_tflops_fp16                 FLOAT NULLABLE
tdp_watts                        INT NULLABLE
gpu_native_interface_type_id     INT NULLABLE REFERENCES interface_type  -- card's rated spec
```

```
system_hardware
──────────────────────────────────────────────────────────
system_hardware_id      SERIAL PRIMARY KEY
system_id               UUID NOT NULL UNIQUE     -- stable per-host UUID (generated at first run)
publisher_id            TEXT NOT NULL            -- submitting user/account
cpu_model               TEXT NULLABLE
ram_total_gb            FLOAT NULLABLE
os                      TEXT NULLABLE
kernel_version          TEXT NULLABLE
```

```
system_gpu_link
──────────────────────────────────────────────────────────
system_hardware_id              INT NOT NULL REFERENCES system_hardware
slot_index                      INT NOT NULL
PRIMARY KEY (system_hardware_id, slot_index)

gpu_hardware_id                 INT NOT NULL REFERENCES gpu_hardware
slot_native_interface_type_id   INT NULLABLE REFERENCES interface_type  -- slot's rated max
actual_interface_type_id        INT NULLABLE REFERENCES interface_type  -- negotiated running state

INDEX: (gpu_hardware_id)
```

```
run_hardware_metrics
──────────────────────────────────────────────────────────
run_id                          TEXT NOT NULL              -- references run record
system_hardware_id              INT NOT NULL
slot_index                      INT NOT NULL
FOREIGN KEY (system_hardware_id, slot_index) REFERENCES system_gpu_link

wall_clock_seconds              FLOAT NULLABLE
time_to_first_token_ms          FLOAT NULLABLE
tokens_per_second               FLOAT NULLABLE
peak_vram_mb                    FLOAT NULLABLE
gpu_sm_utilization_pct          FLOAT NULLABLE
tflops_utilization_pct          FLOAT NULLABLE
cpu_cycles_elapsed              BIGINT NULLABLE
cpu_time_user_ms                FLOAT NULLABLE
cpu_time_sys_ms                 FLOAT NULLABLE
gpu_wall_time_ms                FLOAT NULLABLE            -- optional, runner-dependent
```

```
run_model_metrics
──────────────────────────────────────────────────────────
run_id                          TEXT NOT NULL PRIMARY KEY
model_id                        TEXT NOT NULL
task_id                         TEXT NOT NULL
prompt_id                       TEXT NOT NULL
score                           FLOAT NULLABLE
pass_fail                       BOOLEAN NULLABLE
flops_per_token_theoretical     BIGINT NULLABLE
```

---

Omitted from dump (out of scope for this design thread): `hardware_specs` seed data, `interface_type` pre-seeded rows, `run_model_metrics` index strategy. Those are implementation details, not schema design questions.

Multi-GPU and multi-system distribution — deferred per earlier agreement. The `run_hardware_metrics` compound FK on `(system_hardware_id, slot_index)` is forward-compatible with a `run_gpu_usage` join table when that work is scoped.

Standing by for ratification or final concerns.

— Bastion

---

## issue #20 body — @AlbinoGeek — 2026-05-22T19:52:27Z
<https://github.com/Rethunk-AI/bakeoff/issues/20>

**Parent:** #8 — Additional Performance Metrics
**Depends on:** #17 (`interface_type`), #18 (`gpu_hardware`), #19 (`system_hardware`)

Implement the `system_gpu_link` join table as ratified in #8. This table models the relationship between a physical host and a GPU in a specific slot, including the interface type information.

## Schema

```sql
CREATE TABLE system_gpu_link (
    system_hardware_id             INT NOT NULL REFERENCES system_hardware,
    slot_index                     INT NOT NULL,
    PRIMARY KEY (system_hardware_id, slot_index),

    gpu_hardware_id                INT NOT NULL REFERENCES gpu_hardware,
    slot_native_interface_type_id  INT NULLABLE REFERENCES interface_type,  -- motherboard slot's rated max
    actual_interface_type_id       INT NULLABLE REFERENCES interface_type   -- negotiated running state
);

CREATE INDEX ON system_gpu_link (gpu_hardware_id);
```

## Key design decisions (from #8 thread)

- PK is `(system_hardware_id, slot_index)` — a slot is a fixed property of the system. The GPU occupying it is data, not part of the key. Two identical GPUs in the same system each get their own slot row.
- `gpu_hardware_id` is an FK carrying the GPU model identity, indexed for efficient lookups of "all systems running GPU model X".
- `is_slot_limited` is **not stored** — derivable as `slot_native_interface_type_id != actual_interface_type_id`. Compute in queries and views.
- NVLink deferred to Phase 2 — no NVLink columns.

## Auto-detection requirements

Harness populates `system_gpu_link` from nvidia-smi at startup:

- **`slot_index`** — GPU index from nvidia-smi (`--id=0`, `--id=1`, etc.). Multi-GPU systems get one row per GPU.
- **`gpu_hardware_id`** — FK to the `gpu_hardware` row detected/created in #18
- **`slot_native_interface_type_id`** — FK to `interface_type` matching `pcie.link.gen.max × pcie.link.width.max` from `nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max`
- **`actual_interface_type_id`** — FK to `interface_type` matching `pcie.link.gen.current × pcie.link.width.current` from `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`

When `slot_native != actual`, the slot is running at reduced capability. The display layer generates the limitation description at runtime from both interface_type rows' `description` fields.

## Deduplication / upsert

`(system_hardware_id, slot_index)` uniquely identifies a slot. On subsequent runs, upsert on this PK — update `gpu_hardware_id` and interface FKs if the GPU in the slot has changed.

## Acceptance criteria

- [ ] Migration creates `system_gpu_link` with correct PK, FKs, and index on `gpu_hardware_id`
- [ ] Harness auto-populates one row per GPU at startup (supports multi-GPU systems)
- [ ] `slot_native_interface_type_id` and `actual_interface_type_id` are resolved from seeded `interface_type` rows
- [ ] Upsert behavior: existing slot rows updated if GPU changes, not duplicated
- [ ] Migration is reversible

— Bastion

---

## issue #21 body — @AlbinoGeek — 2026-05-22T19:52:31Z
<https://github.com/Rethunk-AI/bakeoff/issues/21>

**Parent:** #8 — Additional Performance Metrics
**Depends on:** #20 (`system_gpu_link`), #19 (`system_software`)

Update `run_hardware_metrics` to reference `system_gpu_link` (via compound FK) and `system_software`. This wires the per-run hardware measurements to the full hardware context established by the prior sub-issues.

## Schema change

Add two FK references to the existing `run_hardware_metrics` table:

```sql
ALTER TABLE run_hardware_metrics
    ADD COLUMN system_hardware_id INT NULLABLE,
    ADD COLUMN slot_index         INT NULLABLE,
    ADD COLUMN system_software_id INT NULLABLE REFERENCES system_software,
    ADD CONSTRAINT fk_run_hardware_metrics_system_gpu_link
        FOREIGN KEY (system_hardware_id, slot_index)
        REFERENCES system_gpu_link (system_hardware_id, slot_index);
```

Nullable on first addition to avoid breaking existing rows; set NOT NULL once backfill is confirmed clean or if starting fresh.

## Run context binding

At run time, the harness records:
- `(system_hardware_id, slot_index)` — identifies the `system_gpu_link` row active for this run (which GPU in which slot on which host)
- `system_software_id` — the `system_software` row created at startup for this run (software environment snapshot)

Both must be populated before inserting any `run_hardware_metrics` rows.

## Query path for full hardware context

A three-way join from any `run_hardware_metrics` row gives complete hardware context:

```sql
SELECT r.*, sg.*, gw.*, sh.*, ss.*, it_native.description, it_actual.description
FROM run_hardware_metrics r
JOIN system_gpu_link sg ON (r.system_hardware_id, r.slot_index) = (sg.system_hardware_id, sg.slot_index)
JOIN gpu_hardware gw ON sg.gpu_hardware_id = gw.gpu_hardware_id
JOIN system_hardware sh ON sg.system_hardware_id = sh.system_hardware_id
JOIN system_software ss ON r.system_software_id = ss.system_software_id
LEFT JOIN interface_type it_native ON sg.slot_native_interface_type_id = it_native.interface_type_id
LEFT JOIN interface_type it_actual ON sg.actual_interface_type_id = it_actual.interface_type_id;
```

## Multi-GPU future compatibility

This design is forward-compatible with a future `run_gpu_usage` join table for multi-GPU runs. When that is scoped, `run_gpu_usage` will reference `(system_hardware_id, slot_index)` per GPU used per run; `run_hardware_metrics` will reference `run_gpu_usage.run_id` instead of carrying the compound FK directly. No schema changes to the other tables required.

## Acceptance criteria

- [ ] Migration adds `system_hardware_id`, `slot_index`, and `system_software_id` to `run_hardware_metrics`
- [ ] FK constraint to `system_gpu_link` is enforced
- [ ] Harness populates both FKs on every run record insert
- [ ] Existing test suite passes (no regression from FK addition)
- [ ] Migration is reversible

— Bastion

---

## #8 comment 4522303209 — @AlbinoGeek — 2026-05-22T19:53:06Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4522303209>

**Bastion response — 221947ZMAY26**

Sub-issues opened. Implementation dependency order:

| Issue | Title | Depends on |
|-------|-------|-----------|
| #17 | `interface_type` lookup table + seed data | — |
| #18 | `gpu_hardware` table + auto-detection | #17 |
| #19 | `system_hardware` + `system_software` tables | — |
| #20 | `system_gpu_link` join table + PCIe detection | #17, #18, #19 |
| #21 | `run_hardware_metrics` FK wiring | #20, #19 |

#17 and #19 have no dependencies and can proceed in parallel. #18 requires #17 first. #20 requires #17 + #18 + #19. #21 is the terminal step.

Standing by for implementation direction or priority assignment.

— Bastion

---

## issue #22 body — @AlbinoGeek — 2026-05-25T09:53:53Z
<https://github.com/Rethunk-AI/bakeoff/issues/22>

## Context

Hardware schema locked in #17 (#18, #19, #20, #21). This thread analyses which hardware tables warrant disk files, what the layout should be, and how hardware records relate to run disk files.

Follows the disk persistence pattern established in #15 for models/tasks/prompts/runs: disk files carry `schema_version`, `created_at`, `updated_at`; FK columns collapse to embedded objects or UUID/ID references.

---

## Tables in scope

| Table | Issue | Purpose |
|-------|-------|---------|
| `interface_type` | #17 | Lookup — GPU bus type enum |
| `system_hardware` | #19 | Host CPU/RAM snapshot per run |
| `system_software` | #19 | Host OS/driver snapshot per run |
| `gpu_hardware` | #18 | Individual GPU slot spec |
| `system_gpu_link` | #20 | Many-to-many: GPUs in a system snapshot |
| `run_hardware_metrics` | #21 | Per-run hardware context (links run → hw snapshot) |

---

## Open questions

**Q1 — Lookup table disk files**

`interface_type` is a small admin-controlled enum (10 rows). Should it have a seed JSON file (like `seeds/quantization_methods.json`) or is it embedded in schema.sql only?

Proposed: seed file `seeds/interface_types.json`, parallel to existing seeds. Admin-review-gated like model_architectures/model_file_formats.

**Q2 — Hardware snapshot identity**

`system_hardware` and `system_software` rows are generated at run time by the runner probing the host machine. No UUID — they use SERIAL PKs.

Options for disk identity:
- A. Derive a deterministic UUID from the hardware fingerprint fields (cpu_model + cpu_cores + ram_gb + motherboard hash) — allows dedup across runs without DB
- B. Use DB-assigned SERIAL ID embedded in the run disk file — simpler, requires DB round-trip before file is self-contained
- C. No separate disk file for hardware — embed the hardware snapshot inline in the run disk file

**Q3 — gpu_hardware + system_gpu_link disk layout**

A system snapshot involves one or more GPU slots. Should the disk representation be:
- A. A single `hardware-snapshot/<uuid>.json` file containing both system_hardware fields and an embedded `gpus: [...]` array (inlined from system_gpu_link + gpu_hardware) — single file per snapshot
- B. Separate `gpu_hardware/<id>.json` files referenced by ID from the snapshot — allows GPU spec dedup across snapshots

**Q4 — run_hardware_metrics relationship to run disk file**

`run_hardware_metrics` is 1:1 with `runs` (one row per run). Should the hardware context be:
- A. Embedded inline in `runs/<run_id>.json` as a `hardware:` block — avoids a separate file, run file is self-contained
- B. A separate `run-hardware/<run_id>.json` file — parallel to the run, referenced by run_id

**Q5 — Runner hardware reporting timing**

Hardware snapshot is captured by the runner at claim time or job start. Should the runner:
- A. Create/update the hardware snapshot once per process start (reuse across runs in same session)
- B. Snapshot hardware fresh per run (detects GPU slot changes mid-session)

---

Tagging @gissf1 for design input per the pattern established in #15.

— Bastion

---

## #22 comment 4533647917 — @AlbinoGeek — 2026-05-25T10:46:41Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533647917>

**Bastion response — 251044ZMAY26**

@gissf1 — synthesizing all five answers.

---

**Q1 — interface_type seed file: CONFIRMED**

Agreed. `seeds/interface_types.json` will be added alongside the other lookup seed files. Admin-review-gated, same pattern as `model_architectures` and `model_file_formats`.

---

**Q2 — Hardware snapshot identity: A + C combined**

Confirmed reading. The two options are not mutually exclusive:

- Hardware specs are embedded inline in the results disk file (option C basis — portable, air-gap compatible, no round-trip required)
- `hardware_id` UUID is computed deterministically from those embedded specs at ingest time (option A UUID generation) — never stored in the disk file, derived on demand

This means disk files are fully self-contained; the UUID is an ingestion artifact, not a field the runner needs to compute or store.

**Proposed field list for `hardware_id` UUID input:**

System fields:
- `cpu_model` — CPU make/model string
- `cpu_cores` — physical core count
- `cpu_threads` — logical thread count
- `ram_gb` — total RAM in GB
- `motherboard` — motherboard make/model

GPU fields (sorted list of tuples, one per slot):
- `gpu_model` — GPU make/model string
- `vram_mb` — VRAM in MB
- `slot_index` — PCIe slot position

**BIOS / performance-affecting settings** — you raise a valid point. These affect benchmark outcomes and should be part of the UUID if they materially differentiate two otherwise-identical machines. Proposed additional fields (all optional, null if not set or auto):

| Field | Description |
|-------|-------------|
| `memory_speed_mhz` | Active memory clock rate |
| `memory_channels` | Active memory channel count (1 = single, 2 = dual, 4 = quad) |
| `memory_interleave_profile` | XMP / EXPO / DOCP profile name or `manual` |
| `bios_notes` | Free-form for other performance-affecting settings (e.g., fTPM off, SMT off) |

`memory_speed_mhz`, `memory_channels`, `memory_interleave_profile` — include in UUID input. `bios_notes` — stored in snapshot but excluded from UUID (free-form field not stable for dedup). Does this field list look right, or are there other BIOS settings you want to capture or include in UUID generation?

---

**Q3 — GPU/system snapshot disk layout: clarifying Option A, then recommendation**

Option A as proposed: a single `hardware-snapshot/<uuid>.json` that merges system fields and GPU list into one document — no separate per-GPU files at the disk level. Example:

```json
{
  "schema_version": 1,
  "hardware_id": "<uuid5>",
  "cpu_model": "AMD Ryzen 9 7950X",
  "cpu_cores": 16,
  "cpu_threads": 32,
  "ram_gb": 64.0,
  "motherboard": "ASUS ROG Crosshair X670E",
  "gpus": [
    {"slot_index": 0, "gpu_model": "AMD Radeon RX 7900 XTX", "vram_mb": 24576, "interface": "PCIe 4.0 x16"}
  ]
}
```

Option B would split that into `gpu_hardware/<gpu_uuid>.json` and a separate system snapshot referencing GPU UUIDs.

**Recommendation given Q4 (run file embeds full hardware block):** Option A is the right shape for the `hardware-snapshot/` dedup cache — it mirrors exactly what gets embedded in the run file. At ingest, the pipeline:
1. Extracts the hardware block from the run file
2. Computes `hardware_id` and per-GPU UUIDs
3. Writes `hardware-snapshot/<hardware_id>.json` if not already present (dedup cache)
4. Upserts DB rows for `system_hardware`, `gpu_hardware`, `system_gpu_link`

Separate `gpu_hardware/<gpu_uuid>.json` files (Option B) add a level of indirection that only pays off when many snapshots share the same GPU and you want to avoid re-embedding the spec. That optimization can land as the dedup enhancement in Phase 2 alongside the sparse run format.

---

**Q4 — run_hardware_metrics / run disk file: CONFIRMED**

Option A confirmed. Run file embeds full hardware block inline. Portable, self-contained, deduplicable at ingest. Sparse/reference format deferred to Phase 2.

---

**Q5 — Runner hardware snapshot timing: CONFIRMED with caveat**

Option A confirmed — snapshot once at process start, reuse across runs in same session. Consumer hardware does not hot-swap GPUs mid-session.

On interfaces: Thunderbolt 3/4, USB4, and eGPU enclosures are theoretically hot-swap capable (OS can detach/reattach the device). In practice, running a GPU benchmark mid-swap is not a real scenario. NVLink/SXM topologies cannot hot-swap. PCIe cannot hot-swap without platform-specific BIOS support.

Recommendation: Option A for Phase 1. If an operator is running on an eGPU enclosure and wants per-run snapshotting, they can restart the runner process. No need for a per-run snapshot path in the initial implementation.

---

**Open items for your confirmation:**

1. BIOS field list above — is `memory_speed_mhz`, `memory_channels`, `memory_interleave_profile` the right set for UUID inclusion? Any fields to add or remove?
2. Option A hardware-snapshot layout — does the single combined file (system + GPUs embedded) make sense as the disk dedup format, given Option B is deferred?

— Bastion

---

## #22 comment 4553530231 — @gissf1 — 2026-05-27T10:04:55Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553530231>

RE: bios_notes UUID whitelist:
- pcie_gen_override - wouldn't this show up as if the board is limiting the GPU's PCIe rate?  If so, I don't see a need to include it since it's encompassed in the interface link table.  In any case, I don't think it's something to include in the GPU UUID.  I'm not sure about how it fits into the system_hardware table though.
- power_limit_w	- this is surely useful to store, but I'm not sure it's needed in the UUID for phase 1.
- Your "Open question": I believe the join table's measured link width should fully encompass any motherboard specific bus width overrides.  Do you have any example hardware situations or circumstances that you think it might not appropriately represent?

RE: BIOS settings as lookup table (EAV approach):  Agreed, phase 2.

RE: schema_versions: a change I anticipate that doesn't fit the example would be adding or modifying a field where data needs to be mapped from some other place or data:
- another field
- text blob containing output from commands like `lspci`, `cat /proc/cpuinfo`, or `dmidecode`
- a substring of another source
- recomputing a field value with different mapping functions
- changing the sources
- changing the hash function (for example, maybe we will change from sha256 to sha512 hashing)
- adding a new field based on a mapping of an old field
- a combination or manipulation of the above (for example, mapping a substring of another field to a set of values, which are then concatenated with another substring from another field.)

I feel like this whole thing almost needs to be a programming language.  Would it make sense to use something like YAML's parser (Jinja2?) to acquire data and specify the details?  Maybe AWK or LUA?  I feel like JavaScript might be too insecure for this use case, but do you have another suggestion that is effective to store in a database to reference the table data, processing efficient and limits security risk?  or maybe we don't need to store it in the database if we can find a way to store it externally in files that isn't going to become a nightmarish number of files over time.

RE: uuid_migrations: I don't think verification_field and verification_source give you what you need.  As described above, we don't know ahead of time what data we might need to compute the field data, and it may be multiple sources or a complex interaction.  I think the correct action is to use the calculation programming code as described above to generate the new field values from any existing data, or explain how to acquire that data externally for automation to retrieve it.  We also may add a new field that is not present in the existing data.

I'm ok with the verified field to indicate that we have confirmed the UUID migration record is valid to begin migrating records automatically, rather than simply flagging them.  Perhaps you can pick a more descriptive name though?  I'm thinking something like `is_migrating`, `allow_auto_migration`, `begin_migrating`, or `migration_started`, but I'm not sold on any of those, so I'm open to more suggestions.

RE: proposed schema_tables: Approved.

RE: schema_tables verification_source / resolution_strategy: I still don't think that is the right approach.  I think the complexity of the changes could require more fields and manipulation than the simple mapping you are implying in your example.

RE: GPU fields: tflops_sources:
- contact_url: I assume this could be things like `mailto:` or `tel:` links, but what if we want to store multiple contact methods (for example, mailing address + email + phone number + website contact form address) for a single source's contact?  or, similarly, what if we have multiple contacts for a single source (for example: a manager, an engineer, and a project manager)?
- url_template: what if this needs something like gpu_architecture?  or some manipulation like capitalization changes, substrings, and/or a mapping to certain slugs?  I feel like this is related to the uuid_migrations scripting language requirements.

RE: Open items for your confirmation:
- bar_size_mb - Confirmed.
- schema_tables.uses_uuid_identity flag: I think this is adequate for determining which tables have UUIDs and which have another form of PK.
- all others, see notes above.

---

## issue #38 body — @gissf1 — 2026-06-05T08:56:56Z
<https://github.com/Rethunk-AI/bakeoff/issues/38>

## What happened

I was looking at `schema/schema.sql` in the latest version of the bakeoff project and noticed many discrepancies in table designs compared to what we discussed.

## Expected

RE: GPU fields: what happened to all these fields in the gpu_hardware table? (See #18)
- pci_vendor_id
- pci_device_id
- pci_subsystem_vendor_id
- pci_subsystem_device_id
- vram_type (FK)
- gpu_architecture (FK)
- memory_bandwidth_peak_gb_s
- tdp_w

Are others missing or incorrect?

And the tflops_source table?
and the vram_type table?

And in system_gpu_link, where are these fields? (see #20)
- slot_native_interface_type_id
- actual_interface_type_id

and the system_hardware table? (see #19)
and the system_software table? (see #19)

and a bunch of fields in interface_types too (see #17)
Also, on #17, transfer_rate should probably be renamed to lane_transfer_rate to be more clear on its intent.

I don't think run_hardware_metrics is completely correct either. (#8, #21)  I don't think #21 covers all the changes discussed near the end of #8.  Let's discuss this more.

was #8 even ready to close?  It seems that many tickets closed around that time were premature, so maybe we should verify all those referenced tickets from #8, #15, #17, #18, #19, #20, #21, #22 (and possibly others) to ensure things are actually implemented as expected.

Confirm the changes you intend to make and get approval again before making any changes.

---

## #38 comment 4629938143 — @AlbinoGeek — 2026-06-05T09:10:20Z
<https://github.com/Rethunk-AI/bakeoff/issues/38#issuecomment-4629938143>

Acknowledged. Schema audit complete against current `schema/schema.sql`. Confirmed gaps below. No changes will be made until you sign off on the proposed patch approach.

---

**`interface_type` table** (per #17 spec) — current schema has only `interface_type_id` + `name`; missing:
- `bandwidth_peak_gb_s`
- `description` (current `name` column covers only the label)
- `interface_family`
- `transfer_rate` → should be `lane_transfer_rate` per your note
- `lane_count`
- Seed data: current 10-row insert is a stub; full PCIe Gen 1–5 variant matrix + SXM2/4/5, NVLink, OCuLink rows not seeded

**`gpu_hardware` table** (per #18 spec) — current schema is a reduced stub; missing vs. spec:
- `pci_device_id`, `pci_sub_device_id`
- `vram_type` (TEXT per #18 spec; you noted it as FK — clarification requested: is `vram_type` meant to be a free-text field or a FK to a separate lookup table?)
- `memory_bus_width_bits`
- `memory_bandwidth_peak_gb_s`
- `clock_memory_mhz`, `clock_graphics_boost_mhz`
- `tdp_watts`
- Column naming drift: `gpu_model` (current) should be `gpu_name`; `vram_mb` → `vram_total_mb`; `tflops_fp16` → `peak_tflops_fp16`; `interface_type_id` → `gpu_native_interface_type_id`
- `tflops_fp32` and `tflops_bf16` in current schema are not in #18 spec — retain or drop?

Note: `pci_vendor_id`, `pci_subsystem_vendor_id` you listed are not in the #18 spec as authored. Please point to the issue/comment where those were agreed, or confirm they should be added now.

**`tflops_source` table** — not found in current schema or in the issues I reviewed. Please reference the issue where this was specified.

**`system_gpu_link` table** (per #20) — missing:
- `slot_native_interface_type_id`
- `actual_interface_type_id`

**`system_hardware` / `system_software` tables** (per #19) — current columns appear minimal. If there are specific fields you expected from #19 that are missing, please enumerate; I don't have visibility into what the end-of-#19 discussion settled on beyond what's in the schema.

**`run_hardware_metrics`** (#8, #21) — acknowledged concern. Will not treat this as fully closed without your review of the end-of-#8 discussion items. Please identify the specific fields or constraints from that thread you consider unimplemented.

---

**Closed tickets (#8, #15, #17, #18, #19, #20, #21, #22):** Will not assume any of these are complete until you confirm. Recommend a verification pass against live schema before declaring any closed.

**Proposed next step:** Once you confirm the gap list above (including the open clarification questions), Bastion will draft a consolidated schema patch DDL and post it for your review before any migration runs. Prefer single PR or per-table issues?

— Bastion // 050907ZJUN26

---

