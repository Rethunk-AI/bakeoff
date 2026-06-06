---
ticket: 38
type: issue-body
author: gissf1
posted: 2026-06-05T08:56:56Z
topic: system_gpu_link
url: https://github.com/Rethunk-AI/bakeoff/issues/38
title: Tables are not defined properly and missing many fields.
---

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
