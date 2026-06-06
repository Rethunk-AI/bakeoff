---
ticket: 8
type: issue-body
author: gissf1
posted: 2026-05-13T20:53:09Z
topic: run_hardware_metrics
url: https://github.com/Rethunk-AI/bakeoff/issues/8
title: Additional Performance Metrics
---

## Problem / Proposal

I'm unsure of the present details tracked, but among other information in these tests, we should also be tracking: prompt and/or test cost for a given model (perhaps it needs multiple planning cycles to come up with a result vs not), real wall clock time, and cpu/gpu cycle counts required (so we can compare relative efficiency metrics across multiple GPU generations and provide performance estimates on new hardware specifications based on past hardware ratings).

In short, not only should we be tracking "how well did this model do at accomplishing the task overall" but also, using our measurement data, we can determine things like:
- "this X hardware was able to do the task faster/slower/cheaper/more efficiently than the other Y hardware" even across generations and product lines.
- "this X model requires more delay time before responding, but comes out with faster and more accurate output, when compared to other Y model"
- "this X model completes tasks faster than the other Y model"
- "this X model uses N% less gpu clock cycles as the other Y model to solve the same problem"
- "Since X hardware was able to do the task in T time, this unreleased hardware Y with N% faster clock speeds on the same architecture should be able to do the task in T2 time"

It likely makes sense to update the bakeoff-results repo to have that data available for comparison as well.

## Touches benchmark invariants?

- [ ] Yes
- [X] No
