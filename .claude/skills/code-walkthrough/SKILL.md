---
name: code-walkthrough
description: Use this skill when walking a user (or yourself in a future session) through complex performance-tuned code — kernels, hand-scheduled assembly, schedulers, lock-free data structures — where the design is subtle and claims need empirical grounding. Goal is to build a mental model the user can verify and extend, not deliver a lecture. Failure mode to avoid: confidently asserting something wrong because you didn't check, or rationalizing an empirical surprise instead of investigating it.
---

# Code walkthroughs that hold up

The reference example for this skill is `analysis/bf16_gemm/8wave_pingpong_walkthrough.md` (HipKittens 8-wave ping-pong BF16 GEMM). The patterns and failure modes below are drawn from that session's actual errors and corrections — concrete enough to recognize again.

## Core loop

Each turn of a walkthrough follows the same shape:

1. **Read the code first.** Open the file. Get line numbers. Quote what you cite. Do not paraphrase from training data — it ages and lies. When a question comes in, the first reach is the file, not memory.
2. **State the claim with grounding.** Every quantitative claim has a verifiable source ("lines 434–449 are 16 `v_mfma` to 8 different accumulators × 2 K-slices, see ISA dump"). If you can't cite a source, mark it as a hypothesis.
3. **Verify it two ways when you can.** Source-code arithmetic AND empirical measurement, or two independent specs. Note when they agree (sanity check); investigate when they disagree (more interesting).
4. **Mark uncertainty explicitly.** "I think X — let me check" beats "It's X" when you haven't. The cost of hedging is small; the cost of confident wrong answers compounds.
5. **Let pushback drive correction, not defense.** When the user questions a number, treat it as a probe of your model — recompute from scratch. Defending is how rationalizations grow.
6. **Maintain a living artifact.** A walkthrough doc that evolves with corrections is more useful than a perfect-on-the-first-try monologue. Each correction makes the next session sharper.

## Failure modes — concrete examples to recognize

### Confident arithmetic without recomputing
- Said "8 warps × 64 threads = 256 threads." It's 512. Two unrelated 256s (the workgroup tile and a would-be thread count) collided in my head, and I didn't expand the formula.
- Said "each accumulator gets 2 MFMAs per cluster." It's 4: K_STEP=64, MFMA-K=16 → 4.
- **Fix:** when stating any numeric relationship, write the formula explicitly, then evaluate. Don't pattern-match. If you've quoted a number twice in the same conversation, that's a smell — recompute once.

### Asserting field semantics without checking
- Asserted `rocprofv3` CSV's "Latency" column = `duration` from the decoder header, based on the magnitudes making sense. They did (verified later, ratio 0.94). But I should have checked first, not after the user asked "did you check?"
- **Fix:** when you reference a tool's output, find the source/header for the field's actual definition before relying on it. If two estimates of the same quantity exist, compute their ratio as a sanity check.

### Hand-waving an empirical surprise instead of designing a control
- The c2→c3 `s_waitcnt(0)` move broke correctness. I rationalized with a "WAR hazard" story. The user pointed out `s_waitcnt` is per-wave and the WAR story didn't hold up. Only after running two more variants (V2 = drain in c3 before MMA, V3 = drain in both) did the real mechanism emerge (group-load + stagger + per-wave waitcnt).
- **Fix:** when a result contradicts your model, your first response should be to design a control experiment, not refine your story. Two more variants is almost always cheaper than another paragraph of speculation. Speculation that survives one piece of evidence often fails at the second.

### Misreading "stall %" before checking what stall means
- Initially attributed chained-MFMA stalls to RAW. The data showed mfmas #2–8 (targeting *different* accumulators from #1, no RAW with it) all at the same ~86% stall — and the decoder header says `stall` = wait-*before*-issue, ruling out RAW (which is wait-for-result). The real bottleneck was structural XDL single-occupancy.
- **Fix:** before naming a mechanism, check that the data actually distinguishes it from alternatives. If two mechanisms predict the same observation, the observation isn't probative — design a measurement that separates them.

### Picking the wrong baseline for an "efficiency" number
- Compared `489 cyc (wave-time parked at MMA PCs)` to `512 cyc (XDL-busy time)` and called the ratio "95% of floor." Those measure overlapping but different things; the ratio wasn't meaningful.
- **Fix:** before computing a ratio, state what each side measures in physical units ("cycles of what, where"). If they're not the same kind of cycles, the ratio is dimensionless garbage.

### Assuming hardware constants without measuring
- Quoted `~2.4 GHz` for the clock. The card sustains `~1.3 GHz` under MFMA load (power-capped at TDP). 2.4 GHz is the spec max, which dense GEMM never reaches.
- **Fix:** if your conclusion depends on a hardware constant (clock, peak FLOPS, cache size), measure it before quoting. On AMD: `amd-smi metric` across all GPUs, look for the one with high `GFX_ACTIVITY` and TDP-level power — that's where the work is happening, and that's the clock you should use.

### Picking the wrong device/context entirely
- Sampled GPU 0's clock for several rounds and got idle-clock numbers. The actual work was on GPU 3 (`cuda:0` mapped to physical 3 on that node).
- **Fix:** never assume the device you're sampling matches the device doing work. Confirm via activity/power before reporting a number from any device-specific tool.

### Cold-state benchmark outliers in cross-kernel comparisons
- Reported "16x32 is 8.6% faster than 32x16" based on numbers measured at different times in the session. Back-to-back warm-GPU re-bench: same kernel jumped from 1192.6 to 1217.3 TFLOPS — the original 1192.6 was a cold-state outlier. With both kernels measured immediately back-to-back from the same warm state, the real gap is ~10.9%, which makes the clock-ratio arithmetic (13.8% clock ⇒ ~11% TFLOPS) cleaner than I'd hand-waved.
- **Fix:** for any cross-kernel TFLOPS comparison, measure back-to-back in the same script invocation with the GPU already warm — same 50-iter warmup + 200-iter timing for both, kernels alternated without GPU idle in between. Treat any number measured against a different thermal state as an *upper bound on uncertainty*, not a real comparison. The fingerprint of a cold-state outlier: the same kernel re-measured immediately gives a noticeably different number.
- This also generalizes: any **stateful resource** (caches, branch predictors, power state, frequency, link state) can leak across runs and dirty cross-condition comparisons. Same-state back-to-back is the universal fix.

### Incomplete diagrams that hide the actual mechanism
- Drew a "ping-pong overlap" picture showing only row0's clusters. The user pointed out row1's c1 MMA happens during row0's c2 loads on the *same* XDL. The "loads hidden under compute" framing collapsed into a sharper "XDL is a shared resource; both halves alternate feeding it at ~100% duty."
- **Fix:** when a diagram has multiple actors, draw all of them. If you find yourself describing "what one wave does," ask whether the model needs the others to be complete.

### Explaining a tensor-core / MFMA op without showing the register layout
- Talked about "row_l / col_l" for register tiles abstractly, then later had to scramble to ground claims about register pressure and which layout the next op consumes. Without a layout diagram, the reader (and you) can't see why a particular `mma_ABt` chain works, why one variant uses more VGPRs than another, or why `store` un-transposes back to `gl`.
- **Fix:** when an MFMA / tensor-core / matrix-unit shape appears in a walkthrough, include a layout sketch covering:
  - The per-lane element count for each operand and the accumulator (compute `shape_rows × shape_cols / wave_size` so the math is visible).
  - Which axis is "fast" inside one lane (`row_l` = the reduction axis K; `col_l` = the output M axis on AMD CDNA).
  - A small 2-D box drawing showing lanes tiling the sub-tile shape. Mark it as **conceptual** — the exact lane-to-element bit positions are MFMA-ISA-specific (often interleaved row-blocks); the diagram is for register-pressure / dataflow reasoning, not bit-exact reverse engineering.
  - Reference numbers worth deriving alongside: number of accumulator sub-tiles per warp (`reg_tile_rows / shape_rows × reg_tile_cols / shape_cols`), cycles-per-MFMA from the matrix-instruction calculator or LLVM `SISchedule.td`, and per-K-step total flops (`mfmas/cluster × cycles × utilization × clock`).
- Example diagrams in the HK BF16 GEMM walkthroughs (`analysis/bf16_gemm/8wave_pingpong_walkthrough.md` for 32x16 shapes; `16x32_walkthrough.md` for the smaller-shape sibling). The two side-by-side show what "different MFMA shape" actually means at the lane level — useful pedagogy for explaining shape tradeoffs.

## Artifacts a walkthrough should leave behind

- **A reference doc** in the repo's analysis/ tree, version-controlled, structured as: data layout → mechanism → empirical numbers → explicit caveats for things not fully explained. Update it as understanding evolves — show the corrections.
- **Project memory** for findings that future sessions shouldn't relitigate (e.g. "tested c2→c3 waitcnt move on YYYY-MM-DD; broke correctness; don't re-propose"). Put it where future-you will see it before re-deriving.
- **Reusable scripts** if the analysis generalizes (see `.claude/skills/gemm-att-analysis/`) — turn a one-off investigation into a tool the next walkthrough can lean on.
- **Honest caveats in the artifacts themselves.** The reference walkthrough records "c3-end barrier asymmetry: predicted ~185, measured 60, not fully explained" instead of wrapping it in a plausible-but-wrong mechanism. That's healthier than overclaiming, and the next reader gets the open question instead of a fake answer.

## When you're stuck

If the user asks something you genuinely can't answer from the available evidence:
- Say so plainly, and propose the experiment that would answer it ("would need per-wave ATT traces, not the aggregate CSV").
- Don't invent a plausible story to fill the gap. A clean "I don't know, here's what we'd need to find out" is far more useful than a confident wrong answer that survives until it breaks something downstream.

## Tooling shopping list

The tools that actually moved the analysis in this session — not a generic list, just what worked:

- **`hipcc --save-temps=obj`** — produces the gfx950 `.s` for ISA-level checks (instruction ordering, register usage, schedule). The 1206-line .s for this kernel was small enough to grep and diff. Use when a small source change has a behavioral effect that defies your model.
- **`rocprofv3 --att`** for per-instruction stall/duration on a target CU. Needs the separately-installed `rocprof-trace-decoder` — see `gemm-att-analysis` skill for setup. Use to attribute cycles within a kernel.
- **`amd-smi metric`** for real clocks, power, and activity across all devices. Use before quoting any hardware constant.
- **`amd_matrix_instruction_calculator` (`dict_insts` in `matrix_calculator.py`)** for MFMA cycle counts per shape. Use to ground "how long does this instruction take."
- **LLVM `llvm/lib/Target/AMDGPU/SISchedule.td`** for arch-specific scheduling classes. Use as a secondary citation when the matrix calculator hasn't been updated for a new arch — watch for unit conventions (passes vs cycles).
- **Tiny Python harnesses** to verify index math by enumeration. Cheaper and more reliable than re-deriving by hand when the math is fiddly (swizzle, partitioning).
- **Existing project docs (CLAUDE.md, walkthrough docs, project memories)** before you start. Often the question has been asked before.

## Anti-patterns

- Generating "explanation" without reading the file you're explaining.
- Polishing the prose of a wrong claim instead of recomputing.
- Calling a ratio "efficiency" without naming what's in the numerator and denominator.
- Citing a spec max as if it were the operating value.
- Drawing a diagram with one of N actors visible and not noticing.
- Writing a long mechanism paragraph in a doc before any measurement confirms it. The doc's role is to *record* what's been verified, not to *propose* what might be true.
