---
name: gemm-att-analysis
description: Use ATT (rocprofv3 Advanced Thread Trace) to attribute per-cluster cycles, stalls, barrier waits, and XDL utilization for a HipKittens GEMM kernel. Run this when you want to understand or tune the pipeline schedule (ping-pong stagger, drain placement, MMA throughput floor, load/compute balance) and the kernel is built as a `tk_kernel` pybind11 module.
---

# GEMM ATT pipeline analysis

This skill captures an ATT (SQTT) trace of a HipKittens GEMM kernel and turns it into a per-cluster cycle breakdown so you can read off:
- How much of the MMA cluster sits at the XDL throughput floor (single-occupancy `v_mfma_*` on this shape ⇒ 32 cyc each on gfx950 BF16 32x32x16).
- Where global-load latency is hidden (or not) via the c2 `s_waitcnt(0)` drain.
- Which barriers are absorbing imbalance between the two ping-pong halves.
- Steady-state XDL utilization (the right "MFMA efficiency" number for kernel-quality tuning).

The reference walkthrough for the 32x16 BF16 kernel is `analysis/bf16_gemm/8wave_pingpong_walkthrough.md`.

---

## Prerequisites

- ROCm 7.x with `rocprofv3` and `hipcc` (`/opt/rocm/bin/rocprofv3 --att --help` should list `--att-library-path`).
- HipKittens repo, `source env.src` from repo root.
- A buildable GEMM kernel in `kernels/gemm/<...>/` whose Python module is named `tk_kernel` (per the repo's pybind convention).
- A target AMD GPU (CDNA3/CDNA4). Note: with multi-GPU nodes, the busy device is whichever `cuda:0` maps to — check `amd-smi metric` (look for the one with `GFX_ACTIVITY: 100%` and `SOCKET_POWER` near TDP).

---

## Step 1 — Install the trace decoder (one-time)

ROCm ships only the ATT collector; the **decoder** is a separate AMD package. Without it `rocprofv3 --att` errors with `rocprof-trace-decoder library path not found`.

```bash
# Pick the .deb matching your container OS (here: Ubuntu 24.04, decoder v0.1.6).
# Other releases at https://github.com/ROCm/rocprof-trace-decoder/releases
cd /tmp
curl -sL -o rtd.deb \
  https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-ubuntu-24.04-0.1.6-Linux.deb

# Extract without installing (no system-package state changes, no dpkg deps)
dpkg-deb -x rtd.deb rtd
ls /tmp/rtd/opt/rocm/lib/librocprof-trace-decoder.so   # should exist
```

The path `/tmp/rtd/opt/rocm/lib` is what you pass to `--att-library-path` later.

---

## Step 2 — Build the kernel and write a single-invocation driver

ATT traces are large; **trace exactly one kernel launch**, not the bench loop. The driver next to the kernel's Makefile should just allocate tensors and call `dispatch_micro` once.

Use the template at `.claude/skills/gemm-att-analysis/driver_template.py` — edit `M, N, K, dtype` to match the kernel's hardcoded dims (the 32x16 variant pins 8192³ via `#define`).

Build it normally:

```bash
cd kernels/gemm/<your-kernel-dir>
make clean && make SRC=<your-kernel.cpp>
```

Confirm: the compile output's "Occupancy [waves/SIMD]" line tells you how many waves co-reside per SIMD — needed later for XDL-utilization arithmetic.

---

## Step 3 — Capture ATT

```bash
cd kernels/gemm/<your-kernel-dir>
rm -rf att_out
rocprofv3 --att \
  --att-target-cu 0 \
  --att-library-path /tmp/rtd/opt/rocm/lib \
  --kernel-include-regex "<kernel-symbol-substring>" \
  -d att_out \
  -- python3 <your-driver.py>
```

- `--kernel-include-regex "micro_tk"` for the standard HK GEMM symbol; adjust for your kernel.
- `--att-target-cu 0` samples one CU (~32 waves). For most pipeline questions one CU is enough; if you need cross-CU you'll average across multiple runs.

What you get under `att_out/`:
- `stats_ui_output_agent_*_dispatch_*.csv` — **the per-instruction aggregate, this is the file the analyzer reads.**
- `ui_output_*/*.json` — per-wave timelines (only needed for verification).
- `*_code_object_id_*.out` — disassembly with ATT annotations.

---

## Step 4 — Per-cluster analysis (reusable)

Run the analyzer; it auto-detects the loop body (mode of `Hitcount`) and segments instructions between consecutive `s_barrier`s. Outputs per-cluster totals normalized to **cyc/wave/iter** so they're comparable across runs and shapes.

```bash
python3 .claude/skills/gemm-att-analysis/analyze.py \
  att_out/stats_ui_output_*.csv \
  --waves 32 \
  --mfma-cycles 32 \
  --mfmas-per-cluster 16
```

`--waves` defaults to the inferred (Hitcount / iterations) ratio; override if you used a different `--att-target-cu` size. `--mfma-cycles` is the MFMA shape's documented cycle count (CDNA3/CDNA4 32x32x16 BF16 = 32; see `amd_matrix_instruction_calculator`'s `dict_insts` for other shapes). `--mfmas-per-cluster` is from the source loop (16 for the 32x16 GEMM's one `mma_ABt` per cluster over `DOT_SLICE=32`).

The analyzer prints:

1. **Loop-body autodetection** — hitcount distribution and how many iterations it inferred.
2. **Per-cluster table** — total cyc/wave/iter, breakdown by instruction class (`mfma`, `gload`, `dsread`, `waitcnt`, `other`), and end-of-cluster barrier wait. The cluster with the largest barrier wait is the imbalance signature.
3. **Steady-state XDL utilization** — `(num_mma_clusters × mfmas × cycles) / sum(per-cluster wall-clock)`. For 4-cluster ping-pong with 16 mfmas at 32 cyc each, the numerator is 2048 cyc; gaps below ~95% are usually phase-boundary handoff overhead.
4. **Sanity check** — `sum(CSV.Latency) / sum(wave.duration)` from the JSON outputs (should be ~0.93–1.00; the small gap is unattributed wave-launch/cleanup cycles). This confirms the CSV's `Latency` column is the decoder header's `duration` field accumulated across hits.

---

## Step 5 — Interpreting the output

**The terminology, grounded in `rocprof-trace-decoder/include/trace_decoder_types.h`:**

- `duration` (= rocprofv3 CSV's "Latency") = `stall + issue_time` (gfx9) per execution, accumulated across all hits.
- `stall` = time **before issue begins**, *not* wait-for-result. So a high `stall %` on an mfma instruction means the matrix unit can't take the issue yet — structural, not RAW.

**Reading the tables:**

- **MMA clusters near `mfmas × cycles`** (e.g. 16 × 32 = 512 cyc) ⇒ XDL at single-occupancy throughput floor. Lower means RAW or pipeline overlap; higher means contention or the MFMA shape doesn't actually pipeline (often the case on the 32x32 shapes).
- **A `s_waitcnt(0)` row with high stall** ⇒ global prefetch is latency-bound (not landing within the design's depth). Cross-check: is it < the partner half's MMA cluster duration? If yes, it's hidden by ping-pong; if no, it extends wall-clock.
- **Barrier with high stall** ⇒ that's where one half waits for the other. Pair it with the stagger map (row0 cN ‖ row1 c(N−1)) to identify which side is faster.

**XDL utilization vs whole-kernel FLOPS efficiency:**

The steady-state XDL utilization (this script) is the right number for **scheduling quality** — it ignores kernel launch, prologue, epilogue, and grid tails. The whole-kernel FLOPS-vs-peak number (`achieved_TFLOPS / (peak_flops_per_cycle × clock × num_CUs)`) will be lower because it carries those overheads. Use whichever fits your question.

---

## Step 6 — Verifying you read the field correctly (sanity check, optional)

The analyzer already prints the `Latency / wave.duration` ratio. If you want to do it yourself:

```python
import csv, json, glob
lat = sum(int(r['Latency']) for r in csv.DictReader(open(CSV)))
dur = sum(json.load(open(w))['duration'] for w in glob.glob('att_out/ui_output_*/se*_wv*.json'))
print(lat / dur)  # ~0.93–1.00 if the column mapping is right
```

If the ratio drifts significantly from this band, the decoder's column semantics may have changed in a newer ROCm version — refetch `trace_decoder_types.h` from the matching branch.

---

## Common pitfalls

- **Wrong GPU.** On multi-GPU nodes `cuda:0` may not be `amd-smi -g 0`. Sample `amd-smi metric` across all devices and use whichever shows `GFX_ACTIVITY: 100%` and TDP-level power. The clock for that device is the real operating clock; everything else idles at ~95–160 MHz.
- **Boost clock vs power-capped clock.** `MAX_CLK` (2400 MHz on MI355X) is the spec ceiling; sustained dense GEMM runs significantly lower (often ~1.2–1.4 GHz). Use the measured clock for theoretical-peak comparisons, not the spec max.
- **Bench loop instead of one launch.** Don't run `bench.py` under ATT — you'll capture 500+ kernels and the trace becomes unwieldy. Always one launch.
- **`Hitcount` mismatch.** If the analyzer's autodetected `iterations` doesn't match `num_tiles - 1` from the source, the kernel may have a different loop structure (e.g. unrolled) — segmenting may need manual barrier addresses.
- **Stall % on first MFMA of a cluster is small** (~29% in our reference); that's the cold-start, not a kernel bug. The structural signal is the chained mfmas at ~85–90%.

---

## References

- Decoder header (column definitions): `https://github.com/ROCm/rocprof-trace-decoder/blob/amd-mainline/include/trace_decoder_types.h`
- MFMA cycle counts per shape: `https://github.com/ROCm/amd_matrix_instruction_calculator` (`dict_insts` in `matrix_calculator.py`)
- LLVM AMDGPU schedule (gfx942 / gfx950 `Write*PassMAI` classes): `llvm/lib/Target/AMDGPU/SISchedule.td`
- Reference walkthrough using this skill's output: `analysis/bf16_gemm/8wave_pingpong_walkthrough.md`
