# FP8 4-wave interleave GEMM walkthrough

Notes on `kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu`, the FP8 BF16-accumulating
GEMM kernel that implements the *4-wave interleave* pattern (the other "core
overlap pattern" mentioned in the project README alongside the 8-wave
ping-pong). Read the [BF16 8-wave walkthrough](../bf16_gemm/8wave_pingpong_walkthrough.md)
first — most low-level concepts (MFMA layout, single-occupancy XDL, group-load
semantics, ATT terminology) carry over; this doc covers what's structurally
different.

**Canonical AMD reference for this exact kernel:** the ROCm Blogs post
[Deep Dive Into 4-Wave Interleave FP8 GEMM](https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html)
covers the same pattern from AMD's perspective; recommended reading
alongside this walkthrough.

> Built and measured on MI355X (CDNA4, gfx950) at 8192³, ROCm 7.2. Source
> line numbers and shapes verified from the file; MFMA cycle count from LLVM
> `SISchedule.td`; achieved TFLOPS, clock, and power from running the kernel
> and sampling `amd-smi`.

## Headline

| metric | FP8 4-wave (this kernel) | BF16 16x32 8-wave (for context) |
|---|---|---|
| Achieved TFLOPS @ 8192³ | **2553** (avg) / 2594 (best) | 1217 |
| Achieved GFX clock | **~1.5 GHz** (derived from wave durations) | ~1.48 GHz |
| Socket power under load | ~297 W | ~1390 W (TDP-capped) |
| MFMA shape | `mfma_scale_f32_16x16x128_f8f6f4` | `mfma_f32_16x16x32_bf16` |
| MFMA cycles per issue | **32** (Write8PassMAI, gfx950) | 16 |
| Flops per MFMA instruction | 65,536 (16·16·128·2) | 8,192 |
| Per-SIMD peak flops/cyc | **2048** (2× BF16) | 1024 |
| Per-cycle efficiency vs peak | **~81%** | ~78% |
| Waves per workgroup | 4 (2×2 grid) | 8 (2×4 grid) |
| Occupancy (waves/SIMD) | 1 | 2 |
| VGPRs (compile remark) | 256 | 210 |

**Net story:** the FP8 kernel runs at ~1.5 GHz — essentially the same clock as
BF16 — and achieves ~81% of per-cycle peak — essentially the same scheduling
quality as BF16. The **2× TFLOPS advantage** over BF16 comes entirely from the
**2× per-cycle peak** of the FP8 MFMA (`16x16x128` packs 2× the flops of
`16x16x32` in the same 32 cycles), not from clock or power differences.

**A measurement caveat worth flagging up front** (and the methodology section
below explains in detail): the `amd-smi` `CLK` and `GFX_ACTIVITY` readings
were both *misleading* for this kernel's burst-launch pattern — `CLK` reports
the between-launch boost clock (2.4 GHz) rather than the achieved compute
clock; `GFX_ACTIVITY` underreports the active phases. The achieved clock
quoted above (~1.5 GHz) is derived from per-wave cycle counts in the ATT
trace divided by kernel wall-time, which is independent of `amd-smi` and
gives the right answer.

## Structure: three kernels in one file, size-dispatched

The file defines three `__global__` kernels, dispatched by host code based on
M==N==K:
- **`matmul_device`** (line 86) — default, for 8192³. `BLOCK_SIZE_ROW/COL = 256`.
  Uses the **`do_interleaved_cluster`** helper. **This is the kernel this doc
  walks through.**
- **`matmul_device_1024`** (line 322) — `BLOCK_SIZE_ROW/COL = 64`. Inlined
  simpler `load + mma_ABt` pattern (the `do_interleaved_cluster` calls are
  commented out at lines 432, 442, 454, 460).
- **`matmul_device_2048`** (line 571) — `BLOCK_SIZE_ROW/COL = 128`. Same
  inlined pattern as the 1024 variant.

So the interleave optimization is **size-specific** — it ships only for the
largest case where each cluster has enough compute to make the fine-grained
load/MFMA interleave worth the source complexity.

## A/B/C orientation

- **A**: `fp8e4m3`, `gl<fp8e4m3, 1, 1, M, K>` (line 86) — row-major (M, K).
- **B**: `fp8e4m3`, `gl<fp8e4m3, 1, 1, N, K>` — row-major (N, K), i.e. Bᵀ
  stored row-major. Same as BF16 GEMM.
- **C**: **`bf16`**, `gl<bf16, 1, 1, M, N>` — output is wider than the inputs;
  the fp32 accumulators are downcast to bf16 on store. (Note: `fp32fp32` would
  be the precision-preserving choice; bf16 saves bandwidth.)

## Workgroup → C tile → per-warp

Lines 87–91 (in `matmul_device`):
- `WARPS_COL = 2`, `WARPS_ROW = 2`, `NUM_WARPS = 4` (line 13) ⇒ **2×2 warp grid**.
- `__launch_bounds__(256, 1)` (line 86) ⇒ 256 threads (= 4 × 64), **occupancy 1**.
- `BLOCK_SIZE_ROW = BLOCK_SIZE_COL = 256` ⇒ workgroup computes a **256×256 C tile**
  (same as BF16 ping-pong).
- `BLOCK_K = 128` ⇒ K-step is **2× wider** than BF16's 64.
- Per-warp tile = 256/2/2 × 256/2/2 = **128×128** in BF16 fp32 accumulators.

## Shared and register tiles

- Shared: `st_fp8e4m3<128, 128, st_16x128_s>` (lines 98–99). Each shared tile
  is half the M block × full K-step.
- Register operand: `rt_fp8e4m3<64, 128>` (lines 105–106). Per
  `include/types/register/rt.cuh:141`, default sub-tile shape is `rt_16x128`,
  so the 64×128 tile dices into 4 (M-sub-tiles) × 1 (K-sub-tile) = 4 sub-tiles
  per register tile.
- Accumulator: `rt_fl<64, 64, col_l, rt_16x16_s>` (line 107). Four of these
  per warp in `c[2][2]` (line 112) ⇒ each warp owns 4 × 64×64 = 128×128 ✓.

## MFMA: 16×16×128 in one shot

Per `include/ops/warp/register/tile/mma.cuh:119`, HK's fp8 MFMA path emits
`__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` — a single instruction
that consumes K=128 of fp8 input and produces a 16×16 fp32 output.

**Cycle count: 32 cyc per issue** on gfx950 (with FP8 inputs), confirmed from
two sources:

- AMD's CDNA4 ISA reference ([PDF](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)),
  MFMA instruction table: this shape is listed as "16 or 32" cycles with the
  note that **the FP8 input case takes the larger count** (32 cyc), and
  FP6/FP4 inputs take 16 cyc. Community summary in
  [salykova.github.io/matrix-cores-cdna](https://salykova.github.io/matrix-cores-cdna)
  consolidates the same table.
- LLVM `SISchedule.td` (cross-check):
  ```
  def : InstRW<[WriteMFMAScale_16X16X128_F8F6F4, MIMFMARead],
         (instregex "^V_MFMA(_SCALE)?_.32_16X16X128_F8F6F4")>;
  def WriteMFMAScale_16X16X128_F8F6F4 : SchedWriteVariant<[
      SchedVar<PredIsF8_MFMA_SCALE, [Write8PassMAI]>,    // FP8: 8 passes × 4 cyc = 32
      SchedVar<NoSchedPred, [Write4PassMAI]>]>;          // FP6/FP4: 4 × 4 = 16
  ```
  matches the ISA table's precision-dependent behavior.

(Useful corollary: if a future HK kernel uses FP4 / FP6 with this same
`16x16x128` shape, it would get 16 cyc per issue → 4096 flops/cyc/SIMD peak,
2× the FP8 case. A reason to want narrower-precision variants if accumulator
precision allows.)

Flops per instruction (small but important):
- BF16 32x32x16: 32·32·16·2 = 32,768 flops in 32 cyc ⇒ 1024 flops/cyc/SIMD.
- BF16 16x16x32: 16·16·32·2 =  8,192 flops in 16 cyc ⇒  512 flops/cyc/SIMD.
- FP8  16x16x128: 16·16·128·2 = **65,536 flops in 32 cyc ⇒ 2048 flops/cyc/SIMD.**

So at fixed 32 cyc, FP8 packs 2× the flops of BF16 32x32x16 — same instruction
duration, half the M·N output area, but **8× the K**, so 2× the total work.
That's the matrix unit's FP8 advantage: double the per-cycle peak.

`mma_ABt` nest for this kernel: D::height = 64/16 = 4, D::width = 64/16 = 4,
A::width = 128/128 = 1 ⇒ **4×4×1 = 16 MFMAs per `do_interleaved_cluster` call**
(K is covered in a single MFMA, no K-chain).

## The 4-wave interleave pattern (`do_interleaved_cluster`, lines 17–83)

This is the headline structural difference from BF16 8-wave ping-pong.

In BF16 ping-pong, 8 warps **spatially split** into two halves; one half does
MMA while the other does LD, alternating phases via a `warp_row`-driven
barrier stagger. The matrix unit is fed by alternating halves.

In FP8 4-wave interleave, all 4 warps run **identical code in lockstep** (no
row stagger). The overlap of compute and memory happens **temporally inside
each wave's instruction stream**: alternate MFMA instructions with load
instructions so the wave keeps both the matrix unit and the memory pipe busy
simultaneously.

The pattern inside `do_interleaved_cluster` (16 MFMAs + many loads, all
inside one wave):
```
sched_barrier(0)
mma_ABt_one  (#1 of 16)
sched_barrier(0)
precompute addresses                    ← scalar (SGPR-bound) address math
sched_barrier(0)
mma_ABt_one  (#2)
sched_barrier(0)
prefill swizzled offsets
load_one<0>(global → LDS)               ← buffer_load to LDS
load_one<0,0,0>(LDS → register)         ← ds_read
sched_barrier(0)
mma_ABt_one  (#3)
sched_barrier(0)
load_one<0,0,1>
sched_barrier(0)
mma_ABt_one  (#4)
sched_barrier(0)
load_one<1> load_one<1,0,0>
sched_barrier(0)
mma_ABt_one  (#5)  mma_ABt_one  (#6)
sched_barrier(0)
... continues through #16
```

The `sched_barrier(0)` calls everywhere are *compile-time* fences (cost zero
runtime cycles) preventing the compiler from reordering across the carefully
placed boundaries. There is **no `s_barrier` inside `do_interleaved_cluster`**
— the synchronization is at the *instruction-issue* level within one wave,
not cross-wave.

**Why this works for FP8 specifically:**

1. The FP8 MFMA covers K=128 in one shot, so there are *fewer* MFMAs per
   K-step (64 per warp per K-step here) vs BF16. The compute density per
   instruction is high.
2. **Register pressure forces low occupancy.** The compile remark reports
   **256 VGPRs/wave, occupancy 1** for `matmul_device`. 8 waves would need
   8×256 = 2048 VGPRs per SIMD, more than CDNA4's VGPR file can give while
   keeping spill-free. With 4 waves at occupancy 1, each wave has plenty of
   registers but there's only one wave per SIMD — no co-resident wave to feed
   the XDL during a load phase. So the wave **has to** interleave its own
   loads with its own compute to keep the XDL fed.
3. The hardware has separate functional units (XDL for MFMA, memory pipe for
   buffer_load/ds_read). An MFMA issues to the XDL and runs there for 32 cyc;
   during those cycles the wave can issue load instructions to the memory
   pipe without contention. The fine-grained interleave exploits this.

## Loop body (lines 187–215): one K-step per source iteration

```
for k in 0 … k_iters-3:
    s_waitcnt vmcnt(16) + s_barrier        ← cross-wave sync, prefetch-backlog limit
    do_interleaved_cluster → c[0][0]
    s_waitcnt lgkmcnt(0) (no barrier)
    do_interleaved_cluster → c[0][1]
    s_waitcnt vmcnt(16) + s_barrier
    do_interleaved_cluster → c[1][0]
    do_interleaved_cluster → c[1][1]
```

So 4 `do_interleaved_cluster` calls per K-step, one per `c[i][j]` accumulator.
Each cluster contains 16 MFMAs ⇒ **64 MFMAs per warp per K-step**. Same MFMA
count as BF16 16x32 — but each MFMA does **8× the flops** of BF16's 16x16x32,
so the per-K-step compute is dramatically higher.

`k_iters = K / BLOCK_K = 8192/128 = 64`. Loop runs `k_iters - 2 = 62`
iterations, with two peeled epilogue K-steps (lines 217–270 and 272–311) for
buffer-rotation handling.

## Prologue and epilogue waitcnts

Prologue (lines 150–184): preloads both buffers (8 G::loads), then a ladder of
**`vmcnt(28)` → `vmcnt(24)`** to retire enough buffer_loads for the first
register loads. The large `N` values (vs BF16's `vmcnt(4)`/`vmcnt(6)`) reflect
the larger number of buffer_loads per `G::load` on the wider FP8 shared tiles.

Two-stage epilogue (lines 217–270 and 272–311): peels the last 2 K-iters with
progressively tighter `vmcnt`s (16 → 8 → 4 → 2 → 0), draining the prefetch
backlog as no more prefetches are issued. Same "tighten as you drain" pattern
as BF16 16x32's epilogue.

## What's *not* in this kernel (vs BF16 8-wave ping-pong)

1. **No `warp_row`-driven barrier stagger.** All 4 waves run identical code.
2. **No `chiplet_transform_chunked`.** Uses the inline `(wgid%8)*(NW/8) + wgid/8`
   swizzle (line 123), same as 32x16's; only WGM=4 (line 125) vs the 8-wave's
   WGM=8.
3. **No readfirstlane SGPR-hoist of LDS bases** — addresses are hoisted by
   `precompute_addresses` (line 23, defined in `utils.cpp`) at a different
   layer.
4. **No `__launch_bounds__(NUM_THREADS, 2)`** — occupancy is 1, not 2.
5. **Standalone executable, not pybind11.** The file has its own `main()`
   (line 994) with inline bench + correctness harness using rotating input
   buffers. Build with the included `Makefile` (`make` ⇒ `./tk_kernel`).

## Measured numbers (8192³, MI355X, warm GPU)

Build (HK Makefile, COMP_LEVEL=profile):
- VGPRs: **256** (max per wave)
- Occupancy: **1 wave/SIMD**
- No spills

Bench (best of 100 timing iters after 500 warmup, with rotating buffers):
- **Best: 0.424 ms / 2594.63 TFLOPS**
- **Avg: 0.431 ms / 2553.34 TFLOPS**
- Correctness: PASSED (CPU reference comparison, rtol implicit ~0.01)

Achieved clock measurement — the right way (cycle counts) and the wrong way
(amd-smi clock readings):

**The wrong way (amd-smi CLK, what I cited first):** `amd-smi metric -g 3`
under load reports `CLK: 2398–2407 MHz` across all XCDs, `SOCKET_POWER: 297 W`,
and `rocm-smi -d 3 --showclocks` cross-checks to `sclk clock level: 1: (2400Mhz)`.
This *looks* like a sustained 2.4 GHz reading. **But that's the SCLK setpoint
DPM transitions to between launches** — not the achieved clock during the
0.43 ms of active MFMA work in each launch. With 0.43 ms compute / ~1 ms
host overhead per launch, amd-smi mostly samples the inter-launch boost
state, and the SCLK setpoint stays elevated through these gaps so even a
"during-launch" sample reports 2.4 GHz.

(Sidebar: `GFX_ACTIVITY` simultaneously read `0%` during these 2.4 GHz reads —
a second misleading metric, for the same burst-launch-pattern reason. Power
is the only `amd-smi` field that reliably distinguished active from idle
here.)

**The right way (per-wave cycles ÷ per-workgroup wall time):**
- ATT records each wave's `duration` in sclk cycles (per `rocprof-trace-decoder`).
- With occupancy 1, 1024 workgroups, 256 CUs ⇒ each CU runs 4 workgroups
  sequentially, so per-workgroup time = kernel_time / 4 = 0.424/4 ≈ 106 µs.
- Median per-wave duration in the trace: **156,516 cyc**.
- Derived clock: 156,516 cyc / 106 µs = **1.477 GHz**.
- Max-wave variant (the longest-running wave on the CU, which best
  approximates the workgroup time): 163,100 cyc / 106 µs = 1.539 GHz.

So **achieved clock ≈ 1.45–1.54 GHz** — *not* 2.4 GHz. The per-cycle
efficiency math at this clock is:
- Achieved 2553 TFLOPS / (1.5 GHz × 256 CUs × 4 SIMDs/CU × 2048 flops/cyc/SIMD)
  ≈ **81%** of peak.

That's the sane number — almost identical to BF16 16x32's 78%. The 51%
efficiency I'd computed against the (wrong) 2.4 GHz clock was the signal
that the clock measurement itself was off, but I'd missed it.

Per-cycle efficiency (using the cycle-derived ~1.5 GHz clock):
- Achieved: 2553 TFLOPS / (1.5 GHz × 256 CUs × 4 SIMDs/CU) ≈ **1662 flops/cyc/SIMD**
- Theoretical peak for `mfma_scale_f32_16x16x128_f8f6f4`: **2048 flops/cyc/SIMD**
- **⇒ ~81% of per-cycle peak.**

That matches BF16 16x32's ~78% closely. The two kernels have very similar
per-cycle scheduling efficiency; the 2× TFLOPS advantage of FP8 comes from
the 2× per-cycle peak of the FP8 MFMA, not from a better schedule.

## ATT (one CU, fp8 main loop)

Captured with the standalone executable + `rocprofv3 --att`, analyzed with
`.claude/skills/gemm-att-analysis/analyze.py` (which I updated this session to
auto-detect the loop body by `hitcount × num_instructions` instead of plain
max hitcount — the old heuristic mispicked the trace's outlier hitcount=2048
which had only 23 instructions; the real loop body is hitcount=148 with 4761
instructions).

The trace finds **58 segments per loop iter** (delimited by `s_barrier`s),
each containing ~16 MFMAs averaged across waves. Sample mfma per-execution
duration in the loop body: ~24 cyc/mfma (vs the 32 cyc spec). That's lower
than I'd expect — either:
- The XDL has some pipelining for this shape (multiple MFMAs in flight,
  averaging the wave's stay-at-PC down),
- Or the trace's per-wave attribution isn't capturing the full chain stall.

I haven't fully reconciled this; it's a real measurement that needs careful
interpretation. **The BF16 16x32 trace showed clean 32 cyc/chained-mfma**
(single-occupancy XDL); the FP8 trace doesn't show the same pattern.
**Worth digging into if you want to pin "is the FP8 XDL pipelined?"**

Sanity ratio (sum CSV.Latency / sum wave.duration): **0.977** — column
mapping is clean.

## Open questions

1. **Why does the GPU not clock higher given the headroom?** The kernel only
   draws 297 W (well below the 1390 W TDP) and runs at ~1.5 GHz, with the spec
   max being 2.4 GHz. Some throttle is keeping it below peak that isn't power.
   Possibilities: voltage/frequency curve limits at this DPM state, thermal
   throttling, or DPM doesn't transition fully during burst workloads.
   Empirical test: a custom driver that keeps the GPU 100% busy continuously
   for several seconds and see if amd-smi reports a *different* (higher or
   lower) clock + an accordingly different TFLOPS. If both move together,
   it's DPM dynamics; if not, it's a hard limit.
2. **Is the XDL pipelined for `mfma_scale_f32_16x16x128_f8f6f4`?** The ATT
   per-mfma cycles (~24) are lower than the 32-cyc spec, hinting at
   pipelining or measurement subtlety. Compare against running a tight
   chained-mfma microbenchmark.
3. **Could occupancy 2 work with reduced register pressure?** 256 VGPRs is
   right at the limit. If the register tile sizes were halved (4 × 32×32 fp32
   accumulators instead of 4 × 64×64), would 2 waves/SIMD fit and would the
   ping-pong story then beat the interleave?

## What I'd verify next

- **Multi-CU ATT** to check per-CU generalization (we did this for BF16; same
  should hold here).
- **`ncu`-equivalent** PMC counters for `SQ_INSTS_VALU_MFMA_MOPS_F8` and
  related — would tell us if the XDL is truly receiving back-to-back
  instructions.
- **Microbenchmark of chained `mfma_scale_f32_16x16x128_f8f6f4`** in isolation,
  to settle the "is the FP8 XDL pipelined?" question without the rest of the
  kernel's noise.
- **Sweep MFMA shapes that map to the same instruction class** to see if
  there's a more efficient FP8 variant the kernel could use.
