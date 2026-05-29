# 8-wave ping-pong BF16 GEMM (16x32 variant) walkthrough

Notes on `kernels/gemm/bf16fp32/256_256_64_32_with16x32.cpp` — the 16x16x32 MFMA
sibling of `256_256_64_32_with32x16.cpp`. Same 256×256 C tile, same 8-warp 2×4
ping-pong, same group-load + per-wave-waitcnt safety properties, but a different
MFMA shape (16x16x32 BF16 ⇒ 16 cyc/issue vs 32 cyc), 4× the accumulator count,
shared tiles split into M-halves, and harder hand-scheduling (SGPR-hoisted base
addresses, `lgkmcnt(8)` partial drains, 8 MMA blocks per source iteration).

Read the [32x16 walkthrough](8wave_pingpong_walkthrough.md) first — most of the
shared mechanism (workgroup swizzle, ping-pong stagger, group-load semantics,
ATT/decoder terminology) lives there. This doc covers what's *different* and
the surprising performance finding.

> Built and measured on MI355X (CDNA4, gfx950) at 8192³, ROCm 7.2. Shape/index
> arithmetic verified from code; cycle counts from ATT (rocprofv3, one CU);
> clocks from `amd-smi metric` under sustained load. Caveats inline.

## Headline

**16x32 runs 1217.3 TFLOPS vs 32x16's 1098.2 — about 10.9% faster — but
the per-K-step compute cycles, per-K-step wall-clock cycles, and steady-state
XDL utilization are essentially identical between the two kernels.** The
speedup is almost entirely **clock-driven**: 16x32 sustains ~1.48 GHz under the
TDP cap vs 32x16's ~1.30 GHz (a ~13.8% clock ratio, vs 10.9% TFLOPS — the ~3%
discount is consistent with memory latency that doesn't scale with clock,
e.g. the 196-cyc c2 `s_waitcnt(0)` drain becoming a larger fraction of K-step
wall-clock at the higher clock). The hypothesis (consistent with the data, not
independently confirmed): 16x16x32 mfmas are lower-power per cycle than
32x32x16, so the power-capped GPU can clock higher running the smaller shape.

> Benchmark methodology note: these are warm-GPU back-to-back measurements
> (32x16 immediately before 16x32, both at 200 iters with 50-iter warmup).
> An earlier draft of this doc reported the 16x32 number as 1192.6 TFLOPS
> — that was a cold-state outlier (first run after the GPU had been idle).
> Same kernel, warm: 1217.3. Cross-kernel comparisons need same thermal
> state, or the difference you measure is partly the warm-up curve, not the
> kernels.

## A/B/C orientation
Identical to 32x16: C = A·Bᵀ, A=(M,K), B=(N,K) row-major bf16 in HBM, C=(M,N),
fp32 accumulate. `transpose_B=True`. Uses `mma_ABt`.

## Workgroup → C tile
Identical: 256×256 C tile per workgroup, 8 warps in a 2×4 grid (`WARPS_M=2`,
`WARPS_N=4`, `NUM_WARPS=8`, `NUM_THREADS=512`). Each warp owns 128 rows × 64
cols.

## Per-warp tile and MFMA — key structural difference
- **`A_tile`** (line 47): `rt_bf<64, 64, rt_16x32>` — 64×64, half of the warp's
  M-rows. Per warp_row.
- **`B_tile_0/1`** (lines 48–49): `rt_bf<32, 64, rt_16x32>` — 32×64 each, two
  N-stripes. Per warp_col.
- **`C_accum[2][2]`** (line 50): four `rt_fl<64, 32, rt_16x16>` accumulators
  tiling the warp's 128×64 — outer indices are (M-half, N-half).
- **MFMA shape**: `v_mfma_f32_16x16x32_bf16` — **16 cyc** on gfx950
  (LLVM `Write4PassMAI` = 4 passes × 4 cyc/pass; ATT confirms ~16 cyc per
  chained mfma, ~75% stall ⇒ single-occupancy XDL behavior just like 32x16).

`mma_ABt` nest with these shapes: D::height = 64/16 = 4, D::width = 32/16 = 2,
A::width = 64/32 = 2 ⇒ **16 mfmas per `mma_ABt`** — same count as the 32x16
variant. But each mfma is shorter and produces less work:
- 32x32x16 mfma: 32·32·16·2 = 32768 flops over 32 cyc → 1024 flops/cyc/SIMD.
- 16x16x32 mfma: 16·16·32·2 = 8192 flops over 16 cyc → 512 flops/cyc/SIMD.

So the **same `mma_ABt` does ½ the flops** in the 16x32 variant. To match work,
the kernel issues **2× more `mma_ABt` calls per K-step** (4 vs 2). Compute time
per K-step ends up identical: 32x16 = 32 mfmas × 32 = 1024 cyc, 16x32 =
64 mfmas × 16 = 1024 cyc.

### row_l / col_l register layout (16x32 shapes)

(Background and caveats: see the [32x16 walkthrough's row_l/col_l section](8wave_pingpong_walkthrough.md#row_l--col_l--mfma-register-layout)
— same meaning, just different per-lane counts because the MFMA shapes are smaller.)

**Counts:**
- A/B input `rt_16x32_s`: 16 × 32 = 512 bf16 / 64 lanes = **8 bf16/lane**, row_l.
  (Same count as 32x16's `rt_32x16`, just shaped differently: K is wider per lane, M is narrower.)
- C accum `rt_16x16_s`: 16 × 16 = 256 fp32 / 64 lanes = **4 fp32/lane**, col_l.
  (Quarter of 32x16's 16 fp32/lane — same fraction smaller as the accumulator dims, 16² vs 32².)

**row_l layout (operands) — K contiguous in a lane:**

```
A_tile / B_tile sub-tile rt_16x32 (16 M-rows × 32 K-cols, 8 bf16/lane)

         K cols (0..31) →
       ┌── 8 ──┬── 8 ──┬── 8 ──┬── 8 ──┐
M=0    │ lane0 │ lane1 │ lane2 │ lane3 │   4 lanes per M row;
M=1    │ lane4 │ lane5 │ lane6 │ lane7 │   16 rows × 4 = 64 lanes ✓
 ...   │  ...  │  ...  │  ...  │  ...  │   each lane holds 8 bf16 along K.
M=15   │ lane60│ lane61│ lane62│ lane63│
       └───────┴───────┴───────┴───────┘
           ↑
      Wider K (32 vs 32x16's 16) ⇒ 4 lanes per row instead of 2.
      The reduction axis is still "fast within a lane".
```

**col_l layout (accumulator) — M and N share the lane's elements:**

```
C_accum sub-tile rt_16x16 (16 M-rows × 16 N-cols, 4 fp32/lane)

         N cols (0..15) →
       ┌─ 4 ─┬─ 4 ─┬─ 4 ─┬─ 4 ─┐
M=0..3 │ l0  │ l1  │ l2  │ l3  │   each lane: 4 fp32 elements
M=4..7 │ l4  │ l5  │ l6  │ l7  │   in a 4-row × 1-col slice
 ...   │     │     │     │     │   (conceptual; exact lane→block
M=12..15│l60 │ l61 │ l62 │ l63 │    mapping per the MFMA spec).
       └─────┴─────┴─────┴─────┘
   16 M-blocks × 4 N-blocks × 1 lane/block = 64 lanes ✓
   4 fp32/lane × 64 = 256 ✓
```

> Same caveat as the 32x16 doc: the diagrams show the per-lane count and "fast axis," not the exact
> MFMA-ISA bit positions. CDNA MFMA outputs are interleaved across multiple row-blocks; the precise
> formula is in the AMD MFMA spec. The point of the diagram is that you can read off register
> pressure, the count of independent accumulators per warp, and which axis a later op contracts
> along — the bit-exact lane mapping isn't needed for those reasoning steps.

**Compared to 32x16:** A/B input layout has the same 8 bf16/lane but covers a wider K per lane
(32 vs 16), so a single sub-tile contracts more K per MFMA — paid for by the smaller M (16 vs 32).
The accumulator is one-quarter the size (256 vs 1024 fp32), so each lane holds only 4 fp32 instead
of 16, but there are **32 such accumulator sub-tiles per warp** (4 `C_accum[i][j]` × (4 × 2)
sub-tiles each) vs **8** in the 32x16 variant. More, smaller accumulators ⇒ finer-grained
dependency graph but more VGPRs to keep them all live (the build report does show 210 VGPRs for
16x32 vs 200 for 32x16).

## Shared memory: split into M-halves
The 256-row M block is held in LDS as **two 128-row halves**:
- `As[2][2]` and `Bs[2][2]` (lines 44–45). Outer `[2]` = tic/toc; inner `[2]` =
  the M-halves (As[t][0] = rows 0–127, As[t][1] = rows 128–255). Each tile is
  `st_bf<128, 64, st_16x32>`.

(32x16 used `As[2]` — one unsplit 256×64 tile per buffer.)

Why split: it lets the kernel load and consume the two M-halves
**independently** so the `mma_ABt` blocks for `C_accum[0][*]` and
`C_accum[1][*]` interleave with their own LDS reads and global prefetches
rather than waiting on a single big load.

## readfirstlane SGPR hoisting (unique to 16x32)
Lines 83–101. The kernel computes the per-warp LDS base addresses for each of
the four `(As[t][h])` and four `(Bs[t][h])` slots, runs each through
`__builtin_amdgcn_readfirstlane`, and stashes them in SGPRs:

```
uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(
    (uint32_t)((uintptr_t)&As[0][0].data[0] + wid * elem_per_warp * sizeof(bf16)));
// ... a_lds_01, a_lds_10, a_lds_11, b_lds_00 ... b_lds_11
```

These get passed as the final argument to the overloaded `G::load(..., srd,
base_ptr, lds_base)`. Two effects:
1. **Scalar instead of vector address**: an address identical across all 64
   lanes lives in SGPR (1 register) instead of VGPR (64 lanes × 1 register
   each). Saves VGPR pressure and address-generation throughput.
2. **Hoisted out of the loop**: computed once, reused for all K-steps. The
   loop body has no per-iteration address math for these.

(32x16 has no equivalent — its `G::load` uses the simpler signature.)

## Loop body — 8 MMA blocks per source iteration

`for (int tile = 0; tile < num_tiles - 2; tile += 2)` (line 139) — the loop is
**unrolled by 2** (covers 2 K-steps per source iteration), so:

- 8 `mma_ABt` calls per source iteration (lines 153, 167, 180, 194, 209, 223,
  236, 248) ⇒ **4 mma_ABts per K-step**, one for each `C_accum[i][j]`.
- 16 `s_barrier`s per source iteration ⇒ **8 LD/MMA phase boundaries per K-step**.
  Each LD phase loads one A or B half-tile + issues one global prefetch; each
  MMA phase is one `mma_ABt` call.

The schedule rotates through the 4 LDS half-tile buffers and the 2×2
accumulator grid, with prefetches into `tic`-toc-swapped buffers interleaved.

### Pipeline diagram (per K-step)

**Counting units to avoid confusion:** the source loop is unrolled by 2 (`tile += 2`), so each
source iteration covers **2 K-steps** and contains **16 `s_barrier`s and 8 `mma_ABt`s**. The diagram
below shows **one K-step** (half a source iteration) — so 8 phases here, 16 per source iter, 4
mma_ABts per K-step, 8 per source iter. ATT reports per-source-iter numbers; this doc uses
per-K-step throughout for parity with the 32x16 walkthrough.

8 alternating LD/MMA phases per K-step (vs 32x16's 4), each ~half as long.
- `LD_k`: one `ds_read` of A or B half-tile + one `G::load` prefetch into the next toc.
- `MMA_k`: one `mma_ABt` over a `C_accum[i][j]` = 16 mfmas of 16x16x32 ≈ 256 cyc.

```
phase:     p1     p2     p3     p4     p5     p6     p7     p8
row0:    LD0  | MMA0 | LD1  | MMA1 | LD2  | MMA2 | LD3  | MMA3 |  ...
row1:    (B1) | LD0  | MMA0 | LD1  | MMA1 | LD2  | MMA2 | LD3  | MMA3
XDL:     idle | row0 | row1 | row0 | row1 | row0 | row1 | row0 |  ...
mem:     row0 | row1 | row0 | row1 | row0 | row1 | row0 | row1 |  ...
```

Vertical = same wall-clock; `|` = `s_barrier` (all 8 waves meet). row1 is one phase behind via the
`warp_row==1` prologue barrier (B1), same as 32x16. Each MMA phase: `s_setprio(1)` wraps it,
`lgkmcnt(0)` before it; the next prefetch's `vmcnt` stays in flight and is drained at a later phase
(the kernel uses partial `lgkmcnt(8)` / `vmcnt(N)` drains instead of full per-phase drains, so the
exact placement varies — see the source for which phase each drain lives in).

Resource accounting per K-step (single SIMD, both halves alternating):
- XDL serves 4 row0 mma_ABts + 4 row1 mma_ABts = 8 × 16 mfmas × 16 cyc = **2048 cyc XDL-busy**.
- Wall-clock per K-step ≈ **2236 cyc** (ATT).
- XDL utilization ≈ **91.6%** — the ~9% gap is at the 8 phase boundaries (~24 cyc each).

Compared to 32x16 (4 phases × 512 cyc), 16x32 doubles the number of phase boundaries (8 vs 4) but
halves each MMA's wall-clock contribution (~256 vs ~512 cyc). Net XDL utilization is essentially
the same (~91% either way) — the finer-grained pipeline doesn't penalize itself with extra
boundary overhead, but it doesn't unlock more XDL-busy time either. The MFMA-shape's per-cycle
power, not the pipeline granularity, is what differs (see Headline).

### Prologue load order — not load-bearing (tested)

The prologue (lines 117–120) issues `Bs[tic][0], As[tic][0], Bs[tic][1], As[tic][1]` in that order
— B-first per pair. Reasonable hypothesis: the loop's first `ds_read` consumes `Bs[0][0]`, and
issuing B first lets it land in LDS earliest (if buffer-loads complete roughly FIFO), minimizing
the loop's first stall.

**Tested at 8192³ (MI355X, warm GPU, back-to-back):** swapping to A-first
(`As[tic][0], Bs[tic][0], As[tic][1], Bs[tic][1]`) gives 1217.5 TFLOPS vs the B-first baseline's
1216.7 — **within 0.07% (noise)**, both correct. So the consumption-order hypothesis doesn't hold
at this scale; whatever ensures correctness for B-first works for A-first too. By the time the
loop's first `ds_read` actually executes, evidently all four prologue buffer_loads have retired
regardless of issue order — likely because the prologue's `s_barrier`s + subsequent loads provide
plenty of overlap cycles for the early DMAs to land. So **the prologue order is stylistic, not
load-bearing**.

### Fine-grained waitcnts

The 16x32 kernel uses `s_waitcnt lgkmcnt(8)` and `vmcnt(N)` in places (lines
147, 189, 203, 243, 282, 300, 311) where the 32x16 used only full drains:

- `s_waitcnt lgkmcnt(8)` — wait until at most 8 LDS ops are still outstanding.
  Keeps later ds_reads in flight to overlap with the upcoming MMA.
- `s_waitcnt vmcnt(N)` — wait until at most N buffer_loads outstanding. Used
  to bound prefetch backlog at specific schedule points.

These are throughput-tuning, not correctness-load-bearing: the partial
drain lets later memory ops continue while the wave issues compute. Whether
the exact `N` is optimal is a profiler call.

#### lgkmcnt(8) at LD0 of both K-steps — tested, no measurable effect

The two `s_waitcnt lgkmcnt(8)` instructions (lines 147, 203) appear only at
LD0 of each K-step — the two LD phases doing 12 ds_reads (4 for B + 8 for A)
before their respective barriers. Other LD phases (with 4 or 8 ds_reads) skip
the partial drain. Intuitively this might look like a phase-balance trick to
lengthen the heavy LD phases to align with their MMA partners, but ATT
measurement says it's mostly inert.

**Tested at 8192³, MI355X, warm-state back-to-back:** removing both
`lgkmcnt(8)` instructions ("V1") gives 1217.9 TFLOPS vs the baseline's 1216.4 —
within the ~0.15% run-to-run noise floor. ATT shows the cycles redistribute:
the LD segments containing the drain (seg0, seg8) lose ~2 cyc of wave-PC time
and gain ~4 cyc of barrier wait, net ~zero. Total barrier-wait per loop iter
drops 777 → 765 cyc (~12 cyc / ~0.3%); XDL utilization 91.6% → 92.1%. None of
this is large enough to show up in wall-clock TFLOPS.

Best reading: the drain's cycles get absorbed into the `s_barrier` wait that
immediately follows. Without the explicit drain, the wave races to the barrier
and parks there the same total time. The partial drains look vestigial in the
current kernel state — possibly correct for a prior tuning iteration with a
different phase balance, but inert here.

Recommendation: leave them as-is. Removing them buys at most a sub-noise
improvement, and the cost if some workload (different M/N/K, different MI3xx
revision) shifts the balance is real. This is a tested null, not a green light
to refactor.

## Per-cluster ATT measurements (`.claude/skills/gemm-att-analysis`)

Captured at 8192³ on MI355X, one CU, 32 waves, 63 loop iterations (each = 2 K-steps).

Loop-body autodetect: hitcount=2016 = 32 waves × 63 iterations. ✓

The analyzer found 16 segments per loop-iter (the 16 `s_barrier`s). Per-segment
cyc/wave/iter (i.e. per 2 K-steps), abridged:

| seg | role               | total | mfma | dsread | gload | waitcnt | barrier wait |
|----:|--------------------|------:|-----:|-------:|------:|--------:|-------------:|
|   0 | A half-0 LD + prefetch | 299 | – | 172 | 63 | 4 | 20 |
|   1 | MMA C[0][0]        | 313  | 247 | – | – | 4 | 54 |
|   2 | B half-1 LD + prefetch | 278 | – | 39 | 92 | – | 106 |
|   3 | MMA C[0][1]        | 270  | 244 | – | – | 4 | 14 |
|   4 | A half-1 LD + prefetch | 273 | – | 99 | 91 | – | 40 |
|   5 | MMA C[1][0]        | 270  | 244 | – | – | 4 | 14 |
|   6 | B' half-0 LD + prefetch | 262 | – | 37 | 84 | 4 | 97 |
|   7 | MMA C[1][1]        | 272  | 244 | – | – | – | 20 |
| ... | (8–15: second K-step, same pattern) | | | | | | |

Per-K-step (sum of 8 segments / 2): wall-clock ≈ **2236 cyc/wave/K-step**,
mfma sum ≈ 976 cyc, barrier waits ≈ 195 cyc.

**Steady-state XDL utilization** (analyzer output, per 2-K-step iteration):
```
XDL busy per iter: 8 MMA segs × 2 halves × 16 mfmas × 16 cyc = 4096 cyc
Wall-clock per iter (sum of 16 segs):                          4472 cyc
⇒ utilization ≈ 91.6%
```

Compare 32x16: 2048 / 2253 = 90.9%. **Essentially identical.** Both kernels
are right at the XDL throughput floor with ~9% gap at phase-boundary handoffs.

Sanity check: `sum(CSV.Latency) / sum(wave.duration) = 0.9947` — the
Latency-column = decoder `duration` mapping holds tightly (cf. 32x16's 0.9358).

## Why is 16x32 faster, then?

Per the ATT numbers: **it isn't, structurally.** Same XDL utilization, same
per-K-step compute (1024 cyc both ways), same per-K-step wall-clock
(2236 vs 2253, within 1%). If both ran at the same clock, they'd produce the
same TFLOPS.

But they don't run at the same clock. Sampled `amd-smi metric` on GPU 3
(the busy device) under sustained load, with warm-GPU back-to-back TFLOPS:

| kernel | sustained GFX_CLK | SOCKET_POWER | achieved TFLOPS |
|--------|------------------:|-------------:|----------------:|
| 32x16  | ~1.30 GHz          | ~1388 W      | 1098.2          |
| 16x32  | ~1.48 GHz          | ~1390 W      | 1217.3          |

Same power budget; ~13.8% higher clock on 16x32. TFLOPS ratio 1.108 ≈ a
slightly discounted clock ratio. The ~3% discount lines up with per-K-step
constants that don't scale with clock — chiefly the global-load latency
captured in c2's `s_waitcnt(0)` drain (~196 cyc/wave/iter on 32x16) which
becomes a larger fraction of K-step wall-clock at the higher clock.

**Working hypothesis (not independently confirmed):** 16x16x32 mfmas are less
power-intensive per cycle than 32x32x16 (smaller per-mfma data path, fewer
fp32 outputs written per issue), so the power cap leaves headroom for a
higher clock. The schedule isn't smarter; the math runs cooler.

What would confirm it: instrumenting per-instruction power (the
`SQ_INSTS_VALU_MFMA_*` PMC counters), or sweeping `--power-cap` / running on a
non-power-capped configuration. Out of scope for this walkthrough.

## What this means for tuning

- **Don't chase scheduling improvements inside the loop body of either
  kernel.** Both are at ~91% XDL utilization; the ~9% slack is phase-boundary
  handoff overhead, hard to attack without restructuring the ping-pong.
- **If you're power-bound** (this card is, under dense bf16 GEMM), the
  MFMA-shape choice is doing more than you think — picking the lower-power
  shape gets you ~8% for free.
- **If you're not power-bound** (smaller M/N, or a configuration without the
  TDP cap binding), the two kernels should converge to the same wall-clock per
  workgroup. The 16x32 form would lose its edge.

## Open questions (not resolved here)

- **Quantifying the ~3% clock-vs-TFLOPS discount.** Hand-wavy explanation
  matches the magnitude (memory latency doesn't scale with clock), but
  a per-cluster cycle decomposition at each clock would confirm rigorously.
- **Are the 16x32 per-MFMA stall numbers (~75%) lower than 32x16's (~86%)
  because of pipelining or because of a tighter MMA-cluster schedule?**
  Single-occupancy holds for both shapes per the ATT signature, so this is
  more likely "tighter cluster" — but I haven't isolated it.
- **Whether moving the c2 drain rule from the 32x16 finding carries over
  unchanged to 16x32.** The 16x32 file already uses partial `lgkmcnt(N)` drains;
  the analog of "c2-must-stay" hasn't been retested for this kernel. Not
  recommended to refactor without re-running the c2→c3 control matrix.

## Followups (run, blocked, deferred)

### Multi-CU ATT generalization (run, generalizes)
Ran ATT with `--att-target-cu 0` and `--att-target-cu 5` on the same 16x32
kernel; per-cluster numbers match within run-to-run noise:

| CU | K-step wall-clock | XDL utilization | sanity ratio |
|---|---|---|---|
| 0 | 4452 cyc | 92.0% | 0.9950 |
| 5 | 4443 cyc | 92.2% | 0.9948 |

Tried CUs at higher indices (31, 100, 200) but rocprofv3 core-dumped on each —
likely a target-CU id range limit in the decoder. CUs 0 and 5 are enough to
confirm the per-CU picture generalizes.

### Clock hypothesis — indirect confirmation (power-cap modification blocked)
Direct test (sweep `--power-cap` and observe TFLOPS converge) is blocked: the
`rocm-smi --setpoweroverdrive` and `--setperflevel` operations need root in
this environment.

Indirect confirmation via per-cycle efficiency math:

| kernel | TFLOPS | clock | flops/cyc/SIMD | % of peak |
|---|---|---|---|---|
| 32x16 | 1098 | 1.30 GHz | **825** | 80.6% |
| 16x32 | 1217 | 1.48 GHz | **803** | 78.4% |

(Peak = 1024 flops/cyc/SIMD for both shapes; ÷ by `clock × 1024 SIMDs` from each
TFLOPS measurement.)

The two kernels have **near-identical per-cycle efficiency** (~2.7% spread, with
32x16 slightly *more* efficient per cycle). Extrapolating to a same-clock
scenario:

| at 1.30 GHz (32x16's measured clock) | at 1.48 GHz (16x32's measured clock) |
|---|---|
| 32x16: 1099, 16x32: 1069 | 32x16: 1251, 16x32: 1217 |

At identical clock, 32x16 would beat 16x32 by ~3%. The 16x32's measured
TFLOPS advantage is **entirely** clock-driven — 16x16x32 mfmas are
lower-power per cycle than 32x32x16, so the power-capped GPU runs ~14% faster.
This is consistent with all measurements and is the strongest confirmation
possible without privileged power-cap control.

### 16x16x16 shape comparison (deferred)
Would require writing a new HK kernel variant — out of scope for this session.
The cleanest follow-up would still distinguish "MFMA tile size" from "specific
instruction microarchitecture" as the power driver: if a 16x16x16 BF16 variant
also clocks high under TDP cap, it's size; if not, it's specific to the
16x16x32 instruction. Worth doing if someone has reason to suspect a third
shape exists in HK or wants to write it.
