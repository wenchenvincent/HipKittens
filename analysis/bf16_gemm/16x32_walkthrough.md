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

**16x32 runs 1192.6 TFLOPS vs 32x16's 1097.8 — about 8.6% faster — but
the per-K-step compute cycles, per-K-step wall-clock cycles, and steady-state
XDL utilization are essentially identical between the two kernels.** The
speedup is almost entirely **clock-driven**: 16x32 sustains ~1.48 GHz under the
TDP cap vs 32x16's ~1.30 GHz. The hypothesis (consistent with the data, not
independently confirmed): 16x16x32 mfmas are lower-power per cycle than
32x32x16, so the power-capped GPU can clock higher running the smaller shape.

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

`for (int tile = 0; tile < num_tiles - 2; tile += 2)` (line 136) — the loop is
**unrolled by 2** (covers 2 K-steps per source iteration), so:

- 8 `mma_ABt` calls per source iteration (lines 148, 160, 171, 183, 196, 208,
  219, 229) ⇒ **4 mma_ABts per K-step**, one for each `C_accum[i][j]`.
- 16 `s_barrier`s per source iteration ⇒ **8 LD/MMA phase boundaries per K-step**.
  Each LD phase loads one A or B half-tile + issues one global prefetch; each
  MMA phase is one `mma_ABt` call.

The schedule rotates through the 4 LDS half-tile buffers and the 2×2
accumulator grid, with prefetches into `tic`-toc-swapped buffers interleaved.

### Fine-grained waitcnts

The 16x32 kernel uses `s_waitcnt lgkmcnt(8)` and `vmcnt(N)` in places (lines
143, 179, 191, 225, 262, 279, 290) where the 32x16 used only full drains:

- `s_waitcnt lgkmcnt(8)` — wait until at most 8 LDS ops are still outstanding.
  Keeps later ds_reads in flight to overlap with the upcoming MMA.
- `s_waitcnt vmcnt(N)` — wait until at most N buffer_loads outstanding. Used
  to bound prefetch backlog at specific schedule points.

These are throughput-tuning, not correctness-load-bearing: the partial
drain lets later memory ops continue while the wave issues compute. Whether
the exact `N` is optimal is a profiler call.

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
(the busy device) under sustained load:

| kernel | sustained GFX_CLK | SOCKET_POWER | achieved TFLOPS |
|--------|------------------:|-------------:|----------------:|
| 32x16  | ~1.30 GHz          | ~1388 W      | 1097.8          |
| 16x32  | ~1.48 GHz          | ~1390 W      | 1192.6          |

Same power budget; ~14% higher clock on 16x32. TFLOPS ratio 1.087 ≈ a
discounted clock ratio (some loss to per-K-step constants that don't scale
with clock — global-load latency in particular).

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

- **Why does the 14% clock gap only translate to 8.6% TFLOPS?** Per-K-step
  global-load latency (~196 cyc on c2's `s_waitcnt(0)` drain on 32x16) doesn't
  shrink with clock; at higher clock it's a larger fraction of K-step
  wall-clock. Could explain the ~5% discount, but I haven't computed it
  rigorously.
- **Are the 16x32 per-MFMA stall numbers (~75%) lower than 32x16's (~86%)
  because of pipelining or because of a tighter MMA-cluster schedule?**
  Single-occupancy holds for both shapes per the ATT signature, so this is
  more likely "tighter cluster" — but I haven't isolated it.
- **Whether moving the c2 drain rule from the 32x16 finding carries over
  unchanged to 16x32.** The 16x32 file already uses partial `lgkmcnt(N)` drains;
  the analog of "c2-must-stay" hasn't been retested for this kernel. Not
  recommended to refactor without re-running the c2→c3 control matrix.

## What I'd do next

- Sweep `--power-cap` to verify the clock hypothesis. If the 8.6% gap closes
  as power-cap rises, the clock story is confirmed.
- Run ATT with `--att-target-cu` covering multiple CUs and average — checks
  whether the per-CU picture generalizes.
- Compare power and clock signatures of a hypothetical 16x16x16 shape (if HK
  adds one) — would distinguish "shape size" from "specific instruction
  microarchitecture" as the power driver.
