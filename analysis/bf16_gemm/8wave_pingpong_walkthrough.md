# 8-wave ping-pong BF16 GEMM walkthrough

Notes on `kernels/gemm/bf16fp32/256_256_64_32_with32x16.cpp` (32x32x16 MFMA variant), with the
corrected arithmetic. The `16x32` sibling is the same algorithm hand-scheduled harder
(readfirstlane SGPR hoisting, fine-grained waitcnts, 4 accumulators of 16x16, 16x16x32 MFMA).

> Shape/index arithmetic verified from code; all perf/balance/L2 claims are profiler-dependent
> (rocprof, `docs/profiling/`, `analysis/*/profile.sh`).

## A/B/C orientation
Computes C = A·Bᵀ, all bf16 in HBM, fp32 accumulate. Memory (row-major gl): A=(M,K), B=(N,K) i.e.
Bᵀ stored row-major, C=(M,N). bench calls dispatch_micro with transpose_B=True. Kernel uses
`mma_ABt` (B un-transposed, contract on shared K cols: A::cols==B::cols).

## Workgroup → C tile
- One workgroup computes a **256×256** C tile, accumulating over K in steps of 64.
- `NUM_WARPS = WARPS_M(2) × WARPS_N(4) = 8`; `NUM_THREADS = 64 × 8 = 512`. A wavefront is 64
  threads (= AMD "warp"). The two `256`s — 512 threads and a 256×256 tile — are unrelated; `BLOCK_SIZE`
  is the tile, not the thread count.
- 8 warps in a 2×4 grid → each warp owns **256/2 = 128 rows (M) × 256/4 = 64 cols (N)**.

## Per-warp 128×64 tile → MFMAs
- `C_accum = rt_fl<128, 64, rt_32x32_s>` → a 4(M) × 2(N) = **8** sub-tiles of 32×32.
- Each `mfma_f32_32x32x16_bf16` produces 32×32 from K=16. One K-step is 64, so each accumulator
  needs `64/16 = 4` MFMAs → **8 × 4 = 32 MFMAs per warp per K-step** (16 per cluster).
- Lanes: 32×32 = 1024 / 64 = 16 fp32 per lane = `floatx16`.

### row_l / col_l = MFMA operand layout
Not host memory order — which lane holds which element for the MFMA registers. `reductions = row?
cols : rows` (rt_base.cuh:66), so K runs along cols for row_l, rows for col_l. Counts (32x16 variant):
A/B `rt_32x16_s` = 512/64 = 8 bf16/lane in row_l (inputs); C `rt_32x32_s` = 1024/64 = 16 fp32/lane in
col_l (mfma writes result transposed). store un-transposes back to gl. 16x16/16x16x32 only on the
16x32 sibling. Pick layout = what the next MFMA produces/consumes, not memory.

## mma_ABt nesting (include/ops/warp/register/tile/mma.cuh:445)
4×2×2 unrolled: `n<D::height`(4 M) × `m<D::width`(2 N) × `k<A::width`(2 per 32-wide cluster).
`k=0` reads incoming `c`, `k≥1` accumulates into `d`. Leaf `mma_ABt_base` constexpr-dispatches on
shape → 32x16 ops → `mfma323216`; 16x32 ops → `mfma161632`.

## Scheduling

### Ping-pong
8 warps split into halves; `s_setprio`/`s_barrier` hand SIMD issue to the computing half while the
other drives LDS/global, then swap. `tic/toc` double-buffers shared `As[2]/Bs[2]`. ("cluster" =
"phase" below: one barrier-delimited segment of the loop; the loop has 4 per K-step.)

### Stagger (how the lead/lag is set)
Prologue B0: all 8 waves sync. Then `if(warp_row==1) s_barrier` (B1): only row1's 4 waves
wait there. s_barrier needs all 8, so row1 parks at B1 while row0 runs cluster0 and hits its first
loop barrier — that 8th arrival releases both. Now halves are locked exactly 1 cluster apart: row0
computes while row1 loads, swapping each barrier. Structural, not timed: barrier latency is a fixed
tax, doesn't set the offset. 16x32 rebalances `if(warp_row==0) s_barrier` before store. Overlap
*quality* (not the stagger) depends on cluster duration balance.

### Pipeline diagram
Cluster contents (32x16 loop): c0 = ds_read slice0 + `G::load` global prefetch→toc; c1 = mma×8;
c2 = ds_read slice1 + `s_waitcnt(0)` drain; c3 = mma×8. Each `|` is an `s_barrier` (all 8 waves
meet); halves offset 1 phase via the B1 stagger:
```
phase:    p0     p1     p2     p3     p4
row0:   c0 LD | c1 MMA| c2 LD | c3 MMA|  ...
row1:   ----- | c0 LD | c1 MMA| c2 LD | c3 MMA   (1 phase behind, via warp_row==1 barrier B1)
              └─ MMA(0) overlaps LD(1) ┘
```
Vertical = same wall-clock; `|` = barrier. Whichever half is in MMA is paired with the other in LD,
so the matrix unit is fed every phase. Per MMA cell: `setprio(1)` wraps it, `lgkmcnt(0)` before it
(LDS landed; vmcnt prefetch stays in flight). Buffers: compute reads tic ‖ prefetch fills toc → swap
at K-step end.

### s_setprio
Wraps every mma: `setprio(1); mma; setprio(0)`. Both warp-row halves share a SIMD; issue funnels
through one scheduler. setprio(1) lets the computing half win issue slots so MFMAs run full-rate;
the loading half (prio 0) waits to issue its buffer_load/ds_read — but those need ~1 slot then run
async, so the delay is cheap. Per-cycle complement to the stagger: barriers pick the phase, setprio
makes compute dominate within it. Hint only.

### s_barrier vs sched_barrier
`s_barrier` = real instruction, runtime cross-wave sync (the 8-wave rendezvous). `sched_barrier(mask)`
= no instruction, compile-time fence limiting what the scheduler reorders across it (mask=0 = none),
intra-wave only. 32x16 has one `sched_barrier(0)` at epilogue start (line 127): fences loop body from
epilogue so the compiler can't hoist epilogue loads/MMAs into the last iteration (epilogue has no
prefetch to hide). 16x32 also uses it inside the loop between MMA blocks — meaning is "don't reorder
across me" wherever placed. (attn kernels use finer `sched_group_barrier`.)

### Balance assumptions (overlap only pays if)
1. Global prefetch (1 G::load/buffer per iter, consumed 4 clusters later) hides within ~1 K-step =
   double-buffer depth bet; slower HBM → toc not ready at swap → stall.
2. Per cluster, LDS ds_read of next slice ≈ 16 MFMA time; mismatch idles matrix unit or lanes.

K_STEP/DOT_SLICE + MFMA shape are sized for this — fp8/mxfp8/16x32 variants exist for different
balances. Latency sources: MFMA throughput per shape is in CDNA3/4 ISA matrix-instruction tables
(deterministic); LDS/global latency under load (bank conflicts, queue depth, cache) is not fixed —
measure via PMC counters. Balance = the interaction; always profiled per kernel, not predicted.

### why c2's s_waitcnt(0) must stay in c2 (tested)
Tempting idea: c2's `__builtin_amdgcn_s_waitcnt(0)` is a full drain whose `lgkmcnt` part looks
redundant with c3's `lgkmcnt(0)`, so move the `vmcnt`/prefetch drain to c3 to give the As[toc]/Bs[toc]
prefetch more flight time. **Tested at 8192³ on MI355X — it breaks correctness (err ~200 vs bf16
noise 4), with zero perf change (1097–1101 TFLOPS, noise).** Matrix: drain in c2 = PASS; drain in c3
(before OR after the mma) = FAIL; drain in both = PASS. So the c2 drain is load-bearing, not
conservative.

Mechanism (ISA-confirmed: only diff is `s_waitcnt(0)`↔`s_barrier` order; no instruction hoist).

Two facts that combine:
- `G::load` is a **group** load — As[toc] is written by all 512 threads (laneid/warpid partition),
  but the c0 `ds_read` consumes it in a different (warp_row/warp_col) partition ⇒ a wave reads bytes
  *another wave* prefetched.
- `s_waitcnt` is **per-wave**; `s_barrier` syncs execution but does NOT drain memory. So only row1's
  own drain retires row1's prefetch DMA.

row0 leads by one cluster, so the wall-clock alignment is:
```
row0 c1        ‖ row1 c0
row0 c2        ‖ row1 c1
row0 c3        ‖ row1 c2
row0 c0(N+1)   ‖ row1 c3(N)   ← leading half's next-iter read aligns with lagging half's c3
```

Trace with drain in c3:
- row1's prefetch (issued c0 of iter N) is drained in row1 c3(N).
- row1 c3(N) is the same phase as row0 c0(N+1).
- row0's c0(N+1) `ds_read` executes *before* that phase-ending barrier — i.e. before row1's c3 drain
  is synchronized to row0.
- ⇒ row0 reads row1's still-in-flight prefetch slice → garbage. Corruption.

Trace with drain in c2 (baseline):
- row1's prefetch drains in row1 c2(N), which aligns with row0 c1(N) — a full cluster before row0's
  c0(N+1) read, with a barrier in between. Safe.

Takeaway: under ping-pong, the group-prefetch must be drained one cluster ahead of the leading half's
consuming read. Without the stagger, c3 would be fine. (Repro harness `_exp_*` was scratch, removed.)

## WGID swizzle (2 stacked remaps)
HW dispatches workgroups round-robin across 8 XCDs, so consecutive tiles scatter chiplets, killing
L2 reuse. Two remaps fix it:
1. Chiplet: 32x16 inline `new=(id%8)*(NW/8)+id/8` (band=NW/8); 16x32 `chiplet_transform_chunked(64)`.
   XCD = new%8, so 8 consecutive origs pin to one XCD (origs 0-7→XCD0, 8-15→XCD1...).
2. Super-group (WGM=8): band is 8 tiles tall in M × full N. `group_size_m=min(num_pid_m-first,WGM)`
   clamps M (WGM is the M block; num_pid sweeps N). Walk 8 M-tiles → B-col reuse, advance N → A reuse.

Both are bijections (verified: 1024 ids → distinct pid for square; full coverage for 8192×4096).

## Tiling diagrams (32x16 variant)

1. Workgroup C tile → 8 warps (2x4):
```
            N = 256  (4 warp-cols)
      ┌──────┬──────┬──────┬──────┐
M=256 │ w0   │ w1   │ w2   │ w3   │  warp_row 0
(2    │128×64│128×64│128×64│128×64│
warp- ├──────┼──────┼──────┼──────┤
rows) │ w4   │ w5   │ w6   │ w7   │  warp_row 1
      │128×64│128×64│128×64│128×64│
      └──────┴──────┴──────┴──────┘
   warp_id = warp_row*4 + warp_col   (512 threads = 8 waves × 64)
```

2. Warp's 128×64 → 8 accumulators of 32×32 (C_accum[4][2]):
```
        64 (2)
   ┌────┬────┐
   │0,0 │0,1 │
128│1,0 │1,1 │   each cell = rt_32x32 = 16 fp32/lane
(4)│2,0 │2,1 │
   │3,0 │3,1 │
   └────┴────┘
```

3. K-step = 64 → 2 clusters × 2 K-slices (1 cell = 1 MFMA, K=16):
```
 K: |16|16|16|16|   each accumulator: 4 MFMAs
     └c0─┘└c1─┘     8 accs × 4 = 32 MFMAs / warp / K-step
```

4. mma_ABt nest = 4(M) × 2(N) × 2(K per cluster) = 16 mfma323216:
```
A:128×16  ·  Bᵀ:16×64   →   accum 32×32   →  mfma323216
```

## Glossary

### gl<>
`gl<T, b, d, r, c>` = global layout: HBM tensor handle = `raw_ptr` + 4 dims (batch, depth, rows,
cols), each compile-time or `-1` for runtime. `g.a[{b,d,r,c}]` row-major flattens; `stride<axis>()`
feeds make_srsrc. Maps 1:1 to a contiguous CUDA torch tensor (ndim ≤ 4). `rt`/`st` are the
register/shared tiles copied in/out of it. `TMA_Types...` is vestigial — unused on AMD.

### st_bf<> / rt_bf<>
Dtype-specialized tile aliases (`bf`=bf16, `fl`=float). Same data, different home:
`st_bf<rows,cols,shape>` = shared/LDS tile (As/Bs staging), diced into `shape` sub-tiles for
swizzle-free access. `rt_bf<rows,cols,layout,shape>` = register tile across a wave's 64 lanes;
`rt_fl` = fp32 accumulators. shape = MFMA sub-tile (must match mma_ABt_base dispatch); layout =
row_l (inputs) / col_l (accum). Flow: gl(HBM) → st_bf(LDS) → rt_bf(regs) → mma → rt_fl → gl.
