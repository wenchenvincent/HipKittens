# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HipKittens (HK) is the AMD port of ThunderKittens (TK): a header-only C++20 library of embedded primitives for writing fast AMD AI kernels. It targets CDNA3 (MI300X/MI325X, `gfx942`) and CDNA4 (MI350X/MI355X, `gfx950`). The `main` branch targets CDNA4; use the `cdna3` branch for MI300X/MI325X. Everything is included via `include/kittens.cuh`. Code namespace is `kittens` and still mostly says "ThunderKittens"/`tk_kernel` — that is expected.

## Environment

Must run inside the AMD ROCm Docker (see README/`docs/docker/`). Before building anything: `source env.src` — sets `THUNDERKITTENS_ROOT`, `HIP_HOME`, `ROCM_PATH`, and adds `pyutils` to `PYTHONPATH`. Kernels are compiled with `hipcc` into pybind11 `.so` modules and benchmarked against AITER/PyTorch, so install AITER from source (`ROCM/aiter`) for comparisons.

## Build & run

Each kernel/test directory has its own Makefile keyed on `GPU_TARGET` (`CDNA3`→gfx942 / `CDNA4`→gfx950, which sets `-DKITTENS_CDNA3`/`-DKITTENS_CDNA4`). Kernels build to a pybind11 module named `tk_kernel`.

```bash
# A kernel (e.g. GEMM, attention): build then bench/test
cd kernels/gemm/bf16fp32/ && make clean && make && python bench.py
cd kernels/attn/gqa/      && make clean && make && python test_python.py
# Dims (ATTN_N, ATTN_H, ATTN_H_KV, ATTN_D, ...) are set in the Makefile + test_python.py
```

```bash
# Unit tests — ALWAYS parallelize; single-threaded takes hours
cd tests/unit && make -j64 && mkdir -p outputs && ./unit_tests printout
```

Unit-test scope is selected at compile time in `tests/unit/Makefile`: `-DTEST_INTENSITY=1..4` (count) and `-DTEST_ALL` or a section/single flag (e.g. `-DTEST_WARP_MEMORY`, `-DTEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER`) — change these to run one test. MMA tests occasionally fail by one element due to tensor-core float behavior; this is expected. If `hipcc` isn't found, point the Makefile at `/opt/rocm/bin/hipcc`.

## Architecture

Three layers, all header-only:

1. **Types** (`include/types/`): `register/` tiles `rt` + vectors `rv`, `shared/` tiles `st` + vectors `sv`, `global/` layout `gl`. Tile *shapes* (16x16, 32x32, 16x32, …) are sized to MFMA tensor-core units; correct shape/layout selection is core to performance. A wavefront is 64 threads.
2. **Ops** (`include/ops/`): `warp/` is the default scope (single wave); `group/` spans multiple warps. Each has `register/`, `shared/`, `memory/` ops. `assembly/` subdirs hold raw-asm fast paths.
3. **Common** (`include/common/`): `base_types.cuh` (bf16/fp8e4m3/fp4/mxfp8 + constants), `base_ops.cuh`, and `macros.cuh` — hand-written inline asm wrappers: `ds_read/write_bXX` (incl. transpose `_tr` loads), `buffer_load/store_dwordxN` (direct buffer loads to LDS), and `mfma_*` tensor-core ops, all templated on explicit GPR numbers.

Performance levers that differ from CUDA: coalesced/bank-conflict-free memory ops, async direct buffer loads, and explicit scheduling — kernels use `__builtin_amdgcn_sched_group_barrier`/`s_barrier` for the two core overlap patterns (8-wave ping-pong, 4-wave interleave). Tiles eagerly use tensor-core layouts to minimize address computation.

Kernel↔Python: a kernel `.cpp` includes `kittens.cuh` + `pyutils/pyutils.cuh`, defines `gl<dtype,...>` globals, and exports via `PYBIND11_MODULE(tk_kernel, ...)` / `py::bind_function`. Tensors must be CUDA + contiguous, ndim ≤ 4.

## Other directories

- `kernels/` — gemm (bf16fp32, fp8fp32, mxfp8), attn (gqa, gqa_causal, + backwards), rotary, layernorm, softmax.
- `analysis/` — paper benchmark sweeps + plotting; `analysis/baselines/` reproduces Triton/CK/HipBLASLT/Mojo. `tests/unit/` mirrors the `include/` tree for coverage.
- `distributed-kernels/`, `training/` (bert, llama) — separate, CMake/setup.py based.
