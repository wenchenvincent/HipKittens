"""Single-invocation driver template for ATT capture.

Copy next to a built `tk_kernel` module and edit M/N/K + dispatch args to match
the kernel under test. Do NOT loop launches — ATT traces are large; one launch
is enough.

Usage:
    python3 driver_template.py
"""
import torch
import tk_kernel
from utils import init_randn, init_empty  # from the HK kernel dir's utils.py

torch.manual_seed(0)
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

# Edit these to match the kernel's expected dimensions.
# The 32x16 BF16 variant hardcodes 8192^3 via `#define`; the 16x32 variant
# accepts runtime dims.
M = N = K = 8192

A = init_randn((M, K), DTYPE, DEVICE)
B = init_randn((K, N), DTYPE, DEVICE)
# Most HK GEMM kernels use mma_ABt and want B passed as Bt (shape N,K).
Bt = B.t().contiguous()
C = init_empty((M, N), DTYPE, DEVICE)

# Single launch. Adjust the dispatch call if your kernel binds a different name.
tk_kernel.dispatch_micro(A, Bt, C)
torch.cuda.synchronize()
print("done")
