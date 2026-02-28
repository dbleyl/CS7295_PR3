# My Notes for Project 3

## Goal 1: Compute Core and Tensor Core

**Traces:**

- `gemm_float` for compute core `FFMA`
- `gemm_half` for tensor core `HMMA` instructions, `float16` operations: `HMMA.1688.F16`
- Both use **NVbit** for trace generation

> FFMA (Fused Floating-point Multiply-Add)
>
> - Operation: `D = A * B + C` in a single instruction
> - Type: Standard CUDA core instruction
> - Precision: Single-precision floating-point (FP32)
> - Usage: General compute workloads (your gemm_float benchmark uses FFMA)
> - Benefit: Same latency as FMUL or FADD alone, but provides 2x throughput

> HMMA (Half-precision Matrix Multiply-Accumulate)
>
> - Operation: Matrix multiply-accumulate on Tensor Cores
> - Type: Tensor Core instruction (warp-level)
> - Precision: Half-precision (FP16), as seen in your traces: HMMA.1688.F16
> - Usage: Matrix multiplication workloads (your gemm_half benchmark uses HMMA)
> - Benefit: Specialized for deep learning and matrix operations, much higher throughput than FFMA for matrix math

> :warning:
> "`macsim` doesn't utilize the flags that follow the opcode."

### Task 1: Implement Naïve Compute Core

- `core.cpp` $\rightarrow$ `run_a_cycle()`.

1. **Identify compute instructions:** check if it's compute, if so retrieve **latency**.
2. **Buffer compute instructsions:** if so, add to `c_exec_buffer` **per core** with **completion cycle timestamp**.
3. **Handle Buffer Fullness:** stall if full.
4. **Execute Instructions from Buffer:** Periodically, check whether current cycle timestamp exceeps the completion timestamps.
