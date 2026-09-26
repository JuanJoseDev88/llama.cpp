# PTQ1_0 on AMD RDNA (HIP)

Branch `amd-ptq1_0` makes the PTQ1_0 (1.75-bit ternary) quantized type use native kernels on AMD instead of the fp16 dequantize + hipBLAS fallback. On an RX 6700 XT decode goes from ~20 t/s to 37 t/s and prompt processing from ~200 t/s to 256 t/s.

Build instructions are in [build.md](build.md#hip). Graph capture, profiling and the test gate for HIP are in [Runtime notes and profiling](build.md#runtime-notes-and-profiling).

## What was missing on HIP

- `ggml_cuda_should_use_mmq` answered `mmq_supported = turing_mma_available(cc)` for `GGML_TYPE_PTQ1_0`, which is false on AMD, so the MMQ tile path was never selected for prefill.
- PTQ1_0 tile geometries existed only in `mmq-config-ampere.cuh`; the RDNA2, RDNA3, RDNA3.5, RDNA4 and CDNA headers had none.
- Decode on AMD fell through to the generic single-column path: the PTQ1_0 multi-column case in `mmvq.cu` and `vec_dot_ptq1_0_q8_1_multi` in `vecdotq.cuh` sat behind `#if !defined(GGML_USE_HIP)`, the MoE kernel had no PTQ1_0 case, and no path used a planar-transposed activation layout, so the stock kernel walked the activations as scattered 4-byte loads per column.
- The trit decode helpers called CUDA intrinsics that ROCm does not map to hardware: `__byte_perm` and `__vsub4` are software routines in HIP.

## What the branch implements

### Prefill, MMQ tile path

`ggml_cuda_mmq_load_tiles_ptq1_0` and `ggml_cuda_mmq_decode_ptq1_0_qs4` in `mmq-load-tiles.cuh` expand the packed trits straight to int8 in shared memory with integer ALU and `ggml_cuda_dp4a`, so the tile dot is the standard MMQ dot: SIMT dp4a below RDNA3, WMMA/MFMA on RDNA3 and newer. No NVIDIA-only instruction is involved, which is why `mmq_supported` is now unconditionally true for PTQ1_0.

Tile geometries mirroring Q1_0 (the loader produces one int8 per weight and the dot is dp4a over signed bytes, so register pressure and tile sizes match) were added to the five AMD headers: 11 cases for RDNA2, 12 each for RDNA3, RDNA3.5 and RDNA4, 7 for CDNA.

### Decode, mat-vec

The new `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh` holds `mul_mat_vec_ptq1_0_pt` for the scalar, multi-column and gated (fused wkv) paths, and `mul_mat_vec_ptq1_0_pt_switch` is dispatched from the plain 2D launcher. The MoE kernel in `mmvq.cu` gained a PTQ1_0 case as well.

The kernel decodes from a planar-transposed Q8_1 activation layout. For each activation column the 128 quants a thread needs are 8 aligned 16-byte pieces, one per plane, and the 4 (d, s) scales are one more piece; adjacent threads read adjacent pieces of the same plane, so a warp load touches 4 cache lines instead of 32, and a thread reuses each piece for every row it owns. The column stride equals the `block_q8_1` stride, so every stride the mmvq launcher computes in `block_q8_1` units stays valid. The decode, the dp4a sequence and the fp32 epilogue run in the same order as the single-column kernel, so every column count produces the same bits for a given column. `quantize.cu` emits this layout, and the quantizer and the mat-vec dispatch call the same helper, so they cannot disagree.

### HIP decode helpers

`vec_dot_ptq1_0_q8_1` and `vec_dot_ptq1_0_q8_1_multi` in `vecdotq.cuh`, `dequantize_block_ptq1_0` in `convert.cu`, and the four permute/subtract helpers (`ptq1_0_interleave_hi`, `ggml_cuda_mmq_ptq1_0_interleave_hi`, `ptq1_0_dq_byte_perm`, `ptq1_0_pt_byte_perm`) gained HIP branches. The trit interleave, the two widen selectors and the tail subtract are emitted as plain ALU, because HIP resolves `__byte_perm` and `__vsub4` to software routines. The non-HIP branch keeps calling `__byte_perm` and `__vsub4`.

The tail subtract needs one more fix on AMD: there is no `__vsub4`, and a plain 32-bit subtract lets borrows cross byte lanes and corrupt the {0,1,2} digit bytes, so the HIP branch uses the borrow-free per-byte form `(((a | 0x80808080) - (b & 0x7F7F7F7F)) ^ 0x80808080)`.

### Launch geometry

`calc_nwarps` pins PTQ1_0 to 4 warps for every column count, so the fp32 sums of a column do not depend on how many columns shared the launch (batch invariance). `calc_rows_per_block` now takes the quantized type and defers to `ptq1_0_pt_rows_per_block`. On NVIDIA Turing and newer the mmvq preference for PTQ1_0 now covers the full mmvq batch (`ne11 <= MMVQ_MAX_BATCH_SIZE`, it was 7) because the weight decode is shared across columns.

## RDNA decode tuning

| step | change | tg128 |
|---|---|---:|
| native kernels | `ROWS` 4, ALU permute expansion | 30.47 t/s |
| E1 | one row per work item (`ROWS` 4 -> 2 -> 1): the trit decode state shrinks to 83 VGPR and 11 waves stay resident per SIMD | 34.10 t/s |
| E1b | live-range fence | +0.7 t/s at `ROWS` 2 |
| E4 | deterministic row-group picker: utilisation tie-break plus a 160-CTA floor so small tensors still fill the GPU | wash, kept for determinism |
| E5 | `v_perm_b32` for the trit interleave, byte selector `0x03010705` | 35.88 t/s |
| E7 | `v_perm_b32` for the two widen steps, `0x00050004` and `0x00070006` | 37.00 t/s |

The fence in E1b is an empty `asm volatile("" ::: "memory")` before the gated `block_dot`. Without it LLVM interleaves the gated and the plain dot live ranges, which costs 207 VGPR and leaves 4 waves; with it the gate path needs 149 VGPR and 6 waves, which is what keeps `ROWS` 1 resident.

## Selector granularity, the trap in this port

`v_perm_b32` and `__builtin_amdgcn_perm` are byte granular: one whole byte per output position, index 0-7, and an out-of-range index returns 0xff. CUDA `__byte_perm` is nibble granular. A CUDA selector used unchanged (`0x7531`) therefore reads as bytes {0x31, 0x75, 0, 0}, two of them out of range, and silently decodes garbage weights. This is not a performance bug: every benchmark still reports full speed while the model emits token salad.

Convert a nibble selector to a byte selector with nibble n -> byte index `n < 4 ? n+4 : n-4`, keeping the operand order. That gives `0x7531` -> `0x03010705`, `0x4140` -> `0x00050004` and `0x4342` -> `0x00070006`; the verified constants are in `mmvq-ptq1_0.cuh`.

A compile-time constant selector is fine. The claim that ROCm mis-folds `amdgcn_perm` when the selector is a constant is a misdiagnosis of this granularity difference: a constant and a runtime selector return the same bytes.

## Validation

Gate every change on both of these:

- `./build/bin/test-backend-ops -o MUL_MAT -p ".*ptq1_0.*"` (45/45, GPU against CPU over the PTQ1_0 mul_mat cases).
- One greedy generation run: `llama-cli -m <model> -p "The capital of France is" -n 32 --temp 0 --seed 42 -st`, which must produce coherent text.

Host-side transcription tests never execute compiled device code, and benchmarks pass at full speed on a broken kernel, so a green test suite is not evidence that the model still generates text.

## Measurements

RX 6700 XT (gfx1030, ROCm 7.2.3), `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, `llama-bench -n 128 -p 0 -r 3`:

| | dequant + hipBLAS fallback | native kernels | + RDNA decode tuning |
|---|---:|---:|---:|
| tg128 decode | ~20 t/s | 30.47 t/s | **37.00 t/s** |
| pp512 prompt | ~200 t/s | 256 t/s | 256 t/s |

Fallback figures are from `llama-server` logs, the rest from `llama-bench`. The tuned decode holds 37.0 t/s at 15.6k context (`-p 15360`), streams about 87% of the card's measured bandwidth (301 of 347 GB/s), and leaves prompt processing untouched (235.3 -> 235.1 t/s at pp15360, the control measurement). A later re-run measured 37.24 +/- 0.07 t/s.

HIP graph capture is on by default and is worth about 6% here, so leave it on when benchmarking: 37.24 +/- 0.07 t/s with graphs on, 35.16 +/- 0.48 t/s with `GGML_CUDA_DISABLE_GRAPHS=1`.

The PTQ kernel now runs at 87% of the measured streaming ceiling, and the pure-bandwidth floor for the model is about 41-42 t/s, so the remaining decode headroom is small.

## Files

- `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh`: new decode and 2D mat-vec kernels.
- `ggml/src/ggml-cuda/mmvq.cu`: dispatch, launch geometry and the MoE case.
- `ggml/src/ggml-cuda/mmq.cu`, `mmq.cuh`: prefill un-gating and the PTQ1_0 tile support.
- `ggml/src/ggml-cuda/mmq-load-tiles.cuh`: the PTQ1_0 tile loader and decode.
- `ggml/src/ggml-cuda/mmq-config-rdna2.cuh`, `-rdna3.cuh`, `-rdna3-5.cuh`, `-rdna4.cuh`, `-cdna.cuh`: tile geometries.
- `ggml/src/ggml-cuda/vecdotq.cuh`, `convert.cu`, `quantize.cu`: vec dot, dequantize and the activation layout.
- `ggml/src/ggml-cuda/template-instances/`: instance and generator update.
