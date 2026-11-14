{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.

  Copyright (c) 2024-2026 Eudox67

  MIT License

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.

  This is an unofficial project and is not affiliated with or endorsed by AMD.
------------------------------------------------------------------------------}

{ GENERATED FILE -- do not edit.
  Source:     src/rocprim/rocprim_shim_abi.inc
  Generator:  ci/gen-shim-bindings.py
  Regenerate: python3 ci/gen-shim-bindings.py
}

{ Internal raw entry points into the rocPRIM shim.

  Nothing here opens the library or checks the GPU: rocprim_shim_loader.pas
  owns that policy and writes these slots only after the identity exports of
  the opened library agree with the device. The names are prefixed with __ so
  that the public API stays in rocprim_bindings.pas, which is what callers use.

  The ABI types are declared once, in rocprim_bindings (which owns the public
  surface); this unit imports them. FPC allows that direction because the
  dependency back from rocprim_bindings is implementation-only.
}

unit rocprim_shim_api;

{$mode objfpc}{$H+}

interface

uses
  ctypes, hip_types, hip_error, hip_runtime, rocprim_types,
  rocprim_bindings;

const
  { Symbols the shim must export: 5 identity + 144 rocPRIM entry points. }
  ROCPRIM_SHIM_SYMBOL_COUNT = 149;

  { Indexes of the identity exports. They are the first declarations in the
    specification because the loader must call them before it can trust the
    rest of the table (ci/DESIGN.md section 13). }
  ROCPRIM_SHIM_IDX_TARGET = 0;
  ROCPRIM_SHIM_IDX_WAVEFRONT_SIZE = 1;
  ROCPRIM_SHIM_IDX_ROCM_VERSION = 2;
  ROCPRIM_SHIM_IDX_WIDTHS_MASK = 3;
  ROCPRIM_SHIM_IDX_WAVEFRONT_COMPATIBLE = 4;

var
  { Raw entry points, assigned by rocprim_shim_loader. Never call these
    directly -- the wrappers in rocprim_bindings.pas load the library first. }

// --- shim identity and target checks (pure: no HIP calls, no GPU needed) ---
// These describe the library that is actually loaded: the GPU target it was
// built for, its compile-time wavefront width, the SDK it was built against, and
// the widths the hardware can run (bit 0 = 32, bit 1 = 64).
//
// A loader should compare pasrocm_shim_target against the device it detected and
// refuse a library built for a different GPU. pasrocm_wavefront_compatible is
// the same pure test the shim uses at init, exposed so a host-side check can
// reproduce the decision without a GPU.
  __pasrocm_shim_target: function: PAnsiChar; cdecl;

  __pasrocm_shim_wavefront_size: function: cuint; cdecl;

  __pasrocm_shim_rocm_version: function: PAnsiChar; cdecl;

  __pasrocm_shim_wavefront_widths_mask: function: cuint; cdecl;

  __pasrocm_wavefront_compatible: function(compiled, reported, allowed_mask: cuint): cint; cdecl;


// --- rocprim::reduce ---
  __rocprim_reduce_int: function(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_reduce_int128: function(
    d_out: PInt128;
    d_in: PInt128;
    num_items: csize_t;
    initial_value: TInt128;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_reduce_reverse_int: function(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::block_reduce ---
  __rocprim_block_reduce_int: function(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::block_scan ---
  __rocprim_block_scan_int: function(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::block_sort ---
  __rocprim_block_sort_int: function(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::block_histogram ---
  __rocprim_block_histogram_int: function(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    bins: cuint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::warp_reduce ---
  __rocprim_warp_reduce_int8: function(d_out: PShortInt; d_in: PShortInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_reduce_int16: function(d_out: PSmallInt; d_in: PSmallInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_reduce_int32: function(d_out: PCint; d_in: PCint; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_reduce_uint32: function(d_out: PCardinal; d_in: PCardinal; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_reduce_int64: function(d_out: PInt64; d_in: PInt64; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_reduce_uint64: function(d_out: PQWord; d_in: PQWord; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;


// --- rocprim::warp_scan ---
  __rocprim_warp_scan_int8: function(d_out: PShortInt; d_in: PShortInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_scan_int16: function(d_out: PSmallInt; d_in: PSmallInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_scan_int32: function(d_out: PCint; d_in: PCint; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_scan_uint32: function(d_out: PCardinal; d_in: PCardinal; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_scan_int64: function(d_out: PInt64; d_in: PInt64; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_scan_uint64: function(d_out: PQWord; d_in: PQWord; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;


// --- rocprim::warp_sort ---
  __rocprim_warp_sort_int8: function(d_out: PShortInt; d_in: PShortInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_sort_int16: function(d_out: PSmallInt; d_in: PSmallInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_sort_int32: function(d_out: PCint; d_in: PCint; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_sort_uint32: function(d_out: PCardinal; d_in: PCardinal; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_sort_int64: function(d_out: PInt64; d_in: PInt64; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_warp_sort_uint64: function(d_out: PQWord; d_in: PQWord; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;


// --- rocprim::histogram_range ---
  __rocprim_histogram_range_int: function(
    temporary_storage: Pointer;
    storage_size: psize_t;
    samples: PCint;
    size: cuint;
    histogram: PCint;
    levels: cuint;
    level_values: PCint;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl;


// --- rocprim::merge_sort_pairs ---
  __rocprim_merge_sort_pairs_int: function(
    temporary_storage: Pointer;
    storage_size: psize_t;
    keys_input: PCint;
    keys_output: PCint;
    values_input: PCint;
    values_output: PCint;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl;


// --- Gather / Scatter (float) ---
  __rocprim_gather_float: function(
    d_input: PSingle;
    d_index: PSingle;
    d_output: PSingle;
    num_items: csize_t;
    dim: cint;
    rank: cint;
    d_in_strides: PInt64;
    d_idx_strides: PInt64;
    d_out_strides: PInt64;
    d_out_shape: PInt64;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_clamp_float: function(
    d_data: PSingle;
    min_val: Single;
    max_val: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_bernoulli_float: function(
    d_data: PSingle;
    p: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- Elementwise math (float32, async) ---
  __rocprim_exp_float: function(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_log_float: function(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_sqrt_float: function(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_recip_float: function(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_div_float: function(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_scale_inplace_float: function(
    d_data: PSingle;
    scalar: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_relu_bwd_float: function(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- Elementwise math: binary broadcast add/sub/mul (same convention as div) ---
  __rocprim_add_float: function(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_sub_float: function(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_mul_float: function(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- scale (unary * scalar) ---
  __rocprim_scale_float: function(
    d_in: PSingle;
    scalar: Single;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- sigmoid / tanh fwd + bwd (bwd takes grad + input, recomputes activation) ---
  __rocprim_sigmoid_float: function(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_sigmoid_bwd_float: function(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_tanh_float: function(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_tanh_bwd_float: function(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- backward variants consuming the *saved forward output* ---
  __rocprim_sigmoid_bwd_out_float: function(
    d_grad: PSingle;
    d_y: PSingle;     // sigmoid forward output
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_tanh_bwd_out_float: function(
    d_grad: PSingle;
    d_t: PSingle;     // tanh forward output
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- GELU / SiLU fwd + bwd (bwd takes grad + input) ---
  __rocprim_gelu_float: function(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_gelu_bwd_float: function(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_silu_float: function(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_silu_bwd_float: function(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- native FP16 / BF16 activation variants (device pointers are raw
//     16-bit element buffers; kernels upconvert to FP32 in-register) ---
  __rocprim_gelu_half: function(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_gelu_bf16: function(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_gelu_bwd_half: function(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_gelu_bwd_bf16: function(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_silu_half: function(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_silu_bf16: function(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_silu_bwd_half: function(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_silu_bwd_bf16: function(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- broadcast binary ops: out[i] = op(a[i], b[i mod blen]).
//     op: 0=add, 1=sub, 2=mul, 3=div. Correct for contiguous row-major A
//     and contiguous trailing-broadcast B (e.g. [1,16,192] + [192]). ---
  __rocprim_bcast_op_float: function(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    alen: csize_t;
    blen: csize_t;
    op: cint32;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_bcast_op_half: function(
    d_a: Pointer;
    d_b: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    alen: csize_t;
    blen: csize_t;
    op: cint32;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_bcast_op_bf16: function(
    d_a: Pointer;
    d_b: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    alen: csize_t;
    blen: csize_t;
    op: cint32;
    stream: hipStream_t
): hipError_t; cdecl;


// --- general strided gather: materialize any permuted/sliced view.
//     h_shape/h_strides are host arrays of dimcount Int64s (elements). ---
  __rocprim_gather_strided_float: function(
    d_src: PSingle;
    d_dst: PSingle;
    h_shape: PInt64;
    h_strides: PInt64;
    dimcount: cint32;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


{ --- native 16-bit same-shape binary ops (FP32 compute in-register) --- }
  __rocprim_add_half: function(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_add_bf16: function(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_sub_half: function(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_sub_bf16: function(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_mul_half: function(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;

  __rocprim_mul_bf16: function(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;


{ Two-pass global sum: out[0] = sum(in[0..n-1]). }
  __rocprim_sum_float: function(
    in_buf: PSingle;
    out_buf: PSingle;
    n: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- fill ---
  __rocprim_fill_float: function(
    d_out: PSingle;
    value: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- row-wise softmax / logsoftmax fwd + bwd (contiguous last dim) ---
  __rocprim_softmax_float: function(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_softmax_bwd_float: function(
    d_grad: PSingle;
    d_y: PSingle;   // softmax output
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_logsoftmax_float: function(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_logsoftmax_bwd_float: function(
    d_grad: PSingle;
    d_logy: PSingle;  // logsoftmax output
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- cumulative sum along the last (contiguous) dimension ---
  __rocprim_cumsum_float: function(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- strided variants: op along a non-last dim of a contiguous tensor
//     viewed as [outer, len, inner]; one block per slice ---
  __rocprim_softmax_strided_float: function(
    d_in: PSingle;
    d_out: PSingle;
    slices: csize_t;
    len: csize_t;
    inner: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_cumsum_strided_float: function(
    d_in: PSingle;
    d_out: PSingle;
    slices: csize_t;
    len: csize_t;
    inner: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- tiled 2D transpose (out[c][r] = in[r][c], coalesced both sides) ---
  __rocprim_transpose_float: function(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- per-row top-k (largest, each row sorted descending); k <= 1024 ---
  __rocprim_topk_float: function(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    k: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- row reductions along the last (contiguous) dimension ---
  __rocprim_rowsum_float: function(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_rowmax_float: function(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_rowmin_float: function(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- repeat-interleave along one dim of a contiguous tensor ---
  __rocprim_repeat_interleave_float: function(
    d_in: PSingle;
    d_out: PSingle;
    outer: csize_t;
    len: csize_t;
    inner: csize_t;
    n: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- dtype converters (typed-void for Pascal side convenience) ---
  __rocprim_convert_f16_to_f32: function(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_convert_f32_to_f16: function(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_convert_bf16_to_f32: function(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_convert_f32_to_bf16: function(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_scatter_float: function(
    d_output: PSingle;
    d_index: PSingle;
    d_src: PSingle;
    num_items: csize_t;
    dim: cint;
    rank: cint;
    d_out_strides: PInt64;
    d_idx_strides: PInt64;
    d_src_strides: PInt64;
    d_src_shape: PInt64;
    atomic_add: cbool;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::inclusive_scan ---
  __rocprim_inclusive_scan_int: function(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::deterministic_inclusive_scan ---
  __rocprim_deterministic_inclusive_scan_int: function(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::exclusive_scan ---
  __rocprim_exclusive_scan_int: function(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::deterministic_exclusive_scan ---
  __rocprim_deterministic_exclusive_scan_int: function(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::radix_sort_keys (in-place) ---
  __rocprim_radix_sort_keys_int: function(
    d_keys: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::transform ---
  __rocprim_transform_int: function(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    multiplier: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::partition ---
  __rocprim_partition_int: function(
    d_in: PCint;
    d_out: PCint;
    d_selected_count: PCsize_t;
    num_items: csize_t;
    pivot: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::select ---
  __rocprim_select_int: function(
    d_in: PCint;
    d_out: PCint;
    d_selected_count: PCsize_t;
    num_items: csize_t;
    pivot: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::reduce_by_key ---
  __rocprim_reduce_by_key_int: function(
    d_keys_in: PCint;
    d_values_in: PCint;
    num_items: csize_t;
    d_unique_keys_out: PCint;
    d_aggregates_out: PCint;
    d_unique_count_out: PCsize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::inclusive_scan_by_key ---
  __rocprim_inclusive_scan_by_key_int: function(
    d_keys_in: PCint;
    d_values_in: PCint;
    d_values_out: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::exclusive_scan_by_key ---
  __rocprim_exclusive_scan_by_key_int: function(
    d_keys_in: PCint;
    d_values_in: PCint;
    d_values_out: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::histogram_even ---
  __rocprim_histogram_even_int: function(
    d_samples: PCint;
    num_samples: csize_t;
    d_histogram: PCint;
    levels: cuint;
    lower_level: cint;
    upper_level: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::merge ---
  __rocprim_merge_int: function(
    d_input1: PCint;
    d_input2: PCint;
    d_output: PCint;
    input1_size: csize_t;
    input2_size: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::radix_sort_keys (input/output iterators) ---
  __rocprim_radix_sort_keys_int_io: function(
    d_keys_in: PCint;
    d_keys_out: PCint;
    num_items: csize_t;
    begin_bit: cuint;
    end_bit: cuint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::unique ---
  __rocprim_unique_int: function(
    d_input: PCint;
    d_output: PCint;
    d_unique_count_out: PCsize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::copy ---
  __rocprim_copy_int: function(
    d_input: PCint;
    d_output: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::fill ---
  __rocprim_fill_int: function(
    d_output: PCint;
    value: cint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::adjacent_difference ---
  __rocprim_adjacent_difference_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    d_input: PCint;
    d_output: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::adjacent_find ---
  __rocprim_adjacent_find_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    d_input: PCint;
    d_output: PCsize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::merge_inplace ---
  __rocprim_merge_inplace_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    data: PCint;
    left_size: csize_t;
    right_size: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::binary_search ---
  __rocprim_binary_search_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    haystack: PCint;
    needles: PCint;
    output: PCsize_t;
    haystack_size: csize_t;
    needles_size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl;


  __rocprim_find_end_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    input: PCint;
    keys: PCint;
    output: PCsize_t;
    size: csize_t;
    keys_size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl;


  __rocprim_find_first_of_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    input: PCint;
    keys: PCint;
    output: PCsize_t;
    size: csize_t;
    keys_size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl;


  __rocprim_merge_sort_keys_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PCint;
    keys_output: PCint;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl;


  __rocprim_nth_element_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PCint;
    keys_output: PCint;
    nth: csize_t;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl;


  __rocprim_partial_sort_copy_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PInteger;
    keys_output: PInteger;
    middle: csize_t;  // Index (prefix length = middle +1)
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_radix_sort_keys_int_asc: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PCint;
    keys_output: PCint;
    size: csize_t;
    begin_bit: cuint;
    end_bit: cuint;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl;


  __rocprim_radix_sort_pairs_int: function(
	temporary_storage: Pointer;
	storage_size: Pcsize_t;
	keys_input: Pointer;
	keys_output: Pointer;
	values_input: Pointer;
	values_output: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


  __rocprim_partial_sort_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys: PInteger;
    middle: csize_t;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_run_length_encode_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    input: PInteger;
    unique_output: PInteger;
    counts_output: PInteger;
    runs_count_output: PInteger;
    size: cuint;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_run_length_encode_non_trivial_runs_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    input: PInteger;
    offsets_output: PInteger;
    counts_output: PInteger;
    runs_count_output: PInteger;
    size: cuint;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_search_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    haystack: PInteger;
    needle: PInteger;
    output: PInteger;
    haystack_size: csize_t;
    needle_size: csize_t;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_search_n_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    haystack: PInteger;
    output: PSize_t;
    haystack_size: csize_t;
    count: csize_t;
    value: cint;
    op: rocprim_binary_op;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_segmented_reduce_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    input: PInteger;
    output: PInteger;
    segments: cuint;
    begin_offsets: PInteger;
    end_offsets: PInteger;
    initial_value: cint;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_segmented_inclusive_scan_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    input: PInteger;
    output: PInteger;
    segments: cuint;
    begin_offsets: PInteger;
    end_offsets: PInteger;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_segmented_exclusive_scan_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    input: PInteger;
    output: PInteger;
    segments: cuint;
    begin_offsets: PInteger;
    end_offsets: PInteger;
    initial_value: cint;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_segmented_radix_sort_keys_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PInteger;
    keys_output: PInteger;
    size: cuint;
    segments: cuint;
    begin_offsets: PInteger;
    end_offsets: PInteger;
    begin_bit: cuint;
    end_bit: cuint;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_segmented_radix_sort_pairs_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PInteger;
    keys_output: PInteger;
    values_input: PInteger;
    values_output: PInteger;
    size: cuint;
    segments: cuint;
    begin_offsets: PInteger;
    end_offsets: PInteger;
    begin_bit: cuint;
    end_bit: cuint;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_radix_sort_keys_desc_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PInteger;
    keys_output: PInteger;
    size: cuint;
    begin_bit: cuint;
    end_bit: cuint;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_radix_sort_pairs_desc_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PInteger;
    keys_output: PInteger;
    values_input: PInteger;
    values_output: PInteger;
    size: cuint;
    begin_bit: cuint;
    end_bit: cuint;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_segmented_radix_sort_keys_desc_int: function(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PInteger;
    keys_output: PInteger;
    size: cuint;
    segments: cuint;
    begin_offsets: PInteger;
    end_offsets: PInteger;
    begin_bit: cuint;
    end_bit: cuint;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


  __rocprim_numeric_limits_int_min: function: cint; cdecl;

  __rocprim_numeric_limits_int_max: function: cint; cdecl;

  __rocprim_numeric_limits_float_epsilon: function: cfloat; cdecl;

  __rocprim_numeric_limits_float_infinity: function: cfloat; cdecl;

  __rocprim_numeric_limits_float_nan: function: cfloat; cdecl;


{ Row-wise argmax over the last (contiguous) dimension: one block per row.
  out[row] = index (as Single) of the max element. Ties -> lowest index. }
  __rocprim_argmax_float: function(
    in_buf: PSingle;
    out_buf: PSingle;
    cols: Int64;
    rows: Int64;
    stream: hipStream_t
): hipError_t; cdecl;


// --- comparison ops (float mask 1.0/0.0, broadcast-modulo) ---
  __rocprim_gt_float: function(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

  __rocprim_lt_float: function(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

  __rocprim_eq_float: function(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

  __rocprim_ge_float: function(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

  __rocprim_le_float: function(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

  __rocprim_ne_float: function(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;


// --- Where(cond, A, B) and MaskedFill ---
  __rocprim_where_float: function(
    d_cond: PSingle; d_a: PSingle; d_b: PSingle; d_out: PSingle;
    c_n: csize_t; a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

  __rocprim_maskedfill_float: function(
    d_data: PSingle; d_mask: PSingle; value: Single;
    mask_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

{ Table view used by the loader: one name and one slot per symbol. }
function rocprim_shim_symbol_name(AIndex: Integer): PAnsiChar;
function rocprim_shim_symbol_slot(AIndex: Integer): PPointer;

implementation

const
  SHIM_SYMBOL_NAMES: array[0..ROCPRIM_SHIM_SYMBOL_COUNT - 1] of PAnsiChar = (
    'pasrocm_shim_target',
    'pasrocm_shim_wavefront_size',
    'pasrocm_shim_rocm_version',
    'pasrocm_shim_wavefront_widths_mask',
    'pasrocm_wavefront_compatible',
    'rocprim_reduce_int',
    'rocprim_reduce_int128',
    'rocprim_reduce_reverse_int',
    'rocprim_block_reduce_int',
    'rocprim_block_scan_int',
    'rocprim_block_sort_int',
    'rocprim_block_histogram_int',
    'rocprim_warp_reduce_int8',
    'rocprim_warp_reduce_int16',
    'rocprim_warp_reduce_int32',
    'rocprim_warp_reduce_uint32',
    'rocprim_warp_reduce_int64',
    'rocprim_warp_reduce_uint64',
    'rocprim_warp_scan_int8',
    'rocprim_warp_scan_int16',
    'rocprim_warp_scan_int32',
    'rocprim_warp_scan_uint32',
    'rocprim_warp_scan_int64',
    'rocprim_warp_scan_uint64',
    'rocprim_warp_sort_int8',
    'rocprim_warp_sort_int16',
    'rocprim_warp_sort_int32',
    'rocprim_warp_sort_uint32',
    'rocprim_warp_sort_int64',
    'rocprim_warp_sort_uint64',
    'rocprim_histogram_range_int',
    'rocprim_merge_sort_pairs_int',
    'rocprim_gather_float',
    'rocprim_clamp_float',
    'rocprim_bernoulli_float',
    'rocprim_exp_float',
    'rocprim_log_float',
    'rocprim_sqrt_float',
    'rocprim_recip_float',
    'rocprim_div_float',
    'rocprim_scale_inplace_float',
    'rocprim_relu_bwd_float',
    'rocprim_add_float',
    'rocprim_sub_float',
    'rocprim_mul_float',
    'rocprim_scale_float',
    'rocprim_sigmoid_float',
    'rocprim_sigmoid_bwd_float',
    'rocprim_tanh_float',
    'rocprim_tanh_bwd_float',
    'rocprim_sigmoid_bwd_out_float',
    'rocprim_tanh_bwd_out_float',
    'rocprim_gelu_float',
    'rocprim_gelu_bwd_float',
    'rocprim_silu_float',
    'rocprim_silu_bwd_float',
    'rocprim_gelu_half',
    'rocprim_gelu_bf16',
    'rocprim_gelu_bwd_half',
    'rocprim_gelu_bwd_bf16',
    'rocprim_silu_half',
    'rocprim_silu_bf16',
    'rocprim_silu_bwd_half',
    'rocprim_silu_bwd_bf16',
    'rocprim_bcast_op_float',
    'rocprim_bcast_op_half',
    'rocprim_bcast_op_bf16',
    'rocprim_gather_strided_float',
    'rocprim_add_half',
    'rocprim_add_bf16',
    'rocprim_sub_half',
    'rocprim_sub_bf16',
    'rocprim_mul_half',
    'rocprim_mul_bf16',
    'rocprim_sum_float',
    'rocprim_fill_float',
    'rocprim_softmax_float',
    'rocprim_softmax_bwd_float',
    'rocprim_logsoftmax_float',
    'rocprim_logsoftmax_bwd_float',
    'rocprim_cumsum_float',
    'rocprim_softmax_strided_float',
    'rocprim_cumsum_strided_float',
    'rocprim_transpose_float',
    'rocprim_topk_float',
    'rocprim_rowsum_float',
    'rocprim_rowmax_float',
    'rocprim_rowmin_float',
    'rocprim_repeat_interleave_float',
    'rocprim_convert_f16_to_f32',
    'rocprim_convert_f32_to_f16',
    'rocprim_convert_bf16_to_f32',
    'rocprim_convert_f32_to_bf16',
    'rocprim_scatter_float',
    'rocprim_inclusive_scan_int',
    'rocprim_deterministic_inclusive_scan_int',
    'rocprim_exclusive_scan_int',
    'rocprim_deterministic_exclusive_scan_int',
    'rocprim_radix_sort_keys_int',
    'rocprim_transform_int',
    'rocprim_partition_int',
    'rocprim_select_int',
    'rocprim_reduce_by_key_int',
    'rocprim_inclusive_scan_by_key_int',
    'rocprim_exclusive_scan_by_key_int',
    'rocprim_histogram_even_int',
    'rocprim_merge_int',
    'rocprim_radix_sort_keys_int_io',
    'rocprim_unique_int',
    'rocprim_copy_int',
    'rocprim_fill_int',
    'rocprim_adjacent_difference_int',
    'rocprim_adjacent_find_int',
    'rocprim_merge_inplace_int',
    'rocprim_binary_search_int',
    'rocprim_find_end_int',
    'rocprim_find_first_of_int',
    'rocprim_merge_sort_keys_int',
    'rocprim_nth_element_int',
    'rocprim_partial_sort_copy_int',
    'rocprim_radix_sort_keys_int_asc',
    'rocprim_radix_sort_pairs_int',
    'rocprim_partial_sort_int',
    'rocprim_run_length_encode_int',
    'rocprim_run_length_encode_non_trivial_runs_int',
    'rocprim_search_int',
    'rocprim_search_n_int',
    'rocprim_segmented_reduce_int',
    'rocprim_segmented_inclusive_scan_int',
    'rocprim_segmented_exclusive_scan_int',
    'rocprim_segmented_radix_sort_keys_int',
    'rocprim_segmented_radix_sort_pairs_int',
    'rocprim_radix_sort_keys_desc_int',
    'rocprim_radix_sort_pairs_desc_int',
    'rocprim_segmented_radix_sort_keys_desc_int',
    'rocprim_numeric_limits_int_min',
    'rocprim_numeric_limits_int_max',
    'rocprim_numeric_limits_float_epsilon',
    'rocprim_numeric_limits_float_infinity',
    'rocprim_numeric_limits_float_nan',
    'rocprim_argmax_float',
    'rocprim_gt_float',
    'rocprim_lt_float',
    'rocprim_eq_float',
    'rocprim_ge_float',
    'rocprim_le_float',
    'rocprim_ne_float',
    'rocprim_where_float',
    'rocprim_maskedfill_float'
  );

var
  SHIM_SYMBOL_SLOTS: array[0..ROCPRIM_SHIM_SYMBOL_COUNT - 1] of PPointer;

function rocprim_shim_symbol_name(AIndex: Integer): PAnsiChar;
begin
  if (AIndex < 0) or (AIndex >= ROCPRIM_SHIM_SYMBOL_COUNT) then
    Exit(nil);
  Result := SHIM_SYMBOL_NAMES[AIndex];
end;

function rocprim_shim_symbol_slot(AIndex: Integer): PPointer;
begin
  if (AIndex < 0) or (AIndex >= ROCPRIM_SHIM_SYMBOL_COUNT) then
    Exit(nil);
  Result := SHIM_SYMBOL_SLOTS[AIndex];
end;

initialization
  SHIM_SYMBOL_SLOTS[0] := PPointer(@__pasrocm_shim_target);
  SHIM_SYMBOL_SLOTS[1] := PPointer(@__pasrocm_shim_wavefront_size);
  SHIM_SYMBOL_SLOTS[2] := PPointer(@__pasrocm_shim_rocm_version);
  SHIM_SYMBOL_SLOTS[3] := PPointer(@__pasrocm_shim_wavefront_widths_mask);
  SHIM_SYMBOL_SLOTS[4] := PPointer(@__pasrocm_wavefront_compatible);
  SHIM_SYMBOL_SLOTS[5] := PPointer(@__rocprim_reduce_int);
  SHIM_SYMBOL_SLOTS[6] := PPointer(@__rocprim_reduce_int128);
  SHIM_SYMBOL_SLOTS[7] := PPointer(@__rocprim_reduce_reverse_int);
  SHIM_SYMBOL_SLOTS[8] := PPointer(@__rocprim_block_reduce_int);
  SHIM_SYMBOL_SLOTS[9] := PPointer(@__rocprim_block_scan_int);
  SHIM_SYMBOL_SLOTS[10] := PPointer(@__rocprim_block_sort_int);
  SHIM_SYMBOL_SLOTS[11] := PPointer(@__rocprim_block_histogram_int);
  SHIM_SYMBOL_SLOTS[12] := PPointer(@__rocprim_warp_reduce_int8);
  SHIM_SYMBOL_SLOTS[13] := PPointer(@__rocprim_warp_reduce_int16);
  SHIM_SYMBOL_SLOTS[14] := PPointer(@__rocprim_warp_reduce_int32);
  SHIM_SYMBOL_SLOTS[15] := PPointer(@__rocprim_warp_reduce_uint32);
  SHIM_SYMBOL_SLOTS[16] := PPointer(@__rocprim_warp_reduce_int64);
  SHIM_SYMBOL_SLOTS[17] := PPointer(@__rocprim_warp_reduce_uint64);
  SHIM_SYMBOL_SLOTS[18] := PPointer(@__rocprim_warp_scan_int8);
  SHIM_SYMBOL_SLOTS[19] := PPointer(@__rocprim_warp_scan_int16);
  SHIM_SYMBOL_SLOTS[20] := PPointer(@__rocprim_warp_scan_int32);
  SHIM_SYMBOL_SLOTS[21] := PPointer(@__rocprim_warp_scan_uint32);
  SHIM_SYMBOL_SLOTS[22] := PPointer(@__rocprim_warp_scan_int64);
  SHIM_SYMBOL_SLOTS[23] := PPointer(@__rocprim_warp_scan_uint64);
  SHIM_SYMBOL_SLOTS[24] := PPointer(@__rocprim_warp_sort_int8);
  SHIM_SYMBOL_SLOTS[25] := PPointer(@__rocprim_warp_sort_int16);
  SHIM_SYMBOL_SLOTS[26] := PPointer(@__rocprim_warp_sort_int32);
  SHIM_SYMBOL_SLOTS[27] := PPointer(@__rocprim_warp_sort_uint32);
  SHIM_SYMBOL_SLOTS[28] := PPointer(@__rocprim_warp_sort_int64);
  SHIM_SYMBOL_SLOTS[29] := PPointer(@__rocprim_warp_sort_uint64);
  SHIM_SYMBOL_SLOTS[30] := PPointer(@__rocprim_histogram_range_int);
  SHIM_SYMBOL_SLOTS[31] := PPointer(@__rocprim_merge_sort_pairs_int);
  SHIM_SYMBOL_SLOTS[32] := PPointer(@__rocprim_gather_float);
  SHIM_SYMBOL_SLOTS[33] := PPointer(@__rocprim_clamp_float);
  SHIM_SYMBOL_SLOTS[34] := PPointer(@__rocprim_bernoulli_float);
  SHIM_SYMBOL_SLOTS[35] := PPointer(@__rocprim_exp_float);
  SHIM_SYMBOL_SLOTS[36] := PPointer(@__rocprim_log_float);
  SHIM_SYMBOL_SLOTS[37] := PPointer(@__rocprim_sqrt_float);
  SHIM_SYMBOL_SLOTS[38] := PPointer(@__rocprim_recip_float);
  SHIM_SYMBOL_SLOTS[39] := PPointer(@__rocprim_div_float);
  SHIM_SYMBOL_SLOTS[40] := PPointer(@__rocprim_scale_inplace_float);
  SHIM_SYMBOL_SLOTS[41] := PPointer(@__rocprim_relu_bwd_float);
  SHIM_SYMBOL_SLOTS[42] := PPointer(@__rocprim_add_float);
  SHIM_SYMBOL_SLOTS[43] := PPointer(@__rocprim_sub_float);
  SHIM_SYMBOL_SLOTS[44] := PPointer(@__rocprim_mul_float);
  SHIM_SYMBOL_SLOTS[45] := PPointer(@__rocprim_scale_float);
  SHIM_SYMBOL_SLOTS[46] := PPointer(@__rocprim_sigmoid_float);
  SHIM_SYMBOL_SLOTS[47] := PPointer(@__rocprim_sigmoid_bwd_float);
  SHIM_SYMBOL_SLOTS[48] := PPointer(@__rocprim_tanh_float);
  SHIM_SYMBOL_SLOTS[49] := PPointer(@__rocprim_tanh_bwd_float);
  SHIM_SYMBOL_SLOTS[50] := PPointer(@__rocprim_sigmoid_bwd_out_float);
  SHIM_SYMBOL_SLOTS[51] := PPointer(@__rocprim_tanh_bwd_out_float);
  SHIM_SYMBOL_SLOTS[52] := PPointer(@__rocprim_gelu_float);
  SHIM_SYMBOL_SLOTS[53] := PPointer(@__rocprim_gelu_bwd_float);
  SHIM_SYMBOL_SLOTS[54] := PPointer(@__rocprim_silu_float);
  SHIM_SYMBOL_SLOTS[55] := PPointer(@__rocprim_silu_bwd_float);
  SHIM_SYMBOL_SLOTS[56] := PPointer(@__rocprim_gelu_half);
  SHIM_SYMBOL_SLOTS[57] := PPointer(@__rocprim_gelu_bf16);
  SHIM_SYMBOL_SLOTS[58] := PPointer(@__rocprim_gelu_bwd_half);
  SHIM_SYMBOL_SLOTS[59] := PPointer(@__rocprim_gelu_bwd_bf16);
  SHIM_SYMBOL_SLOTS[60] := PPointer(@__rocprim_silu_half);
  SHIM_SYMBOL_SLOTS[61] := PPointer(@__rocprim_silu_bf16);
  SHIM_SYMBOL_SLOTS[62] := PPointer(@__rocprim_silu_bwd_half);
  SHIM_SYMBOL_SLOTS[63] := PPointer(@__rocprim_silu_bwd_bf16);
  SHIM_SYMBOL_SLOTS[64] := PPointer(@__rocprim_bcast_op_float);
  SHIM_SYMBOL_SLOTS[65] := PPointer(@__rocprim_bcast_op_half);
  SHIM_SYMBOL_SLOTS[66] := PPointer(@__rocprim_bcast_op_bf16);
  SHIM_SYMBOL_SLOTS[67] := PPointer(@__rocprim_gather_strided_float);
  SHIM_SYMBOL_SLOTS[68] := PPointer(@__rocprim_add_half);
  SHIM_SYMBOL_SLOTS[69] := PPointer(@__rocprim_add_bf16);
  SHIM_SYMBOL_SLOTS[70] := PPointer(@__rocprim_sub_half);
  SHIM_SYMBOL_SLOTS[71] := PPointer(@__rocprim_sub_bf16);
  SHIM_SYMBOL_SLOTS[72] := PPointer(@__rocprim_mul_half);
  SHIM_SYMBOL_SLOTS[73] := PPointer(@__rocprim_mul_bf16);
  SHIM_SYMBOL_SLOTS[74] := PPointer(@__rocprim_sum_float);
  SHIM_SYMBOL_SLOTS[75] := PPointer(@__rocprim_fill_float);
  SHIM_SYMBOL_SLOTS[76] := PPointer(@__rocprim_softmax_float);
  SHIM_SYMBOL_SLOTS[77] := PPointer(@__rocprim_softmax_bwd_float);
  SHIM_SYMBOL_SLOTS[78] := PPointer(@__rocprim_logsoftmax_float);
  SHIM_SYMBOL_SLOTS[79] := PPointer(@__rocprim_logsoftmax_bwd_float);
  SHIM_SYMBOL_SLOTS[80] := PPointer(@__rocprim_cumsum_float);
  SHIM_SYMBOL_SLOTS[81] := PPointer(@__rocprim_softmax_strided_float);
  SHIM_SYMBOL_SLOTS[82] := PPointer(@__rocprim_cumsum_strided_float);
  SHIM_SYMBOL_SLOTS[83] := PPointer(@__rocprim_transpose_float);
  SHIM_SYMBOL_SLOTS[84] := PPointer(@__rocprim_topk_float);
  SHIM_SYMBOL_SLOTS[85] := PPointer(@__rocprim_rowsum_float);
  SHIM_SYMBOL_SLOTS[86] := PPointer(@__rocprim_rowmax_float);
  SHIM_SYMBOL_SLOTS[87] := PPointer(@__rocprim_rowmin_float);
  SHIM_SYMBOL_SLOTS[88] := PPointer(@__rocprim_repeat_interleave_float);
  SHIM_SYMBOL_SLOTS[89] := PPointer(@__rocprim_convert_f16_to_f32);
  SHIM_SYMBOL_SLOTS[90] := PPointer(@__rocprim_convert_f32_to_f16);
  SHIM_SYMBOL_SLOTS[91] := PPointer(@__rocprim_convert_bf16_to_f32);
  SHIM_SYMBOL_SLOTS[92] := PPointer(@__rocprim_convert_f32_to_bf16);
  SHIM_SYMBOL_SLOTS[93] := PPointer(@__rocprim_scatter_float);
  SHIM_SYMBOL_SLOTS[94] := PPointer(@__rocprim_inclusive_scan_int);
  SHIM_SYMBOL_SLOTS[95] := PPointer(@__rocprim_deterministic_inclusive_scan_int);
  SHIM_SYMBOL_SLOTS[96] := PPointer(@__rocprim_exclusive_scan_int);
  SHIM_SYMBOL_SLOTS[97] := PPointer(@__rocprim_deterministic_exclusive_scan_int);
  SHIM_SYMBOL_SLOTS[98] := PPointer(@__rocprim_radix_sort_keys_int);
  SHIM_SYMBOL_SLOTS[99] := PPointer(@__rocprim_transform_int);
  SHIM_SYMBOL_SLOTS[100] := PPointer(@__rocprim_partition_int);
  SHIM_SYMBOL_SLOTS[101] := PPointer(@__rocprim_select_int);
  SHIM_SYMBOL_SLOTS[102] := PPointer(@__rocprim_reduce_by_key_int);
  SHIM_SYMBOL_SLOTS[103] := PPointer(@__rocprim_inclusive_scan_by_key_int);
  SHIM_SYMBOL_SLOTS[104] := PPointer(@__rocprim_exclusive_scan_by_key_int);
  SHIM_SYMBOL_SLOTS[105] := PPointer(@__rocprim_histogram_even_int);
  SHIM_SYMBOL_SLOTS[106] := PPointer(@__rocprim_merge_int);
  SHIM_SYMBOL_SLOTS[107] := PPointer(@__rocprim_radix_sort_keys_int_io);
  SHIM_SYMBOL_SLOTS[108] := PPointer(@__rocprim_unique_int);
  SHIM_SYMBOL_SLOTS[109] := PPointer(@__rocprim_copy_int);
  SHIM_SYMBOL_SLOTS[110] := PPointer(@__rocprim_fill_int);
  SHIM_SYMBOL_SLOTS[111] := PPointer(@__rocprim_adjacent_difference_int);
  SHIM_SYMBOL_SLOTS[112] := PPointer(@__rocprim_adjacent_find_int);
  SHIM_SYMBOL_SLOTS[113] := PPointer(@__rocprim_merge_inplace_int);
  SHIM_SYMBOL_SLOTS[114] := PPointer(@__rocprim_binary_search_int);
  SHIM_SYMBOL_SLOTS[115] := PPointer(@__rocprim_find_end_int);
  SHIM_SYMBOL_SLOTS[116] := PPointer(@__rocprim_find_first_of_int);
  SHIM_SYMBOL_SLOTS[117] := PPointer(@__rocprim_merge_sort_keys_int);
  SHIM_SYMBOL_SLOTS[118] := PPointer(@__rocprim_nth_element_int);
  SHIM_SYMBOL_SLOTS[119] := PPointer(@__rocprim_partial_sort_copy_int);
  SHIM_SYMBOL_SLOTS[120] := PPointer(@__rocprim_radix_sort_keys_int_asc);
  SHIM_SYMBOL_SLOTS[121] := PPointer(@__rocprim_radix_sort_pairs_int);
  SHIM_SYMBOL_SLOTS[122] := PPointer(@__rocprim_partial_sort_int);
  SHIM_SYMBOL_SLOTS[123] := PPointer(@__rocprim_run_length_encode_int);
  SHIM_SYMBOL_SLOTS[124] := PPointer(@__rocprim_run_length_encode_non_trivial_runs_int);
  SHIM_SYMBOL_SLOTS[125] := PPointer(@__rocprim_search_int);
  SHIM_SYMBOL_SLOTS[126] := PPointer(@__rocprim_search_n_int);
  SHIM_SYMBOL_SLOTS[127] := PPointer(@__rocprim_segmented_reduce_int);
  SHIM_SYMBOL_SLOTS[128] := PPointer(@__rocprim_segmented_inclusive_scan_int);
  SHIM_SYMBOL_SLOTS[129] := PPointer(@__rocprim_segmented_exclusive_scan_int);
  SHIM_SYMBOL_SLOTS[130] := PPointer(@__rocprim_segmented_radix_sort_keys_int);
  SHIM_SYMBOL_SLOTS[131] := PPointer(@__rocprim_segmented_radix_sort_pairs_int);
  SHIM_SYMBOL_SLOTS[132] := PPointer(@__rocprim_radix_sort_keys_desc_int);
  SHIM_SYMBOL_SLOTS[133] := PPointer(@__rocprim_radix_sort_pairs_desc_int);
  SHIM_SYMBOL_SLOTS[134] := PPointer(@__rocprim_segmented_radix_sort_keys_desc_int);
  SHIM_SYMBOL_SLOTS[135] := PPointer(@__rocprim_numeric_limits_int_min);
  SHIM_SYMBOL_SLOTS[136] := PPointer(@__rocprim_numeric_limits_int_max);
  SHIM_SYMBOL_SLOTS[137] := PPointer(@__rocprim_numeric_limits_float_epsilon);
  SHIM_SYMBOL_SLOTS[138] := PPointer(@__rocprim_numeric_limits_float_infinity);
  SHIM_SYMBOL_SLOTS[139] := PPointer(@__rocprim_numeric_limits_float_nan);
  SHIM_SYMBOL_SLOTS[140] := PPointer(@__rocprim_argmax_float);
  SHIM_SYMBOL_SLOTS[141] := PPointer(@__rocprim_gt_float);
  SHIM_SYMBOL_SLOTS[142] := PPointer(@__rocprim_lt_float);
  SHIM_SYMBOL_SLOTS[143] := PPointer(@__rocprim_eq_float);
  SHIM_SYMBOL_SLOTS[144] := PPointer(@__rocprim_ge_float);
  SHIM_SYMBOL_SLOTS[145] := PPointer(@__rocprim_le_float);
  SHIM_SYMBOL_SLOTS[146] := PPointer(@__rocprim_ne_float);
  SHIM_SYMBOL_SLOTS[147] := PPointer(@__rocprim_where_float);
  SHIM_SYMBOL_SLOTS[148] := PPointer(@__rocprim_maskedfill_float);

end.
