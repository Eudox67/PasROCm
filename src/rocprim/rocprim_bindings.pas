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

{ Low-level extern declarations into librocprim_shim. }
{ Low-level extern declarations into librocprim_shim. }
unit rocprim_bindings;

interface

uses
  ctypes, hip_types, hip_error, hip_runtime, rocprim_types; // Removed lgint128, added rocprim_types

type
  PInt128 = ^int128_t; // Using rocprim_types.int128_t
  TInt128 = int128_t;

const
  // This must point to the compiled shim library.
  {$IFDEF MSWINDOWS}
    librocprimshim = 'rocprim_shim.dll';
  {$ELSE}
    librocprimshim = 'rocprim_shim';
  {$ENDIF}

// --- shim identity and target checks (pure: no HIP calls, no GPU needed) ---
// These describe the library that is actually loaded: the GPU target it was
// built for, its compile-time wavefront width, the SDK it was built against, and
// the widths the hardware can run (bit 0 = 32, bit 1 = 64).
//
// A loader should compare pasrocm_shim_target against the device it detected and
// refuse a library built for a different GPU. pasrocm_wavefront_compatible is
// the same pure test the shim uses at init, exposed so a host-side check can
// reproduce the decision without a GPU.
function pasrocm_shim_target: PAnsiChar; cdecl; external librocprimshim;
function pasrocm_shim_wavefront_size: cuint; cdecl; external librocprimshim;
function pasrocm_shim_rocm_version: PAnsiChar; cdecl; external librocprimshim;
function pasrocm_shim_wavefront_widths_mask: cuint; cdecl; external librocprimshim;
function pasrocm_wavefront_compatible(compiled, reported, allowed_mask: cuint): cint;
    cdecl; external librocprimshim;

// --- rocprim::reduce ---
function rocprim_reduce_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_reduce_int128(
    d_out: PInt128;
    d_in: PInt128;
    num_items: csize_t;
    initial_value: TInt128;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_reduce_reverse_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::block_reduce ---
function rocprim_block_reduce_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::block_scan ---
function rocprim_block_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::block_sort ---
function rocprim_block_sort_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::block_histogram ---
function rocprim_block_histogram_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    bins: cuint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::warp_reduce ---
function rocprim_warp_reduce_int8(d_out: PShortInt; d_in: PShortInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_reduce_int16(d_out: PSmallInt; d_in: PSmallInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_reduce_int32(d_out: PCint; d_in: PCint; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_reduce_uint32(d_out: PCardinal; d_in: PCardinal; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_reduce_int64(d_out: PInt64; d_in: PInt64; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_reduce_uint64(d_out: PQWord; d_in: PQWord; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;

// --- rocprim::warp_scan ---
function rocprim_warp_scan_int8(d_out: PShortInt; d_in: PShortInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_scan_int16(d_out: PSmallInt; d_in: PSmallInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_scan_int32(d_out: PCint; d_in: PCint; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_scan_uint32(d_out: PCardinal; d_in: PCardinal; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_scan_int64(d_out: PInt64; d_in: PInt64; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_scan_uint64(d_out: PQWord; d_in: PQWord; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;

// --- rocprim::warp_sort ---
function rocprim_warp_sort_int8(d_out: PShortInt; d_in: PShortInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_sort_int16(d_out: PSmallInt; d_in: PSmallInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_sort_int32(d_out: PCint; d_in: PCint; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_sort_uint32(d_out: PCardinal; d_in: PCardinal; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_sort_int64(d_out: PInt64; d_in: PInt64; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_warp_sort_uint64(d_out: PQWord; d_in: PQWord; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;

// --- rocprim::histogram_range ---
function rocprim_histogram_range_int(
    temporary_storage: Pointer;
    storage_size: psize_t;
    samples: PCint;
    size: cuint;
    histogram: PCint;
    levels: cuint;
    level_values: PCint;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::merge_sort_pairs ---
function rocprim_merge_sort_pairs_int(
    temporary_storage: Pointer;
    storage_size: psize_t;
    keys_input: PCint;
    keys_output: PCint;
    values_input: PCint;
    values_output: PCint;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl; external librocprimshim;

// --- Gather / Scatter (float) ---
function rocprim_gather_float(
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
): hipError_t; cdecl; external librocprimshim;

function rocprim_clamp_float(
    d_data: PSingle;
    min_val: Single;
    max_val: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_bernoulli_float(
    d_data: PSingle;
    p: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- Elementwise math (float32, async) ---
function rocprim_exp_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_log_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_sqrt_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_recip_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_div_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_scale_inplace_float(
    d_data: PSingle;
    scalar: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_relu_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- Elementwise math: binary broadcast add/sub/mul (same convention as div) ---
function rocprim_add_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_sub_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_mul_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- scale (unary * scalar) ---
function rocprim_scale_float(
    d_in: PSingle;
    scalar: Single;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- sigmoid / tanh fwd + bwd (bwd takes grad + input, recomputes activation) ---
function rocprim_sigmoid_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_sigmoid_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_tanh_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_tanh_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- backward variants consuming the *saved forward output* ---
function rocprim_sigmoid_bwd_out_float(
    d_grad: PSingle;
    d_y: PSingle;     // sigmoid forward output
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_tanh_bwd_out_float(
    d_grad: PSingle;
    d_t: PSingle;     // tanh forward output
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- GELU / SiLU fwd + bwd (bwd takes grad + input) ---
function rocprim_gelu_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_gelu_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_silu_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_silu_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- native FP16 / BF16 activation variants (device pointers are raw
//     16-bit element buffers; kernels upconvert to FP32 in-register) ---
function rocprim_gelu_half(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_gelu_bf16(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_gelu_bwd_half(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_gelu_bwd_bf16(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_silu_half(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_silu_bf16(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_silu_bwd_half(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_silu_bwd_bf16(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- broadcast binary ops: out[i] = op(a[i], b[i mod blen]).
//     op: 0=add, 1=sub, 2=mul, 3=div. Correct for contiguous row-major A
//     and contiguous trailing-broadcast B (e.g. [1,16,192] + [192]). ---
function rocprim_bcast_op_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    alen: csize_t;
    blen: csize_t;
    op: cint32;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_bcast_op_half(
    d_a: Pointer;
    d_b: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    alen: csize_t;
    blen: csize_t;
    op: cint32;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_bcast_op_bf16(
    d_a: Pointer;
    d_b: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    alen: csize_t;
    blen: csize_t;
    op: cint32;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- general strided gather: materialize any permuted/sliced view.
//     h_shape/h_strides are host arrays of dimcount Int64s (elements). ---
function rocprim_gather_strided_float(
    d_src: PSingle;
    d_dst: PSingle;
    h_shape: PInt64;
    h_strides: PInt64;
    dimcount: cint32;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

{ --- native 16-bit same-shape binary ops (FP32 compute in-register) --- }
function rocprim_add_half(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_add_bf16(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_sub_half(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_sub_bf16(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_mul_half(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_mul_bf16(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;

{ Two-pass global sum: out[0] = sum(in[0..n-1]). }
function rocprim_sum_float(
    in_buf: PSingle;
    out_buf: PSingle;
    n: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- fill ---
function rocprim_fill_float(
    d_out: PSingle;
    value: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- row-wise softmax / logsoftmax fwd + bwd (contiguous last dim) ---
function rocprim_softmax_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_softmax_bwd_float(
    d_grad: PSingle;
    d_y: PSingle;   // softmax output
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_logsoftmax_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_logsoftmax_bwd_float(
    d_grad: PSingle;
    d_logy: PSingle;  // logsoftmax output
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- cumulative sum along the last (contiguous) dimension ---
function rocprim_cumsum_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- strided variants: op along a non-last dim of a contiguous tensor
//     viewed as [outer, len, inner]; one block per slice ---
function rocprim_softmax_strided_float(
    d_in: PSingle;
    d_out: PSingle;
    slices: csize_t;
    len: csize_t;
    inner: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_cumsum_strided_float(
    d_in: PSingle;
    d_out: PSingle;
    slices: csize_t;
    len: csize_t;
    inner: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- tiled 2D transpose (out[c][r] = in[r][c], coalesced both sides) ---
function rocprim_transpose_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- per-row top-k (largest, each row sorted descending); k <= 1024 ---
function rocprim_topk_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    k: cint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- row reductions along the last (contiguous) dimension ---
function rocprim_rowsum_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_rowmax_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_rowmin_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- repeat-interleave along one dim of a contiguous tensor ---
function rocprim_repeat_interleave_float(
    d_in: PSingle;
    d_out: PSingle;
    outer: csize_t;
    len: csize_t;
    inner: csize_t;
    n: cint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- dtype converters (typed-void for Pascal side convenience) ---
function rocprim_convert_f16_to_f32(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_convert_f32_to_f16(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_convert_bf16_to_f32(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_convert_f32_to_bf16(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_scatter_float(
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
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::inclusive_scan ---
function rocprim_inclusive_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::deterministic_inclusive_scan ---
function rocprim_deterministic_inclusive_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::exclusive_scan ---
function rocprim_exclusive_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::deterministic_exclusive_scan ---
function rocprim_deterministic_exclusive_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::radix_sort_keys (in-place) ---
function rocprim_radix_sort_keys_int(
    d_keys: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::transform ---
function rocprim_transform_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    multiplier: cint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::partition ---
function rocprim_partition_int(
    d_in: PCint;
    d_out: PCint;
    d_selected_count: PCsize_t;
    num_items: csize_t;
    pivot: cint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::select ---
function rocprim_select_int(
    d_in: PCint;
    d_out: PCint;
    d_selected_count: PCsize_t;
    num_items: csize_t;
    pivot: cint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::reduce_by_key ---
function rocprim_reduce_by_key_int(
    d_keys_in: PCint;
    d_values_in: PCint;
    num_items: csize_t;
    d_unique_keys_out: PCint;
    d_aggregates_out: PCint;
    d_unique_count_out: PCsize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::inclusive_scan_by_key ---
function rocprim_inclusive_scan_by_key_int(
    d_keys_in: PCint;
    d_values_in: PCint;
    d_values_out: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::exclusive_scan_by_key ---
function rocprim_exclusive_scan_by_key_int(
    d_keys_in: PCint;
    d_values_in: PCint;
    d_values_out: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::histogram_even ---
function rocprim_histogram_even_int(
    d_samples: PCint;
    num_samples: csize_t;
    d_histogram: PCint;
    levels: cuint;
    lower_level: cint;
    upper_level: cint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::merge ---
function rocprim_merge_int(
    d_input1: PCint;
    d_input2: PCint;
    d_output: PCint;
    input1_size: csize_t;
    input2_size: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::radix_sort_keys (input/output iterators) ---
function rocprim_radix_sort_keys_int_io(
    d_keys_in: PCint;
    d_keys_out: PCint;
    num_items: csize_t;
    begin_bit: cuint;
    end_bit: cuint;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::unique ---
function rocprim_unique_int(
    d_input: PCint;
    d_output: PCint;
    d_unique_count_out: PCsize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::copy ---
function rocprim_copy_int(
    d_input: PCint;
    d_output: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::fill ---
function rocprim_fill_int(
    d_output: PCint;
    value: cint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::adjacent_difference ---
function rocprim_adjacent_difference_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    d_input: PCint;
    d_output: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::adjacent_find ---
function rocprim_adjacent_find_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    d_input: PCint;
    d_output: PCsize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::merge_inplace ---
function rocprim_merge_inplace_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    data: PCint;
    left_size: csize_t;
    right_size: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- rocprim::binary_search ---
function rocprim_binary_search_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    haystack: PCint;
    needles: PCint;
    output: PCsize_t;
    haystack_size: csize_t;
    needles_size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl; external librocprimshim;

function rocprim_find_end_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    input: PCint;
    keys: PCint;
    output: PCsize_t;
    size: csize_t;
    keys_size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl; external librocprimshim;

function rocprim_find_first_of_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    input: PCint;
    keys: PCint;
    output: PCsize_t;
    size: csize_t;
    keys_size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl; external librocprimshim;

function rocprim_merge_sort_keys_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PCint;
    keys_output: PCint;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl; external librocprimshim;

function rocprim_nth_element_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PCint;
    keys_output: PCint;
    nth: csize_t;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl; external librocprimshim;

function rocprim_partial_sort_copy_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PInteger;
    keys_output: PInteger;
    middle: csize_t;  // Index (prefix length = middle +1)
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl; external librocprimshim;

function rocprim_radix_sort_keys_int_asc(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PCint;
    keys_output: PCint;
    size: csize_t;
    begin_bit: cuint;
    end_bit: cuint;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl; external librocprimshim;

function rocprim_radix_sort_pairs_int(
	temporary_storage: Pointer;
	storage_size: Pcsize_t;
	keys_input: Pointer;
	keys_output: Pointer;
	values_input: Pointer;
	values_output: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

function rocprim_partial_sort_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys: PInteger;
    middle: csize_t;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl; external librocprimshim;

function rocprim_run_length_encode_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    input: PInteger;
    unique_output: PInteger;
    counts_output: PInteger;
    runs_count_output: PInteger;
    size: cuint;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl; external librocprimshim;

function rocprim_run_length_encode_non_trivial_runs_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    input: PInteger;
    offsets_output: PInteger;
    counts_output: PInteger;
    runs_count_output: PInteger;
    size: cuint;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl; external librocprimshim;

function rocprim_search_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    haystack: PInteger;
    needle: PInteger;
    output: PInteger;
    haystack_size: csize_t;
    needle_size: csize_t;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl; external librocprimshim;

type
  rocprim_binary_op = (
    rocprim_binary_op_equal_to,
    rocprim_binary_op_not_equal_to,
    rocprim_binary_op_greater_than,
    rocprim_binary_op_less_than,
    rocprim_binary_op_greater_equal,
    rocprim_binary_op_less_equal
  );

function rocprim_search_n_int(
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
): hipError_t; cdecl; external librocprimshim;

function rocprim_segmented_reduce_int(
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
): hipError_t; cdecl; external librocprimshim;

function rocprim_segmented_inclusive_scan_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    input: PInteger;
    output: PInteger;
    segments: cuint;
    begin_offsets: PInteger;
    end_offsets: PInteger;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl; external librocprimshim;

function rocprim_segmented_exclusive_scan_int(
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
): hipError_t; cdecl; external librocprimshim;

function rocprim_segmented_radix_sort_keys_int(
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
): hipError_t; cdecl; external librocprimshim;

function rocprim_segmented_radix_sort_pairs_int(
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
): hipError_t; cdecl; external librocprimshim;

function rocprim_radix_sort_keys_desc_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PInteger;
    keys_output: PInteger;
    size: cuint;
    begin_bit: cuint;
    end_bit: cuint;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl; external librocprimshim;

function rocprim_radix_sort_pairs_desc_int(
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
): hipError_t; cdecl; external librocprimshim;

function rocprim_segmented_radix_sort_keys_desc_int(
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
): hipError_t; cdecl; external librocprimshim;

function rocprim_numeric_limits_int_min: cint; cdecl; external librocprimshim;
function rocprim_numeric_limits_int_max: cint; cdecl; external librocprimshim;
function rocprim_numeric_limits_float_epsilon: cfloat; cdecl; external librocprimshim;
function rocprim_numeric_limits_float_infinity: cfloat; cdecl; external librocprimshim;
function rocprim_numeric_limits_float_nan: cfloat; cdecl; external librocprimshim;

{ Row-wise argmax over the last (contiguous) dimension: one block per row.
  out[row] = index (as Single) of the max element. Ties -> lowest index. }
function rocprim_argmax_float(
    in_buf: PSingle;
    out_buf: PSingle;
    cols: Int64;
    rows: Int64;
    stream: hipStream_t
): hipError_t; cdecl; external librocprimshim;

// --- comparison ops (float mask 1.0/0.0, broadcast-modulo) ---
function rocprim_gt_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_lt_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_eq_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_ge_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_le_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_ne_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl; external librocprimshim;

// --- Where(cond, A, B) and MaskedFill ---
function rocprim_where_float(
    d_cond: PSingle; d_a: PSingle; d_b: PSingle; d_out: PSingle;
    c_n: csize_t; a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl; external librocprimshim;
function rocprim_maskedfill_float(
    d_data: PSingle; d_mask: PSingle; value: Single;
    mask_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl; external librocprimshim;

implementation
uses
  math;

end.
