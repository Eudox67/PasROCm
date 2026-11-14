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

{ The public rocPRIM shim surface. Each function is a thin wrapper: the call
  asks rocprim_shim_loader.pas for the shim that matches the detected GPU, and
  only then dispatches through the pointer table in rocprim_shim_api.pas. A
  missing library, a library built for another GPU, or a missing symbol raises
  an exception naming the problem -- it never calls through a nil pointer.

  This unit keeps the original names and signatures, so existing programs need
  no change; nothing here is resolved at link time (DESIGN.md section 13). }

unit rocprim_bindings;

{$mode objfpc}{$H+}

interface

uses
  ctypes, hip_types, hip_error, hip_runtime, rocprim_types;

type
  PInt128 = ^int128_t;
  TInt128 = int128_t;

{ Types shared by the declarations. Types must be declared before their first
  use, so they live here rather than between the declarations. }
type
  rocprim_binary_op = (
    rocprim_binary_op_equal_to,
    rocprim_binary_op_not_equal_to,
    rocprim_binary_op_greater_than,
    rocprim_binary_op_less_than,
    rocprim_binary_op_greater_equal,
    rocprim_binary_op_less_equal
  );

const
  { Monotonic ABI revision of this surface. Bump it for any change a consumer
    could notice: a removed or renamed symbol, a changed parameter or return
    type, a changed enum value. Adding a symbol is also a bump, because the
    strict gate counts. The loader refuses a library whose value differs from
    the one its generated units were built with. }
  PASROCM_ABI_REVISION = 1;

  { Kept for source compatibility only: nothing links against it any more. }
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
function pasrocm_shim_target: PAnsiChar; cdecl;

function pasrocm_shim_wavefront_size: cuint; cdecl;

function pasrocm_shim_rocm_version: PAnsiChar; cdecl;

function pasrocm_shim_wavefront_widths_mask: cuint; cdecl;

function pasrocm_wavefront_compatible(compiled, reported, allowed_mask: cuint): cint; cdecl;


// Monotonic ABI revision (see PASROCM_ABI_REVISION). It belongs to the
// identity block because a loader must be able to reject a mismatched library
// before it binds anything.
function pasrocm_shim_abi_version: cuint; cdecl;


// Deterministic provenance string: version, git commit, GPU target, wavefront
// width, widths mask, SDK stamp and compiler version. It never contains a build
// timestamp, so two builds of one commit stay byte-identical (the packaging
// determinism gate).
function pasrocm_shim_build_info: PAnsiChar; cdecl;


// --- rocprim::reduce ---
function rocprim_reduce_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_reduce_int128(
    d_out: PInt128;
    d_in: PInt128;
    num_items: csize_t;
    initial_value: TInt128;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_reduce_reverse_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::block_reduce ---
function rocprim_block_reduce_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::block_scan ---
function rocprim_block_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::block_sort ---
function rocprim_block_sort_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::block_histogram ---
function rocprim_block_histogram_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    bins: cuint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::warp_reduce ---
function rocprim_warp_reduce_int8(d_out: PShortInt; d_in: PShortInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_reduce_int16(d_out: PSmallInt; d_in: PSmallInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_reduce_int32(d_out: PCint; d_in: PCint; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_reduce_uint32(d_out: PCardinal; d_in: PCardinal; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_reduce_int64(d_out: PInt64; d_in: PInt64; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_reduce_uint64(d_out: PQWord; d_in: PQWord; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;


// --- rocprim::warp_scan ---
function rocprim_warp_scan_int8(d_out: PShortInt; d_in: PShortInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_scan_int16(d_out: PSmallInt; d_in: PSmallInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_scan_int32(d_out: PCint; d_in: PCint; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_scan_uint32(d_out: PCardinal; d_in: PCardinal; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_scan_int64(d_out: PInt64; d_in: PInt64; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_scan_uint64(d_out: PQWord; d_in: PQWord; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;


// --- rocprim::warp_sort ---
function rocprim_warp_sort_int8(d_out: PShortInt; d_in: PShortInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_sort_int16(d_out: PSmallInt; d_in: PSmallInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_sort_int32(d_out: PCint; d_in: PCint; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_sort_uint32(d_out: PCardinal; d_in: PCardinal; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_sort_int64(d_out: PInt64; d_in: PInt64; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_warp_sort_uint64(d_out: PQWord; d_in: PQWord; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


function rocprim_clamp_float(
    d_data: PSingle;
    min_val: Single;
    max_val: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_bernoulli_float(
    d_data: PSingle;
    p: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- Elementwise math (float32, async) ---
function rocprim_exp_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_log_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_sqrt_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_recip_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_div_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_scale_inplace_float(
    d_data: PSingle;
    scalar: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_relu_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- Elementwise math: binary broadcast add/sub/mul (same convention as div) ---
function rocprim_add_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_sub_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_mul_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- scale (unary * scalar) ---
function rocprim_scale_float(
    d_in: PSingle;
    scalar: Single;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- sigmoid / tanh fwd + bwd (bwd takes grad + input, recomputes activation) ---
function rocprim_sigmoid_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_sigmoid_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_tanh_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_tanh_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- backward variants consuming the *saved forward output* ---
function rocprim_sigmoid_bwd_out_float(
    d_grad: PSingle;
    d_y: PSingle;     // sigmoid forward output
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_tanh_bwd_out_float(
    d_grad: PSingle;
    d_t: PSingle;     // tanh forward output
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- GELU / SiLU fwd + bwd (bwd takes grad + input) ---
function rocprim_gelu_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_gelu_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_silu_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_silu_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- native FP16 / BF16 activation variants (device pointers are raw
//     16-bit element buffers; kernels upconvert to FP32 in-register) ---
function rocprim_gelu_half(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_gelu_bf16(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_gelu_bwd_half(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_gelu_bwd_bf16(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_silu_half(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_silu_bf16(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_silu_bwd_half(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_silu_bwd_bf16(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


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
): hipError_t; cdecl;


function rocprim_bcast_op_half(
    d_a: Pointer;
    d_b: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    alen: csize_t;
    blen: csize_t;
    op: cint32;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_bcast_op_bf16(
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
function rocprim_gather_strided_float(
    d_src: PSingle;
    d_dst: PSingle;
    h_shape: PInt64;
    h_strides: PInt64;
    dimcount: cint32;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


{ --- native 16-bit same-shape binary ops (FP32 compute in-register) --- }
function rocprim_add_half(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_add_bf16(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_sub_half(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_sub_bf16(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_mul_half(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;

function rocprim_mul_bf16(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;


{ Two-pass global sum: out[0] = sum(in[0..n-1]). }
function rocprim_sum_float(
    in_buf: PSingle;
    out_buf: PSingle;
    n: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- fill ---
function rocprim_fill_float(
    d_out: PSingle;
    value: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- row-wise softmax / logsoftmax fwd + bwd (contiguous last dim) ---
function rocprim_softmax_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_softmax_bwd_float(
    d_grad: PSingle;
    d_y: PSingle;   // softmax output
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_logsoftmax_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_logsoftmax_bwd_float(
    d_grad: PSingle;
    d_logy: PSingle;  // logsoftmax output
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- cumulative sum along the last (contiguous) dimension ---
function rocprim_cumsum_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- strided variants: op along a non-last dim of a contiguous tensor
//     viewed as [outer, len, inner]; one block per slice ---
function rocprim_softmax_strided_float(
    d_in: PSingle;
    d_out: PSingle;
    slices: csize_t;
    len: csize_t;
    inner: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_cumsum_strided_float(
    d_in: PSingle;
    d_out: PSingle;
    slices: csize_t;
    len: csize_t;
    inner: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- tiled 2D transpose (out[c][r] = in[r][c], coalesced both sides) ---
function rocprim_transpose_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- per-row top-k (largest, each row sorted descending); k <= 1024 ---
function rocprim_topk_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    k: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- row reductions along the last (contiguous) dimension ---
function rocprim_rowsum_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_rowmax_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_rowmin_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- repeat-interleave along one dim of a contiguous tensor ---
function rocprim_repeat_interleave_float(
    d_in: PSingle;
    d_out: PSingle;
    outer: csize_t;
    len: csize_t;
    inner: csize_t;
    n: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- dtype converters (typed-void for Pascal side convenience) ---
function rocprim_convert_f16_to_f32(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_convert_f32_to_f16(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_convert_bf16_to_f32(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_convert_f32_to_bf16(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


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
): hipError_t; cdecl;


// --- rocprim::inclusive_scan ---
function rocprim_inclusive_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::deterministic_inclusive_scan ---
function rocprim_deterministic_inclusive_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::exclusive_scan ---
function rocprim_exclusive_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::deterministic_exclusive_scan ---
function rocprim_deterministic_exclusive_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::radix_sort_keys (in-place) ---
function rocprim_radix_sort_keys_int(
    d_keys: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::transform ---
function rocprim_transform_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    multiplier: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::partition ---
function rocprim_partition_int(
    d_in: PCint;
    d_out: PCint;
    d_selected_count: PCsize_t;
    num_items: csize_t;
    pivot: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::select ---
function rocprim_select_int(
    d_in: PCint;
    d_out: PCint;
    d_selected_count: PCsize_t;
    num_items: csize_t;
    pivot: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::reduce_by_key ---
function rocprim_reduce_by_key_int(
    d_keys_in: PCint;
    d_values_in: PCint;
    num_items: csize_t;
    d_unique_keys_out: PCint;
    d_aggregates_out: PCint;
    d_unique_count_out: PCsize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::inclusive_scan_by_key ---
function rocprim_inclusive_scan_by_key_int(
    d_keys_in: PCint;
    d_values_in: PCint;
    d_values_out: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::exclusive_scan_by_key ---
function rocprim_exclusive_scan_by_key_int(
    d_keys_in: PCint;
    d_values_in: PCint;
    d_values_out: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::histogram_even ---
function rocprim_histogram_even_int(
    d_samples: PCint;
    num_samples: csize_t;
    d_histogram: PCint;
    levels: cuint;
    lower_level: cint;
    upper_level: cint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::merge ---
function rocprim_merge_int(
    d_input1: PCint;
    d_input2: PCint;
    d_output: PCint;
    input1_size: csize_t;
    input2_size: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::radix_sort_keys (input/output iterators) ---
function rocprim_radix_sort_keys_int_io(
    d_keys_in: PCint;
    d_keys_out: PCint;
    num_items: csize_t;
    begin_bit: cuint;
    end_bit: cuint;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::unique ---
function rocprim_unique_int(
    d_input: PCint;
    d_output: PCint;
    d_unique_count_out: PCsize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::copy ---
function rocprim_copy_int(
    d_input: PCint;
    d_output: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::fill ---
function rocprim_fill_int(
    d_output: PCint;
    value: cint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::adjacent_difference ---
function rocprim_adjacent_difference_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    d_input: PCint;
    d_output: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::adjacent_find ---
function rocprim_adjacent_find_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    d_input: PCint;
    d_output: PCsize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


// --- rocprim::merge_inplace ---
function rocprim_merge_inplace_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    data: PCint;
    left_size: csize_t;
    right_size: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


function rocprim_merge_sort_keys_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PCint;
    keys_output: PCint;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl;


function rocprim_nth_element_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PCint;
    keys_output: PCint;
    nth: csize_t;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl;


function rocprim_partial_sort_copy_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PInteger;
    keys_output: PInteger;
    middle: csize_t;  // Index (prefix length = middle +1)
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


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
): hipError_t; cdecl;


function rocprim_radix_sort_pairs_int(
	temporary_storage: Pointer;
	storage_size: Pcsize_t;
	keys_input: Pointer;
	keys_output: Pointer;
	values_input: Pointer;
	values_output: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;


function rocprim_partial_sort_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys: PInteger;
    middle: csize_t;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


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
): hipError_t; cdecl;


function rocprim_numeric_limits_int_min: cint; cdecl;

function rocprim_numeric_limits_int_max: cint; cdecl;

function rocprim_numeric_limits_float_epsilon: cfloat; cdecl;

function rocprim_numeric_limits_float_infinity: cfloat; cdecl;

function rocprim_numeric_limits_float_nan: cfloat; cdecl;


{ Row-wise argmax over the last (contiguous) dimension: one block per row.
  out[row] = index (as Single) of the max element. Ties -> lowest index. }
function rocprim_argmax_float(
    in_buf: PSingle;
    out_buf: PSingle;
    cols: Int64;
    rows: Int64;
    stream: hipStream_t
): hipError_t; cdecl;


// --- comparison ops (float mask 1.0/0.0, broadcast-modulo) ---
function rocprim_gt_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

function rocprim_lt_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

function rocprim_eq_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

function rocprim_ge_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

function rocprim_le_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

function rocprim_ne_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;


// --- Where(cond, A, B) and MaskedFill ---
function rocprim_where_float(
    d_cond: PSingle; d_a: PSingle; d_b: PSingle; d_out: PSingle;
    c_n: csize_t; a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

function rocprim_maskedfill_float(
    d_data: PSingle; d_mask: PSingle; value: Single;
    mask_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;

implementation

uses
  rocprim_shim_api,
  rocprim_shim_loader;

function pasrocm_shim_target: PAnsiChar; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__pasrocm_shim_target();
end;

function pasrocm_shim_wavefront_size: cuint; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__pasrocm_shim_wavefront_size();
end;

function pasrocm_shim_rocm_version: PAnsiChar; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__pasrocm_shim_rocm_version();
end;

function pasrocm_shim_wavefront_widths_mask: cuint; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__pasrocm_shim_wavefront_widths_mask();
end;

function pasrocm_wavefront_compatible(compiled, reported, allowed_mask: cuint): cint; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__pasrocm_wavefront_compatible(compiled, reported, allowed_mask);
end;

function pasrocm_shim_abi_version: cuint; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__pasrocm_shim_abi_version();
end;

function pasrocm_shim_build_info: PAnsiChar; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__pasrocm_shim_build_info();
end;

function rocprim_reduce_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_reduce_int(d_out, d_in, num_items, initial_value, stream);
end;

function rocprim_reduce_int128(
    d_out: PInt128;
    d_in: PInt128;
    num_items: csize_t;
    initial_value: TInt128;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_reduce_int128(d_out, d_in, num_items, initial_value, stream);
end;

function rocprim_reduce_reverse_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_reduce_reverse_int(d_out, d_in, num_items, initial_value, stream);
end;

function rocprim_block_reduce_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_block_reduce_int(d_out, d_in, num_items, block_size, stream);
end;

function rocprim_block_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_block_scan_int(d_out, d_in, num_items, block_size, stream);
end;

function rocprim_block_sort_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_block_sort_int(d_out, d_in, num_items, block_size, stream);
end;

function rocprim_block_histogram_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    block_size: cuint;
    bins: cuint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_block_histogram_int(d_out, d_in, num_items, block_size, bins, stream);
end;

function rocprim_warp_reduce_int8(d_out: PShortInt; d_in: PShortInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_reduce_int8(d_out, d_in, num_items, stream);
end;

function rocprim_warp_reduce_int16(d_out: PSmallInt; d_in: PSmallInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_reduce_int16(d_out, d_in, num_items, stream);
end;

function rocprim_warp_reduce_int32(d_out: PCint; d_in: PCint; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_reduce_int32(d_out, d_in, num_items, stream);
end;

function rocprim_warp_reduce_uint32(d_out: PCardinal; d_in: PCardinal; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_reduce_uint32(d_out, d_in, num_items, stream);
end;

function rocprim_warp_reduce_int64(d_out: PInt64; d_in: PInt64; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_reduce_int64(d_out, d_in, num_items, stream);
end;

function rocprim_warp_reduce_uint64(d_out: PQWord; d_in: PQWord; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_reduce_uint64(d_out, d_in, num_items, stream);
end;

function rocprim_warp_scan_int8(d_out: PShortInt; d_in: PShortInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_scan_int8(d_out, d_in, num_items, stream);
end;

function rocprim_warp_scan_int16(d_out: PSmallInt; d_in: PSmallInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_scan_int16(d_out, d_in, num_items, stream);
end;

function rocprim_warp_scan_int32(d_out: PCint; d_in: PCint; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_scan_int32(d_out, d_in, num_items, stream);
end;

function rocprim_warp_scan_uint32(d_out: PCardinal; d_in: PCardinal; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_scan_uint32(d_out, d_in, num_items, stream);
end;

function rocprim_warp_scan_int64(d_out: PInt64; d_in: PInt64; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_scan_int64(d_out, d_in, num_items, stream);
end;

function rocprim_warp_scan_uint64(d_out: PQWord; d_in: PQWord; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_scan_uint64(d_out, d_in, num_items, stream);
end;

function rocprim_warp_sort_int8(d_out: PShortInt; d_in: PShortInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_sort_int8(d_out, d_in, num_items, stream);
end;

function rocprim_warp_sort_int16(d_out: PSmallInt; d_in: PSmallInt; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_sort_int16(d_out, d_in, num_items, stream);
end;

function rocprim_warp_sort_int32(d_out: PCint; d_in: PCint; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_sort_int32(d_out, d_in, num_items, stream);
end;

function rocprim_warp_sort_uint32(d_out: PCardinal; d_in: PCardinal; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_sort_uint32(d_out, d_in, num_items, stream);
end;

function rocprim_warp_sort_int64(d_out: PInt64; d_in: PInt64; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_sort_int64(d_out, d_in, num_items, stream);
end;

function rocprim_warp_sort_uint64(d_out: PQWord; d_in: PQWord; num_items: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_warp_sort_uint64(d_out, d_in, num_items, stream);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_histogram_range_int(temporary_storage, storage_size, samples, size, histogram, levels, level_values, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_merge_sort_pairs_int(temporary_storage, storage_size, keys_input, keys_output, values_input, values_output, size, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_gather_float(d_input, d_index, d_output, num_items, dim, rank, d_in_strides, d_idx_strides, d_out_strides, d_out_shape, stream);
end;

function rocprim_clamp_float(
    d_data: PSingle;
    min_val: Single;
    max_val: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_clamp_float(d_data, min_val, max_val, num_items, stream);
end;

function rocprim_bernoulli_float(
    d_data: PSingle;
    p: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_bernoulli_float(d_data, p, num_items, stream);
end;

function rocprim_exp_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_exp_float(d_in, d_out, num_items, stream);
end;

function rocprim_log_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_log_float(d_in, d_out, num_items, stream);
end;

function rocprim_sqrt_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_sqrt_float(d_in, d_out, num_items, stream);
end;

function rocprim_recip_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_recip_float(d_in, d_out, num_items, stream);
end;

function rocprim_div_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_div_float(d_a, d_b, d_out, a_count, b_count, num_items, stream);
end;

function rocprim_scale_inplace_float(
    d_data: PSingle;
    scalar: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_scale_inplace_float(d_data, scalar, num_items, stream);
end;

function rocprim_relu_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_relu_bwd_float(d_grad, d_input, d_out, num_items, stream);
end;

function rocprim_add_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_add_float(d_a, d_b, d_out, a_count, b_count, num_items, stream);
end;

function rocprim_sub_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_sub_float(d_a, d_b, d_out, a_count, b_count, num_items, stream);
end;

function rocprim_mul_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    a_count: csize_t;
    b_count: csize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_mul_float(d_a, d_b, d_out, a_count, b_count, num_items, stream);
end;

function rocprim_scale_float(
    d_in: PSingle;
    scalar: Single;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_scale_float(d_in, scalar, d_out, num_items, stream);
end;

function rocprim_sigmoid_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_sigmoid_float(d_in, d_out, num_items, stream);
end;

function rocprim_sigmoid_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_sigmoid_bwd_float(d_grad, d_input, d_out, num_items, stream);
end;

function rocprim_tanh_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_tanh_float(d_in, d_out, num_items, stream);
end;

function rocprim_tanh_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_tanh_bwd_float(d_grad, d_input, d_out, num_items, stream);
end;

function rocprim_sigmoid_bwd_out_float(
    d_grad: PSingle;
    d_y: PSingle;     // sigmoid forward output
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_sigmoid_bwd_out_float(d_grad, d_y, d_out, num_items, stream);
end;

function rocprim_tanh_bwd_out_float(
    d_grad: PSingle;
    d_t: PSingle;     // tanh forward output
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_tanh_bwd_out_float(d_grad, d_t, d_out, num_items, stream);
end;

function rocprim_gelu_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_gelu_float(d_in, d_out, num_items, stream);
end;

function rocprim_gelu_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_gelu_bwd_float(d_grad, d_input, d_out, num_items, stream);
end;

function rocprim_silu_float(
    d_in: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_silu_float(d_in, d_out, num_items, stream);
end;

function rocprim_silu_bwd_float(
    d_grad: PSingle;
    d_input: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_silu_bwd_float(d_grad, d_input, d_out, num_items, stream);
end;

function rocprim_gelu_half(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_gelu_half(d_in, d_out, num_items, stream);
end;

function rocprim_gelu_bf16(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_gelu_bf16(d_in, d_out, num_items, stream);
end;

function rocprim_gelu_bwd_half(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_gelu_bwd_half(d_grad, d_input, d_out, num_items, stream);
end;

function rocprim_gelu_bwd_bf16(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_gelu_bwd_bf16(d_grad, d_input, d_out, num_items, stream);
end;

function rocprim_silu_half(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_silu_half(d_in, d_out, num_items, stream);
end;

function rocprim_silu_bf16(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_silu_bf16(d_in, d_out, num_items, stream);
end;

function rocprim_silu_bwd_half(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_silu_bwd_half(d_grad, d_input, d_out, num_items, stream);
end;

function rocprim_silu_bwd_bf16(
    d_grad: Pointer;
    d_input: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_silu_bwd_bf16(d_grad, d_input, d_out, num_items, stream);
end;

function rocprim_bcast_op_float(
    d_a: PSingle;
    d_b: PSingle;
    d_out: PSingle;
    num_items: csize_t;
    alen: csize_t;
    blen: csize_t;
    op: cint32;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_bcast_op_float(d_a, d_b, d_out, num_items, alen, blen, op, stream);
end;

function rocprim_bcast_op_half(
    d_a: Pointer;
    d_b: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    alen: csize_t;
    blen: csize_t;
    op: cint32;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_bcast_op_half(d_a, d_b, d_out, num_items, alen, blen, op, stream);
end;

function rocprim_bcast_op_bf16(
    d_a: Pointer;
    d_b: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    alen: csize_t;
    blen: csize_t;
    op: cint32;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_bcast_op_bf16(d_a, d_b, d_out, num_items, alen, blen, op, stream);
end;

function rocprim_gather_strided_float(
    d_src: PSingle;
    d_dst: PSingle;
    h_shape: PInt64;
    h_strides: PInt64;
    dimcount: cint32;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_gather_strided_float(d_src, d_dst, h_shape, h_strides, dimcount, num_items, stream);
end;

function rocprim_add_half(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_add_half(a, b, out_buf, n, stream);
end;

function rocprim_add_bf16(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_add_bf16(a, b, out_buf, n, stream);
end;

function rocprim_sub_half(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_sub_half(a, b, out_buf, n, stream);
end;

function rocprim_sub_bf16(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_sub_bf16(a, b, out_buf, n, stream);
end;

function rocprim_mul_half(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_mul_half(a, b, out_buf, n, stream);
end;

function rocprim_mul_bf16(a, b, out_buf: Pointer; n: csize_t; stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_mul_bf16(a, b, out_buf, n, stream);
end;

function rocprim_sum_float(
    in_buf: PSingle;
    out_buf: PSingle;
    n: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_sum_float(in_buf, out_buf, n, stream);
end;

function rocprim_fill_float(
    d_out: PSingle;
    value: Single;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_fill_float(d_out, value, num_items, stream);
end;

function rocprim_softmax_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_softmax_float(d_in, d_out, rows, cols, stream);
end;

function rocprim_softmax_bwd_float(
    d_grad: PSingle;
    d_y: PSingle;   // softmax output
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_softmax_bwd_float(d_grad, d_y, d_out, rows, cols, stream);
end;

function rocprim_logsoftmax_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_logsoftmax_float(d_in, d_out, rows, cols, stream);
end;

function rocprim_logsoftmax_bwd_float(
    d_grad: PSingle;
    d_logy: PSingle;  // logsoftmax output
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_logsoftmax_bwd_float(d_grad, d_logy, d_out, rows, cols, stream);
end;

function rocprim_cumsum_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_cumsum_float(d_in, d_out, rows, cols, stream);
end;

function rocprim_softmax_strided_float(
    d_in: PSingle;
    d_out: PSingle;
    slices: csize_t;
    len: csize_t;
    inner: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_softmax_strided_float(d_in, d_out, slices, len, inner, stream);
end;

function rocprim_cumsum_strided_float(
    d_in: PSingle;
    d_out: PSingle;
    slices: csize_t;
    len: csize_t;
    inner: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_cumsum_strided_float(d_in, d_out, slices, len, inner, stream);
end;

function rocprim_transpose_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_transpose_float(d_in, d_out, rows, cols, stream);
end;

function rocprim_topk_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    k: cint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_topk_float(d_in, d_out, rows, cols, k, stream);
end;

function rocprim_rowsum_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_rowsum_float(d_in, d_out, rows, cols, stream);
end;

function rocprim_rowmax_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_rowmax_float(d_in, d_out, rows, cols, stream);
end;

function rocprim_rowmin_float(
    d_in: PSingle;
    d_out: PSingle;
    rows: csize_t;
    cols: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_rowmin_float(d_in, d_out, rows, cols, stream);
end;

function rocprim_repeat_interleave_float(
    d_in: PSingle;
    d_out: PSingle;
    outer: csize_t;
    len: csize_t;
    inner: csize_t;
    n: cint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_repeat_interleave_float(d_in, d_out, outer, len, inner, n, stream);
end;

function rocprim_convert_f16_to_f32(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_convert_f16_to_f32(d_in, d_out, num_items, stream);
end;

function rocprim_convert_f32_to_f16(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_convert_f32_to_f16(d_in, d_out, num_items, stream);
end;

function rocprim_convert_bf16_to_f32(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_convert_bf16_to_f32(d_in, d_out, num_items, stream);
end;

function rocprim_convert_f32_to_bf16(
    d_in: Pointer;
    d_out: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_convert_f32_to_bf16(d_in, d_out, num_items, stream);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_scatter_float(d_output, d_index, d_src, num_items, dim, rank, d_out_strides, d_idx_strides, d_src_strides, d_src_shape, atomic_add, stream);
end;

function rocprim_inclusive_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_inclusive_scan_int(d_out, d_in, num_items, stream);
end;

function rocprim_deterministic_inclusive_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_deterministic_inclusive_scan_int(d_out, d_in, num_items, stream);
end;

function rocprim_exclusive_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_exclusive_scan_int(d_out, d_in, num_items, initial_value, stream);
end;

function rocprim_deterministic_exclusive_scan_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_deterministic_exclusive_scan_int(d_out, d_in, num_items, initial_value, stream);
end;

function rocprim_radix_sort_keys_int(
    d_keys: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_radix_sort_keys_int(d_keys, num_items, stream);
end;

function rocprim_transform_int(
    d_out: PCint;
    d_in: PCint;
    num_items: csize_t;
    multiplier: cint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_transform_int(d_out, d_in, num_items, multiplier, stream);
end;

function rocprim_partition_int(
    d_in: PCint;
    d_out: PCint;
    d_selected_count: PCsize_t;
    num_items: csize_t;
    pivot: cint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_partition_int(d_in, d_out, d_selected_count, num_items, pivot, stream);
end;

function rocprim_select_int(
    d_in: PCint;
    d_out: PCint;
    d_selected_count: PCsize_t;
    num_items: csize_t;
    pivot: cint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_select_int(d_in, d_out, d_selected_count, num_items, pivot, stream);
end;

function rocprim_reduce_by_key_int(
    d_keys_in: PCint;
    d_values_in: PCint;
    num_items: csize_t;
    d_unique_keys_out: PCint;
    d_aggregates_out: PCint;
    d_unique_count_out: PCsize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_reduce_by_key_int(d_keys_in, d_values_in, num_items, d_unique_keys_out, d_aggregates_out, d_unique_count_out, stream);
end;

function rocprim_inclusive_scan_by_key_int(
    d_keys_in: PCint;
    d_values_in: PCint;
    d_values_out: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_inclusive_scan_by_key_int(d_keys_in, d_values_in, d_values_out, num_items, stream);
end;

function rocprim_exclusive_scan_by_key_int(
    d_keys_in: PCint;
    d_values_in: PCint;
    d_values_out: PCint;
    num_items: csize_t;
    initial_value: cint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_exclusive_scan_by_key_int(d_keys_in, d_values_in, d_values_out, num_items, initial_value, stream);
end;

function rocprim_histogram_even_int(
    d_samples: PCint;
    num_samples: csize_t;
    d_histogram: PCint;
    levels: cuint;
    lower_level: cint;
    upper_level: cint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_histogram_even_int(d_samples, num_samples, d_histogram, levels, lower_level, upper_level, stream);
end;

function rocprim_merge_int(
    d_input1: PCint;
    d_input2: PCint;
    d_output: PCint;
    input1_size: csize_t;
    input2_size: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_merge_int(d_input1, d_input2, d_output, input1_size, input2_size, stream);
end;

function rocprim_radix_sort_keys_int_io(
    d_keys_in: PCint;
    d_keys_out: PCint;
    num_items: csize_t;
    begin_bit: cuint;
    end_bit: cuint;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_radix_sort_keys_int_io(d_keys_in, d_keys_out, num_items, begin_bit, end_bit, stream);
end;

function rocprim_unique_int(
    d_input: PCint;
    d_output: PCint;
    d_unique_count_out: PCsize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_unique_int(d_input, d_output, d_unique_count_out, num_items, stream);
end;

function rocprim_copy_int(
    d_input: PCint;
    d_output: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_copy_int(d_input, d_output, num_items, stream);
end;

function rocprim_fill_int(
    d_output: PCint;
    value: cint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_fill_int(d_output, value, num_items, stream);
end;

function rocprim_adjacent_difference_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    d_input: PCint;
    d_output: PCint;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_adjacent_difference_int(temporary_storage, storage_size, d_input, d_output, num_items, stream);
end;

function rocprim_adjacent_find_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    d_input: PCint;
    d_output: PCsize_t;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_adjacent_find_int(temporary_storage, storage_size, d_input, d_output, num_items, stream);
end;

function rocprim_merge_inplace_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    data: PCint;
    left_size: csize_t;
    right_size: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_merge_inplace_int(temporary_storage, storage_size, data, left_size, right_size, stream);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_binary_search_int(temporary_storage, storage_size, haystack, needles, output, haystack_size, needles_size, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_find_end_int(temporary_storage, storage_size, input, keys, output, size, keys_size, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_find_first_of_int(temporary_storage, storage_size, input, keys, output, size, keys_size, stream, debug_synchronous);
end;

function rocprim_merge_sort_keys_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PCint;
    keys_output: PCint;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_merge_sort_keys_int(temporary_storage, storage_size, keys_input, keys_output, size, stream, debug_synchronous);
end;

function rocprim_nth_element_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PCint;
    keys_output: PCint;
    nth: csize_t;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: cbool
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_nth_element_int(temporary_storage, storage_size, keys_input, keys_output, nth, size, stream, debug_synchronous);
end;

function rocprim_partial_sort_copy_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys_input: PInteger;
    keys_output: PInteger;
    middle: csize_t;  // Index (prefix length = middle +1)
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_partial_sort_copy_int(temporary_storage, storage_size, keys_input, keys_output, middle, size, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_radix_sort_keys_int_asc(temporary_storage, storage_size, keys_input, keys_output, size, begin_bit, end_bit, stream, debug_synchronous);
end;

function rocprim_radix_sort_pairs_int(
	temporary_storage: Pointer;
	storage_size: Pcsize_t;
	keys_input: Pointer;
	keys_output: Pointer;
	values_input: Pointer;
	values_output: Pointer;
    num_items: csize_t;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_radix_sort_pairs_int(temporary_storage, storage_size, keys_input, keys_output, values_input, values_output, num_items, stream);
end;

function rocprim_partial_sort_int(
    temporary_storage: Pointer;
    storage_size: PCsize_t;
    keys: PInteger;
    middle: csize_t;
    size: csize_t;
    stream: hipStream_t;
    debug_synchronous: Boolean
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_partial_sort_int(temporary_storage, storage_size, keys, middle, size, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_run_length_encode_int(temporary_storage, storage_size, input, unique_output, counts_output, runs_count_output, size, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_run_length_encode_non_trivial_runs_int(temporary_storage, storage_size, input, offsets_output, counts_output, runs_count_output, size, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_search_int(temporary_storage, storage_size, haystack, needle, output, haystack_size, needle_size, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_search_n_int(temporary_storage, storage_size, haystack, output, haystack_size, count, value, op, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_segmented_reduce_int(temporary_storage, storage_size, input, output, segments, begin_offsets, end_offsets, initial_value, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_segmented_inclusive_scan_int(temporary_storage, storage_size, input, output, segments, begin_offsets, end_offsets, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_segmented_exclusive_scan_int(temporary_storage, storage_size, input, output, segments, begin_offsets, end_offsets, initial_value, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_segmented_radix_sort_keys_int(temporary_storage, storage_size, keys_input, keys_output, size, segments, begin_offsets, end_offsets, begin_bit, end_bit, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_segmented_radix_sort_pairs_int(temporary_storage, storage_size, keys_input, keys_output, values_input, values_output, size, segments, begin_offsets, end_offsets, begin_bit, end_bit, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_radix_sort_keys_desc_int(temporary_storage, storage_size, keys_input, keys_output, size, begin_bit, end_bit, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_radix_sort_pairs_desc_int(temporary_storage, storage_size, keys_input, keys_output, values_input, values_output, size, begin_bit, end_bit, stream, debug_synchronous);
end;

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
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_segmented_radix_sort_keys_desc_int(temporary_storage, storage_size, keys_input, keys_output, size, segments, begin_offsets, end_offsets, begin_bit, end_bit, stream, debug_synchronous);
end;

function rocprim_numeric_limits_int_min: cint; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_numeric_limits_int_min();
end;

function rocprim_numeric_limits_int_max: cint; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_numeric_limits_int_max();
end;

function rocprim_numeric_limits_float_epsilon: cfloat; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_numeric_limits_float_epsilon();
end;

function rocprim_numeric_limits_float_infinity: cfloat; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_numeric_limits_float_infinity();
end;

function rocprim_numeric_limits_float_nan: cfloat; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_numeric_limits_float_nan();
end;

function rocprim_argmax_float(
    in_buf: PSingle;
    out_buf: PSingle;
    cols: Int64;
    rows: Int64;
    stream: hipStream_t
): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_argmax_float(in_buf, out_buf, cols, rows, stream);
end;

function rocprim_gt_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_gt_float(d_a, d_b, d_out, a_n, b_n, n, stream);
end;

function rocprim_lt_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_lt_float(d_a, d_b, d_out, a_n, b_n, n, stream);
end;

function rocprim_eq_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_eq_float(d_a, d_b, d_out, a_n, b_n, n, stream);
end;

function rocprim_ge_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_ge_float(d_a, d_b, d_out, a_n, b_n, n, stream);
end;

function rocprim_le_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_le_float(d_a, d_b, d_out, a_n, b_n, n, stream);
end;

function rocprim_ne_float(
    d_a: PSingle; d_b: PSingle; d_out: PSingle;
    a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_ne_float(d_a, d_b, d_out, a_n, b_n, n, stream);
end;

function rocprim_where_float(
    d_cond: PSingle; d_a: PSingle; d_b: PSingle; d_out: PSingle;
    c_n: csize_t; a_n: csize_t; b_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_where_float(d_cond, d_a, d_b, d_out, c_n, a_n, b_n, n, stream);
end;

function rocprim_maskedfill_float(
    d_data: PSingle; d_mask: PSingle; value: Single;
    mask_n: csize_t; n: csize_t;
    stream: hipStream_t): hipError_t; cdecl;
begin
  PasROCmShimEnsureLoaded;
  Result := rocprim_shim_api.__rocprim_maskedfill_float(d_data, d_mask, value, mask_n, n, stream);
end;

end.
