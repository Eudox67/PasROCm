/*
------------------------------------------------------------------------------
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
------------------------------------------------------------------------------
*/

/*
 * GENERATED FILE -- do not edit.
 *   Source:    the definitions in rocprim_shim.cpp and
 *              rocprim_shim_math_kernels.cpp, gated by the 149 names in
 *              rocprim_shim_abi.inc (which owns the ABI contract).
 *   Generator: ci/gen-shim-header.py
 *   Regenerate: python3 ci/gen-shim-header.py
 *
 * Every prototype carries PASROCM_EXPORT because the library is compiled with
 * -fvisibility=hidden: this header is the complete list of exported symbols.
 * The SONAME is the on-disk file name (libpasrocm_shim_<gfx>.so), because the
 * Pascal loader resolves the library by exact file name; ABI breaks are gated
 * by ci/check-abi.sh --strict and the package version, not by a soname suffix.
 */
#ifndef PASROCM_ROCprim_SHIM_H
#define PASROCM_ROCprim_SHIM_H

#include <hip/hip_runtime.h> /* hipError_t, hipStream_t */
#include <hip/hip_fp16.h>     /* __half */
#include <hip/hip_bfloat16.h> /* hip_bfloat16 */
#include <stdbool.h>
#include <stddef.h> /* size_t */

#if defined(_WIN32)
#  define PASROCM_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define PASROCM_EXPORT __attribute__((visibility("default")))
#else
#  define PASROCM_EXPORT
#endif

/* ABI revision, generated from the spec. A consumer compares this with
 * pasrocm_shim_abi_version() before trusting the prototypes below. */
#define PASROCM_SHIM_ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

/* ABI enum, generated from the spec's TYPES section so the C and Pascal sides
 * cannot drift. Same order means the same values. */
typedef enum rocprim_binary_op {
    rocprim_binary_op_equal_to,
    rocprim_binary_op_not_equal_to,
    rocprim_binary_op_greater_than,
    rocprim_binary_op_less_than,
    rocprim_binary_op_greater_equal,
    rocprim_binary_op_less_equal
} rocprim_binary_op;

PASROCM_EXPORT const char* pasrocm_shim_target(void);
PASROCM_EXPORT unsigned    pasrocm_shim_wavefront_size(void);
PASROCM_EXPORT const char* pasrocm_shim_rocm_version(void);
PASROCM_EXPORT unsigned pasrocm_shim_wavefront_widths_mask(void);
PASROCM_EXPORT int pasrocm_wavefront_compatible(unsigned compiled,
                                 unsigned reported,
                                 unsigned allowed_mask);
PASROCM_EXPORT unsigned pasrocm_shim_abi_version(void);
PASROCM_EXPORT const char* pasrocm_shim_build_info(void);
PASROCM_EXPORT hipError_t rocprim_reduce_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    int initial_value,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_reduce_int128(
    __int128_t* d_out,
    const __int128_t* d_in,
    size_t num_items,
    __int128_t initial_value,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_reduce_reverse_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    int initial_value,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_block_reduce_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    unsigned int block_size,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_block_scan_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    unsigned int block_size,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_block_sort_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    unsigned int block_size,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_block_histogram_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    unsigned int block_size,
    unsigned int bins,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_reduce_int8(int8_t* d_out, const int8_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_reduce_int16(int16_t* d_out, const int16_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_reduce_int32(int32_t* d_out, const int32_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_reduce_uint32(uint32_t* d_out, const uint32_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_reduce_int64(int64_t* d_out, const int64_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_reduce_uint64(uint64_t* d_out, const uint64_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_scan_int8(int8_t* d_out, const int8_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_scan_int16(int16_t* d_out, const int16_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_scan_int32(int32_t* d_out, const int32_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_scan_uint32(uint32_t* d_out, const uint32_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_scan_int64(int64_t* d_out, const int64_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_scan_uint64(uint64_t* d_out, const uint64_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_sort_int8(int8_t* d_out, const int8_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_sort_int16(int16_t* d_out, const int16_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_sort_int32(int32_t* d_out, const int32_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_sort_uint32(uint32_t* d_out, const uint32_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_sort_int64(int64_t* d_out, const int64_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_warp_sort_uint64(uint64_t* d_out, const uint64_t* d_in, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_histogram_range_int(
    void* temporary_storage,
    size_t* storage_size,
    const int* samples,
    unsigned int size,
    int* histogram,
    unsigned int levels,
    int* level_values,
    hipStream_t stream,
    bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_merge_sort_pairs_int(
    void* temporary_storage,
    size_t* storage_size,
    const int* keys_input,
    int* keys_output,
    const int* values_input,
    int* values_output,
    size_t size,
    hipStream_t stream,
    bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_gather_float(
    const float* d_input,
    const float* d_index,
    float* d_output,
    size_t num_items,
    int dim,
    int rank,
    const long long* d_in_strides,
    const long long* d_idx_strides,
    const long long* d_out_strides,
    const long long* d_out_shape,
    hipStream_t stream
);
PASROCM_EXPORT hipError_t rocprim_clamp_float(float* d_data, float min_val, float max_val, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_bernoulli_float(float* d_data, float p, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_exp_float(const float* in, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_log_float(const float* in, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_sqrt_float(const float* in, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_recip_float(const float* in, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_div_float(const float* a, const float* b, float* out, size_t a_n, size_t b_n, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_scale_inplace_float(float* data, float scalar, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_relu_bwd_float(const float* grad, const float* input, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_add_float(const float* a, const float* b, float* out, size_t a_n, size_t b_n, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_sub_float(const float* a, const float* b, float* out, size_t a_n, size_t b_n, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_mul_float(const float* a, const float* b, float* out, size_t a_n, size_t b_n, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_scale_float(const float* in, float scalar, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_sigmoid_float(const float* in, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_sigmoid_bwd_float(const float* grad, const float* input, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_tanh_float(const float* in, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_tanh_bwd_float(const float* grad, const float* input, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_sigmoid_bwd_out_float(const float* grad, const float* y, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_tanh_bwd_out_float(const float* grad, const float* t, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_gelu_float(const float* in, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_gelu_bwd_float(const float* grad, const float* input, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_silu_float(const float* in, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_silu_bwd_float(const float* grad, const float* input, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_gelu_half(const __half* in, __half* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_gelu_bf16(const hip_bfloat16* in, hip_bfloat16* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_gelu_bwd_half(const __half* grad, const __half* input, __half* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_gelu_bwd_bf16(const hip_bfloat16* grad, const hip_bfloat16* input, hip_bfloat16* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_silu_half(const __half* in, __half* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_silu_bf16(const hip_bfloat16* in, hip_bfloat16* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_silu_bwd_half(const __half* grad, const __half* input, __half* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_silu_bwd_bf16(const hip_bfloat16* grad, const hip_bfloat16* input, hip_bfloat16* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_bcast_op_float(
    const float* d_a, const float* d_b, float* d_out,
    size_t num_items, size_t alen, size_t blen, int op, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_bcast_op_half(
    const void* d_a, const void* d_b, void* d_out,
    size_t num_items, size_t alen, size_t blen, int op, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_bcast_op_bf16(
    const void* d_a, const void* d_b, void* d_out,
    size_t num_items, size_t alen, size_t blen, int op, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_gather_strided_float(
    const float* d_src, float* d_dst,
    const long long* h_shape, const long long* h_strides,
    int dimcount, size_t num_items, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_add_half(const void* a, const void* b, void* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_add_bf16(const void* a, const void* b, void* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_sub_half(const void* a, const void* b, void* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_sub_bf16(const void* a, const void* b, void* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_mul_half(const void* a, const void* b, void* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_mul_bf16(const void* a, const void* b, void* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_sum_float(
    const float* in, float* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_fill_float(float* out, float value, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_softmax_float(const float* in, float* out, size_t rows, size_t cols, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_softmax_bwd_float(const float* grad, const float* y, float* out, size_t rows, size_t cols, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_logsoftmax_float(const float* in, float* out, size_t rows, size_t cols, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_logsoftmax_bwd_float(const float* grad, const float* y, float* out, size_t rows, size_t cols, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_cumsum_float(const float* in, float* out, size_t rows, size_t cols, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_softmax_strided_float(const float* in, float* out,
                                                    size_t slices, size_t len, size_t inner, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_cumsum_strided_float(const float* in, float* out,
                                                   size_t slices, size_t len, size_t inner, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_transpose_float(const float* in, float* out,
                                              size_t rows, size_t cols, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_topk_float(const float* in, float* out,
                                         size_t rows, size_t cols, int k, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_rowsum_float(const float* in, float* out,
                                           size_t rows, size_t cols, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_rowmax_float(const float* in, float* out,
                                           size_t rows, size_t cols, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_rowmin_float(const float* in, float* out,
                                           size_t rows, size_t cols, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_repeat_interleave_float(const float* in, float* out,
                                                      size_t outer, size_t len, size_t inner,
                                                      int n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_convert_f16_to_f32(const void* in, void* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_convert_f32_to_f16(const void* in, void* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_convert_bf16_to_f32(const void* in, void* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_convert_f32_to_bf16(const void* in, void* out, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_scatter_float(
    float* d_output,
    const float* d_index,
    const float* d_src,
    size_t num_items,
    int dim,
    int rank,
    const long long* d_out_strides,
    const long long* d_idx_strides,
    const long long* d_src_strides,
    const long long* d_src_shape,
    bool atomic_add,
    hipStream_t stream
);
PASROCM_EXPORT hipError_t rocprim_inclusive_scan_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_deterministic_inclusive_scan_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_exclusive_scan_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    int initial_value,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_deterministic_exclusive_scan_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    int initial_value,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_radix_sort_keys_int(
    int* d_keys,
    size_t num_items,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_transform_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    int multiplier,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_partition_int(
    int* d_in,
    int* d_out,
    size_t* d_selected_count,
    size_t num_items,
    int pivot,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_select_int(
    int* d_in,
    int* d_out,
    size_t* d_selected_count,
    size_t num_items,
    int pivot,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_reduce_by_key_int(
    int* d_keys_in,
    int* d_values_in,
    size_t num_items,
    int* d_unique_keys_out,
    int* d_aggregates_out,
    size_t* d_unique_count_out,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_inclusive_scan_by_key_int(
    int* d_keys_in,
    int* d_values_in,
    int* d_values_out,
    size_t num_items,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_exclusive_scan_by_key_int(
    int* d_keys_in,
    int* d_values_in,
    int* d_values_out,
    size_t num_items,
    int initial_value,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_histogram_even_int(
    int* d_samples,
    size_t num_samples,
    int* d_histogram,
    unsigned int levels,
    int lower_level,
    int upper_level,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_merge_int(
    int* d_input1,
    int* d_input2,
    int* d_output,
    size_t input1_size,
    size_t input2_size,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_radix_sort_keys_int_io(
    int* d_keys_in,
    int* d_keys_out,
    size_t num_items,
    unsigned int begin_bit,
    unsigned int end_bit,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_unique_int(
    int* d_input,
    int* d_output,
    size_t* d_unique_count_out,
    size_t num_items,
    hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_copy_int(
    int* d_input,
    int* d_output,
    size_t num_items,
    hipStream_t stream);
PASROCM_EXPORT void rocprim_fill_int(void* output, int value, size_t count, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_adjacent_difference_int(void* temporary_storage,
                                           size_t* storage_size,
                                           const int* input,
                                           int* output,
                                           size_t size,
                                           hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_adjacent_find_int(void* temporary_storage,
                                     size_t* storage_size,
                                     const int* input,
                                     size_t* output,
                                     size_t size,
                                     hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_merge_inplace_int(void* temporary_storage,
                                     size_t* storage_size,
                                     int* data,
                                     size_t left_size,
                                     size_t right_size,
                                     hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_binary_search_int(void* temporary_storage,
                                     size_t* storage_size,
                                     const int* haystack,
                                     const int* needles,
                                     size_t* output,
                                     size_t haystack_size,
                                     size_t needles_size,
                                     hipStream_t stream,
                                     bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_find_end_int(void* temporary_storage,
                                  size_t* storage_size,
                                  const int* input,
                                  const int* keys,
                                  size_t* output,
                                  size_t size,
                                  size_t keys_size,
                                  hipStream_t stream,
                                  bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_find_first_of_int(void* temporary_storage,
                                     size_t* storage_size,
                                     const int* input,
                                     const int* keys,
                                     size_t* output,
                                     size_t size,
                                     size_t keys_size,
                                     hipStream_t stream,
                                     bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_merge_sort_keys_int(void* temporary_storage,
                                       size_t* storage_size,
                                       const int* keys_input,
                                       int* keys_output,
                                       size_t size,
                                       hipStream_t stream,
                                       bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_nth_element_int(void* temporary_storage,
                                   size_t* storage_size,
                                   const int* keys_input,
                                   int* keys_output,
                                   size_t nth,
                                   size_t size,
                                   hipStream_t stream,
                                   bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_partial_sort_copy_int(
    void* temporary_storage,      // Temp storage buffer (nullptr for query)
    size_t* storage_size,         // Output: Required temp size (queried from radix_sort for prefix)
    const int* keys_input,        // Device input (full array)
    int* keys_output,             // Device output (sorted prefix + original tail)
    size_t middle,                // Middle index (prefix_length = middle + 1)
    size_t size_,                 // Total array size
    hipStream_t stream,           // Stream for async execution
    bool debug_synchronous        // If true, synchronize after operations
);
PASROCM_EXPORT hipError_t rocprim_radix_sort_keys_int_asc(void* temporary_storage,
                                           size_t* storage_size,
                                           const int* keys_input,
                                           int* keys_output,
                                           size_t size,
                                           unsigned int begin_bit,
                                           unsigned int end_bit,
                                           hipStream_t stream,
                                           bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_radix_sort_pairs_int(
    void* temporary_storage,      // Temp for rocPRIM large n (ignored for small n)
    size_t* storage_size,         // Output: RocPRIM temp size (0 for small n)
    const int* keys_input,        // Device input keys (const)
    int* keys_output,             // Device output keys (sorted)
    const int* values_input,      // Device input values
    int* values_output,           // Device output values (sorted by key)
    size_t num_items,             // Number of pairs
    hipStream_t stream            // Stream for async
);
PASROCM_EXPORT hipError_t rocprim_partial_sort_int(void* temporary_storage,
                                    size_t* storage_size,
                                    int* keys,
                                    size_t middle,
                                    size_t size,
                                    hipStream_t stream,
                                    bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_run_length_encode_int(void* temporary_storage,
                                         size_t* storage_size,
                                         const int* input,
                                         int* unique_output,
                                         int* counts_output,
                                         int* runs_count_output,
                                         unsigned int size,
                                         hipStream_t stream,
                                         bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_run_length_encode_non_trivial_runs_int(void* temporary_storage,
                                                        size_t* storage_size,
                                                        const int* input,
                                                        int* offsets_output,
                                                        int* counts_output,
                                                        int* runs_count_output,
                                                        unsigned int size,
                                                        hipStream_t stream,
                                                        bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_search_int(void* temporary_storage,
                              size_t* storage_size,
                              const int* haystack,
                              const int* needle,
                              int* output,
                              size_t haystack_size,
                              size_t needle_size,
                              hipStream_t stream,
                              bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_search_n_int(
    void* temporary_storage,
    size_t* storage_size,
    const int* haystack,
    size_t* output,
    size_t haystack_size,
    size_t count,
    int value,
    rocprim_binary_op op,
    hipStream_t stream,
    bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_segmented_reduce_int(void* temporary_storage,
                                        size_t* storage_size,
                                        const int* input,
                                        int* output,
                                        unsigned int segments,
                                        const int* begin_offsets,
                                        const int* end_offsets,
                                        int initial_value,
                                        hipStream_t stream,
                                        bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_segmented_inclusive_scan_int(void* temporary_storage,
                                                size_t* storage_size,
                                                const int* input,
                                                int* output,
                                                unsigned int segments,
                                                const int* begin_offsets,
                                                const int* end_offsets,
                                                hipStream_t stream,
                                                bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_segmented_exclusive_scan_int(void* temporary_storage,
                                                size_t* storage_size,
                                                const int* input,
                                                int* output,
                                                unsigned int segments,
                                                const int* begin_offsets,
                                                const int* end_offsets,
                                                int initial_value,
                                                hipStream_t stream,
                                                bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_segmented_radix_sort_keys_int(void* temporary_storage,
                                                 size_t* storage_size,
                                                 const int* keys_input,
                                                 int* keys_output,
                                                 unsigned int size,
                                                 unsigned int segments,
                                                 const int* begin_offsets,
                                                 const int* end_offsets,
                                                 unsigned int begin_bit,
                                                 unsigned int end_bit,
                                                 hipStream_t stream,
                                                 bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_segmented_radix_sort_pairs_int(void* temporary_storage,
                                                  size_t* storage_size,
                                                  const int* keys_input,
                                                  int* keys_output,
                                                  const int* values_input,
                                                  int* values_output,
                                                  unsigned int size,
                                                  unsigned int segments,
                                                  const int* begin_offsets,
                                                  const int* end_offsets,
                                                  unsigned int begin_bit,
                                                  unsigned int end_bit,
                                                  hipStream_t stream,
                                                  bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_radix_sort_keys_desc_int(void* temporary_storage,
                                            size_t* storage_size,
                                            const int* keys_input,
                                            int* keys_output,
                                            unsigned int size,
                                            unsigned int begin_bit,
                                            unsigned int end_bit,
                                            hipStream_t stream,
                                            bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_radix_sort_pairs_desc_int(void* temporary_storage,
                                             size_t* storage_size,
                                             const int* keys_input,
                                             int* keys_output,
                                             const int* values_input,
                                             int* values_output,
                                             unsigned int size,
                                             unsigned int begin_bit,
                                             unsigned int end_bit,
                                             hipStream_t stream,
                                             bool debug_synchronous);
PASROCM_EXPORT hipError_t rocprim_segmented_radix_sort_keys_desc_int(void* temporary_storage,
                                                      size_t* storage_size,
                                                      const int* keys_input,
                                                      int* keys_output,
                                                      unsigned int size,
                                                      unsigned int segments,
                                                      const int* begin_offsets,
                                                      const int* end_offsets,
                                                      unsigned int begin_bit,
                                                      unsigned int end_bit,
                                                      hipStream_t stream,
                                                      bool debug_synchronous);
PASROCM_EXPORT int rocprim_numeric_limits_int_min();
PASROCM_EXPORT int rocprim_numeric_limits_int_max();
PASROCM_EXPORT float rocprim_numeric_limits_float_epsilon();
PASROCM_EXPORT float rocprim_numeric_limits_float_infinity();
PASROCM_EXPORT float rocprim_numeric_limits_float_nan();
PASROCM_EXPORT hipError_t rocprim_argmax_float(
    const float* in, float* out, long cols, long rows, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_gt_float(const float* a, const float* b, float* out,
    size_t a_n, size_t b_n, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_lt_float(const float* a, const float* b, float* out,
    size_t a_n, size_t b_n, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_eq_float(const float* a, const float* b, float* out,
    size_t a_n, size_t b_n, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_ge_float(const float* a, const float* b, float* out,
    size_t a_n, size_t b_n, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_le_float(const float* a, const float* b, float* out,
    size_t a_n, size_t b_n, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_ne_float(const float* a, const float* b, float* out,
    size_t a_n, size_t b_n, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_where_float(
    const float* cond, const float* a, const float* b, float* out,
    size_t c_n, size_t a_n, size_t b_n, size_t n, hipStream_t stream);
PASROCM_EXPORT hipError_t rocprim_maskedfill_float(
    float* data, const float* mask, float value,
    size_t m_n, size_t n, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PASROCM_ROCprim_SHIM_H */
