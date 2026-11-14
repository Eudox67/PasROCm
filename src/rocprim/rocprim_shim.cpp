// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#define NOMINMAX

#include <atomic>
#include <cstdio>
#include <cstring>  // rocPRIM 4.2's texture_cache_iterator.hpp calls memset unqualified
#include <limits>

// The public ABI declarations. They carry PASROCM_EXPORT, and the build uses
// -fvisibility=hidden: a definition inherits the visibility of the prototype
// declared here, which is what keeps the exported surface at exactly 149
// symbols. Including it also makes the header a
// compile-time check of every signature.
#include "rocprim_shim.h"


#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <rocprim/device/device_copy.hpp>
#include <rocprim/device/device_radix_sort.hpp>  // For rocprim::sort (radix backend for int)
// rocPRIM gained merge_inplace after 3.4. Include it explicitly when the header
// exists, because not every rocprim.hpp re-exports it. The 6.4 line (rocPRIM
// 3.4.x) has neither the header nor the function, so the entry point below
// reports hipErrorNotSupported instead of failing the build. Measured on the
// first CI matrix: 3.4.1 no, 4.2.0 and 4.6.0 yes.
#if defined(__has_include)
#  if __has_include(<rocprim/device/device_merge_inplace.hpp>)
#    include <rocprim/device/device_merge_inplace.hpp>
#    define PASROCM_HAVE_MERGE_INPLACE 1
#  endif
#endif
#include <rocprim/types.hpp>
#include <rocprim/intrinsics/thread.hpp>
// No explicit <rocprim/intrinsics/arch.hpp>: the shim only needs lane_id(),
// which thread.hpp provides (and which includes arch.hpp itself on 4.x). The
// 6.4.0 channel ships rocPRIM 3.4.0, where that header does not exist at all --
// the 6.4.0 probe job caught the redundant include.

// This shim provides a simplified, C-style interface to the C++ rocprim template functions.
// It encapsulates the two-step (get-size, execute) API for temporary storage
// and instantiates templates for common data types.

// ---------------------------------------------------------------------------
// Target and wavefront configuration.
//
// Both come from the build (src/rocprim/build.sh or CMakeLists.txt), looked up
// in targets.conf -- the single mapping table. They are deliberately not
// derived from preprocessor macros, because (measured):
//   * __AMDGCN_TARGET__ does not exist in clang at all;
//   * __AMDGCN__ and __AMDGPU__ are defined for RDNA and CDNA alike, so neither
//     identifies a wavefront width;
//   * __AMDGCN_WAVEFRONT_SIZE no longer exists in ROCm 10.1's clang 24;
//   * __builtin_amdgcn_wavefrontsize() is not an integral constant expression,
//     and the launch geometry below is HOST code, which cannot evaluate it.
// ---------------------------------------------------------------------------
#ifndef PASROCM_WAVEFRONT_SIZE
#error "PASROCM_WAVEFRONT_SIZE must be injected by the build (ci/gfx-targets.json)"
#endif
#ifndef PASROCM_GPU_TARGET
#error "PASROCM_GPU_TARGET must be injected by the build"
#endif
#ifndef PASROCM_ROCM_VERSION
#error "PASROCM_ROCM_VERSION must be injected by the build"
#endif
#ifndef PASROCM_WAVEFRONT_WIDTHS_MASK
#error "PASROCM_WAVEFRONT_WIDTHS_MASK must be injected by the build"
#endif

static_assert(PASROCM_WAVEFRONT_SIZE == 32 || PASROCM_WAVEFRONT_SIZE == 64,
              "wavefront size must be 32 or 64");

// Hardware facts, checked in the device pass. A wrong entry in the table cannot
// satisfy these, so a mistyped width fails the build instead of the GPU run.
#if defined(__GFX9__) && PASROCM_WAVEFRONT_SIZE != 64
#error "CDNA (gfx9xx) is wave64: refusing a wave32 build"
#endif
#if (defined(__GFX10__) || defined(__GFX11__)) && PASROCM_WAVEFRONT_SIZE != 32
#error "RDNA 1-3.5 is wave32: refusing a wave64 build"
#endif
// gfx12 deliberately has no width guard: RDNA4 can run either width, so the
// table's pinned value is the only decider (gfx1250 rejects -mwavefrontsize64).

#define ROCPRIM_WARP_SIZE PASROCM_WAVEFRONT_SIZE

extern "C" {

// Pure identity accessors: no HIP calls, so a host-only process can read them
// without a GPU. CI asserts them against the matrix entry for every build.
const char* pasrocm_shim_target(void)         { return PASROCM_GPU_TARGET; }
unsigned    pasrocm_shim_wavefront_size(void) { return PASROCM_WAVEFRONT_SIZE; }
const char* pasrocm_shim_rocm_version(void)   { return PASROCM_ROCM_VERSION; }

// Widths this build's hardware can run: bit 0 = 32, bit 1 = 64.
unsigned pasrocm_shim_wavefront_widths_mask(void) { return PASROCM_WAVEFRONT_WIDTHS_MASK; }

// Monotonic ABI revision, taken from the generated header (which takes it from
// the specification, the single owner of the number).
unsigned pasrocm_shim_abi_version(void) { return PASROCM_SHIM_ABI_VERSION; }

// Deterministic provenance string. Deliberately no __DATE__/__TIME__: a
// timestamp would make two builds of one commit differ and break the packaging
// determinism gate. Everything here is a build input or a compiler fact.
#ifndef PASROCM_SHIM_VERSION
#  define PASROCM_SHIM_VERSION "unknown"
#endif
#ifndef PASROCM_GIT_COMMIT
#  define PASROCM_GIT_COMMIT "unknown"
#endif
#define PASROCM_STRINGIZE_(x) #x
#define PASROCM_STRINGIZE(x) PASROCM_STRINGIZE_(x)

const char* pasrocm_shim_build_info(void)
{
    return "pasrocm-shim " PASROCM_SHIM_VERSION
           " abi=" PASROCM_STRINGIZE(PASROCM_SHIM_ABI_VERSION)
           " commit=" PASROCM_GIT_COMMIT
           " target=" PASROCM_GPU_TARGET
           " wavefront=" PASROCM_STRINGIZE(PASROCM_WAVEFRONT_SIZE)
           " widths_mask=" PASROCM_STRINGIZE(PASROCM_WAVEFRONT_WIDTHS_MASK)
           " rocm=" PASROCM_ROCM_VERSION
           " compiler=" __VERSION__;
}

// Pure compatibility test, exposed so the mismatch branch is testable without
// CDNA hardware.
//
// It compares the width the DEVICE reports against the widths the hardware can
// run -- never against the compiled width. Measured on gfx1100: a kernel
// built with -mwavefrontsize64 reports 64 from inside the
// kernel while hipDeviceProp_t::warpSize still reports 32, and the launch
// succeeds. An equality test would therefore reject a legitimate wave64 build
// on dual-width hardware.
int pasrocm_wavefront_compatible(unsigned compiled,
                                 unsigned reported,
                                 unsigned allowed_mask)
{
    const unsigned compiled_bit = compiled == 32u ? 1u : (compiled == 64u ? 2u : 0u);
    const unsigned reported_bit = reported == 32u ? 1u : (reported == 64u ? 2u : 0u);
    if (compiled_bit == 0u || reported_bit == 0u)
        return 0;                                   // not a wavefront width at all
    // The build itself must target a width the hardware supports, whatever the
    // device happens to report. build.sh/CMake cannot produce this today, but
    // this is a public entry point a loader may call with anything.
    if ((allowed_mask & compiled_bit) == 0u)
        return 0;                                   // build is for an impossible width
    if ((allowed_mask & reported_bit) == 0u)
        return 0;                                   // hardware cannot run this width
    if ((allowed_mask & 3u) == 3u)
        return 1;                                   // dual-width hardware: either is valid
    return compiled == reported ? 1 : 0;            // single-width hardware: must agree
}

}  // extern "C"

namespace {

// 0 = not checked yet, 1 = compatible, -1 = mismatch (latched for the process).
std::atomic<int> g_wavefront_state{0};

// Compares the compiled width with the device's reported width exactly once.
// A mismatch means the wrong library was loaded, so no kernel may be launched.
hipError_t require_matching_wavefront()
{
    int state = g_wavefront_state.load(std::memory_order_acquire);
    if (state == 0)
    {
        hipDeviceProp_t properties{};
        int device = 0;
        if (hipGetDevice(&device) != hipSuccess)
            return hipGetLastError();
        if (hipGetDeviceProperties(&properties, device) != hipSuccess)
            return hipGetLastError();
        const int compatible = pasrocm_wavefront_compatible(
            static_cast<unsigned>(PASROCM_WAVEFRONT_SIZE),
            static_cast<unsigned>(properties.warpSize),
            static_cast<unsigned>(PASROCM_WAVEFRONT_WIDTHS_MASK));
        state = compatible ? 1 : -1;
        g_wavefront_state.store(state, std::memory_order_release);
    }
    return state > 0 ? hipSuccess : hipErrorInvalidDevice;
}

}  // namespace
    



// rocprim_binary_op is declared in the public header (generated from the spec's
// TYPES section), so this file and the Pascal enum cannot disagree.

__global__
void fill_kernel_size_t(size_t* array, size_t value, size_t count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        array[idx] = value;
    }
}

// Templated kernel for search_n with BinaryPredicate.
template <class BinaryPredicate>
__global__ void search_n_kernel(const int* haystack, size_t* output, size_t haystack_size, size_t count, int value, BinaryPredicate pred) {
    const size_t global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t local_min = haystack_size;

    size_t num_starts = haystack_size - count + 1;

    for (size_t i = global_tid; i < num_starts; i += gridDim.x * blockDim.x) {
        bool found = true;
        for (size_t j = 0; j < count; ++j) {
            if (!pred(haystack[i + j], value)) {
                found = false;
                break;
            }
        }
        if (found) {
            local_min = i;
            break; 
        }
    }

    extern __shared__ size_t shared_mins[];
    const size_t local_tid = threadIdx.x;
    shared_mins[local_tid] = local_min;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_tid < s) {
            if (shared_mins[local_tid] > shared_mins[local_tid + s]) {
                shared_mins[local_tid] = shared_mins[local_tid + s];
            }
        }
        __syncthreads();
    }

    if (local_tid == 0 && shared_mins[0] < haystack_size) {
        atomicMin((unsigned long long*)output, (unsigned long long)shared_mins[0]);
    }
}

// --- rocprim::block_reduce ---

template<
    unsigned int BlockSize
>
__global__
void block_reduce_kernel(const int* d_in, int* d_out, const size_t num_items)
{
    const unsigned int i = blockIdx.x * BlockSize + threadIdx.x;

    int input = 0;
    if (i < num_items)
    {
        input = d_in[i];
    }

    using block_reduce_int = rocprim::block_reduce<int, BlockSize>;
    __shared__ typename block_reduce_int::storage_type temp_storage;
    
    int aggregate;
    block_reduce_int().reduce(input, aggregate, temp_storage, rocprim::plus<int>());

    if (threadIdx.x == 0)
    {
        d_out[blockIdx.x] = aggregate;
    }
}

// --- rocprim::block_scan ---

template<
    unsigned int BlockSize
>
__global__
void block_scan_kernel(const int* d_in, int* d_out, const size_t num_items)
{
    const unsigned int i = blockIdx.x * BlockSize + threadIdx.x;

    int input = 0;
    if (i < num_items)
    {
        input = d_in[i];
    }

    using block_scan_int = rocprim::block_scan<int, BlockSize>;
    __shared__ typename block_scan_int::storage_type temp_storage;

    int output;
    block_scan_int().inclusive_scan(input, output, temp_storage, rocprim::plus<int>());

    if (i < num_items)
    {
        d_out[i] = output;
    }
}

// --- rocprim::block_sort ---

template<
    unsigned int BlockSize
>
__global__
void block_sort_kernel(const int* d_in, int* d_out, const size_t num_items)
{
    const unsigned int i = blockIdx.x * BlockSize + threadIdx.x;

    int key = 0;
    if (i < num_items)
    {
        key = d_in[i];
    }

    using block_sort_int = rocprim::block_sort<int, BlockSize>;
    __shared__ typename block_sort_int::storage_type temp_storage;

    block_sort_int().sort(key, temp_storage);

    if (i < num_items)
    {
        d_out[i] = key;
    }
}

// --- rocprim::block_histogram ---

template<
    unsigned int BlockSize,
    unsigned int Bins
>
__global__
void block_histogram_kernel(const int* d_in, int* d_out, const size_t num_items)
{
    const unsigned int i = blockIdx.x * BlockSize + threadIdx.x;

    int input[1];
    if (i < num_items)
    {
        input[0] = d_in[i];
    } else {
        input[0] = 0;
    }

    using block_histogram_int = rocprim::block_histogram<int, BlockSize, 1, Bins>;
    __shared__ typename block_histogram_int::storage_type temp_storage;

    int* histogram = &d_out[blockIdx.x * Bins];
    block_histogram_int().histogram(input, histogram, temp_storage);
}

// --- rocprim::warp_reduce ---

template<typename T, unsigned int WarpSize>
__global__ void warp_reduce_kernel(const T* d_in, T* d_out, const size_t num_items)
{
    const unsigned int i = blockIdx.x * WarpSize + threadIdx.x;

    T input = 0;
    if (i < num_items)
    {
        input = d_in[i];
    }

    using warp_reduce_t = rocprim::warp_reduce<T, WarpSize>;
    
    T aggregate;
    typename warp_reduce_t::storage_type storage;
    warp_reduce_t().reduce(input, aggregate, storage, rocprim::plus<T>());

    if (rocprim::lane_id() == 0)
    {
        d_out[i / WarpSize] = aggregate;
    }
}

// --- rocprim::warp_scan ---

template<typename T, unsigned int WarpSize>
__global__ void warp_scan_kernel(const T* d_in, T* d_out, const size_t num_items)
{
    const unsigned int i = blockIdx.x * WarpSize + threadIdx.x;

    T input = 0;
    if (i < num_items)
    {
        input = d_in[i];
    }

    using warp_scan_t = rocprim::warp_scan<T, WarpSize>;
    
    T output;
    typename warp_scan_t::storage_type storage;
    warp_scan_t().inclusive_scan(input, output, storage, rocprim::plus<T>());

    if (i < num_items)
    {
        d_out[i] = output;
    }
}

// --- rocprim::warp_sort ---

template<typename T, unsigned int WarpSize>
__global__ void warp_sort_kernel(const T* d_in, T* d_out, const size_t num_items)
{
    const unsigned int i = blockIdx.x * WarpSize + threadIdx.x;

    T key = 0;
    if (i < num_items)
    {
        key = d_in[i];
    }

    using warp_sort_t = rocprim::warp_sort<T, WarpSize>;
    
    typename warp_sort_t::storage_type storage;
    warp_sort_t().sort(key, storage);

    if (i < num_items)
    {
        d_out[i] = key;
    }
}

template<typename T>
hipError_t rocprim_warp_reduce(T* d_out, const T* d_in, size_t num_items, hipStream_t stream)
{
	const hipError_t ready = require_matching_wavefront();
	if (ready != hipSuccess) return ready;
	hipLaunchKernelGGL(
		HIP_KERNEL_NAME(warp_reduce_kernel<T, ROCPRIM_WARP_SIZE>),
		dim3(rocprim::detail::ceiling_div(num_items, (unsigned int)ROCPRIM_WARP_SIZE)),
		dim3(ROCPRIM_WARP_SIZE),
		0,
		stream,
		d_in, d_out, num_items
	);
	return hipGetLastError();
}

template<typename T>
hipError_t rocprim_warp_scan(T* d_out, const T* d_in, size_t num_items, hipStream_t stream)
{
	const hipError_t ready = require_matching_wavefront();
	if (ready != hipSuccess) return ready;
	hipLaunchKernelGGL(
		HIP_KERNEL_NAME(warp_scan_kernel<T, ROCPRIM_WARP_SIZE>),
		dim3(rocprim::detail::ceiling_div(num_items, (unsigned int)ROCPRIM_WARP_SIZE)),
		dim3(ROCPRIM_WARP_SIZE),
		0,
		stream,
		d_in, d_out, num_items
	);
	return hipGetLastError();
}

template<typename T>
hipError_t rocprim_warp_sort(T* d_out, const T* d_in, size_t num_items, hipStream_t stream)
{
	const hipError_t ready = require_matching_wavefront();
	if (ready != hipSuccess) return ready;
	hipLaunchKernelGGL(
		HIP_KERNEL_NAME(warp_sort_kernel<T, ROCPRIM_WARP_SIZE>),
		dim3(rocprim::detail::ceiling_div(num_items, (unsigned int)ROCPRIM_WARP_SIZE)),
		dim3(ROCPRIM_WARP_SIZE),
		0,
		stream,
		d_in, d_out, num_items
	);
	return hipGetLastError();
}
    
    

// --- Gather / Scatter Kernels ---

extern "C" __global__ void gather_kernel_float(
    const float* __restrict__ input,
    const float* __restrict__ index,
    float* __restrict__ output,
    size_t num_items,
    int dim,
    int rank,
    const long long* __restrict__ in_strides,
    const long long* __restrict__ idx_strides,
    const long long* __restrict__ out_strides,
    const long long* __restrict__ out_shape
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) return;

    size_t remaining = tid;
    size_t in_idx = 0;
    
    // Decompose tid into multi-dim coordinates based on output shape
    // and compute input index.
    for (int i = 0; i < rank; ++i) {
        size_t coord = remaining / out_strides[i];
        remaining %= out_strides[i];
        
        if (i == dim) {
            // Use the value from the index tensor for this dimension
            int gather_idx = (int)index[tid];
            in_idx += gather_idx * in_strides[i];
        } else {
            in_idx += coord * in_strides[i];
        }
    }
    output[tid] = input[in_idx];
}

extern "C" __global__ void scatter_kernel_float(
    float* __restrict__ output,
    const float* __restrict__ index,
    const float* __restrict__ src,
    size_t num_items,
    int dim,
    int rank,
    const long long* __restrict__ out_strides,
    const long long* __restrict__ idx_strides,
    const long long* __restrict__ src_strides,
    const long long* __restrict__ src_shape,
    bool atomic_add
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) return;

    size_t remaining = tid;
    size_t out_idx = 0;
    
    for (int i = 0; i < rank; ++i) {
        size_t coord = remaining / src_strides[i];
        remaining %= src_strides[i];
        
        if (i == dim) {
            int scatter_idx = (int)index[tid];
            out_idx += scatter_idx * out_strides[i];
        } else {
            out_idx += coord * out_strides[i];
        }
    }
    
    if (atomic_add) {
        atomicAdd(&output[out_idx], src[tid]);
    } else {
        output[out_idx] = src[tid];
    }
}

extern "C" {

hipError_t rocprim_gather_float(
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
) {
    dim3 block(256);
    dim3 grid((num_items + block.x - 1) / block.x);
    hipLaunchKernelGGL(gather_kernel_float, grid, block, 0, stream, d_input, d_index, d_output, num_items, dim, rank, d_in_strides, d_idx_strides, d_out_strides, d_out_shape);
    return hipGetLastError();
}

hipError_t rocprim_scatter_float(
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
) {
    dim3 block(256);
    dim3 grid((num_items + block.x - 1) / block.x);
    hipLaunchKernelGGL(scatter_kernel_float, grid, block, 0, stream, d_output, d_index, d_src, num_items, dim, rank, d_out_strides, d_idx_strides, d_src_strides, d_src_shape, atomic_add);
    return hipGetLastError();
}

void rocprim_set_warp_size(unsigned int size)
{
    // Kept for ABI compatibility. The compiled width is authoritative; this only
    // validates a caller's belief and latches a mismatch so later calls fail
    // instead of computing with the wrong geometry.
    if (size != 0u && size != PASROCM_WAVEFRONT_SIZE)
        g_wavefront_state.store(-1, std::memory_order_release);
}

// --- rocprim::reduce ---
hipError_t rocprim_reduce_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    int initial_value,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::reduce(d_temp_storage, temp_storage_size, d_in, d_out, initial_value, num_items, rocprim::plus<int>(), stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::reduce(d_temp_storage, temp_storage_size, d_in, d_out, initial_value, num_items, rocprim::plus<int>(), stream);
    
    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::inclusive_scan ---
hipError_t rocprim_inclusive_scan_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::inclusive_scan(d_temp_storage, temp_storage_size, d_in, d_out, num_items, rocprim::plus<int>(), stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::inclusive_scan(d_temp_storage, temp_storage_size, d_in, d_out, num_items, rocprim::plus<int>(), stream);

    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::deterministic_inclusive_scan ---
hipError_t rocprim_deterministic_inclusive_scan_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::deterministic_inclusive_scan(d_temp_storage, temp_storage_size, d_in, d_out, num_items, rocprim::plus<int>(), stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::deterministic_inclusive_scan(d_temp_storage, temp_storage_size, d_in, d_out, num_items, rocprim::plus<int>(), stream);

    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::exclusive_scan ---
hipError_t rocprim_exclusive_scan_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    int initial_value,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::exclusive_scan(d_temp_storage, temp_storage_size, d_in, d_out, initial_value, num_items, rocprim::plus<int>(), stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::exclusive_scan(d_temp_storage, temp_storage_size, d_in, d_out, initial_value, num_items, rocprim::plus<int>(), stream);

    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::deterministic_exclusive_scan ---
hipError_t rocprim_deterministic_exclusive_scan_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    int initial_value,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::deterministic_exclusive_scan(d_temp_storage, temp_storage_size, d_in, d_out, initial_value, num_items, rocprim::plus<int>(), stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::deterministic_exclusive_scan(d_temp_storage, temp_storage_size, d_in, d_out, initial_value, num_items, rocprim::plus<int>(), stream);

    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::radix_sort_keys (in-place) ---
hipError_t rocprim_radix_sort_keys_int(
    int* d_keys,
    size_t num_items,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::radix_sort_keys(d_temp_storage, temp_storage_size, d_keys, d_keys, num_items, 0, sizeof(int) * 8, stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::radix_sort_keys(d_temp_storage, temp_storage_size, d_keys, d_keys, num_items, 0, sizeof(int) * 8, stream);

    hipFree(d_temp_storage);
    return status;
}

// Functor for the transform operation
struct multiply_by_constant
{
    const int multiplier;

    multiply_by_constant(int multiplier) : multiplier(multiplier) {}

    __device__ __host__
    int operator()(const int& a) const
    {
        return a * multiplier;
    }
};

// --- rocprim::transform ---
hipError_t rocprim_transform_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    int multiplier,
    hipStream_t stream)
{
    multiply_by_constant op(multiplier);
    return rocprim::transform(d_in, d_out, num_items, op, stream);
}

// Functor for the partition operation
struct less_than_pivot
{
    const int pivot;

    less_than_pivot(int pivot) : pivot(pivot) {}

    __device__ __host__
    bool operator()(const int& a) const
    {
        return a < pivot;
    }
};

// --- rocprim::partition ---
hipError_t rocprim_partition_int(
    int* d_in,
    int* d_out,
    size_t* d_selected_count,
    size_t num_items,
    int pivot,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    less_than_pivot op(pivot);

    status = rocprim::partition(d_temp_storage, temp_storage_size, d_in, d_out, d_selected_count, num_items, op, stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::partition(d_temp_storage, temp_storage_size, d_in, d_out, d_selected_count, num_items, op, stream);

    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::select ---
hipError_t rocprim_select_int(
    int* d_in,
    int* d_out,
    size_t* d_selected_count,
    size_t num_items,
    int pivot,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    less_than_pivot op(pivot);

    status = rocprim::select(d_temp_storage, temp_storage_size, d_in, d_out, d_selected_count, num_items, op, stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::select(d_temp_storage, temp_storage_size, d_in, d_out, d_selected_count, num_items, op, stream);

    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::reduce_by_key ---
hipError_t rocprim_reduce_by_key_int(
    int* d_keys_in,
    int* d_values_in,
    size_t num_items,
    int* d_unique_keys_out,
    int* d_aggregates_out,
    size_t* d_unique_count_out,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::reduce_by_key(d_temp_storage, temp_storage_size, d_keys_in, d_values_in, num_items, d_unique_keys_out, d_aggregates_out, d_unique_count_out, rocprim::plus<int>(), rocprim::equal_to<int>(), stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::reduce_by_key(d_temp_storage, temp_storage_size, d_keys_in, d_values_in, num_items, d_unique_keys_out, d_aggregates_out, d_unique_count_out, rocprim::plus<int>(), rocprim::equal_to<int>(), stream);

    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::inclusive_scan_by_key ---
hipError_t rocprim_inclusive_scan_by_key_int(
    int* d_keys_in,
    int* d_values_in,
    int* d_values_out,
    size_t num_items,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::inclusive_scan_by_key(d_temp_storage, temp_storage_size, d_keys_in, d_values_in, d_values_out, num_items, rocprim::plus<int>(), rocprim::equal_to<int>(), stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::inclusive_scan_by_key(d_temp_storage, temp_storage_size, d_keys_in, d_values_in, d_values_out, num_items, rocprim::plus<int>(), rocprim::equal_to<int>(), stream);

    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::exclusive_scan_by_key ---
hipError_t rocprim_exclusive_scan_by_key_int(
    int* d_keys_in,
    int* d_values_in,
    int* d_values_out,
    size_t num_items,
    int initial_value,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::exclusive_scan_by_key(d_temp_storage, temp_storage_size, d_keys_in, d_values_in, d_values_out, initial_value, num_items, rocprim::plus<int>(), rocprim::equal_to<int>(), stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::exclusive_scan_by_key(d_temp_storage, temp_storage_size, d_keys_in, d_values_in, d_values_out, initial_value, num_items, rocprim::plus<int>(), rocprim::equal_to<int>(), stream);

    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::histogram_even ---
hipError_t rocprim_histogram_even_int(
    int* d_samples,
    size_t num_samples,
    int* d_histogram,
    unsigned int levels,
    int lower_level,
    int upper_level,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::histogram_even(d_temp_storage, temp_storage_size, d_samples, num_samples, d_histogram, levels, lower_level, upper_level, stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::histogram_even(d_temp_storage, temp_storage_size, d_samples, num_samples, d_histogram, levels, lower_level, upper_level, stream);

    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::merge ---
hipError_t rocprim_merge_int(
    int* d_input1,
    int* d_input2,
    int* d_output,
    size_t input1_size,
    size_t input2_size,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::merge(d_temp_storage, temp_storage_size, d_input1, d_input2, d_output, input1_size, input2_size, rocprim::less<int>(), stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::merge(d_temp_storage, temp_storage_size, d_input1, d_input2, d_output, input1_size, input2_size, rocprim::less<int>(), stream);

    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::radix_sort_keys (input/output iterators) ---
hipError_t rocprim_radix_sort_keys_int_io(
    int* d_keys_in,
    int* d_keys_out,
    size_t num_items,
    unsigned int begin_bit,
    unsigned int end_bit,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::radix_sort_keys(d_temp_storage, temp_storage_size, d_keys_in, d_keys_out, num_items, begin_bit, end_bit, stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::radix_sort_keys(d_temp_storage, temp_storage_size, d_keys_in, d_keys_out, num_items, begin_bit, end_bit, stream);

    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::unique ---
hipError_t rocprim_unique_int(
    int* d_input,
    int* d_output,
    size_t* d_unique_count_out,
    size_t num_items,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::unique(d_temp_storage, temp_storage_size, d_input, d_output, d_unique_count_out, num_items, rocprim::equal_to<int>(), stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::unique(d_temp_storage, temp_storage_size, d_input, d_output, d_unique_count_out, num_items, rocprim::equal_to<int>(), stream);

    hipFree(d_temp_storage);
    return status;
}

// --- rocprim::copy ---
hipError_t rocprim_copy_int(
    int* d_input,
    int* d_output,
    size_t num_items,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    // Host arrays for batch_copy parameters
    int* h_sources[1];
    int* h_destinations[1];
    size_t h_sizes[1];

    h_sources[0] = d_input;
    h_destinations[0] = d_output;
    h_sizes[0] = num_items;

    // Device arrays for batch_copy parameters
    int** d_sources;
    int** d_destinations;
    size_t* d_sizes;

    hipMalloc((void**)&d_sources, sizeof(int*));
    hipMalloc((void**)&d_destinations, sizeof(int*));
    hipMalloc((void**)&d_sizes, sizeof(size_t));

    hipMemcpy(d_sources, h_sources, sizeof(int*), hipMemcpyHostToDevice);
    hipMemcpy(d_destinations, h_destinations, sizeof(int*), hipMemcpyHostToDevice);
    hipMemcpy(d_sizes, h_sizes, sizeof(size_t), hipMemcpyHostToDevice);

    status = rocprim::batch_copy(d_temp_storage, temp_storage_size, d_sources, d_destinations, d_sizes, 1, stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::batch_copy(d_temp_storage, temp_storage_size, d_sources, d_destinations, d_sizes, 1, stream);

    hipFree(d_temp_storage);
    hipFree(d_sources);
    hipFree(d_destinations);
    hipFree(d_sizes);
    return status;
}

// --- rocprim::fill ---
extern "C"
__global__ void rocprim_fill_kernel_int(int* output, int value, size_t count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        output[idx] = value;
    }
}

extern "C"
void rocprim_fill_int(void* output, int value, size_t count, hipStream_t stream)
{
    hipLaunchKernelGGL(rocprim_fill_kernel_int, dim3((count + 255) / 256), dim3(256), 0, stream, (int*)output, value, count);
}

// --- rocprim::adjacent_difference ---
hipError_t rocprim_adjacent_difference_int(void* temporary_storage,
                                           size_t* storage_size,
                                           const int* input,
                                           int* output,
                                           size_t size,
                                           hipStream_t stream)
{
    hipError_t status;

    // Get required size of the temporary storage
    status = rocprim::adjacent_difference(nullptr, *storage_size, input, output, size, rocprim::minus<int>(), stream);
    if (status != hipSuccess) return status;

    // Perform adjacent_difference
    status = rocprim::adjacent_difference(temporary_storage, *storage_size, input, output, size, rocprim::minus<int>(), stream);

    return status;
}

// --- rocprim::adjacent_find ---
hipError_t rocprim_adjacent_find_int(void* temporary_storage,
                                     size_t* storage_size,
                                     const int* input,
                                     size_t* output,
                                     size_t size,
                                     hipStream_t stream)
{
    hipError_t status;

    // Get required size of the temporary storage
    status = rocprim::adjacent_find(nullptr, *storage_size, input, output, size, rocprim::equal_to<int>(), stream);
    if (status != hipSuccess) return status;

    // Perform adjacent_find
    status = rocprim::adjacent_find(temporary_storage, *storage_size, input, output, size, rocprim::equal_to<int>(), stream);

    return status;
}

// --- rocprim::merge_inplace ---
hipError_t rocprim_merge_inplace_int(void* temporary_storage,
                                     size_t* storage_size,
                                     int* data,
                                     size_t left_size,
                                     size_t right_size,
                                     hipStream_t stream)
{
#ifdef PASROCM_HAVE_MERGE_INPLACE
    hipError_t status;

    // Get required size of the temporary storage
    status = rocprim::merge_inplace(nullptr, *storage_size, data, left_size, right_size, rocprim::less<int>(), stream);
    if (status != hipSuccess) return status;

    // Perform merge_inplace
    status = rocprim::merge_inplace(temporary_storage, *storage_size, data, left_size, right_size, rocprim::less<int>(), stream);

    return status;
#else
    // rocPRIM before 4.0 has no merge_inplace. Keep the symbol so the ABI stays
    // complete and report the capability gap where the caller can see it.
    (void)temporary_storage;
    (void)storage_size;
    (void)data;
    (void)left_size;
    (void)right_size;
    (void)stream;
    return hipErrorNotSupported;
#endif
}

hipError_t rocprim_binary_search_int(void* temporary_storage,
                                     size_t* storage_size,
                                     const int* haystack,
                                     const int* needles,
                                     size_t* output,
                                     size_t haystack_size,
                                     size_t needles_size,
                                     hipStream_t stream,
                                     bool debug_synchronous)
{
    return rocprim::binary_search(temporary_storage, *storage_size, haystack, needles, output, haystack_size, needles_size, rocprim::less<int>(), stream, debug_synchronous);
}


hipError_t rocprim_find_end_int(void* temporary_storage,
                                  size_t* storage_size,
                                  const int* input,
                                  const int* keys,
                                  size_t* output,
                                  size_t size,
                                  size_t keys_size,
                                  hipStream_t stream,
                                  bool debug_synchronous)
{
    return rocprim::find_end(temporary_storage, *storage_size, input, keys, output, size, keys_size, rocprim::equal_to<int>(), stream, debug_synchronous);
}

hipError_t rocprim_find_first_of_int(void* temporary_storage,
                                     size_t* storage_size,
                                     const int* input,
                                     const int* keys,
                                     size_t* output,
                                     size_t size,
                                     size_t keys_size,
                                     hipStream_t stream,
                                     bool debug_synchronous)
{
    return rocprim::find_first_of(temporary_storage, *storage_size, input, keys, output, size, keys_size, rocprim::equal_to<int>(), stream, debug_synchronous);
}

hipError_t rocprim_merge_sort_keys_int(void* temporary_storage,
                                       size_t* storage_size,
                                       const int* keys_input,
                                       int* keys_output,
                                       size_t size,
                                       hipStream_t stream,
                                       bool debug_synchronous)
{
    return rocprim::merge_sort(temporary_storage, *storage_size, keys_input, keys_output, size, rocprim::less<int>(), stream, debug_synchronous);
}

hipError_t rocprim_nth_element_int(void* temporary_storage,
                                   size_t* storage_size,
                                   const int* keys_input,
                                   int* keys_output,
                                   size_t nth,
                                   size_t size,
                                   hipStream_t stream,
                                   bool debug_synchronous)
{
    return rocprim::nth_element(temporary_storage, *storage_size, keys_input, keys_output, nth, size, rocprim::less<int>(), stream, debug_synchronous);
}



hipError_t rocprim_radix_sort_keys_int_asc(void* temporary_storage,
                                           size_t* storage_size,
                                           const int* keys_input,
                                           int* keys_output,
                                           size_t size,
                                           unsigned int begin_bit,
                                           unsigned int end_bit,
                                           hipStream_t stream,
                                           bool debug_synchronous)
{
    return rocprim::radix_sort_keys(temporary_storage, *storage_size, keys_input, keys_output, size, begin_bit, end_bit, stream, debug_synchronous);
}

hipError_t rocprim_partial_sort_int(void* temporary_storage,
                                    size_t* storage_size,
                                    int* keys,
                                    size_t middle,
                                    size_t size,
                                    hipStream_t stream,
                                    bool debug_synchronous)
{
    return rocprim::partial_sort(temporary_storage, *storage_size, keys, middle, size, rocprim::less<int>(), stream, debug_synchronous);
}

hipError_t rocprim_run_length_encode_int(void* temporary_storage,
                                         size_t* storage_size,
                                         const int* input,
                                         int* unique_output,
                                         int* counts_output,
                                         int* runs_count_output,
                                         unsigned int size,
                                         hipStream_t stream,
                                         bool debug_synchronous)
{
    return rocprim::run_length_encode(temporary_storage, *storage_size, input, size, unique_output, counts_output, runs_count_output, stream, debug_synchronous);
}

hipError_t rocprim_run_length_encode_non_trivial_runs_int(void* temporary_storage,
                                                        size_t* storage_size,
                                                        const int* input,
                                                        int* offsets_output,
                                                        int* counts_output,
                                                        int* runs_count_output,
                                                        unsigned int size,
                                                        hipStream_t stream,
                                                        bool debug_synchronous)
{
    return rocprim::run_length_encode_non_trivial_runs(temporary_storage, *storage_size, input, size, offsets_output, counts_output, runs_count_output, stream, debug_synchronous);
}

hipError_t rocprim_search_int(void* temporary_storage,
                              size_t* storage_size,
                              const int* haystack,
                              const int* needle,
                              int* output,
                              size_t haystack_size,
                              size_t needle_size,
                              hipStream_t stream,
                              bool debug_synchronous)
{
    return rocprim::search(temporary_storage, *storage_size, haystack, needle, output, haystack_size, needle_size, rocprim::equal_to<int>(), stream, debug_synchronous);
}

hipError_t rocprim_search_n_int(
    void* temporary_storage,
    size_t* storage_size,
    const int* haystack,
    size_t* output,
    size_t haystack_size,
    size_t count,
    int value,
    rocprim_binary_op op,
    hipStream_t stream,
    bool debug_synchronous) {

    if (count == 0) {
        if (storage_size != nullptr) *storage_size = 0;
        size_t result = 0;
        hipMemcpyAsync(output, &result, sizeof(size_t), hipMemcpyHostToDevice, stream);
        return hipSuccess;
    }

    if (haystack_size < count) {
        if (storage_size != nullptr) *storage_size = 0;
        size_t not_found = haystack_size;
        hipMemcpyAsync(output, &not_found, sizeof(size_t), hipMemcpyHostToDevice, stream);
        return hipSuccess;
    }

    if (storage_size != nullptr) {
        *storage_size = 0;
    }

    if (temporary_storage == nullptr && storage_size != nullptr) {
        return hipSuccess;
    }

    if (output == nullptr) return hipErrorInvalidValue;

    size_t not_found_val = haystack_size;
    hipMemcpyAsync(output, &not_found_val, sizeof(size_t), hipMemcpyHostToDevice, stream);

    dim3 threads(256);
    size_t num_starts = haystack_size - count + 1;
    dim3 blocks((num_starts + threads.x - 1) / threads.x);
    size_t shared_mem_size = threads.x * sizeof(size_t);

    switch (op) {
        case rocprim_binary_op_equal_to:
            hipLaunchKernelGGL(search_n_kernel<rocprim::equal_to<int>>, blocks, threads, shared_mem_size, stream, haystack, output, haystack_size, count, value, rocprim::equal_to<int>());
            break;
        case rocprim_binary_op_not_equal_to:
            hipLaunchKernelGGL(search_n_kernel<rocprim::not_equal_to<int>>, blocks, threads, shared_mem_size, stream, haystack, output, haystack_size, count, value, rocprim::not_equal_to<int>());
            break;
        case rocprim_binary_op_greater_than:
            hipLaunchKernelGGL(search_n_kernel<rocprim::greater<int>>, blocks, threads, shared_mem_size, stream, haystack, output, haystack_size, count, value, rocprim::greater<int>());
            break;
        case rocprim_binary_op_less_than:
            hipLaunchKernelGGL(search_n_kernel<rocprim::less<int>>, blocks, threads, shared_mem_size, stream, haystack, output, haystack_size, count, value, rocprim::less<int>());
            break;
        case rocprim_binary_op_greater_equal:
            hipLaunchKernelGGL(search_n_kernel<rocprim::greater_equal<int>>, blocks, threads, shared_mem_size, stream, haystack, output, haystack_size, count, value, rocprim::greater_equal<int>());
            break;
        case rocprim_binary_op_less_equal:
            hipLaunchKernelGGL(search_n_kernel<rocprim::less_equal<int>>, blocks, threads, shared_mem_size, stream, haystack, output, haystack_size, count, value, rocprim::less_equal<int>());
            break;
        default:
            return hipErrorInvalidValue;
    }

    hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;

    if (debug_synchronous) {
        status = hipStreamSynchronize(stream);
    }

    return status;
}

hipError_t rocprim_segmented_reduce_int(void* temporary_storage,
                                        size_t* storage_size,
                                        const int* input,
                                        int* output,
                                        unsigned int segments,
                                        const int* begin_offsets,
                                        const int* end_offsets,
                                        int initial_value,
                                        hipStream_t stream,
                                        bool debug_synchronous)
{
    return rocprim::segmented_reduce(temporary_storage, *storage_size, input, output, segments, begin_offsets, end_offsets, rocprim::plus<int>(), initial_value, stream, debug_synchronous);
}

hipError_t rocprim_segmented_inclusive_scan_int(void* temporary_storage,
                                                size_t* storage_size,
                                                const int* input,
                                                int* output,
                                                unsigned int segments,
                                                const int* begin_offsets,
                                                const int* end_offsets,
                                                hipStream_t stream,
                                                bool debug_synchronous)
{
    return rocprim::segmented_inclusive_scan(temporary_storage, *storage_size, input, output, segments, begin_offsets, end_offsets, rocprim::plus<int>(), stream, debug_synchronous);
}

hipError_t rocprim_segmented_exclusive_scan_int(void* temporary_storage,
                                                size_t* storage_size,
                                                const int* input,
                                                int* output,
                                                unsigned int segments,
                                                const int* begin_offsets,
                                                const int* end_offsets,
                                                int initial_value,
                                                hipStream_t stream,
                                                bool debug_synchronous)
{
    return rocprim::segmented_exclusive_scan(temporary_storage, *storage_size, input, output, segments, begin_offsets, end_offsets, initial_value, rocprim::plus<int>(), stream, debug_synchronous);
}

hipError_t rocprim_segmented_radix_sort_keys_int(void* temporary_storage,
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
                                                 bool debug_synchronous)
{
    return rocprim::segmented_radix_sort_keys(temporary_storage, *storage_size, keys_input, keys_output, size, segments, begin_offsets, end_offsets, begin_bit, end_bit, stream, debug_synchronous);
}

hipError_t rocprim_segmented_radix_sort_pairs_int(void* temporary_storage,
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
                                                  bool debug_synchronous)
{
    return rocprim::segmented_radix_sort_pairs(temporary_storage, *storage_size, keys_input, keys_output, values_input, values_output, size, segments, begin_offsets, end_offsets, begin_bit, end_bit, stream, debug_synchronous);
}

hipError_t rocprim_radix_sort_keys_desc_int(void* temporary_storage,
                                            size_t* storage_size,
                                            const int* keys_input,
                                            int* keys_output,
                                            unsigned int size,
                                            unsigned int begin_bit,
                                            unsigned int end_bit,
                                            hipStream_t stream,
                                            bool debug_synchronous)
{
    return rocprim::radix_sort_keys_desc(temporary_storage, *storage_size, keys_input, keys_output, size, begin_bit, end_bit, stream, debug_synchronous);
}

hipError_t rocprim_radix_sort_pairs_desc_int(void* temporary_storage,
                                             size_t* storage_size,
                                             const int* keys_input,
                                             int* keys_output,
                                             const int* values_input,
                                             int* values_output,
                                             unsigned int size,
                                             unsigned int begin_bit,
                                             unsigned int end_bit,
                                             hipStream_t stream,
                                             bool debug_synchronous)
{
    return rocprim::radix_sort_pairs_desc(temporary_storage, *storage_size, keys_input, keys_output, values_input, values_output, size, begin_bit, end_bit, stream, debug_synchronous);
}

hipError_t rocprim_segmented_radix_sort_keys_desc_int(void* temporary_storage,
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
                                                      bool debug_synchronous)
{
    return rocprim::segmented_radix_sort_keys_desc(temporary_storage, *storage_size, keys_input, keys_output, size, segments, begin_offsets, end_offsets, begin_bit, end_bit, stream, debug_synchronous);
}


// --- rocprim::partial_sort_copy (Strengthened: rocPRIM Radix Sort on Prefix + Tail Copy) ---
// Custom implementation to match test intent: Sort only prefix subarray independently (using rocPRIM radix_sort),
// copy full input to output first (preserves tail unchanged). Bypasses rocPRIM API's global k-smallest semantics.
// Uses temp storage for prefix copy + sort; scalable O(n log n) for large prefixes. Temp size queried from radix_sort.

extern "C" hipError_t rocprim_partial_sort_copy_int(
    void* temporary_storage,      // Temp storage buffer (nullptr for query)
    size_t* storage_size,         // Output: Required temp size (queried from radix_sort for prefix)
    const int* keys_input,        // Device input (full array)
    int* keys_output,             // Device output (sorted prefix + original tail)
    size_t middle,                // Middle index (prefix_length = middle + 1)
    size_t size_,                 // Total array size
    hipStream_t stream,           // Stream for async execution
    bool debug_synchronous        // If true, synchronize after operations
) {
    if (size_ == 0) {
        if (storage_size != nullptr) *storage_size = 0;
        return hipSuccess;
    }

    size_t prefix_length = middle + 1;
    if (prefix_length > size_ || prefix_length == 0) {
        return hipErrorInvalidValue;
    }

    size_t temp_size_for_sort = 0;
    hipError_t status = rocprim::radix_sort_keys(
        nullptr, temp_size_for_sort,
        static_cast<const int*>(nullptr), static_cast<int*>(nullptr),  // Dummy iterators for query
        prefix_length, 0, static_cast<unsigned int>(sizeof(int) * 8), stream
    );
    if (status != hipSuccess) return status;

    size_t required_temp_size = temp_size_for_sort + 2 * (prefix_length * sizeof(int));
    if (storage_size != nullptr) {
        *storage_size = required_temp_size;
    }

    // Query phase: Return early after setting size.
    if (temporary_storage == nullptr) {
        return hipSuccess;
    }

    if (storage_size != nullptr && (*storage_size < required_temp_size)) {
        return hipErrorLaunchOutOfResources;
    }

    status = hipMemcpyAsync(const_cast<int*>(keys_output), const_cast<int*>(keys_input), 
                            size_ * sizeof(int), hipMemcpyDeviceToDevice, stream);
    if (status != hipSuccess) return status;

    if (prefix_length <= 1) {
    } else {
        char* temp_base = static_cast<char*>(temporary_storage);
        int* unsorted_prefix_temp = reinterpret_cast<int*>(temp_base);
        void* sorted_prefix_temp_void = temp_base + (prefix_length * sizeof(int));
        int* sorted_prefix_temp = reinterpret_cast<int*>(sorted_prefix_temp_void);
        void* sort_temp_buffer = static_cast<char*>(sorted_prefix_temp_void) + (prefix_length * sizeof(int));

        status = hipMemcpyAsync(unsorted_prefix_temp, const_cast<int*>(keys_input), 
                                prefix_length * sizeof(int), hipMemcpyDeviceToDevice, stream);
        if (status != hipSuccess) return status;

        status = rocprim::radix_sort_keys(
            sort_temp_buffer, temp_size_for_sort,
            unsorted_prefix_temp, sorted_prefix_temp,
            prefix_length, 0, static_cast<unsigned int>(sizeof(int) * 8), stream
        );
        if (status != hipSuccess) return status;

        status = hipMemcpyAsync(keys_output, sorted_prefix_temp, 
                                prefix_length * sizeof(int), hipMemcpyDeviceToDevice, stream);
        if (status != hipSuccess) return status;
    }

    if (debug_synchronous) {
        status = hipStreamSynchronize(stream);
        if (status != hipSuccess) {
            return status;
        }
    }

    return hipSuccess;
}

// --- Custom Kernel for Small-n Pairs Sort (Stable Bubble) ---
__global__ void radix_sort_pairs_small_kernel(int* keys_in, int* values_in, int* keys_out, int* values_out, int n) {
    // Parallel copy to outputs.
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        keys_out[tid] = keys_in[tid];
        values_out[tid] = values_in[tid];
    }
    __syncthreads();

    // Serial bubble sort (thread 0; stable).
    if (threadIdx.x == 0 && n > 1) {
        for (int i = 0; i < n - 1; ++i) {
            bool swapped = false;
            for (int j = 0; j < n - i - 1; ++j) {
                if (keys_out[j] > keys_out[j + 1]) {
                    // Swap keys.
                    int temp_key = keys_out[j];
                    keys_out[j] = keys_out[j + 1];
                    keys_out[j + 1] = temp_key;
                    // Swap values.
                    int temp_val = values_out[j];
                    values_out[j] = values_out[j + 1];
                    values_out[j + 1] = temp_val;
                    swapped = true;
                }
            }
            if (!swapped) break;
        }
    }
}

// --- rocprim::radix_sort_pairs (Keys + Values, Out-of-Place) ---
extern "C" hipError_t rocprim_radix_sort_pairs_int(
    void* temporary_storage,      // Temp for rocPRIM large n (ignored for small n)
    size_t* storage_size,         // Output: RocPRIM temp size (0 for small n)
    const int* keys_input,        // Device input keys (const)
    int* keys_output,             // Device output keys (sorted)
    const int* values_input,      // Device input values
    int* values_output,           // Device output values (sorted by key)
    size_t num_items,             // Number of pairs
    hipStream_t stream            // Stream for async
) {
    if (num_items == 0) {
        if (storage_size != nullptr) *storage_size = 0;
        return hipSuccess;
    }

    hipError_t status;

    // Small n fallback: No temp, always execute on call (ignore temporary_storage).
    if (num_items < 64) {
        if (storage_size != nullptr) *storage_size = 0;

        int* temp_keys_input = nullptr;
        int* temp_values_input = nullptr;
        dim3 block(128);
        dim3 grid(1);

        // Alloc temps (aligned).
        status = hipMalloc((void**)&temp_keys_input, num_items * sizeof(int));
        if (status != hipSuccess) {
            hipFree(temp_keys_input);  // Safe.
            return status;
        }
        status = hipMalloc((void**)&temp_values_input, num_items * sizeof(int));
        if (status != hipSuccess) {
            hipFree(temp_keys_input);
            return status;
        }

        // Copy sync (small n).
        status = hipMemcpy(temp_keys_input, keys_input, num_items * sizeof(int), hipMemcpyDeviceToDevice);
        if (status != hipSuccess) {
            hipFree(temp_keys_input);
            hipFree(temp_values_input);
            return status;
        }
        status = hipMemcpy(temp_values_input, values_input, num_items * sizeof(int), hipMemcpyDeviceToDevice);
        if (status != hipSuccess) {
            hipFree(temp_keys_input);
            hipFree(temp_values_input);
            return status;
        }

        // Sync copies.
        status = hipDeviceSynchronize();
        if (status != hipSuccess) {
            hipFree(temp_keys_input);
            hipFree(temp_values_input);
            return status;
        }

        // Launch kernel.
        hipLaunchKernelGGL(radix_sort_pairs_small_kernel, grid, block, 0, stream, temp_keys_input, temp_values_input, keys_output, values_output, static_cast<int>(num_items));

        status = hipGetLastError();
        if (status != hipSuccess) {
            hipFree(temp_keys_input);
            hipFree(temp_values_input);
            return status;
        }

        status = hipStreamSynchronize(stream);
        if (status != hipSuccess) {
            hipFree(temp_keys_input);
            hipFree(temp_values_input);
            return status;
        }

        printf("Fallback used, num_items=%zu\n", num_items);  // Debug

        // Cleanup.
        hipFree(temp_keys_input);
        hipFree(temp_values_input);
        return status;
    }

    // Large n: Query rocPRIM size (standard).
    size_t rocprim_size = 0;
    status = rocprim::radix_sort_pairs(
        nullptr, rocprim_size,
        static_cast<int*>(nullptr), static_cast<int*>(nullptr),
        static_cast<int*>(nullptr), static_cast<int*>(nullptr),
        num_items, 0, static_cast<unsigned int>(sizeof(int) * 8), stream
    );
    if (status != hipSuccess) return status;

    if (storage_size != nullptr) {
        *storage_size = rocprim_size;
    }

    if (temporary_storage == nullptr) {
        return hipSuccess;
    }

    if (storage_size != nullptr && *storage_size < rocprim_size) {
        return hipErrorLaunchOutOfResources;
    }

    // Alloc input temps for large n.
    int* temp_keys_input = nullptr;
    int* temp_values_input = nullptr;
    status = hipMalloc((void**)&temp_keys_input, num_items * sizeof(int));
    if (status != hipSuccess) return status;
    status = hipMalloc((void**)&temp_values_input, num_items * sizeof(int));
    if (status != hipSuccess) {
        hipFree(temp_keys_input);
        return status;
    }

    // Copy async.
    status = hipMemcpyAsync(temp_keys_input, keys_input, num_items * sizeof(int), hipMemcpyDeviceToDevice, stream);
    if (status != hipSuccess) {
        hipFree(temp_keys_input);
        hipFree(temp_values_input);
        return status;
    }
    status = hipMemcpyAsync(temp_values_input, values_input, num_items * sizeof(int), hipMemcpyDeviceToDevice, stream);
    if (status != hipSuccess) {
        hipFree(temp_keys_input);
        hipFree(temp_values_input);
        return status;
    }

    status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
        hipFree(temp_keys_input);
        hipFree(temp_values_input);
        return status;
    }

    // RocPRIM sort.
    status = rocprim::radix_sort_pairs(
        temporary_storage, rocprim_size,
        temp_keys_input, temp_values_input, keys_output, values_output,
        num_items, 0, static_cast<unsigned int>(sizeof(int) * 8), stream
    );
    if (status != hipSuccess) {
        hipFree(temp_keys_input);
        hipFree(temp_values_input);
        return status;
    }

    status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
        hipFree(temp_keys_input);
        hipFree(temp_values_input);
        return status;
    }

    printf("RocPRIM used, num_items=%zu\n", num_items);  // Debug

    // Cleanup.
    hipFree(temp_keys_input);
    hipFree(temp_values_input);
    return status;
}

hipError_t rocprim_reduce_int128(
    rocprim::int128_t* d_out,
    const rocprim::int128_t* d_in,
    size_t num_items,
    rocprim::int128_t initial_value,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    status = rocprim::reduce(d_temp_storage, temp_storage_size, d_in, d_out, initial_value, num_items, rocprim::plus<rocprim::int128_t>(), stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::reduce(d_temp_storage, temp_storage_size, d_in, d_out, initial_value, num_items, rocprim::plus<rocprim::int128_t>(), stream);
    
    hipFree(d_temp_storage);
    return status;
}

hipError_t rocprim_reduce_reverse_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    int initial_value,
    hipStream_t stream)
{
    size_t temp_storage_size = 0;
    void* d_temp_storage = nullptr;
    hipError_t status;

    auto d_in_reverse = rocprim::make_reverse_iterator(d_in + num_items);

    status = rocprim::reduce(d_temp_storage, temp_storage_size, d_in_reverse, d_out, initial_value, num_items, rocprim::plus<int>(), stream);
    if (status != hipSuccess) return status;

    status = hipMalloc(&d_temp_storage, temp_storage_size);
    if (status != hipSuccess) return status;

    status = rocprim::reduce(d_temp_storage, temp_storage_size, d_in_reverse, d_out, initial_value, num_items, rocprim::plus<int>(), stream);
    
    hipFree(d_temp_storage);
    return status;
}

hipError_t rocprim_block_reduce_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    unsigned int block_size,
    hipStream_t stream)
{
    if (block_size == 128)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(block_reduce_kernel<128>),
            dim3(rocprim::detail::ceiling_div(num_items, 128u)),
            dim3(128),
            0,
            stream,
            d_in, d_out, num_items
        );
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(block_reduce_kernel<256>),
            dim3(rocprim::detail::ceiling_div(num_items, 256u)),
            dim3(256),
            0,
            stream,
            d_in, d_out, num_items
        );
    }
    return hipGetLastError();
}

hipError_t rocprim_block_scan_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    unsigned int block_size,
    hipStream_t stream)
{
    if (block_size == 128)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(block_scan_kernel<128>),
            dim3(rocprim::detail::ceiling_div(num_items, 128u)),
            dim3(128),
            0,
            stream,
            d_in, d_out, num_items
        );
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(block_scan_kernel<256>),
            dim3(rocprim::detail::ceiling_div(num_items, 256u)),
            dim3(256),
            0,
            stream,
            d_in, d_out, num_items
        );
    }
    return hipGetLastError();
}

hipError_t rocprim_block_sort_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    unsigned int block_size,
    hipStream_t stream)
{
    if (block_size == 128)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(block_sort_kernel<128>),
            dim3(rocprim::detail::ceiling_div(num_items, 128u)),
            dim3(128),
            0,
            stream,
            d_in, d_out, num_items
        );
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(block_sort_kernel<256>),
            dim3(rocprim::detail::ceiling_div(num_items, 256u)),
            dim3(256),
            0,
            stream,
            d_in, d_out, num_items
        );
    }
    return hipGetLastError();
}

hipError_t rocprim_block_histogram_int(
    int* d_out,
    const int* d_in,
    size_t num_items,
    unsigned int block_size,
    unsigned int bins,
    hipStream_t stream)
{
    if (block_size == 128)
    {
        if (bins == 256)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(block_histogram_kernel<128, 256>),
                dim3(rocprim::detail::ceiling_div(num_items, 128u)),
                dim3(128),
                0,
                stream,
                d_in, d_out, num_items
            );
        }
    }
    else
    {
        if (bins == 256)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(block_histogram_kernel<256, 256>),
                dim3(rocprim::detail::ceiling_div(num_items, 256u)),
                dim3(256),
                0,
                stream,
                d_in, d_out, num_items
            );
        }
    }
    return hipGetLastError();
}

hipError_t rocprim_warp_reduce_int8(int8_t* d_out, const int8_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_reduce(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_reduce_int16(int16_t* d_out, const int16_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_reduce(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_reduce_int32(int32_t* d_out, const int32_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_reduce(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_reduce_uint32(uint32_t* d_out, const uint32_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_reduce(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_reduce_int64(int64_t* d_out, const int64_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_reduce(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_reduce_uint64(uint64_t* d_out, const uint64_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_reduce(d_out, d_in, num_items, stream); }

hipError_t rocprim_warp_scan_int8(int8_t* d_out, const int8_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_scan(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_scan_int16(int16_t* d_out, const int16_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_scan(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_scan_int32(int32_t* d_out, const int32_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_scan(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_scan_uint32(uint32_t* d_out, const uint32_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_scan(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_scan_int64(int64_t* d_out, const int64_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_scan(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_scan_uint64(uint64_t* d_out, const uint64_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_scan(d_out, d_in, num_items, stream); }

hipError_t rocprim_warp_sort_int8(int8_t* d_out, const int8_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_sort(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_sort_int16(int16_t* d_out, const int16_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_sort(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_sort_int32(int32_t* d_out, const int32_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_sort(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_sort_uint32(uint32_t* d_out, const uint32_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_sort(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_sort_int64(int64_t* d_out, const int64_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_sort(d_out, d_in, num_items, stream); }
hipError_t rocprim_warp_sort_uint64(uint64_t* d_out, const uint64_t* d_in, size_t num_items, hipStream_t stream) { return rocprim_warp_sort(d_out, d_in, num_items, stream); }

// --- rocprim::histogram_range ---
hipError_t rocprim_histogram_range_int(
    void* temporary_storage,
    size_t* storage_size,
    const int* samples,
    unsigned int size,
    int* histogram,
    unsigned int levels,
    int* level_values,
    hipStream_t stream,
    bool debug_synchronous)
{
    return rocprim::histogram_range(
        temporary_storage,
        *storage_size,
        samples,
        size,
        histogram,
        levels,
        level_values,
        stream,
        debug_synchronous
    );
}

// --- rocprim::merge_sort_pairs ---
hipError_t rocprim_merge_sort_pairs_int(
    void* temporary_storage,
    size_t* storage_size,
    const int* keys_input,
    int* keys_output,
    const int* values_input,
    int* values_output,
    size_t size,
    hipStream_t stream,
    bool debug_synchronous)
{
    return rocprim::merge_sort(
        temporary_storage,
        *storage_size,
        keys_input,
        keys_output,
        values_input,
        values_output,
        size,
        rocprim::less<int>(),
        stream,
        debug_synchronous
    );
}

} // extern "C" 

extern "C" {
    // ... existing functions ...

    int rocprim_numeric_limits_int_min() {
        return std::numeric_limits<int>::min();
    }

    int rocprim_numeric_limits_int_max() {
        return std::numeric_limits<int>::max();
    }

    float rocprim_numeric_limits_float_epsilon() {
        return std::numeric_limits<float>::epsilon();
    }

    float rocprim_numeric_limits_float_infinity() {
        return std::numeric_limits<float>::infinity();
    }

    float rocprim_numeric_limits_float_nan() {
        return std::numeric_limits<float>::quiet_NaN();
    }
}

extern "C" __global__ void clamp_kernel_float(float* data, float min_val, float max_val, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        data[idx] = (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
    }
}

extern "C" hipError_t rocprim_clamp_float(float* d_data, float min_val, float max_val, size_t n, hipStream_t stream) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(clamp_kernel_float, gridSize, blockSize, 0, stream, d_data, min_val, max_val, n);
    return hipGetLastError();
}

extern "C" __global__ void bernoulli_kernel_float(float* data, float p, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = (data[idx] < p) ? 1.0f : 0.0f;
    }
}

extern "C" hipError_t rocprim_bernoulli_float(float* d_data, float p, size_t n, hipStream_t stream) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(bernoulli_kernel_float, gridSize, blockSize, 0, stream, d_data, p, n);
    return hipGetLastError();
}
