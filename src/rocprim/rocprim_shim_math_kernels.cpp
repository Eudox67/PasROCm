// Elementwise math kernels for ROCm (float32).
// Compiled into rocprim_shim.so next to the existing kernels.
// All kernels are stream-parallel (async), no host sync.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstddef>

// Same reason as rocprim_shim.cpp: the prototypes carry PASROCM_EXPORT, so the
// definitions in this translation unit inherit default visibility and nothing
// else leaks out of a -fvisibility=hidden build.
#include "rocprim_shim.h"

extern "C" __global__ void elew_exp_float(const float* __restrict__ in, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = expf(in[idx]);
}

extern "C" __global__ void elew_log_float(const float* __restrict__ in, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Matches Pascal: Ln(x + 1e-12) — epsilon keeps cross-entropy stable.
        out[idx] = logf(in[idx] + 1e-12f);
    }
}

extern "C" __global__ void elew_sqrt_float(const float* __restrict__ in, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float v = in[idx];
        out[idx] = (v > 0.0f) ? sqrtf(v) : 0.0f;
    }
}

extern "C" __global__ void elew_recip_float(const float* __restrict__ in, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float v = in[idx];
        out[idx] = (v != 0.0f) ? (1.0f / v) : 0.0f;
    }
}

extern "C" __global__ void elew_div_float(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, size_t a_n, size_t b_n, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float denom = b[idx % b_n];
        float num = a[idx % a_n];
        out[idx] = (denom != 0.0f) ? (num / denom) : 0.0f;
    }
}

extern "C" __global__ void elew_relu_bwd_float(const float* __restrict__ grad, const float* __restrict__ input, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = (input[idx] > 0.0f) ? grad[idx] : 0.0f;
    }
}

#define LAUNCH_ELEW(kernel)                                          \
    int blockSize = 256;                                             \
    int gridSize = (int)((n + blockSize - 1) / blockSize);           \
    if (gridSize < 1) gridSize = 1;                                  \
    hipLaunchKernelGGL(kernel, gridSize, blockSize, 0, stream,       \
                       in, out, n);                                  \
    return hipGetLastError();

extern "C" hipError_t rocprim_exp_float(const float* in, float* out, size_t n, hipStream_t stream)
{
    LAUNCH_ELEW(elew_exp_float)
}

extern "C" hipError_t rocprim_log_float(const float* in, float* out, size_t n, hipStream_t stream)
{
    LAUNCH_ELEW(elew_log_float)
}

extern "C" hipError_t rocprim_sqrt_float(const float* in, float* out, size_t n, hipStream_t stream)
{
    LAUNCH_ELEW(elew_sqrt_float)
}

extern "C" hipError_t rocprim_recip_float(const float* in, float* out, size_t n, hipStream_t stream)
{
    LAUNCH_ELEW(elew_recip_float)
}

extern "C" hipError_t rocprim_div_float(const float* a, const float* b, float* out, size_t a_n, size_t b_n, size_t n, hipStream_t stream)
{
    int blockSize = 256;
    int gridSize = (int)((n + blockSize - 1) / blockSize);
    if (gridSize < 1) gridSize = 1;
    hipLaunchKernelGGL(elew_div_float, gridSize, blockSize, 0, stream, a, b, out, a_n, b_n, n);
    return hipGetLastError();
}

extern "C" hipError_t rocprim_relu_bwd_float(const float* grad, const float* input, float* out, size_t n, hipStream_t stream)
{
    int blockSize = 256;
    int gridSize = (int)((n + blockSize - 1) / blockSize);
    if (gridSize < 1) gridSize = 1;
    hipLaunchKernelGGL(elew_relu_bwd_float, gridSize, blockSize, 0, stream, grad, input, out, n);
    return hipGetLastError();
}

// ------------------------------------------------------------------
// Binary elementwise ops with scalar-broadcast (matches div signature)
// ------------------------------------------------------------------
#define ELEW_BINARY(NAME, EXPR)                                            \
extern "C" __global__ void NAME(                                           \
    const float* __restrict__ a, const float* __restrict__ b,              \
    float* __restrict__ out, size_t a_n, size_t b_n, size_t n)             \
{                                                                          \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                    \
    if (idx < n)                                                           \
    {                                                                      \
        float av = a[idx % a_n];                                           \
        float bv = b[idx % b_n];                                           \
        out[idx] = EXPR;                                                   \
    }                                                                      \
}

ELEW_BINARY(elew_add_float, av + bv)
ELEW_BINARY(elew_sub_float, av - bv)
ELEW_BINARY(elew_mul_float, av * bv)

#define LAUNCH_ELEW_BINARY(kernel)                                   \
    int blockSize = 256;                                             \
    int gridSize = (int)((n + blockSize - 1) / blockSize);           \
    if (gridSize < 1) gridSize = 1;                                  \
    hipLaunchKernelGGL(kernel, gridSize, blockSize, 0, stream,       \
                       a, b, out, a_n, b_n, n);                      \
    return hipGetLastError();

extern "C" hipError_t rocprim_add_float(const float* a, const float* b, float* out, size_t a_n, size_t b_n, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_BINARY(elew_add_float) }

extern "C" hipError_t rocprim_sub_float(const float* a, const float* b, float* out, size_t a_n, size_t b_n, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_BINARY(elew_sub_float) }

extern "C" hipError_t rocprim_mul_float(const float* a, const float* b, float* out, size_t a_n, size_t b_n, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_BINARY(elew_mul_float) }

// ----------------- native 16-bit add/sub/mul (half, bf16) -----------------
// FP32 compute in-register, 16-bit store. Same elementwise-shape only
// (a_n = b_n = n) — broadcast stays on the FP32 bridge in FIRE.

#define ELEW_BIN16(NAME, EXPR)                                               \
template <typename T>                                                        \
__global__ void NAME(const T* __restrict__ a, const T* __restrict__ b,       \
                     T* __restrict__ out, size_t n)                          \
{                                                                            \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                      \
    if (idx < n)                                                             \
    {                                                                        \
        float av = (float)a[idx];                                            \
        float bv = (float)b[idx];                                            \
        out[idx] = (T)(EXPR);                                                \
    }                                                                        \
}

ELEW_BIN16(elew_add16_t, av + bv)
ELEW_BIN16(elew_sub16_t, av - bv)
ELEW_BIN16(elew_mul16_t, av * bv)

#define LAUNCH_BIN16(kernel, T)                                              \
    int b16BlockSize = 256;                                                  \
    int b16Grid = (int)((n + b16BlockSize - 1) / b16BlockSize);               \
    if (b16Grid < 1) b16Grid = 1;                                            \
    hipLaunchKernelGGL((kernel<T>), b16Grid, b16BlockSize, 0, stream,        \
                       (const T*)a, (const T*)b, (T*)out, n);                \
    return hipGetLastError();

extern "C" hipError_t rocprim_add_half(const void* a, const void* b, void* out, size_t n, hipStream_t stream)
{ LAUNCH_BIN16(elew_add16_t, __half) }

extern "C" hipError_t rocprim_add_bf16(const void* a, const void* b, void* out, size_t n, hipStream_t stream)
{ LAUNCH_BIN16(elew_add16_t, hip_bfloat16) }

extern "C" hipError_t rocprim_sub_half(const void* a, const void* b, void* out, size_t n, hipStream_t stream)
{ LAUNCH_BIN16(elew_sub16_t, __half) }

extern "C" hipError_t rocprim_sub_bf16(const void* a, const void* b, void* out, size_t n, hipStream_t stream)
{ LAUNCH_BIN16(elew_sub16_t, hip_bfloat16) }

extern "C" hipError_t rocprim_mul_half(const void* a, const void* b, void* out, size_t n, hipStream_t stream)
{ LAUNCH_BIN16(elew_mul16_t, __half) }

extern "C" hipError_t rocprim_mul_bf16(const void* a, const void* b, void* out, size_t n, hipStream_t stream)
{ LAUNCH_BIN16(elew_mul16_t, hip_bfloat16) }

// ---------------------- two-pass global sum (float) ----------------------
// Pass 1: each block reduces a contiguous chunk into partials[blockIdx].
// Pass 2: one block reduces the partials. ~5x faster than the rocBLAS
// sgemm-based sum trick for large n.

__global__ void reduce_partial_float(
    const float* __restrict__ in, float* __restrict__ partials,
    size_t n, size_t perBlock)
{
    __shared__ float sh[256];
    size_t start = (size_t)blockIdx.x * perBlock;
    size_t end   = start + perBlock;
    if (end > n) end = n;
    float s = 0.0f;
    for (size_t j = start + threadIdx.x; j < end; j += blockDim.x)
        s += in[j];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1)
    {
        if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = sh[0];
}

__global__ void reduce_final_float(const float* __restrict__ partials, float* __restrict__ out, int nb)
{
    __shared__ float sh[256];
    float s = 0.0f;
    for (int j = threadIdx.x; j < nb; j += blockDim.x) s += partials[j];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1)
    {
        if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = sh[0];
}

extern "C" hipError_t rocprim_sum_float(
    const float* in, float* out, size_t n, hipStream_t stream)
{
    if (n < 1) return hipSuccess;
    size_t blocks = (n + 8191) / 8192;
    if (blocks > 2048) blocks = 2048;
    if (blocks < 1) blocks = 1;
    float* partials = nullptr;
    if (hipMalloc(&partials, blocks * sizeof(float)) != hipSuccess)
        return hipGetLastError();
    hipLaunchKernelGGL(reduce_partial_float, dim3((unsigned)blocks), dim3(256),
                       0, stream, in, partials, n,
                       (n + blocks - 1) / blocks);
    hipError_t e = hipGetLastError();
    if (e == hipSuccess)
    {
        hipLaunchKernelGGL(reduce_final_float, dim3(1), dim3(256),
                           0, stream, partials, out, (int)blocks);
        e = hipGetLastError();
    }
    hipFree(partials);
    return e;
}

// ------------------------- scale (unary * scalar) -------------------------
extern "C" __global__ void elew_scale_float(const float* __restrict__ in, float scalar, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * scalar;
}

extern "C" hipError_t rocprim_scale_float(const float* in, float scalar, float* out, size_t n, hipStream_t stream)
{
    int blockSize = 256;
    int gridSize = (int)((n + blockSize - 1) / blockSize);
    if (gridSize < 1) gridSize = 1;
    hipLaunchKernelGGL(elew_scale_float, gridSize, blockSize, 0, stream, in, scalar, out, n);
    return hipGetLastError();
}

// In-place variant: data[i] *= scalar — safe when out == in.
extern "C" __global__ void elew_scale_inplace_float(float* __restrict__ data, float scalar, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= scalar;
}

extern "C" hipError_t rocprim_scale_inplace_float(float* data, float scalar, size_t n, hipStream_t stream)
{
    int blockSize = 256;
    int gridSize = (int)((n + blockSize - 1) / blockSize);
    if (gridSize < 1) gridSize = 1;
    hipLaunchKernelGGL(elew_scale_inplace_float, gridSize, blockSize, 0, stream, data, scalar, n);
    return hipGetLastError();
}

// --------------------- sigmoid / tanh, fwd + bwd ---------------------
// Backward variants take grad + input and recompute the activation,
// mirroring elew_relu_bwd_float's (grad, input, out) convention.
extern "C" __global__ void elew_sigmoid_float(const float* __restrict__ in, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 1.0f / (1.0f + expf(-in[idx]));
}

extern "C" __global__ void elew_sigmoid_bwd_float(const float* __restrict__ grad, const float* __restrict__ input, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float s = 1.0f / (1.0f + expf(-input[idx]));
        out[idx] = grad[idx] * s * (1.0f - s);
    }
}

extern "C" __global__ void elew_tanh_float(const float* __restrict__ in, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = tanhf(in[idx]);
}

extern "C" __global__ void elew_tanh_bwd_float(const float* __restrict__ grad, const float* __restrict__ input, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float t = tanhf(input[idx]);
        out[idx] = grad[idx] * (1.0f - t * t);
    }
}

#define LAUNCH_ELEW_GRADIN(kernel2)                                  \
    int blockSize = 256;                                             \
    int gridSize = (int)((n + blockSize - 1) / blockSize);           \
    if (gridSize < 1) gridSize = 1;                                  \
    hipLaunchKernelGGL(kernel2, gridSize, blockSize, 0, stream,      \
                       grad, input, out, n);                         \
    return hipGetLastError();

extern "C" hipError_t rocprim_sigmoid_float(const float* in, float* out, size_t n, hipStream_t stream)
{
    int blockSize = 256;
    int gridSize = (int)((n + blockSize - 1) / blockSize);
    if (gridSize < 1) gridSize = 1;
    hipLaunchKernelGGL(elew_sigmoid_float, gridSize, blockSize, 0, stream, in, out, n);
    return hipGetLastError();
}

extern "C" hipError_t rocprim_sigmoid_bwd_float(const float* grad, const float* input, float* out, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_GRADIN(elew_sigmoid_bwd_float) }

extern "C" hipError_t rocprim_tanh_float(const float* in, float* out, size_t n, hipStream_t stream)
{
    int blockSize = 256;
    int gridSize = (int)((n + blockSize - 1) / blockSize);
    if (gridSize < 1) gridSize = 1;
    hipLaunchKernelGGL(elew_tanh_float, gridSize, blockSize, 0, stream, in, out, n);
    return hipGetLastError();
}

extern "C" hipError_t rocprim_tanh_bwd_float(const float* grad, const float* input, float* out, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_GRADIN(elew_tanh_bwd_float) }

// --------------------- GELU / SiLU, fwd + bwd ---------------------
// GELU (exact)  : y = 0.5 * x * (1 + erf(x / sqrt(2)))
// SiLU (Swish)  : y = x * sigmoid(x)
// Backward variants take (grad, input) like elew_relu_bwd_float / sigmoid bwd.
static constexpr float kInvSqrt2      = 0.70710678118654752440f; // 1/sqrt(2)
static constexpr float kInvSqrt2pi    = 0.39894228040143267794f; // 1/sqrt(2*pi)

extern "C" __global__ void elew_gelu_float(const float* __restrict__ in, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = in[idx];
        out[idx] = 0.5f * x * (1.0f + erff(x * kInvSqrt2));
    }
}

extern "C" __global__ void elew_gelu_bwd_float(const float* __restrict__ grad, const float* __restrict__ input, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = input[idx];
        float phi = 0.5f * (1.0f + erff(x * kInvSqrt2));       // cdf of N(0,1)
        float pdf = kInvSqrt2pi * expf(-0.5f * x * x);         // pdf of N(0,1)
        out[idx] = grad[idx] * (phi + x * pdf);
    }
}

extern "C" __global__ void elew_silu_float(const float* __restrict__ in, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = in[idx];
        out[idx] = x / (1.0f + expf(-x));
    }
}

extern "C" __global__ void elew_silu_bwd_float(const float* __restrict__ grad, const float* __restrict__ input, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = input[idx];
        float s = 1.0f / (1.0f + expf(-x));        // sigmoid(x)
        out[idx] = grad[idx] * (s + x * s * (1.0f - s));
    }
}

extern "C" hipError_t rocprim_gelu_float(const float* in, float* out, size_t n, hipStream_t stream)
{
    LAUNCH_ELEW(elew_gelu_float)
}

extern "C" hipError_t rocprim_gelu_bwd_float(const float* grad, const float* input, float* out, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_GRADIN(elew_gelu_bwd_float) }

extern "C" hipError_t rocprim_silu_float(const float* in, float* out, size_t n, hipStream_t stream)
{
    LAUNCH_ELEW(elew_silu_float)
}

extern "C" hipError_t rocprim_silu_bwd_float(const float* grad, const float* input, float* out, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_GRADIN(elew_silu_bwd_float) }
// ------------------------------------------------------------------
// Native FP16 / BF16 variants: half/bfloat16 storage with in-register
// FP32 compute, converting once per element. One launch, one pass —
// replaces the old convert→compute→convert FP32 bridge (3 launches,
// 3 passes, 2 temporaries). Same numerics: FP32 accumulation is what
// the bridge did anyway, just redundantly through HBM.
// ------------------------------------------------------------------
template <typename T>
__global__ void elew_gelu_t(const T* __restrict__ in, T* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = (float)in[idx];
        out[idx] = (T)(0.5f * x * (1.0f + erff(x * kInvSqrt2)));
    }
}

template <typename T>
__global__ void elew_gelu_bwd_t(const T* __restrict__ grad, const T* __restrict__ input, T* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = (float)input[idx];
        float g = (float)grad[idx];
        float phi = 0.5f * (1.0f + erff(x * kInvSqrt2));       // cdf of N(0,1)
        float pdf = kInvSqrt2pi * expf(-0.5f * x * x);         // pdf of N(0,1)
        out[idx] = (T)(g * (phi + x * pdf));
    }
}

template <typename T>
__global__ void elew_silu_t(const T* __restrict__ in, T* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = (float)in[idx];
        out[idx] = (T)(x / (1.0f + expf(-x)));
    }
}

template <typename T>
__global__ void elew_silu_bwd_t(const T* __restrict__ grad, const T* __restrict__ input, T* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = (float)input[idx];
        float g = (float)grad[idx];
        float s = 1.0f / (1.0f + expf(-x));                    // sigmoid(x)
        out[idx] = (T)(g * (s + x * s * (1.0f - s)));
    }
}

#define LAUNCH_ELEW_T(kernel, T)                                     \
    int blockSize = 256;                                             \
    int gridSize = (int)((n + blockSize - 1) / blockSize);           \
    if (gridSize < 1) gridSize = 1;                                  \
    hipLaunchKernelGGL(kernel<T>, gridSize, blockSize, 0, stream,    \
                       in, out, n);                                  \
    return hipGetLastError();

#define LAUNCH_ELEW_GRADIN_T(kernel, T)                              \
    int blockSize = 256;                                             \
    int gridSize = (int)((n + blockSize - 1) / blockSize);           \
    if (gridSize < 1) gridSize = 1;                                  \
    hipLaunchKernelGGL(kernel<T>, gridSize, blockSize, 0, stream,    \
                       grad, input, out, n);                         \
    return hipGetLastError();

extern "C" hipError_t rocprim_gelu_half(const __half* in, __half* out, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_T(elew_gelu_t, __half) }

extern "C" hipError_t rocprim_gelu_bf16(const hip_bfloat16* in, hip_bfloat16* out, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_T(elew_gelu_t, hip_bfloat16) }

extern "C" hipError_t rocprim_gelu_bwd_half(const __half* grad, const __half* input, __half* out, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_GRADIN_T(elew_gelu_bwd_t, __half) }

extern "C" hipError_t rocprim_gelu_bwd_bf16(const hip_bfloat16* grad, const hip_bfloat16* input, hip_bfloat16* out, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_GRADIN_T(elew_gelu_bwd_t, hip_bfloat16) }

extern "C" hipError_t rocprim_silu_half(const __half* in, __half* out, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_T(elew_silu_t, __half) }

extern "C" hipError_t rocprim_silu_bf16(const hip_bfloat16* in, hip_bfloat16* out, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_T(elew_silu_t, hip_bfloat16) }

extern "C" hipError_t rocprim_silu_bwd_half(const __half* grad, const __half* input, __half* out, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_GRADIN_T(elew_silu_bwd_t, __half) }

extern "C" hipError_t rocprim_silu_bwd_bf16(const hip_bfloat16* grad, const hip_bfloat16* input, hip_bfloat16* out, size_t n, hipStream_t stream)
{ LAUNCH_ELEW_GRADIN_T(elew_silu_bwd_t, hip_bfloat16) }


// ------------------------------------------------------------------
// Backward variants consuming the *saved forward output* y (FIRE's
// autograd stores outputs, not inputs):
//   sigmoid: dx = g * y * (1 - y)
//   tanh:    dx = g * (1 - t^2)
// Macro not usable here because its parameter list is named differently.
// ------------------------------------------------------------------
extern "C" __global__ void elew_sigmoid_bwd_y_float(const float* __restrict__ grad, const float* __restrict__ y, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = grad[idx] * y[idx] * (1.0f - y[idx]);
}

extern "C" __global__ void elew_tanh_bwd_y_float(const float* __restrict__ grad, const float* __restrict__ t, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = grad[idx] * (1.0f - t[idx] * t[idx]);
}

extern "C" hipError_t rocprim_sigmoid_bwd_out_float(const float* grad, const float* y, float* out, size_t n, hipStream_t stream)
{
    int blockSize = 256;
    int gridSize = (int)((n + blockSize - 1) / blockSize);
    if (gridSize < 1) gridSize = 1;
    hipLaunchKernelGGL(elew_sigmoid_bwd_y_float, gridSize, blockSize, 0, stream, grad, y, out, n);
    return hipGetLastError();
}

extern "C" hipError_t rocprim_tanh_bwd_out_float(const float* grad, const float* t, float* out, size_t n, hipStream_t stream)
{
    int blockSize = 256;
    int gridSize = (int)((n + blockSize - 1) / blockSize);
    if (gridSize < 1) gridSize = 1;
    hipLaunchKernelGGL(elew_tanh_bwd_y_float, gridSize, blockSize, 0, stream, grad, t, out, n);
    return hipGetLastError();
}

// NOTE: clamp lives in rocprim_shim.cpp (rocprim_clamp_float) — do not
// redefine it here; it caused a duplicate-symbol link error.

extern "C" __global__ void elew_fill_float(float* __restrict__ out, float value, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = value;
}

extern "C" hipError_t rocprim_fill_float(float* out, float value, size_t n, hipStream_t stream)
{
    int blockSize = 256;
    int gridSize = (int)((n + blockSize - 1) / blockSize);
    if (gridSize < 1) gridSize = 1;
    hipLaunchKernelGGL(elew_fill_float, gridSize, blockSize, 0, stream, out, value, n);
    return hipGetLastError();
}

// ------------------------------------------------------------------
// Row-wise softmax / log-softmax (+ backwards). Layout: rows * cols,
// contiguous last dim. One thread block per row, 256 threads.
// Two-pass algorithm (max then sum); second global read hits L2.
// ------------------------------------------------------------------
#define SOFTMAX_BLOCK 256

__device__ __forceinline__ float warpReduceSumV(float v)
{
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down(v, off);
    return v;
}

__device__ __forceinline__ float warpReduceMaxV(float v)
{
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down(v, off));
    return v;
}

// softmax fwd: out = exp(x - max) / sum(exp(x - max))
extern "C" __global__ void soft_max_rows(const float* __restrict__ in, float* __restrict__ out, size_t cols)
{
    size_t row = blockIdx.x;
    const float* x = in + row * cols;
    float* o = out + row * cols;
    __shared__ float sh[SOFTMAX_BLOCK];

    // pass 1: row max
    float m = -INFINITY;
    for (size_t i = threadIdx.x; i < cols; i += SOFTMAX_BLOCK)
        m = fmaxf(m, x[i]);
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int s = SOFTMAX_BLOCK / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    float rmax = sh[0];

    // pass 2: sum of exp
    float acc = 0.0f;
    for (size_t i = threadIdx.x; i < cols; i += SOFTMAX_BLOCK)
        acc += expf(x[i] - rmax);
    float wsum = warpReduceSumV(acc);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = wsum;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float t = 0.0f;
        for (int w = 0; w < SOFTMAX_BLOCK / 32; ++w) t += sh[w];
        sh[0] = t;
    }
    __syncthreads();
    float denom = sh[0];

    for (size_t i = threadIdx.x; i < cols; i += SOFTMAX_BLOCK)
        o[i] = expf(x[i] - rmax) / denom;
}

// logsoftmax fwd: out = x - max - log(sum(exp(x - max)))
extern "C" __global__ void soft_logmax_rows(const float* __restrict__ in, float* __restrict__ out, size_t cols)
{
    size_t row = blockIdx.x;
    const float* x = in + row * cols;
    float* o = out + row * cols;
    __shared__ float sh[SOFTMAX_BLOCK];

    float m = -INFINITY;
    for (size_t i = threadIdx.x; i < cols; i += SOFTMAX_BLOCK)
        m = fmaxf(m, x[i]);
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int s = SOFTMAX_BLOCK / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    float rmax = sh[0];

    float acc = 0.0f;
    for (size_t i = threadIdx.x; i < cols; i += SOFTMAX_BLOCK)
        acc += expf(x[i] - rmax);
    float wsum = warpReduceSumV(acc);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = wsum;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float t = 0.0f;
        for (int w = 0; w < SOFTMAX_BLOCK / 32; ++w) t += sh[w];
        sh[0] = logf(t);
    }
    __syncthreads();
    float lse = sh[0];   // log-sum-exp

    for (size_t i = threadIdx.x; i < cols; i += SOFTMAX_BLOCK)
        o[i] = x[i] - rmax - lse;
}

#define LAUNCH_ROWS(kernel)                                          \
    int blockSize = SOFTMAX_BLOCK;                                   \
    int gridSize = (int)(rows);                                      \
    if (gridSize < 1) gridSize = 1;                                  \
    hipLaunchKernelGGL(kernel, dim3(gridSize), dim3(blockSize),      \
                       0, stream, in, out, cols);                    \
    return hipGetLastError();

extern "C" hipError_t rocprim_softmax_float(const float* in, float* out, size_t rows, size_t cols, hipStream_t stream)
{ LAUNCH_ROWS(soft_max_rows) }

extern "C" hipError_t rocprim_logsoftmax_float(const float* in, float* out, size_t rows, size_t cols, hipStream_t stream)
{ LAUNCH_ROWS(soft_logmax_rows) }

// softmax bwd: dx_i = y_i * (g_i - dot(g, y))   [y = softmax output]
extern "C" __global__ void soft_max_rows_bwd(const float* __restrict__ grad, const float* __restrict__ y, float* __restrict__ out, size_t cols)
{
    size_t row = blockIdx.x;
    const float* g = grad + row * cols;
    const float* p = y + row * cols;
    float* o = out + row * cols;
    __shared__ float sh[SOFTMAX_BLOCK];

    float dot = 0.0f;
    for (size_t i = threadIdx.x; i < cols; i += SOFTMAX_BLOCK)
        dot += g[i] * p[i];
    float wsum = warpReduceSumV(dot);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = wsum;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float t = 0.0f;
        for (int w = 0; w < SOFTMAX_BLOCK / 32; ++w) t += sh[w];
        sh[0] = t;
    }
    __syncthreads();
    float d = sh[0];

    for (size_t i = threadIdx.x; i < cols; i += SOFTMAX_BLOCK)
        o[i] = p[i] * (g[i] - d);
}

// logsoftmax bwd: dx_i = g_i - exp(y_i) * sum(g)
extern "C" __global__ void soft_logmax_rows_bwd(const float* __restrict__ grad, const float* __restrict__ ly, float* __restrict__ out, size_t cols)
{
    size_t row = blockIdx.x;
    const float* g = grad + row * cols;
    const float* l = ly + row * cols;
    float* o = out + row * cols;
    __shared__ float sh[SOFTMAX_BLOCK];

    float acc = 0.0f;
    for (size_t i = threadIdx.x; i < cols; i += SOFTMAX_BLOCK)
        acc += g[i];
    float wsum = warpReduceSumV(acc);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = wsum;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float t = 0.0f;
        for (int w = 0; w < SOFTMAX_BLOCK / 32; ++w) t += sh[w];
        sh[0] = t;
    }
    __syncthreads();
    float s = sh[0];

    for (size_t i = threadIdx.x; i < cols; i += SOFTMAX_BLOCK)
        o[i] = g[i] - expf(l[i]) * s;
}

#define LAUNCH_ROWS_BWD(kernel)                                      \
    int blockSize = SOFTMAX_BLOCK;                                   \
    int gridSize = (int)(rows);                                      \
    if (gridSize < 1) gridSize = 1;                                  \
    hipLaunchKernelGGL(kernel, dim3(gridSize), dim3(blockSize),      \
                       0, stream, grad, y, out, cols);               \
    return hipGetLastError();

extern "C" hipError_t rocprim_softmax_bwd_float(const float* grad, const float* y, float* out, size_t rows, size_t cols, hipStream_t stream)
{ LAUNCH_ROWS_BWD(soft_max_rows_bwd) }

extern "C" hipError_t rocprim_logsoftmax_bwd_float(const float* grad, const float* y, float* out, size_t rows, size_t cols, hipStream_t stream)
{ LAUNCH_ROWS_BWD(soft_logmax_rows_bwd) }

// ------------------------------------------------------------------
// Dtype conversion kernels: FP32 <-> IEEE half, FP32 <-> BF16.
// Conversions are round-to-nearest-even and handle NaN/Inf/subnormals,
// mirroring fire.core.half.pas.
// ------------------------------------------------------------------
static __device__ __forceinline__ unsigned short f32_to_f16_rne(float f)
{
    return __half_as_ushort(__float2half(f));
}

static __device__ __forceinline__ float f16_to_f32_conv(unsigned short h)
{
    return __half2float(__ushort_as_half(h));
}

static __device__ __forceinline__ unsigned short f32_to_bf16_rne(float f)
{
    unsigned int x = __float_as_uint(f);
    // NaN -> keep sign, quiet NaN payload (matches FIRE's half unit style)
    if ((x & 0x7fffffffu) > 0x7f800000u)
        return (unsigned short)(((x >> 24) & 0x80u) | 0x7fc0u);
    x += 0x7fffu + ((x >> 16) & 1u);   // round-to-nearest-even
    return (unsigned short)(x >> 16);
}

static __device__ __forceinline__ float bf16_to_f32_conv(unsigned short h)
{
    return __uint_as_float(((unsigned int)h) << 16);  // NaN/Inf preserved exactly
}

#define CONV_KERNEL(NAME, SRC_T, DST_T, EXPR)                        \
extern "C" __global__ void NAME(const SRC_T* __restrict__ in,        \
                                DST_T* __restrict__ out, size_t n)   \
{                                                                    \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;              \
    if (idx < n) out[idx] = EXPR(in[idx]);                           \
}

CONV_KERNEL(conv_f16_to_f32, unsigned short, float, f16_to_f32_conv)
CONV_KERNEL(conv_bf16_to_f32, unsigned short, float, bf16_to_f32_conv)
CONV_KERNEL(conv_f32_to_f16, float, unsigned short, f32_to_f16_rne)
CONV_KERNEL(conv_f32_to_bf16, float, unsigned short, f32_to_bf16_rne)

#define LAUNCH_CONV(fnName, kernel, srcT, dstT)                      \
    int blockSize = 256;                                             \
    int gridSize = (int)((n + blockSize - 1) / blockSize);           \
    if (gridSize < 1) gridSize = 1;                                  \
    hipLaunchKernelGGL(kernel, gridSize, blockSize, 0, stream,       \
                       (const srcT*)in, (dstT*)out, n);              \
    return hipGetLastError();

extern "C" hipError_t rocprim_convert_f16_to_f32(const void* in, void* out, size_t n, hipStream_t stream)
{ LAUNCH_CONV(f16, conv_f16_to_f32, unsigned short, float) }

extern "C" hipError_t rocprim_convert_bf16_to_f32(const void* in, void* out, size_t n, hipStream_t stream)
{ LAUNCH_CONV(bf16, conv_bf16_to_f32, unsigned short, float) }

extern "C" hipError_t rocprim_convert_f32_to_f16(const void* in, void* out, size_t n, hipStream_t stream)
{ LAUNCH_CONV(f16, conv_f32_to_f16, float, unsigned short) }

extern "C" hipError_t rocprim_convert_f32_to_bf16(const void* in, void* out, size_t n, hipStream_t stream)
{ LAUNCH_CONV(bf16, conv_f32_to_bf16, float, unsigned short) }

// ---------------------------------------------------------------------------
// Row-wise argmax over the last (contiguous) dimension.
// One block per row: strided per-thread scan, then shared-memory tree
// reduction. Ties resolve to the LOWEST index (strict > keeps the first
// occurrence), matching the CPU/host implementation semantics.
// out[row] = (float) index of the maximum element in in[row*cols .. +cols).
// ---------------------------------------------------------------------------
extern "C" __global__ void row_argmax_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    long cols, long rows)
{
    __shared__ float sval[256];
    __shared__ int   sidx[256];

    long row = blockIdx.x;
    if (row >= rows) return;
    const float* p = in + row * cols;

    float best = -3.4e38f;
    int   idx  = 0;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        float v = p[j];
        if (v > best) { best = v; idx = j; }
    }
    sval[threadIdx.x] = best;
    sidx[threadIdx.x] = idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            // strict > : left (lower thread => lower index region) wins ties
            if (sval[threadIdx.x + s] > sval[threadIdx.x])
            {
                sval[threadIdx.x] = sval[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) out[row] = (float)sidx[0];
}

extern "C" hipError_t rocprim_argmax_float(
    const float* in, float* out, long cols, long rows, hipStream_t stream)
{
    if (rows < 1 || cols < 1) return hipSuccess;
    // Always use a power-of-two block: the shared-memory tree reduction
    // below halves blockDim each step and silently drops threads when
    // blockDim is not a power of two (e.g. cols=3 lost every max that
    // landed on thread 2). Threads with no elements init to -FLT_MAX,
    // so they never win the reduction for real data.
    int threads = 256;
    hipLaunchKernelGGL(row_argmax_kernel, dim3((unsigned)rows), dim3(threads),
                       0, stream, in, out, cols, rows);
    return hipGetLastError();
}




// ---------------------------------------------------------------------------
// Comparison ops: float masks (1.0f / 0.0f), broadcasting via modulo indexing.
// ---------------------------------------------------------------------------

// Helper: broadcast-compatible binary elementwise with modulo indices.
// a[idx % a_n] OP b[idx % b_n] → 1.0f or 0.0f.
#define COMP_KERNEL(NAME, OP)                                                          \
extern "C" __global__ void NAME(                                                       \
    const float* __restrict__ a, const float* __restrict__ b,                          \
    float* __restrict__ out, size_t a_n, size_t b_n, size_t n)                         \
{                                                                                      \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                                \
    if (idx < n) out[idx] = (a[idx % a_n] OP b[idx % b_n]) ? 1.0f : 0.0f;             \
}

COMP_KERNEL(elew_gt_float, >)
COMP_KERNEL(elew_lt_float, <)
COMP_KERNEL(elew_eq_float, ==)
COMP_KERNEL(elew_ge_float, >=)
COMP_KERNEL(elew_le_float, <=)
COMP_KERNEL(elew_ne_float, !=)

#define LAUNCH_COMP(kernel)                                                            \
    int blockSize = 256;                                                               \
    int gridSize = (int)((n + blockSize - 1) / blockSize);                             \
    if (gridSize < 1) gridSize = 1;                                                    \
    hipLaunchKernelGGL(kernel, gridSize, blockSize, 0, stream, a, b, out, a_n, b_n, n);\
    return hipGetLastError();

extern "C" hipError_t rocprim_gt_float(const float* a, const float* b, float* out,
    size_t a_n, size_t b_n, size_t n, hipStream_t stream)  { LAUNCH_COMP(elew_gt_float) }
extern "C" hipError_t rocprim_lt_float(const float* a, const float* b, float* out,
    size_t a_n, size_t b_n, size_t n, hipStream_t stream)  { LAUNCH_COMP(elew_lt_float) }
extern "C" hipError_t rocprim_eq_float(const float* a, const float* b, float* out,
    size_t a_n, size_t b_n, size_t n, hipStream_t stream)  { LAUNCH_COMP(elew_eq_float) }
extern "C" hipError_t rocprim_ge_float(const float* a, const float* b, float* out,
    size_t a_n, size_t b_n, size_t n, hipStream_t stream)  { LAUNCH_COMP(elew_ge_float) }
extern "C" hipError_t rocprim_le_float(const float* a, const float* b, float* out,
    size_t a_n, size_t b_n, size_t n, hipStream_t stream)  { LAUNCH_COMP(elew_le_float) }
extern "C" hipError_t rocprim_ne_float(const float* a, const float* b, float* out,
    size_t a_n, size_t b_n, size_t n, hipStream_t stream)  { LAUNCH_COMP(elew_ne_float) }

// ---------------------------------------------------------------------------
// Where(cond, A, B): out[i] = (cond[i] != 0) ? a[i] : b[i]
// cond is a float mask (1.0/0.0 from comparison ops, but any nonzero works).
// Broadcast via modulo indexing on all three inputs.
// ---------------------------------------------------------------------------
extern "C" __global__ void elew_where_float(
    const float* __restrict__ cond, const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, size_t c_n, size_t a_n, size_t b_n, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = (cond[idx % c_n] != 0.0f) ? a[idx % a_n] : b[idx % b_n];
}

extern "C" hipError_t rocprim_where_float(
    const float* cond, const float* a, const float* b, float* out,
    size_t c_n, size_t a_n, size_t b_n, size_t n, hipStream_t stream)
{
    int blockSize = 256;
    int gridSize = (int)((n + blockSize - 1) / blockSize);
    if (gridSize < 1) gridSize = 1;
    hipLaunchKernelGGL(elew_where_float, gridSize, blockSize, 0, stream,
                       cond, a, b, out, c_n, a_n, b_n, n);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// MaskedFill(data, mask, value): data[i] = (mask[i % m_n] != 0) ? value : data[i]
// In-place: overwrites data where mask is nonzero.
// ---------------------------------------------------------------------------
extern "C" __global__ void elew_maskedfill_float(
    float* __restrict__ data, const float* __restrict__ mask, float value,
    size_t m_n, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && mask[idx % m_n] != 0.0f)
        data[idx] = value;
}

extern "C" hipError_t rocprim_maskedfill_float(
    float* data, const float* mask, float value,
    size_t m_n, size_t n, hipStream_t stream)
{
    int blockSize = 256;
    int gridSize = (int)((n + blockSize - 1) / blockSize);
    if (gridSize < 1) gridSize = 1;
    hipLaunchKernelGGL(elew_maskedfill_float, gridSize, blockSize, 0, stream,
                       data, mask, value, m_n, n);
    return hipGetLastError();
}


// ------------------------------------------------------------------
// Cumulative sum along the last (contiguous) dimension.
// Layout: rows * cols. One thread block per row; the row is processed
// in 256-element chunks with a shared-memory Hillis-Steele scan per
// chunk and a running carry between chunks. Handles any cols.
// ------------------------------------------------------------------
#define CUMSUM_BLOCK 256
extern "C" __global__ void cumsum_rows(const float* __restrict__ in, float* __restrict__ out, size_t cols)
{
    size_t row = blockIdx.x;
    const float* x = in + row * cols;
    float* o = out + row * cols;
    __shared__ float sh[CUMSUM_BLOCK];

    float carry = 0.0f;
    for (size_t base = 0; base < cols; base += CUMSUM_BLOCK)
    {
        size_t i = base + threadIdx.x;
        float v = (i < cols) ? x[i] : 0.0f;
        // Hillis-Steele inclusive scan within the chunk
        sh[threadIdx.x] = v;
        __syncthreads();
        for (int off = 1; off < CUMSUM_BLOCK; off <<= 1)
        {
            float t = (threadIdx.x >= off) ? sh[threadIdx.x - off] : 0.0f;
            __syncthreads();
            sh[threadIdx.x] += t;
            __syncthreads();
        }
        if (i < cols) o[i] = carry + sh[threadIdx.x];
        __syncthreads();
        // carry = last valid element's cumulative value
        if (threadIdx.x == (CUMSUM_BLOCK - 1))
        {
            size_t last = base + CUMSUM_BLOCK - 1;
            if (last < cols) carry += sh[CUMSUM_BLOCK - 1];
            else
            {
                // tail chunk: recompute carry from last valid element
                size_t valid = cols - base; // 1..CUMSUM_BLOCK-1
                carry += sh[valid - 1];
            }
        }
        __syncthreads();
    }
}

extern "C" hipError_t rocprim_cumsum_float(const float* in, float* out, size_t rows, size_t cols, hipStream_t stream)
{
    int blockSize = CUMSUM_BLOCK;
    int gridSize = (int)(rows);
    if (gridSize < 1) gridSize = 1;
    hipLaunchKernelGGL(cumsum_rows, dim3(gridSize), dim3(blockSize), 0, stream, in, out, cols);
    return hipGetLastError();
}

// ------------------------------------------------------------------
// Strided variants: reduce/scan along a non-last dimension of a
// CONTIGUOUS tensor viewed as [outer, len, inner]. Element j of slice
// s = (o*inner + k) lives at (o*len*inner) + j*inner + k, i.e. base =
// (s/inner)*len*inner + (s%inner) with element stride `inner`.
// One block per slice; strided access pattern.
// ------------------------------------------------------------------
extern "C" __global__ void softmax_strided(const float* __restrict__ in, float* __restrict__ out,
                                           long slices, long len, long inner)
{
    long s = blockIdx.x;
    if (s >= slices) return;
    size_t base = (size_t)(s / inner) * (size_t)len * (size_t)inner + (size_t)(s % inner);
    const float* x = in + base;
    float* o = out + base;
    __shared__ float sh[SOFTMAX_BLOCK];

    float mx = -3.4e38f;
    for (long j = threadIdx.x; j < len; j += SOFTMAX_BLOCK)
    { float v = x[(size_t)j * (size_t)inner]; if (v > mx) mx = v; }
    // block max reduce
    float w = warpReduceMaxV(mx);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = w;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float t = -3.4e38f;
        for (int wi = 0; wi < SOFTMAX_BLOCK / 32; ++wi) if (sh[wi] > t) t = sh[wi];
        sh[0] = t;
    }
    __syncthreads();
    mx = sh[0];

    float se = 0.0f;
    for (long j = threadIdx.x; j < len; j += SOFTMAX_BLOCK)
    { float e = expf(x[(size_t)j * (size_t)inner] - mx); o[(size_t)j * (size_t)inner] = e; se += e; }
    w = warpReduceSumV(se);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = w;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float t = 0.0f;
        for (int wi = 0; wi < SOFTMAX_BLOCK / 32; ++wi) t += sh[wi];
        sh[0] = t;
    }
    __syncthreads();
    se = sh[0] + 1e-12f;

    for (long j = threadIdx.x; j < len; j += SOFTMAX_BLOCK)
        o[(size_t)j * (size_t)inner] /= se;
}

extern "C" __global__ void cumsum_strided(const float* __restrict__ in, float* __restrict__ out,
                                          long slices, long len, long inner)
{
    long s = blockIdx.x;
    if (s >= slices) return;
    size_t base = (size_t)(s / inner) * (size_t)len * (size_t)inner + (size_t)(s % inner);
    const float* x = in + base;
    float* o = out + base;
    __shared__ float sh[CUMSUM_BLOCK];

    float carry = 0.0f;
    for (long cbase = 0; cbase < len; cbase += CUMSUM_BLOCK)
    {
        long j = cbase + threadIdx.x;
        float v = (j < len) ? x[(size_t)j * (size_t)inner] : 0.0f;
        sh[threadIdx.x] = v;
        __syncthreads();
        for (int off = 1; off < CUMSUM_BLOCK; off <<= 1)
        {
            float t = (threadIdx.x >= off) ? sh[threadIdx.x - off] : 0.0f;
            __syncthreads();
            sh[threadIdx.x] += t;
            __syncthreads();
        }
        if (j < len) o[(size_t)j * (size_t)inner] = carry + sh[threadIdx.x];
        __syncthreads();
        if (threadIdx.x == CUMSUM_BLOCK - 1)
        {
            long valid = (len - cbase < CUMSUM_BLOCK) ? (len - cbase) : CUMSUM_BLOCK;
            carry += sh[valid - 1];
        }
        __syncthreads();
    }
}

#define LAUNCH_STRIDED(kernel)                                        \
    int blockSize = SOFTMAX_BLOCK;                                    \
    int gridSize = (int)(slices);                                     \
    if (gridSize < 1) gridSize = 1;                                   \
    hipLaunchKernelGGL(kernel, dim3(gridSize), dim3(blockSize),       \
                       0, stream, in, out, (long)slices, (long)len, (long)inner); \
    return hipGetLastError();

// ---------------------------------------------------------------------------
// General strided gather: dst[i] = src[offset(i)] where offset is computed
// by decomposing i into multi-dim coords (row-major) and dotting with the
// source strides. Handles ANY permuted/sliced view materialization.
// shape_d/strides_d are device buffers of dimcount Int64s.
// ---------------------------------------------------------------------------
__global__ void gather_strided_float(const float* src, float* dst,
                                     const long long* shape_d,
                                     const long long* strides_d,
                                     int dimcount, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long off = 0;
    size_t rem = i;
    for (int d = dimcount - 1; d >= 0; --d)
    {
        long long c = (long long)(rem % (size_t)shape_d[d]);
        rem /= (size_t)shape_d[d];
        off += c * strides_d[d];
    }
    dst[i] = src[off];
}

extern "C" hipError_t rocprim_gather_strided_float(
    const float* d_src, float* d_dst,
    const long long* h_shape, const long long* h_strides,
    int dimcount, size_t num_items, hipStream_t stream)
{
    long long* shape_d = nullptr;
    long long* strides_d = nullptr;
    if ((d_src == nullptr) || (d_dst == nullptr) || (h_shape == nullptr) ||
        (h_strides == nullptr) || (dimcount <= 0) || (num_items == 0))
        return hipErrorInvalidValue;
    if (hipMalloc((void**)&shape_d, sizeof(long long) * dimcount) != hipSuccess)
        return hipErrorOutOfMemory;
    if (hipMalloc((void**)&strides_d, sizeof(long long) * dimcount) != hipSuccess)
    {
        hipFree(shape_d);
        return hipErrorOutOfMemory;
    }
    hipMemcpy(shape_d, h_shape, sizeof(long long) * dimcount, hipMemcpyHostToDevice);
    hipMemcpy(strides_d, h_strides, sizeof(long long) * dimcount, hipMemcpyHostToDevice);
    size_t blocks = (num_items + 255) / 256;
    hipLaunchKernelGGL(gather_strided_float, dim3(blocks), dim3(256),
                       0, stream, d_src, d_dst, shape_d, strides_d, dimcount, num_items);
    hipError_t err = hipGetLastError();
    hipDeviceSynchronize();
    hipFree(shape_d);
    hipFree(strides_d);
    return err;
}

//   op 0..3 : out[i] = op(a[i],      b[i mod blen])   (A direct, B broadcast)
//   op 10..13: out[i] = op(a[i mod alen], b[i])       (A broadcast, B direct)
// op: 0/10=add, 1/11=sub, 2/12=mul, 3/13=div.
// alen is ignored for op 0..3; blen is ignored for op 10..13.
// Correct for contiguous row-major operands.
// ---------------------------------------------------------------------------
__global__ void bcast_op_float(const float* a, const float* b, float* out,
                               size_t n, size_t alen, size_t blen, int op)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float av, bv;
    if (op < 10) { av = a[i];         bv = b[i % blen]; }
    else         { av = a[i % alen];  bv = b[i];        op -= 10; }
    switch (op)
    {
        case 0:  out[i] = av + bv; break;
        case 1:  out[i] = av - bv; break;
        case 2:  out[i] = av * bv; break;
        default: out[i] = (bv != 0.0f) ? av / bv : 0.0f; break;
    }
}

extern "C" hipError_t rocprim_bcast_op_float(
    const float* d_a, const float* d_b, float* d_out,
    size_t num_items, size_t alen, size_t blen, int op, hipStream_t stream)
{
    if ((d_a == nullptr) || (d_b == nullptr) || (d_out == nullptr) ||
        (num_items == 0))
        return hipErrorInvalidValue;
    size_t mlen = (op < 10) ? blen : alen;
    if ((mlen == 0) || (num_items % mlen != 0))
        return hipErrorInvalidValue;
    size_t blocks = (num_items + 255) / 256;
    hipLaunchKernelGGL(bcast_op_float, dim3(blocks), dim3(256),
                       0, stream, d_a, d_b, d_out, num_items, alen, blen, op);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// 16-bit variants: FP32 compute in-register, 16-bit store.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void bcast_op16_t(const T* a, const T* b, T* out,
                             size_t n, size_t alen, size_t blen, int op)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float av, bv, r;
    if (op < 10) { av = (float)a[i];        bv = (float)b[i % blen]; }
    else         { av = (float)a[i % alen]; bv = (float)b[i];  op -= 10; }
    switch (op)
    {
        case 0:  r = av + bv; break;
        case 1:  r = av - bv; break;
        case 2:  r = av * bv; break;
        default: r = (bv != 0.0f) ? av / bv : 0.0f; break;
    }
    out[i] = (T)r;
}

static hipError_t bcast_op16_dispatch(const void* a, const void* b, void* out,
                                      size_t n, size_t alen, size_t blen, int op,
                                      bool bf16, hipStream_t stream)
{
    size_t mlen = (op < 10) ? blen : alen;
    if ((mlen == 0) || (n % mlen != 0))
        return hipErrorInvalidValue;
    size_t blocks = (n + 255) / 256;
    if (bf16)
        hipLaunchKernelGGL((bcast_op16_t<hip_bfloat16>), dim3(blocks), dim3(256),
                           0, stream, (const hip_bfloat16*)a, (const hip_bfloat16*)b,
                           (hip_bfloat16*)out, n, alen, blen, op);
    else
        hipLaunchKernelGGL((bcast_op16_t<__half>), dim3(blocks), dim3(256),
                           0, stream, (const __half*)a, (const __half*)b,
                           (__half*)out, n, alen, blen, op);
    return hipGetLastError();
}

extern "C" hipError_t rocprim_bcast_op_half(
    const void* d_a, const void* d_b, void* d_out,
    size_t num_items, size_t alen, size_t blen, int op, hipStream_t stream)
{
    if ((d_a == nullptr) || (d_b == nullptr) || (d_out == nullptr) || (num_items == 0))
        return hipErrorInvalidValue;
    return bcast_op16_dispatch(d_a, d_b, d_out, num_items, alen, blen, op, false, stream);
}

extern "C" hipError_t rocprim_bcast_op_bf16(
    const void* d_a, const void* d_b, void* d_out,
    size_t num_items, size_t alen, size_t blen, int op, hipStream_t stream)
{
    if ((d_a == nullptr) || (d_b == nullptr) || (d_out == nullptr) || (num_items == 0))
        return hipErrorInvalidValue;
    return bcast_op16_dispatch(d_a, d_b, d_out, num_items, alen, blen, op, true, stream);
}

extern "C" hipError_t rocprim_softmax_strided_float(const float* in, float* out,
                                                    size_t slices, size_t len, size_t inner, hipStream_t stream)
{ LAUNCH_STRIDED(softmax_strided) }

extern "C" hipError_t rocprim_cumsum_strided_float(const float* in, float* out,
                                                   size_t slices, size_t len, size_t inner, hipStream_t stream)
{ LAUNCH_STRIDED(cumsum_strided) }

// ---------------------------------------------------------------------------
// Tiled 2D transpose: out[c][r] = in[r][c]. 32x32 tiles, +1 padded shared
// memory to avoid bank conflicts. Coalesced on both read and write.
// ---------------------------------------------------------------------------
#define TR_BLOCK 32
extern "C" __global__ void transpose_tiled(const float* __restrict__ in, float* __restrict__ out,
                                           size_t rows, size_t cols)
{
    __shared__ float tile[TR_BLOCK][TR_BLOCK + 1];
    size_t x = (size_t)blockIdx.x * TR_BLOCK + threadIdx.x;   // col index
    size_t y = (size_t)blockIdx.y * TR_BLOCK + threadIdx.y;   // row index

    if (x < cols)
        for (size_t i = y; i < rows; i += (size_t)gridDim.y * TR_BLOCK)
            tile[threadIdx.y][threadIdx.x] = in[i * cols + x];
    __syncthreads();

    size_t x2 = (size_t)blockIdx.y * TR_BLOCK + threadIdx.x;  // new col = old row
    size_t y2 = (size_t)blockIdx.x * TR_BLOCK + threadIdx.y;  // new row = old col
    if (x2 < rows)
        for (size_t i = y2; i < cols; i += (size_t)gridDim.x * TR_BLOCK)
            out[i * rows + x2] = tile[threadIdx.x][threadIdx.y];
}

extern "C" hipError_t rocprim_transpose_float(const float* in, float* out,
                                              size_t rows, size_t cols, hipStream_t stream)
{
    dim3 block(TR_BLOCK, TR_BLOCK);
    dim3 grid((unsigned)((cols + TR_BLOCK - 1) / TR_BLOCK),
              (unsigned)((rows + TR_BLOCK - 1) / TR_BLOCK));
    hipLaunchKernelGGL(transpose_tiled, grid, block, 0, stream, in, out, rows, cols);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Per-row top-k (largest values, each row's output sorted descending),
// one block per row. Radix select over float->sortable-uint keys with
// 4 x 8-bit histogram passes (each pass re-reads the row, coalesced),
// then compaction with atomic cursors, then a single-thread insertion
// sort of the k selected values. Requires 1 <= k <= TOPK_MAX_K.
// ---------------------------------------------------------------------------
#define TOPK_BLOCK 256
#define TOPK_MAX_K 1024
__device__ __forceinline__ unsigned float_to_sortable(float f)
{
    unsigned bits = __float_as_uint(f);
    // total-order ascending key: negatives flipped, positives biased
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

extern "C" __global__ void row_topk_kernel(const float* __restrict__ in, float* __restrict__ out,
                                           long cols, long rows, int k)
{
    __shared__ unsigned hist[256];
    __shared__ unsigned s_cursor_gt;
    __shared__ unsigned s_eq_taken;
    __shared__ float svals[TOPK_MAX_K];

    long row = blockIdx.x;
    if (row >= rows) return;
    const float* x = in + (size_t)row * cols;

    // We want the k largest values. In ascending sortable order they are the
    // last k, i.e. threshold = (cols-k)-th smallest key (0-based).
    uint target_rank = (uint)(cols - k);
    __shared__ unsigned s_prefix;   // broadcast: thread 0 updates, all read
    uint prefix = 0;
    if (threadIdx.x == 0) s_prefix = 0;
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass)
    {
        int shift = 24 - 8 * pass;
        int shift_check = 32 - 8 * pass;
        if (threadIdx.x < 256) hist[threadIdx.x] = 0;
        __syncthreads();
        prefix = s_prefix;   // every thread reads the broadcast prefix
        for (long j = threadIdx.x; j < cols; j += TOPK_BLOCK)
        {
            unsigned key = float_to_sortable(x[j]);
            if (pass == 0 || ((key >> shift_check) == prefix))
                atomicAdd(&hist[(key >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            uint acc = 0;
            for (int b = 0; b < 256; ++b)
            {
                uint c = hist[b];
                if (acc + c > target_rank) { s_prefix = prefix | ((uint)b << shift); target_rank -= acc; break; }
                acc += c;
            }
        }
        __syncthreads();
    }
    prefix = s_prefix;
    // prefix = sortable key of the k-th largest value (the threshold)

    __shared__ unsigned s_out_cursor;
    if (threadIdx.x == 0) { s_out_cursor = 0; }
    __syncthreads();

    // count of elements strictly greater than threshold
    __shared__ unsigned s_cnt_gt;
    if (threadIdx.x == 0) s_cnt_gt = 0;
    __syncthreads();
    for (long j = threadIdx.x; j < cols; j += TOPK_BLOCK)
    {
        if (float_to_sortable(x[j]) > prefix) atomicAdd(&s_cnt_gt, 1u);
    }
    __syncthreads();
    uint need_eq = (uint)k - s_cnt_gt;   // ties at threshold still required

    // compact: cursor 0 for '>' elements, cursor 1 for '==' elements
    if (threadIdx.x == 0) { s_cursor_gt = 0; s_eq_taken = 0; }
    __syncthreads();
    for (long j = threadIdx.x; j < cols; j += TOPK_BLOCK)
    {
        float v = x[j];
        unsigned key = float_to_sortable(v);
        if (key > prefix)
        {
            uint pos = atomicAdd(&s_cursor_gt, 1u);
            if (pos < (uint)k) svals[pos] = v;
        }
    }
    __syncthreads();
    for (long j = threadIdx.x; j < cols; j += TOPK_BLOCK)
    {
        float v = x[j];
        if (float_to_sortable(v) == prefix)
        {
            uint pos = atomicAdd(&s_eq_taken, 1u);
            if (pos < need_eq)
            {
                uint slot = s_cursor_gt + pos;
                if (slot < (uint)k) svals[slot] = v;
            }
        }
    }
    __syncthreads();

    // sort svals[0..k) descending (single thread; k is small in practice)
    if (threadIdx.x == 0)
    {
        for (int i = 1; i < k; ++i)
        {
            float v = svals[i];
            int jj = i - 1;
            while (jj >= 0 && svals[jj] < v) { svals[jj + 1] = svals[jj]; --jj; }
            svals[jj + 1] = v;
        }
        for (int i = 0; i < k; ++i) out[(size_t)row * k + i] = svals[i];
    }
}

extern "C" hipError_t rocprim_topk_float(const float* in, float* out,
                                         size_t rows, size_t cols, int k, hipStream_t stream)
{
    if (rows < 1 || cols < 1) return hipSuccess;
    if (k < 1 || k > TOPK_MAX_K) return hipErrorInvalidValue;
    hipLaunchKernelGGL(row_topk_kernel, dim3((unsigned)rows), dim3(TOPK_BLOCK),
                       0, stream, in, out, (long)cols, (long)rows, k);
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// Row reductions along the last (contiguous) dimension: sum, max, min.
// One block (256 threads, power-of-two tree reduce) per row. Same structure
// as row_argmax_kernel; out has one element per row.
// ---------------------------------------------------------------------------
#define ROWRED_BLOCK 256

extern "C" __global__ void row_sum_kernel(const float* __restrict__ in, float* __restrict__ out,
                                          long cols, long rows)
{
    __shared__ float sh[ROWRED_BLOCK];
    long row = blockIdx.x;
    if (row >= rows) return;
    const float* p = in + (size_t)row * cols;
    float acc = 0.0f;
    for (long j = threadIdx.x; j < cols; j += ROWRED_BLOCK) acc += p[j];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = ROWRED_BLOCK / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = sh[0];
}

extern "C" __global__ void row_max_kernel(const float* __restrict__ in, float* __restrict__ out,
                                          long cols, long rows)
{
    __shared__ float sh[ROWRED_BLOCK];
    long row = blockIdx.x;
    if (row >= rows) return;
    const float* p = in + (size_t)row * cols;
    float best = -3.4e38f;
    for (long j = threadIdx.x; j < cols; j += ROWRED_BLOCK)
    { float v = p[j]; if (v > best) best = v; }
    sh[threadIdx.x] = best;
    __syncthreads();
    for (int s = ROWRED_BLOCK / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            if (sh[threadIdx.x + s] > sh[threadIdx.x]) sh[threadIdx.x] = sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = sh[0];
}

extern "C" __global__ void row_min_kernel(const float* __restrict__ in, float* __restrict__ out,
                                          long cols, long rows)
{
    __shared__ float sh[ROWRED_BLOCK];
    long row = blockIdx.x;
    if (row >= rows) return;
    const float* p = in + (size_t)row * cols;
    float best = 3.4e38f;
    for (long j = threadIdx.x; j < cols; j += ROWRED_BLOCK)
    { float v = p[j]; if (v < best) best = v; }
    sh[threadIdx.x] = best;
    __syncthreads();
    for (int s = ROWRED_BLOCK / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            if (sh[threadIdx.x + s] < sh[threadIdx.x]) sh[threadIdx.x] = sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = sh[0];
}

#define LAUNCH_ROWRED(kernel)                                          \
    if (rows < 1 || cols < 1) return hipSuccess;                       \
    hipLaunchKernelGGL(kernel, dim3((unsigned)rows), dim3(ROWRED_BLOCK), \
                       0, stream, in, out, (long)cols, (long)rows);    \
    return hipGetLastError();

extern "C" hipError_t rocprim_rowsum_float(const float* in, float* out,
                                           size_t rows, size_t cols, hipStream_t stream)
{ LAUNCH_ROWRED(row_sum_kernel) }

extern "C" hipError_t rocprim_rowmax_float(const float* in, float* out,
                                           size_t rows, size_t cols, hipStream_t stream)
{ LAUNCH_ROWRED(row_max_kernel) }

extern "C" hipError_t rocprim_rowmin_float(const float* in, float* out,
                                           size_t rows, size_t cols, hipStream_t stream)
{ LAUNCH_ROWRED(row_min_kernel) }

// ---------------------------------------------------------------------------
// Repeat-interleave along one dim of a CONTIGUOUS tensor:
// out[o][r*n+rep][k] = in[o][r][k] for rep in [0,n).
// Viewed as [outer, len, inner] with output [outer, len*n, inner].
// One thread per output element (embarrassingly parallel, coalesced).
// ---------------------------------------------------------------------------
extern "C" __global__ void repeat_interleave_kernel(const float* __restrict__ in,
                                                     float* __restrict__ out,
                                                     size_t outer, size_t len, size_t inner, int n)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = outer * len * (size_t)n * inner;
    if (tid >= total) return;
    size_t k  = tid % inner;
    size_t r  = (tid / inner) % ((size_t)len * n);
    size_t o  = tid / ((size_t)inner * len * n);
    out[tid] = in[o * len * inner + (size_t)(r / n) * inner + k];
}

extern "C" hipError_t rocprim_repeat_interleave_float(const float* in, float* out,
                                                      size_t outer, size_t len, size_t inner,
                                                      int n, hipStream_t stream)
{
    size_t total = outer * len * (size_t)n * inner;
    if (total == 0) return hipSuccess;
    dim3 block(256);
    dim3 grid((unsigned)((total + 255) / 256));
    hipLaunchKernelGGL(repeat_interleave_kernel, grid, block, 0, stream,
                       in, out, outer, len, inner, n);
    return hipGetLastError();
}
