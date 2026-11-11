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

unit Test_rocblas_lvl3_strided_batched_64;

{$mode objfpc}{$H+}
{$linklib rocblas}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  TTestRocBLAS_Level3_strided_batched_64 = class(TTestRocBLAS_Base)
  published
    procedure Test_SGEMM_Strided_Batched_64;
    procedure Test_DGEMM_Strided_Batched_64;
    procedure Test_CGEMM_Strided_Batched_64;
    procedure Test_ZGEMM_Strided_Batched_64;
    procedure Test_SSYMM_Strided_Batched_64;
    procedure Test_DSYMM_Strided_Batched_64;
    procedure Test_CSYMM_Strided_Batched_64;
    procedure Test_ZSYMM_Strided_Batched_64;
    procedure Test_STRMM_Strided_Batched_64;
    procedure Test_DTRMM_Strided_Batched_64;
    procedure Test_CTRMM_Strided_Batched_64;
    procedure Test_ZTRMM_Strided_Batched_64;
    procedure Test_STRSM_Strided_Batched_64;
    procedure Test_DTRSM_Strided_Batched_64;
    procedure Test_CTRSM_Strided_Batched_64;
    procedure Test_ZTRSM_Strided_Batched_64;
    procedure Test_SSYRK_Strided_Batched_64;
    procedure Test_DSYRK_Strided_Batched_64;
    procedure Test_CSYRK_Strided_Batched_64;
    procedure Test_ZSYRK_Strided_Batched_64;
    procedure Test_CHERK_Strided_Batched_64;
    procedure Test_ZHERK_Strided_Batched_64;
    procedure Test_SSyr2k_Strided_Batched_64;
    procedure Test_DSyr2k_Strided_Batched_64;
    procedure Test_CSyr2k_Strided_Batched_64;
    procedure Test_ZSyr2k_Strided_Batched_64;
    procedure Test_CHER2K_Strided_Batched_64;
    procedure Test_ZHER2K_Strided_Batched_64;
  end;

implementation

procedure TTestRocBLAS_Level3_strided_batched_64.Test_SGEMM_Strided_Batched_64;
const
  M: Int64 = 128;
  N: Int64 = 128;
  K: Int64 = 128;
  batch_count: Int64 = 3;
  transA = rocblas_operation_none;
  transB = rocblas_operation_none;
var
  lda, ldb, ldc: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of Single;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, beta: Single;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_expected, pC_initial: PSingle;
  temp: Single;
  tolerance: Single = 1e-3;
begin
  // Setup dimensions and leading dimensions
  lda := M;
  ldb := K;
  ldc := M;

  // Setup strides
  stride_A := M * K;
  stride_B := K * N;
  stride_C := M * N;

  // Calculate total sizes
  size_A := batch_count * stride_A * SizeOf(Single);
  size_B := batch_count * stride_B * SizeOf(Single);
  size_C := batch_count * stride_C * SizeOf(Single);

  // Allocate host memory
  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  // Initialize host data
  alpha := 1.5;
  beta := 2.5;
  for i := 0 to Length(h_A) - 1 do h_A[i] := (i mod 100) / 100.0 + 1.0;
  for i := 0 to Length(h_B) - 1 do h_B[i] := (i mod 100) / 100.0 + 1.0;
  for i := 0 to Length(h_C) - 1 do h_C[i] := (i mod 100) / 100.0 + 1.0;

  // Copy initial C to expected C
  System.Move(h_C[0], h_C_expected[0], size_C);

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to M - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        temp := 0;
        for k_idx := 0 to K - 1 do
        begin
          temp := temp + pA[i + k_idx * lda] * pB[k_idx + j * ldb];
        end;
        pC_expected[i + j * ldc] := alpha * temp + beta * pC_initial[i + j * ldc];
      end;
    end;
  end;

  // Allocate device memory
  hip_status := hipMalloc(d_A, size_A);
  AssertEquals('hipMalloc for d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B);
  AssertEquals('hipMalloc for d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C);
  AssertEquals('hipMalloc for d_C failed', HIP_SUCCESS, hip_status);

  // Copy data from host to device
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy to d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy to d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy to d_C failed', HIP_SUCCESS, hip_status);

  // Call rocBLAS function
  blas_status := rocblas_sgemm_strided_batched_64(handle, transA, transB, M, N, K,
    @alpha, d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_sgemm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy result from device to host
  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  // Verify result
  for i := 0 to Length(h_C_result) - 1 do
  begin
    CheckEquals(h_C_expected[i], h_C_result[i], tolerance, Format('Mismatch at index %d', [i]));
  end;

  // Cleanup
  hip_status := hipFree(d_A);
  AssertEquals('hipFree for d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B);
  AssertEquals('hipFree for d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C);
  AssertEquals('hipFree for d_C failed', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_DGEMM_Strided_Batched_64;
const
  M: Int64 = 128;
  N: Int64 = 128;
  K: Int64 = 128;
  batch_count: Int64 = 3;
  transA = rocblas_operation_none;
  transB = rocblas_operation_none;
var
  lda, ldb, ldc: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of Double;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, beta: Double;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_expected, pC_initial: PDouble;
  temp: Double;
  tolerance: Double = 1e-12;
begin
  // Setup dimensions and leading dimensions
  lda := M;
  ldb := K;
  ldc := M;

  // Setup strides
  stride_A := M * K;
  stride_B := K * N;
  stride_C := M * N;

  // Calculate total sizes
  size_A := batch_count * stride_A * SizeOf(Double);
  size_B := batch_count * stride_B * SizeOf(Double);
  size_C := batch_count * stride_C * SizeOf(Double);

  // Allocate host memory
  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  // Initialize host data
  alpha := 1.5;
  beta := 2.5;
  for i := 0 to Length(h_A) - 1 do h_A[i] := (i mod 100) / 100.0 + 1.0;
  for i := 0 to Length(h_B) - 1 do h_B[i] := (i mod 100) / 100.0 + 1.0;
  for i := 0 to Length(h_C) - 1 do h_C[i] := (i mod 100) / 100.0 + 1.0;

  // Copy initial C to expected C
  System.Move(h_C[0], h_C_expected[0], size_C);

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to M - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        temp := 0;
        for k_idx := 0 to K - 1 do
        begin
          temp := temp + pA[i + k_idx * lda] * pB[k_idx + j * ldb];
        end;
        pC_expected[i + j * ldc] := alpha * temp + beta * pC_initial[i + j * ldc];
      end;
    end;
  end;

  // Allocate device memory
  hip_status := hipMalloc(d_A, size_A);
  AssertEquals('hipMalloc for d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B);
  AssertEquals('hipMalloc for d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C);
  AssertEquals('hipMalloc for d_C failed', HIP_SUCCESS, hip_status);

  // Copy data from host to device
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy to d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy to d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy to d_C failed', HIP_SUCCESS, hip_status);

  // Call rocBLAS function
  blas_status := rocblas_dgemm_strided_batched_64(handle, transA, transB, M, N, K,
    @alpha, d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_dgemm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy result from device to host
  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  // Verify result
  for i := 0 to Length(h_C_result) - 1 do
  begin
    CheckEquals(h_C_expected[i], h_C_result[i], tolerance, Format('Mismatch at index %d', [i]));
  end;

  // Cleanup
  hip_status := hipFree(d_A);
  AssertEquals('hipFree for d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B);
  AssertEquals('hipFree for d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C);
  AssertEquals('hipFree for d_C failed', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_CGEMM_Strided_Batched_64;
const
  M: Int64 = 64;
  N: Int64 = 64;
  K: Int64 = 64;
  batch_count: Int64 = 2;
  transA = rocblas_operation_none;
  transB = rocblas_operation_none;
var
  lda, ldb, ldc: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, beta: rocblas_float_complex;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_expected, pC_initial: Procfloat_complex;
  temp, prod: rocblas_float_complex;
  tolerance: Single = 1e-3;
begin
  // Setup dimensions and leading dimensions
  lda := M;
  ldb := K;
  ldc := M;

  // Setup strides
  stride_A := M * K;
  stride_B := K * N;
  stride_C := M * N;

  // Calculate total sizes
  size_A := batch_count * stride_A * SizeOf(rocblas_float_complex);
  size_B := batch_count * stride_B * SizeOf(rocblas_float_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_float_complex);

  // Allocate host memory
  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  // Initialize host data
  alpha.x := 1.5; alpha.y := 0.5;
  beta.x  := 2.5; beta.y  := 1.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 50) / 50.0 + 1.0; h_A[i].y := (i mod 25) / 25.0; end;
  for i := 0 to Length(h_B) - 1 do begin h_B[i].x := (i mod 50) / 50.0 + 1.0; h_B[i].y := (i mod 25) / 25.0; end;
  for i := 0 to Length(h_C) - 1 do begin h_C[i].x := (i mod 50) / 50.0 + 1.0; h_C[i].y := (i mod 25) / 25.0; end;

  // Copy initial C to expected C
  System.Move(h_C[0], h_C_expected[0], size_C);

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to M - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        temp.x := 0.0; temp.y := 0.0;
        for k_idx := 0 to K - 1 do
        begin
          // prod := pA[i,k] * pB[k,j]
          prod := pA[i + k_idx * lda];
          ComplexMul(prod, pB[k_idx + j * ldb]);
          // temp := temp + prod
          ComplexAdd(temp, prod);
        end;
        // pC_expected[i,j] = alpha * temp + beta * pC_initial[i,j]
        ComplexMul(temp, alpha);
        prod := pC_initial[i + j * ldc];
        ComplexMul(prod, beta);
        ComplexAdd(temp, prod);
        pC_expected[i + j * ldc] := temp;
      end;
    end;
  end;

  // Allocate device memory
  hip_status := hipMalloc(d_A, size_A);
  AssertEquals('hipMalloc for d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B);
  AssertEquals('hipMalloc for d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C);
  AssertEquals('hipMalloc for d_C failed', HIP_SUCCESS, hip_status);

  // Copy data from host to device
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy to d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy to d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy to d_C failed', HIP_SUCCESS, hip_status);

  // Call rocBLAS function
  blas_status := rocblas_cgemm_strided_batched_64(handle, transA, transB, M, N, K,
    @alpha, d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_cgemm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy result from device to host
  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  // Verify result
  for i := 0 to Length(h_C_result) - 1 do
  begin
    CheckEquals(h_C_expected[i].x, h_C_result[i].x, tolerance, Format('Mismatch at index %d .x', [i]));
    CheckEquals(h_C_expected[i].y, h_C_result[i].y, tolerance, Format('Mismatch at index %d .y', [i]));
  end;

  // Cleanup
  hip_status := hipFree(d_A);
  AssertEquals('hipFree for d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B);
  AssertEquals('hipFree for d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C);
  AssertEquals('hipFree for d_C failed', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_ZGEMM_Strided_Batched_64;
const
  M: Int64 = 64;
  N: Int64 = 64;
  K: Int64 = 64;
  batch_count: Int64 = 2;
  transA = rocblas_operation_none;
  transB = rocblas_operation_none;
var
  lda, ldb, ldc: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, beta: rocblas_double_complex;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_expected, pC_initial: Procdouble_complex;
  temp, prod: rocblas_double_complex;
  tolerance: Double = 1e-11;
begin
  // Setup dimensions and leading dimensions
  lda := M;
  ldb := K;
  ldc := M;

  // Setup strides
  stride_A := M * K;
  stride_B := K * N;
  stride_C := M * N;

  // Calculate total sizes
  size_A := batch_count * stride_A * SizeOf(rocblas_double_complex);
  size_B := batch_count * stride_B * SizeOf(rocblas_double_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_double_complex);

  // Allocate host memory
  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  // Initialize host data
  alpha.x := 1.5; alpha.y := 0.5;
  beta.x  := 2.5; beta.y  := 1.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 50) / 50.0 + 1.0; h_A[i].y := (i mod 25) / 25.0; end;
  for i := 0 to Length(h_B) - 1 do begin h_B[i].x := (i mod 50) / 50.0 + 1.0; h_B[i].y := (i mod 25) / 25.0; end;
  for i := 0 to Length(h_C) - 1 do begin h_C[i].x := (i mod 50) / 50.0 + 1.0; h_C[i].y := (i mod 25) / 25.0; end;

  // Copy initial C to expected C
  System.Move(h_C[0], h_C_expected[0], size_C);

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to M - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        temp.x := 0.0; temp.y := 0.0;
        for k_idx := 0 to K - 1 do
        begin
          // prod := pA[i,k] * pB[k,j]
          prod := pA[i + k_idx * lda];
          ComplexMul(prod, pB[k_idx + j * ldb]);
          // temp := temp + prod
          ComplexAdd(temp, prod);
        end;
        // pC_expected[i,j] = alpha * temp + beta * pC_initial[i,j]
        ComplexMul(temp, alpha);
        prod := pC_initial[i + j * ldc];
        ComplexMul(prod, beta);
        ComplexAdd(temp, prod);
        pC_expected[i + j * ldc] := temp;
      end;
    end;
  end;

  // Allocate device memory
  hip_status := hipMalloc(d_A, size_A);
  AssertEquals('hipMalloc for d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B);
  AssertEquals('hipMalloc for d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C);
  AssertEquals('hipMalloc for d_C failed', HIP_SUCCESS, hip_status);

  // Copy data from host to device
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy to d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy to d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy to d_C failed', HIP_SUCCESS, hip_status);

  // Call rocBLAS function
  blas_status := rocblas_zgemm_strided_batched_64(handle, transA, transB, M, N, K,
    @alpha, d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_zgemm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy result from device to host
  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  // Verify result
  for i := 0 to Length(h_C_result) - 1 do
  begin
    CheckEquals(h_C_expected[i].x, h_C_result[i].x, tolerance, Format('Mismatch at index %d .x', [i]));
    CheckEquals(h_C_expected[i].y, h_C_result[i].y, tolerance, Format('Mismatch at index %d .y', [i]));
  end;

  // Cleanup
  hip_status := hipFree(d_A);
  AssertEquals('hipFree for d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B);
  AssertEquals('hipFree for d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C);
  AssertEquals('hipFree for d_C failed', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_SSYMM_Strided_Batched_64;
const
  M: Int64 = 72;
  N: Int64 = 72;
  batch_count: Int64 = 2;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
var
  lda, ldb, ldc, size_A_mat, size_B_mat, size_C_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of Single;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, beta: Single;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_initial, pC_expected: PSingle;
  temp, A_ik: Single;
  tolerance: Single = 1e-3;
begin
  // SYMM: C = alpha*A*B + beta*C (side=left) or C = alpha*B*A + beta*C (side=right)
  // A is symmetric.
  // If side is left, A is M x M. If side is right, A is N x N.

  // Setup dimensions
  if side = rocblas_side_left then
  begin
    lda := M;
    size_A_mat := M * M;
  end
  else // side = rocblas_side_right
  begin
    lda := N;
    size_A_mat := N * N;
  end;
  ldb := M;
  ldc := M;
  size_B_mat := M * N;
  size_C_mat := M * N;

  // Setup strides
  stride_A := size_A_mat;
  stride_B := size_B_mat;
  stride_C := size_C_mat;

  // Calculate total allocation sizes
  size_A := batch_count * stride_A * SizeOf(Single);
  size_B := batch_count * stride_B * SizeOf(Single);
  size_C := batch_count * stride_C * SizeOf(Single);

  // Allocate host memory
  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  // Initialize host data
  alpha := 1.5;
  beta := 2.5;
  for i := 0 to Length(h_A) - 1 do h_A[i] := (i mod 100) / 100.0 + 1.0;
  for i := 0 to Length(h_B) - 1 do h_B[i] := (i mod 100) / 100.0 + 1.0;
  for i := 0 to Length(h_C) - 1 do h_C[i] := (i mod 100) / 100.0 + 1.0;

  // Copy initial C to expected C for calculation
  System.Move(h_C[0], h_C_expected[0], size_C);

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    if side = rocblas_side_left then
    begin
      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          temp := 0.0;
          for k_idx := 0 to M - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if i <= k_idx then
                A_ik := pA[i + k_idx * lda]
              else
                A_ik := pA[k_idx + i * lda];
            end
            else // lower
            begin
              if i >= k_idx then
                A_ik := pA[i + k_idx * lda]
              else
                A_ik := pA[k_idx + i * lda];
            end;
            temp := temp + A_ik * pB[k_idx + j * ldb];
          end;
          pC_expected[i + j * ldc] := alpha * temp + beta * pC_initial[i + j * ldc];
        end;
      end;
    end
    else // side = rocblas_side_right
    begin
      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          temp := 0.0;
          for k_idx := 0 to N - 1 do
          begin
             if uplo = rocblas_fill_upper then
             begin
                if j <= k_idx then
                  A_ik := pA[j + k_idx * lda]
                else
                  A_ik := pA[k_idx + j * lda];
             end
             else // lower
             begin
                if j >= k_idx then
                  A_ik := pA[j + k_idx * lda]
                else
                  A_ik := pA[k_idx + j * lda];
             end;
             temp := temp + pB[i + k_idx * ldb] * A_ik;
          end;
          pC_expected[i + j * ldc] := alpha * temp + beta * pC_initial[i + j * ldc];
        end;
      end;
    end;
  end;

  // Allocate and copy to device
  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  // Call rocBLAS
  blas_status := rocblas_ssymm_strided_batched_64(handle, side, uplo, M, N, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_ssymm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy result back
  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  // Verify
  for i := 0 to Length(h_C_result) - 1 do
  begin
    CheckEquals(h_C_expected[i], h_C_result[i], tolerance, Format('Mismatch at index %d', [i]));
  end;

  // Cleanup
  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_DSYMM_Strided_Batched_64;
const
  M: Int64 = 72;
  N: Int64 = 72;
  batch_count: Int64 = 2;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
var
  lda, ldb, ldc, size_A_mat, size_B_mat, size_C_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of Double;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, beta: Double;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_initial, pC_expected: PDouble;
  temp, A_ik: Double;
  tolerance: Double = 1e-12;
begin
  if side = rocblas_side_left then
  begin
    lda := M;
    size_A_mat := M * M;
  end
  else
  begin
    lda := N;
    size_A_mat := N * N;
  end;
  ldb := M;
  ldc := M;
  size_B_mat := M * N;
  size_C_mat := M * N;

  stride_A := size_A_mat;
  stride_B := size_B_mat;
  stride_C := size_C_mat;

  size_A := batch_count * stride_A * SizeOf(Double);
  size_B := batch_count * stride_B * SizeOf(Double);
  size_C := batch_count * stride_C * SizeOf(Double);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha := 1.5;
  beta := 2.5;
  for i := 0 to Length(h_A) - 1 do h_A[i] := (i mod 100) / 100.0 + 1.0;
  for i := 0 to Length(h_B) - 1 do h_B[i] := (i mod 100) / 100.0 + 1.0;
  for i := 0 to Length(h_C) - 1 do h_C[i] := (i mod 100) / 100.0 + 1.0;

  System.Move(h_C[0], h_C_expected[0], size_C);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    if side = rocblas_side_left then
    begin
      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          temp := 0.0;
          for k_idx := 0 to M - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if i <= k_idx then
                A_ik := pA[i + k_idx * lda]
              else
                A_ik := pA[k_idx + i * lda];
            end
            else
            begin
              if i >= k_idx then
                A_ik := pA[i + k_idx * lda]
              else
                A_ik := pA[k_idx + i * lda];
            end;
            temp := temp + A_ik * pB[k_idx + j * ldb];
          end;
          pC_expected[i + j * ldc] := alpha * temp + beta * pC_initial[i + j * ldc];
        end;
      end;
    end
    else
    begin
      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          temp := 0.0;
          for k_idx := 0 to N - 1 do
          begin
             if uplo = rocblas_fill_upper then
             begin
                if j <= k_idx then
                  A_ik := pA[j + k_idx * lda]
                else
                  A_ik := pA[k_idx + j * lda];
             end
             else
             begin
                if j >= k_idx then
                  A_ik := pA[j + k_idx * lda]
                else
                  A_ik := pA[k_idx + j * lda];
             end;
             temp := temp + pB[i + k_idx * ldb] * A_ik;
          end;
          pC_expected[i + j * ldc] := alpha * temp + beta * pC_initial[i + j * ldc];
        end;
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_dsymm_strided_batched_64(handle, side, uplo, M, N, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_dsymm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to Length(h_C_result) - 1 do
  begin
    CheckEquals(h_C_expected[i], h_C_result[i], tolerance, Format('Mismatch at index %d', [i]));
  end;

  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_CSYMM_Strided_Batched_64;
const
  M: Int64 = 32;
  N: Int64 = 32;
  batch_count: Int64 = 2;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
var
  lda, ldb, ldc, size_A_mat, size_B_mat, size_C_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, beta: rocblas_float_complex;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_initial, pC_expected: Procfloat_complex;
  temp, prod, A_ik: rocblas_float_complex;
  tolerance: Single = 1e-5;
begin
  if side = rocblas_side_left then
  begin
    lda := M;
    size_A_mat := M * M;
  end
  else
  begin
    lda := N;
    size_A_mat := N * N;
  end;
  ldb := M;
  ldc := M;
  size_B_mat := M * N;
  size_C_mat := M * N;

  stride_A := size_A_mat;
  stride_B := size_B_mat;
  stride_C := size_C_mat;

  size_A := batch_count * stride_A * SizeOf(rocblas_float_complex);
  size_B := batch_count * stride_B * SizeOf(rocblas_float_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_float_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha.x := 1.5; alpha.y := 0.5;
  beta.x  := 2.5; beta.y  := 1.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20) / 20.0; h_A[i].y := (i mod 10) / 10.0; end;
  for i := 0 to Length(h_B) - 1 do begin h_B[i].x := (i mod 20) / 20.0; h_B[i].y := (i mod 10) / 10.0; end;
  for i := 0 to Length(h_C) - 1 do begin h_C[i].x := (i mod 20) / 20.0; h_C[i].y := (i mod 10) / 10.0; end;

  System.Move(h_C[0], h_C_expected[0], size_C);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    if side = rocblas_side_left then
    begin
      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          temp.x := 0.0; temp.y := 0.0;
          for k_idx := 0 to M - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if i <= k_idx then
                A_ik := pA[i + k_idx * lda]
              else
                A_ik := pA[k_idx + i * lda];
            end
            else
            begin
              if i >= k_idx then
                A_ik := pA[i + k_idx * lda]
              else
                A_ik := pA[k_idx + i * lda];
            end;
            prod := A_ik;
            ComplexMul(prod, pB[k_idx + j * ldb]);
            ComplexAdd(temp, prod);
          end;
          prod := pC_initial[i + j * ldc];
          ComplexMul(prod, beta);
          ComplexMul(temp, alpha);
          ComplexAdd(temp, prod);
          pC_expected[i + j * ldc] := temp;
        end;
      end;
    end
    else // side = rocblas_side_right
    begin
      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          temp.x := 0.0; temp.y := 0.0;
          for k_idx := 0 to N - 1 do
          begin
             if uplo = rocblas_fill_upper then
             begin
                if j <= k_idx then
                  A_ik := pA[j + k_idx * lda]
                else
                  A_ik := pA[k_idx + j * lda];
             end
             else
             begin
                if j >= k_idx then
                  A_ik := pA[j + k_idx * lda]
                else
                  A_ik := pA[k_idx + j * lda];
             end;
             prod := pB[i + k_idx * ldb];
             ComplexMul(prod, A_ik);
             ComplexAdd(temp, prod);
          end;
          prod := pC_initial[i + j * ldc];
          ComplexMul(prod, beta);
          ComplexMul(temp, alpha);
          ComplexAdd(temp, prod);
          pC_expected[i + j * ldc] := temp;
        end;
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_csymm_strided_batched_64(handle, side, uplo, M, N, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_csymm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to Length(h_C_result) - 1 do
  begin
    CheckEquals(h_C_expected[i].x, h_C_result[i].x, tolerance, Format('Mismatch at index %d .x', [i]));
    CheckEquals(h_C_expected[i].y, h_C_result[i].y, tolerance, Format('Mismatch at index %d .y', [i]));
  end;

  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_ZSYMM_Strided_Batched_64;
const
  M: Int64 = 32;
  N: Int64 = 32;
  batch_count: Int64 = 2;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
var
  lda, ldb, ldc, size_A_mat, size_B_mat, size_C_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, beta: rocblas_double_complex;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_initial, pC_expected: Procdouble_complex;
  temp, prod, A_ik: rocblas_double_complex;
  tolerance: Double = 1e-11;
begin
  if side = rocblas_side_left then
  begin
    lda := M;
    size_A_mat := M * M;
  end
  else
  begin
    lda := N;
    size_A_mat := N * N;
  end;
  ldb := M;
  ldc := M;
  size_B_mat := M * N;
  size_C_mat := M * N;

  stride_A := size_A_mat;
  stride_B := size_B_mat;
  stride_C := size_C_mat;

  size_A := batch_count * stride_A * SizeOf(rocblas_double_complex);
  size_B := batch_count * stride_B * SizeOf(rocblas_double_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_double_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha.x := 1.5; alpha.y := 0.5;
  beta.x  := 2.5; beta.y  := 1.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20) / 20.0; h_A[i].y := (i mod 10) / 10.0; end;
  for i := 0 to Length(h_B) - 1 do begin h_B[i].x := (i mod 20) / 20.0; h_B[i].y := (i mod 10) / 10.0; end;
  for i := 0 to Length(h_C) - 1 do begin h_C[i].x := (i mod 20) / 20.0; h_C[i].y := (i mod 10) / 10.0; end;

  System.Move(h_C[0], h_C_expected[0], size_C);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    if side = rocblas_side_left then
    begin
      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          temp.x := 0.0; temp.y := 0.0;
          for k_idx := 0 to M - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if i <= k_idx then
                A_ik := pA[i + k_idx * lda]
              else
                A_ik := pA[k_idx + i * lda];
            end
            else
            begin
              if i >= k_idx then
                A_ik := pA[i + k_idx * lda]
              else
                A_ik := pA[k_idx + i * lda];
            end;
            prod := A_ik;
            ComplexMul(prod, pB[k_idx + j * ldb]);
            ComplexAdd(temp, prod);
          end;
          prod := pC_initial[i + j * ldc];
          ComplexMul(prod, beta);
          ComplexMul(temp, alpha);
          ComplexAdd(temp, prod);
          pC_expected[i + j * ldc] := temp;
        end;
      end;
    end
    else
    begin
      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          temp.x := 0.0; temp.y := 0.0;
          for k_idx := 0 to N - 1 do
          begin
             if uplo = rocblas_fill_upper then
             begin
                if j <= k_idx then
                  A_ik := pA[j + k_idx * lda]
                else
                  A_ik := pA[k_idx + j * lda];
             end
             else
             begin
                if j >= k_idx then
                  A_ik := pA[j + k_idx * lda]
                else
                  A_ik := pA[k_idx + j * lda];
             end;
             prod := pB[i + k_idx * ldb];
             ComplexMul(prod, A_ik);
             ComplexAdd(temp, prod);
          end;
          prod := pC_initial[i + j * ldc];
          ComplexMul(prod, beta);
          ComplexMul(temp, alpha);
          ComplexAdd(temp, prod);
          pC_expected[i + j * ldc] := temp;
        end;
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_zsymm_strided_batched_64(handle, side, uplo, M, N, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_zsymm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to Length(h_C_result) - 1 do
  begin
    CheckEquals(h_C_expected[i].x, h_C_result[i].x, tolerance, Format('Mismatch at index %d .x', [i]));
    CheckEquals(h_C_expected[i].y, h_C_result[i].y, tolerance, Format('Mismatch at index %d .y', [i]));
  end;

  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_STRMM_Strided_Batched_64;
const
  M: Int64 = 64;
  N: Int64 = 64;
  batch_count: Int64 = 3;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  lda, ldb, ldc, size_A_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C_result, h_C_expected: array of Single;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha: Single;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_expected: PSingle;
  temp, A_val: Single;
  tolerance: Single = 1e-3;
  K: Int64;
begin
  // TRMM: C := alpha * op(A) * B (side=left) or C := alpha * B * op(A) (side=right)
  // A is triangular, B is general, C is general output.
  if side = rocblas_side_left then K := M else K := N;

  // Setup dimensions
  lda := K;
  ldb := M;
  ldc := M;
  size_A_mat := K * K;

  // Setup strides
  stride_A := size_A_mat;
  stride_B := M * N;
  stride_C := M * N;

  // Calculate total allocation sizes
  size_A := batch_count * stride_A * SizeOf(Single);
  size_B := batch_count * stride_B * SizeOf(Single);
  size_C := batch_count * stride_C * SizeOf(Single);

  // Allocate host memory
  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  // Initialize host data
  alpha := 1.5;
  for i := 0 to Length(h_A) - 1 do h_A[i] := (i mod 100) / 100.0 + 1.0;
  for i := 0 to Length(h_B) - 1 do h_B[i] := (i mod 100) / 100.0 + 1.0;
  FillChar(h_C_expected[0], Length(h_C_expected) * SizeOf(Single), 0);

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to M - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        temp := 0.0;
        if side = rocblas_side_left then
        begin
          for k_idx := 0 to K - 1 do // K = M
          begin
            if uplo = rocblas_fill_upper then
            begin
              if i > k_idx then A_val := 0.0
              else if (i = k_idx) and (diag = rocblas_diagonal_unit) then A_val := 1.0
              else A_val := pA[i + k_idx * lda];
            end
            else // lower
            begin
              if i < k_idx then A_val := 0.0
              else if (i = k_idx) and (diag = rocblas_diagonal_unit) then A_val := 1.0
              else A_val := pA[i + k_idx * lda];
            end;
            temp := temp + A_val * pB[k_idx + j * ldb];
          end;
        end
        else // side = rocblas_side_right
        begin
          for k_idx := 0 to K - 1 do // K = N
          begin
            if uplo = rocblas_fill_upper then
            begin
              if k_idx > j then A_val := 0.0
              else if (k_idx = j) and (diag = rocblas_diagonal_unit) then A_val := 1.0
              else A_val := pA[k_idx + j * lda];
            end
            else // lower
            begin
              if k_idx < j then A_val := 0.0
              else if (k_idx = j) and (diag = rocblas_diagonal_unit) then A_val := 1.0
              else A_val := pA[k_idx + j * lda];
            end;
            temp := temp + pB[i + k_idx * ldb] * A_val;
          end;
        end;
        pC_expected[i + j * ldc] := alpha * temp;
      end;
    end;
  end;

  // Allocate and copy to device
  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);

  // Call rocBLAS
  blas_status := rocblas_strmm_strided_batched_64(handle, side, uplo, transA, diag, M, N, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_strmm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy result back
  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  // Verify
  for i := 0 to Length(h_C_result) - 1 do
  begin
    CheckEquals(h_C_expected[i], h_C_result[i], tolerance, Format('Mismatch at index %d', [i]));
  end;

  // Cleanup
  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_DTRMM_Strided_Batched_64;
const
  M: Int64 = 64;
  N: Int64 = 64;
  batch_count: Int64 = 3;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  lda, ldb, ldc, size_A_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C_result, h_C_expected: array of Double;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha: Double;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_expected: PDouble;
  temp, A_val: Double;
  tolerance: Double = 1e-12;
  K: Int64;
begin
  if side = rocblas_side_left then K := M else K := N;

  lda := K;
  ldb := M;
  ldc := M;
  size_A_mat := K * K;

  stride_A := size_A_mat;
  stride_B := M * N;
  stride_C := M * N;

  size_A := batch_count * stride_A * SizeOf(Double);
  size_B := batch_count * stride_B * SizeOf(Double);
  size_C := batch_count * stride_C * SizeOf(Double);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha := 1.5;
  for i := 0 to Length(h_A) - 1 do h_A[i] := (i mod 100) / 100.0 + 1.0;
  for i := 0 to Length(h_B) - 1 do h_B[i] := (i mod 100) / 100.0 + 1.0;
  FillChar(h_C_expected[0], Length(h_C_expected) * SizeOf(Double), 0);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to M - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        temp := 0.0;
        if side = rocblas_side_left then
        begin
          for k_idx := 0 to K - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if i > k_idx then A_val := 0.0
              else if (i = k_idx) and (diag = rocblas_diagonal_unit) then A_val := 1.0
              else A_val := pA[i + k_idx * lda];
            end
            else
            begin
              if i < k_idx then A_val := 0.0
              else if (i = k_idx) and (diag = rocblas_diagonal_unit) then A_val := 1.0
              else A_val := pA[i + k_idx * lda];
            end;
            temp := temp + A_val * pB[k_idx + j * ldb];
          end;
        end
        else
        begin
          for k_idx := 0 to K - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if k_idx > j then A_val := 0.0
              else if (k_idx = j) and (diag = rocblas_diagonal_unit) then A_val := 1.0
              else A_val := pA[k_idx + j * lda];
            end
            else
            begin
              if k_idx < j then A_val := 0.0
              else if (k_idx = j) and (diag = rocblas_diagonal_unit) then A_val := 1.0
              else A_val := pA[k_idx + j * lda];
            end;
            temp := temp + pB[i + k_idx * ldb] * A_val;
          end;
        end;
        pC_expected[i + j * ldc] := alpha * temp;
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);

  blas_status := rocblas_dtrmm_strided_batched_64(handle, side, uplo, transA, diag, M, N, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_dtrmm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to Length(h_C_result) - 1 do
  begin
    CheckEquals(h_C_expected[i], h_C_result[i], tolerance, Format('Mismatch at index %d', [i]));
  end;

  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_CTRMM_Strided_Batched_64;
const
  M: Int64 = 32;
  N: Int64 = 32;
  batch_count: Int64 = 2;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  lda, ldb, ldc, size_A_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C_result, h_C_expected: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha: rocblas_float_complex;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_expected: Procfloat_complex;
  temp, prod, A_val: rocblas_float_complex;
  tolerance: Single = 1e-5;
  K: Int64;
begin
  if side = rocblas_side_left then K := M else K := N;

  lda := K;
  ldb := M;
  ldc := M;
  size_A_mat := K * K;

  stride_A := size_A_mat;
  stride_B := M * N;
  stride_C := M * N;

  size_A := batch_count * stride_A * SizeOf(rocblas_float_complex);
  size_B := batch_count * stride_B * SizeOf(rocblas_float_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_float_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha.x := 1.5; alpha.y := 0.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20)/20.0; h_A[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_B) - 1 do begin h_B[i].x := (i mod 20)/20.0; h_B[i].y := (i mod 10)/10.0; end;
  FillChar(h_C_expected[0], Length(h_C_expected) * SizeOf(rocblas_float_complex), 0);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to M - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        temp.x := 0.0; temp.y := 0.0;
        if side = rocblas_side_left then
        begin
          for k_idx := 0 to K - 1 do
          begin
            A_val.x := 0.0; A_val.y := 0.0;
            if uplo = rocblas_fill_upper then
            begin
              if i <= k_idx then
              begin
                if (i = k_idx) and (diag = rocblas_diagonal_unit) then A_val.x := 1.0
                else A_val := pA[i + k_idx * lda];
              end;
            end
            else
            begin
              if i >= k_idx then
              begin
                if (i = k_idx) and (diag = rocblas_diagonal_unit) then A_val.x := 1.0
                else A_val := pA[i + k_idx * lda];
              end;
            end;
            prod := A_val;
            ComplexMul(prod, pB[k_idx + j * ldb]);
            ComplexAdd(temp, prod);
          end;
        end
        else // side = rocblas_side_right
        begin
          for k_idx := 0 to K - 1 do
          begin
            A_val.x := 0.0; A_val.y := 0.0;
            if uplo = rocblas_fill_upper then
            begin
              if k_idx <= j then
              begin
                if (k_idx = j) and (diag = rocblas_diagonal_unit) then A_val.x := 1.0
                else A_val := pA[k_idx + j * lda];
              end;
            end
            else
            begin
              if k_idx >= j then
              begin
                if (k_idx = j) and (diag = rocblas_diagonal_unit) then A_val.x := 1.0
                else A_val := pA[k_idx + j * lda];
              end;
            end;
            prod := pB[i + k_idx * ldb];
            ComplexMul(prod, A_val);
            ComplexAdd(temp, prod);
          end;
        end;
        ComplexMul(temp, alpha);
        pC_expected[i + j * ldc] := temp;
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);

  blas_status := rocblas_ctrmm_strided_batched_64(handle, side, uplo, transA, diag, M, N, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_ctrmm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to Length(h_C_result) - 1 do
  begin
    CheckEquals(h_C_expected[i].x, h_C_result[i].x, tolerance, Format('Mismatch at index %d .x', [i]));
    CheckEquals(h_C_expected[i].y, h_C_result[i].y, tolerance, Format('Mismatch at index %d .y', [i]));
  end;

  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_ZTRMM_Strided_Batched_64;
const
  M: Int64 = 32;
  N: Int64 = 32;
  batch_count: Int64 = 2;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  lda, ldb, ldc, size_A_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C_result, h_C_expected: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha: rocblas_double_complex;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_expected: Procdouble_complex;
  temp, prod, A_val: rocblas_double_complex;
  tolerance: Double = 1e-11;
  K: Int64;
begin
  if side = rocblas_side_left then K := M else K := N;

  lda := K;
  ldb := M;
  ldc := M;
  size_A_mat := K * K;

  stride_A := size_A_mat;
  stride_B := M * N;
  stride_C := M * N;

  size_A := batch_count * stride_A * SizeOf(rocblas_double_complex);
  size_B := batch_count * stride_B * SizeOf(rocblas_double_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_double_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha.x := 1.5; alpha.y := 0.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20)/20.0; h_A[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_B) - 1 do begin h_B[i].x := (i mod 20)/20.0; h_B[i].y := (i mod 10)/10.0; end;
  FillChar(h_C_expected[0], Length(h_C_expected) * SizeOf(rocblas_double_complex), 0);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to M - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        temp.x := 0.0; temp.y := 0.0;
        if side = rocblas_side_left then
        begin
          for k_idx := 0 to K - 1 do
          begin
            A_val.x := 0.0; A_val.y := 0.0;
            if uplo = rocblas_fill_upper then
            begin
              if i <= k_idx then
              begin
                if (i = k_idx) and (diag = rocblas_diagonal_unit) then A_val.x := 1.0
                else A_val := pA[i + k_idx * lda];
              end;
            end
            else
            begin
              if i >= k_idx then
              begin
                if (i = k_idx) and (diag = rocblas_diagonal_unit) then A_val.x := 1.0
                else A_val := pA[i + k_idx * lda];
              end;
            end;
            prod := A_val;
            ComplexMul(prod, pB[k_idx + j * ldb]);
            ComplexAdd(temp, prod);
          end;
        end
        else
        begin
          for k_idx := 0 to K - 1 do
          begin
            A_val.x := 0.0; A_val.y := 0.0;
            if uplo = rocblas_fill_upper then
            begin
              if k_idx <= j then
              begin
                if (k_idx = j) and (diag = rocblas_diagonal_unit) then A_val.x := 1.0
                else A_val := pA[k_idx + j * lda];
              end;
            end
            else
            begin
              if k_idx >= j then
              begin
                if (k_idx = j) and (diag = rocblas_diagonal_unit) then A_val.x := 1.0
                else A_val := pA[k_idx + j * lda];
              end;
            end;
            prod := pB[i + k_idx * ldb];
            ComplexMul(prod, A_val);
            ComplexAdd(temp, prod);
          end;
        end;
        ComplexMul(temp, alpha);
        pC_expected[i + j * ldc] := temp;
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);

  blas_status := rocblas_ztrmm_strided_batched_64(handle, side, uplo, transA, diag, M, N, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_ztrmm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to Length(h_C_result) - 1 do
  begin
    CheckEquals(h_C_expected[i].x, h_C_result[i].x, tolerance, Format('Mismatch at index %d .x', [i]));
    CheckEquals(h_C_expected[i].y, h_C_result[i].y, tolerance, Format('Mismatch at index %d .y', [i]));
  end;

  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_STRSM_Strided_Batched_64;
const
  M: Int64 = 64;
  N: Int64 = 64;
  batch_count: Int64 = 2;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  lda, ldb, size_A_mat, K: Int64;
  stride_A, stride_B: rocblas_stride;
  h_A, h_B_input, h_X_initial, h_X_expected, h_B_result: array of Single;
  d_A, d_B: hipDeviceptr_t;
  size_A, size_B: SizeUInt;
  alpha: Single;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB_input, pX_initial: PSingle;
  temp, A_val: Single;
  tolerance: Single = 1e-4;
begin
  // Reverse verification strategy per guide
  if side = rocblas_side_left then K := M else K := N;
  lda := K;
  ldb := M;
  size_A_mat := K * K;
  stride_A := size_A_mat;
  stride_B := M * N;
  size_A := batch_count * stride_A * SizeOf(Single);
  size_B := batch_count * stride_B * SizeOf(Single);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_X_initial, batch_count * stride_B);
  SetLength(h_B_input, batch_count * stride_B);
  SetLength(h_X_expected, batch_count * stride_B);
  SetLength(h_B_result, batch_count * stride_B);

  alpha := 1.5;
  for i := 0 to Length(h_A) - 1 do h_A[i] := (i mod 100) / 50.0 + 1.0;
  for i := 0 to Length(h_X_initial) - 1 do h_X_initial[i] := (i mod 100) / 100.0 + 1.0;
  for b := 0 to batch_count - 1 do
    for i := 0 to K - 1 do
      h_A[b * stride_A + i + i * lda] := K; // Make A diagonally dominant

  // Calculate B_input = op(A) * X_initial (CPU)
  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pX_initial := @h_X_initial[b * stride_B];
    pB_input   := @h_B_input[b * stride_B];

    for i := 0 to M - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        temp := 0.0;
        if side = rocblas_side_left then
        begin
          for k_idx := 0 to K - 1 do
          begin
            A_val := 0.0;
            if uplo = rocblas_fill_upper then
            begin
              if i <= k_idx then A_val := pA[i + k_idx * lda];
            end
            else // lower
            begin
              if i >= k_idx then A_val := pA[i + k_idx * lda];
            end;
            if (i = k_idx) and (diag = rocblas_diagonal_unit) then A_val := 1.0;
            temp := temp + A_val * pX_initial[k_idx + j * ldb];
          end;
        end
        else // side = rocblas_side_right
        begin
          for k_idx := 0 to K - 1 do
          begin
            A_val := 0.0;
            if uplo = rocblas_fill_upper then
            begin
               if k_idx <= j then A_val := pA[k_idx + j * lda];
            end
            else
            begin
               if k_idx >= j then A_val := pA[k_idx + j * lda];
            end;
            if (k_idx = j) and (diag = rocblas_diagonal_unit) then A_val := 1.0;
            temp := temp + pX_initial[i + k_idx * ldb] * A_val;
          end;
        end;
        pB_input[i + j * ldb] := temp;
      end;
    end;
  end;

  // Calculate X_expected = alpha * X_initial
  for i := 0 to Length(h_X_initial) - 1 do
    h_X_expected[i] := alpha * h_X_initial[i];

  // Execute on GPU
  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B_input[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);

  blas_status := rocblas_strsm_strided_batched_64(handle, side, uplo, transA, diag, M, N, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, batch_count);
  AssertEquals('rocblas_strsm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_B_result[0], d_B, size_B, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_B failed', HIP_SUCCESS, hip_status);

  // Verify
  for i := 0 to Length(h_B_result) - 1 do
  begin
    CheckEquals(h_X_expected[i], h_B_result[i], tolerance, Format('Mismatch at index %d', [i]));
  end;

  hipFree(d_A); hipFree(d_B);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_DTRSM_Strided_Batched_64;
const
  M: Int64 = 64;
  N: Int64 = 64;
  batch_count: Int64 = 2;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  lda, ldb, size_A_mat, K: Int64;
  stride_A, stride_B: rocblas_stride;
  h_A, h_B_input, h_X_initial, h_X_expected, h_B_result: array of Double;
  d_A, d_B: hipDeviceptr_t;
  size_A, size_B: SizeUInt;
  alpha: Double;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB_input, pX_initial: PDouble;
  temp, A_val: Double;
  tolerance: Double = 1e-11;
begin
  if side = rocblas_side_left then K := M else K := N;
  lda := K;
  ldb := M;
  size_A_mat := K * K;
  stride_A := size_A_mat;
  stride_B := M * N;
  size_A := batch_count * stride_A * SizeOf(Double);
  size_B := batch_count * stride_B * SizeOf(Double);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_X_initial, batch_count * stride_B);
  SetLength(h_B_input, batch_count * stride_B);
  SetLength(h_X_expected, batch_count * stride_B);
  SetLength(h_B_result, batch_count * stride_B);

  alpha := 1.5;
  for i := 0 to Length(h_A) - 1 do h_A[i] := (i mod 100) / 50.0 + 1.0;
  for i := 0 to Length(h_X_initial) - 1 do h_X_initial[i] := (i mod 100) / 100.0 + 1.0;
  for b := 0 to batch_count - 1 do
    for i := 0 to K - 1 do
      h_A[b * stride_A + i + i * lda] := K;

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pX_initial := @h_X_initial[b * stride_B];
    pB_input   := @h_B_input[b * stride_B];
    for i := 0 to M - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        temp := 0.0;
        if side = rocblas_side_left then
        begin
          for k_idx := 0 to K - 1 do
          begin
            A_val := 0.0;
            if uplo = rocblas_fill_upper then
            begin
              if i <= k_idx then A_val := pA[i + k_idx * lda];
            end
            else
            begin
              if i >= k_idx then A_val := pA[i + k_idx * lda];
            end;
            if (i=k_idx) and (diag = rocblas_diagonal_unit) then A_val := 1.0;
            temp := temp + A_val * pX_initial[k_idx + j * ldb];
          end;
        end
        else
        begin
          for k_idx := 0 to K - 1 do
          begin
            A_val := 0.0;
            if uplo = rocblas_fill_upper then
            begin
               if k_idx <= j then A_val := pA[k_idx + j * lda];
            end
            else
            begin
               if k_idx >= j then A_val := pA[k_idx + j * lda];
            end;
            if (k_idx=j) and (diag = rocblas_diagonal_unit) then A_val := 1.0;
            temp := temp + pX_initial[i + k_idx * ldb] * A_val;
          end;
        end;
        pB_input[i + j * ldb] := temp;
      end;
    end;
  end;

  for i := 0 to Length(h_X_initial) - 1 do
    h_X_expected[i] := alpha * h_X_initial[i];

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B_input[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);

  blas_status := rocblas_dtrsm_strided_batched_64(handle, side, uplo, transA, diag, M, N, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, batch_count);
  AssertEquals('rocblas_dtrsm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_B_result[0], d_B, size_B, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_B failed', HIP_SUCCESS, hip_status);

  for i := 0 to Length(h_B_result) - 1 do
  begin
    CheckEquals(h_X_expected[i], h_B_result[i], tolerance, Format('Mismatch at index %d', [i]));
  end;

  hipFree(d_A); hipFree(d_B);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_CTRSM_Strided_Batched_64;
const
  M: Int64 = 32;
  N: Int64 = 32;
  batch_count: Int64 = 2;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  lda, ldb, size_A_mat, K: Int64;
  stride_A, stride_B: rocblas_stride;
  h_A, h_B_input, h_X_initial, h_X_expected, h_B_result: array of rocblas_float_complex;
  d_A, d_B: hipDeviceptr_t;
  size_A, size_B: SizeUInt;
  alpha: rocblas_float_complex;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB_input, pX_initial: Procfloat_complex;
  temp, prod, A_val: rocblas_float_complex;
  tolerance: Single = 1e-4;
begin
  if side = rocblas_side_left then K := M else K := N;
  lda := K;
  ldb := M;
  size_A_mat := K * K;
  stride_A := size_A_mat;
  stride_B := M * N;
  size_A := batch_count * stride_A * SizeOf(rocblas_float_complex);
  size_B := batch_count * stride_B * SizeOf(rocblas_float_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_X_initial, batch_count * stride_B);
  SetLength(h_B_input, batch_count * stride_B);
  SetLength(h_X_expected, batch_count * stride_B);
  SetLength(h_B_result, batch_count * stride_B);

  alpha.x := 1.5; alpha.y := 0.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20)/20.0+0.5; h_A[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_X_initial) - 1 do begin h_X_initial[i].x := (i mod 20)/20.0; h_X_initial[i].y := (i mod 10)/10.0; end;
  for b := 0 to batch_count - 1 do
    for i := 0 to K - 1 do
      h_A[b * stride_A + i + i * lda].x := K;

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pX_initial := @h_X_initial[b * stride_B];
    pB_input   := @h_B_input[b * stride_B];
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
      begin
        temp.x := 0.0; temp.y := 0.0;
        if side = rocblas_side_left then
        begin
          for k_idx := 0 to K - 1 do
          begin
            A_val.x := 0.0; A_val.y := 0.0;
            if uplo = rocblas_fill_upper then
            begin
              if i <= k_idx then A_val := pA[i + k_idx * lda];
            end
            else
            begin
              if i >= k_idx then A_val := pA[i + k_idx * lda];
            end;
            if (i = k_idx) and (diag = rocblas_diagonal_unit) then begin A_val.x := 1.0; A_val.y := 0.0; end;
            prod := A_val; ComplexMul(prod, pX_initial[k_idx + j * ldb]); ComplexAdd(temp, prod);
          end;
        end
        else
        begin
          for k_idx := 0 to K - 1 do
          begin
            A_val.x := 0.0; A_val.y := 0.0;
            if uplo = rocblas_fill_upper then
            begin
              if k_idx <= j then A_val := pA[k_idx + j * lda];
            end
            else
            begin
              if k_idx >= j then A_val := pA[k_idx + j * lda];
            end;
            if (k_idx=j) and (diag = rocblas_diagonal_unit) then begin A_val.x := 1.0; A_val.y := 0.0; end;
            prod := pX_initial[i + k_idx * ldb]; ComplexMul(prod, A_val); ComplexAdd(temp, prod);
          end;
        end;
        pB_input[i + j * ldb] := temp;
      end;
  end;

  for i := 0 to Length(h_X_initial) - 1 do
  begin
    h_X_expected[i] := h_X_initial[i];
    ComplexMul(h_X_expected[i], alpha);
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B_input[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);

  blas_status := rocblas_ctrsm_strided_batched_64(handle, side, uplo, transA, diag, M, N, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, batch_count);
  AssertEquals('rocblas_ctrsm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_B_result[0], d_B, size_B, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_B failed', HIP_SUCCESS, hip_status);

  for i := 0 to Length(h_B_result) - 1 do
  begin
    CheckEquals(h_X_expected[i].x, h_B_result[i].x, tolerance, Format('Mismatch at index %d .x', [i]));
    CheckEquals(h_X_expected[i].y, h_B_result[i].y, tolerance, Format('Mismatch at index %d .y', [i]));
  end;

  hipFree(d_A); hipFree(d_B);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_ZTRSM_Strided_Batched_64;
const
  M: Int64 = 32;
  N: Int64 = 32;
  batch_count: Int64 = 2;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  lda, ldb, size_A_mat, K: Int64;
  stride_A, stride_B: rocblas_stride;
  h_A, h_B_input, h_X_initial, h_X_expected, h_B_result: array of rocblas_double_complex;
  d_A, d_B: hipDeviceptr_t;
  size_A, size_B: SizeUInt;
  alpha: rocblas_double_complex;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB_input, pX_initial: Procdouble_complex;
  temp, prod, A_val: rocblas_double_complex;
  tolerance: Double = 1e-11;
begin
  if side = rocblas_side_left then K := M else K := N;
  lda := K;
  ldb := M;
  size_A_mat := K * K;
  stride_A := size_A_mat;
  stride_B := M * N;
  size_A := batch_count * stride_A * SizeOf(rocblas_double_complex);
  size_B := batch_count * stride_B * SizeOf(rocblas_double_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_X_initial, batch_count * stride_B);
  SetLength(h_B_input, batch_count * stride_B);
  SetLength(h_X_expected, batch_count * stride_B);
  SetLength(h_B_result, batch_count * stride_B);

  alpha.x := 1.5; alpha.y := 0.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20)/20.0+0.5; h_A[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_X_initial) - 1 do begin h_X_initial[i].x := (i mod 20)/20.0; h_X_initial[i].y := (i mod 10)/10.0; end;
  for b := 0 to batch_count - 1 do
    for i := 0 to K - 1 do
      h_A[b * stride_A + i + i * lda].x := K;

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pX_initial := @h_X_initial[b * stride_B];
    pB_input   := @h_B_input[b * stride_B];
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
      begin
        temp.x := 0.0; temp.y := 0.0;
        if side = rocblas_side_left then
        begin
          for k_idx := 0 to K - 1 do
          begin
            A_val.x := 0.0; A_val.y := 0.0;
            if uplo = rocblas_fill_upper then
            begin
              if i <= k_idx then A_val := pA[i + k_idx * lda];
            end
            else
            begin
              if i >= k_idx then A_val := pA[i + k_idx * lda];
            end;
            if (i = k_idx) and (diag = rocblas_diagonal_unit) then begin A_val.x := 1.0; A_val.y := 0.0; end;
            prod := A_val; ComplexMul(prod, pX_initial[k_idx + j * ldb]); ComplexAdd(temp, prod);
          end;
        end
        else
        begin
          for k_idx := 0 to K - 1 do
          begin
            A_val.x := 0.0; A_val.y := 0.0;
            if uplo = rocblas_fill_upper then
            begin
              if k_idx <= j then A_val := pA[k_idx + j * lda];
            end
            else
            begin
              if k_idx >= j then A_val := pA[k_idx + j * lda];
            end;
            if (k_idx=j) and (diag = rocblas_diagonal_unit) then begin A_val.x := 1.0; A_val.y := 0.0; end;
            prod := pX_initial[i + k_idx * ldb]; ComplexMul(prod, A_val); ComplexAdd(temp, prod);
          end;
        end;
        pB_input[i + j * ldb] := temp;
      end;
  end;

  for i := 0 to Length(h_X_initial) - 1 do
  begin
    h_X_expected[i] := h_X_initial[i];
    ComplexMul(h_X_expected[i], alpha);
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B_input[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);

  blas_status := rocblas_ztrsm_strided_batched_64(handle, side, uplo, transA, diag, M, N, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, batch_count);
  AssertEquals('rocblas_ztrsm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_B_result[0], d_B, size_B, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_B failed', HIP_SUCCESS, hip_status);

  for i := 0 to Length(h_B_result) - 1 do
  begin
    CheckEquals(h_X_expected[i].x, h_B_result[i].x, tolerance, Format('Mismatch at index %d .x', [i]));
    CheckEquals(h_X_expected[i].y, h_B_result[i].y, tolerance, Format('Mismatch at index %d .y', [i]));
  end;

  hipFree(d_A); hipFree(d_B);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_SSYRK_Strided_Batched_64;
const
  N: Int64 = 72;
  K: Int64 = 72;
  batch_count: Int64 = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
var
  lda, ldc, size_A_mat: Int64;
  stride_A, stride_C: rocblas_stride;
  h_A, h_C, h_C_result, h_C_expected: array of Single;
  d_A, d_C: hipDeviceptr_t;
  size_A, size_C: SizeUInt;
  alpha, beta: Single;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pC_initial, pC_expected: PSingle;
  temp: Single;
  tolerance: Single = 2e-4;
begin
  // SYRK: C := alpha*op(A)*op(A)^T + beta*C
  if transA = rocblas_operation_none then
  begin
    lda := N;
    size_A_mat := N * K;
  end
  else // transpose
  begin
    lda := K;
    size_A_mat := K * N;
  end;
  ldc := N;

  stride_A := size_A_mat;
  stride_C := N * N;
  size_A := batch_count * stride_A * SizeOf(Single);
  size_C := batch_count * stride_C * SizeOf(Single);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha := 1.5;
  beta := 2.5;
  for i := 0 to Length(h_A) - 1 do h_A[i] := (i mod 50) / 50.0 + 1.0;
  for i := 0 to Length(h_C) - 1 do h_C[i] := (i mod 50) / 50.0 + 1.0;
  System.Move(h_C[0], h_C_expected[0], size_C);

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;

        temp := 0.0;
        if transA = rocblas_operation_none then
        begin
          for k_idx := 0 to K - 1 do
            temp := temp + pA[i + k_idx * lda] * pA[j + k_idx * lda];
        end
        else // transpose
        begin
          for k_idx := 0 to K - 1 do
            temp := temp + pA[k_idx + i * lda] * pA[k_idx + j * lda];
        end;
        pC_expected[i + j * ldc] := alpha * temp + beta * pC_initial[i + j * ldc];
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_ssyrk_strided_batched_64(handle, uplo, transA, N, K, @alpha,
    d_A, lda, stride_A, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_ssyrk_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  // Verify
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;
        CheckEquals(h_C_expected[b*stride_C + i+j*ldc], h_C_result[b*stride_C + i+j*ldc],
                    tolerance, Format('Mismatch at batch %d, index (%d, %d)', [b, i, j]));
      end;

  hipFree(d_A); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_DSYRK_Strided_Batched_64;
const
  N: Int64 = 72;
  K: Int64 = 72;
  batch_count: Int64 = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
var
  lda, ldc, size_A_mat: Int64;
  stride_A, stride_C: rocblas_stride;
  h_A, h_C, h_C_result, h_C_expected: array of Double;
  d_A, d_C: hipDeviceptr_t;
  size_A, size_C: SizeUInt;
  alpha, beta: Double;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pC_initial, pC_expected: PDouble;
  temp: Double;
  tolerance: Double = 1e-12;
begin
  if transA = rocblas_operation_none then
  begin
    lda := N;
    size_A_mat := N * K;
  end
  else
  begin
    lda := K;
    size_A_mat := K * N;
  end;
  ldc := N;

  stride_A := size_A_mat;
  stride_C := N * N;
  size_A := batch_count * stride_A * SizeOf(Double);
  size_C := batch_count * stride_C * SizeOf(Double);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha := 1.5;
  beta := 2.5;
  for i := 0 to Length(h_A) - 1 do h_A[i] := (i mod 50) / 50.0 + 1.0;
  for i := 0 to Length(h_C) - 1 do h_C[i] := (i mod 50) / 50.0 + 1.0;
  System.Move(h_C[0], h_C_expected[0], size_C);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;

        temp := 0.0;
        if transA = rocblas_operation_none then
        begin
          for k_idx := 0 to K - 1 do
            temp := temp + pA[i + k_idx * lda] * pA[j + k_idx * lda];
        end
        else
        begin
          for k_idx := 0 to K - 1 do
            temp := temp + pA[k_idx + i * lda] * pA[k_idx + j * lda];
        end;
        pC_expected[i + j * ldc] := alpha * temp + beta * pC_initial[i + j * ldc];
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_dsyrk_strided_batched_64(handle, uplo, transA, N, K, @alpha,
    d_A, lda, stride_A, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_dsyrk_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;
        CheckEquals(h_C_expected[b*stride_C + i+j*ldc], h_C_result[b*stride_C + i+j*ldc],
                    tolerance, Format('Mismatch at batch %d, index (%d, %d)', [b, i, j]));
      end;

  hipFree(d_A); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_CSYRK_Strided_Batched_64;
const
  N: Int64 = 48;
  K: Int64 = 48;
  batch_count: Int64 = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
var
  lda, ldc, size_A_mat: Int64;
  stride_A, stride_C: rocblas_stride;
  h_A, h_C, h_C_result, h_C_expected: array of rocblas_float_complex;
  d_A, d_C: hipDeviceptr_t;
  size_A, size_C: SizeUInt;
  alpha, beta: rocblas_float_complex;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pC_initial, pC_expected: Procfloat_complex;
  temp, prod: rocblas_float_complex;
  tolerance: Single = 1e-4;
begin
  if transA = rocblas_operation_none then
  begin
    lda := N;
    size_A_mat := N * K;
  end
  else
  begin
    lda := K;
    size_A_mat := K * N;
  end;
  ldc := N;

  stride_A := size_A_mat;
  stride_C := N * N;
  size_A := batch_count * stride_A * SizeOf(rocblas_float_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_float_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha.x := 1.5; alpha.y := 0.5;
  beta.x  := 2.5; beta.y  := 1.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20)/20.0; h_A[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_C) - 1 do begin h_C[i].x := (i mod 20)/20.0; h_C[i].y := (i mod 10)/10.0; end;
  System.Move(h_C[0], h_C_expected[0], size_C);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;

        temp.x := 0.0; temp.y := 0.0;
        if transA = rocblas_operation_none then
        begin
          for k_idx := 0 to K - 1 do
          begin
            prod := pA[i + k_idx * lda];
            ComplexMul(prod, pA[j + k_idx * lda]); // A * A^T
            ComplexAdd(temp, prod);
          end;
        end
        else
        begin
          for k_idx := 0 to K - 1 do
          begin
            prod := pA[k_idx + i * lda];
            ComplexMul(prod, pA[k_idx + j * lda]); // A^T * A
            ComplexAdd(temp, prod);
          end;
        end;
        ComplexMul(temp, alpha);
        prod := pC_initial[i + j * ldc];
        ComplexMul(prod, beta);
        ComplexAdd(temp, prod);
        pC_expected[i + j * ldc] := temp;
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_csyrk_strided_batched_64(handle, uplo, transA, N, K, @alpha,
    d_A, lda, stride_A, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_csyrk_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;
        CheckEquals(h_C_expected[b*stride_C + i+j*ldc].x, h_C_result[b*stride_C + i+j*ldc].x,
                    tolerance, Format('Mismatch at batch %d, index (%d, %d) .x', [b, i, j]));
        CheckEquals(h_C_expected[b*stride_C + i+j*ldc].y, h_C_result[b*stride_C + i+j*ldc].y,
                    tolerance, Format('Mismatch at batch %d, index (%d, %d) .y', [b, i, j]));
      end;

  hipFree(d_A); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_ZSYRK_Strided_Batched_64;
const
  N: Int64 = 48;
  K: Int64 = 48;
  batch_count: Int64 = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
var
  lda, ldc, size_A_mat: Int64;
  stride_A, stride_C: rocblas_stride;
  h_A, h_C, h_C_result, h_C_expected: array of rocblas_double_complex;
  d_A, d_C: hipDeviceptr_t;
  size_A, size_C: SizeUInt;
  alpha, beta: rocblas_double_complex;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pC_initial, pC_expected: Procdouble_complex;
  temp, prod: rocblas_double_complex;
  tolerance: Double = 1e-11;
begin
  if transA = rocblas_operation_none then
  begin
    lda := N;
    size_A_mat := N * K;
  end
  else
  begin
    lda := K;
    size_A_mat := K * N;
  end;
  ldc := N;

  stride_A := size_A_mat;
  stride_C := N * N;
  size_A := batch_count * stride_A * SizeOf(rocblas_double_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_double_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha.x := 1.5; alpha.y := 0.5;
  beta.x  := 2.5; beta.y  := 1.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20)/20.0; h_A[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_C) - 1 do begin h_C[i].x := (i mod 20)/20.0; h_C[i].y := (i mod 10)/10.0; end;
  System.Move(h_C[0], h_C_expected[0], size_C);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;

        temp.x := 0.0; temp.y := 0.0;
        if transA = rocblas_operation_none then
        begin
          for k_idx := 0 to K - 1 do
          begin
            prod := pA[i + k_idx * lda];
            ComplexMul(prod, pA[j + k_idx * lda]);
            ComplexAdd(temp, prod);
          end;
        end
        else
        begin
          for k_idx := 0 to K - 1 do
          begin
            prod := pA[k_idx + i * lda];
            ComplexMul(prod, pA[k_idx + j * lda]);
            ComplexAdd(temp, prod);
          end;
        end;
        ComplexMul(temp, alpha);
        prod := pC_initial[i + j * ldc];
        ComplexMul(prod, beta);
        ComplexAdd(temp, prod);
        pC_expected[i + j * ldc] := temp;
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_zsyrk_strided_batched_64(handle, uplo, transA, N, K, @alpha,
    d_A, lda, stride_A, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_zsyrk_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;
        CheckEquals(h_C_expected[b*stride_C + i+j*ldc].x, h_C_result[b*stride_C + i+j*ldc].x,
                    tolerance, Format('Mismatch at batch %d, index (%d, %d) .x', [b, i, j]));
        CheckEquals(h_C_expected[b*stride_C + i+j*ldc].y, h_C_result[b*stride_C + i+j*ldc].y,
                    tolerance, Format('Mismatch at batch %d, index (%d, %d) .y', [b, i, j]));
      end;

  hipFree(d_A); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_CHERK_Strided_Batched_64;
const
  N: Int64 = 48;
  K: Int64 = 48;
  batch_count: Int64 = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
var
  lda, ldc, size_A_mat: Int64;
  stride_A, stride_C: rocblas_stride;
  h_A, h_C, h_C_result, h_C_expected: array of rocblas_float_complex;
  d_A, d_C: hipDeviceptr_t;
  size_A, size_C: SizeUInt;
  alpha, beta: Single;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pC_initial, pC_expected: Procfloat_complex;
  temp, prod, op_A_ik, op_A_jk_conj: rocblas_float_complex;
  tolerance: Single = 1e-4;
  abs_tolerance: Single = 1e-6; // For checking against zero
begin
  // HERK: C := alpha*op(A)*op(A)^H + beta*C
  if transA = rocblas_operation_none then
  begin
    lda := N;
    size_A_mat := N * K;
  end
  else // conjugate transpose
  begin
    lda := K;
    size_A_mat := K * N;
  end;
  ldc := N;

  stride_A := size_A_mat;
  stride_C := N * N;
  size_A := batch_count * stride_A * SizeOf(rocblas_float_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_float_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha := 1.5;
  beta  := 2.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20)/20.0; h_A[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_C) - 1 do begin h_C[i].x := (i mod 20)/20.0; h_C[i].y := 0.0; end; // Imaginary on C diagonal is zero
  System.Move(h_C[0], h_C_expected[0], size_C);

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;

        temp.x := 0.0; temp.y := 0.0;
        for k_idx := 0 to K - 1 do
        begin
          if transA = rocblas_operation_none then
          begin
            op_A_ik := pA[i + k_idx * lda];
            op_A_jk_conj := pA[j + k_idx * lda]; op_A_jk_conj.y := -op_A_jk_conj.y;
          end
          else // conjugate transpose
          begin
            op_A_ik := pA[k_idx + i * lda]; op_A_ik.y := -op_A_ik.y;
            op_A_jk_conj := pA[k_idx + j * lda];
          end;
          prod := op_A_ik;
          ComplexMul(prod, op_A_jk_conj);
          ComplexAdd(temp, prod);
        end;

        // C[i,j] = alpha * temp + beta * C_initial[i,j]
        // Since alpha and beta are real, this is a simple scaling
        pC_expected[i + j * ldc].x := alpha * temp.x + beta * pC_initial[i + j * ldc].x;
        pC_expected[i + j * ldc].y := alpha * temp.y + beta * pC_initial[i + j * ldc].y;

        // Diagonal of a Hermitian matrix must be real
        if i = j then pC_expected[i + j * ldc].y := 0.0;
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_cherk_strided_batched_64(handle, uplo, transA, N, K, @alpha,
    d_A, lda, stride_A, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_cherk_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;
        CheckEqualsRelative(h_C_expected[b*stride_C + i+j*ldc].x, h_C_result[b*stride_C + i+j*ldc].x,
                            tolerance, Format('Mismatch at batch %d, index (%d, %d) .x', [b, i, j]));
        if i = j then // Diagonal - imaginary part must be zero
          CheckEquals(0.0, h_C_result[b*stride_C + i+j*ldc].y, abs_tolerance, Format('Imaginary part of diagonal non-zero at batch %d, index (%d, %d)', [b, i, j]))
        else
          CheckEqualsRelative(h_C_expected[b*stride_C + i+j*ldc].y, h_C_result[b*stride_C + i+j*ldc].y,
                              tolerance, Format('Mismatch at batch %d, index (%d, %d) .y', [b, i, j]));
      end;

  hipFree(d_A); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_ZHERK_Strided_Batched_64;
const
  N: Int64 = 48;
  K: Int64 = 48;
  batch_count: Int64 = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
var
  lda, ldc, size_A_mat: Int64;
  stride_A, stride_C: rocblas_stride;
  h_A, h_C, h_C_result, h_C_expected: array of rocblas_double_complex;
  d_A, d_C: hipDeviceptr_t;
  size_A, size_C: SizeUInt;
  alpha, beta: Double;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pC_initial, pC_expected: Procdouble_complex;
  temp, prod, op_A_ik, op_A_jk_conj: rocblas_double_complex;
  tolerance: Double = 1e-11;
  abs_tolerance: Double = 1e-12;
begin
  if transA = rocblas_operation_none then
  begin
    lda := N;
    size_A_mat := N * K;
  end
  else
  begin
    lda := K;
    size_A_mat := K * N;
  end;
  ldc := N;

  stride_A := size_A_mat;
  stride_C := N * N;
  size_A := batch_count * stride_A * SizeOf(rocblas_double_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_double_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha := 1.5;
  beta  := 2.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20)/20.0; h_A[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_C) - 1 do begin h_C[i].x := (i mod 20)/20.0; h_C[i].y := 0.0; end;
  System.Move(h_C[0], h_C_expected[0], size_C);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;

        temp.x := 0.0; temp.y := 0.0;
        for k_idx := 0 to K - 1 do
        begin
          if transA = rocblas_operation_none then
          begin
            op_A_ik := pA[i + k_idx * lda];
            op_A_jk_conj := pA[j + k_idx * lda]; op_A_jk_conj.y := -op_A_jk_conj.y;
          end
          else
          begin
            op_A_ik := pA[k_idx + i * lda]; op_A_ik.y := -op_A_ik.y;
            op_A_jk_conj := pA[k_idx + j * lda];
          end;
          prod := op_A_ik;
          ComplexMul(prod, op_A_jk_conj);
          ComplexAdd(temp, prod);
        end;

        pC_expected[i + j * ldc].x := alpha * temp.x + beta * pC_initial[i + j * ldc].x;
        pC_expected[i + j * ldc].y := alpha * temp.y + beta * pC_initial[i + j * ldc].y;

        if i = j then pC_expected[i + j * ldc].y := 0.0;
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_zherk_strided_batched_64(handle, uplo, transA, N, K, @alpha,
    d_A, lda, stride_A, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_zherk_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;
        CheckEqualsRelative(h_C_expected[b*stride_C + i+j*ldc].x, h_C_result[b*stride_C + i+j*ldc].x,
                            tolerance, Format('Mismatch at batch %d, index (%d, %d) .x', [b, i, j]));
        if i = j then
          CheckEquals(0.0, h_C_result[b*stride_C + i+j*ldc].y, abs_tolerance, Format('Imaginary part of diagonal non-zero at batch %d, index (%d, %d)', [b, i, j]))
        else
          CheckEqualsRelative(h_C_expected[b*stride_C + i+j*ldc].y, h_C_result[b*stride_C + i+j*ldc].y,
                              tolerance, Format('Mismatch at batch %d, index (%d, %d) .y', [b, i, j]));
      end;

  hipFree(d_A); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_SSyr2k_Strided_Batched_64;
const
  N: Int64 = 64;
  K: Int64 = 64;
  batch_count: Int64 = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
var
  lda, ldb, ldc, size_A_mat, size_B_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of Single;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, beta: Single;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_initial, pC_expected: PSingle;
  term1, term2: Single;
  tolerance: Single = 1e-3;
begin
  // SYR2K: C := alpha*(op(A)*op(B)^T + op(B)*op(A)^T) + beta*C
  if transA = rocblas_operation_none then
  begin
    lda := N; ldb := N;
    size_A_mat := N * K; size_B_mat := N * K;
  end
  else // transpose
  begin
    lda := K; ldb := K;
    size_A_mat := K * N; size_B_mat := K * N;
  end;
  ldc := N;

  stride_A := size_A_mat;
  stride_B := size_B_mat;
  stride_C := N * N;
  size_A := batch_count * stride_A * SizeOf(Single);
  size_B := batch_count * stride_B * SizeOf(Single);
  size_C := batch_count * stride_C * SizeOf(Single);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha := 1.5;
  beta := 2.5;
  for i := 0 to Length(h_A) - 1 do h_A[i] := (i mod 50) / 50.0 + 1.0;
  for i := 0 to Length(h_B) - 1 do h_B[i] := (i mod 50) / 50.0 + 1.0;
  for i := 0 to Length(h_C) - 1 do h_C[i] := (i mod 50) / 50.0 + 1.0;
  System.Move(h_C[0], h_C_expected[0], size_C);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;

        term1 := 0.0; term2 := 0.0;
        if transA = rocblas_operation_none then
        begin
          for k_idx := 0 to K - 1 do
          begin
            term1 := term1 + pA[i + k_idx * lda] * pB[j + k_idx * ldb];
            term2 := term2 + pB[i + k_idx * ldb] * pA[j + k_idx * lda];
          end;
        end
        else
        begin
          for k_idx := 0 to K - 1 do
          begin
            term1 := term1 + pA[k_idx + i * lda] * pB[k_idx + j * ldb];
            term2 := term2 + pB[k_idx + i * ldb] * pA[k_idx + j * lda];
          end;
        end;
        pC_expected[i + j * ldc] := alpha * (term1 + term2) + beta * pC_initial[i + j * ldc];
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_ssyr2k_strided_batched_64(handle, uplo, transA, N, K, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_ssyr2k_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;
        CheckEquals(h_C_expected[b*stride_C + i+j*ldc], h_C_result[b*stride_C + i+j*ldc],
                    tolerance, Format('Mismatch at batch %d, index (%d, %d)', [b, i, j]));
      end;

  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_DSyr2k_Strided_Batched_64;
const
  N: Int64 = 64;
  K: Int64 = 64;
  batch_count: Int64 = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
var
  lda, ldb, ldc, size_A_mat, size_B_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of Double;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, beta: Double;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_initial, pC_expected: PDouble;
  term1, term2: Double;
  tolerance: Double = 1e-11;
begin
  if transA = rocblas_operation_none then
  begin
    lda := N; ldb := N;
    size_A_mat := N * K; size_B_mat := N * K;
  end
  else
  begin
    lda := K; ldb := K;
    size_A_mat := K * N; size_B_mat := K * N;
  end;
  ldc := N;

  stride_A := size_A_mat;
  stride_B := size_B_mat;
  stride_C := N * N;
  size_A := batch_count * stride_A * SizeOf(Double);
  size_B := batch_count * stride_B * SizeOf(Double);
  size_C := batch_count * stride_C * SizeOf(Double);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha := 1.5;
  beta := 2.5;
  for i := 0 to Length(h_A) - 1 do h_A[i] := (i mod 50) / 50.0 + 1.0;
  for i := 0 to Length(h_B) - 1 do h_B[i] := (i mod 50) / 50.0 + 1.0;
  for i := 0 to Length(h_C) - 1 do h_C[i] := (i mod 50) / 50.0 + 1.0;
  System.Move(h_C[0], h_C_expected[0], size_C);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;

        term1 := 0.0; term2 := 0.0;
        if transA = rocblas_operation_none then
        begin
          for k_idx := 0 to K - 1 do
          begin
            term1 := term1 + pA[i + k_idx * lda] * pB[j + k_idx * ldb];
            term2 := term2 + pB[i + k_idx * ldb] * pA[j + k_idx * lda];
          end;
        end
        else
        begin
          for k_idx := 0 to K - 1 do
          begin
            term1 := term1 + pA[k_idx + i * lda] * pB[k_idx + j * ldb];
            term2 := term2 + pB[k_idx + i * ldb] * pA[k_idx + j * lda];
          end;
        end;
        pC_expected[i + j * ldc] := alpha * (term1 + term2) + beta * pC_initial[i + j * ldc];
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_dsyr2k_strided_batched_64(handle, uplo, transA, N, K, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_dsyr2k_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;
        CheckEquals(h_C_expected[b*stride_C + i+j*ldc], h_C_result[b*stride_C + i+j*ldc],
                    tolerance, Format('Mismatch at batch %d, index (%d, %d)', [b, i, j]));
      end;

  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_CSyr2k_Strided_Batched_64;
const
  N: Int64 = 32;
  K: Int64 = 32;
  batch_count: Int64 = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
var
  lda, ldb, ldc, size_A_mat, size_B_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, beta: rocblas_float_complex;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_initial, pC_expected: Procfloat_complex;
  term1, term2, temp: rocblas_float_complex;
  tolerance: Single = 1e-4;
begin
  if transA = rocblas_operation_none then
  begin
    lda := N; ldb := N;
    size_A_mat := N * K; size_B_mat := N * K;
  end
  else
  begin
    lda := K; ldb := K;
    size_A_mat := K * N; size_B_mat := K * N;
  end;
  ldc := N;

  stride_A := size_A_mat;
  stride_B := size_B_mat;
  stride_C := N * N;
  size_A := batch_count * stride_A * SizeOf(rocblas_float_complex);
  size_B := batch_count * stride_B * SizeOf(rocblas_float_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_float_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha.x := 1.5; alpha.y := 0.5;
  beta.x  := 2.5; beta.y  := 1.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20)/20.0; h_A[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_B) - 1 do begin h_B[i].x := (i mod 20)/20.0; h_B[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_C) - 1 do begin h_C[i].x := (i mod 20)/20.0; h_C[i].y := (i mod 10)/10.0; end;
  System.Move(h_C[0], h_C_expected[0], size_C);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;

        term1.x := 0.0; term1.y := 0.0;
        term2.x := 0.0; term2.y := 0.0;
        if transA = rocblas_operation_none then
        begin
          for k_idx := 0 to K - 1 do
          begin
            temp := pA[i + k_idx*lda]; ComplexMul(temp, pB[j + k_idx*ldb]); ComplexAdd(term1, temp);
            temp := pB[i + k_idx*ldb]; ComplexMul(temp, pA[j + k_idx*lda]); ComplexAdd(term2, temp);
          end;
        end
        else
        begin
          for k_idx := 0 to K - 1 do
          begin
            temp := pA[k_idx + i*lda]; ComplexMul(temp, pB[k_idx + j*ldb]); ComplexAdd(term1, temp);
            temp := pB[k_idx + i*ldb]; ComplexMul(temp, pA[k_idx + j*lda]); ComplexAdd(term2, temp);
          end;
        end;
        ComplexAdd(term1, term2);
        ComplexMul(term1, alpha);
        temp := pC_initial[i + j*ldc];
        ComplexMul(temp, beta);
        ComplexAdd(term1, temp);
        pC_expected[i + j*ldc] := term1;
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_csyr2k_strided_batched_64(handle, uplo, transA, N, K, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_csyr2k_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;
        CheckEquals(h_C_expected[b*stride_C + i+j*ldc].x, h_C_result[b*stride_C + i+j*ldc].x,
                    tolerance, Format('Mismatch at batch %d, index (%d, %d) .x', [b, i, j]));
        CheckEquals(h_C_expected[b*stride_C + i+j*ldc].y, h_C_result[b*stride_C + i+j*ldc].y,
                    tolerance, Format('Mismatch at batch %d, index (%d, %d) .y', [b, i, j]));
      end;

  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_ZSyr2k_Strided_Batched_64;
const
  N: Int64 = 32;
  K: Int64 = 32;
  batch_count: Int64 = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
var
  lda, ldb, ldc, size_A_mat, size_B_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, beta: rocblas_double_complex;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_initial, pC_expected: Procdouble_complex;
  term1, term2, temp: rocblas_double_complex;
  tolerance: Double = 1e-11;
begin
  if transA = rocblas_operation_none then
  begin
    lda := N; ldb := N;
    size_A_mat := N * K; size_B_mat := N * K;
  end
  else
  begin
    lda := K; ldb := K;
    size_A_mat := K * N; size_B_mat := K * N;
  end;
  ldc := N;

  stride_A := size_A_mat;
  stride_B := size_B_mat;
  stride_C := N * N;
  size_A := batch_count * stride_A * SizeOf(rocblas_double_complex);
  size_B := batch_count * stride_B * SizeOf(rocblas_double_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_double_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha.x := 1.5; alpha.y := 0.5;
  beta.x  := 2.5; beta.y  := 1.5;
  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20)/20.0; h_A[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_B) - 1 do begin h_B[i].x := (i mod 20)/20.0; h_B[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_C) - 1 do begin h_C[i].x := (i mod 20)/20.0; h_C[i].y := (i mod 10)/10.0; end;
  System.Move(h_C[0], h_C_expected[0], size_C);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;

        term1.x := 0.0; term1.y := 0.0;
        term2.x := 0.0; term2.y := 0.0;
        if transA = rocblas_operation_none then
        begin
          for k_idx := 0 to K - 1 do
          begin
            temp := pA[i + k_idx*lda]; ComplexMul(temp, pB[j + k_idx*ldb]); ComplexAdd(term1, temp);
            temp := pB[i + k_idx*ldb]; ComplexMul(temp, pA[j + k_idx*lda]); ComplexAdd(term2, temp);
          end;
        end
        else
        begin
          for k_idx := 0 to K - 1 do
          begin
            temp := pA[k_idx + i*lda]; ComplexMul(temp, pB[k_idx + j*ldb]); ComplexAdd(term1, temp);
            temp := pB[k_idx + i*ldb]; ComplexMul(temp, pA[k_idx + j*lda]); ComplexAdd(term2, temp);
          end;
        end;
        ComplexAdd(term1, term2);
        ComplexMul(term1, alpha);
        temp := pC_initial[i + j*ldc];
        ComplexMul(temp, beta);
        ComplexAdd(term1, temp);
        pC_expected[i + j*ldc] := term1;
      end;
    end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_zsyr2k_strided_batched_64(handle, uplo, transA, N, K, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_zsyr2k_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;
        CheckEquals(h_C_expected[b*stride_C + i+j*ldc].x, h_C_result[b*stride_C + i+j*ldc].x,
                    tolerance, Format('Mismatch at batch %d, index (%d, %d) .x', [b, i, j]));
        CheckEquals(h_C_expected[b*stride_C + i+j*ldc].y, h_C_result[b*stride_C + i+j*ldc].y,
                    tolerance, Format('Mismatch at batch %d, index (%d, %d) .y', [b, i, j]));
      end;

  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_CHER2K_Strided_Batched_64;
const
  N: Int64 = 48;
  K: Int64 = 48;
  batch_count: Int64 = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
var
  lda, ldb, ldc, size_A_mat, size_B_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, alpha_conj: rocblas_float_complex;
  beta: Single; // Beta is REAL for HER2K
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_initial, pC_expected: Procfloat_complex;
  term1, term2, temp, op_A_ik, op_B_jk_conj, op_B_ik, op_A_jk_conj: rocblas_float_complex;
  tolerance: Single = 1e-4;
  abs_tolerance: Single = 1e-6;
begin
  if transA = rocblas_operation_none then
  begin
    lda := N; ldb := N;
    size_A_mat := N * K; size_B_mat := N * K;
  end
  else
  begin
    lda := K; ldb := K;
    size_A_mat := K * N; size_B_mat := K * N;
  end;
  ldc := N;

  stride_A := size_A_mat; stride_B := size_B_mat; stride_C := N * N;
  size_A := batch_count * stride_A * SizeOf(rocblas_float_complex);
  size_B := batch_count * stride_B * SizeOf(rocblas_float_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_float_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha.x := 1.5; alpha.y := 0.5;
  beta := 2.5;
  alpha_conj := alpha; alpha_conj.y := -alpha_conj.y;

  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20)/20.0; h_A[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_B) - 1 do begin h_B[i].x := (i mod 20)/20.0; h_B[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_C) - 1 do begin h_C[i].x := (i mod 20)/20.0; h_C[i].y := 0.0; end;
  System.Move(h_C[0], h_C_expected[0], size_C);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;

        term1.x := 0.0; term1.y := 0.0;
        term2.x := 0.0; term2.y := 0.0;
        for k_idx := 0 to K - 1 do
        begin
          if transA = rocblas_operation_none then
          begin
            op_A_ik := pA[i + k_idx*lda];
            op_B_jk_conj := pB[j + k_idx*ldb]; op_B_jk_conj.y := -op_B_jk_conj.y;
            op_B_ik := pB[i + k_idx*ldb];
            op_A_jk_conj := pA[j + k_idx*lda]; op_A_jk_conj.y := -op_A_jk_conj.y;
          end
          else
          begin
            op_A_ik := pA[k_idx + i*lda]; op_A_ik.y := -op_A_ik.y;
            op_B_jk_conj := pB[k_idx + j*ldb];
            op_B_ik := pB[k_idx + i*ldb]; op_B_ik.y := -op_B_ik.y;
            op_A_jk_conj := pA[k_idx + j*lda];
          end;
          temp := op_A_ik; ComplexMul(temp, op_B_jk_conj); ComplexAdd(term1, temp);
          temp := op_B_ik; ComplexMul(temp, op_A_jk_conj); ComplexAdd(term2, temp);
        end;

        ComplexMul(term1, alpha);
        ComplexMul(term2, alpha_conj);
        ComplexAdd(term1, term2);

        pC_expected[i+j*ldc].x := beta * pC_initial[i+j*ldc].x + term1.x;
        pC_expected[i+j*ldc].y := beta * pC_initial[i+j*ldc].y + term1.y;

        if i = j then pC_expected[i+j*ldc].y := 0.0;
      end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_cher2k_strided_batched_64(handle, uplo, transA, N, K, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_cher2k_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;
        CheckEqualsRelative(h_C_expected[b*stride_C+i+j*ldc].x, h_C_result[b*stride_C+i+j*ldc].x,
                            tolerance, Format('Mismatch at batch %d, index (%d, %d) .x', [b, i, j]));
        if i = j then
          CheckEquals(0.0, h_C_result[b*stride_C+i+j*ldc].y, abs_tolerance, Format('Imaginary part of diagonal non-zero at batch %d, index (%d, %d)', [b, i, j]))
        else
          CheckEqualsRelative(h_C_expected[b*stride_C+i+j*ldc].y, h_C_result[b*stride_C+i+j*ldc].y,
                              tolerance, Format('Mismatch at batch %d, index (%d, %d) .y', [b, i, j]));
      end;

  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

procedure TTestRocBLAS_Level3_strided_batched_64.Test_ZHER2K_Strided_Batched_64;
const
  N: Int64 = 48;
  K: Int64 = 48;
  batch_count: Int64 = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
var
  lda, ldb, ldc, size_A_mat, size_B_mat: Int64;
  stride_A, stride_B, stride_C: rocblas_stride;
  h_A, h_B, h_C, h_C_result, h_C_expected: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  size_A, size_B, size_C: SizeUInt;
  alpha, alpha_conj: rocblas_double_complex;
  beta: Double; // Beta is REAL for HER2K
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, k_idx: Int64;
  pA, pB, pC_initial, pC_expected: Procdouble_complex;
  term1, term2, temp, op_A_ik, op_B_jk_conj, op_B_ik, op_A_jk_conj: rocblas_double_complex;
  relative_tolerance: Double = 1e-11;
  absolute_tolerance: Double = 1e-12;
  relative_error, expected_val, actual_val: Double;
  error_msg: String;
begin
  if transA = rocblas_operation_none then
  begin
    lda := N; ldb := N;
    size_A_mat := N * K; size_B_mat := N * K;
  end
  else
  begin
    lda := K; ldb := K;
    size_A_mat := K * N; size_B_mat := K * N;
  end;
  ldc := N;

  stride_A := size_A_mat; stride_B := size_B_mat; stride_C := N * N;
  size_A := batch_count * stride_A * SizeOf(rocblas_double_complex);
  size_B := batch_count * stride_B * SizeOf(rocblas_double_complex);
  size_C := batch_count * stride_C * SizeOf(rocblas_double_complex);

  SetLength(h_A, batch_count * stride_A);
  SetLength(h_B, batch_count * stride_B);
  SetLength(h_C, batch_count * stride_C);
  SetLength(h_C_result, batch_count * stride_C);
  SetLength(h_C_expected, batch_count * stride_C);

  alpha.x := 1.5; alpha.y := 0.5;
  beta := 2.5;
  alpha_conj := alpha; alpha_conj.y := -alpha_conj.y;

  for i := 0 to Length(h_A) - 1 do begin h_A[i].x := (i mod 20)/20.0; h_A[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_B) - 1 do begin h_B[i].x := (i mod 20)/20.0; h_B[i].y := (i mod 10)/10.0; end;
  for i := 0 to Length(h_C) - 1 do begin h_C[i].x := (i mod 20)/20.0; h_C[i].y := 0.0; end;
  System.Move(h_C[0], h_C_expected[0], size_C);

  for b := 0 to batch_count - 1 do
  begin
    pA := @h_A[b * stride_A];
    pB := @h_B[b * stride_B];
    pC_initial := @h_C[b * stride_C];
    pC_expected := @h_C_expected[b * stride_C];

    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;

        term1.x := 0.0; term1.y := 0.0;
        term2.x := 0.0; term2.y := 0.0;
        for k_idx := 0 to K - 1 do
        begin
          if transA = rocblas_operation_none then
          begin
            op_A_ik := pA[i + k_idx*lda];
            op_B_jk_conj := pB[j + k_idx*ldb]; op_B_jk_conj.y := -op_B_jk_conj.y;
            op_B_ik := pB[i + k_idx*ldb];
            op_A_jk_conj := pA[j + k_idx*lda]; op_A_jk_conj.y := -op_A_jk_conj.y;
          end
          else
          begin
            op_A_ik := pA[k_idx + i*lda]; op_A_ik.y := -op_A_ik.y;
            op_B_jk_conj := pB[k_idx + j*ldb];
            op_B_ik := pB[k_idx + i*ldb]; op_B_ik.y := -op_B_ik.y;
            op_A_jk_conj := pA[k_idx + j*lda];
          end;
          temp := op_A_ik; ComplexMul(temp, op_B_jk_conj); ComplexAdd(term1, temp);
          temp := op_B_ik; ComplexMul(temp, op_A_jk_conj); ComplexAdd(term2, temp);
        end;

        ComplexMul(term1, alpha);
        ComplexMul(term2, alpha_conj);
        ComplexAdd(term1, term2);

        pC_expected[i+j*ldc].x := beta * pC_initial[i+j*ldc].x + term1.x;
        pC_expected[i+j*ldc].y := beta * pC_initial[i+j*ldc].y + term1.y;

        if i = j then pC_expected[i+j*ldc].y := 0.0;
      end;
  end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_zher2k_strided_batched_64(handle, uplo, transA, N, K, @alpha,
    d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
  AssertEquals('rocblas_zher2k_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_C_result[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy from d_C failed', HIP_SUCCESS, hip_status);

  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
      for j := 0 to N - 1 do
      begin
        if (uplo = rocblas_fill_upper) and (j < i) then continue;
        if (uplo = rocblas_fill_lower) and (j > i) then continue;

        // Manual relative check for .x component
        expected_val := h_C_expected[b*stride_C+i+j*ldc].x;
        actual_val := h_C_result[b*stride_C+i+j*ldc].x;
        if expected_val <> 0.0 then
          relative_error := Abs((expected_val - actual_val) / expected_val)
        else
          relative_error := Abs(expected_val - actual_val);
        error_msg := Format('Mismatch at batch %d, index (%d,%d) .x. RelErr: %.12f, Exp: %.12f, Act: %.12f', [b, i, j, relative_error, expected_val, actual_val]);
        AssertTrue(error_msg, relative_error <= relative_tolerance);

        // Hybrid check for .y component
        expected_val := h_C_expected[b*stride_C+i+j*ldc].y;
        actual_val := h_C_result[b*stride_C+i+j*ldc].y;
        if i = j then // Diagonal - imaginary part must be zero, use absolute tolerance
          CheckEquals(0.0, actual_val, absolute_tolerance, Format('Imaginary part of diagonal non-zero at batch %d, index (%d, %d)', [b, i, j]))
        else // Off-diagonal, use relative tolerance
        begin
          if expected_val <> 0.0 then
            relative_error := Abs((expected_val - actual_val) / expected_val)
          else
            relative_error := Abs(expected_val - actual_val);
          error_msg := Format('Mismatch at batch %d, index (%d,%d) .y. RelErr: %.12f, Exp: %.12f, Act: %.12f', [b, i, j, relative_error, expected_val, actual_val]);
          AssertTrue(error_msg, relative_error <= relative_tolerance);
        end;
      end;

  hipFree(d_A); hipFree(d_B); hipFree(d_C);
end;

initialization
  RegisterTest(TTestRocBLAS_Level3_strided_batched_64);

end.

