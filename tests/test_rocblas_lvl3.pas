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

  Signature checks for the rocBLAS level-3 (matrix-matrix) bindings: every
  call is made and its status is asserted, so a changed prototype or a
  missing symbol fails here first.
------------------------------------------------------------------------------}

unit Test_rocblas_lvl3;

{$mode objfpc}{$H+}
{$linklib rocblas}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  TTestRocBLAS_Level3 = class(TTestRocBLAS_Base)
  published
    procedure Test_SGEMM;
    procedure Test_DGEMM;
    procedure Test_CGEMM;
    procedure Test_ZGEMM;
    procedure Test_CHEMM;
    procedure Test_ZHEMM;
    procedure Test_STRMM;
    procedure Test_DTRMM;
    procedure Test_CTRMM;
    procedure Test_ZTRMM;
    procedure Test_STRSM;
    procedure Test_DTRSM;
    procedure Test_CTRSM;
    procedure Test_ZTRSM;
    procedure Test_SSYMM;
    procedure Test_DSYMM;
    procedure Test_CSYMM;
    procedure Test_ZSYMM;
    procedure Test_CHERK;
    procedure Test_ZHERK;
    procedure Test_SSYRK;
    procedure Test_DSYRK;
    procedure Test_CSYRK;
    procedure Test_ZSYRK;
    procedure Test_CHER2K;
    procedure Test_ZHER2K;
    procedure Test_SSyr2k;
    procedure Test_DSyr2k;
    procedure Test_CSyr2k;
    procedure Test_ZSyr2k;
    procedure Test_GEMM_EX_FP16;
  end;

implementation

procedure TTestRocBLAS_Level3.Test_SGEMM;
const
  M = 4; N = 3; K = 5;
  alpha: Single = 2.0; beta: Single = 1.5;
var
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC, expectedC: array of Single;
  devA_array, devB_array, devC_array: array of PSingle;
  i, j, l: Integer;
begin
  SetLength(hostA, M*K); SetLength(hostB, K*N); SetLength(hostC, M*N); SetLength(expectedC, M*N);

  for i := 0 to M*K-1 do hostA[i] := i + 1;
  for i := 0 to K*N-1 do hostB[i] := i + 1;
  for i := 0 to M*N-1 do hostC[i] := i + 1;

  for i := 0 to M - 1 do for j := 0 to N - 1 do
  begin
    expectedC[j*M+i] := beta * hostC[j*M+i];
    for l := 0 to K - 1 do
      expectedC[j*M+i] := expectedC[j*M+i] + alpha * hostA[l*M+i] * hostB[j*K+l];
  end;

  hipMalloc(devA, M*K*SizeOf(Single)); hipMalloc(devB, K*N*SizeOf(Single)); hipMalloc(devC, M*N*SizeOf(Single));
  hipMemcpy(devA, @hostA[0], M*K*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devB, @hostB[0], K*N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devC, @hostC[0], M*N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_sgemm should return success', rocblas_status_success, rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, @alpha, devA, M, devB, K, @beta, devC, M));

  hipMemcpy(@hostC[0], devC, M*N*SizeOf(Single), hipMemcpyDeviceToHost);
  for i := 0 to M*N-1 do AssertEquals('SGEMM C[' + IntToStr(i) + ']', expectedC[i], hostC[i], 1e-4);

  hipFree(devA); hipFree(devB); hipFree(devC);
end;

procedure TTestRocBLAS_Level3.Test_DGEMM;
const
  M = 4; N = 3; K = 5;
  alpha: Double = 2.0; beta: Double = 1.5;
var
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC, expectedC: array of Double;
  devA_array, devB_array, devC_array: array of PDouble;
  i, j, l: Integer;
begin
  SetLength(hostA, M*K); SetLength(hostB, K*N); SetLength(hostC, M*N); SetLength(expectedC, M*N);

  for i := 0 to M*K-1 do hostA[i] := i + 1;
  for i := 0 to K*N-1 do hostB[i] := i + 1;
  for i := 0 to M*N-1 do hostC[i] := i + 1;

  for i := 0 to M - 1 do for j := 0 to N - 1 do
  begin
    expectedC[j*M+i] := beta * hostC[j*M+i];
    for l := 0 to K - 1 do
      expectedC[j*M+i] := expectedC[j*M+i] + alpha * hostA[l*M+i] * hostB[j*K+l];
  end;

  hipMalloc(devA, M*K*SizeOf(Double)); hipMalloc(devB, K*N*SizeOf(Double)); hipMalloc(devC, M*N*SizeOf(Double));
  hipMemcpy(devA, @hostA[0], M*K*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devB, @hostB[0], K*N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devC, @hostC[0], M*N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dgemm should return success', rocblas_status_success, rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, @alpha, devA, M, devB, K, @beta, devC, M));

  hipMemcpy(@hostC[0], devC, M*N*SizeOf(Double), hipMemcpyDeviceToHost);
  for i := 0 to M*N-1 do AssertEquals('DGEMM C[' + IntToStr(i) + ']', expectedC[i], hostC[i], 1e-12);

  hipFree(devA); hipFree(devB); hipFree(devC);
end;

procedure TTestRocBLAS_Level3.Test_CGEMM;
var
  M, N, K, lda, ldb, ldc: Integer;
  h_A, h_B, h_C_initial, h_C_computed: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t; // Use hipDeviceptr_t for device pointers
  alpha, beta: rocblas_float_complex;
  blas_status: rocblas_status; // Use blas_status instead of status
  pm_status: rocblas_status;
  i: Integer;
begin

  // Initialize matrix dimensions
  M := 3;
  N := 3;
  K := 3;
  lda := M;
  ldb := K;
  ldc := M;

  // Set up test data
  SetLength(h_A, lda * K);
  SetLength(h_B, ldb * N);
  SetLength(h_C_initial, ldc * N);
  SetLength(h_C_computed, ldc * N);

  // Fill matrices A, B, and C with simple test data
  for i := 0 to High(h_A) do
  begin
    h_A[i].x := i + 1;
    h_A[i].y := 0;
  end;
  for i := 0 to High(h_B) do
  begin
    h_B[i].x := i + 1;
    h_B[i].y := 0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 1;
    h_C_initial[i].y := 0;
  end;

  // Copy initial C matrix for computation
  h_C_computed := Copy(h_C_initial);

  alpha.x := 1.0;
  alpha.y := 0.0;
  beta.x := 1.0;
  beta.y := 0.0;

  // Allocate device memory
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_float_complex) * lda * K));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_float_complex) * ldb * N));
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_float_complex) * ldc * N));

  // Copy data from host to device - Use @ to get pointer to host arrays
  AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_float_complex) * lda * K, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_float_complex) * ldb * N, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyHostToDevice));

  // Set pointer mode to host (alpha, beta are on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // Call rocblas_cgemm
  blas_status := rocblas_cgemm(
    handle,
    rocblas_operation_none, // transA
    rocblas_operation_none, // transB
    M, N, K,
    @alpha,
    d_A, lda,
    d_B, ldb,
    @beta,
    d_C, ldc
  );
  AssertEquals('rocblas_cgemm', rocblas_status_success, blas_status);

  // Copy result back to host - Use @ to get pointer to host array
  AssertEquals('hipMemcpy d_C result', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyDeviceToHost));

  // Basic check: verify result is not the initial matrix (indicating computation happened)
  // A more thorough check would involve calculating the expected result manually or with a reference.
  AssertTrue('Result matrix should differ from initial matrix', not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N));

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_ZGEMM;
var
  M, N, K, lda, ldb, ldc: Integer;
  h_A, h_B, h_C_initial, h_C_computed: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t; // Use hipDeviceptr_t for device pointers
  alpha, beta: rocblas_double_complex;
  blas_status: rocblas_status; // Use blas_status instead of status
  pm_status: rocblas_status;
  i: Integer;
begin

  // Initialize matrix dimensions
  M := 3;
  N := 3;
  K := 3;
  lda := M;
  ldb := K;
  ldc := M;

  // Set up test data
  SetLength(h_A, lda * K);
  SetLength(h_B, ldb * N);
  SetLength(h_C_initial, ldc * N);
  SetLength(h_C_computed, ldc * N);

  // Fill matrices A, B, and C with simple test data
  for i := 0 to High(h_A) do
  begin
    h_A[i].x := i + 1;
    h_A[i].y := 0;
  end;
  for i := 0 to High(h_B) do
  begin
    h_B[i].x := i + 1;
    h_B[i].y := 0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 1;
    h_C_initial[i].y := 0;
  end;

  // Copy initial C matrix for computation
  h_C_computed := Copy(h_C_initial);

  alpha.x := 1.0;
  alpha.y := 0.0;
  beta.x := 1.0;
  beta.y := 0.0;

  // Allocate device memory
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_double_complex) * lda * K));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_double_complex) * ldb * N));
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_double_complex) * ldc * N));

  // Copy data from host to device - Use @ to get pointer to host arrays
  AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_double_complex) * lda * K, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_double_complex) * ldb * N, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyHostToDevice));

  // Set pointer mode to host (alpha, beta are on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // Call rocblas_zgemm
  blas_status := rocblas_zgemm(
    handle,
    rocblas_operation_none, // transA
    rocblas_operation_none, // transB
    M, N, K,
    @alpha,
    d_A, lda,
    d_B, ldb,
    @beta,
    d_C, ldc
  );
  AssertEquals('rocblas_zgemm', rocblas_status_success, blas_status);

  // Copy result back to host - Use @ to get pointer to host array
  AssertEquals('hipMemcpy d_C result', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyDeviceToHost));

  // Basic check: verify result is not the initial matrix (indicating computation happened)
  AssertTrue('Result matrix should differ from initial matrix', not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N));

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_CHEMM;
var
  M, N, lda, ldb, ldc: Integer;
  side: rocblas_side;
  uplo: rocblas_fill;
  h_A, h_B, h_C_initial, h_C_computed: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t; // Use hipDeviceptr_t for device pointers
  alpha, beta: rocblas_float_complex;
  blas_status: rocblas_status; // Use blas_status instead of status
  pm_status: rocblas_status;
  i: Integer;
begin
  // Initialize matrix dimensions and parameters
  M := 3;
  N := 3;
  side := rocblas_side_left;
  uplo := rocblas_fill_upper; // Assuming Upper triangular part of A is used
  if side = rocblas_side_left then
    lda := M
  else
    lda := N;
  ldb := M;
  ldc := M;

  // Set up test data
  SetLength(h_A, lda * (M)); // Size for square matrix A
  SetLength(h_B, ldb * N);
  SetLength(h_C_initial, ldc * N);
  SetLength(h_C_computed, ldc * N);

  // Fill matrices A (Hermitian), B, and C with simple test data
  // For Hermitian A (3x3), fill upper triangle and set diagonal real, lower = conjugate transpose of upper
  // A = [ 1+0i  2+1i  3+0i ]
  //     [ 2-1i  4+0i  5+2i ]
  //     [ 3+0i  5-2i  6+0i ]
  h_A[0].x := 1.0; h_A[0].y := 0.0; // A[0][0]
  h_A[1].x := 2.0; h_A[1].y := 1.0; // A[1][0] (stored, but will be conjugated for lower)
  h_A[2].x := 3.0; h_A[2].y := 0.0; // A[2][0] (stored, but will be conjugated for lower)
  h_A[3].x := 2.0; h_A[3].y := -1.0; // A[0][1] (conjugate of A[1][0])
  h_A[4].x := 4.0; h_A[4].y := 0.0; // A[1][1]
  h_A[5].x := 5.0; h_A[5].y := 2.0; // A[2][1] (stored, but will be conjugated for lower)
  h_A[6].x := 3.0; h_A[6].y := 0.0; // A[0][2] (conjugate of A[2][0])
  h_A[7].x := 5.0; h_A[7].y := -2.0; // A[1][2] (conjugate of A[2][1])
  h_A[8].x := 6.0; h_A[8].y := 0.0; // A[2][2]

  for i := 0 to High(h_B) do
  begin
    h_B[i].x := i + 1;
    h_B[i].y := 0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 1;
    h_C_initial[i].y := 0;
  end;

  // Copy initial C matrix for computation
  h_C_computed := Copy(h_C_initial);

  alpha.x := 1.0;
  alpha.y := 0.0;
  beta.x := 1.0;
  beta.y := 0.0;

  // Allocate device memory
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_float_complex) * lda * (M)));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_float_complex) * ldb * N));
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_float_complex) * ldc * N));

  // Copy data from host to device - Use @ to get pointer to host arrays
  AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_float_complex) * lda * (M), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_float_complex) * ldb * N, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyHostToDevice));

  // Set pointer mode to host (alpha, beta are on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // Call rocblas_chemm
  blas_status := rocblas_chemm(
    handle,
    side,
    uplo,
    M, N,
    @alpha,
    d_A, lda,
    d_B, ldb,
    @beta,
    d_C, ldc
  );
  AssertEquals('rocblas_chemm', rocblas_status_success, blas_status);

  // Copy result back to host - Use @ to get pointer to host array
  AssertEquals('hipMemcpy d_C result', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyDeviceToHost));

  // Basic check: verify result is not the initial matrix (indicating computation happened)
  AssertTrue('Result matrix should differ from initial matrix', not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N));

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_ZHEMM;
var
  M, N, lda, ldb, ldc: Integer;
  side: rocblas_side;
  uplo: rocblas_fill;
  h_A, h_B, h_C_initial, h_C_computed: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t; // Use hipDeviceptr_t for device pointers
  alpha, beta: rocblas_double_complex;
  blas_status: rocblas_status; // Use blas_status instead of status
  pm_status: rocblas_status;
  i: Integer;
begin
  // Initialize matrix dimensions and parameters
  M := 3;
  N := 3;
  side := rocblas_side_left;
  uplo := rocblas_fill_upper; // Assuming Upper triangular part of A is used
  if side = rocblas_side_left then
    lda := M
  else
    lda := N;
  ldb := M;
  ldc := M;

  // Set up test data
  SetLength(h_A, lda * (M)); // Size for square matrix A
  SetLength(h_B, ldb * N);
  SetLength(h_C_initial, ldc * N);
  SetLength(h_C_computed, ldc * N);

  // Fill matrices A (Hermitian), B, and C with simple test data
  // For Hermitian A (3x3), fill upper triangle and set diagonal real, lower = conjugate transpose of upper
  // A = [ 1+0i  2+1i  3+0i ]
  //     [ 2-1i  4+0i  5+2i ]
  //     [ 3+0i  5-2i  6+0i ]
  h_A[0].x := 1.0; h_A[0].y := 0.0; // A[0][0]
  h_A[1].x := 2.0; h_A[1].y := 1.0; // A[1][0] (stored, but will be conjugated for lower)
  h_A[2].x := 3.0; h_A[2].y := 0.0; // A[2][0] (stored, but will be conjugated for lower)
  h_A[3].x := 2.0; h_A[3].y := -1.0; // A[0][1] (conjugate of A[1][0])
  h_A[4].x := 4.0; h_A[4].y := 0.0; // A[1][1]
  h_A[5].x := 5.0; h_A[5].y := 2.0; // A[2][1] (stored, but will be conjugated for lower)
  h_A[6].x := 3.0; h_A[6].y := 0.0; // A[0][2] (conjugate of A[2][0])
  h_A[7].x := 5.0; h_A[7].y := -2.0; // A[1][2] (conjugate of A[2][1])
  h_A[8].x := 6.0; h_A[8].y := 0.0; // A[2][2]

  for i := 0 to High(h_B) do
  begin
    h_B[i].x := i + 1;
    h_B[i].y := 0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 1;
    h_C_initial[i].y := 0;
  end;

  // Copy initial C matrix for computation
  h_C_computed := Copy(h_C_initial);

  alpha.x := 1.0;
  alpha.y := 0.0;
  beta.x := 1.0;
  beta.y := 0.0;

  // Allocate device memory
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_double_complex) * lda * (M)));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_double_complex) * ldb * N));
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_double_complex) * ldc * N));

  // Copy data from host to device - Use @ to get pointer to host arrays
  AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_double_complex) * lda * (M), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_double_complex) * ldb * N, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyHostToDevice));

  // Set pointer mode to host (alpha, beta are on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // Call rocblas_zhemm
  blas_status := rocblas_zhemm(
    handle,
    side,
    uplo,
    M, N,
    @alpha,
    d_A, lda,
    d_B, ldb,
    @beta,
    d_C, ldc
  );
  AssertEquals('rocblas_zhemm', rocblas_status_success, blas_status);

  // Copy result back to host - Use @ to get pointer to host array
  AssertEquals('hipMemcpy d_C result', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyDeviceToHost));

  // Basic check: verify result is not the initial matrix (indicating computation happened)
  AssertTrue('Result matrix should differ from initial matrix', not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N));

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;

procedure TTestRocBLAS_Level3.Test_STRMM;
var
  M, N, lda, ldb, ldc: Integer;
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  h_A, h_B_initial, h_B_computed: array of Single; // Real Single Precision
  d_A, d_B: hipDeviceptr_t;
  alpha: Single; // Real Single Precision Scalar
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i: Integer;
  idx: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  M := 2;
  N := 2;
  side := rocblas_side_left;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  lda := M; // A is MxM since side=left
  ldb := M; // Leading dimension of input B
  ldc := ldb; // Leading dimension of output C. For in-place, ldc = ldb.

  // 2. Set up test data
  SetLength(h_A, lda * M); // A is MxM
  SetLength(h_B_initial, ldb * N); // B is MxN
  SetLength(h_B_computed, ldb * N);

  // 3. Initialize host data
  for idx := 0 to High(h_A) do
  begin
    h_A[idx] := 0.0;
  end;
  for idx := 0 to High(h_B_initial) do
  begin
    h_B_initial[idx] := 0.0;
  end;

  // Fill triangular matrix A (Upper triangle)
  if uplo = rocblas_fill_upper then
  begin
    h_A[0 + 0 * lda] := 2.0; // A[0,0]
    h_A[0 + 1 * lda] := 1.0; // A[0,1]
    h_A[1 + 1 * lda] := 3.0; // A[1,1]
  end;

  // Fill matrix B initial
  for i := 0 to (M * N - 1) do
  begin
    h_B_initial[i] := i + 1;
  end;
  Move(h_B_initial[0], h_B_computed[0], SizeOf(Single) * Length(h_B_initial));

  alpha := 1.5;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil; // Will be used for both B (input) and C (output)
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Single) * Length(h_A)));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(Single) * Length(h_B_computed)));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Single) * Length(h_A), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B_computed[0], SizeOf(Single) * Length(h_B_computed), hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_strmm (Assuming FIXED 14-parameter signature)
  blas_status := rocblas_strmm(
    handle,
    side,
    uplo,
    transA,
    diag,
    M, N,
    @alpha,   // alpha (host pointer)
    d_A, lda, // A, lda
    d_B, ldb, // B (input matrix)
    d_B, ldc  // C (output matrix) - SAME POINTER, SAME LD for in-place
  );
  AssertEquals('rocblas_strmm should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_strmm', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_B D2H', HIP_SUCCESS, hipMemcpy(@h_B_computed[0], d_B, SizeOf(Single) * Length(h_B_computed), hipMemcpyDeviceToHost));

  // 10. Basic check
  AssertTrue('Result matrix B should differ from initial matrix B', not CompareMem(@h_B_initial[0], @h_B_computed[0], SizeOf(Single) * Length(h_B_computed)));

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
end;


procedure TTestRocBLAS_Level3.Test_DTRMM;
var
  M, N, lda, ldb, ldc: Integer;
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  h_A, h_B_initial, h_B_computed: array of Double; // Real Double Precision
  d_A, d_B: hipDeviceptr_t;
  alpha: Double; // Real Double Precision Scalar
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i: Integer;
  idx: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  M := 2;
  N := 2;
  side := rocblas_side_right; // Different side
  uplo := rocblas_fill_lower; // Different uplo
  transA := rocblas_operation_transpose; // Different transA
  diag := rocblas_diagonal_unit; // Different diag

  // lda for triangular matrix A (NxN if side=right)
  // ldb for matrix B (MxN)
  lda := N; // Since side=right, A is NxN
  ldb := M; // ldb >= max(1, M)
  ldc := ldb; // For in-place operation

  // 2. Set up test data
  SetLength(h_A, lda * N); // A is NxN
  SetLength(h_B_initial, ldb * N); // B is MxN
  SetLength(h_B_computed, ldb * N);

  // 3. Initialize host data
  for idx := 0 to High(h_A) do
  begin
    h_A[idx] := 0.0;
  end;
  for idx := 0 to High(h_B_initial) do
  begin
    h_B_initial[idx] := 0.0;
  end;

  // Fill triangular matrix A (Lower triangle)
  if uplo = rocblas_fill_lower then
  begin
    // Column 0
    // h_A[0 + 0 * lda] is diagonal, ignored for unit diag
    h_A[1 + 0 * lda] := 1.5; // A[1,0]
    // Column 1
    // h_A[0 + 1 * lda] is upper, ignored
    // h_A[1 + 1 * lda] is diagonal, ignored for unit diag
    if diag = rocblas_diagonal_unit then
    begin
       h_A[0 + 0 * lda] := 1.0; // Explicit unit diag for clarity
       h_A[1 + 1 * lda] := 1.0; // Explicit unit diag for clarity
    end
    else
    begin
       h_A[0 + 0 * lda] := 2.5; // A[0,0]
       h_A[1 + 1 * lda] := 4.5; // A[1,1]
    end;
  end;

  // Fill matrix B initial
  for i := 0 to (M * N - 1) do
  begin
    h_B_initial[i] := i + 1 + (i + 1) * 0.1;
  end;
  Move(h_B_initial[0], h_B_computed[0], SizeOf(Double) * Length(h_B_initial));

  alpha := 0.8;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Double) * Length(h_A)));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(Double) * Length(h_B_computed)));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Double) * Length(h_A), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B_computed[0], SizeOf(Double) * Length(h_B_computed), hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_dtrmm (Assuming FIXED 14-parameter signature)
  blas_status := rocblas_dtrmm(
    handle,
    side,
    uplo,
    transA,
    diag,
    M, N,
    @alpha,   // alpha (host pointer)
    d_A, lda, // A, lda
    d_B, ldb, // B (input matrix)
    d_B, ldc  // C (output matrix) - SAME POINTER, SAME LD for in-place
  );
  AssertEquals('rocblas_dtrmm should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_dtrmm', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_B D2H', HIP_SUCCESS, hipMemcpy(@h_B_computed[0], d_B, SizeOf(Double) * Length(h_B_computed), hipMemcpyDeviceToHost));

  // 10. Basic check
  AssertTrue('Result matrix B should differ from initial matrix B', not CompareMem(@h_B_initial[0], @h_B_computed[0], SizeOf(Double) * Length(h_B_computed)));

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
end;

procedure TTestRocBLAS_Level3.Test_CTRMM;
var
  M, N, lda, ldb, ldc: Integer; // Add ldc
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  h_A, h_B_initial, h_B_computed: array of rocblas_float_complex;
  d_A, d_B: hipDeviceptr_t; // d_B will be used for both B and C
  alpha: rocblas_float_complex;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i: Integer;
  idx: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  M := 2;
  N := 2;
  side := rocblas_side_left;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  lda := M; // A is MxM since side=left
  ldb := M; // Leading dimension of input B
  ldc := ldb; // Leading dimension of output C. For in-place, ldc = ldb.

  // 2. Set up test data
  SetLength(h_A, lda * M); // A is MxM
  SetLength(h_B_initial, ldb * N); // B is MxN
  SetLength(h_B_computed, ldb * N);

  // 3. Initialize host data
  for idx := 0 to High(h_A) do
  begin
    h_A[idx].x := 0.0;
    h_A[idx].y := 0.0;
  end;
  for idx := 0 to High(h_B_initial) do
  begin
    h_B_initial[idx].x := 0.0;
    h_B_initial[idx].y := 0.0;
  end;

  // Fill triangular matrix A (Upper triangle)
  if uplo = rocblas_fill_upper then
  begin
    h_A[0 + 0 * lda].x := 2.0; h_A[0 + 0 * lda].y := 1.0; // A[0,0]
    h_A[0 + 1 * lda].x := 1.0; h_A[0 + 1 * lda].y := 0.0; // A[0,1]
    h_A[1 + 1 * lda].x := 3.0; h_A[1 + 1 * lda].y := 0.0; // A[1,1]
  end;

  // Fill matrix B initial
  for i := 0 to (M * N - 1) do
  begin
    h_B_initial[i].x := i + 1;
    h_B_initial[i].y := 0.0;
  end;
  Move(h_B_initial[0], h_B_computed[0], SizeOf(rocblas_float_complex) * Length(h_B_initial));

  alpha.x := 1.5;
  alpha.y := -0.5;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil; // Will be used for both B (input) and C (output)
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_float_complex) * Length(h_A)));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_float_complex) * Length(h_B_computed)));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_float_complex) * Length(h_A), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B_computed[0], SizeOf(rocblas_float_complex) * Length(h_B_computed), hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  blas_status := rocblas_ctrmm(
    handle,
    side,
    uplo,
    transA,
    diag,
    M, N,
    @alpha,
    d_A, lda,
    d_B, ldb,  // B (input matrix)
    d_B, ldc   // C (output matrix) - SAME POINTER, SAME LD for in-place
  );

  AssertEquals('rocblas_ctrmm should return success', rocblas_status_success, blas_status);

  // 7. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ctrmm', HIP_SUCCESS, sync_status);

  // 8. Copy result D2H
  AssertEquals('hipMemcpy d_B D2H', HIP_SUCCESS, hipMemcpy(@h_B_computed[0], d_B, SizeOf(rocblas_float_complex) * Length(h_B_computed), hipMemcpyDeviceToHost));

  // 9. Basic check
  AssertTrue('Result matrix B should differ from initial matrix B', not CompareMem(@h_B_initial[0], @h_B_computed[0], SizeOf(rocblas_float_complex) * Length(h_B_computed)));

  // 10. Free device memory
  hipFree(d_A);
  hipFree(d_B);
end;

procedure TTestRocBLAS_Level3.Test_ZTRMM;
var
  M, N, lda, ldb, ldc: Integer;
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  h_A, h_B_initial, h_B_computed: array of rocblas_double_complex;
  d_A, d_B: hipDeviceptr_t;
  alpha: rocblas_double_complex;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i: Integer;
  idx: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  M := 2;
  N := 2;
  side := rocblas_side_right; // Different side
  uplo := rocblas_fill_lower; // Different uplo
  transA := rocblas_operation_transpose; // Different transA
  diag := rocblas_diagonal_unit; // Different diag

  // lda for triangular matrix A (NxN if side=right)
  // ldb for matrix B (MxN)
  lda := N; // Since side=right, A is NxN
  ldb := M; // ldb >= max(1, M)
  ldc := ldb; // For in-place operation

  // 2. Set up test data
  SetLength(h_A, lda * N); // A is NxN
  SetLength(h_B_initial, ldb * N); // B is MxN
  SetLength(h_B_computed, ldb * N);

  // 3. Initialize host data
  for idx := 0 to High(h_A) do
  begin
    h_A[idx].x := 0.0;
    h_A[idx].y := 0.0;
  end;
  for idx := 0 to High(h_B_initial) do
  begin
    h_B_initial[idx].x := 0.0;
    h_B_initial[idx].y := 0.0;
  end;

  // Fill triangular matrix A (Lower triangle)
  // A = [ 2.5-1.0i  0.0+0.0i ] (Upper part ignored)
  //     [ 1.0+0.5i  4.0-0.5i ] (Lower triangle)
  if uplo = rocblas_fill_lower then
  begin
    // Column 0
    // h_A[0 + 0 * lda] is diagonal, ignored for unit diag
    h_A[1 + 0 * lda].x := 1.0; h_A[1 + 0 * lda].y := 0.5; // A[1,0]
    // Column 1
    // h_A[0 + 1 * lda] is upper, ignored
    // h_A[1 + 1 * lda] is diagonal, ignored for unit diag
    if diag = rocblas_diagonal_unit then
    begin
       h_A[0 + 0 * lda].x := 1.0; h_A[0 + 0 * lda].y := 0.0; // Explicit unit diag for clarity
       h_A[1 + 1 * lda].x := 1.0; h_A[1 + 1 * lda].y := 0.0; // Explicit unit diag for clarity
    end
    else
    begin
       h_A[0 + 0 * lda].x := 2.5; h_A[0 + 0 * lda].y := -1.0; // A[0,0]
       h_A[1 + 1 * lda].x := 4.0; h_A[1 + 1 * lda].y := -0.5; // A[1,1]
    end;
  end;

  // Fill matrix B initial
  for i := 0 to (M * N - 1) do
  begin
    h_B_initial[i].x := i + 1;
    h_B_initial[i].y := (i + 1) * 0.1;
  end;
  Move(h_B_initial[0], h_B_computed[0], SizeOf(rocblas_double_complex) * Length(h_B_initial));

  alpha.x := 0.8;
  alpha.y := 0.2;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_double_complex) * Length(h_A)));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_double_complex) * Length(h_B_computed)));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_double_complex) * Length(h_A), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B_computed[0], SizeOf(rocblas_double_complex) * Length(h_B_computed), hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_ztrmm (Fixed Signature)
  blas_status := rocblas_ztrmm(
    handle,
    side,
    uplo,
    transA,
    diag,
    M, N,
    @alpha,
    d_A, lda,
    d_B, ldb,  // B (input matrix)
    d_B, ldc   // C (output matrix) - SAME POINTER, SAME LD for in-place
  );
  AssertEquals('rocblas_ztrmm should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ztrmm', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_B D2H', HIP_SUCCESS, hipMemcpy(@h_B_computed[0], d_B, SizeOf(rocblas_double_complex) * Length(h_B_computed), hipMemcpyDeviceToHost));

  // 10. Basic check
  AssertTrue('Result matrix B should differ from initial matrix B', not CompareMem(@h_B_initial[0], @h_B_computed[0], SizeOf(rocblas_double_complex) * Length(h_B_computed)));

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
end;

procedure TTestRocBLAS_Level3.Test_STRSM;
var
  M, N, lda, ldb: Integer;
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  h_A, h_B_initial, h_B_computed: array of Single; // Real Single Precision
  d_A, d_B: hipDeviceptr_t;
  alpha: Single; // Real Single Precision Scalar
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i: Integer;
  idx: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  M := 2;
  N := 2;
  side := rocblas_side_left;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  lda := M; // A is MxM since side=left
  ldb := M; // Leading dimension of B

  // 2. Set up test data
  SetLength(h_A, lda * M); // A is MxM
  SetLength(h_B_initial, ldb * N); // B is MxN
  SetLength(h_B_computed, ldb * N);

  // 3. Initialize host data
  for idx := 0 to High(h_A) do
  begin
    h_A[idx] := 0.0;
  end;
  for idx := 0 to High(h_B_initial) do
  begin
    h_B_initial[idx] := 0.0;
  end;

  // Fill triangular matrix A (Upper triangle, non-singular)
  // A = [ 2.0  1.0 ]
  //     [ 0.0  3.0 ]
  if uplo = rocblas_fill_upper then
  begin
    h_A[0 + 0 * lda] := 2.0; // A[0,0]
    h_A[0 + 1 * lda] := 1.0; // A[0,1]
    h_A[1 + 1 * lda] := 3.0; // A[1,1]
  end;

  // Fill matrix B initial (RHS)
  for i := 0 to (M * N - 1) do
  begin
    h_B_initial[i] := i + 1;
  end;
  Move(h_B_initial[0], h_B_computed[0], SizeOf(Single) * Length(h_B_initial));

  alpha := 2.0;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil; // Will be used for B (input) and X (output)
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Single) * Length(h_A)));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(Single) * Length(h_B_computed)));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Single) * Length(h_A), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B_computed[0], SizeOf(Single) * Length(h_B_computed), hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_strsm (12 parameters - CORRECT for TRSM)
  blas_status := rocblas_strsm(
    handle,
    side,
    uplo,
    transA,
    diag,
    M, N,
    @alpha,   // alpha (host pointer)
    d_A, lda, // A, lda
    d_B, ldb  // B (input RHS, contains solution X on output)
    // No separate C or ldc parameters for TRSM
  );
  AssertEquals('rocblas_strsm should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_strsm', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_B D2H', HIP_SUCCESS, hipMemcpy(@h_B_computed[0], d_B, SizeOf(Single) * Length(h_B_computed), hipMemcpyDeviceToHost));

  // 10. Basic check
  AssertTrue('Solution matrix B should differ from initial RHS matrix B', not CompareMem(@h_B_initial[0], @h_B_computed[0], SizeOf(Single) * Length(h_B_computed)));

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
end;


procedure TTestRocBLAS_Level3.Test_DTRSM;
var
  M, N, lda, ldb: Integer;
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  h_A, h_B_initial, h_B_computed: array of Double; // Real Double Precision
  d_A, d_B: hipDeviceptr_t;
  alpha: Double; // Real Double Precision Scalar
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i: Integer;
  idx: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  M := 2;
  N := 2;
  side := rocblas_side_right; // Different side
  uplo := rocblas_fill_lower; // Different uplo
  transA := rocblas_operation_transpose; // Different transA
  diag := rocblas_diagonal_unit; // Different diag

  // lda for triangular matrix A (NxN if side=right)
  // ldb for matrix B (MxN)
  lda := N; // Since side=right, A is NxN
  ldb := M; // ldb >= max(1, M)

  // 2. Set up test data
  SetLength(h_A, lda * N); // A is NxN
  SetLength(h_B_initial, ldb * N); // B is MxN
  SetLength(h_B_computed, ldb * N);

  // 3. Initialize host data
  for idx := 0 to High(h_A) do
  begin
    h_A[idx] := 0.0;
  end;
  for idx := 0 to High(h_B_initial) do
  begin
    h_B_initial[idx] := 0.0;
  end;

  // Fill triangular matrix A (Lower triangle, unit diagonal)
  // A = [ 1.0  0.0 ] (Upper part ignored)
  //     [ 1.5  1.0 ] (Lower triangle, diag implicitly 1)
  if uplo = rocblas_fill_lower then
  begin
    // Column 0
    // h_A[0 + 0 * lda] is diagonal, implicitly 1 for unit diag
    h_A[1 + 0 * lda] := 1.5; // A[1,0]
    // Column 1
    // h_A[0 + 1 * lda] is upper, ignored
    // h_A[1 + 1 * lda] is diagonal, implicitly 1 for unit diag
    // Explicitly set diagonal to 1 for clarity, though rocBLAS ignores it
    h_A[0 + 0 * lda] := 1.0;
    h_A[1 + 1 * lda] := 1.0;
  end;

  // Fill matrix B initial (RHS)
  for i := 0 to (M * N - 1) do
  begin
    h_B_initial[i] := i + 1 + (i + 1) * 0.1;
  end;
  Move(h_B_initial[0], h_B_computed[0], SizeOf(Double) * Length(h_B_initial));

  alpha := 0.5;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Double) * Length(h_A)));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(Double) * Length(h_B_computed)));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Double) * Length(h_A), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B_computed[0], SizeOf(Double) * Length(h_B_computed), hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_dtrsm (12 parameters - CORRECT for TRSM)
  blas_status := rocblas_dtrsm(
    handle,
    side,
    uplo,
    transA,
    diag,
    M, N,
    @alpha,   // alpha (host pointer)
    d_A, lda, // A, lda
    d_B, ldb  // B (input RHS, contains solution X on output)
    // No separate C or ldc parameters for TRSM
  );
  AssertEquals('rocblas_dtrsm should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_dtrsm', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_B D2H', HIP_SUCCESS, hipMemcpy(@h_B_computed[0], d_B, SizeOf(Double) * Length(h_B_computed), hipMemcpyDeviceToHost));

  // 10. Basic check
  AssertTrue('Solution matrix B should differ from initial RHS matrix B', not CompareMem(@h_B_initial[0], @h_B_computed[0], SizeOf(Double) * Length(h_B_computed)));

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
end;


procedure TTestRocBLAS_Level3.Test_CTRSM;
var
  M, N, lda, ldb: Integer; // No ldc needed for TRSM
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  h_A, h_B_initial, h_B_computed: array of rocblas_float_complex;
  d_A, d_B: hipDeviceptr_t;
  alpha: rocblas_float_complex;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i: Integer;
  idx: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  M := 2;
  N := 2;
  side := rocblas_side_left;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  lda := M; // A is MxM since side=left
  ldb := M; // Leading dimension of B
  // ldc := ldb; // Not needed for TRSM

  // 2. Set up test data
  SetLength(h_A, lda * M); // A is MxM
  SetLength(h_B_initial, ldb * N); // B is MxN
  SetLength(h_B_computed, ldb * N);

  // 3. Initialize host data
  for idx := 0 to High(h_A) do
  begin
    h_A[idx].x := 0.0;
    h_A[idx].y := 0.0;
  end;
  for idx := 0 to High(h_B_initial) do
  begin
    h_B_initial[idx].x := 0.0;
    h_B_initial[idx].y := 0.0;
  end;

  // Fill triangular matrix A (Upper triangle, non-singular)
  // A = [ 2.0+0.0i  1.0+1.0i ]
  //     [ 0.0+0.0i  3.0+0.0i ]
  if uplo = rocblas_fill_upper then
  begin
    h_A[0 + 0 * lda].x := 2.0; h_A[0 + 0 * lda].y := 0.0; // A[0,0]
    h_A[0 + 1 * lda].x := 1.0; h_A[0 + 1 * lda].y := 1.0; // A[0,1]
    h_A[1 + 1 * lda].x := 3.0; h_A[1 + 1 * lda].y := 0.0; // A[1,1]
  end;

  // Fill matrix B initial (RHS)
  for i := 0 to (M * N - 1) do
  begin
    h_B_initial[i].x := i + 1;
    h_B_initial[i].y := (i + 1) * 0.1;
  end;
  Move(h_B_initial[0], h_B_computed[0], SizeOf(rocblas_float_complex) * Length(h_B_initial));

  alpha.x := 2.0;
  alpha.y := 0.0;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil; // Will be used for B (input) and X (output)
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_float_complex) * Length(h_A)));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_float_complex) * Length(h_B_computed)));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_float_complex) * Length(h_A), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B_computed[0], SizeOf(rocblas_float_complex) * Length(h_B_computed), hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_ctrsm (12 parameters - CORRECT for TRSM)
  blas_status := rocblas_ctrsm(
    handle,
    side,
    uplo,
    transA,
    diag,
    M, N,
    @alpha,   // alpha (host pointer)
    d_A, lda, // A, lda
    d_B, ldb  // B (input RHS, contains solution X on output)
    // No separate C or ldc parameters for TRSM
  );
  AssertEquals('rocblas_ctrsm should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ctrsm', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_B D2H', HIP_SUCCESS, hipMemcpy(@h_B_computed[0], d_B, SizeOf(rocblas_float_complex) * Length(h_B_computed), hipMemcpyDeviceToHost));

  // 10. Basic check
  AssertTrue('Solution matrix B should differ from initial RHS matrix B', not CompareMem(@h_B_initial[0], @h_B_computed[0], SizeOf(rocblas_float_complex) * Length(h_B_computed)));

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
end;


procedure TTestRocBLAS_Level3.Test_ZTRSM;
var
  M, N, lda, ldb: Integer;
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  h_A, h_B_initial, h_B_computed: array of rocblas_double_complex;
  d_A, d_B: hipDeviceptr_t;
  alpha: rocblas_double_complex;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i: Integer;
  idx: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  M := 2;
  N := 2;
  side := rocblas_side_right; // Different side
  uplo := rocblas_fill_lower; // Different uplo
  transA := rocblas_operation_conjugate_transpose; // Different transA
  diag := rocblas_diagonal_unit; // Different diag

  // lda for triangular matrix A (NxN if side=right)
  // ldb for matrix B (MxN)
  lda := N; // Since side=right, A is NxN
  ldb := M; // ldb >= max(1, M)
  // ldc := ldb; // Not needed for TRSM

  // 2. Set up test data
  SetLength(h_A, lda * N); // A is NxN
  SetLength(h_B_initial, ldb * N); // B is MxN
  SetLength(h_B_computed, ldb * N);

  // 3. Initialize host data
  for idx := 0 to High(h_A) do
  begin
    h_A[idx].x := 0.0;
    h_A[idx].y := 0.0;
  end;
  for idx := 0 to High(h_B_initial) do
  begin
    h_B_initial[idx].x := 0.0;
    h_B_initial[idx].y := 0.0;
  end;

  // Fill triangular matrix A (Lower triangle, unit diagonal)
  // A = [ 1.0+0.0i  0.0+0.0i ] (Upper part ignored)
  //     [ 1.0+0.5i  1.0+0.0i ] (Lower triangle, diag implicitly 1)
  if uplo = rocblas_fill_lower then
  begin
    // Column 0
    // h_A[0 + 0 * lda] is diagonal, implicitly 1 for unit diag
    h_A[1 + 0 * lda].x := 1.0; h_A[1 + 0 * lda].y := 0.5; // A[1,0]
    // Column 1
    // h_A[0 + 1 * lda] is upper, ignored
    // h_A[1 + 1 * lda] is diagonal, implicitly 1 for unit diag
    // Explicitly set diagonal to 1 for clarity, though rocBLAS ignores it
    h_A[0 + 0 * lda].x := 1.0; h_A[0 + 0 * lda].y := 0.0;
    h_A[1 + 1 * lda].x := 1.0; h_A[1 + 1 * lda].y := 0.0;
  end;

  // Fill matrix B initial (RHS)
  for i := 0 to (M * N - 1) do
  begin
    h_B_initial[i].x := i + 1;
    h_B_initial[i].y := -(i + 1) * 0.1;
  end;
  Move(h_B_initial[0], h_B_computed[0], SizeOf(rocblas_double_complex) * Length(h_B_initial));

  alpha.x := 0.5;
  alpha.y := 0.0;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_double_complex) * Length(h_A)));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_double_complex) * Length(h_B_computed)));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_double_complex) * Length(h_A), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B_computed[0], SizeOf(rocblas_double_complex) * Length(h_B_computed), hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_ztrsm (12 parameters - CORRECT for TRSM)
  blas_status := rocblas_ztrsm(
    handle,
    side,
    uplo,
    transA,
    diag,
    M, N,
    @alpha,   // alpha (host pointer)
    d_A, lda, // A, lda
    d_B, ldb  // B (input RHS, contains solution X on output)
    // No separate C or ldc parameters for TRSM
  );
  AssertEquals('rocblas_ztrsm should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ztrsm', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_B D2H', HIP_SUCCESS, hipMemcpy(@h_B_computed[0], d_B, SizeOf(rocblas_double_complex) * Length(h_B_computed), hipMemcpyDeviceToHost));

  // 10. Basic check
  AssertTrue('Solution matrix B should differ from initial RHS matrix B', not CompareMem(@h_B_initial[0], @h_B_computed[0], SizeOf(rocblas_double_complex) * Length(h_B_computed)));

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
end;


procedure TTestRocBLAS_Level3.Test_SSYMM;
var
  M, N, lda, ldb, ldc: Integer;
  side: rocblas_side;
  uplo: rocblas_fill;
  h_A, h_B, h_C_initial, h_C_computed: array of Single; // Real Single Precision
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: Single; // Real Single Precision Scalars
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
  dim_A: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  M := 3;
  N := 3;
  side := rocblas_side_left;
  uplo := rocblas_fill_upper;
  if side = rocblas_side_left then
    dim_A := M
  else
    dim_A := N;

  lda := dim_A;
  ldb := M;
  ldc := M;

  // 2. Set up test data
  SetLength(h_A, lda * dim_A); // A is dim_A x dim_A
  SetLength(h_B, ldb * N);     // B is M x N
  SetLength(h_C_initial, ldc * N); // C is M x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data
  for i := 0 to High(h_A) do
  begin
    h_A[i] := 0.0;
  end;
  for i := 0 to High(h_B) do
  begin
    h_B[i] := 0.0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i] := 0.0;
  end;

  // Fill the specified triangle (upper) of symmetric matrix A and make it fully symmetric
  if uplo = rocblas_fill_upper then
  begin
    // Upper triangle values
    h_A[0 + 0 * lda] := 1.0; // A[0][0]
    h_A[0 + 1 * lda] := 2.0; // A[0][1]
    h_A[0 + 2 * lda] := 3.0; // A[0][2]
    h_A[1 + 1 * lda] := 4.0; // A[1][1]
    h_A[1 + 2 * lda] := 5.0; // A[1][2]
    h_A[2 + 2 * lda] := 6.0; // A[2][2]
    // Copy upper triangle to lower triangle to make it symmetric
    for i := 1 to dim_A - 1 do
      for j := 0 to i - 1 do
      begin
        h_A[i + j * lda] := h_A[j + i * lda];
      end;
  end;

  // Fill matrix B
  for i := 0 to High(h_B) do
  begin
    h_B[i] := i + 1;
  end;

  // Fill initial C matrix
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i] := 1.0;
  end;
  Move(h_C_initial[0], h_C_computed[0], SizeOf(Single) * Length(h_C_initial));

  alpha := 1.0;
  beta := 1.0;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  d_C := nil;
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Single) * lda * dim_A));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(Single) * ldb * N));
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(Single) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Single) * lda * dim_A, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(Single) * ldb * N, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(Single) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_ssymm (14 parameters)
  blas_status := rocblas_ssymm(
    handle,
    side,
    uplo,
    M, N,
    @alpha,   // alpha (host pointer)
    d_A, lda, // A, lda
    d_B, ldb, // B, ldb
    @beta,    // beta (host pointer)
    d_C, ldc  // C, ldc (C is output)
  );
  AssertEquals('rocblas_ssymm should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ssymm', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(Single) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check
  AssertTrue('Result matrix C should differ from initial matrix C', not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(Single) * ldc * N));

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_DSYMM;
var
  M, N, lda, ldb, ldc: Integer;
  side: rocblas_side;
  uplo: rocblas_fill;
  h_A, h_B, h_C_initial, h_C_computed: array of Double; // Real Double Precision
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: Double; // Real Double Precision Scalars
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
  dim_A: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  M := 3;
  N := 3;
  side := rocblas_side_right; // Different side
  uplo := rocblas_fill_lower; // Different uplo
  if side = rocblas_side_left then
    dim_A := M
  else
    dim_A := N;

  lda := dim_A;
  ldb := M;
  ldc := M;

  // 2. Set up test data
  SetLength(h_A, lda * dim_A); // A is dim_A x dim_A
  SetLength(h_B, ldb * N);     // B is M x N
  SetLength(h_C_initial, ldc * N); // C is M x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data
  for i := 0 to High(h_A) do
  begin
    h_A[i] := 0.0;
  end;
  for i := 0 to High(h_B) do
  begin
    h_B[i] := 0.0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i] := 0.0;
  end;

  // Fill the specified triangle (lower) of symmetric matrix A and make it fully symmetric
  if uplo = rocblas_fill_lower then
  begin
    // Lower triangle values
    h_A[0 + 0 * lda] := 1.5; // A[0][0]
    h_A[1 + 0 * lda] := 2.5; // A[1][0]
    h_A[2 + 0 * lda] := 3.5; // A[2][0]
    h_A[1 + 1 * lda] := 4.5; // A[1][1]
    h_A[2 + 1 * lda] := 5.5; // A[2][1]
    h_A[2 + 2 * lda] := 6.5; // A[2][2]
    // Copy lower triangle to upper triangle to make it symmetric
    for i := 0 to dim_A - 2 do
      for j := i + 1 to dim_A - 1 do
      begin
        h_A[i + j * lda] := h_A[j + i * lda];
      end;
  end;

  // Fill matrix B
  for i := 0 to High(h_B) do
  begin
    h_B[i] := i + 1 + (i + 1) * 0.1;
  end;

  // Fill initial C matrix
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i] := 1.5;
  end;
  Move(h_C_initial[0], h_C_computed[0], SizeOf(Double) * Length(h_C_initial));

  alpha := 0.8;
  beta := 1.2;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  d_C := nil;
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Double) * lda * dim_A));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(Double) * ldb * N));
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(Double) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Double) * lda * dim_A, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(Double) * ldb * N, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(Double) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_dsymm (14 parameters)
  blas_status := rocblas_dsymm(
    handle,
    side,
    uplo,
    M, N,
    @alpha,   // alpha (host pointer)
    d_A, lda, // A, lda
    d_B, ldb, // B, ldb
    @beta,    // beta (host pointer)
    d_C, ldc  // C, ldc (C is output)
  );
  AssertEquals('rocblas_dsymm should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_dsymm', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(Double) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check
  AssertTrue('Result matrix C should differ from initial matrix C', not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(Double) * ldc * N));

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;


// Assuming Test_CSYMM is already implemented correctly as per previous discussions
// and uses the 14-parameter signature for rocblas_csymm.
procedure TTestRocBLAS_Level3.Test_CSYMM;
var
  M, N, lda, ldb, ldc: Integer;
  side: rocblas_side;
  uplo: rocblas_fill;
  h_A, h_B, h_C_initial, h_C_computed: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_float_complex;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
  dim_A: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  M := 3;
  N := 3;
  side := rocblas_side_left;
  uplo := rocblas_fill_upper;
  if side = rocblas_side_left then
    dim_A := M
  else
    dim_A := N;

  lda := dim_A;
  ldb := M;
  ldc := M;

  // 2. Set up test data
  SetLength(h_A, lda * dim_A);
  SetLength(h_B, ldb * N);
  SetLength(h_C_initial, ldc * N);
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data
  for i := 0 to High(h_A) do
  begin
    h_A[i].x := 0.0;
    h_A[i].y := 0.0;
  end;
  for i := 0 to High(h_B) do
  begin
    h_B[i].x := 0.0;
    h_B[i].y := 0.0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 0.0;
    h_C_initial[i].y := 0.0;
  end;

  // Fill the specified triangle (upper) of symmetric matrix A
  if uplo = rocblas_fill_upper then
  begin
    h_A[0 + 0 * lda].x := 1.0; h_A[0 + 0 * lda].y := 0.0; // A[0][0]
    h_A[0 + 1 * lda].x := 2.0; h_A[0 + 1 * lda].y := 1.0; // A[0][1]
    h_A[0 + 2 * lda].x := 3.0; h_A[0 + 2 * lda].y := 0.0; // A[0][2]
    h_A[1 + 1 * lda].x := 4.0; h_A[1 + 1 * lda].y := 0.0; // A[1][1]
    h_A[1 + 2 * lda].x := 5.0; h_A[1 + 2 * lda].y := 2.0; // A[1][2]
    h_A[2 + 2 * lda].x := 6.0; h_A[2 + 2 * lda].y := 0.0; // A[2][2]
    // Copy upper triangle to lower triangle to make it symmetric
    for i := 1 to dim_A - 1 do
      for j := 0 to i - 1 do
      begin
        h_A[i + j * lda].x := h_A[j + i * lda].x;
        h_A[i + j * lda].y := h_A[j + i * lda].y; // Simple copy for symmetric
      end;
  end;

  // Fill matrix B
  for i := 0 to High(h_B) do
  begin
    h_B[i].x := i + 1;
    h_B[i].y := (i + 1) * 0.1;
  end;

  // Fill initial C matrix
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 1.0;
    h_C_initial[i].y := 0.0;
  end;
  Move(h_C_initial[0], h_C_computed[0], SizeOf(rocblas_float_complex) * Length(h_C_initial));

  alpha.x := 1.0;
  alpha.y := 0.0;
  beta.x := 1.0;
  beta.y := 0.0;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  d_C := nil;
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_float_complex) * lda * dim_A));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_float_complex) * ldb * N));
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_float_complex) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_float_complex) * lda * dim_A, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_float_complex) * ldb * N, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_csymm (14 parameters)
  blas_status := rocblas_csymm(
    handle,
    side,
    uplo,
    M, N,
    @alpha,   // alpha (host pointer)
    d_A, lda, // A, lda
    d_B, ldb, // B, ldb
    @beta,    // beta (host pointer)
    d_C, ldc  // C, ldc (C is output)
  );
  AssertEquals('rocblas_csymm should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_csymm', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check
  AssertTrue('Result matrix C should differ from initial matrix C', not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N));

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;


// Assuming Test_ZSYMM is already implemented correctly as per previous discussions
// and uses the 14-parameter signature for rocblas_zsymm.
procedure TTestRocBLAS_Level3.Test_ZSYMM;
var
  M, N, lda, ldb, ldc: Integer;
  side: rocblas_side;
  uplo: rocblas_fill;
  h_A, h_B, h_C_initial, h_C_computed: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_double_complex;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
  dim_A: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  M := 3;
  N := 3;
  side := rocblas_side_right; // Different side
  uplo := rocblas_fill_lower; // Different uplo
  if side = rocblas_side_left then
    dim_A := M
  else
    dim_A := N;

  lda := dim_A;
  ldb := M;
  ldc := M;

  // 2. Set up test data
  SetLength(h_A, lda * dim_A);
  SetLength(h_B, ldb * N);
  SetLength(h_C_initial, ldc * N);
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data
  for i := 0 to High(h_A) do
  begin
    h_A[i].x := 0.0;
    h_A[i].y := 0.0;
  end;
  for i := 0 to High(h_B) do
  begin
    h_B[i].x := 0.0;
    h_B[i].y := 0.0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 0.0;
    h_C_initial[i].y := 0.0;
  end;

  // Fill the specified triangle (lower) of symmetric matrix A
  if uplo = rocblas_fill_lower then
  begin
    h_A[0 + 0 * lda].x := 1.5; h_A[0 + 0 * lda].y := -0.5; // A[0][0]
    h_A[1 + 0 * lda].x := 2.5; h_A[1 + 0 * lda].y := 0.5;  // A[1][0]
    h_A[2 + 0 * lda].x := 3.5; h_A[2 + 0 * lda].y := 1.5;  // A[2][0]
    h_A[1 + 1 * lda].x := 4.5; h_A[1 + 1 * lda].y := -0.5; // A[1][1]
    h_A[2 + 1 * lda].x := 5.5; h_A[2 + 1 * lda].y := 0.0;  // A[2][1]
    h_A[2 + 2 * lda].x := 6.5; h_A[2 + 2 * lda].y := 0.5;  // A[2][2]
    // Copy lower triangle to upper triangle to make it symmetric
    for i := 0 to dim_A - 2 do
      for j := i + 1 to dim_A - 1 do
      begin
        h_A[i + j * lda].x := h_A[j + i * lda].x;
        h_A[i + j * lda].y := h_A[j + i * lda].y; // Simple copy for symmetric
      end;
  end;

  // Fill matrix B
  for i := 0 to High(h_B) do
  begin
    h_B[i].x := i + 1;
    h_B[i].y := -(i + 1) * 0.1;
  end;

  // Fill initial C matrix
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 1.5;
    h_C_initial[i].y := 0.5;
  end;
  Move(h_C_initial[0], h_C_computed[0], SizeOf(rocblas_double_complex) * Length(h_C_initial));

  alpha.x := 0.8;
  alpha.y := 0.2;
  beta.x := 1.2;
  beta.y := -0.3;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  d_C := nil;
  AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_double_complex) * lda * dim_A));
  AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_double_complex) * ldb * N));
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_double_complex) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_double_complex) * lda * dim_A, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_double_complex) * ldb * N, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_zsymm (14 parameters)
  blas_status := rocblas_zsymm(
    handle,
    side,
    uplo,
    M, N,
    @alpha,   // alpha (host pointer)
    d_A, lda, // A, lda
    d_B, ldb, // B, ldb
    @beta,    // beta (host pointer)
    d_C, ldc  // C, ldc (C is output)
  );
  AssertEquals('rocblas_zsymm should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_zsymm', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check
  AssertTrue('Result matrix C should differ from initial matrix C', not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N));

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;

procedure TTestRocBLAS_Level3.Test_CHERK;
var
  N, K, lda, ldc: Integer;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  h_A, h_C_initial, h_C_computed: array of rocblas_float_complex;
  d_A, d_C: hipDeviceptr_t;
  alpha, beta: Single; // alpha and beta are real for HERK
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  N := 3; // C is N x N
  K := 2; // A is N x K (assuming no transA) or K x N (assuming transA)
  uplo := rocblas_fill_upper; // Update upper triangle of C
  transA := rocblas_operation_none; // Operation: C = alpha*A*A^H + beta*C

  if transA = rocblas_operation_none then
    lda := N // A is N x K
  else
    lda := K; // A is K x N

  ldc := N; // C is N x N

  // 2. Set up test data
  if transA = rocblas_operation_none then
    SetLength(h_A, lda * K) // A is N x K
  else
    SetLength(h_A, lda * N); // A is K x N

  SetLength(h_C_initial, ldc * N); // C is N x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data
  for i := 0 to High(h_A) do
  begin
    h_A[i].x := 0.0;
    h_A[i].y := 0.0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 0.0;
    h_C_initial[i].y := 0.0;
  end;

  // Fill matrix A with test data
  // A (3x2) = [ 1+1i  2+0i ]
  //           [ 3+0i  4+1i ]
  //           [ 5+0i  6+0i ]
  if transA = rocblas_operation_none then
  begin
    h_A[0 + 0 * lda].x := 1.0; h_A[0 + 0 * lda].y := 1.0; // A[0][0]
    h_A[1 + 0 * lda].x := 3.0; h_A[1 + 0 * lda].y := 0.0; // A[1][0]
    h_A[2 + 0 * lda].x := 5.0; h_A[2 + 0 * lda].y := 0.0; // A[2][0]
    h_A[0 + 1 * lda].x := 2.0; h_A[0 + 1 * lda].y := 0.0; // A[0][1]
    h_A[1 + 1 * lda].x := 4.0; h_A[1 + 1 * lda].y := 1.0; // A[1][1]
    h_A[2 + 1 * lda].x := 6.0; h_A[2 + 1 * lda].y := 0.0; // A[2][1]
  end;

  // Fill initial Hermitian matrix C
  // C (3x3) = [ 2+0i  1+1i  3+0i ] (Upper triangle)
  //           [ 1-1i  3+0i  2+1i ]
  //           [ 3+0i  2-1i  1+0i ] (Lower triangle, conjugate transpose of upper)
  h_C_initial[0 + 0 * ldc].x := 2.0; h_C_initial[0 + 0 * ldc].y := 0.0; // C[0][0]
  h_C_initial[1 + 1 * ldc].x := 3.0; h_C_initial[1 + 1 * ldc].y := 0.0; // C[1][1]
  h_C_initial[2 + 2 * ldc].x := 1.0; h_C_initial[2 + 2 * ldc].y := 0.0; // C[2][2]

  if uplo = rocblas_fill_upper then
  begin
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1]
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0; // C[0][2]
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0; // C[1][2]
    // Fill lower triangle based on upper (Hermitian property)
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := -1.0; // C[1][0] = conj(C[0][1])
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0] = conj(C[0][2])
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := -1.0; // C[2][1] = conj(C[1][2])
  end
  else // rocblas_fill_lower
  begin
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := -1.0; // C[1][0]
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0]
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := -1.0; // C[2][1]
    // Fill upper triangle based on lower (Hermitian property)
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1] = conj(C[1][0])
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0;  // C[0][2] = conj(C[2][0])
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0;  // C[1][2] = conj(C[2][1])
  end;

  // Copy initial C matrix for computation
  Move(h_C_initial[0], h_C_computed[0], SizeOf(rocblas_float_complex) * Length(h_C_initial));

  alpha := 1.0;
  beta := 1.0;

  // 4. Allocate device memory
  d_A := nil;
  d_C := nil;
  if transA = rocblas_operation_none then
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_float_complex) * N * K))
  else
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_float_complex) * K * N));

  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_float_complex) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  if transA = rocblas_operation_none then
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_float_complex) * N * K, hipMemcpyHostToDevice))
  else
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_float_complex) * K * N, hipMemcpyHostToDevice));

  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode (alpha, beta are real scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_cherk (11 parameters)
  blas_status := rocblas_cherk(
    handle,
    uplo,
    transA,
    N, K,
    @alpha,   // alpha (real, host pointer)
    d_A, lda, // A, lda
    @beta,    // beta (real, host pointer)
    d_C, ldc  // C, ldc (C is Hermitian matrix, output)
  );
  AssertEquals('rocblas_cherk should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_cherk', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check (check if updated part differs)
  AssertTrue('Result matrix C should differ from initial matrix C in updated triangle',
    not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N));
  // A more thorough check would validate the mathematical result against a reference.

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_ZHERK;
var
  N, K, lda, ldc: Integer;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  h_A, h_C_initial, h_C_computed: array of rocblas_double_complex;
  d_A, d_C: hipDeviceptr_t;
  alpha, beta: Double; // alpha and beta are real for HERK
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  N := 3; // C is N x N
  K := 2; // A is N x K (assuming no transA) or K x N (assuming transA)
  uplo := rocblas_fill_lower; // Update lower triangle of C
  transA := rocblas_operation_conjugate_transpose; // Operation: C = alpha*A^H*A + beta*C

  if transA = rocblas_operation_none then
    lda := N // A is N x K
  else
    lda := K; // A is K x N

  ldc := N; // C is N x N

  // 2. Set up test data
  if transA = rocblas_operation_none then
    SetLength(h_A, lda * K) // A is N x K
  else
    SetLength(h_A, lda * N); // A is K x N

  SetLength(h_C_initial, ldc * N); // C is N x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data
  for i := 0 to High(h_A) do
  begin
    h_A[i].x := 0.0;
    h_A[i].y := 0.0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 0.0;
    h_C_initial[i].y := 0.0;
  end;

  // Fill matrix A with test data
  // A (2x3) = [ 1+1i  3+0i  5+0i ]
  //           [ 2+0i  4+1i  6+0i ]
  if transA = rocblas_operation_conjugate_transpose then
  begin
    h_A[0 + 0 * lda].x := 1.0; h_A[0 + 0 * lda].y := 1.0; // A[0][0]
    h_A[0 + 1 * lda].x := 3.0; h_A[0 + 1 * lda].y := 0.0; // A[0][1]
    h_A[0 + 2 * lda].x := 5.0; h_A[0 + 2 * lda].y := 0.0; // A[0][2]
    h_A[1 + 0 * lda].x := 2.0; h_A[1 + 0 * lda].y := 0.0; // A[1][0]
    h_A[1 + 1 * lda].x := 4.0; h_A[1 + 1 * lda].y := 1.0; // A[1][1]
    h_A[1 + 2 * lda].x := 6.0; h_A[1 + 2 * lda].y := 0.0; // A[1][2]
  end;

  // Fill initial Hermitian matrix C
  // C (3x3) = [ 2+0i  1+1i  3+0i ]
  //           [ 1-1i  3+0i  2+1i ]
  //           [ 3+0i  2-1i  1+0i ]
  h_C_initial[0 + 0 * ldc].x := 2.0; h_C_initial[0 + 0 * ldc].y := 0.0; // C[0][0]
  h_C_initial[1 + 1 * ldc].x := 3.0; h_C_initial[1 + 1 * ldc].y := 0.0; // C[1][1]
  h_C_initial[2 + 2 * ldc].x := 1.0; h_C_initial[2 + 2 * ldc].y := 0.0; // C[2][2]

  if uplo = rocblas_fill_upper then
  begin
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1]
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0; // C[0][2]
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0; // C[1][2]
    // Fill lower triangle based on upper (Hermitian property)
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := -1.0; // C[1][0] = conj(C[0][1])
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0] = conj(C[0][2])
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := -1.0; // C[2][1] = conj(C[1][2])
  end
  else // rocblas_fill_lower
  begin
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := -1.0; // C[1][0]
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0]
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := -1.0; // C[2][1]
    // Fill upper triangle based on lower (Hermitian property)
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1] = conj(C[1][0])
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0;  // C[0][2] = conj(C[2][0])
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0;  // C[1][2] = conj(C[2][1])
  end;

  // Copy initial C matrix for computation
  Move(h_C_initial[0], h_C_computed[0], SizeOf(rocblas_double_complex) * Length(h_C_initial));

  alpha := 0.5;
  beta := 2.0;

  // 4. Allocate device memory
  d_A := nil;
  d_C := nil;
  if transA = rocblas_operation_none then
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_double_complex) * N * K))
  else
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_double_complex) * K * N));

  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_double_complex) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  if transA = rocblas_operation_none then
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_double_complex) * N * K, hipMemcpyHostToDevice))
  else
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_double_complex) * K * N, hipMemcpyHostToDevice));

  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode (alpha, beta are real scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_zherk (11 parameters)
  blas_status := rocblas_zherk(
    handle,
    uplo,
    transA,
    N, K,
    @alpha,   // alpha (real, host pointer)
    d_A, lda, // A, lda
    @beta,    // beta (real, host pointer)
    d_C, ldc  // C, ldc (C is Hermitian matrix, output)
  );
  AssertEquals('rocblas_zherk should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_zherk', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check (check if updated part differs)
  AssertTrue('Result matrix C should differ from initial matrix C in updated triangle',
    not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N));
  // A more thorough check would validate the mathematical result against a reference.

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_SSYRK;
var
  N, K, lda, ldc: Integer;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  h_A, h_C_initial, h_C_computed: array of Single; // Real Single Precision
  d_A, d_C: hipDeviceptr_t;
  alpha, beta: Single; // alpha and beta are real scalars
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  N := 3; // C is N x N (symmetric)
  K := 2; // A is N x K (if no trans) or K x N (if trans)
  uplo := rocblas_fill_upper; // Update upper triangle of C
  transA := rocblas_operation_none; // Operation: C = alpha*A*A^T + beta*C

  // Calculate leading dimensions
  if transA = rocblas_operation_none then
    lda := N // A is N x K
  else
    lda := K; // A is K x N
  ldc := N; // C is N x N

  // 2. Set up test data sizes
  if transA = rocblas_operation_none then
    SetLength(h_A, lda * K) // A is N x K
  else
    SetLength(h_A, lda * N); // A is K x N
  SetLength(h_C_initial, ldc * N); // C is N x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data to zero
  for i := 0 to High(h_A) do
    h_A[i] := 0.0;
  for i := 0 to High(h_C_initial) do
    h_C_initial[i] := 0.0;

  // Fill matrix A with test data
  // Example: A (3x2) = [ 1.0  2.0 ]
  //                   [ 3.0  4.0 ]
  //                   [ 5.0  6.0 ]
  if transA = rocblas_operation_none then
  begin
    h_A[0 + 0 * lda] := 1.0; // A[0][0]
    h_A[1 + 0 * lda] := 3.0; // A[1][0]
    h_A[2 + 0 * lda] := 5.0; // A[2][0]
    h_A[0 + 1 * lda] := 2.0; // A[0][1]
    h_A[1 + 1 * lda] := 4.0; // A[1][1]
    h_A[2 + 1 * lda] := 6.0; // A[2][1]
  end;

  // Fill initial symmetric matrix C
  // Example: C (3x3) = [ 2.0  1.0  3.0 ] (Upper triangle)
  //                    [ 1.0  3.0  2.0 ]
  //                    [ 3.0  2.0  1.0 ] (Lower triangle, copy of upper)
  h_C_initial[0 + 0 * ldc] := 2.0; // C[0][0]
  h_C_initial[1 + 1 * ldc] := 3.0; // C[1][1]
  h_C_initial[2 + 2 * ldc] := 1.0; // C[2][2]

  if uplo = rocblas_fill_upper then
  begin
    h_C_initial[0 + 1 * ldc] := 1.0; // C[0][1]
    h_C_initial[0 + 2 * ldc] := 3.0; // C[0][2]
    h_C_initial[1 + 2 * ldc] := 2.0; // C[1][2]
    // Fill lower triangle based on upper (symmetric property)
    h_C_initial[1 + 0 * ldc] := 1.0; // C[1][0] = C[0][1]
    h_C_initial[2 + 0 * ldc] := 3.0; // C[2][0] = C[0][2]
    h_C_initial[2 + 1 * ldc] := 2.0; // C[2][1] = C[1][2]
  end
  else // rocblas_fill_lower
  begin
    h_C_initial[1 + 0 * ldc] := 1.0; // C[1][0]
    h_C_initial[2 + 0 * ldc] := 3.0; // C[2][0]
    h_C_initial[2 + 1 * ldc] := 2.0; // C[2][1]
    // Fill upper triangle based on lower (symmetric property)
    h_C_initial[0 + 1 * ldc] := 1.0; // C[0][1] = C[1][0]
    h_C_initial[0 + 2 * ldc] := 3.0; // C[0][2] = C[2][0]
    h_C_initial[1 + 2 * ldc] := 2.0; // C[1][2] = C[2][1]
  end;

  // Copy initial C matrix for computation
  Move(h_C_initial[0], h_C_computed[0], SizeOf(Single) * Length(h_C_initial));

  alpha := 1.0;
  beta := 1.0;

  // 4. Allocate device memory
  d_A := nil;
  d_C := nil;
  if transA = rocblas_operation_none then
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Single) * N * K))
  else
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Single) * K * N));
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(Single) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  if transA = rocblas_operation_none then
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Single) * N * K, hipMemcpyHostToDevice))
  else
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Single) * K * N, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(Single) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode (alpha, beta are real scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_ssyrk (11 parameters)
  blas_status := rocblas_ssyrk(
    handle,
    uplo,
    transA,
    N, K,
    @alpha,   // alpha (real, host pointer)
    d_A, lda, // A, lda
    @beta,    // beta (real, host pointer)
    d_C, ldc  // C, ldc (C is symmetric matrix, output)
  );
  AssertEquals('rocblas_ssyrk should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ssyrk', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(Single) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check (check if the matrix differs)
  AssertTrue('Result matrix C should differ from initial matrix C',
    not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(Single) * ldc * N));
  // A more thorough check would validate the mathematical result against a reference.

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_DSYRK;
var
  N, K, lda, ldc: Integer;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  h_A, h_C_initial, h_C_computed: array of Double; // Real Double Precision
  d_A, d_C: hipDeviceptr_t;
  alpha, beta: Double; // alpha and beta are real scalars
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  N := 3; // C is N x N (symmetric)
  K := 2; // A is N x K (if no trans) or K x N (if trans)
  uplo := rocblas_fill_lower; // Update lower triangle of C
  transA := rocblas_operation_transpose; // Operation: C = alpha*A^T*A + beta*C

  // Calculate leading dimensions
  if transA = rocblas_operation_none then
    lda := N
  else
    lda := K;
  ldc := N; // C is N x N

  // 2. Set up test data sizes
  if transA = rocblas_operation_none then
    SetLength(h_A, lda * K) // A is N x K
  else
    SetLength(h_A, lda * N); // A is K x N
  SetLength(h_C_initial, ldc * N); // C is N x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data to zero
  for i := 0 to High(h_A) do
    h_A[i] := 0.0;
  for i := 0 to High(h_C_initial) do
    h_C_initial[i] := 0.0;

  // Fill matrix A with test data
  // Example: A (2x3) = [ 1.0  3.0  5.0 ]
  //                   [ 2.0  4.0  6.0 ]
  if transA = rocblas_operation_transpose then
  begin
    h_A[0 + 0 * lda] := 1.0; // A[0][0]
    h_A[0 + 1 * lda] := 3.0; // A[0][1]
    h_A[0 + 2 * lda] := 5.0; // A[0][2]
    h_A[1 + 0 * lda] := 2.0; // A[1][0]
    h_A[1 + 1 * lda] := 4.0; // A[1][1]
    h_A[1 + 2 * lda] := 6.0; // A[1][2]
  end;

  // Fill initial symmetric matrix C
  // Example: C (3x3) = [ 2.0  1.0  3.0 ]
  //                    [ 1.0  3.0  2.0 ]
  //                    [ 3.0  2.0  1.0 ]
  h_C_initial[0 + 0 * ldc] := 2.0; // C[0][0]
  h_C_initial[1 + 1 * ldc] := 3.0; // C[1][1]
  h_C_initial[2 + 2 * ldc] := 1.0; // C[2][2]

  if uplo = rocblas_fill_upper then
  begin
    h_C_initial[0 + 1 * ldc] := 1.0; // C[0][1]
    h_C_initial[0 + 2 * ldc] := 3.0; // C[0][2]
    h_C_initial[1 + 2 * ldc] := 2.0; // C[1][2]
    // Fill lower triangle based on upper (symmetric property)
    h_C_initial[1 + 0 * ldc] := 1.0; // C[1][0] = C[0][1]
    h_C_initial[2 + 0 * ldc] := 3.0; // C[2][0] = C[0][2]
    h_C_initial[2 + 1 * ldc] := 2.0; // C[2][1] = C[1][2]
  end
  else // rocblas_fill_lower
  begin
    h_C_initial[1 + 0 * ldc] := 1.0; // C[1][0]
    h_C_initial[2 + 0 * ldc] := 3.0; // C[2][0]
    h_C_initial[2 + 1 * ldc] := 2.0; // C[2][1]
    // Fill upper triangle based on lower (symmetric property)
    h_C_initial[0 + 1 * ldc] := 1.0; // C[0][1] = C[1][0]
    h_C_initial[0 + 2 * ldc] := 3.0; // C[0][2] = C[2][0]
    h_C_initial[1 + 2 * ldc] := 2.0; // C[1][2] = C[2][1]
  end;

  // Copy initial C matrix for computation
  Move(h_C_initial[0], h_C_computed[0], SizeOf(Double) * Length(h_C_initial));

  alpha := 0.5;
  beta := 2.0;

  // 4. Allocate device memory
  d_A := nil;
  d_C := nil;
  if transA = rocblas_operation_none then
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Double) * N * K))
  else
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Double) * K * N));
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(Double) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  if transA = rocblas_operation_none then
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Double) * N * K, hipMemcpyHostToDevice))
  else
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Double) * K * N, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(Double) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode (alpha, beta are real scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_dsyrk (11 parameters)
  blas_status := rocblas_dsyrk(
    handle,
    uplo,
    transA,
    N, K,
    @alpha,   // alpha (real, host pointer)
    d_A, lda, // A, lda
    @beta,    // beta (real, host pointer)
    d_C, ldc  // C, ldc (C is symmetric matrix, output)
  );
  AssertEquals('rocblas_dsyrk should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_dsyrk', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(Double) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check (check if the matrix differs)
  AssertTrue('Result matrix C should differ from initial matrix C',
    not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(Double) * ldc * N));
  // A more thorough check would validate the mathematical result against a reference.

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_CSYRK;
var
  N, K, lda, ldc: Integer;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  h_A, h_C_initial, h_C_computed: array of rocblas_float_complex;
  d_A, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_float_complex; // alpha and beta are complex scalars
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  N := 3; // C is N x N (symmetric)
  K := 2; // A is N x K (if no trans) or K x N (if trans)
  uplo := rocblas_fill_upper; // Update upper triangle of C
  transA := rocblas_operation_none; // Operation: C = alpha*A*A^T + beta*C

  // Calculate leading dimensions
  if transA = rocblas_operation_none then
    lda := N // A is N x K
  else
    lda := K; // A is K x N
  ldc := N; // C is N x N

  // 2. Set up test data sizes
  if transA = rocblas_operation_none then
    SetLength(h_A, lda * K) // A is N x K
  else
    SetLength(h_A, lda * N); // A is K x N
  SetLength(h_C_initial, ldc * N); // C is N x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data to zero
  for i := 0 to High(h_A) do
  begin
    h_A[i].x := 0.0;
    h_A[i].y := 0.0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 0.0;
    h_C_initial[i].y := 0.0;
  end;

  // Fill matrix A with test data
  // Example: A (3x2) = [ 1+1i  2+0i ]
  //                   [ 3+0i  4+1i ]
  //                   [ 5+0i  6+0i ]
  if transA = rocblas_operation_none then
  begin
    h_A[0 + 0 * lda].x := 1.0; h_A[0 + 0 * lda].y := 1.0; // A[0][0]
    h_A[1 + 0 * lda].x := 3.0; h_A[1 + 0 * lda].y := 0.0; // A[1][0]
    h_A[2 + 0 * lda].x := 5.0; h_A[2 + 0 * lda].y := 0.0; // A[2][0]
    h_A[0 + 1 * lda].x := 2.0; h_A[0 + 1 * lda].y := 0.0; // A[0][1]
    h_A[1 + 1 * lda].x := 4.0; h_A[1 + 1 * lda].y := 1.0; // A[1][1]
    h_A[2 + 1 * lda].x := 6.0; h_A[2 + 1 * lda].y := 0.0; // A[2][1]
  end;

  // Fill initial symmetric matrix C
  // Example: C (3x3) = [ 2+0i  1+1i  3+0i ] (Upper triangle)
  //                    [ 1-1i  3+0i  2+1i ] (Lower triangle would be conj-transpose for Hermitian,
  //                                          but for SYRK it's just symmetric, so copy upper)
  //                    [ 3+0i  2-1i  1+0i ]
  // For SYMMETRIC (not Hermitian), C[i][j] = C[j][i] (no conjugate).
  h_C_initial[0 + 0 * ldc].x := 2.0; h_C_initial[0 + 0 * ldc].y := 0.0; // C[0][0]
  h_C_initial[1 + 1 * ldc].x := 3.0; h_C_initial[1 + 1 * ldc].y := 0.0; // C[1][1]
  h_C_initial[2 + 2 * ldc].x := 1.0; h_C_initial[2 + 2 * ldc].y := 0.0; // C[2][2]

  if uplo = rocblas_fill_upper then
  begin
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1]
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0; // C[0][2]
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0; // C[1][2]
    // Fill lower triangle based on upper (symmetric property - copy, no conjugate)
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := 1.0; // C[1][0] = C[0][1]
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0] = C[0][2]
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := 1.0;  // C[2][1] = C[1][2]
  end
  else // rocblas_fill_lower
  begin
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := 1.0; // C[1][0]
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0]
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := 1.0;  // C[2][1]
    // Fill upper triangle based on lower (symmetric property - copy, no conjugate)
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1] = C[1][0]
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0;  // C[0][2] = C[2][0]
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0;  // C[1][2] = C[2][1]
  end;

  // Copy initial C matrix for computation
  Move(h_C_initial[0], h_C_computed[0], SizeOf(rocblas_float_complex) * Length(h_C_initial));

  alpha.x := 1.0;
  alpha.y := 0.0;
  beta.x := 1.0;
  beta.y := 0.0;

  // 4. Allocate device memory
  d_A := nil;
  d_C := nil;
  if transA = rocblas_operation_none then
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_float_complex) * N * K))
  else
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_float_complex) * K * N));
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_float_complex) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  if transA = rocblas_operation_none then
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_float_complex) * N * K, hipMemcpyHostToDevice))
  else
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_float_complex) * K * N, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode (alpha, beta are complex scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_csyrk (11 parameters)
  blas_status := rocblas_csyrk(
    handle,
    uplo,
    transA,
    N, K,
    @alpha,   // alpha (complex, host pointer)
    d_A, lda, // A, lda
    @beta,    // beta (complex, host pointer)
    d_C, ldc  // C, ldc (C is symmetric matrix, output)
  );
  AssertEquals('rocblas_csyrk should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_csyrk', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check (check if the matrix differs)
  AssertTrue('Result matrix C should differ from initial matrix C',
    not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N));
  // A more thorough check would validate the mathematical result against a reference.

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_ZSYRK;
var
  N, K, lda, ldc: Integer;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  h_A, h_C_initial, h_C_computed: array of rocblas_double_complex;
  d_A, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_double_complex; // alpha and beta are complex scalars
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  N := 3; // C is N x N (symmetric)
  K := 2; // A is N x K (if no trans) or K x N (if trans)
  uplo := rocblas_fill_lower; // Update lower triangle of C
  transA := rocblas_operation_transpose; // Operation: C = alpha*A^T*A + beta*C

  // Calculate leading dimensions
  if transA = rocblas_operation_none then
    lda := N
  else
    lda := K;
  ldc := N; // C is N x N

  // 2. Set up test data sizes
  if transA = rocblas_operation_none then
    SetLength(h_A, lda * K) // A is N x K
  else
    SetLength(h_A, lda * N); // A is K x N
  SetLength(h_C_initial, ldc * N); // C is N x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data to zero
  for i := 0 to High(h_A) do
  begin
    h_A[i].x := 0.0;
    h_A[i].y := 0.0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 0.0;
    h_C_initial[i].y := 0.0;
  end;

  // Fill matrix A with test data
  // Example: A (2x3) = [ 1+1i  3+0i  5+0i ]
  //                   [ 2+0i  4+1i  6+0i ]
  if transA = rocblas_operation_transpose then
  begin
    h_A[0 + 0 * lda].x := 1.0; h_A[0 + 0 * lda].y := 1.0; // A[0][0]
    h_A[0 + 1 * lda].x := 3.0; h_A[0 + 1 * lda].y := 0.0; // A[0][1]
    h_A[0 + 2 * lda].x := 5.0; h_A[0 + 2 * lda].y := 0.0; // A[0][2]
    h_A[1 + 0 * lda].x := 2.0; h_A[1 + 0 * lda].y := 0.0; // A[1][0]
    h_A[1 + 1 * lda].x := 4.0; h_A[1 + 1 * lda].y := 1.0; // A[1][1]
    h_A[1 + 2 * lda].x := 6.0; h_A[1 + 2 * lda].y := 0.0; // A[1][2]
  end;

  // Fill initial symmetric matrix C
  // Example: C (3x3) = [ 2+0i  1+1i  3+0i ]
  //                    [ 1-1i  3+0i  2+1i ]
  //                    [ 3+0i  2-1i  1+0i ]
  h_C_initial[0 + 0 * ldc].x := 2.0; h_C_initial[0 + 0 * ldc].y := 0.0; // C[0][0]
  h_C_initial[1 + 1 * ldc].x := 3.0; h_C_initial[1 + 1 * ldc].y := 0.0; // C[1][1]
  h_C_initial[2 + 2 * ldc].x := 1.0; h_C_initial[2 + 2 * ldc].y := 0.0; // C[2][2]

  if uplo = rocblas_fill_upper then
  begin
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1]
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0; // C[0][2]
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0; // C[1][2]
    // Fill lower triangle based on upper (symmetric property - copy, no conjugate)
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := 1.0; // C[1][0] = C[0][1]
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0] = C[0][2]
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := 1.0;  // C[2][1] = C[1][2]
  end
  else // rocblas_fill_lower
  begin
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := 1.0; // C[1][0]
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0]
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := 1.0;  // C[2][1]
    // Fill upper triangle based on lower (symmetric property - copy, no conjugate)
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1] = C[1][0]
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0;  // C[0][2] = C[2][0]
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0;  // C[1][2] = C[2][1]
  end;

  // Copy initial C matrix for computation
  Move(h_C_initial[0], h_C_computed[0], SizeOf(rocblas_double_complex) * Length(h_C_initial));

  alpha.x := 0.5;
  alpha.y := 0.1;
  beta.x := 2.0;
  beta.y := -0.3;

  // 4. Allocate device memory
  d_A := nil;
  d_C := nil;
  if transA = rocblas_operation_none then
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_double_complex) * N * K))
  else
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_double_complex) * K * N));
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_double_complex) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  if transA = rocblas_operation_none then
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_double_complex) * N * K, hipMemcpyHostToDevice))
  else
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_double_complex) * K * N, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode (alpha, beta are complex scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_zsyrk (11 parameters)
  blas_status := rocblas_zsyrk(
    handle,
    uplo,
    transA,
    N, K,
    @alpha,   // alpha (complex, host pointer)
    d_A, lda, // A, lda
    @beta,    // beta (complex, host pointer)
    d_C, ldc  // C, ldc (C is symmetric matrix, output)
  );
  AssertEquals('rocblas_zsyrk should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_zsyrk', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check (check if the matrix differs)
  AssertTrue('Result matrix C should differ from initial matrix C',
    not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N));
  // A more thorough check would validate the mathematical result against a reference.

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_CHER2K;
var
  N, K, lda, ldb, ldc: Integer;
  uplo: rocblas_fill;
  trans: rocblas_operation;
  h_A, h_B, h_C_initial, h_C_computed: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha: rocblas_float_complex; // alpha is complex
  beta: Single;                 // beta is real for HER2K
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  N := 3; // C is N x N (Hermitian)
  K := 2; // A, B are N x K (if no trans) or K x N (if trans)
  uplo := rocblas_fill_upper; // Update upper triangle of C
  trans := rocblas_operation_none; // Operation: C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C

  // Calculate leading dimensions
  if trans = rocblas_operation_none then
  begin
    lda := N; // A is N x K
    ldb := N; // B is N x K
  end
  else
  begin
    lda := K; // A is K x N
    ldb := K; // B is K x N
  end;
  ldc := N; // C is N x N

  // 2. Set up test data sizes
  if trans = rocblas_operation_none then
  begin
    SetLength(h_A, lda * K); // A is N x K
    SetLength(h_B, ldb * K); // B is N x K
  end
  else
  begin
    SetLength(h_A, lda * N); // A is K x N
    SetLength(h_B, ldb * N); // B is K x N
  end;
  SetLength(h_C_initial, ldc * N); // C is N x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data to zero
  for i := 0 to High(h_A) do
  begin
    h_A[i].x := 0.0;
    h_A[i].y := 0.0;
  end;
  for i := 0 to High(h_B) do
  begin
    h_B[i].x := 0.0;
    h_B[i].y := 0.0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 0.0;
    h_C_initial[i].y := 0.0;
  end;

  // Fill matrix A with test data
  // Example: A (3x2) = [ 1+1i  2+0i ]
  //                   [ 3+0i  4+1i ]
  //                   [ 5+0i  6+0i ]
  if trans = rocblas_operation_none then
  begin
    h_A[0 + 0 * lda].x := 1.0; h_A[0 + 0 * lda].y := 1.0; // A[0][0]
    h_A[1 + 0 * lda].x := 3.0; h_A[1 + 0 * lda].y := 0.0; // A[1][0]
    h_A[2 + 0 * lda].x := 5.0; h_A[2 + 0 * lda].y := 0.0; // A[2][0]
    h_A[0 + 1 * lda].x := 2.0; h_A[0 + 1 * lda].y := 0.0; // A[0][1]
    h_A[1 + 1 * lda].x := 4.0; h_A[1 + 1 * lda].y := 1.0; // A[1][1]
    h_A[2 + 1 * lda].x := 6.0; h_A[2 + 1 * lda].y := 0.0; // A[2][1]
  end;

  // Fill matrix B with test data
  // Example: B (3x2) = [ 0+1i  1+0i ]
  //                   [ 2+0i  0+1i ]
  //                   [ 1+1i  2+0i ]
  if trans = rocblas_operation_none then
  begin
    h_B[0 + 0 * ldb].x := 0.0; h_B[0 + 0 * ldb].y := 1.0; // B[0][0]
    h_B[1 + 0 * ldb].x := 2.0; h_B[1 + 0 * ldb].y := 0.0; // B[1][0]
    h_B[2 + 0 * ldb].x := 1.0; h_B[2 + 0 * ldb].y := 1.0; // B[2][0]
    h_B[0 + 1 * ldb].x := 1.0; h_B[0 + 1 * ldb].y := 0.0; // B[0][1]
    h_B[1 + 1 * ldb].x := 0.0; h_B[1 + 1 * ldb].y := 1.0; // B[1][1]
    h_B[2 + 1 * ldb].x := 2.0; h_B[2 + 1 * ldb].y := 0.0; // B[2][1]
  end;

  // Fill initial Hermitian matrix C
  // Example: C (3x3) = [ 2+0i  1+1i  3+0i ] (Upper triangle)
  //                    [ 1-1i  3+0i  2+1i ] (Lower triangle = conj-transpose of upper)
  //                    [ 3+0i  2-1i  1+0i ]
  h_C_initial[0 + 0 * ldc].x := 2.0; h_C_initial[0 + 0 * ldc].y := 0.0; // C[0][0] (Real diagonal)
  h_C_initial[1 + 1 * ldc].x := 3.0; h_C_initial[1 + 1 * ldc].y := 0.0; // C[1][1] (Real diagonal)
  h_C_initial[2 + 2 * ldc].x := 1.0; h_C_initial[2 + 2 * ldc].y := 0.0; // C[2][2] (Real diagonal)

  if uplo = rocblas_fill_upper then
  begin
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1]
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0; // C[0][2]
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0; // C[1][2]
    // Fill lower triangle based on upper (Hermitian property)
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := -1.0; // C[1][0] = conj(C[0][1])
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0] = conj(C[0][2])
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := -1.0; // C[2][1] = conj(C[1][2])
  end
  else // rocblas_fill_lower
  begin
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := -1.0; // C[1][0]
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0]
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := -1.0; // C[2][1]
    // Fill upper triangle based on lower (Hermitian property)
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1] = conj(C[1][0])
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0;  // C[0][2] = conj(C[2][0])
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0;  // C[1][2] = conj(C[2][1])
  end;

  // Copy initial C matrix for computation
  Move(h_C_initial[0], h_C_computed[0], SizeOf(rocblas_float_complex) * Length(h_C_initial));

  alpha.x := 1.0;
  alpha.y := 0.5;
  beta := 1.0; // Real beta

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  d_C := nil;
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_float_complex) * N * K));
    AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_float_complex) * N * K));
  end
  else
  begin
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_float_complex) * K * N));
    AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_float_complex) * K * N));
  end;
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_float_complex) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_float_complex) * N * K, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_float_complex) * N * K, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_float_complex) * K * N, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_float_complex) * K * N, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  // alpha is complex (device or host), beta is real (device or host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_cher2k (13 parameters)
  blas_status := rocblas_cher2k(
    handle,
    uplo,
    trans,
    N, K,
    @alpha,   // alpha (complex, host pointer)
    d_A, lda, // A, lda
    d_B, ldb, // B, ldb
    @beta,    // beta (real, host pointer)
    d_C, ldc  // C, ldc (C is Hermitian matrix, output)
  );
  AssertEquals('rocblas_cher2k should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_cher2k', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check (check if the matrix differs)
  // For Hermitian matrices, only the updated triangle and diagonal should be checked properly.
  // This simple check just sees if memory changed.
  AssertTrue('Result matrix C should differ from initial matrix C',
    not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N));
  // A more thorough check would validate the mathematical result against a reference.

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_ZHER2K;
var
  N, K, lda, ldb, ldc: Integer;
  uplo: rocblas_fill;
  trans: rocblas_operation;
  h_A, h_B, h_C_initial, h_C_computed: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha: rocblas_double_complex; // alpha is complex
  beta: Double;                  // beta is real for HER2K
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  N := 3; // C is N x N (Hermitian)
  K := 2; // A, B are N x K (if no trans) or K x N (if trans)
  uplo := rocblas_fill_lower; // Update lower triangle of C
  trans := rocblas_operation_conjugate_transpose; // Operation: C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C

  // Calculate leading dimensions
  if trans = rocblas_operation_none then
  begin
    lda := N;
    ldb := N;
  end
  else
  begin
    lda := K; // A is K x N
    ldb := K; // B is K x N
  end;
  ldc := N; // C is N x N

  // 2. Set up test data sizes
  if trans = rocblas_operation_none then
  begin
    SetLength(h_A, lda * K); // A is N x K
    SetLength(h_B, ldb * K); // B is N x K
  end
  else
  begin
    SetLength(h_A, lda * N); // A is K x N
    SetLength(h_B, ldb * N); // B is K x N
  end;
  SetLength(h_C_initial, ldc * N); // C is N x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data to zero
  for i := 0 to High(h_A) do
  begin
    h_A[i].x := 0.0;
    h_A[i].y := 0.0;
  end;
  for i := 0 to High(h_B) do
  begin
    h_B[i].x := 0.0;
    h_B[i].y := 0.0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 0.0;
    h_C_initial[i].y := 0.0;
  end;

  // Fill matrix A with test data
  // Example: A (2x3) = [ 1+1i  3+0i  5+0i ]
  //                   [ 2+0i  4+1i  6+0i ]
  if trans = rocblas_operation_conjugate_transpose then
  begin
    h_A[0 + 0 * lda].x := 1.0; h_A[0 + 0 * lda].y := 1.0; // A[0][0]
    h_A[0 + 1 * lda].x := 3.0; h_A[0 + 1 * lda].y := 0.0; // A[0][1]
    h_A[0 + 2 * lda].x := 5.0; h_A[0 + 2 * lda].y := 0.0; // A[0][2]
    h_A[1 + 0 * lda].x := 2.0; h_A[1 + 0 * lda].y := 0.0; // A[1][0]
    h_A[1 + 1 * lda].x := 4.0; h_A[1 + 1 * lda].y := 1.0; // A[1][1]
    h_A[1 + 2 * lda].x := 6.0; h_A[1 + 2 * lda].y := 0.0; // A[1][2]
  end;

  // Fill matrix B with test data
  // Example: B (2x3) = [ 0+1i  2+0i  1+1i ]
  //                   [ 1+0i  0+1i  2+0i ]
  if trans = rocblas_operation_conjugate_transpose then
  begin
    h_B[0 + 0 * ldb].x := 0.0; h_B[0 + 0 * ldb].y := 1.0; // B[0][0]
    h_B[0 + 1 * ldb].x := 2.0; h_B[0 + 1 * ldb].y := 0.0; // B[0][1]
    h_B[0 + 2 * ldb].x := 1.0; h_B[0 + 2 * ldb].y := 1.0; // B[0][2]
    h_B[1 + 0 * ldb].x := 1.0; h_B[1 + 0 * ldb].y := 0.0; // B[1][0]
    h_B[1 + 1 * ldb].x := 0.0; h_B[1 + 1 * ldb].y := 1.0; // B[1][1]
    h_B[1 + 2 * ldb].x := 2.0; h_B[1 + 2 * ldb].y := 0.0; // B[1][2]
  end;

  // Fill initial Hermitian matrix C
  // Example: C (3x3) = [ 2+0i  1+1i  3+0i ]
  //                    [ 1-1i  3+0i  2+1i ]
  //                    [ 3+0i  2-1i  1+0i ]
  h_C_initial[0 + 0 * ldc].x := 2.0; h_C_initial[0 + 0 * ldc].y := 0.0; // C[0][0] (Real diagonal)
  h_C_initial[1 + 1 * ldc].x := 3.0; h_C_initial[1 + 1 * ldc].y := 0.0; // C[1][1] (Real diagonal)
  h_C_initial[2 + 2 * ldc].x := 1.0; h_C_initial[2 + 2 * ldc].y := 0.0; // C[2][2] (Real diagonal)

  if uplo = rocblas_fill_upper then
  begin
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1]
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0; // C[0][2]
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0; // C[1][2]
    // Fill lower triangle based on upper (Hermitian property)
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := -1.0; // C[1][0] = conj(C[0][1])
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0] = conj(C[0][2])
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := -1.0; // C[2][1] = conj(C[1][2])
  end
  else // rocblas_fill_lower
  begin
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := -1.0; // C[1][0]
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0]
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := -1.0; // C[2][1]
    // Fill upper triangle based on lower (Hermitian property)
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1] = conj(C[1][0])
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0;  // C[0][2] = conj(C[2][0])
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0;  // C[1][2] = conj(C[2][1])
  end;

  // Copy initial C matrix for computation
  Move(h_C_initial[0], h_C_computed[0], SizeOf(rocblas_double_complex) * Length(h_C_initial));

  alpha.x := 0.5;
  alpha.y := -0.2;
  beta := 2.0; // Real beta

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  d_C := nil;
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_double_complex) * N * K));
    AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_double_complex) * N * K));
  end
  else
  begin
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_double_complex) * K * N));
    AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_double_complex) * K * N));
  end;
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_double_complex) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_double_complex) * N * K, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_double_complex) * N * K, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_double_complex) * K * N, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_double_complex) * K * N, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode
  // alpha is complex (device or host), beta is real (device or host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_zher2k (13 parameters)
  blas_status := rocblas_zher2k(
    handle,
    uplo,
    trans,
    N, K,
    @alpha,   // alpha (complex, host pointer)
    d_A, lda, // A, lda
    d_B, ldb, // B, ldb
    @beta,    // beta (real, host pointer)
    d_C, ldc  // C, ldc (C is Hermitian matrix, output)
  );
  AssertEquals('rocblas_zher2k should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_zher2k', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check (check if the matrix differs)
  AssertTrue('Result matrix C should differ from initial matrix C',
    not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N));
  // A more thorough check would validate the mathematical result against a reference.

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;

procedure TTestRocBLAS_Level3.Test_SSyr2k;
var
  N, K, lda, ldb, ldc: Integer;
  uplo: rocblas_fill;
  trans: rocblas_operation;
  h_A, h_B, h_C_initial, h_C_computed: array of Single; // Real Single Precision
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: Single; // alpha and beta are real scalars
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  N := 3; // C is N x N (symmetric)
  K := 2; // A, B are N x K (if no trans) or K x N (if trans)
  uplo := rocblas_fill_upper; // Update upper triangle of C
  trans := rocblas_operation_none; // Operation: C = alpha*A*B^T + alpha*B*A^T + beta*C

  // Calculate leading dimensions
  if trans = rocblas_operation_none then
  begin
    lda := N; // A is N x K
    ldb := N; // B is N x K
  end
  else
  begin
    lda := K; // A is K x N
    ldb := K; // B is K x N
  end;
  ldc := N; // C is N x N

  // 2. Set up test data sizes
  if trans = rocblas_operation_none then
  begin
    SetLength(h_A, lda * K); // A is N x K
    SetLength(h_B, ldb * K); // B is N x K
  end
  else
  begin
    SetLength(h_A, lda * N); // A is K x N
    SetLength(h_B, ldb * N); // B is K x N
  end;
  SetLength(h_C_initial, ldc * N); // C is N x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data to zero
  for i := 0 to High(h_A) do
    h_A[i] := 0.0;
  for i := 0 to High(h_B) do
    h_B[i] := 0.0;
  for i := 0 to High(h_C_initial) do
    h_C_initial[i] := 0.0;

  // Fill matrix A with test data
  // Example: A (3x2) = [ 1.0  2.0 ]
  //                   [ 3.0  4.0 ]
  //                   [ 5.0  6.0 ]
  if trans = rocblas_operation_none then
  begin
    h_A[0 + 0 * lda] := 1.0; // A[0][0]
    h_A[1 + 0 * lda] := 3.0; // A[1][0]
    h_A[2 + 0 * lda] := 5.0; // A[2][0]
    h_A[0 + 1 * lda] := 2.0; // A[0][1]
    h_A[1 + 1 * lda] := 4.0; // A[1][1]
    h_A[2 + 1 * lda] := 6.0; // A[2][1]
  end;

  // Fill matrix B with test data
  // Example: B (3x2) = [ 0.5  1.5 ]
  //                   [ 2.5  0.5 ]
  //                   [ 1.0  2.0 ]
  if trans = rocblas_operation_none then
  begin
    h_B[0 + 0 * ldb] := 0.5; // B[0][0]
    h_B[1 + 0 * ldb] := 2.5; // B[1][0]
    h_B[2 + 0 * ldb] := 1.0; // B[2][0]
    h_B[0 + 1 * ldb] := 1.5; // B[0][1]
    h_B[1 + 1 * ldb] := 0.5; // B[1][1]
    h_B[2 + 1 * ldb] := 2.0; // B[2][1]
  end;

  // Fill initial symmetric matrix C
  // Example: C (3x3) = [ 2.0  1.0  3.0 ] (Upper triangle)
  //                    [ 1.0  3.0  2.0 ]
  //                    [ 3.0  2.0  1.0 ] (Lower triangle = transpose of upper for symmetric)
  h_C_initial[0 + 0 * ldc] := 2.0; // C[0][0]
  h_C_initial[1 + 1 * ldc] := 3.0; // C[1][1]
  h_C_initial[2 + 2 * ldc] := 1.0; // C[2][2]

  if uplo = rocblas_fill_upper then
  begin
    h_C_initial[0 + 1 * ldc] := 1.0; // C[0][1]
    h_C_initial[0 + 2 * ldc] := 3.0; // C[0][2]
    h_C_initial[1 + 2 * ldc] := 2.0; // C[1][2]
    // Fill lower triangle based on upper (symmetric property)
    h_C_initial[1 + 0 * ldc] := 1.0; // C[1][0] = C[0][1]
    h_C_initial[2 + 0 * ldc] := 3.0; // C[2][0] = C[0][2]
    h_C_initial[2 + 1 * ldc] := 2.0; // C[2][1] = C[1][2]
  end
  else // rocblas_fill_lower
  begin
    h_C_initial[1 + 0 * ldc] := 1.0; // C[1][0]
    h_C_initial[2 + 0 * ldc] := 3.0; // C[2][0]
    h_C_initial[2 + 1 * ldc] := 2.0; // C[2][1]
    // Fill upper triangle based on lower (symmetric property)
    h_C_initial[0 + 1 * ldc] := 1.0; // C[0][1] = C[1][0]
    h_C_initial[0 + 2 * ldc] := 3.0; // C[0][2] = C[2][0]
    h_C_initial[1 + 2 * ldc] := 2.0; // C[1][2] = C[2][1]
  end;

  // Copy initial C matrix for computation
  Move(h_C_initial[0], h_C_computed[0], SizeOf(Single) * Length(h_C_initial));

  alpha := 1.0;
  beta := 1.0;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  d_C := nil;
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Single) * N * K));
    AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(Single) * N * K));
  end
  else
  begin
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Single) * K * N));
    AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(Single) * K * N));
  end;
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(Single) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Single) * N * K, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(Single) * N * K, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Single) * K * N, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(Single) * K * N, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(Single) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode (alpha, beta are real scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_ssyr2k (13 parameters)
  blas_status := rocblas_ssyr2k(
    handle,
    uplo,
    trans,
    N, K,
    @alpha,   // alpha (real, host pointer)
    d_A, lda, // A, lda
    d_B, ldb, // B, ldb
    @beta,    // beta (real, host pointer)
    d_C, ldc  // C, ldc (C is symmetric matrix, output)
  );
  AssertEquals('rocblas_ssyr2k should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ssyr2k', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(Single) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check (check if the matrix differs)
  AssertTrue('Result matrix C should differ from initial matrix C',
    not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(Single) * ldc * N));
  // A more thorough check would validate the mathematical result against a reference.

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_DSyr2k;
var
  N, K, lda, ldb, ldc: Integer;
  uplo: rocblas_fill;
  trans: rocblas_operation;
  h_A, h_B, h_C_initial, h_C_computed: array of Double; // Real Double Precision
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: Double; // alpha and beta are real scalars
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  N := 3; // C is N x N (symmetric)
  K := 2; // A, B are N x K (if no trans) or K x N (if trans)
  uplo := rocblas_fill_lower; // Update lower triangle of C
  trans := rocblas_operation_transpose; // Operation: C = alpha*A^T*B + alpha*B^T*A + beta*C

  // Calculate leading dimensions
  if trans = rocblas_operation_none then
  begin
    lda := N;
    ldb := N;
  end
  else
  begin
    lda := K; // A is K x N
    ldb := K; // B is K x N
  end;
  ldc := N; // C is N x N

  // 2. Set up test data sizes
  if trans = rocblas_operation_none then
  begin
    SetLength(h_A, lda * K); // A is N x K
    SetLength(h_B, ldb * K); // B is N x K
  end
  else
  begin
    SetLength(h_A, lda * N); // A is K x N
    SetLength(h_B, ldb * N); // B is K x N
  end;
  SetLength(h_C_initial, ldc * N); // C is N x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data to zero
  for i := 0 to High(h_A) do
    h_A[i] := 0.0;
  for i := 0 to High(h_B) do
    h_B[i] := 0.0;
  for i := 0 to High(h_C_initial) do
    h_C_initial[i] := 0.0;

  // Fill matrix A with test data
  // Example: A (2x3) = [ 1.0  3.0  5.0 ]
  //                   [ 2.0  4.0  6.0 ]
  if trans = rocblas_operation_transpose then
  begin
    h_A[0 + 0 * lda] := 1.0; // A[0][0]
    h_A[0 + 1 * lda] := 3.0; // A[0][1]
    h_A[0 + 2 * lda] := 5.0; // A[0][2]
    h_A[1 + 0 * lda] := 2.0; // A[1][0]
    h_A[1 + 1 * lda] := 4.0; // A[1][1]
    h_A[1 + 2 * lda] := 6.0; // A[1][2]
  end;

  // Fill matrix B with test data
  // Example: B (2x3) = [ 0.5  2.5  1.0 ]
  //                   [ 1.5  0.5  2.0 ]
  if trans = rocblas_operation_transpose then
  begin
    h_B[0 + 0 * ldb] := 0.5; // B[0][0]
    h_B[0 + 1 * ldb] := 2.5; // B[0][1]
    h_B[0 + 2 * ldb] := 1.0; // B[0][2]
    h_B[1 + 0 * ldb] := 1.5; // B[1][0]
    h_B[1 + 1 * ldb] := 0.5; // B[1][1]
    h_B[1 + 2 * ldb] := 2.0; // B[1][2]
  end;

  // Fill initial symmetric matrix C
  // Example: C (3x3) = [ 2.0  1.0  3.0 ]
  //                    [ 1.0  3.0  2.0 ]
  //                    [ 3.0  2.0  1.0 ]
  h_C_initial[0 + 0 * ldc] := 2.0; // C[0][0]
  h_C_initial[1 + 1 * ldc] := 3.0; // C[1][1]
  h_C_initial[2 + 2 * ldc] := 1.0; // C[2][2]

  if uplo = rocblas_fill_upper then
  begin
    h_C_initial[0 + 1 * ldc] := 1.0; // C[0][1]
    h_C_initial[0 + 2 * ldc] := 3.0; // C[0][2]
    h_C_initial[1 + 2 * ldc] := 2.0; // C[1][2]
    // Fill lower triangle based on upper (symmetric property)
    h_C_initial[1 + 0 * ldc] := 1.0; // C[1][0] = C[0][1]
    h_C_initial[2 + 0 * ldc] := 3.0; // C[2][0] = C[0][2]
    h_C_initial[2 + 1 * ldc] := 2.0; // C[2][1] = C[1][2]
  end
  else // rocblas_fill_lower
  begin
    h_C_initial[1 + 0 * ldc] := 1.0; // C[1][0]
    h_C_initial[2 + 0 * ldc] := 3.0; // C[2][0]
    h_C_initial[2 + 1 * ldc] := 2.0; // C[2][1]
    // Fill upper triangle based on lower (symmetric property)
    h_C_initial[0 + 1 * ldc] := 1.0; // C[0][1] = C[1][0]
    h_C_initial[0 + 2 * ldc] := 3.0; // C[0][2] = C[2][0]
    h_C_initial[1 + 2 * ldc] := 2.0; // C[1][2] = C[2][1]
  end;

  // Copy initial C matrix for computation
  Move(h_C_initial[0], h_C_computed[0], SizeOf(Double) * Length(h_C_initial));

  alpha := 0.5;
  beta := 2.0;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  d_C := nil;
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Double) * N * K));
    AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(Double) * N * K));
  end
  else
  begin
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(Double) * K * N));
    AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(Double) * K * N));
  end;
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(Double) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Double) * N * K, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(Double) * N * K, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(Double) * K * N, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(Double) * K * N, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(Double) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode (alpha, beta are real scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_dsyr2k (13 parameters)
  blas_status := rocblas_dsyr2k(
    handle,
    uplo,
    trans,
    N, K,
    @alpha,   // alpha (real, host pointer)
    d_A, lda, // A, lda
    d_B, ldb, // B, ldb
    @beta,    // beta (real, host pointer)
    d_C, ldc  // C, ldc (C is symmetric matrix, output)
  );
  AssertEquals('rocblas_dsyr2k should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_dsyr2k', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(Double) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check (check if the matrix differs)
  AssertTrue('Result matrix C should differ from initial matrix C',
    not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(Double) * ldc * N));
  // A more thorough check would validate the mathematical result against a reference.

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_CSyr2k;
var
  N, K, lda, ldb, ldc: Integer;
  uplo: rocblas_fill;
  trans: rocblas_operation;
  h_A, h_B, h_C_initial, h_C_computed: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_float_complex; // alpha and beta are complex scalars
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  N := 3; // C is N x N (symmetric)
  K := 2; // A, B are N x K (if no trans) or K x N (if trans)
  uplo := rocblas_fill_upper; // Update upper triangle of C
  trans := rocblas_operation_none; // Operation: C = alpha*A*B^T + alpha*B*A^T + beta*C

  // Calculate leading dimensions
  if trans = rocblas_operation_none then
  begin
    lda := N; // A is N x K
    ldb := N; // B is N x K
  end
  else
  begin
    lda := K; // A is K x N
    ldb := K; // B is K x N
  end;
  ldc := N; // C is N x N

  // 2. Set up test data sizes
  if trans = rocblas_operation_none then
  begin
    SetLength(h_A, lda * K); // A is N x K
    SetLength(h_B, ldb * K); // B is N x K
  end
  else
  begin
    SetLength(h_A, lda * N); // A is K x N
    SetLength(h_B, ldb * N); // B is K x N
  end;
  SetLength(h_C_initial, ldc * N); // C is N x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data to zero
  for i := 0 to High(h_A) do
  begin
    h_A[i].x := 0.0;
    h_A[i].y := 0.0;
  end;
  for i := 0 to High(h_B) do
  begin
    h_B[i].x := 0.0;
    h_B[i].y := 0.0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 0.0;
    h_C_initial[i].y := 0.0;
  end;

  // Fill matrix A with test data
  // Example: A (3x2) = [ 1+1i  2+0i ]
  //                   [ 3+0i  4+1i ]
  //                   [ 5+0i  6+0i ]
  if trans = rocblas_operation_none then
  begin
    h_A[0 + 0 * lda].x := 1.0; h_A[0 + 0 * lda].y := 1.0; // A[0][0]
    h_A[1 + 0 * lda].x := 3.0; h_A[1 + 0 * lda].y := 0.0; // A[1][0]
    h_A[2 + 0 * lda].x := 5.0; h_A[2 + 0 * lda].y := 0.0; // A[2][0]
    h_A[0 + 1 * lda].x := 2.0; h_A[0 + 1 * lda].y := 0.0; // A[0][1]
    h_A[1 + 1 * lda].x := 4.0; h_A[1 + 1 * lda].y := 1.0; // A[1][1]
    h_A[2 + 1 * lda].x := 6.0; h_A[2 + 1 * lda].y := 0.0; // A[2][1]
  end;

  // Fill matrix B with test data
  // Example: B (3x2) = [ 0+1i  1+0i ]
  //                   [ 2+0i  0+1i ]
  //                   [ 1+1i  2+0i ]
  if trans = rocblas_operation_none then
  begin
    h_B[0 + 0 * ldb].x := 0.0; h_B[0 + 0 * ldb].y := 1.0; // B[0][0]
    h_B[1 + 0 * ldb].x := 2.0; h_B[1 + 0 * ldb].y := 0.0; // B[1][0]
    h_B[2 + 0 * ldb].x := 1.0; h_B[2 + 0 * ldb].y := 1.0; // B[2][0]
    h_B[0 + 1 * ldb].x := 1.0; h_B[0 + 1 * ldb].y := 0.0; // B[0][1]
    h_B[1 + 1 * ldb].x := 0.0; h_B[1 + 1 * ldb].y := 1.0; // B[1][1]
    h_B[2 + 1 * ldb].x := 2.0; h_B[2 + 1 * ldb].y := 0.0; // B[2][1]
  end;

  // Fill initial symmetric matrix C
  // Example: C (3x3) = [ 2+0i  1+1i  3+0i ] (Upper triangle)
  //                    [ 1-1i  3+0i  2+1i ] (Lower triangle would be conj-transpose for Hermitian,
  //                                          but for SYR2K it's just symmetric, so copy upper)
  //                    [ 3+0i  2-1i  1+0i ]
  // For SYMMETRIC (not Hermitian), C[i][j] = C[j][i] (no conjugate).
  h_C_initial[0 + 0 * ldc].x := 2.0; h_C_initial[0 + 0 * ldc].y := 0.0; // C[0][0]
  h_C_initial[1 + 1 * ldc].x := 3.0; h_C_initial[1 + 1 * ldc].y := 0.0; // C[1][1]
  h_C_initial[2 + 2 * ldc].x := 1.0; h_C_initial[2 + 2 * ldc].y := 0.0; // C[2][2]

  if uplo = rocblas_fill_upper then
  begin
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1]
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0; // C[0][2]
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0; // C[1][2]
    // Fill lower triangle based on upper (symmetric property - copy, no conjugate)
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := 1.0; // C[1][0] = C[0][1]
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0] = C[0][2]
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := 1.0;  // C[2][1] = C[1][2]
  end
  else // rocblas_fill_lower
  begin
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := 1.0; // C[1][0]
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0]
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := 1.0;  // C[2][1]
    // Fill upper triangle based on lower (symmetric property - copy, no conjugate)
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1] = C[1][0]
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0;  // C[0][2] = C[2][0]
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0;  // C[1][2] = C[2][1]
  end;

  // Copy initial C matrix for computation
  Move(h_C_initial[0], h_C_computed[0], SizeOf(rocblas_float_complex) * Length(h_C_initial));

  alpha.x := 1.0;
  alpha.y := 0.5;
  beta.x := 1.0;
  beta.y := 0.0;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  d_C := nil;
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_float_complex) * N * K));
    AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_float_complex) * N * K));
  end
  else
  begin
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_float_complex) * K * N));
    AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_float_complex) * K * N));
  end;
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_float_complex) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_float_complex) * N * K, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_float_complex) * N * K, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_float_complex) * K * N, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_float_complex) * K * N, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode (alpha, beta are complex scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_csyr2k (13 parameters)
  blas_status := rocblas_csyr2k(
    handle,
    uplo,
    trans,
    N, K,
    @alpha,   // alpha (complex, host pointer)
    d_A, lda, // A, lda
    d_B, ldb, // B, ldb
    @beta,    // beta (complex, host pointer)
    d_C, ldc  // C, ldc (C is symmetric matrix, output)
  );
  AssertEquals('rocblas_csyr2k should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_csyr2k', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_float_complex) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check (check if the matrix differs)
  AssertTrue('Result matrix C should differ from initial matrix C',
    not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_float_complex) * ldc * N));
  // A more thorough check would validate the mathematical result against a reference.

  // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;


procedure TTestRocBLAS_Level3.Test_ZSyr2k;
var
  N, K, lda, ldb, ldc: Integer;
  uplo: rocblas_fill;
  trans: rocblas_operation;
  h_A, h_B, h_C_initial, h_C_computed: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_double_complex; // alpha and beta are complex scalars
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  i, j: Integer;
begin
  // 1. Initialize matrix dimensions and parameters
  N := 3; // C is N x N (symmetric)
  K := 2; // A, B are N x K (if no trans) or K x N (if trans)
  uplo := rocblas_fill_lower; // Update lower triangle of C
  trans := rocblas_operation_transpose; // Operation: C = alpha*A^T*B + alpha*B^T*A + beta*C

  // Calculate leading dimensions
  if trans = rocblas_operation_none then
  begin
    lda := N;
    ldb := N;
  end
  else
  begin
    lda := K; // A is K x N
    ldb := K; // B is K x N
  end;
  ldc := N; // C is N x N

  // 2. Set up test data sizes
  if trans = rocblas_operation_none then
  begin
    SetLength(h_A, lda * K); // A is N x K
    SetLength(h_B, ldb * K); // B is N x K
  end
  else
  begin
    SetLength(h_A, lda * N); // A is K x N
    SetLength(h_B, ldb * N); // B is K x N
  end;
  SetLength(h_C_initial, ldc * N); // C is N x N
  SetLength(h_C_computed, ldc * N);

  // 3. Initialize host data to zero
  for i := 0 to High(h_A) do
  begin
    h_A[i].x := 0.0;
    h_A[i].y := 0.0;
  end;
  for i := 0 to High(h_B) do
  begin
    h_B[i].x := 0.0;
    h_B[i].y := 0.0;
  end;
  for i := 0 to High(h_C_initial) do
  begin
    h_C_initial[i].x := 0.0;
    h_C_initial[i].y := 0.0;
  end;

  // Fill matrix A with test data
  // Example: A (2x3) = [ 1+1i  3+0i  5+0i ]
  //                   [ 2+0i  4+1i  6+0i ]
  if trans = rocblas_operation_transpose then
  begin
    h_A[0 + 0 * lda].x := 1.0; h_A[0 + 0 * lda].y := 1.0; // A[0][0]
    h_A[0 + 1 * lda].x := 3.0; h_A[0 + 1 * lda].y := 0.0; // A[0][1]
    h_A[0 + 2 * lda].x := 5.0; h_A[0 + 2 * lda].y := 0.0; // A[0][2]
    h_A[1 + 0 * lda].x := 2.0; h_A[1 + 0 * lda].y := 0.0; // A[1][0]
    h_A[1 + 1 * lda].x := 4.0; h_A[1 + 1 * lda].y := 1.0; // A[1][1]
    h_A[1 + 2 * lda].x := 6.0; h_A[1 + 2 * lda].y := 0.0; // A[1][2]
  end;

  // Fill matrix B with test data
  // Example: B (2x3) = [ 0+1i  2+0i  1+1i ]
  //                   [ 1+0i  0+1i  2+0i ]
  if trans = rocblas_operation_transpose then
  begin
    h_B[0 + 0 * ldb].x := 0.0; h_B[0 + 0 * ldb].y := 1.0; // B[0][0]
    h_B[0 + 1 * ldb].x := 2.0; h_B[0 + 1 * ldb].y := 0.0; // B[0][1]
    h_B[0 + 2 * ldb].x := 1.0; h_B[0 + 2 * ldb].y := 1.0; // B[0][2]
    h_B[1 + 0 * ldb].x := 1.0; h_B[1 + 0 * ldb].y := 0.0; // B[1][0]
    h_B[1 + 1 * ldb].x := 0.0; h_B[1 + 1 * ldb].y := 1.0; // B[1][1]
    h_B[1 + 2 * ldb].x := 2.0; h_B[1 + 2 * ldb].y := 0.0; // B[1][2]
  end;

  // Fill initial symmetric matrix C
  // Example: C (3x3) = [ 2+0i  1+1i  3+0i ]
  //                    [ 1-1i  3+0i  2+1i ]
  //                    [ 3+0i  2-1i  1+0i ]
  h_C_initial[0 + 0 * ldc].x := 2.0; h_C_initial[0 + 0 * ldc].y := 0.0; // C[0][0]
  h_C_initial[1 + 1 * ldc].x := 3.0; h_C_initial[1 + 1 * ldc].y := 0.0; // C[1][1]
  h_C_initial[2 + 2 * ldc].x := 1.0; h_C_initial[2 + 2 * ldc].y := 0.0; // C[2][2]

  if uplo = rocblas_fill_upper then
  begin
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1]
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0; // C[0][2]
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0; // C[1][2]
    // Fill lower triangle based on upper (symmetric property - copy, no conjugate)
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := 1.0; // C[1][0] = C[0][1]
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0] = C[0][2]
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := 1.0;  // C[2][1] = C[1][2]
  end
  else // rocblas_fill_lower
  begin
    h_C_initial[1 + 0 * ldc].x := 1.0; h_C_initial[1 + 0 * ldc].y := 1.0; // C[1][0]
    h_C_initial[2 + 0 * ldc].x := 3.0; h_C_initial[2 + 0 * ldc].y := 0.0;  // C[2][0]
    h_C_initial[2 + 1 * ldc].x := 2.0; h_C_initial[2 + 1 * ldc].y := 1.0;  // C[2][1]
    // Fill upper triangle based on lower (symmetric property - copy, no conjugate)
    h_C_initial[0 + 1 * ldc].x := 1.0; h_C_initial[0 + 1 * ldc].y := 1.0; // C[0][1] = C[1][0]
    h_C_initial[0 + 2 * ldc].x := 3.0; h_C_initial[0 + 2 * ldc].y := 0.0;  // C[0][2] = C[2][0]
    h_C_initial[1 + 2 * ldc].x := 2.0; h_C_initial[1 + 2 * ldc].y := 1.0;  // C[1][2] = C[2][1]
  end;

  // Copy initial C matrix for computation
  Move(h_C_initial[0], h_C_computed[0], SizeOf(rocblas_double_complex) * Length(h_C_initial));

  alpha.x := 0.5;
  alpha.y := -0.2;
  beta.x := 2.0;
  beta.y := 0.3;

  // 4. Allocate device memory
  d_A := nil;
  d_B := nil;
  d_C := nil;
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_double_complex) * N * K));
    AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_double_complex) * N * K));
  end
  else
  begin
    AssertEquals('hipMalloc d_A', HIP_SUCCESS, hipMalloc(d_A, SizeOf(rocblas_double_complex) * K * N));
    AssertEquals('hipMalloc d_B', HIP_SUCCESS, hipMalloc(d_B, SizeOf(rocblas_double_complex) * K * N));
  end;
  AssertEquals('hipMalloc d_C', HIP_SUCCESS, hipMalloc(d_C, SizeOf(rocblas_double_complex) * ldc * N));
  AssertTrue('Device pointer d_A should not be nil', d_A <> nil);
  AssertTrue('Device pointer d_B should not be nil', d_B <> nil);
  AssertTrue('Device pointer d_C should not be nil', d_C <> nil);

  // 5. Copy data H2D
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_double_complex) * N * K, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_double_complex) * N * K, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy d_A H2D', HIP_SUCCESS, hipMemcpy(d_A, @h_A[0], SizeOf(rocblas_double_complex) * K * N, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy d_B H2D', HIP_SUCCESS, hipMemcpy(d_B, @h_B[0], SizeOf(rocblas_double_complex) * K * N, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy d_C H2D', HIP_SUCCESS, hipMemcpy(d_C, @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyHostToDevice));

  // 6. Set rocBLAS pointer mode (alpha, beta are complex scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', rocblas_status_success, pm_status);

  // 7. Call rocblas_zsyr2k (13 parameters)
  blas_status := rocblas_zsyr2k(
    handle,
    uplo,
    trans,
    N, K,
    @alpha,   // alpha (complex, host pointer)
    d_A, lda, // A, lda
    d_B, ldb, // B, ldb
    @beta,    // beta (complex, host pointer)
    d_C, ldc  // C, ldc (C is symmetric matrix, output)
  );
  AssertEquals('rocblas_zsyr2k should return success', rocblas_status_success, blas_status);

  // 8. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_zsyr2k', HIP_SUCCESS, sync_status);

  // 9. Copy result D2H
  AssertEquals('hipMemcpy d_C D2H', HIP_SUCCESS, hipMemcpy(@h_C_computed[0], d_C, SizeOf(rocblas_double_complex) * ldc * N, hipMemcpyDeviceToHost));

  // 10. Basic check (check if the matrix differs)
  AssertTrue('Result matrix C should differ from initial matrix C',
    not CompareMem(@h_C_initial[0], @h_C_computed[0], SizeOf(rocblas_double_complex) * ldc * N));
  // A more thorough check would validate the mathematical result against a reference.

    // 11. Free device memory
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
end;

procedure TTestRocBLAS_Level3.Test_GEMM_EX_FP16;
{ Reproduces the FP16 gemm_ex bug: missing D/d_type/ldd params shifted
  all args from C onward, returning 0.0889 instead of the correct 0.5.
  Config: A/B FP16, C/D FP16, compute FP32 accumulation (rocBLAS recommended). }
const
  M = 2; N = 2; K = 2;
  alpha_f32: Single = 1.0;
  beta_f32: Single = 0.0;
var
  devA, devB, devC: hipDeviceptr_t;
  hostA: array[0..3] of rocblas_half;     { [[1,2],[3,4]] }
  hostB: array[0..3] of rocblas_half;     { diag(0.5,0.5) }
  hostD: array[0..3] of rocblas_half;     { output }
  i: Integer;
  blas_status: rocblas_status;
  expected: array[0..3] of Single;
  actual: array[0..3] of Single;
begin
  { Build FP16 inputs: A=[[1,2],[3,4]], B=diag(0.5,0.5) }
  hostA[0] := SingleToHalf(1.0); hostA[1] := SingleToHalf(2.0);
  hostA[2] := SingleToHalf(3.0); hostA[3] := SingleToHalf(4.0);
  hostB[0] := SingleToHalf(0.5); hostB[1] := SingleToHalf(0.0);
  hostB[2] := SingleToHalf(0.0); hostB[3] := SingleToHalf(0.5);

  { Expected: A*B = [[0.5,1],[1.5,2]] }
  expected[0] := 0.5; expected[1] := 1.0;
  expected[2] := 1.5; expected[3] := 2.0;

  hipMalloc(devA, M*K*SizeOf(rocblas_half));
  hipMalloc(devB, K*N*SizeOf(rocblas_half));
  hipMalloc(devC, M*N*SizeOf(rocblas_half));

  hipMemcpy(devA, @hostA[0], M*K*SizeOf(rocblas_half), hipMemcpyHostToDevice);
  hipMemcpy(devB, @hostB[0], K*N*SizeOf(rocblas_half), hipMemcpyHostToDevice);

  { beta=0, so C contents are irrelevant; D aliases devC }
  blas_status := rocblas_gemm_ex(handle,
    rocblas_operation_none, rocblas_operation_none,  { transA, transB }
    M, N, K,                                         { m, n, k }
    @alpha_f32,                                      { alpha (FP32 pointer) }
    devA, rocblas_datatype_f16_r, M,                 { A, a_type, lda }
    devB, rocblas_datatype_f16_r, K,                 { B, b_type, ldb }
    @beta_f32,                                       { beta (FP32 pointer) }
    devC, rocblas_datatype_f16_r, M,                 { C, c_type, ldc }
    devC, rocblas_datatype_f16_r, M,                 { D, d_type, ldd }
    rocblas_datatype_f32_r,                          { compute_type }
    rocblas_gemm_algo_standard,                      { algo }
    0,                                               { solution_index }
    0);                                              { flags }

  AssertEquals('rocblas_gemm_ex FP16 should return success', rocblas_status_success, blas_status);

  hipMemcpy(@hostD[0], devC, M*N*SizeOf(rocblas_half), hipMemcpyDeviceToHost);

  { Unpack half -> single and compare }
  for i := 0 to M*N-1 do
  begin
    actual[i] := HalfToSingle(hostD[i]);
    AssertEquals('GEMM_EX FP16 result[' + IntToStr(i) + ']', expected[i], actual[i], 0.01);
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
end;


initialization
  RegisterTest(TTestRocBLAS_Level3);

end.

