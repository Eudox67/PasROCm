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

unit Test_rocblas_lvl3_batched_64;

{$mode objfpc}{$H+}
{$linklib rocblas}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  TTestRocBLAS_Level3_Batched_64 = class(TTestRocBLAS_Base)
  published
    procedure Test_SGEMM_Batched_64;
    procedure Test_DGEMM_Batched_64;
    procedure Test_CGEMM_Batched_64;
    procedure Test_ZGEMM_Batched_64;
    procedure Test_SSYMM_Batched_64;
    procedure Test_DSYMM_Batched_64;
    procedure Test_CSYMM_Batched_64;
    procedure Test_ZSYMM_Batched_64;
    procedure Test_STRMM_Batched_64;
    procedure Test_DTRMM_Batched_64;
    procedure Test_CTRMM_Batched_64;
    procedure Test_ZTRMM_Batched_64;
    procedure Test_STRSM_Batched_64;
    procedure Test_DTRSM_Batched_64;
    procedure Test_CTRSM_Batched_64;
    procedure Test_ZTRSM_Batched_64;
    procedure Test_SSYRK_Batched_64;
    procedure Test_DSYRK_Batched_64;
    procedure Test_CSYRK_Batched_64;
    procedure Test_ZSYRK_Batched_64;
    procedure Test_CHERK_Batched_64;
    procedure Test_ZHERK_Batched_64;
    procedure Test_SSyr2k_Batched_64;
    procedure Test_DSyr2k_Batched_64;
    procedure Test_CSyr2k_Batched_64;
    procedure Test_ZSyr2k_Batched_64;
    procedure Test_CHER2K_Batched_64;
    procedure Test_ZHER2K_Batched_64;
    procedure Test_CHEMM_Batched_64;
    procedure Test_ZHEMM_Batched_64;
  end;

implementation

procedure TTestRocBLAS_Level3_Batched_64.Test_SGEMM_Batched_64;
const
  M: Int64 = 4;
  N: Int64 = 3;
  K: Int64 = 5;
  batch_count: Int64 = 2;
  lda: Int64 = 4; // Column-major leading dimension for A, must be >= M
  ldb: Int64 = 5; // Column-major leading dimension for B, must be >= N if transB=T
  ldc: Int64 = 4; // Column-major leading dimension for C, must be >= M
  transA: rocblas_operation = rocblas_operation_none;
  transB: rocblas_operation = rocblas_operation_transpose;
  ALPHA_VALUE: Single = 2.0;
  BETA_VALUE: Single = 1.5;
  rel_epsilon: Single = 1e-5;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C, host_C_initial, host_C_expected: array of Single;
  host_alpha, host_beta: Single;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of PSingle;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  hip_status: hipError_t;
  blas_status: rocblas_status;
  sum: Single;
  idx_A, idx_B, idx_C, base_A, base_B, base_C: Int64;
begin
  host_alpha := ALPHA_VALUE;
  host_beta := BETA_VALUE;

  // Correct calculation for B (n x k matrix) size
  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_B := Int64(batch_count) * ldb * K; // If B is N x K, size is ldb * K
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C, total_elements_C);
  SetLength(host_C_initial, total_elements_C); // Allocate space for initial C
  SetLength(host_C_expected, total_elements_C);

  // Initialize host matrices
  for i := 0 to High(host_A) do host_A[i] := (Random - 0.5) / 10;
  for i := 0 to High(host_B) do host_B[i] := (Random - 0.5) / 10;
  for i := 0 to High(host_C) do host_C[i] := (Random - 0.5) / 10;

  // *** FIX: Save the initial state of host_C ***
  System.Move(host_C[0], host_C_initial[0], total_elements_C * SizeOf(Single));

  // CPU reference calculation: C = beta*C + alpha*A*B^T
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_B := b * ldb * K;
    base_C := b * ldc * N;
    for j := 0 to M - 1 do      // Rows of C and A
    begin
      for i := 0 to N - 1 do  // Cols of C and B^T
      begin
        sum := 0.0;
        for l := 0 to K - 1 do // Inner dimension
        begin
          // A(j, l) * B^T(l, i) which is B(i, l)
          idx_A := base_A + l * lda + j;
          idx_B := base_B + l * ldb + i;
          sum := sum + host_A[idx_A] * host_B[idx_B];
        end;
        idx_C := base_C + i * ldc + j;
        host_C_expected[idx_C] := host_beta * host_C[idx_C] + host_alpha * sum;
      end;
    end;
  end;

  // Allocate device memory
  size_A := total_elements_A * SizeOf(Single);
  size_B := total_elements_B * SizeOf(Single);
  size_C := total_elements_C * SizeOf(Single);

  hip_status := hipMalloc(devA, size_A);
  AssertEquals('hipMalloc devA failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(devB, size_B);
  AssertEquals('hipMalloc devB failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(devC, size_C);
  AssertEquals('hipMalloc devC failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // Copy initial host data to device
  hip_status := hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H->D devA failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H->D devB failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice); // Use initial C
  AssertEquals('hipMemcpy H->D devC failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- Setup for batched pointers ---
  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);

  hip_status := hipMalloc(devA_array_device, batch_count * SizeOf(PSingle));
  AssertEquals('hipMalloc devA_array_device failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(devB_array_device, batch_count * SizeOf(PSingle));
  AssertEquals('hipMalloc devB_array_device failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(devC_array_device, batch_count * SizeOf(PSingle));
  AssertEquals('hipMalloc devC_array_device failed', Integer(HIP_SUCCESS), Integer(hip_status));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PSingle(PtrUInt(devA) + (b * lda * K * SizeOf(Single)));
    devB_array[b] := PSingle(PtrUInt(devB) + (b * ldb * K * SizeOf(Single)));
    devC_array[b] := PSingle(PtrUInt(devC) + (b * ldc * N * SizeOf(Single)));
  end;

  hip_status := hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H->D devA_array_device failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H->D devB_array_device failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H->D devC_array_device failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- Test 1: Pointer Mode Host ---
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode host failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sgemm_batched_64(handle, transA, transB, M, N, K, @host_alpha,
    PPSingle(devA_array_device), lda, PPSingle(devB_array_device), ldb, @host_beta,
    PPSingle(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_sgemm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));

  // Get result from host-mode test
  hip_status := hipMemcpy(@host_C[0], devC, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy D->H devC (host mode) failed', Integer(HIP_SUCCESS), Integer(hip_status));

  for i := 0 to High(host_C) do
  begin
    CheckEqualsRelative(host_C_expected[i], host_C[i], rel_epsilon,
      Format('SGEMM_Batched_64 result mismatch at index %d (Host Mode)', [i]));
  end;

  // --- Test 2: Pointer Mode Device ---

  // *** FIX: Reset device C matrix to its original state before the next run ***
  hip_status := hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H->D devC reset failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // Allocate and set device scalars
  hip_status := hipMalloc(dev_alpha_mem, SizeOf(Single));
  AssertEquals('hipMalloc dev_alpha failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(dev_beta_mem, SizeOf(Single));
  AssertEquals('hipMalloc dev_beta failed', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H->D dev_alpha failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(dev_beta_mem, @host_beta, SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H->D dev_beta failed', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode device failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sgemm_batched_64(handle, transA, transB, M, N, K, PSingle(dev_alpha_mem),
    PPSingle(devA_array_device), lda, PPSingle(devB_array_device), ldb, PSingle(dev_beta_mem),
    PPSingle(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_sgemm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));

  // Get result from device-mode test
  hip_status := hipMemcpy(@host_C[0], devC, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy D->H devC (device mode) failed', Integer(HIP_SUCCESS), Integer(hip_status));

  for i := 0 to High(host_C) do
  begin
    CheckEqualsRelative(host_C_expected[i], host_C[i], rel_epsilon,
      Format('SGEMM_Batched_64 result mismatch at index %d (Device Mode)', [i]));
  end;

  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
  hipFree(dev_alpha_mem);
  hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_DGEMM_Batched_64;
const
  M: Int64 = 4;
  N: Int64 = 3;
  K: Int64 = 5;
  batch_count: Int64 = 2;
  lda: Int64 = 4;
  ldb: Int64 = 5;
  ldc: Int64 = 4;
  transA: rocblas_operation = rocblas_operation_none;
  transB: rocblas_operation = rocblas_operation_transpose;
  ALPHA_VALUE: Double = 2.0;
  BETA_VALUE: Double = 1.5;
  rel_epsilon: Double = 1e-12; // Tolerance for Double precision
var
  size_A, size_B, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C, host_C_initial, host_C_expected: array of Double;
  host_alpha, host_beta: Double;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of PDouble;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  hip_status: hipError_t;
  blas_status: rocblas_status;
  sum: Double;
  idx_A, idx_B, idx_C, base_A, base_B, base_C: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha := ALPHA_VALUE;
  host_beta := BETA_VALUE;

  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_B := Int64(batch_count) * ldb * K;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do host_A[i] := (Random - 0.5) / 10.0;
  for i := 0 to High(host_B) do host_B[i] := (Random - 0.5) / 10.0;
  for i := 0 to High(host_C) do host_C[i] := (Random - 0.5) / 10.0;

  System.Move(host_C[0], host_C_initial[0], total_elements_C * SizeOf(Double));

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_B := b * ldb * K;
    base_C := b * ldc * N;
    for j := 0 to M - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        sum := 0.0;
        for l := 0 to K - 1 do
        begin
          idx_A := base_A + l * lda + j;
          idx_B := base_B + l * ldb + i;
          sum := sum + host_A[idx_A] * host_B[idx_B];
        end;
        idx_C := base_C + i * ldc + j;
        host_C_expected[idx_C] := host_beta * host_C[idx_C] + host_alpha * sum;
      end;
    end;
  end;

  size_A := total_elements_A * SizeOf(Double);
  size_B := total_elements_B * SizeOf(Double);
  size_C := total_elements_C * SizeOf(Double);

  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(PDouble));
  hipMalloc(devB_array_device, batch_count * SizeOf(PDouble));
  hipMalloc(devC_array_device, batch_count * SizeOf(PDouble));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PDouble(PtrUInt(devA) + (b * lda * K * SizeOf(Double)));
    devB_array[b] := PDouble(PtrUInt(devB) + (b * ldb * K * SizeOf(Double)));
    devC_array[b] := PDouble(PtrUInt(devC) + (b * ldc * N * SizeOf(Double)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);

  // --- Test 1: Pointer Mode Host ---
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode host failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dgemm_batched_64(handle, transA, transB, M, N, K, @host_alpha,
    PPDouble(devA_array_device), lda, PPDouble(devB_array_device), ldb, @host_beta,
    PPDouble(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_dgemm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@host_C[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C) do
  begin
    if host_C_expected[i] <> 0.0 then
      relative_error := Abs((host_C_expected[i] - host_C[i]) / host_C_expected[i])
    else
      relative_error := Abs(host_C[i]);
    error_msg := Format('DGEMM_Batched_64 result mismatch at index %d (Host Mode) Relative Error: %.15e', [i, relative_error]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  // --- Test 2: Pointer Mode Device ---
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(Double));
  hipMalloc(dev_beta_mem, SizeOf(Double));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(Double), hipMemcpyHostToDevice);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode device failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dgemm_batched_64(handle, transA, transB, M, N, K, PDouble(dev_alpha_mem),
    PPDouble(devA_array_device), lda, PPDouble(devB_array_device), ldb, PDouble(dev_beta_mem),
    PPDouble(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_dgemm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@host_C[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C) do
  begin
    if host_C_expected[i] <> 0.0 then
      relative_error := Abs((host_C_expected[i] - host_C[i]) / host_C_expected[i])
    else
      relative_error := Abs(host_C[i]);
    error_msg := Format('DGEMM_Batched_64 result mismatch at index %d (Device Mode) Relative Error: %.15e', [i, relative_error]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  // Cleanup
  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
  hipFree(dev_alpha_mem);
  hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_CGEMM_Batched_64;
const
  M: Int64 = 4;
  N: Int64 = 3;
  K: Int64 = 5;
  batch_count: Int64 = 2;
  lda: Int64 = 4;
  ldb: Int64 = 5;
  ldc: Int64 = 4;
  transA: rocblas_operation = rocblas_operation_none;
  transB: rocblas_operation = rocblas_operation_transpose;
  rel_epsilon: Single = 1e-4;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C, host_C_initial, host_C_expected: array of rocblas_float_complex;
  host_alpha, host_beta: rocblas_float_complex;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of Procfloat_complex;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  hip_status: hipError_t;
  blas_status: rocblas_status;
  sum, term: rocblas_float_complex;
  idx_A, idx_B, idx_C, base_A, base_B, base_C: Int64;
begin
  host_alpha.x := 2.0; host_alpha.y := -0.5;
  host_beta.x  := 1.5; host_beta.y  :=  1.0;

  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_B := Int64(batch_count) * ldb * K;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do begin host_A[i].x := (Random-0.5)/10; host_A[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_B) do begin host_B[i].x := (Random-0.5)/10; host_B[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C) do begin host_C[i].x := (Random-0.5)/10; host_C[i].y := (Random-0.5)/10; end;

  System.Move(host_C[0], host_C_initial[0], total_elements_C * SizeOf(rocblas_float_complex));

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_B := b * ldb * K;
    base_C := b * ldc * N;
    for j := 0 to M - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        sum.x := 0.0; sum.y := 0.0;
        for l := 0 to K - 1 do
        begin
          idx_A := base_A + l * lda + j;
          idx_B := base_B + l * ldb + i;
          // term = A * B^T
          term.x := host_A[idx_A].x * host_B[idx_B].x - host_A[idx_A].y * host_B[idx_B].y;
          term.y := host_A[idx_A].x * host_B[idx_B].y + host_A[idx_A].y * host_B[idx_B].x;
          ComplexAdd(sum, term);
        end;
        idx_C := base_C + i * ldc + j;
        // C_exp = beta*C + alpha*sum
        term := host_C[idx_C];
        ComplexMul(term, host_beta); // term = beta*C
        ComplexMul(sum, host_alpha); // sum = alpha*sum
        ComplexAdd(term, sum);       // term = beta*C + alpha*sum
        host_C_expected[idx_C] := term;
      end;
    end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_float_complex);
  size_B := total_elements_B * SizeOf(rocblas_float_complex);
  size_C := total_elements_C * SizeOf(rocblas_float_complex);

  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procfloat_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procfloat_complex(PtrUInt(devA) + (b * lda * K * SizeOf(rocblas_float_complex)));
    devB_array[b] := Procfloat_complex(PtrUInt(devB) + (b * ldb * K * SizeOf(rocblas_float_complex)));
    devC_array[b] := Procfloat_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_float_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);

  // --- Test 1: Pointer Mode Host ---
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode host failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgemm_batched_64(handle, transA, transB, M, N, K, @host_alpha,
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb, @host_beta,
    PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_cgemm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@host_C[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C) do
  begin
    CheckEqualsRelative(host_C_expected[i].x, host_C[i].x, rel_epsilon, Format('CGEMM_Batched_64 result.x mismatch at index %d (Host Mode)', [i]));
    CheckEqualsRelative(host_C_expected[i].y, host_C[i].y, rel_epsilon, Format('CGEMM_Batched_64 result.y mismatch at index %d (Host Mode)', [i]));
  end;

  // --- Test 2: Pointer Mode Device ---
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_float_complex));
  hipMalloc(dev_beta_mem, SizeOf(rocblas_float_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode device failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgemm_batched_64(handle, transA, transB, M, N, K, Procfloat_complex(dev_alpha_mem),
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb, Procfloat_complex(dev_beta_mem),
    PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_cgemm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@host_C[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C) do
  begin
    CheckEqualsRelative(host_C_expected[i].x, host_C[i].x, rel_epsilon, Format('CGEMM_Batched_64 result.x mismatch at index %d (Device Mode)', [i]));
    CheckEqualsRelative(host_C_expected[i].y, host_C[i].y, rel_epsilon, Format('CGEMM_Batched_64 result.y mismatch at index %d (Device Mode)', [i]));
  end;

  // Cleanup
  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
  hipFree(dev_alpha_mem);
  hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_ZGEMM_Batched_64;
const
  M: Int64 = 4;
  N: Int64 = 3;
  K: Int64 = 5;
  batch_count: Int64 = 2;
  lda: Int64 = 4;
  ldb: Int64 = 5;
  ldc: Int64 = 4;
  transA: rocblas_operation = rocblas_operation_none;
  transB: rocblas_operation = rocblas_operation_transpose;
  rel_epsilon: Double = 1e-12; // Tolerance for Double precision
var
  size_A, size_B, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C, host_C_initial, host_C_expected: array of rocblas_double_complex;
  host_alpha, host_beta: rocblas_double_complex;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of Procdouble_complex;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  hip_status: hipError_t;
  blas_status: rocblas_status;
  sum, term: rocblas_double_complex;
  idx_A, idx_B, idx_C, base_A, base_B, base_C: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha.x := 2.0; host_alpha.y := -0.5;
  host_beta.x  := 1.5; host_beta.y  :=  1.0;

  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_B := Int64(batch_count) * ldb * K;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do begin host_A[i].x := (Random-0.5)/10; host_A[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_B) do begin host_B[i].x := (Random-0.5)/10; host_B[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C) do begin host_C[i].x := (Random-0.5)/10; host_C[i].y := (Random-0.5)/10; end;

  System.Move(host_C[0], host_C_initial[0], total_elements_C * SizeOf(rocblas_double_complex));

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_B := b * ldb * K;
    base_C := b * ldc * N;
    for j := 0 to M - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        sum.x := 0.0; sum.y := 0.0;
        for l := 0 to K - 1 do
        begin
          idx_A := base_A + l * lda + j;
          idx_B := base_B + l * ldb + i;
          term.x := host_A[idx_A].x * host_B[idx_B].x - host_A[idx_A].y * host_B[idx_B].y;
          term.y := host_A[idx_A].x * host_B[idx_B].y + host_A[idx_A].y * host_B[idx_B].x;
          ComplexAdd(sum, term);
        end;
        idx_C := base_C + i * ldc + j;
        term := host_C[idx_C];
        ComplexMul(term, host_beta);
        ComplexMul(sum, host_alpha);
        ComplexAdd(term, sum);
        host_C_expected[idx_C] := term;
      end;
    end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_double_complex);
  size_B := total_elements_B * SizeOf(rocblas_double_complex);
  size_C := total_elements_C * SizeOf(rocblas_double_complex);

  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procdouble_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + (b * lda * K * SizeOf(rocblas_double_complex)));
    devB_array[b] := Procdouble_complex(PtrUInt(devB) + (b * ldb * K * SizeOf(rocblas_double_complex)));
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_double_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);

  // --- Test 1: Pointer Mode Host ---
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode host failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgemm_batched_64(handle, transA, transB, M, N, K, @host_alpha,
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb, @host_beta,
    PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zgemm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@host_C[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C) do
  begin
    if host_C_expected[i].x <> 0.0 then relative_error := Abs((host_C_expected[i].x - host_C[i].x) / host_C_expected[i].x) else relative_error := Abs(host_C[i].x);
    error_msg := Format('ZGEMM_Batched_64 result.x mismatch at index %d (Host Mode) Relative Error: %.15e', [i, relative_error]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);

    if host_C_expected[i].y <> 0.0 then relative_error := Abs((host_C_expected[i].y - host_C[i].y) / host_C_expected[i].y) else relative_error := Abs(host_C[i].y);
    error_msg := Format('ZGEMM_Batched_64 result.y mismatch at index %d (Host Mode) Relative Error: %.15e', [i, relative_error]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  // --- Test 2: Pointer Mode Device ---
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_double_complex));
  hipMalloc(dev_beta_mem, SizeOf(rocblas_double_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode device failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgemm_batched_64(handle, transA, transB, M, N, K, Procdouble_complex(dev_alpha_mem),
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb, Procdouble_complex(dev_beta_mem),
    PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zgemm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@host_C[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C) do
  begin
    if host_C_expected[i].x <> 0.0 then relative_error := Abs((host_C_expected[i].x - host_C[i].x) / host_C_expected[i].x) else relative_error := Abs(host_C[i].x);
    error_msg := Format('ZGEMM_Batched_64 result.x mismatch at index %d (Device Mode) Relative Error: %.15e', [i, relative_error]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);

    if host_C_expected[i].y <> 0.0 then relative_error := Abs((host_C_expected[i].y - host_C[i].y) / host_C_expected[i].y) else relative_error := Abs(host_C[i].y);
    error_msg := Format('ZGEMM_Batched_64 result.y mismatch at index %d (Device Mode) Relative Error: %.15e', [i, relative_error]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  // Cleanup
  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
  hipFree(dev_alpha_mem);
  hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_SSYMM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  lda: Int64 = 5; // For side_left, A is M x M, so lda >= M
  ldb: Int64 = 5; // B is M x N, so ldb >= M
  ldc: Int64 = 5; // C is M x N, so ldc >= M
  ALPHA_VALUE: Single = 1.5;
  BETA_VALUE: Single = 2.5;
  rel_epsilon: Single = 1e-5;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_initial, host_C_expected: array of Single;
  host_alpha, host_beta: Single;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of PSingle;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, valA: Single;
  base_A, base_B, base_C: Int64;
begin
  host_alpha := ALPHA_VALUE;
  host_beta := BETA_VALUE;

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do host_A[i] := (Random - 0.5) / 10;
  for i := 0 to High(host_B) do host_B[i] := (Random - 0.5) / 10;
  for i := 0 to High(host_C_initial) do host_C_initial[i] := (Random - 0.5) / 10;

  // CPU reference: C = beta*C_initial + alpha*A*B, with A symmetric, uplo=lower
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do       // Columns of C and B
      for i := 0 to M - 1 do   // Rows of C, B, and A
      begin
        sum := 0.0;
        for k := 0 to M - 1 do // Inner dimension (cols of A, rows of B)
        begin
          // Get A[i,k], respecting symmetric lower storage
          if i < k then valA := host_A[base_A + i * lda + k]
          else valA := host_A[base_A + k * lda + i];

          sum := sum + valA * host_B[base_B + j * ldb + k];
        end;
        host_C_expected[base_C + j * ldc + i] := host_beta * host_C_initial[base_C + j * ldc + i] + host_alpha * sum;
      end;
  end;

  size_A := total_elements_A * SizeOf(Single);
  size_B := total_elements_B * SizeOf(Single);
  size_C := total_elements_C * SizeOf(Single);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(PSingle));
  hipMalloc(devB_array_device, batch_count * SizeOf(PSingle));
  hipMalloc(devC_array_device, batch_count * SizeOf(PSingle));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PSingle(PtrUInt(devA) + (b * lda * M * SizeOf(Single)));
    devB_array[b] := PSingle(PtrUInt(devB) + (b * ldb * N * SizeOf(Single)));
    devC_array[b] := PSingle(PtrUInt(devC) + (b * ldc * N * SizeOf(Single)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);

  // --- Test 1: Pointer Mode Host ---
  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_ssymm_batched_64(handle, side, uplo, M, N, @host_alpha,
    PPSingle(devA_array_device), lda, PPSingle(devB_array_device), ldb, @host_beta,
    PPSingle(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_ssymm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
    CheckEqualsRelative(host_C_expected[i], host_C_result[i], rel_epsilon, Format('SSYMM_Batched_64 result mismatch at index %d (Host Mode)', [i]));

  // --- Test 2: Pointer Mode Device ---
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice); // Reset C
  hipMalloc(dev_alpha_mem, SizeOf(Single));
  hipMalloc(dev_beta_mem, SizeOf(Single));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(Single), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_ssymm_batched_64(handle, side, uplo, M, N, PSingle(dev_alpha_mem),
    PPSingle(devA_array_device), lda, PPSingle(devB_array_device), ldb, PSingle(dev_beta_mem),
    PPSingle(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_ssymm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
    CheckEqualsRelative(host_C_expected[i], host_C_result[i], rel_epsilon, Format('SSYMM_Batched_64 result mismatch at index %d (Device Mode)', [i]));

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_DSYMM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ldc: Int64 = 5;
  ALPHA_VALUE: Double = 1.5;
  BETA_VALUE: Double = 2.5;
  rel_epsilon: Double = 1e-12;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_initial, host_C_expected: array of Double;
  host_alpha, host_beta: Double;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of PDouble;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, valA: Double;
  base_A, base_B, base_C: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha := ALPHA_VALUE;
  host_beta := BETA_VALUE;

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do host_A[i] := (Random - 0.5) / 10.0;
  for i := 0 to High(host_B) do host_B[i] := (Random - 0.5) / 10.0;
  for i := 0 to High(host_C_initial) do host_C_initial[i] := (Random - 0.5) / 10.0;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        sum := 0.0;
        for k := 0 to M - 1 do
        begin
          if i < k then valA := host_A[base_A + i * lda + k]
          else valA := host_A[base_A + k * lda + i];
          sum := sum + valA * host_B[base_B + j * ldb + k];
        end;
        host_C_expected[base_C + j * ldc + i] := host_beta * host_C_initial[base_C + j * ldc + i] + host_alpha * sum;
      end;
  end;

  size_A := total_elements_A * SizeOf(Double);
  size_B := total_elements_B * SizeOf(Double);
  size_C := total_elements_C * SizeOf(Double);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(PDouble));
  hipMalloc(devB_array_device, batch_count * SizeOf(PDouble));
  hipMalloc(devC_array_device, batch_count * SizeOf(PDouble));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PDouble(PtrUInt(devA) + (b * lda * M * SizeOf(Double)));
    devB_array[b] := PDouble(PtrUInt(devB) + (b * ldb * N * SizeOf(Double)));
    devC_array[b] := PDouble(PtrUInt(devC) + (b * ldc * N * SizeOf(Double)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_dsymm_batched_64(handle, side, uplo, M, N, @host_alpha,
    PPDouble(devA_array_device), lda, PPDouble(devB_array_device), ldb, @host_beta,
    PPDouble(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_dsymm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    if host_C_expected[i] <> 0.0 then relative_error := Abs((host_C_expected[i] - host_C_result[i]) / host_C_expected[i]) else relative_error := Abs(host_C_result[i]);
    error_msg := Format('DSYMM_Batched_64 result mismatch at index %d (Host Mode) Relative Error: %.15e', [i, relative_error]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(Double));
  hipMalloc(dev_beta_mem, SizeOf(Double));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(Double), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_dsymm_batched_64(handle, side, uplo, M, N, PDouble(dev_alpha_mem),
    PPDouble(devA_array_device), lda, PPDouble(devB_array_device), ldb, PDouble(dev_beta_mem),
    PPDouble(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_dsymm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    if host_C_expected[i] <> 0.0 then relative_error := Abs((host_C_expected[i] - host_C_result[i]) / host_C_expected[i]) else relative_error := Abs(host_C_result[i]);
    error_msg := Format('DSYMM_Batched_64 result mismatch at index %d (Device Mode) Relative Error: %.15e', [i, relative_error]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_CSYMM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ldc: Int64 = 5;
  rel_epsilon: Single = 1e-4;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_initial, host_C_expected: array of rocblas_float_complex;
  host_alpha, host_beta: rocblas_float_complex;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of Procfloat_complex;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, term, valA: rocblas_float_complex;
  base_A, base_B, base_C: Int64;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;
  host_beta.x  := 2.5; host_beta.y  :=  1.0;

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do begin host_A[i].x := (Random-0.5)/10; host_A[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_B) do begin host_B[i].x := (Random-0.5)/10; host_B[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C_initial) do begin host_C_initial[i].x := (Random-0.5)/10; host_C_initial[i].y := (Random-0.5)/10; end;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        sum.x := 0.0; sum.y := 0.0;
        for k := 0 to M - 1 do
        begin
          if i < k then valA := host_A[base_A + i * lda + k]
          else valA := host_A[base_A + k * lda + i];
          term := host_B[base_B + j * ldb + k];
          ComplexMul(valA, term);
          ComplexAdd(sum, valA);
        end;
        term := host_C_initial[base_C + j * ldc + i];
        ComplexMul(term, host_beta);
        ComplexMul(sum, host_alpha);
        ComplexAdd(term, sum);
        host_C_expected[base_C + j * ldc + i] := term;
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_float_complex);
  size_B := total_elements_B * SizeOf(rocblas_float_complex);
  size_C := total_elements_C * SizeOf(rocblas_float_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procfloat_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procfloat_complex(PtrUInt(devA) + (b * lda * M * SizeOf(rocblas_float_complex)));
    devB_array[b] := Procfloat_complex(PtrUInt(devB) + (b * ldb * N * SizeOf(rocblas_float_complex)));
    devC_array[b] := Procfloat_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_float_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_csymm_batched_64(handle, side, uplo, M, N, @host_alpha,
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb, @host_beta,
    PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_csymm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    CheckEqualsRelative(host_C_expected[i].x, host_C_result[i].x, rel_epsilon, Format('CSYMM_Batched_64 result.x mismatch at index %d (Host Mode)', [i]));
    CheckEqualsRelative(host_C_expected[i].y, host_C_result[i].y, rel_epsilon, Format('CSYMM_Batched_64 result.y mismatch at index %d (Host Mode)', [i]));
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_float_complex));
  hipMalloc(dev_beta_mem, SizeOf(rocblas_float_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_csymm_batched_64(handle, side, uplo, M, N, Procfloat_complex(dev_alpha_mem),
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb, Procfloat_complex(dev_beta_mem),
    PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_csymm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    CheckEqualsRelative(host_C_expected[i].x, host_C_result[i].x, rel_epsilon, Format('CSYMM_Batched_64 result.x mismatch at index %d (Device Mode)', [i]));
    CheckEqualsRelative(host_C_expected[i].y, host_C_result[i].y, rel_epsilon, Format('CSYMM_Batched_64 result.y mismatch at index %d (Device Mode)', [i]));
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_ZSYMM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ldc: Int64 = 5;
  rel_epsilon: Double = 1e-12;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_initial, host_C_expected: array of rocblas_double_complex;
  host_alpha, host_beta: rocblas_double_complex;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of Procdouble_complex;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, term, valA: rocblas_double_complex;
  base_A, base_B, base_C: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;
  host_beta.x  := 2.5; host_beta.y  :=  1.0;

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do begin host_A[i].x := (Random-0.5)/10; host_A[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_B) do begin host_B[i].x := (Random-0.5)/10; host_B[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C_initial) do begin host_C_initial[i].x := (Random-0.5)/10; host_C_initial[i].y := (Random-0.5)/10; end;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        sum.x := 0.0; sum.y := 0.0;
        for k := 0 to M - 1 do
        begin
          if i < k then valA := host_A[base_A + i * lda + k]
          else valA := host_A[base_A + k * lda + i];
          term := host_B[base_B + j * ldb + k];
          ComplexMul(valA, term);
          ComplexAdd(sum, valA);
        end;
        term := host_C_initial[base_C + j * ldc + i];
        ComplexMul(term, host_beta);
        ComplexMul(sum, host_alpha);
        ComplexAdd(term, sum);
        host_C_expected[base_C + j * ldc + i] := term;
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_double_complex);
  size_B := total_elements_B * SizeOf(rocblas_double_complex);
  size_C := total_elements_C * SizeOf(rocblas_double_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procdouble_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + (b * lda * M * SizeOf(rocblas_double_complex)));
    devB_array[b] := Procdouble_complex(PtrUInt(devB) + (b * ldb * N * SizeOf(rocblas_double_complex)));
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_double_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_zsymm_batched_64(handle, side, uplo, M, N, @host_alpha,
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb, @host_beta,
    PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zsymm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    if host_C_expected[i].x <> 0.0 then relative_error := Abs((host_C_expected[i].x - host_C_result[i].x) / host_C_expected[i].x) else relative_error := Abs(host_C_result[i].x);
    error_msg := Format('ZSYMM_Batched_64 result.x mismatch at index %d (Host Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
    if host_C_expected[i].y <> 0.0 then relative_error := Abs((host_C_expected[i].y - host_C_result[i].y) / host_C_expected[i].y) else relative_error := Abs(host_C_result[i].y);
    error_msg := Format('ZSYMM_Batched_64 result.y mismatch at index %d (Host Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_double_complex));
  hipMalloc(dev_beta_mem, SizeOf(rocblas_double_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_zsymm_batched_64(handle, side, uplo, M, N, Procdouble_complex(dev_alpha_mem),
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb, Procdouble_complex(dev_beta_mem),
    PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zsymm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    if host_C_expected[i].x <> 0.0 then relative_error := Abs((host_C_expected[i].x - host_C_result[i].x) / host_C_expected[i].x) else relative_error := Abs(host_C_result[i].x);
    error_msg := Format('ZSYMM_Batched_64 result.x mismatch at index %d (Device Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
    if host_C_expected[i].y <> 0.0 then relative_error := Abs((host_C_expected[i].y - host_C_result[i].y) / host_C_expected[i].y) else relative_error := Abs(host_C_result[i].y);
    error_msg := Format('ZSYMM_Batched_64 result.y mismatch at index %d (Device Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_STRMM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  diag: rocblas_diagonal = rocblas_diagonal_non_unit;
  lda: Int64 = 5; // if side=L, A is MxM -> lda>=M
  ldb: Int64 = 5; // B is MxN -> ldb>=M
  ldc: Int64 = 5; // C is MxN -> ldc>=M
  ALPHA_VALUE: Single = 1.5;
  rel_epsilon: Single = 1e-4;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_expected: array of Single;
  host_alpha: Single;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of PSingle;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, valA: Single;
  base_A, base_B, base_C: Int64;
begin
  host_alpha := ALPHA_VALUE;

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  // Initialize A as lower triangular
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    for j := 0 to M - 1 do
      for i := 0 to M - 1 do
        if i >= j then host_A[base_A + j * lda + i] := (Random - 0.5) / 10.0
        else host_A[base_A + j * lda + i] := 0.0;
  end;
  for i := 0 to High(host_B) do host_B[i] := (Random - 0.5) / 10.0;

  // CPU reference: C_expected = alpha * A * B
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        sum := 0.0;
        for k := 0 to M - 1 do
        begin
          valA := host_A[base_A + k * lda + i];
          sum := sum + valA * host_B[base_B + j * ldb + k];
        end;
        host_C_expected[base_C + j * ldc + i] := host_alpha * sum;
      end;
  end;

  size_A := total_elements_A * SizeOf(Single);
  size_B := total_elements_B * SizeOf(Single);
  size_C := total_elements_C * SizeOf(Single);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(PSingle));
  hipMalloc(devB_array_device, batch_count * SizeOf(PSingle));
  hipMalloc(devC_array_device, batch_count * SizeOf(PSingle));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PSingle(PtrUInt(devA) + (b * lda * M * SizeOf(Single)));
    devB_array[b] := PSingle(PtrUInt(devB) + (b * ldb * N * SizeOf(Single)));
    devC_array[b] := PSingle(PtrUInt(devC) + (b * ldc * N * SizeOf(Single)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);

  // --- Test 1: Pointer Mode Host ---
  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_strmm_batched_64(handle, side, uplo, transA, diag, M, N, @host_alpha,
    PPSingle(devA_array_device), lda, PPSingle(devB_array_device), ldb, PPSingle(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_strmm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
    CheckEqualsRelative(host_C_expected[i], host_C_result[i], rel_epsilon, Format('STRMM_Batched_64 result mismatch at index %d (Host Mode)', [i]));

  // --- Test 2: Pointer Mode Device ---
  hipMalloc(dev_alpha_mem, SizeOf(Single));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Single), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_strmm_batched_64(handle, side, uplo, transA, diag, M, N, PSingle(dev_alpha_mem),
    PPSingle(devA_array_device), lda, PPSingle(devB_array_device), ldb, PPSingle(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_strmm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
    CheckEqualsRelative(host_C_expected[i], host_C_result[i], rel_epsilon, Format('STRMM_Batched_64 result mismatch at index %d (Device Mode)', [i]));

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_DTRMM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  diag: rocblas_diagonal = rocblas_diagonal_non_unit;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ldc: Int64 = 5;
  ALPHA_VALUE: Double = 1.5;
  rel_epsilon: Double = 1e-12;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_expected: array of Double;
  host_alpha: Double;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of PDouble;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, valA: Double;
  base_A, base_B, base_C: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha := ALPHA_VALUE;

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    for j := 0 to M - 1 do
      for i := 0 to M - 1 do
        if i >= j then host_A[base_A + j * lda + i] := (Random - 0.5) / 10.0
        else host_A[base_A + j * lda + i] := 0.0;
  end;
  for i := 0 to High(host_B) do host_B[i] := (Random - 0.5) / 10.0;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        sum := 0.0;
        for k := 0 to M - 1 do
        begin
          valA := host_A[base_A + k * lda + i];
          sum := sum + valA * host_B[base_B + j * ldb + k];
        end;
        host_C_expected[base_C + j * ldc + i] := host_alpha * sum;
      end;
  end;

  size_A := total_elements_A * SizeOf(Double);
  size_B := total_elements_B * SizeOf(Double);
  size_C := total_elements_C * SizeOf(Double);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(PDouble));
  hipMalloc(devB_array_device, batch_count * SizeOf(PDouble));
  hipMalloc(devC_array_device, batch_count * SizeOf(PDouble));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PDouble(PtrUInt(devA) + (b * lda * M * SizeOf(Double)));
    devB_array[b] := PDouble(PtrUInt(devB) + (b * ldb * N * SizeOf(Double)));
    devC_array[b] := PDouble(PtrUInt(devC) + (b * ldc * N * SizeOf(Double)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_dtrmm_batched_64(handle, side, uplo, transA, diag, M, N, @host_alpha,
    PPDouble(devA_array_device), lda, PPDouble(devB_array_device), ldb, PPDouble(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_dtrmm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    if host_C_expected[i] <> 0.0 then relative_error := Abs((host_C_expected[i] - host_C_result[i]) / host_C_expected[i]) else relative_error := Abs(host_C_result[i]);
    error_msg := Format('DTRMM_Batched_64 result mismatch at index %d (Host Mode) Relative Error: %.15e', [i, relative_error]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipMalloc(dev_alpha_mem, SizeOf(Double));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Double), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_dtrmm_batched_64(handle, side, uplo, transA, diag, M, N, PDouble(dev_alpha_mem),
    PPDouble(devA_array_device), lda, PPDouble(devB_array_device), ldb, PPDouble(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_dtrmm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    if host_C_expected[i] <> 0.0 then relative_error := Abs((host_C_expected[i] - host_C_result[i]) / host_C_expected[i]) else relative_error := Abs(host_C_result[i]);
    error_msg := Format('DTRMM_Batched_64 result mismatch at index %d (Device Mode) Relative Error: %.15e', [i, relative_error]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_CTRMM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  diag: rocblas_diagonal = rocblas_diagonal_non_unit;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ldc: Int64 = 5;
  rel_epsilon: Single = 1e-4;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_expected: array of rocblas_float_complex;
  host_alpha: rocblas_float_complex;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of Procfloat_complex;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, term, valA: rocblas_float_complex;
  base_A, base_B, base_C: Int64;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    for j := 0 to M - 1 do
      for i := 0 to M - 1 do
        if i >= j then begin host_A[base_A + j * lda + i].x := (Random-0.5)/10; host_A[base_A + j * lda + i].y := (Random-0.5)/10; end
        else begin host_A[base_A + j * lda + i].x := 0.0; host_A[base_A + j * lda + i].y := 0.0; end;
  end;
  for i := 0 to High(host_B) do begin host_B[i].x := (Random-0.5)/10; host_B[i].y := (Random-0.5)/10; end;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        sum.x := 0; sum.y := 0;
        for k := 0 to M - 1 do
        begin
          valA := host_A[base_A + k * lda + i];
          term := host_B[base_B + j * ldb + k];
          ComplexMul(valA, term);
          ComplexAdd(sum, valA);
        end;
        ComplexMul(sum, host_alpha);
        host_C_expected[base_C + j * ldc + i] := sum;
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_float_complex);
  size_B := total_elements_B * SizeOf(rocblas_float_complex);
  size_C := total_elements_C * SizeOf(rocblas_float_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procfloat_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procfloat_complex(PtrUInt(devA) + (b * lda * M * SizeOf(rocblas_float_complex)));
    devB_array[b] := Procfloat_complex(PtrUInt(devB) + (b * ldb * N * SizeOf(rocblas_float_complex)));
    devC_array[b] := Procfloat_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_float_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_ctrmm_batched_64(handle, side, uplo, transA, diag, M, N, @host_alpha,
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb,
    PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_ctrmm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    CheckEqualsRelative(host_C_expected[i].x, host_C_result[i].x, rel_epsilon, Format('CTRMM_Batched_64 result.x mismatch at index %d (Host Mode)', [i]));
    CheckEqualsRelative(host_C_expected[i].y, host_C_result[i].y, rel_epsilon, Format('CTRMM_Batched_64 result.y mismatch at index %d (Host Mode)', [i]));
  end;

  hipMalloc(dev_alpha_mem, SizeOf(rocblas_float_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_ctrmm_batched_64(handle, side, uplo, transA, diag, M, N, Procfloat_complex(dev_alpha_mem),
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb,
    PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_ctrmm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    CheckEqualsRelative(host_C_expected[i].x, host_C_result[i].x, rel_epsilon, Format('CTRMM_Batched_64 result.x mismatch at index %d (Device Mode)', [i]));
    CheckEqualsRelative(host_C_expected[i].y, host_C_result[i].y, rel_epsilon, Format('CTRMM_Batched_64 result.y mismatch at index %d (Device Mode)', [i]));
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_ZTRMM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  diag: rocblas_diagonal = rocblas_diagonal_non_unit;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ldc: Int64 = 5;
  rel_epsilon: Double = 1e-12;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_expected: array of rocblas_double_complex;
  host_alpha: rocblas_double_complex;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of Procdouble_complex;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, term, valA: rocblas_double_complex;
  base_A, base_B, base_C: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    for j := 0 to M - 1 do
      for i := 0 to M - 1 do
        if i >= j then begin host_A[base_A + j * lda + i].x := (Random-0.5)/10; host_A[base_A + j * lda + i].y := (Random-0.5)/10; end
        else begin host_A[base_A + j * lda + i].x := 0.0; host_A[base_A + j * lda + i].y := 0.0; end;
  end;
  for i := 0 to High(host_B) do begin host_B[i].x := (Random-0.5)/10; host_B[i].y := (Random-0.5)/10; end;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        sum.x := 0; sum.y := 0;
        for k := 0 to M - 1 do
        begin
          valA := host_A[base_A + k * lda + i];
          term := host_B[base_B + j * ldb + k];
          ComplexMul(valA, term);
          ComplexAdd(sum, valA);
        end;
        ComplexMul(sum, host_alpha);
        host_C_expected[base_C + j * ldc + i] := sum;
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_double_complex);
  size_B := total_elements_B * SizeOf(rocblas_double_complex);
  size_C := total_elements_C * SizeOf(rocblas_double_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procdouble_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + (b * lda * M * SizeOf(rocblas_double_complex)));
    devB_array[b] := Procdouble_complex(PtrUInt(devB) + (b * ldb * N * SizeOf(rocblas_double_complex)));
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_double_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_ztrmm_batched_64(handle, side, uplo, transA, diag, M, N, @host_alpha,
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb,
    PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_ztrmm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    if host_C_expected[i].x <> 0.0 then relative_error := Abs((host_C_expected[i].x - host_C_result[i].x) / host_C_expected[i].x) else relative_error := Abs(host_C_result[i].x);
    error_msg := Format('ZTRMM_Batched_64 result.x mismatch at index %d (Host Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
    if host_C_expected[i].y <> 0.0 then relative_error := Abs((host_C_expected[i].y - host_C_result[i].y) / host_C_expected[i].y) else relative_error := Abs(host_C_result[i].y);
    error_msg := Format('ZTRMM_Batched_64 result.y mismatch at index %d (Host Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipMalloc(dev_alpha_mem, SizeOf(rocblas_double_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_ztrmm_batched_64(handle, side, uplo, transA, diag, M, N, Procdouble_complex(dev_alpha_mem),
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb,
    PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_ztrmm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    if host_C_expected[i].x <> 0.0 then relative_error := Abs((host_C_expected[i].x - host_C_result[i].x) / host_C_expected[i].x) else relative_error := Abs(host_C_result[i].x);
    error_msg := Format('ZTRMM_Batched_64 result.x mismatch at index %d (Device Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
    if host_C_expected[i].y <> 0.0 then relative_error := Abs((host_C_expected[i].y - host_C_result[i].y) / host_C_expected[i].y) else relative_error := Abs(host_C_result[i].y);
    error_msg := Format('ZTRMM_Batched_64 result.y mismatch at index %d (Device Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_STRSM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  diag: rocblas_diagonal = rocblas_diagonal_non_unit;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ALPHA_VALUE: Single = 1.5;
  rel_epsilon: Single = 1e-4; // Solvers can have slightly larger errors
var
  size_A, size_B: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B: Int64;

  host_A, host_B_rhs, host_X_solution, host_B_initial_rhs: array of Single;
  host_alpha, inv_alpha: Single;

  devA, devB: hipDeviceptr_t;
  devA_array, devB_array: array of PSingle;
  devA_array_device, devB_array_device: hipDeviceptr_t;
  dev_alpha_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum: Single;
  base_A, base_B: Int64;
begin
  host_alpha := ALPHA_VALUE;
  inv_alpha := 1.0 / host_alpha;

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B_rhs, total_elements_B);
  SetLength(host_X_solution, total_elements_B);
  SetLength(host_B_initial_rhs, total_elements_B);

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    for j := 0 to M - 1 do
      for i := 0 to M - 1 do
        if i > j then host_A[base_A + j * lda + i] := (Random - 0.5) / 10.0
        else if i = j then host_A[base_A + j * lda + i] := (Random - 0.5) / 5.0 + 1.0
        else host_A[base_A + j * lda + i] := 0.0;
  end;
  for i := 0 to High(host_X_solution) do host_X_solution[i] := (Random - 0.5) / 10.0;

  // FIX: Calculate B = (1/alpha) * op(A) * X
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        sum := 0.0;
        for k := 0 to M - 1 do
          sum := sum + host_A[base_A + k * lda + i] * host_X_solution[base_B + j * ldb + k];
        host_B_rhs[base_B + j * ldb + i] := inv_alpha * sum;
      end;
  end;

  System.Move(host_B_rhs[0], host_B_initial_rhs[0], total_elements_B * SizeOf(Single));

  size_A := total_elements_A * SizeOf(Single);
  size_B := total_elements_B * SizeOf(Single);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B_initial_rhs[0], size_B, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(PSingle));
  hipMalloc(devB_array_device, batch_count * SizeOf(PSingle));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PSingle(PtrUInt(devA) + (b * lda * M * SizeOf(Single)));
    devB_array[b] := PSingle(PtrUInt(devB) + (b * ldb * N * SizeOf(Single)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_strsm_batched_64(handle, side, uplo, transA, diag, M, N, @host_alpha,
    PPSingle(devA_array_device), lda, PPSingle(devB_array_device), ldb, batch_count);
  AssertEquals('rocblas_strsm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_B_rhs[0], devB, size_B, hipMemcpyDeviceToHost);

  for i := 0 to High(host_X_solution) do
    CheckEqualsRelative(host_X_solution[i], host_B_rhs[i], rel_epsilon, Format('STRSM_Batched_64 result mismatch at index %d (Host Mode)', [i]));

  hipMemcpy(devB, @host_B_initial_rhs[0], size_B, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(Single));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Single), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_strsm_batched_64(handle, side, uplo, transA, diag, M, N, PSingle(dev_alpha_mem),
    PPSingle(devA_array_device), lda, PPSingle(devB_array_device), ldb, batch_count);
  AssertEquals('rocblas_strsm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_B_rhs[0], devB, size_B, hipMemcpyDeviceToHost);

  for i := 0 to High(host_X_solution) do
    CheckEqualsRelative(host_X_solution[i], host_B_rhs[i], rel_epsilon, Format('STRSM_Batched_64 result mismatch at index %d (Device Mode)', [i]));

  hipFree(devA); hipFree(devB);
  hipFree(devA_array_device); hipFree(devB_array_device);
  hipFree(dev_alpha_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_DTRSM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  diag: rocblas_diagonal = rocblas_diagonal_non_unit;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ALPHA_VALUE: Double = 1.5;
  rel_epsilon: Double = 1e-12;
var
  size_A, size_B: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B: Int64;

  host_A, host_B_rhs, host_X_solution, host_B_initial_rhs: array of Double;
  host_alpha, inv_alpha: Double;

  devA, devB: hipDeviceptr_t;
  devA_array, devB_array: array of PDouble;
  devA_array_device, devB_array_device: hipDeviceptr_t;
  dev_alpha_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum: Double;
  base_A, base_B: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha := ALPHA_VALUE;
  inv_alpha := 1.0 / host_alpha;

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B_rhs, total_elements_B);
  SetLength(host_X_solution, total_elements_B);
  SetLength(host_B_initial_rhs, total_elements_B);

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    for j := 0 to M - 1 do
      for i := 0 to M - 1 do
        if i > j then host_A[base_A + j * lda + i] := (Random - 0.5) / 10.0
        else if i = j then host_A[base_A + j * lda + i] := (Random - 0.5) / 5.0 + 1.0
        else host_A[base_A + j * lda + i] := 0.0;
  end;
  for i := 0 to High(host_X_solution) do host_X_solution[i] := (Random - 0.5) / 10.0;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        sum := 0.0;
        for k := 0 to M - 1 do
          sum := sum + host_A[base_A + k * lda + i] * host_X_solution[base_B + j * ldb + k];
        host_B_rhs[base_B + j * ldb + i] := inv_alpha * sum;
      end;
  end;

  System.Move(host_B_rhs[0], host_B_initial_rhs[0], total_elements_B * SizeOf(Double));

  size_A := total_elements_A * SizeOf(Double);
  size_B := total_elements_B * SizeOf(Double);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B_initial_rhs[0], size_B, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(PDouble));
  hipMalloc(devB_array_device, batch_count * SizeOf(PDouble));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PDouble(PtrUInt(devA) + (b * lda * M * SizeOf(Double)));
    devB_array[b] := PDouble(PtrUInt(devB) + (b * ldb * N * SizeOf(Double)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_dtrsm_batched_64(handle, side, uplo, transA, diag, M, N, @host_alpha,
    PPDouble(devA_array_device), lda, PPDouble(devB_array_device), ldb, batch_count);
  AssertEquals('rocblas_dtrsm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_B_rhs[0], devB, size_B, hipMemcpyDeviceToHost);

  for i := 0 to High(host_X_solution) do
  begin
    if host_X_solution[i] <> 0.0 then relative_error := Abs((host_X_solution[i] - host_B_rhs[i]) / host_X_solution[i]) else relative_error := Abs(host_B_rhs[i]);
    error_msg := Format('DTRSM_Batched_64 result mismatch at index %d (Host Mode) Relative Error: %.15e', [i, relative_error]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipMemcpy(devB, @host_B_initial_rhs[0], size_B, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(Double));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Double), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_dtrsm_batched_64(handle, side, uplo, transA, diag, M, N, PDouble(dev_alpha_mem),
    PPDouble(devA_array_device), lda, PPDouble(devB_array_device), ldb, batch_count);
  AssertEquals('rocblas_dtrsm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_B_rhs[0], devB, size_B, hipMemcpyDeviceToHost);

  for i := 0 to High(host_X_solution) do
  begin
    if host_X_solution[i] <> 0.0 then relative_error := Abs((host_X_solution[i] - host_B_rhs[i]) / host_X_solution[i]) else relative_error := Abs(host_B_rhs[i]);
    error_msg := Format('DTRSM_Batched_64 result mismatch at index %d (Device Mode) Relative Error: %.15e', [i, relative_error]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipFree(devA); hipFree(devB);
  hipFree(devA_array_device); hipFree(devB_array_device);
  hipFree(dev_alpha_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_CTRSM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  diag: rocblas_diagonal = rocblas_diagonal_non_unit;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  rel_epsilon: Single = 1e-4;
var
  size_A, size_B: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B: Int64;

  host_A, host_B_rhs, host_X_solution, host_B_initial_rhs: array of rocblas_float_complex;
  host_alpha, inv_alpha: rocblas_float_complex;

  devA, devB: hipDeviceptr_t;
  devA_array, devB_array: array of Procfloat_complex;
  devA_array_device, devB_array_device: hipDeviceptr_t;
  dev_alpha_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, term, one_complex: rocblas_float_complex;
  base_A, base_B: Int64;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;

  // Correctly calculate inv_alpha = 1.0 / host_alpha
  one_complex.x := 1.0;
  one_complex.y := 0.0;
  inv_alpha := one_complex;
  ComplexDiv(inv_alpha, host_alpha);

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B_rhs, total_elements_B);
  SetLength(host_X_solution, total_elements_B);
  SetLength(host_B_initial_rhs, total_elements_B);

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    for j := 0 to M - 1 do
      for i := 0 to M - 1 do
        if i > j then begin host_A[base_A + j * lda + i].x := (Random-0.5)/10; host_A[base_A + j * lda + i].y := (Random-0.5)/10; end
        else if i = j then begin host_A[base_A + j * lda + i].x := (Random-0.5)/5+1.0; host_A[base_A + j * lda + i].y := (Random-0.5)/5; end
        else begin host_A[base_A + j * lda + i].x := 0.0; host_A[base_A + j * lda + i].y := 0.0; end;
  end;
  for i := 0 to High(host_X_solution) do begin host_X_solution[i].x := (Random-0.5)/10; host_X_solution[i].y := (Random-0.5)/10; end;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        sum.x := 0; sum.y := 0;
        for k := 0 to M - 1 do
        begin
          term := host_A[base_A + k * lda + i];
          ComplexMul(term, host_X_solution[base_B + j * ldb + k]);
          ComplexAdd(sum, term);
        end;
        ComplexMul(sum, inv_alpha);
        host_B_rhs[base_B + j * ldb + i] := sum;
      end;
  end;

  System.Move(host_B_rhs[0], host_B_initial_rhs[0], total_elements_B * SizeOf(rocblas_float_complex));

  size_A := total_elements_A * SizeOf(rocblas_float_complex);
  size_B := total_elements_B * SizeOf(rocblas_float_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B_initial_rhs[0], size_B, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procfloat_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procfloat_complex(PtrUInt(devA) + (b * lda * M * SizeOf(rocblas_float_complex)));
    devB_array[b] := Procfloat_complex(PtrUInt(devB) + (b * ldb * N * SizeOf(rocblas_float_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_ctrsm_batched_64(handle, side, uplo, transA, diag, M, N, @host_alpha,
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb, batch_count);
  AssertEquals('rocblas_ctrsm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_B_rhs[0], devB, size_B, hipMemcpyDeviceToHost);

  for i := 0 to High(host_X_solution) do
  begin
    CheckEqualsRelative(host_X_solution[i].x, host_B_rhs[i].x, rel_epsilon, Format('CTRSM_Batched_64 result.x mismatch at index %d (Host Mode)', [i]));
    CheckEqualsRelative(host_X_solution[i].y, host_B_rhs[i].y, rel_epsilon, Format('CTRSM_Batched_64 result.y mismatch at index %d (Host Mode)', [i]));
  end;

  hipMemcpy(devB, @host_B_initial_rhs[0], size_B, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_float_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_ctrsm_batched_64(handle, side, uplo, transA, diag, M, N, Procfloat_complex(dev_alpha_mem),
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb, batch_count);
  AssertEquals('rocblas_ctrsm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_B_rhs[0], devB, size_B, hipMemcpyDeviceToHost);

  for i := 0 to High(host_X_solution) do
  begin
    CheckEqualsRelative(host_X_solution[i].x, host_B_rhs[i].x, rel_epsilon, Format('CTRSM_Batched_64 result.x mismatch at index %d (Device Mode)', [i]));
    CheckEqualsRelative(host_X_solution[i].y, host_B_rhs[i].y, rel_epsilon, Format('CTRSM_Batched_64 result.y mismatch at index %d (Device Mode)', [i]));
  end;

  hipFree(devA); hipFree(devB);
  hipFree(devA_array_device); hipFree(devB_array_device);
  hipFree(dev_alpha_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_ZTRSM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  diag: rocblas_diagonal = rocblas_diagonal_non_unit;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  rel_epsilon: Double = 1e-12;
var
  size_A, size_B: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B: Int64;

  host_A, host_B_rhs, host_X_solution, host_B_initial_rhs: array of rocblas_double_complex;
  host_alpha, inv_alpha, one_complex: rocblas_double_complex;

  devA, devB: hipDeviceptr_t;
  devA_array, devB_array: array of Procdouble_complex;
  devA_array_device, devB_array_device: hipDeviceptr_t;
  dev_alpha_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, term: rocblas_double_complex;
  base_A, base_B: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;
  one_complex.x := 1.0; one_complex.y := 0.0;
  inv_alpha := one_complex;
  ComplexDiv(inv_alpha, host_alpha);

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B_rhs, total_elements_B);
  SetLength(host_X_solution, total_elements_B);
  SetLength(host_B_initial_rhs, total_elements_B);

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    for j := 0 to M - 1 do
      for i := 0 to M - 1 do
        if i > j then begin host_A[base_A + j * lda + i].x := (Random-0.5)/10; host_A[base_A + j * lda + i].y := (Random-0.5)/10; end
        else if i = j then begin host_A[base_A + j * lda + i].x := (Random-0.5)/5+1.0; host_A[base_A + j * lda + i].y := (Random-0.5)/5; end
        else begin host_A[base_A + j * lda + i].x := 0.0; host_A[base_A + j * lda + i].y := 0.0; end;
  end;
  for i := 0 to High(host_X_solution) do begin host_X_solution[i].x := (Random-0.5)/10; host_X_solution[i].y := (Random-0.5)/10; end;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        sum.x := 0; sum.y := 0;
        for k := 0 to M - 1 do
        begin
          term := host_A[base_A + k * lda + i];
          ComplexMul(term, host_X_solution[base_B + j * ldb + k]);
          ComplexAdd(sum, term);
        end;
        ComplexMul(sum, inv_alpha);
        host_B_rhs[base_B + j * ldb + i] := sum;
      end;
  end;

  System.Move(host_B_rhs[0], host_B_initial_rhs[0], total_elements_B * SizeOf(rocblas_double_complex));

  size_A := total_elements_A * SizeOf(rocblas_double_complex);
  size_B := total_elements_B * SizeOf(rocblas_double_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B_initial_rhs[0], size_B, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procdouble_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + (b * lda * M * SizeOf(rocblas_double_complex)));
    devB_array[b] := Procdouble_complex(PtrUInt(devB) + (b * ldb * N * SizeOf(rocblas_double_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_ztrsm_batched_64(handle, side, uplo, transA, diag, M, N, @host_alpha,
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb, batch_count);
  AssertEquals('rocblas_ztrsm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_B_rhs[0], devB, size_B, hipMemcpyDeviceToHost);

  for i := 0 to High(host_X_solution) do
  begin
    if host_X_solution[i].x <> 0.0 then relative_error := Abs((host_X_solution[i].x - host_B_rhs[i].x) / host_X_solution[i].x) else relative_error := Abs(host_B_rhs[i].x);
    error_msg := Format('ZTRSM_Batched_64 result.x mismatch at index %d (Host Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
    if host_X_solution[i].y <> 0.0 then relative_error := Abs((host_X_solution[i].y - host_B_rhs[i].y) / host_X_solution[i].y) else relative_error := Abs(host_B_rhs[i].y);
    error_msg := Format('ZTRSM_Batched_64 result.y mismatch at index %d (Host Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipMemcpy(devB, @host_B_initial_rhs[0], size_B, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_double_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_ztrsm_batched_64(handle, side, uplo, transA, diag, M, N, Procdouble_complex(dev_alpha_mem),
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb, batch_count);
  AssertEquals('rocblas_ztrsm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_B_rhs[0], devB, size_B, hipMemcpyDeviceToHost);

  for i := 0 to High(host_X_solution) do
  begin
    if host_X_solution[i].x <> 0.0 then relative_error := Abs((host_X_solution[i].x - host_B_rhs[i].x) / host_X_solution[i].x) else relative_error := Abs(host_B_rhs[i].x);
    error_msg := Format('ZTRSM_Batched_64 result.x mismatch at index %d (Device Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
    if host_X_solution[i].y <> 0.0 then relative_error := Abs((host_X_solution[i].y - host_B_rhs[i].y) / host_X_solution[i].y) else relative_error := Abs(host_B_rhs[i].y);
    error_msg := Format('ZTRSM_Batched_64 result.y mismatch at index %d (Device Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipFree(devA); hipFree(devB);
  hipFree(devA_array_device); hipFree(devB_array_device);
  hipFree(dev_alpha_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_SSYRK_Batched_64;
const
  N: Int64 = 5;
  K: Int64 = 4;
  batch_count: Int64 = 2;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  lda: Int64 = 5; // if transA=N, lda >= N
  ldc: Int64 = 5; // ldc >= N
  ALPHA_VALUE: Single = 1.5;
  BETA_VALUE: Single = 2.5;
  rel_epsilon: Single = 1e-5;
var
  size_A, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_C: Int64;

  host_A, host_C, host_C_initial, host_C_expected: array of Single;
  host_alpha, host_beta: Single;

  devA, devC: hipDeviceptr_t;
  devA_array, devC_array: array of PSingle;
  devA_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum: Single;
  idx, base_A, base_C: Int64;
begin
  host_alpha := ALPHA_VALUE;
  host_beta := BETA_VALUE;

  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_C, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do host_A[i] := (Random - 0.5) / 10;
  for i := 0 to High(host_C) do host_C[i] := (Random - 0.5) / 10;
  System.Move(host_C[0], host_C_initial[0], total_elements_C * SizeOf(Single));

  // CPU reference: C = beta*C_initial + alpha*A*A^T
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if i >= j then // Calculate lower triangle
        begin
          sum := 0.0;
          for l := 0 to K - 1 do
            sum := sum + host_A[base_A + l * lda + i] * host_A[base_A + l * lda + j];
          host_C_expected[idx] := host_beta * host_C_initial[idx] + host_alpha * sum;
        end
        else // Upper triangle is not touched by rocBLAS, so expected is initial
        begin
          host_C_expected[idx] := host_C_initial[idx];
        end;
      end;
  end;

  size_A := total_elements_A * SizeOf(Single);
  size_C := total_elements_C * SizeOf(Single);
  hipMalloc(devA, size_A);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(PSingle));
  hipMalloc(devC_array_device, batch_count * SizeOf(PSingle));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PSingle(PtrUInt(devA) + (b * lda * K * SizeOf(Single)));
    devC_array[b] := PSingle(PtrUInt(devC) + (b * ldc * N * SizeOf(Single)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_ssyrk_batched_64(handle, uplo, transA, N, K, @host_alpha,
    PPSingle(devA_array_device), lda, @host_beta, PPSingle(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_ssyrk_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do // Check only lower triangle
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx], host_C[idx], rel_epsilon, Format('SSYRK_Batched_64 result mismatch at index %d (Host Mode)', [idx]));
      end;
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(Single));
  hipMalloc(dev_beta_mem, SizeOf(Single));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(Single), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_ssyrk_batched_64(handle, uplo, transA, N, K, PSingle(dev_alpha_mem),
    PPSingle(devA_array_device), lda, PSingle(dev_beta_mem), PPSingle(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_ssyrk_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do // Check only lower triangle
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx], host_C[idx], rel_epsilon, Format('SSYRK_Batched_64 result mismatch at index %d (Device Mode)', [idx]));
      end;
  end;

  hipFree(devA); hipFree(devC);
  hipFree(devA_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_DSYRK_Batched_64;
const
  N: Int64 = 5;
  K: Int64 = 4;
  batch_count: Int64 = 2;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  lda: Int64 = 5;
  ldc: Int64 = 5;
  ALPHA_VALUE: Double = 1.5;
  BETA_VALUE: Double = 2.5;
  rel_epsilon: Double = 1e-12;
var
  size_A, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_C: Int64;

  host_A, host_C_result, host_C_initial, host_C_expected: array of Double;
  host_alpha, host_beta: Double;

  devA, devC: hipDeviceptr_t;
  devA_array, devC_array: array of PDouble;
  devA_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum: Double;
  idx, base_A, base_C: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha := ALPHA_VALUE;
  host_beta := BETA_VALUE;

  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do host_A[i] := (Random - 0.5) / 10.0;
  for i := 0 to High(host_C_initial) do host_C_initial[i] := (Random - 0.5) / 10.0;

  // CPU reference: C = beta*C_initial + alpha*A*A^T
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if i >= j then // Calculate lower triangle
        begin
          sum := 0.0;
          for l := 0 to K - 1 do
            sum := sum + host_A[base_A + l * lda + i] * host_A[base_A + l * lda + j];
          host_C_expected[idx] := host_beta * host_C_initial[idx] + host_alpha * sum;
        end
        else // Upper triangle is untouched
        begin
          host_C_expected[idx] := host_C_initial[idx];
        end;
      end;
  end;

  size_A := total_elements_A * SizeOf(Double);
  size_C := total_elements_C * SizeOf(Double);
  hipMalloc(devA, size_A);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(PDouble));
  hipMalloc(devC_array_device, batch_count * SizeOf(PDouble));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PDouble(PtrUInt(devA) + (b * lda * K * SizeOf(Double)));
    devC_array[b] := PDouble(PtrUInt(devC) + (b * ldc * N * SizeOf(Double)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_dsyrk_batched_64(handle, uplo, transA, N, K, @host_alpha,
    PPDouble(devA_array_device), lda, @host_beta, PPDouble(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_dsyrk_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if host_C_expected[idx] <> 0.0 then relative_error := Abs((host_C_expected[idx] - host_C_result[idx]) / host_C_expected[idx]) else relative_error := Abs(host_C_result[idx]);
        error_msg := Format('DSYRK_Batched_64 result mismatch at index %d (Host Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
      end;
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(Double));
  hipMalloc(dev_beta_mem, SizeOf(Double));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(Double), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_dsyrk_batched_64(handle, uplo, transA, N, K, PDouble(dev_alpha_mem),
    PPDouble(devA_array_device), lda, PDouble(dev_beta_mem), PPDouble(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_dsyrk_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if host_C_expected[idx] <> 0.0 then relative_error := Abs((host_C_expected[idx] - host_C_result[idx]) / host_C_expected[idx]) else relative_error := Abs(host_C_result[idx]);
        error_msg := Format('DSYRK_Batched_64 result mismatch at index %d (Device Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
      end;
  end;

  hipFree(devA); hipFree(devC);
  hipFree(devA_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_CSYRK_Batched_64;
const
  N: Int64 = 5;
  K: Int64 = 4;
  batch_count: Int64 = 2;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  lda: Int64 = 5;
  ldc: Int64 = 5;
  rel_epsilon: Single = 1e-4;
var
  size_A, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_C: Int64;

  host_A, host_C_result, host_C_initial, host_C_expected: array of rocblas_float_complex;
  host_alpha, host_beta: rocblas_float_complex;

  devA, devC: hipDeviceptr_t;
  devA_array, devC_array: array of Procfloat_complex;
  devA_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, term: rocblas_float_complex;
  idx, base_A, base_C: Int64;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;
  host_beta.x  := 2.5; host_beta.y  :=  1.0;

  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do begin host_A[i].x := (Random-0.5)/10; host_A[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C_initial) do begin host_C_initial[i].x := (Random-0.5)/10; host_C_initial[i].y := (Random-0.5)/10; end;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if i >= j then // Lower triangle
        begin
          sum.x := 0.0; sum.y := 0.0;
          for l := 0 to K - 1 do
          begin
            term := host_A[base_A + l * lda + i];
            ComplexMul(term, host_A[base_A + l * lda + j]); // A(i,l)*A(j,l)
            ComplexAdd(sum, term);
          end;
          term := host_C_initial[idx];
          ComplexMul(term, host_beta); // beta * C_initial
          ComplexMul(sum, host_alpha); // alpha * sum
          ComplexAdd(term, sum);
          host_C_expected[idx] := term;
        end
        else
        begin
          host_C_expected[idx] := host_C_initial[idx];
        end;
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_float_complex);
  size_C := total_elements_C * SizeOf(rocblas_float_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procfloat_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procfloat_complex(PtrUInt(devA) + (b * lda * K * SizeOf(rocblas_float_complex)));
    devC_array[b] := Procfloat_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_float_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_csyrk_batched_64(handle, uplo, transA, N, K, @host_alpha,
    PPProcfloat_complex(devA_array_device), lda, @host_beta, PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_csyrk_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx].x, host_C_result[idx].x, rel_epsilon, Format('CSYRK_Batched_64 result.x mismatch at index %d (Host Mode)', [idx]));
        CheckEqualsRelative(host_C_expected[idx].y, host_C_result[idx].y, rel_epsilon, Format('CSYRK_Batched_64 result.y mismatch at index %d (Host Mode)', [idx]));
      end;
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_float_complex));
  hipMalloc(dev_beta_mem, SizeOf(rocblas_float_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_csyrk_batched_64(handle, uplo, transA, N, K, Procfloat_complex(dev_alpha_mem),
    PPProcfloat_complex(devA_array_device), lda, Procfloat_complex(dev_beta_mem), PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_csyrk_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx].x, host_C_result[idx].x, rel_epsilon, Format('CSYRK_Batched_64 result.x mismatch at index %d (Device Mode)', [idx]));
        CheckEqualsRelative(host_C_expected[idx].y, host_C_result[idx].y, rel_epsilon, Format('CSYRK_Batched_64 result.y mismatch at index %d (Device Mode)', [idx]));
      end;
  end;

  hipFree(devA); hipFree(devC);
  hipFree(devA_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_ZSYRK_Batched_64;
const
  N: Int64 = 5;
  K: Int64 = 4;
  batch_count: Int64 = 2;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  lda: Int64 = 5;
  ldc: Int64 = 5;
  rel_epsilon: Double = 1e-12;
var
  size_A, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_C: Int64;

  host_A, host_C_result, host_C_initial, host_C_expected: array of rocblas_double_complex;
  host_alpha, host_beta: rocblas_double_complex;

  devA, devC: hipDeviceptr_t;
  devA_array, devC_array: array of Procdouble_complex;
  devA_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, term: rocblas_double_complex;
  idx, base_A, base_C: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;
  host_beta.x  := 2.5; host_beta.y  :=  1.0;

  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do begin host_A[i].x := (Random-0.5)/10; host_A[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C_initial) do begin host_C_initial[i].x := (Random-0.5)/10; host_C_initial[i].y := (Random-0.5)/10; end;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if i >= j then // Lower triangle
        begin
          sum.x := 0.0; sum.y := 0.0;
          for l := 0 to K - 1 do
          begin
            term := host_A[base_A + l * lda + i];
            ComplexMul(term, host_A[base_A + l * lda + j]);
            ComplexAdd(sum, term);
          end;
          term := host_C_initial[idx];
          ComplexMul(term, host_beta);
          ComplexMul(sum, host_alpha);
          ComplexAdd(term, sum);
          host_C_expected[idx] := term;
        end
        else
        begin
          host_C_expected[idx] := host_C_initial[idx];
        end;
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_double_complex);
  size_C := total_elements_C * SizeOf(rocblas_double_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procdouble_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + (b * lda * K * SizeOf(rocblas_double_complex)));
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_double_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_zsyrk_batched_64(handle, uplo, transA, N, K, @host_alpha,
    PPProcdouble_complex(devA_array_device), lda, @host_beta, PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zsyrk_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if host_C_expected[idx].x <> 0.0 then relative_error := Abs((host_C_expected[idx].x - host_C_result[idx].x) / host_C_expected[idx].x) else relative_error := Abs(host_C_result[idx].x);
        error_msg := Format('ZSYRK_Batched_64 result.x mismatch at index %d (Host Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
        if host_C_expected[idx].y <> 0.0 then relative_error := Abs((host_C_expected[idx].y - host_C_result[idx].y) / host_C_expected[idx].y) else relative_error := Abs(host_C_result[idx].y);
        error_msg := Format('ZSYRK_Batched_64 result.y mismatch at index %d (Host Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
      end;
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_double_complex));
  hipMalloc(dev_beta_mem, SizeOf(rocblas_double_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_zsyrk_batched_64(handle, uplo, transA, N, K, Procdouble_complex(dev_alpha_mem),
    PPProcdouble_complex(devA_array_device), lda, Procdouble_complex(dev_beta_mem), PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zsyrk_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if host_C_expected[idx].x <> 0.0 then relative_error := Abs((host_C_expected[idx].x - host_C_result[idx].x) / host_C_expected[idx].x) else relative_error := Abs(host_C_result[idx].x);
        error_msg := Format('ZSYRK_Batched_64 result.x mismatch at index %d (Device Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
        if host_C_expected[idx].y <> 0.0 then relative_error := Abs((host_C_expected[idx].y - host_C_result[idx].y) / host_C_expected[idx].y) else relative_error := Abs(host_C_result[idx].y);
        error_msg := Format('ZSYRK_Batched_64 result.y mismatch at index %d (Device Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
      end;
  end;

  hipFree(devA); hipFree(devC);
  hipFree(devA_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_CHERK_Batched_64;
const
  N: Int64 = 5;
  K: Int64 = 4;
  batch_count: Int64 = 2;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  lda: Int64 = 5;
  ldc: Int64 = 5;
  ALPHA_VALUE: Single = 1.5;
  BETA_VALUE: Single = 2.5;
  rel_epsilon: Single = 1e-4;
var
  size_A, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_C: Int64;

  host_A, host_C_result, host_C_initial, host_C_expected: array of rocblas_float_complex;
  host_alpha, host_beta: Single;

  devA, devC: hipDeviceptr_t;
  devA_array, devC_array: array of Procfloat_complex;
  devA_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum: rocblas_float_complex;
  term, conj_A: rocblas_float_complex;
  idx, base_A, base_C: Int64;
begin
  host_alpha := ALPHA_VALUE;
  host_beta := BETA_VALUE;

  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do begin host_A[i].x := (Random-0.5)/10; host_A[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C_initial) do begin host_C_initial[i].x := (Random-0.5)/10; host_C_initial[i].y := (Random-0.5)/10; end;

  // CPU reference: C = beta*C_initial + alpha*A*A^H
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if i >= j then // Lower triangle
        begin
          sum.x := 0.0; sum.y := 0.0;
          for l := 0 to K - 1 do
          begin
            conj_A := host_A[base_A + l * lda + j];
            conj_A.y := -conj_A.y; // Conjugate
            term := host_A[base_A + l * lda + i];
            ComplexMul(term, conj_A); // A(i,l) * conj(A(j,l))
            ComplexAdd(sum, term);
          end;
          // C_exp = (beta * C_initial) + (alpha * sum)
          host_C_expected[idx].x := host_beta * host_C_initial[idx].x + host_alpha * sum.x;
          host_C_expected[idx].y := host_beta * host_C_initial[idx].y + host_alpha * sum.y;
          if i = j then host_C_expected[idx].y := 0.0; // Diagonal of Hermitian is real
        end
        else
        begin
          host_C_expected[idx] := host_C_initial[idx];
        end;
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_float_complex);
  size_C := total_elements_C * SizeOf(rocblas_float_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procfloat_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procfloat_complex(PtrUInt(devA) + (b * lda * K * SizeOf(rocblas_float_complex)));
    devC_array[b] := Procfloat_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_float_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_cherk_batched_64(handle, uplo, transA, N, K, @host_alpha,
    PPProcfloat_complex(devA_array_device), lda, @host_beta, PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_cherk_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx].x, host_C_result[idx].x, rel_epsilon, Format('CHERK_Batched_64 result.x mismatch at index %d (Host Mode)', [idx]));
        CheckEqualsRelative(host_C_expected[idx].y, host_C_result[idx].y, rel_epsilon, Format('CHERK_Batched_64 result.y mismatch at index %d (Host Mode)', [idx]));
      end;
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(Single));
  hipMalloc(dev_beta_mem, SizeOf(Single));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(Single), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_cherk_batched_64(handle, uplo, transA, N, K, PSingle(dev_alpha_mem),
    PPProcfloat_complex(devA_array_device), lda, PSingle(dev_beta_mem), PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_cherk_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx].x, host_C_result[idx].x, rel_epsilon, Format('CHERK_Batched_64 result.x mismatch at index %d (Device Mode)', [idx]));
        CheckEqualsRelative(host_C_expected[idx].y, host_C_result[idx].y, rel_epsilon, Format('CHERK_Batched_64 result.y mismatch at index %d (Device Mode)', [idx]));
      end;
  end;

  hipFree(devA); hipFree(devC);
  hipFree(devA_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_ZHERK_Batched_64;
const
  N: Int64 = 5;
  K: Int64 = 4;
  batch_count: Int64 = 2;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  lda: Int64 = 5;
  ldc: Int64 = 5;
  ALPHA_VALUE: Double = 1.5;
  BETA_VALUE: Double = 2.5;
  rel_epsilon: Double = 1e-12;
var
  size_A, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_C: Int64;

  host_A, host_C_result, host_C_initial, host_C_expected: array of rocblas_double_complex;
  host_alpha, host_beta: Double;

  devA, devC: hipDeviceptr_t;
  devA_array, devC_array: array of Procdouble_complex;
  devA_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum: rocblas_double_complex;
  term, conj_A: rocblas_double_complex;
  idx, base_A, base_C: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha := ALPHA_VALUE;
  host_beta := BETA_VALUE;

  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do begin host_A[i].x := (Random-0.5)/10; host_A[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C_initial) do begin host_C_initial[i].x := (Random-0.5)/10; host_C_initial[i].y := (Random-0.5)/10; end;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if i >= j then // Lower triangle
        begin
          sum.x := 0.0; sum.y := 0.0;
          for l := 0 to K - 1 do
          begin
            conj_A := host_A[base_A + l * lda + j];
            conj_A.y := -conj_A.y;
            term := host_A[base_A + l * lda + i];
            ComplexMul(term, conj_A);
            ComplexAdd(sum, term);
          end;
          host_C_expected[idx].x := host_beta * host_C_initial[idx].x + host_alpha * sum.x;
          host_C_expected[idx].y := host_beta * host_C_initial[idx].y + host_alpha * sum.y;
          if i = j then host_C_expected[idx].y := 0.0;
        end
        else
        begin
          host_C_expected[idx] := host_C_initial[idx];
        end;
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_double_complex);
  size_C := total_elements_C * SizeOf(rocblas_double_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procdouble_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + (b * lda * K * SizeOf(rocblas_double_complex)));
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_double_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_zherk_batched_64(handle, uplo, transA, N, K, @host_alpha,
    PPProcdouble_complex(devA_array_device), lda, @host_beta, PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zherk_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if host_C_expected[idx].x <> 0.0 then relative_error := Abs((host_C_expected[idx].x - host_C_result[idx].x) / host_C_expected[idx].x) else relative_error := Abs(host_C_result[idx].x);
        error_msg := Format('ZHERK_Batched_64 result.x mismatch at index %d (Host Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
        if host_C_expected[idx].y <> 0.0 then relative_error := Abs((host_C_expected[idx].y - host_C_result[idx].y) / host_C_expected[idx].y) else relative_error := Abs(host_C_result[idx].y);
        error_msg := Format('ZHERK_Batched_64 result.y mismatch at index %d (Host Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
      end;
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(Double));
  hipMalloc(dev_beta_mem, SizeOf(Double));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(Double), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_zherk_batched_64(handle, uplo, transA, N, K, PDouble(dev_alpha_mem),
    PPProcdouble_complex(devA_array_device), lda, PDouble(dev_beta_mem), PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zherk_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if host_C_expected[idx].x <> 0.0 then relative_error := Abs((host_C_expected[idx].x - host_C_result[idx].x) / host_C_expected[idx].x) else relative_error := Abs(host_C_result[idx].x);
        error_msg := Format('ZHERK_Batched_64 result.x mismatch at index %d (Device Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
        if host_C_expected[idx].y <> 0.0 then relative_error := Abs((host_C_expected[idx].y - host_C_result[idx].y) / host_C_expected[idx].y) else relative_error := Abs(host_C_result[idx].y);
        error_msg := Format('ZHERK_Batched_64 result.y mismatch at index %d (Device Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
      end;
  end;

  hipFree(devA); hipFree(devC);
  hipFree(devA_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_SSyr2k_Batched_64;
const
  N: Int64 = 5;
  K: Int64 = 4;
  batch_count: Int64 = 2;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  lda: Int64 = 5; // if transA=N, lda >= N
  ldb: Int64 = 5; // if transA=N, ldb >= N
  ldc: Int64 = 5; // ldc >= N
  ALPHA_VALUE: Single = 1.5;
  BETA_VALUE: Single = 2.5;
  rel_epsilon: Single = 1e-5;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_initial, host_C_expected: array of Single;
  host_alpha, host_beta: Single;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of PSingle;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum_ABt, sum_BAt: Single;
  idx, base_A, base_B, base_C: Int64;
begin
  host_alpha := ALPHA_VALUE;
  host_beta := BETA_VALUE;

  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_B := Int64(batch_count) * ldb * K;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do host_A[i] := (Random - 0.5) / 10;
  for i := 0 to High(host_B) do host_B[i] := (Random - 0.5) / 10;
  for i := 0 to High(host_C_initial) do host_C_initial[i] := (Random - 0.5) / 10;

  // CPU reference: C = beta*C_initial + alpha*(A*B^T + B*A^T)
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_B := b * ldb * K;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if i >= j then // Lower triangle
        begin
          sum_ABt := 0.0;
          sum_BAt := 0.0;
          for l := 0 to K - 1 do
          begin
            sum_ABt := sum_ABt + host_A[base_A + l * lda + i] * host_B[base_A + l * ldb + j];
            sum_BAt := sum_BAt + host_B[base_B + l * ldb + i] * host_A[base_A + l * lda + j];
          end;
          host_C_expected[idx] := host_beta * host_C_initial[idx] + host_alpha * (sum_ABt + sum_BAt);
        end
        else
        begin
          host_C_expected[idx] := host_C_initial[idx];
        end;
      end;
  end;

  size_A := total_elements_A * SizeOf(Single);
  size_B := total_elements_B * SizeOf(Single);
  size_C := total_elements_C * SizeOf(Single);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(PSingle));
  hipMalloc(devB_array_device, batch_count * SizeOf(PSingle));
  hipMalloc(devC_array_device, batch_count * SizeOf(PSingle));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PSingle(PtrUInt(devA) + (b * lda * K * SizeOf(Single)));
    devB_array[b] := PSingle(PtrUInt(devB) + (b * ldb * K * SizeOf(Single)));
    devC_array[b] := PSingle(PtrUInt(devC) + (b * ldc * N * SizeOf(Single)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(PSingle), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_ssyr2k_batched_64(handle, uplo, transA, N, K, @host_alpha,
    PPSingle(devA_array_device), lda, PPSingle(devB_array_device), ldb, @host_beta,
    PPSingle(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_ssyr2k_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx], host_C_result[idx], rel_epsilon, Format('SSYR2K_Batched_64 result mismatch at index %d (Host Mode)', [idx]));
      end;
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(Single));
  hipMalloc(dev_beta_mem, SizeOf(Single));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(Single), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_ssyr2k_batched_64(handle, uplo, transA, N, K, PSingle(dev_alpha_mem),
    PPSingle(devA_array_device), lda, PPSingle(devB_array_device), ldb, PSingle(dev_beta_mem),
    PPSingle(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_ssyr2k_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx], host_C_result[idx], rel_epsilon, Format('SSYR2K_Batched_64 result mismatch at index %d (Device Mode)', [idx]));
      end;
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_DSyr2k_Batched_64;
const
  N: Int64 = 5;
  K: Int64 = 4;
  batch_count: Int64 = 2;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ldc: Int64 = 5;
  ALPHA_VALUE: Double = 1.5;
  BETA_VALUE: Double = 2.5;
  rel_epsilon: Double = 1e-12;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_initial, host_C_expected: array of Double;
  host_alpha, host_beta: Double;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of PDouble;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum_ABt, sum_BAt: Double;
  idx, base_A, base_B, base_C: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha := ALPHA_VALUE;
  host_beta := BETA_VALUE;

  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_B := Int64(batch_count) * ldb * K;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do host_A[i] := (Random - 0.5) / 10.0;
  for i := 0 to High(host_B) do host_B[i] := (Random - 0.5) / 10.0;
  for i := 0 to High(host_C_initial) do host_C_initial[i] := (Random - 0.5) / 10.0;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_B := b * ldb * K;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if i >= j then
        begin
          sum_ABt := 0.0;
          sum_BAt := 0.0;
          for l := 0 to K - 1 do
          begin
            sum_ABt := sum_ABt + host_A[base_A + l * lda + i] * host_B[base_A + l * ldb + j];
            sum_BAt := sum_BAt + host_B[base_B + l * ldb + i] * host_A[base_A + l * lda + j];
          end;
          host_C_expected[idx] := host_beta * host_C_initial[idx] + host_alpha * (sum_ABt + sum_BAt);
        end
        else
        begin
          host_C_expected[idx] := host_C_initial[idx];
        end;
      end;
  end;

  size_A := total_elements_A * SizeOf(Double);
  size_B := total_elements_B * SizeOf(Double);
  size_C := total_elements_C * SizeOf(Double);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(PDouble));
  hipMalloc(devB_array_device, batch_count * SizeOf(PDouble));
  hipMalloc(devC_array_device, batch_count * SizeOf(PDouble));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PDouble(PtrUInt(devA) + (b * lda * K * SizeOf(Double)));
    devB_array[b] := PDouble(PtrUInt(devB) + (b * ldb * K * SizeOf(Double)));
    devC_array[b] := PDouble(PtrUInt(devC) + (b * ldc * N * SizeOf(Double)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(PDouble), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_dsyr2k_batched_64(handle, uplo, transA, N, K, @host_alpha,
    PPDouble(devA_array_device), lda, PPDouble(devB_array_device), ldb, @host_beta,
    PPDouble(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_dsyr2k_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if host_C_expected[idx] <> 0.0 then relative_error := Abs((host_C_expected[idx] - host_C_result[idx]) / host_C_expected[idx]) else relative_error := Abs(host_C_result[idx]);
        error_msg := Format('DSYR2K_Batched_64 result mismatch at index %d (Host Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
      end;
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(Double));
  hipMalloc(dev_beta_mem, SizeOf(Double));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(Double), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_dsyr2k_batched_64(handle, uplo, transA, N, K, PDouble(dev_alpha_mem),
    PPDouble(devA_array_device), lda, PPDouble(devB_array_device), ldb, PDouble(dev_beta_mem),
    PPDouble(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_dsyr2k_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if host_C_expected[idx] <> 0.0 then relative_error := Abs((host_C_expected[idx] - host_C_result[idx]) / host_C_expected[idx]) else relative_error := Abs(host_C_result[idx]);
        error_msg := Format('DSYR2K_Batched_64 result mismatch at index %d (Device Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
      end;
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_CSyr2k_Batched_64;
const
  N: Int64 = 5;
  K: Int64 = 4;
  batch_count: Int64 = 2;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ldc: Int64 = 5;
  rel_epsilon: Single = 1e-4;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;
  host_A, host_B, host_C_result, host_C_initial, host_C_expected: array of rocblas_float_complex;
  host_alpha, host_beta: rocblas_float_complex;
  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of Procfloat_complex;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;
  blas_status: rocblas_status;
  sum, term: rocblas_float_complex;
  idx, base_A, base_B, base_C: Int64;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;
  host_beta.x := 2.5; host_beta.y := 1.0;

  total_elements_A := batch_count * lda * K;
  total_elements_B := batch_count * ldb * K;
  total_elements_C := batch_count * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do begin host_A[i].x := (Random-0.5)/10; host_A[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_B) do begin host_B[i].x := (Random-0.5)/10; host_B[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C_initial) do begin host_C_initial[i].x := (Random-0.5)/10; host_C_initial[i].y := (Random-0.5)/10; end;

  // CPU reference: C = beta*C_initial + alpha*(A*B^T + B*A^T), symmetric (no conjugation)
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_B := b * ldb * K;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if i >= j then // Lower triangle
        begin
          sum.x := 0.0; sum.y := 0.0;
          for l := 0 to K - 1 do
          begin
            term := host_A[base_A + l * lda + i];
            ComplexMul(term, host_B[base_B + l * ldb + j]); // A(i,l)*B(j,l)
            ComplexAdd(sum, term);
            term := host_B[base_B + l * ldb + i];
            ComplexMul(term, host_A[base_A + l * lda + j]); // B(i,l)*A(j,l)
            ComplexAdd(sum, term);
          end;
          ComplexMul(sum, host_alpha);
          term := host_C_initial[idx];
          ComplexMul(term, host_beta);   // beta * C_initial
          ComplexAdd(term, sum);
          host_C_expected[idx] := term;
        end
        else
          host_C_expected[idx] := host_C_initial[idx];
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_float_complex);
  size_B := total_elements_B * SizeOf(rocblas_float_complex);
  size_C := total_elements_C * SizeOf(rocblas_float_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procfloat_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procfloat_complex(PtrUInt(devA) + b * lda * K * SizeOf(rocblas_float_complex));
    devB_array[b] := Procfloat_complex(PtrUInt(devB) + b * ldb * K * SizeOf(rocblas_float_complex));
    devC_array[b] := Procfloat_complex(PtrUInt(devC) + b * ldc * N * SizeOf(rocblas_float_complex));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_csyr2k_batched_64(handle, uplo, transA, N, K, @host_alpha,
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb,
    @host_beta, PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_csyr2k_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  AssertEquals('hipDeviceSynchronize post rocblas_csyr2k_batched_64', Integer(HIP_SUCCESS), Integer(hipDeviceSynchronize));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx].x, host_C_result[idx].x, rel_epsilon, Format('CSYR2K_Batched_64 result.x mismatch at index %d (Host Mode)', [idx]));
        CheckEqualsRelative(host_C_expected[idx].y, host_C_result[idx].y, rel_epsilon, Format('CSYR2K_Batched_64 result.y mismatch at index %d (Host Mode)', [idx]));
      end;
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_float_complex));
  hipMalloc(dev_beta_mem, SizeOf(rocblas_float_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_csyr2k_batched_64(handle, uplo, transA, N, K, Procfloat_complex(dev_alpha_mem),
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb,
    Procfloat_complex(dev_beta_mem), PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_csyr2k_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx].x, host_C_result[idx].x, rel_epsilon, Format('CSYR2K_Batched_64 result.x mismatch at index %d (Device Mode)', [idx]));
        CheckEqualsRelative(host_C_expected[idx].y, host_C_result[idx].y, rel_epsilon, Format('CSYR2K_Batched_64 result.y mismatch at index %d (Device Mode)', [idx]));
      end;
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_ZSyr2k_Batched_64;
const
  N: Int64 = 5;
  K: Int64 = 4;
  batch_count: Int64 = 2;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ldc: Int64 = 5;
  rel_epsilon: Double = 1e-12;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;
  host_A, host_B, host_C_result, host_C_initial, host_C_expected: array of rocblas_double_complex;
  host_alpha, host_beta: rocblas_double_complex;
  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of Procdouble_complex;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;
  blas_status: rocblas_status;
  sum, term: rocblas_double_complex;
  idx, base_A, base_B, base_C: Int64;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;
  host_beta.x := 2.5; host_beta.y := 1.0;

  total_elements_A := batch_count * lda * K;
  total_elements_B := batch_count * ldb * K;
  total_elements_C := batch_count * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do begin host_A[i].x := (Random-0.5)/10; host_A[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_B) do begin host_B[i].x := (Random-0.5)/10; host_B[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C_initial) do begin host_C_initial[i].x := (Random-0.5)/10; host_C_initial[i].y := (Random-0.5)/10; end;

  // CPU reference: C = beta*C_initial + alpha*(A*B^T + B*A^T), symmetric (no conjugation)
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_B := b * ldb * K;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if i >= j then // Lower triangle
        begin
          sum.x := 0.0; sum.y := 0.0;
          for l := 0 to K - 1 do
          begin
            term := host_A[base_A + l * lda + i];
            ComplexMul(term, host_B[base_B + l * ldb + j]); // A(i,l)*B(j,l)
            ComplexAdd(sum, term);
            term := host_B[base_B + l * ldb + i];
            ComplexMul(term, host_A[base_A + l * lda + j]); // B(i,l)*A(j,l)
            ComplexAdd(sum, term);
          end;
          ComplexMul(sum, host_alpha);
          term := host_C_initial[idx];
          ComplexMul(term, host_beta);   // beta * C_initial
          ComplexAdd(term, sum);
          host_C_expected[idx] := term;
        end
        else
          host_C_expected[idx] := host_C_initial[idx];
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_double_complex);
  size_B := total_elements_B * SizeOf(rocblas_double_complex);
  size_C := total_elements_C * SizeOf(rocblas_double_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procdouble_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + b * lda * K * SizeOf(rocblas_double_complex));
    devB_array[b] := Procdouble_complex(PtrUInt(devB) + b * ldb * K * SizeOf(rocblas_double_complex));
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + b * ldc * N * SizeOf(rocblas_double_complex));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_zsyr2k_batched_64(handle, uplo, transA, N, K, @host_alpha,
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb,
    @host_beta, PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zsyr2k_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  AssertEquals('hipDeviceSynchronize post rocblas_zsyr2k_batched_64', Integer(HIP_SUCCESS), Integer(hipDeviceSynchronize));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx].x, host_C_result[idx].x, rel_epsilon, Format('ZSYR2K_Batched_64 result.x mismatch at index %d (Host Mode)', [idx]));
        CheckEqualsRelative(host_C_expected[idx].y, host_C_result[idx].y, rel_epsilon, Format('ZSYR2K_Batched_64 result.y mismatch at index %d (Host Mode)', [idx]));
      end;
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_double_complex));
  hipMalloc(dev_beta_mem, SizeOf(rocblas_double_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_zsyr2k_batched_64(handle, uplo, transA, N, K, Procdouble_complex(dev_alpha_mem),
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb,
    Procdouble_complex(dev_beta_mem), PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zsyr2k_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx].x, host_C_result[idx].x, rel_epsilon, Format('ZSYR2K_Batched_64 result.x mismatch at index %d (Device Mode)', [idx]));
        CheckEqualsRelative(host_C_expected[idx].y, host_C_result[idx].y, rel_epsilon, Format('ZSYR2K_Batched_64 result.y mismatch at index %d (Device Mode)', [idx]));
      end;
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_CHER2K_Batched_64;
const
  N: Int64 = 5;
  K: Int64 = 4;
  batch_count: Int64 = 2;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ldc: Int64 = 5;
  BETA_VALUE: Single = 2.5;
  rel_epsilon: Single = 1e-4;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_initial, host_C_expected: array of rocblas_float_complex;
  host_alpha, conj_alpha: rocblas_float_complex;
  host_beta: Single;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of Procfloat_complex;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum1, sum2, term1, term2, conj_A, conj_B: rocblas_float_complex;
  idx, base_A, base_B, base_C: Int64;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;
  conj_alpha.x := host_alpha.x; conj_alpha.y := -host_alpha.y;
  host_beta := BETA_VALUE;

  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_B := Int64(batch_count) * ldb * K;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do begin host_A[i].x := (Random-0.5)/10; host_A[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_B) do begin host_B[i].x := (Random-0.5)/10; host_B[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C_initial) do begin host_C_initial[i].x := (Random-0.5)/10; host_C_initial[i].y := (Random-0.5)/10; end;

  // CPU reference: C = beta*C_initial + alpha*A*B^H + conj(alpha)*B*A^H
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_B := b * ldb * K;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if i >= j then // Lower triangle
        begin
          sum1.x := 0; sum1.y := 0;
          sum2.x := 0; sum2.y := 0;
          for l := 0 to K - 1 do
          begin
            // alpha * A(i,l)*B(j,l)^H
            conj_B := host_B[base_B + l*ldb + j]; conj_B.y := -conj_B.y;
            term1 := host_A[base_A + l*lda + i];
            ComplexMul(term1, conj_B);
            ComplexAdd(sum1, term1);
            // conj(alpha) * B(i,l)*A(j,l)^H
            conj_A := host_A[base_A + l*lda + j]; conj_A.y := -conj_A.y;
            term2 := host_B[base_B + l*ldb + i];
            ComplexMul(term2, conj_A);
            ComplexAdd(sum2, term2);
          end;
          ComplexMul(sum1, host_alpha);
          ComplexMul(sum2, conj_alpha);
          host_C_expected[idx].x := host_beta * host_C_initial[idx].x + sum1.x + sum2.x;
          host_C_expected[idx].y := host_beta * host_C_initial[idx].y + sum1.y + sum2.y;
          if i = j then host_C_expected[idx].y := 0.0;
        end
        else
          host_C_expected[idx] := host_C_initial[idx];
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_float_complex);
  size_B := total_elements_B * SizeOf(rocblas_float_complex);
  size_C := total_elements_C * SizeOf(rocblas_float_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procfloat_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procfloat_complex(PtrUInt(devA) + (b * lda * K * SizeOf(rocblas_float_complex)));
    devB_array[b] := Procfloat_complex(PtrUInt(devB) + (b * ldb * K * SizeOf(rocblas_float_complex)));
    devC_array[b] := Procfloat_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_float_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_cher2k_batched_64(handle, uplo, transA, N, K, @host_alpha,
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb,
    @host_beta, PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_cher2k_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx].x, host_C_result[idx].x, rel_epsilon, Format('CHER2K_Batched_64 result.x mismatch at index %d (Host Mode)', [idx]));
        CheckEqualsRelative(host_C_expected[idx].y, host_C_result[idx].y, rel_epsilon, Format('CHER2K_Batched_64 result.y mismatch at index %d (Host Mode)', [idx]));
      end;
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_float_complex));
  hipMalloc(dev_beta_mem, SizeOf(Single));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(Single), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_cher2k_batched_64(handle, uplo, transA, N, K, Procfloat_complex(dev_alpha_mem),
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb,
    PSingle(dev_beta_mem), PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_cher2k_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        CheckEqualsRelative(host_C_expected[idx].x, host_C_result[idx].x, rel_epsilon, Format('CHER2K_Batched_64 result.x mismatch at index %d (Device Mode)', [idx]));
        CheckEqualsRelative(host_C_expected[idx].y, host_C_result[idx].y, rel_epsilon, Format('CHER2K_Batched_64 result.y mismatch at index %d (Device Mode)', [idx]));
      end;
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_ZHER2K_Batched_64;
const
  N: Int64 = 5;
  K: Int64 = 4;
  batch_count: Int64 = 2;
  uplo: rocblas_fill = rocblas_fill_lower;
  transA: rocblas_operation = rocblas_operation_none;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ldc: Int64 = 5;
  BETA_VALUE: Double = 2.5;
  rel_epsilon: Double = 1e-12;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, l, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_initial, host_C_expected: array of rocblas_double_complex;
  host_alpha, conj_alpha: rocblas_double_complex;
  host_beta: Double;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of Procdouble_complex;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum1, sum2, term1, term2, conj_A, conj_B: rocblas_double_complex;
  idx, base_A, base_B, base_C: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;
  conj_alpha.x := host_alpha.x; conj_alpha.y := -host_alpha.y;
  host_beta := BETA_VALUE;

  total_elements_A := Int64(batch_count) * lda * K;
  total_elements_B := Int64(batch_count) * ldb * K;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  for i := 0 to High(host_A) do begin host_A[i].x := (Random-0.5)/10; host_A[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_B) do begin host_B[i].x := (Random-0.5)/10; host_B[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C_initial) do begin host_C_initial[i].x := (Random-0.5)/10; host_C_initial[i].y := (Random-0.5)/10; end;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * K;
    base_B := b * ldb * K;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if i >= j then
        begin
          sum1.x := 0; sum1.y := 0;
          sum2.x := 0; sum2.y := 0;
          for l := 0 to K - 1 do
          begin
            conj_B := host_B[base_B + l*ldb + j]; conj_B.y := -conj_B.y;
            term1 := host_A[base_A + l*lda + i];
            ComplexMul(term1, conj_B);
            ComplexAdd(sum1, term1);
            conj_A := host_A[base_A + l*lda + j]; conj_A.y := -conj_A.y;
            term2 := host_B[base_B + l*ldb + i];
            ComplexMul(term2, conj_A);
            ComplexAdd(sum2, term2);
          end;
          ComplexMul(sum1, host_alpha);
          ComplexMul(sum2, conj_alpha);
          host_C_expected[idx].x := host_beta * host_C_initial[idx].x + sum1.x + sum2.x;
          host_C_expected[idx].y := host_beta * host_C_initial[idx].y + sum1.y + sum2.y;
          if i = j then host_C_expected[idx].y := 0.0;
        end
        else
          host_C_expected[idx] := host_C_initial[idx];
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_double_complex);
  size_B := total_elements_B * SizeOf(rocblas_double_complex);
  size_C := total_elements_C * SizeOf(rocblas_double_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procdouble_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + (b * lda * K * SizeOf(rocblas_double_complex)));
    devB_array[b] := Procdouble_complex(PtrUInt(devB) + (b * ldb * K * SizeOf(rocblas_double_complex)));
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_double_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_zher2k_batched_64(handle, uplo, transA, N, K, @host_alpha,
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb,
    @host_beta, PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zher2k_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if host_C_expected[idx].x <> 0.0 then relative_error := Abs((host_C_expected[idx].x - host_C_result[idx].x) / host_C_expected[idx].x) else relative_error := Abs(host_C_result[idx].x);
        error_msg := Format('ZHER2K_Batched_64 result.x mismatch at index %d (Host Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
        if host_C_expected[idx].y <> 0.0 then relative_error := Abs((host_C_expected[idx].y - host_C_result[idx].y) / host_C_expected[idx].y) else relative_error := Abs(host_C_result[idx].y);
        error_msg := Format('ZHER2K_Batched_64 result.y mismatch at index %d (Host Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
      end;
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_double_complex));
  hipMalloc(dev_beta_mem, SizeOf(Double));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(Double), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_zher2k_batched_64(handle, uplo, transA, N, K, Procdouble_complex(dev_alpha_mem),
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb,
    PDouble(dev_beta_mem), PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zher2k_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := j to N - 1 do
      begin
        idx := base_C + j * ldc + i;
        if host_C_expected[idx].x <> 0.0 then relative_error := Abs((host_C_expected[idx].x - host_C_result[idx].x) / host_C_expected[idx].x) else relative_error := Abs(host_C_result[idx].x);
        error_msg := Format('ZHER2K_Batched_64 result.x mismatch at index %d (Device Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
        if host_C_expected[idx].y <> 0.0 then relative_error := Abs((host_C_expected[idx].y - host_C_result[idx].y) / host_C_expected[idx].y) else relative_error := Abs(host_C_result[idx].y);
        error_msg := Format('ZHER2K_Batched_64 result.y mismatch at index %d (Device Mode)', [idx]);
        AssertTrue(error_msg, relative_error <= rel_epsilon);
      end;
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_CHEMM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ldc: Int64 = 5;
  rel_epsilon: Single = 1e-4;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_initial, host_C_expected: array of rocblas_float_complex;
  host_alpha, host_beta: rocblas_float_complex;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of Procfloat_complex;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, term, valA, product: rocblas_float_complex;
  base_A, base_B, base_C: Int64;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;
  host_beta.x  := 2.5; host_beta.y  :=  1.0;

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  // *** FIX: Initialize A as a valid Hermitian matrix ***
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    for j := 0 to M - 1 do
      for i := j to M - 1 do // Only need to set lower triangle
      begin
        host_A[base_A + j * lda + i].x := (Random-0.5)/10;
        if i = j then
          host_A[base_A + j * lda + i].y := 0.0 // Diagonal must be real
        else
          host_A[base_A + j * lda + i].y := (Random-0.5)/10;
      end;
  end;
  for i := 0 to High(host_B) do begin host_B[i].x := (Random-0.5)/10; host_B[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C_initial) do begin host_C_initial[i].x := (Random-0.5)/10; host_C_initial[i].y := (Random-0.5)/10; end;

  // CPU reference: C = beta*C_initial + alpha*A*B
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        sum.x := 0.0; sum.y := 0.0;
        for k := 0 to M - 1 do
        begin
          if i < k then
          begin
            valA := host_A[base_A + i * lda + k];
            valA.y := -valA.y;
          end
          else
            valA := host_A[base_A + k * lda + i];

          term := host_B[base_B + j * ldb + k];
          product := valA;
          ComplexMul(product, term);
          ComplexAdd(sum, product);
        end;
        term := host_C_initial[base_C + j * ldc + i];
        ComplexMul(term, host_beta);
        ComplexMul(sum, host_alpha);
        ComplexAdd(term, sum);
        host_C_expected[base_C + j * ldc + i] := term;
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_float_complex);
  size_B := total_elements_B * SizeOf(rocblas_float_complex);
  size_C := total_elements_C * SizeOf(rocblas_float_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procfloat_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procfloat_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procfloat_complex(PtrUInt(devA) + (b * lda * M * SizeOf(rocblas_float_complex)));
    devB_array[b] := Procfloat_complex(PtrUInt(devB) + (b * ldb * N * SizeOf(rocblas_float_complex)));
    devC_array[b] := Procfloat_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_float_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procfloat_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_chemm_batched_64(handle, side, uplo, M, N, @host_alpha,
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb, @host_beta,
    PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_chemm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    CheckEqualsRelative(host_C_expected[i].x, host_C_result[i].x, rel_epsilon, Format('CHEMM_Batched_64 result.x mismatch at index %d (Host Mode)', [i]));
    CheckEqualsRelative(host_C_expected[i].y, host_C_result[i].y, rel_epsilon, Format('CHEMM_Batched_64 result.y mismatch at index %d (Host Mode)', [i]));
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_float_complex));
  hipMalloc(dev_beta_mem, SizeOf(rocblas_float_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_chemm_batched_64(handle, side, uplo, M, N, Procfloat_complex(dev_alpha_mem),
    PPProcfloat_complex(devA_array_device), lda, PPProcfloat_complex(devB_array_device), ldb, Procfloat_complex(dev_beta_mem),
    PPProcfloat_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_chemm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    CheckEqualsRelative(host_C_expected[i].x, host_C_result[i].x, rel_epsilon, Format('CHEMM_Batched_64 result.x mismatch at index %d (Device Mode)', [i]));
    CheckEqualsRelative(host_C_expected[i].y, host_C_result[i].y, rel_epsilon, Format('CHEMM_Batched_64 result.y mismatch at index %d (Device Mode)', [i]));
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

procedure TTestRocBLAS_Level3_Batched_64.Test_ZHEMM_Batched_64;
const
  M: Int64 = 5;
  N: Int64 = 4;
  batch_count: Int64 = 2;
  side: rocblas_side = rocblas_side_left;
  uplo: rocblas_fill = rocblas_fill_lower;
  lda: Int64 = 5;
  ldb: Int64 = 5;
  ldc: Int64 = 5;
  rel_epsilon: Double = 1e-12;
var
  size_A, size_B, size_C: SizeUInt;
  i, j, k, b: Integer;
  total_elements_A, total_elements_B, total_elements_C: Int64;

  host_A, host_B, host_C_result, host_C_initial, host_C_expected: array of rocblas_double_complex;
  host_alpha, host_beta: rocblas_double_complex;

  devA, devB, devC: hipDeviceptr_t;
  devA_array, devB_array, devC_array: array of Procdouble_complex;
  devA_array_device, devB_array_device, devC_array_device: hipDeviceptr_t;
  dev_alpha_mem, dev_beta_mem: hipDeviceptr_t;

  blas_status: rocblas_status;
  sum, term, valA, product: rocblas_double_complex;
  base_A, base_B, base_C: Int64;
  relative_error: Double;
  error_msg: String;
begin
  host_alpha.x := 1.5; host_alpha.y := -0.5;
  host_beta.x  := 2.5; host_beta.y  :=  1.0;

  total_elements_A := Int64(batch_count) * lda * M;
  total_elements_B := Int64(batch_count) * ldb * N;
  total_elements_C := Int64(batch_count) * ldc * N;

  SetLength(host_A, total_elements_A);
  SetLength(host_B, total_elements_B);
  SetLength(host_C_result, total_elements_C);
  SetLength(host_C_initial, total_elements_C);
  SetLength(host_C_expected, total_elements_C);

  // *** FIX: Initialize A as a valid Hermitian matrix ***
  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    for j := 0 to M - 1 do
      for i := j to M - 1 do
      begin
        host_A[base_A + j * lda + i].x := (Random-0.5)/10;
        if i = j then
          host_A[base_A + j * lda + i].y := 0.0
        else
          host_A[base_A + j * lda + i].y := (Random-0.5)/10;
      end;
  end;
  for i := 0 to High(host_B) do begin host_B[i].x := (Random-0.5)/10; host_B[i].y := (Random-0.5)/10; end;
  for i := 0 to High(host_C_initial) do begin host_C_initial[i].x := (Random-0.5)/10; host_C_initial[i].y := (Random-0.5)/10; end;

  for b := 0 to batch_count - 1 do
  begin
    base_A := b * lda * M;
    base_B := b * ldb * N;
    base_C := b * ldc * N;
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        sum.x := 0.0; sum.y := 0.0;
        for k := 0 to M - 1 do
        begin
          if i < k then
          begin
            valA := host_A[base_A + i * lda + k];
            valA.y := -valA.y;
          end
          else
            valA := host_A[base_A + k * lda + i];

          term := host_B[base_B + j * ldb + k];
          product := valA;
          ComplexMul(product, term);
          ComplexAdd(sum, product);
        end;
        term := host_C_initial[base_C + j * ldc + i];
        ComplexMul(term, host_beta);
        ComplexMul(sum, host_alpha);
        ComplexAdd(term, sum);
        host_C_expected[base_C + j * ldc + i] := term;
      end;
  end;

  size_A := total_elements_A * SizeOf(rocblas_double_complex);
  size_B := total_elements_B * SizeOf(rocblas_double_complex);
  size_C := total_elements_C * SizeOf(rocblas_double_complex);
  hipMalloc(devA, size_A);
  hipMalloc(devB, size_B);
  hipMalloc(devC, size_C);
  hipMemcpy(devA, @host_A[0], size_A, hipMemcpyHostToDevice);
  hipMemcpy(devB, @host_B[0], size_B, hipMemcpyHostToDevice);
  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);

  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);
  hipMalloc(devA_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devB_array_device, batch_count * SizeOf(Procdouble_complex));
  hipMalloc(devC_array_device, batch_count * SizeOf(Procdouble_complex));

  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + (b * lda * M * SizeOf(rocblas_double_complex)));
    devB_array[b] := Procdouble_complex(PtrUInt(devB) + (b * ldb * N * SizeOf(rocblas_double_complex)));
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + (b * ldc * N * SizeOf(rocblas_double_complex)));
  end;

  hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);
  hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Procdouble_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  blas_status := rocblas_zhemm_batched_64(handle, side, uplo, M, N, @host_alpha,
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb, @host_beta,
    PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zhemm_batched_64 (host mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    if host_C_expected[i].x <> 0.0 then relative_error := Abs((host_C_expected[i].x - host_C_result[i].x) / host_C_expected[i].x) else relative_error := Abs(host_C_result[i].x);
    error_msg := Format('ZHEMM_Batched_64 result.x mismatch at index %d (Host Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
    if host_C_expected[i].y <> 0.0 then relative_error := Abs((host_C_expected[i].y - host_C_result[i].y) / host_C_expected[i].y) else relative_error := Abs(host_C_result[i].y);
    error_msg := Format('ZHEMM_Batched_64 result.y mismatch at index %d (Host Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipMemcpy(devC, @host_C_initial[0], size_C, hipMemcpyHostToDevice);
  hipMalloc(dev_alpha_mem, SizeOf(rocblas_double_complex));
  hipMalloc(dev_beta_mem, SizeOf(rocblas_double_complex));
  hipMemcpy(dev_alpha_mem, @host_alpha, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(dev_beta_mem, @host_beta, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_zhemm_batched_64(handle, side, uplo, M, N, Procdouble_complex(dev_alpha_mem),
    PPProcdouble_complex(devA_array_device), lda, PPProcdouble_complex(devB_array_device), ldb, Procdouble_complex(dev_beta_mem),
    PPProcdouble_complex(devC_array_device), ldc, batch_count);
  AssertEquals('rocblas_zhemm_batched_64 (device mode) failed', Integer(rocblas_status_success), Integer(blas_status));
  hipMemcpy(@host_C_result[0], devC, size_C, hipMemcpyDeviceToHost);

  for i := 0 to High(host_C_expected) do
  begin
    if host_C_expected[i].x <> 0.0 then relative_error := Abs((host_C_expected[i].x - host_C_result[i].x) / host_C_expected[i].x) else relative_error := Abs(host_C_result[i].x);
    error_msg := Format('ZHEMM_Batched_64 result.x mismatch at index %d (Device Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
    if host_C_expected[i].y <> 0.0 then relative_error := Abs((host_C_expected[i].y - host_C_result[i].y) / host_C_expected[i].y) else relative_error := Abs(host_C_result[i].y);
    error_msg := Format('ZHEMM_Batched_64 result.y mismatch at index %d (Device Mode)', [i]);
    AssertTrue(error_msg, relative_error <= rel_epsilon);
  end;

  hipFree(devA); hipFree(devB); hipFree(devC);
  hipFree(devA_array_device); hipFree(devB_array_device); hipFree(devC_array_device);
  hipFree(dev_alpha_mem); hipFree(dev_beta_mem);
end;

initialization
  RegisterTest(TTestRocBLAS_Level3_Batched_64);

end.
