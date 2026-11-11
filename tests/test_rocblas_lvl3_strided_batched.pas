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

unit Test_rocblas_lvl3_strided_batched;

{$mode objfpc}{$H+}
{$linklib rocblas}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  TTestRocBLAS_Level3_Strided_Batched = class(TTestRocBLAS_Base)
  published
    procedure Test_SGEMM_Strided_Batched;
    procedure Test_DGEMM_Strided_Batched;
    procedure Test_CGEMM_Strided_Batched;
    procedure Test_ZGEMM_Strided_Batched;
    procedure Test_SSYMM_Strided_Batched;
    procedure Test_DSYMM_Strided_Batched;
    procedure Test_CSYMM_Strided_Batched;
    procedure Test_ZSYMM_Strided_Batched;
    procedure Test_STRMM_Strided_Batched;
    procedure Test_DTRMM_Strided_Batched;
    procedure Test_CTRMM_Strided_Batched;
    procedure Test_ZTRMM_Strided_Batched;
    procedure Test_STRSM_Strided_Batched;
    procedure Test_DTRSM_Strided_Batched;
    procedure Test_CTRSM_Strided_Batched;
    procedure Test_ZTRSM_Strided_Batched;
    procedure Test_SSYRK_Strided_Batched;
    procedure Test_DSYRK_Strided_Batched;
    procedure Test_CSYRK_Strided_Batched;
    procedure Test_ZSYRK_Strided_Batched;
    procedure Test_CHERK_Strided_Batched;
    procedure Test_ZHERK_Strided_Batched;
    procedure Test_SSyr2k_Strided_Batched;
    procedure Test_DSyr2k_Strided_Batched;
    procedure Test_CSyr2k_Strided_Batched;
    procedure Test_ZSyr2k_Strided_Batched;
    procedure Test_CHER2K_Strided_Batched;
    procedure Test_ZHER2K_Strided_Batched;
  end;

implementation

procedure TTestRocBLAS_Level3_Strided_Batched.Test_SGEMM_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: PSingle;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, l: Integer;
  alpha, beta: Single;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC_expected: PSingle;
  sum: Single;
const
  M = 64;
  N = 64;
  K = 64;
  transA = rocblas_operation_none;
  transB = rocblas_operation_none;
  lda = M;
  ldb = K;
  ldc = M;
  stride_A = M * K;
  stride_B = K * N;
  stride_C = M * N;
  batch_count = 3;
  tolerance: Single = 1e-5;
begin
  alpha := 1.2;
  beta := 0.8;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(Single);
  size_B := stride_B * batch_count * SizeOf(Single);
  size_C := stride_C * batch_count * SizeOf(Single);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    // Initialize host matrices
    for i := 0 to (stride_A * batch_count) - 1 do h_A[i] := (i mod 100) / 100.0;
    for i := 0 to (stride_B * batch_count) - 1 do h_B[i] := (i mod 100) / 100.0;
    for i := 0 to (stride_C * batch_count) - 1 do h_C[i] := (i mod 100) / 100.0;
    Move(h_C^, h_C_expected^, size_C);

    // CPU-based GEMM calculation for verification
    for b := 0 to batch_count - 1 do
    begin
      pA := PSingle(PByte(h_A) + b * stride_A * SizeOf(Single));
      pB := PSingle(PByte(h_B) + b * stride_B * SizeOf(Single));
      pC_expected := PSingle(PByte(h_C_expected) + b * stride_C * SizeOf(Single));

      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          sum := 0.0;
          for l := 0 to K - 1 do
          begin
            sum := sum + pA[i + l * lda] * pB[l + j * ldb];
          end;
          pC_expected[i + j * ldc] := alpha * sum + beta * pC_expected[i + j * ldc];
        end;
      end;
    end;

    // Allocate device memory
    hip_status := hipMalloc(d_A, size_A);
    AssertEquals('hipMalloc for d_A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B);
    AssertEquals('hipMalloc for d_B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C);
    AssertEquals('hipMalloc for d_C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    // Copy data from host to device
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    // Set pointer mode to host
    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    // Call rocBLAS function
    blas_status := rocblas_sgemm_strided_batched(handle, transA, transB, M, N, K, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_sgemm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    // Copy result from device to host
    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H for C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipDeviceSynchronize();
    AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    // Verify results
    for i := 0 to (stride_C * batch_count) - 1 do
    begin
      CheckEqualsRelative(h_C_expected[i], h_C[i], tolerance, 'Mismatch at index ' + IntToStr(i));
    end;

  finally
    // Free memory
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_DGEMM_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: PDouble;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, l: Integer;
  alpha, beta: Double;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC_expected: PDouble;
  sum: Double;
const
  M = 64;
  N = 64;
  K = 64;
  transA = rocblas_operation_none;
  transB = rocblas_operation_none;
  lda = M;
  ldb = K;
  ldc = M;
  stride_A = M * K;
  stride_B = K * N;
  stride_C = M * N;
  batch_count = 3;
  tolerance: Double = 1e-12;
begin
  alpha := 1.2;
  beta := 0.8;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(Double);
  size_B := stride_B * batch_count * SizeOf(Double);
  size_C := stride_C * batch_count * SizeOf(Double);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    // Initialize host matrices
    for i := 0 to (stride_A * batch_count) - 1 do h_A[i] := (i mod 100) / 100.0;
    for i := 0 to (stride_B * batch_count) - 1 do h_B[i] := (i mod 100) / 100.0;
    for i := 0 to (stride_C * batch_count) - 1 do h_C[i] := (i mod 100) / 100.0;
    Move(h_C^, h_C_expected^, size_C);

    // CPU-based GEMM calculation for verification
    for b := 0 to batch_count - 1 do
    begin
      pA := PDouble(PByte(h_A) + b * stride_A * SizeOf(Double));
      pB := PDouble(PByte(h_B) + b * stride_B * SizeOf(Double));
      pC_expected := PDouble(PByte(h_C_expected) + b * stride_C * SizeOf(Double));

      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          sum := 0.0;
          for l := 0 to K - 1 do
          begin
            sum := sum + pA[i + l * lda] * pB[l + j * ldb];
          end;
          pC_expected[i + j * ldc] := alpha * sum + beta * pC_expected[i + j * ldc];
        end;
      end;
    end;

    // Allocate device memory
    hip_status := hipMalloc(d_A, size_A);
    AssertEquals('hipMalloc for d_A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B);
    AssertEquals('hipMalloc for d_B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C);
    AssertEquals('hipMalloc for d_C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    // Copy data from host to device
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    // Set pointer mode to host
    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    // Call rocBLAS function
    blas_status := rocblas_dgemm_strided_batched(handle, transA, transB, M, N, K, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_dgemm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    // Copy result from device to host
    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H for C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipDeviceSynchronize();
    AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    // Verify results
    for i := 0 to (stride_C * batch_count) - 1 do
    begin
      CheckEquals(h_C_expected[i], h_C[i], tolerance, 'Mismatch at index ' + IntToStr(i));
    end;

  finally
    // Free memory
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_CGEMM_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: Procfloat_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, l: Integer;
  alpha, beta: rocblas_float_complex;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC_expected: Procfloat_complex;
  sum, term, beta_C: rocblas_float_complex;
const
  M = 64;
  N = 64;
  K = 64;
  transA = rocblas_operation_none;
  transB = rocblas_operation_none;
  lda = M;
  ldb = K;
  ldc = M;
  stride_A = M * K;
  stride_B = K * N;
  stride_C = M * N;
  batch_count = 3;
  tolerance: Single = 1e-5;
begin
  alpha.x := 1.2; alpha.y := -0.5;
  beta.x := 0.8; beta.y := 0.3;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_float_complex);
  size_B := stride_B * batch_count * SizeOf(rocblas_float_complex);
  size_C := stride_C * batch_count * SizeOf(rocblas_float_complex);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    // Initialize host matrices
    for i := 0 to (stride_A * batch_count) - 1 do
    begin
      h_A[i].x := (i mod 10) / 10.0;
      h_A[i].y := (i mod 20) / 20.0;
    end;
    for i := 0 to (stride_B * batch_count) - 1 do
    begin
      h_B[i].x := (i mod 30) / 30.0;
      h_B[i].y := (i mod 40) / 40.0;
    end;
    for i := 0 to (stride_C * batch_count) - 1 do
    begin
      h_C[i].x := (i mod 50) / 50.0;
      h_C[i].y := (i mod 60) / 60.0;
    end;
    Move(h_C^, h_C_expected^, size_C);

    // CPU-based GEMM calculation for verification
    for b := 0 to batch_count - 1 do
    begin
      pA := Procfloat_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_float_complex));
      pB := Procfloat_complex(PByte(h_B) + b * stride_B * SizeOf(rocblas_float_complex));
      pC_expected := Procfloat_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_float_complex));

      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          sum.x := 0.0; sum.y := 0.0;
          for l := 0 to K - 1 do
          begin
            term.x := pA[i + l * lda].x * pB[l + j * ldb].x - pA[i + l * lda].y * pB[l + j * ldb].y;
            term.y := pA[i + l * lda].x * pB[l + j * ldb].y + pA[i + l * lda].y * pB[l + j * ldb].x;
            ComplexAdd(sum, term);
          end;
          beta_C := pC_expected[i + j * ldc];
          ComplexMul(beta_C, beta);
          ComplexMul(sum, alpha);
          ComplexAdd(sum, beta_C);
          pC_expected[i + j * ldc] := sum;
        end;
      end;
    end;

    // Allocate device memory
    hip_status := hipMalloc(d_A, size_A);
    AssertEquals('hipMalloc for d_A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B);
    AssertEquals('hipMalloc for d_B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C);
    AssertEquals('hipMalloc for d_C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    // Copy data from host to device
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    // Set pointer mode to host
    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    // Call rocBLAS function
    blas_status := rocblas_cgemm_strided_batched(handle, transA, transB, M, N, K, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_cgemm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    // Copy result from device to host
    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H for C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipDeviceSynchronize();
    AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    // Verify results
    for i := 0 to (stride_C * batch_count) - 1 do
    begin
      CheckEqualsRelative(h_C_expected[i].x, h_C[i].x, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (real)');
      CheckEqualsRelative(h_C_expected[i].y, h_C[i].y, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (imag)');
    end;

  finally
    // Free memory
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_ZGEMM_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: Procdouble_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, b, l: Integer;
  alpha, beta: rocblas_double_complex;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC_expected: Procdouble_complex;
  sum, term, beta_C: rocblas_double_complex;
const
  M = 64;
  N = 64;
  K = 64;
  transA = rocblas_operation_none;
  transB = rocblas_operation_none;
  lda = M;
  ldb = K;
  ldc = M;
  stride_A = M * K;
  stride_B = K * N;
  stride_C = M * N;
  batch_count = 3;
  tolerance: Double = 1e-12;
begin
  alpha.x := 1.2; alpha.y := -0.5;
  beta.x := 0.8; beta.y := 0.3;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_double_complex);
  size_B := stride_B * batch_count * SizeOf(rocblas_double_complex);
  size_C := stride_C * batch_count * SizeOf(rocblas_double_complex);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    // Initialize host matrices
    for i := 0 to (stride_A * batch_count) - 1 do
    begin
      h_A[i].x := (i mod 10) / 10.0;
      h_A[i].y := (i mod 20) / 20.0;
    end;
    for i := 0 to (stride_B * batch_count) - 1 do
    begin
      h_B[i].x := (i mod 30) / 30.0;
      h_B[i].y := (i mod 40) / 40.0;
    end;
    for i := 0 to (stride_C * batch_count) - 1 do
    begin
      h_C[i].x := (i mod 50) / 50.0;
      h_C[i].y := (i mod 60) / 60.0;
    end;
    Move(h_C^, h_C_expected^, size_C);

    // CPU-based GEMM calculation for verification
    for b := 0 to batch_count - 1 do
    begin
      pA := Procdouble_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_double_complex));
      pB := Procdouble_complex(PByte(h_B) + b * stride_B * SizeOf(rocblas_double_complex));
      pC_expected := Procdouble_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_double_complex));

      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          sum.x := 0.0; sum.y := 0.0;
          for l := 0 to K - 1 do
          begin
            term.x := pA[i + l * lda].x * pB[l + j * ldb].x - pA[i + l * lda].y * pB[l + j * ldb].y;
            term.y := pA[i + l * lda].x * pB[l + j * ldb].y + pA[i + l * lda].y * pB[l + j * ldb].x;
            ComplexAdd(sum, term);
          end;
          beta_C := pC_expected[i + j * ldc];
          ComplexMul(beta_C, beta);
          ComplexMul(sum, alpha);
          ComplexAdd(sum, beta_C);
          pC_expected[i + j * ldc] := sum;
        end;
      end;
    end;

    // Allocate device memory
    hip_status := hipMalloc(d_A, size_A);
    AssertEquals('hipMalloc for d_A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B);
    AssertEquals('hipMalloc for d_B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C);
    AssertEquals('hipMalloc for d_C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    // Copy data from host to device
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    // Set pointer mode to host
    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    // Call rocBLAS function
    blas_status := rocblas_zgemm_strided_batched(handle, transA, transB, M, N, K, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_zgemm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    // Copy result from device to host
    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H for C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipDeviceSynchronize();
    AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    // Verify results
    for i := 0 to (stride_C * batch_count) - 1 do
    begin
      CheckEquals(h_C_expected[i].x, h_C[i].x, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (real)');
      CheckEquals(h_C_expected[i].y, h_C[i].y, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (imag)');
    end;

  finally
    // Free memory
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_SSYMM_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: PSingle;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, k, b: Integer;
  alpha, beta: Single;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC_expected: PSingle;
  temp_A_row, temp_A_col, temp_swap: Integer;
  sum: Single;
const
  M = 64;
  N = 72; // Use non-square B,C for better testing
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  batch_count = 3;
  lda = M;
  ldb = M;
  ldc = M;
  stride_A = lda * M;
  stride_B = ldb * N;
  stride_C = ldc * N;
  tolerance: Single = 1e-5;
begin
  alpha := 1.2;
  beta := 0.8;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(Single);
  size_B := stride_B * batch_count * SizeOf(Single);
  size_C := stride_C * batch_count * SizeOf(Single);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    // Initialize host matrices
    for i := 0 to (size_A div SizeOf(Single)) - 1 do h_A[i] := (i mod 100) / 100.0;
    for i := 0 to (size_B div SizeOf(Single)) - 1 do h_B[i] := (i mod 100) / 100.0;
    for i := 0 to (size_C div SizeOf(Single)) - 1 do h_C[i] := (i mod 100) / 100.0;
    Move(h_C^, h_C_expected^, size_C);

    // CPU-based SYMM calculation for verification
    for b := 0 to batch_count - 1 do
    begin
      pA := PSingle(PByte(h_A) + b * stride_A * SizeOf(Single));
      pB := PSingle(PByte(h_B) + b * stride_B * SizeOf(Single));
      pC_expected := PSingle(PByte(h_C_expected) + b * stride_C * SizeOf(Single));

      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          sum := 0.0;
          for k := 0 to M - 1 do
          begin
            // A is symmetric, access upper triangle
            temp_A_row := i;
            temp_A_col := k;
            if temp_A_row > temp_A_col then
            begin
              temp_swap := temp_A_row;
              temp_A_row := temp_A_col;
              temp_A_col := temp_swap;
            end;
            sum := sum + pA[temp_A_row + temp_A_col * lda] * pB[k + j * ldb];
          end;
          pC_expected[i + j * ldc] := alpha * sum + beta * pC_expected[i + j * ldc];
        end;
      end;
    end;

    // Allocate device memory and copy data
    hip_status := hipMalloc(d_A, size_A);
    AssertEquals('hipMalloc for d_A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B);
    AssertEquals('hipMalloc for d_B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C);
    AssertEquals('hipMalloc for d_C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    // Call rocBLAS function
    blas_status := rocblas_ssymm_strided_batched(handle, side, uplo, M, N, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_ssymm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H for C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipDeviceSynchronize();
    AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    // Verify results
    for i := 0 to (size_C div SizeOf(Single)) - 1 do
      CheckEqualsRelative(h_C_expected[i], h_C[i], tolerance, 'Mismatch at index ' + IntToStr(i));

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_DSYMM_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: PDouble;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, k, b: Integer;
  alpha, beta: Double;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC_expected: PDouble;
  temp_A_row, temp_A_col, temp_swap: Integer;
  sum: Double;
const
  M = 64;
  N = 72;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  batch_count = 3;
  lda = M;
  ldb = M;
  ldc = M;
  stride_A = lda * M;
  stride_B = ldb * N;
  stride_C = ldc * N;
  tolerance: Double = 1e-12;
begin
  alpha := 1.2;
  beta := 0.8;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(Double);
  size_B := stride_B * batch_count * SizeOf(Double);
  size_C := stride_C * batch_count * SizeOf(Double);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    for i := 0 to (size_A div SizeOf(Double)) - 1 do h_A[i] := (i mod 100) / 100.0;
    for i := 0 to (size_B div SizeOf(Double)) - 1 do h_B[i] := (i mod 100) / 100.0;
    for i := 0 to (size_C div SizeOf(Double)) - 1 do h_C[i] := (i mod 100) / 100.0;
    Move(h_C^, h_C_expected^, size_C);

    for b := 0 to batch_count - 1 do
    begin
      pA := PDouble(PByte(h_A) + b * stride_A * SizeOf(Double));
      pB := PDouble(PByte(h_B) + b * stride_B * SizeOf(Double));
      pC_expected := PDouble(PByte(h_C_expected) + b * stride_C * SizeOf(Double));
      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          sum := 0.0;
          for k := 0 to M - 1 do
          begin
            temp_A_row := i;
            temp_A_col := k;
            if temp_A_row > temp_A_col then
            begin
                temp_swap := temp_A_row;
                temp_A_row := temp_A_col;
                temp_A_col := temp_swap;
            end;
            sum := sum + pA[temp_A_row + temp_A_col * lda] * pB[k + j * ldb];
          end;
          pC_expected[i + j * ldc] := alpha * sum + beta * pC_expected[i + j * ldc];
        end;
      end;
    end;

    hip_status := hipMalloc(d_A, size_A);
    AssertEquals('hipMalloc for d_A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B);
    AssertEquals('hipMalloc for d_B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C);
    AssertEquals('hipMalloc for d_C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H->D for C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_dsymm_strided_batched(handle, side, uplo, M, N, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_dsymm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H for C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipDeviceSynchronize();
    AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for i := 0 to (size_C div SizeOf(Double)) - 1 do
      CheckEquals(h_C_expected[i], h_C[i], tolerance, 'Mismatch at index ' + IntToStr(i));

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_CSYMM_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: Procfloat_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, k, b: Integer;
  alpha, beta: rocblas_float_complex;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC_expected: Procfloat_complex;
  temp_A_row, temp_A_col, temp_swap: Integer;
  sum, term, beta_C: rocblas_float_complex;
const
  M = 64;
  N = 72;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  batch_count = 3;
  lda = M;
  ldb = M;
  ldc = M;
  stride_A = lda * M;
  stride_B = ldb * N;
  stride_C = ldc * N;
  tolerance: Single = 1e-5;
begin
  alpha.x := 1.2; alpha.y := -0.5;
  beta.x := 0.8; beta.y := 0.3;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_float_complex);
  size_B := stride_B * batch_count * SizeOf(rocblas_float_complex);
  size_C := stride_C * batch_count * SizeOf(rocblas_float_complex);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    for i := 0 to (size_A div SizeOf(rocblas_float_complex)) - 1 do begin h_A[i].x := (i mod 10)/10; h_A[i].y := (i mod 20)/20; end;
    for i := 0 to (size_B div SizeOf(rocblas_float_complex)) - 1 do begin h_B[i].x := (i mod 30)/30; h_B[i].y := (i mod 40)/40; end;
    for i := 0 to (size_C div SizeOf(rocblas_float_complex)) - 1 do begin h_C[i].x := (i mod 50)/50; h_C[i].y := (i mod 60)/60; end;
    Move(h_C^, h_C_expected^, size_C);

    for b := 0 to batch_count - 1 do
    begin
      pA := Procfloat_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_float_complex));
      pB := Procfloat_complex(PByte(h_B) + b * stride_B * SizeOf(rocblas_float_complex));
      pC_expected := Procfloat_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_float_complex));
      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          sum.x := 0; sum.y := 0;
          for k := 0 to M - 1 do
          begin
            temp_A_row := i;
            temp_A_col := k;
            if temp_A_row > temp_A_col then
            begin
                temp_swap := temp_A_row;
                temp_A_row := temp_A_col;
                temp_A_col := temp_swap;
            end;
            term := pA[temp_A_row + temp_A_col * lda];
            ComplexMul(term, pB[k + j * ldb]);
            ComplexAdd(sum, term);
          end;
          beta_C := pC_expected[i + j * ldc];
          ComplexMul(beta_C, beta);
          ComplexMul(sum, alpha);
          ComplexAdd(sum, beta_C);
          pC_expected[i + j * ldc] := sum;
        end;
      end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_csymm_strided_batched(handle, side, uplo, M, N, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_csymm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost); AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for i := 0 to (size_C div SizeOf(rocblas_float_complex)) - 1 do
    begin
      CheckEqualsRelative(h_C_expected[i].x, h_C[i].x, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (real)');
      CheckEqualsRelative(h_C_expected[i].y, h_C[i].y, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (imag)');
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_ZSYMM_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: Procdouble_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, k, b: Integer;
  alpha, beta: rocblas_double_complex;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC_expected: Procdouble_complex;
  temp_A_row, temp_A_col, temp_swap: Integer;
  sum, term, beta_C: rocblas_double_complex;
const
  M = 64;
  N = 72;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  batch_count = 3;
  lda = M;
  ldb = M;
  ldc = M;
  stride_A = lda * M;
  stride_B = ldb * N;
  stride_C = ldc * N;
  tolerance: Double = 1e-12;
begin
  alpha.x := 1.2; alpha.y := -0.5;
  beta.x := 0.8; beta.y := 0.3;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_double_complex);
  size_B := stride_B * batch_count * SizeOf(rocblas_double_complex);
  size_C := stride_C * batch_count * SizeOf(rocblas_double_complex);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    for i := 0 to (size_A div SizeOf(rocblas_double_complex)) - 1 do begin h_A[i].x := (i mod 10)/10; h_A[i].y := (i mod 20)/20; end;
    for i := 0 to (size_B div SizeOf(rocblas_double_complex)) - 1 do begin h_B[i].x := (i mod 30)/30; h_B[i].y := (i mod 40)/40; end;
    for i := 0 to (size_C div SizeOf(rocblas_double_complex)) - 1 do begin h_C[i].x := (i mod 50)/50; h_C[i].y := (i mod 60)/60; end;
    Move(h_C^, h_C_expected^, size_C);

    for b := 0 to batch_count - 1 do
    begin
      pA := Procdouble_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_double_complex));
      pB := Procdouble_complex(PByte(h_B) + b * stride_B * SizeOf(rocblas_double_complex));
      pC_expected := Procdouble_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_double_complex));
      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          sum.x := 0; sum.y := 0;
          for k := 0 to M - 1 do
          begin
            temp_A_row := i;
            temp_A_col := k;
            if temp_A_row > temp_A_col then
            begin
              temp_swap := temp_A_row;
              temp_A_row := temp_A_col;
              temp_A_col := temp_swap;
            end;
            term := pA[temp_A_row + temp_A_col * lda];
            ComplexMul(term, pB[k + j * ldb]);
            ComplexAdd(sum, term);
          end;
          beta_C := pC_expected[i + j * ldc];
          ComplexMul(beta_C, beta);
          ComplexMul(sum, alpha);
          ComplexAdd(sum, beta_C);
          pC_expected[i + j * ldc] := sum;
        end;
      end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_zsymm_strided_batched(handle, side, uplo, M, N, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_zsymm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost); AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for i := 0 to (size_C div SizeOf(rocblas_double_complex)) - 1 do
    begin
      CheckEquals(h_C_expected[i].x, h_C[i].x, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (real)');
      CheckEquals(h_C_expected[i].y, h_C[i].y, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (imag)');
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_STRMM_Strided_Batched;
var
  h_A, h_B, h_C_result, h_C_expected: PSingle;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, k, b: Integer;
  alpha: Single;
  size_A, size_B_C: SizeUInt;
  pA, pB, pC_expected: PSingle;
  sum, A_ik: Single;
const
  M = 64;
  N = 72;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  batch_count = 3;
  lda = M;
  ldb = M;
  ldc = M;
  stride_A = lda * M;
  stride_B = ldb * N;
  stride_C = ldc * N;
  tolerance: Single = 1e-5;
begin
  alpha := 1.25;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C_result := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(Single);
  size_B_C := stride_B * batch_count * SizeOf(Single);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B_C);
  GetMem(h_C_result, size_B_C);
  GetMem(h_C_expected, size_B_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do h_A[i] := (i mod 100) / 100.0 + 0.01;
    for i := 0 to (stride_B * batch_count) - 1 do h_B[i] := (i mod 100) / 100.0;

    for b := 0 to batch_count - 1 do
    begin
      pA := PSingle(PByte(h_A) + b * stride_A * SizeOf(Single));
      pB := PSingle(PByte(h_B) + b * stride_B * SizeOf(Single));
      pC_expected := PSingle(PByte(h_C_expected) + b * stride_C * SizeOf(Single));

      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          sum := 0.0;
          for k := 0 to M - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if (i > k) then A_ik := 0.0 else A_ik := pA[i + k * lda];
            end
            else
            begin
              if (i < k) then A_ik := 0.0 else A_ik := pA[i + k * lda];
            end;
            sum := sum + A_ik * pB[k + j * ldb];
          end;
          pC_expected[i + j * ldc] := alpha * sum;
        end;
      end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B_C); AssertEquals('hipMalloc B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_B_C); AssertEquals('hipMalloc C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D for A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D for B failed', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_strmm_strided_batched(handle, side, uplo, transA, diag, M, N, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_strmm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C_result, d_C, size_B_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H for result C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipDeviceSynchronize();
    AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for i := 0 to (stride_C * batch_count) - 1 do
      CheckEqualsRelative(h_C_expected[i], h_C_result[i], tolerance, 'Mismatch at index ' + IntToStr(i));

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C_result) then FreeMem(h_C_result);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_DTRMM_Strided_Batched;
var
  h_A, h_B, h_C_result, h_C_expected: PDouble;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, k, b: Integer;
  alpha: Double;
  size_A, size_B_C: SizeUInt;
  pA, pB, pC_expected: PDouble;
  sum, A_ik: Double;
const
  M = 64;
  N = 72;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  batch_count = 3;
  lda = M;
  ldb = M;
  ldc = M;
  stride_A = lda * M;
  stride_B = ldb * N;
  stride_C = ldc * N;
  tolerance: Double = 1e-12;
begin
  alpha := 1.25;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C_result := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(Double);
  size_B_C := stride_B * batch_count * SizeOf(Double);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B_C);
  GetMem(h_C_result, size_B_C);
  GetMem(h_C_expected, size_B_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do h_A[i] := (i mod 100) / 100.0 + 0.01;
    for i := 0 to (stride_B * batch_count) - 1 do h_B[i] := (i mod 100) / 100.0;

    for b := 0 to batch_count - 1 do
    begin
      pA := PDouble(PByte(h_A) + b * stride_A * SizeOf(Double));
      pB := PDouble(PByte(h_B) + b * stride_B * SizeOf(Double));
      pC_expected := PDouble(PByte(h_C_expected) + b * stride_C * SizeOf(Double));

      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          sum := 0.0;
          for k := 0 to M - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if (i > k) then A_ik := 0.0 else A_ik := pA[i + k * lda];
            end
            else
            begin
              if (i < k) then A_ik := 0.0 else A_ik := pA[i + k * lda];
            end;
            sum := sum + A_ik * pB[k + j * ldb];
          end;
          pC_expected[i + j * ldc] := alpha * sum;
        end;
      end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B_C); AssertEquals('hipMalloc B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_B_C); AssertEquals('hipMalloc C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D for A failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D for B failed', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_dtrmm_strided_batched(handle, side, uplo, transA, diag, M, N, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_dtrmm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C_result, d_C, size_B_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H for result C failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipDeviceSynchronize();
    AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for i := 0 to (stride_C * batch_count) - 1 do
      CheckEquals(h_C_expected[i], h_C_result[i], tolerance, 'Mismatch at index ' + IntToStr(i));

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C_result) then FreeMem(h_C_result);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_CTRMM_Strided_Batched;
var
  h_A, h_B, h_C_result, h_C_expected: Procfloat_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, k, b: Integer;
  alpha: rocblas_float_complex;
  size_A, size_B_C: SizeUInt;
  pA, pB, pC_expected: Procfloat_complex;
  sum, term, A_ik: rocblas_float_complex;
const
  M = 64;
  N = 72;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  batch_count = 3;
  lda = M;
  ldb = M;
  ldc = M;
  stride_A = lda * M;
  stride_B = ldb * N;
  stride_C = ldc * N;
  tolerance: Single = 1e-5;
begin
  alpha.x := 1.25; alpha.y := -0.5;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C_result := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_float_complex);
  size_B_C := stride_B * batch_count * SizeOf(rocblas_float_complex);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B_C);
  GetMem(h_C_result, size_B_C);
  GetMem(h_C_expected, size_B_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do begin h_A[i].x := (i mod 10)/10+0.1; h_A[i].y := (i mod 20)/20; end;
    for i := 0 to (stride_B * batch_count) - 1 do begin h_B[i].x := (i mod 30)/30; h_B[i].y := (i mod 40)/40; end;

    for b := 0 to batch_count - 1 do
    begin
      pA := Procfloat_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_float_complex));
      pB := Procfloat_complex(PByte(h_B) + b * stride_B * SizeOf(rocblas_float_complex));
      pC_expected := Procfloat_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_float_complex));

      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          sum.x := 0.0; sum.y := 0.0;
          for k := 0 to M - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if (i > k) then begin A_ik.x := 0.0; A_ik.y := 0.0; end
              else A_ik := pA[i + k * lda];
            end
            else
            begin
              if (i < k) then begin A_ik.x := 0.0; A_ik.y := 0.0; end
              else A_ik := pA[i + k * lda];
            end;
            term := A_ik;
            ComplexMul(term, pB[k + j * ldb]);
            ComplexAdd(sum, term);
          end;
          ComplexMul(sum, alpha);
          pC_expected[i + j * ldc] := sum;
        end;
      end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B_C); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_B_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_ctrmm_strided_batched(handle, side, uplo, transA, diag, M, N, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_ctrmm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C_result, d_C, size_B_C, hipMemcpyDeviceToHost); AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for i := 0 to (stride_C * batch_count) - 1 do
    begin
      CheckEqualsRelative(h_C_expected[i].x, h_C_result[i].x, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (real)');
      CheckEqualsRelative(h_C_expected[i].y, h_C_result[i].y, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (imag)');
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C_result) then FreeMem(h_C_result);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_ZTRMM_Strided_Batched;
var
  h_A, h_B, h_C_result, h_C_expected: Procdouble_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, k, b: Integer;
  alpha: rocblas_double_complex;
  size_A, size_B_C: SizeUInt;
  pA, pB, pC_expected: Procdouble_complex;
  sum, term, A_ik: rocblas_double_complex;
const
  M = 64;
  N = 72;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  batch_count = 3;
  lda = M;
  ldb = M;
  ldc = M;
  stride_A = lda * M;
  stride_B = ldb * N;
  stride_C = ldc * N;
  tolerance: Double = 1e-12;
begin
  alpha.x := 1.25; alpha.y := -0.5;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C_result := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_double_complex);
  size_B_C := stride_B * batch_count * SizeOf(rocblas_double_complex);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B_C);
  GetMem(h_C_result, size_B_C);
  GetMem(h_C_expected, size_B_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do begin h_A[i].x := (i mod 10)/10+0.1; h_A[i].y := (i mod 20)/20; end;
    for i := 0 to (stride_B * batch_count) - 1 do begin h_B[i].x := (i mod 30)/30; h_B[i].y := (i mod 40)/40; end;

    for b := 0 to batch_count - 1 do
    begin
      pA := Procdouble_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_double_complex));
      pB := Procdouble_complex(PByte(h_B) + b * stride_B * SizeOf(rocblas_double_complex));
      pC_expected := Procdouble_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_double_complex));

      for i := 0 to M - 1 do
      begin
        for j := 0 to N - 1 do
        begin
          sum.x := 0.0; sum.y := 0.0;
          for k := 0 to M - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if (i > k) then begin A_ik.x := 0.0; A_ik.y := 0.0; end
              else A_ik := pA[i + k * lda];
            end
            else
            begin
              if (i < k) then begin A_ik.x := 0.0; A_ik.y := 0.0; end
              else A_ik := pA[i + k * lda];
            end;
            term := A_ik;
            ComplexMul(term, pB[k + j * ldb]);
            ComplexAdd(sum, term);
          end;
          ComplexMul(sum, alpha);
          pC_expected[i + j * ldc] := sum;
        end;
      end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B_C); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_B_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_ztrmm_strided_batched(handle, side, uplo, transA, diag, M, N, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_ztrmm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C_result, d_C, size_B_C, hipMemcpyDeviceToHost); AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for i := 0 to (stride_C * batch_count) - 1 do
    begin
      CheckEquals(h_C_expected[i].x, h_C_result[i].x, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (real)');
      CheckEquals(h_C_expected[i].y, h_C_result[i].y, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (imag)');
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C_result) then FreeMem(h_C_result);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_STRSM_Strided_Batched;
var
  h_A, h_X_initial, h_B_input, h_B_result, h_X_expected: PSingle;
  d_A, d_B: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, k, b: Integer;
  alpha: Single;
  size_A, size_B: SizeUInt;
  pA, pX_initial, pB_input: PSingle;
  sum, A_ik: Single;
const
  M = 64;
  N = 72;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  batch_count = 3;
  lda = M;
  ldb = M;
  stride_A = lda * M;
  stride_B = ldb * N;
  tolerance: Single = 1e-3; // Relaxed tolerance for single-precision solver
begin
  alpha := 1.25;
  d_A := nil;
  d_B := nil;
  h_A := nil;
  h_X_initial := nil;
  h_B_input := nil;
  h_B_result := nil;
  h_X_expected := nil;

  size_A := stride_A * batch_count * SizeOf(Single);
  size_B := stride_B * batch_count * SizeOf(Single);

  GetMem(h_A, size_A);
  GetMem(h_X_initial, size_B);
  GetMem(h_B_input, size_B);
  GetMem(h_B_result, size_B);
  GetMem(h_X_expected, size_B);

  try
    for b := 0 to batch_count - 1 do
      for j := 0 to M - 1 do
        for i := 0 to M - 1 do
        begin
          pA := PSingle(PByte(h_A) + b * stride_A * SizeOf(Single));
          if i = j then pA[i + j*lda] := 1.0 + (i mod 10)/10.0
          else if i < j then pA[i + j*lda] := ( (i+j) mod 20 ) / 20.0
          else pA[i + j*lda] := 0.0;
        end;
    for i := 0 to (stride_B * batch_count) - 1 do h_X_initial[i] := (i mod 50) / 50.0;

    for b := 0 to batch_count - 1 do
    begin
      pA := PSingle(PByte(h_A) + b * stride_A * SizeOf(Single));
      pX_initial := PSingle(PByte(h_X_initial) + b * stride_B * SizeOf(Single));
      pB_input := PSingle(PByte(h_B_input) + b * stride_B * SizeOf(Single));
      for i := 0 to M - 1 do
        for j := 0 to N - 1 do
        begin
          sum := 0.0;
          for k := 0 to M - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if (i > k) then A_ik := 0.0 else A_ik := pA[i + k * lda];
            end
            else
            begin
              if (i < k) then A_ik := 0.0 else A_ik := pA[i + k * lda];
            end;
            sum := sum + A_ik * pX_initial[k + j * ldb];
          end;
          pB_input[i + j * ldb] := sum;
        end;
    end;

    for i := 0 to (stride_B * batch_count) - 1 do
      h_X_expected[i] := alpha * h_X_initial[i];

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B_input, size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_strsm_strided_batched(handle, side, uplo, transA, diag, M, N, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, batch_count);
    AssertEquals('rocblas_strsm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_B_result, d_B, size_B, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H for result B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for i := 0 to (stride_B * batch_count) - 1 do
      CheckEqualsRelative(h_X_expected[i], h_B_result[i], tolerance, 'Mismatch at index ' + IntToStr(i));

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_X_initial) then FreeMem(h_X_initial);
    if Assigned(h_B_input) then FreeMem(h_B_input);
    if Assigned(h_B_result) then FreeMem(h_B_result);
    if Assigned(h_X_expected) then FreeMem(h_X_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_DTRSM_Strided_Batched;
var
  h_A, h_X_initial, h_B_input, h_B_result, h_X_expected: PDouble;
  d_A, d_B: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, k, b: Integer;
  alpha: Double;
  size_A, size_B: SizeUInt;
  pA, pX_initial, pB_input: PDouble;
  sum, A_ik: Double;
const
  M = 64;
  N = 72;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  batch_count = 3;
  lda = M;
  ldb = M;
  stride_A = lda * M;
  stride_B = ldb * N;
  tolerance: Double = 1e-11;
begin
  alpha := 1.25;
  d_A := nil;
  d_B := nil;
  h_A := nil;
  h_X_initial := nil;
  h_B_input := nil;
  h_B_result := nil;
  h_X_expected := nil;

  size_A := stride_A * batch_count * SizeOf(Double);
  size_B := stride_B * batch_count * SizeOf(Double);

  GetMem(h_A, size_A);
  GetMem(h_X_initial, size_B);
  GetMem(h_B_input, size_B);
  GetMem(h_B_result, size_B);
  GetMem(h_X_expected, size_B);

  try
    for b := 0 to batch_count - 1 do
      for j := 0 to M - 1 do
        for i := 0 to M - 1 do
        begin
          pA := PDouble(PByte(h_A) + b * stride_A * SizeOf(Double));
          if i = j then pA[i + j*lda] := 1.0 + (i mod 10)/10.0
          else if i < j then pA[i + j*lda] := ( (i+j) mod 20 ) / 20.0
          else pA[i + j*lda] := 0.0;
        end;
    for i := 0 to (stride_B * batch_count) - 1 do h_X_initial[i] := (i mod 50) / 50.0;

    for b := 0 to batch_count - 1 do
    begin
      pA := PDouble(PByte(h_A) + b * stride_A * SizeOf(Double));
      pX_initial := PDouble(PByte(h_X_initial) + b * stride_B * SizeOf(Double));
      pB_input := PDouble(PByte(h_B_input) + b * stride_B * SizeOf(Double));
      for i := 0 to M - 1 do
        for j := 0 to N - 1 do
        begin
          sum := 0.0;
          for k := 0 to M - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if (i > k) then A_ik := 0.0 else A_ik := pA[i + k * lda];
            end
            else
            begin
              if (i < k) then A_ik := 0.0 else A_ik := pA[i + k * lda];
            end;
            sum := sum + A_ik * pX_initial[k + j * ldb];
          end;
          pB_input[i + j * ldb] := sum;
        end;
    end;

    for i := 0 to (stride_B * batch_count) - 1 do
      h_X_expected[i] := alpha * h_X_initial[i];

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B_input, size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_dtrsm_strided_batched(handle, side, uplo, transA, diag, M, N, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, batch_count);
    AssertEquals('rocblas_dtrsm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_B_result, d_B, size_B, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H for result B failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for i := 0 to (stride_B * batch_count) - 1 do
      CheckEquals(h_X_expected[i], h_B_result[i], tolerance, 'Mismatch at index ' + IntToStr(i));

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_X_initial) then FreeMem(h_X_initial);
    if Assigned(h_B_input) then FreeMem(h_B_input);
    if Assigned(h_B_result) then FreeMem(h_B_result);
    if Assigned(h_X_expected) then FreeMem(h_X_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_CTRSM_Strided_Batched;
var
  h_A, h_X_initial, h_B_input, h_B_result, h_X_expected: Procfloat_complex;
  d_A, d_B: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, k, b: Integer;
  alpha: rocblas_float_complex;
  size_A, size_B: SizeUInt;
  pA, pX_initial, pB_input: Procfloat_complex;
  sum, term, A_ik: rocblas_float_complex;
const
  M = 64;
  N = 72;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  batch_count = 3;
  lda = M;
  ldb = M;
  stride_A = lda * M;
  stride_B = ldb * N;
  tolerance: Single = 1e-2; // Further relaxed tolerance for complex single-precision solver
begin
  alpha.x := 1.25; alpha.y := -0.5;
  d_A := nil;
  d_B := nil;
  h_A := nil;
  h_X_initial := nil;
  h_B_input := nil;
  h_B_result := nil;
  h_X_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_float_complex);
  size_B := stride_B * batch_count * SizeOf(rocblas_float_complex);

  GetMem(h_A, size_A);
  GetMem(h_X_initial, size_B);
  GetMem(h_B_input, size_B);
  GetMem(h_B_result, size_B);
  GetMem(h_X_expected, size_B);

  try
    for b := 0 to batch_count - 1 do
      for j := 0 to M - 1 do
        for i := 0 to M - 1 do
        begin
          pA := Procfloat_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_float_complex));
          if i = j then begin pA[i + j*lda].x := 1.0 + (i mod 10)/10.0; pA[i + j*lda].y := 0.1; end
          else if i < j then begin pA[i + j*lda].x := ((i+j) mod 20)/20.0; pA[i + j*lda].y := ((i+j) mod 15)/15.0; end
          else begin pA[i + j*lda].x := 0.0; pA[i + j*lda].y := 0.0; end;
        end;
    for i := 0 to (stride_B * batch_count) - 1 do begin h_X_initial[i].x := (i mod 50)/50.0; h_X_initial[i].y := (i mod 30)/30.0; end;

    for b := 0 to batch_count - 1 do
    begin
      pA := Procfloat_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_float_complex));
      pX_initial := Procfloat_complex(PByte(h_X_initial) + b * stride_B * SizeOf(rocblas_float_complex));
      pB_input := Procfloat_complex(PByte(h_B_input) + b * stride_B * SizeOf(rocblas_float_complex));
      for i := 0 to M - 1 do
        for j := 0 to N - 1 do
        begin
          sum.x := 0.0; sum.y := 0.0;
          for k := 0 to M - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if (i > k) then begin A_ik.x := 0.0; A_ik.y := 0.0; end
              else A_ik := pA[i + k * lda];
            end
            else
            begin
              if (i < k) then begin A_ik.x := 0.0; A_ik.y := 0.0; end
              else A_ik := pA[i + k * lda];
            end;
            term := A_ik;
            ComplexMul(term, pX_initial[k + j * ldb]);
            ComplexAdd(sum, term);
          end;
          pB_input[i + j * ldb] := sum;
        end;
    end;

    for i := 0 to (stride_B * batch_count) - 1 do
    begin
      h_X_expected[i] := h_X_initial[i];
      ComplexMul(h_X_expected[i], alpha);
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B_input, size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_ctrsm_strided_batched(handle, side, uplo, transA, diag, M, N, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, batch_count);
    AssertEquals('rocblas_ctrsm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_B_result, d_B, size_B, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for i := 0 to (stride_B * batch_count) - 1 do
    begin
      CheckEqualsRelative(h_X_expected[i].x, h_B_result[i].x, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (real)');
      CheckEqualsRelative(h_X_expected[i].y, h_B_result[i].y, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (imag)');
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_X_initial) then FreeMem(h_X_initial);
    if Assigned(h_B_input) then FreeMem(h_B_input);
    if Assigned(h_B_result) then FreeMem(h_B_result);
    if Assigned(h_X_expected) then FreeMem(h_X_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_ZTRSM_Strided_Batched;
var
  h_A, h_X_initial, h_B_input, h_B_result, h_X_expected: Procdouble_complex;
  d_A, d_B: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, k, b: Integer;
  alpha: rocblas_double_complex;
  size_A, size_B: SizeUInt;
  pA, pX_initial, pB_input: Procdouble_complex;
  sum, term, A_ik: rocblas_double_complex;
const
  M = 64;
  N = 72;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  batch_count = 3;
  lda = M;
  ldb = M;
  stride_A = lda * M;
  stride_B = ldb * N;
  tolerance: Double = 1e-11;
begin
  alpha.x := 1.25; alpha.y := -0.5;
  d_A := nil;
  d_B := nil;
  h_A := nil;
  h_X_initial := nil;
  h_B_input := nil;
  h_B_result := nil;
  h_X_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_double_complex);
  size_B := stride_B * batch_count * SizeOf(rocblas_double_complex);

  GetMem(h_A, size_A);
  GetMem(h_X_initial, size_B);
  GetMem(h_B_input, size_B);
  GetMem(h_B_result, size_B);
  GetMem(h_X_expected, size_B);

  try
    for b := 0 to batch_count - 1 do
      for j := 0 to M - 1 do
        for i := 0 to M - 1 do
        begin
          pA := Procdouble_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_double_complex));
          if i = j then begin pA[i + j*lda].x := 1.0 + (i mod 10)/10.0; pA[i + j*lda].y := 0.1; end
          else if i < j then begin pA[i + j*lda].x := ((i+j) mod 20)/20.0; pA[i + j*lda].y := ((i+j) mod 15)/15.0; end
          else begin pA[i + j*lda].x := 0.0; pA[i + j*lda].y := 0.0; end;
        end;
    for i := 0 to (stride_B * batch_count) - 1 do begin h_X_initial[i].x := (i mod 50)/50.0; h_X_initial[i].y := (i mod 30)/30.0; end;

    for b := 0 to batch_count - 1 do
    begin
      pA := Procdouble_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_double_complex));
      pX_initial := Procdouble_complex(PByte(h_X_initial) + b * stride_B * SizeOf(rocblas_double_complex));
      pB_input := Procdouble_complex(PByte(h_B_input) + b * stride_B * SizeOf(rocblas_double_complex));
      for i := 0 to M - 1 do
        for j := 0 to N - 1 do
        begin
          sum.x := 0.0; sum.y := 0.0;
          for k := 0 to M - 1 do
          begin
            if uplo = rocblas_fill_upper then
            begin
              if (i > k) then begin A_ik.x := 0.0; A_ik.y := 0.0; end
              else A_ik := pA[i + k * lda];
            end
            else
            begin
              if (i < k) then begin A_ik.x := 0.0; A_ik.y := 0.0; end
              else A_ik := pA[i + k * lda];
            end;
            term := A_ik;
            ComplexMul(term, pX_initial[k + j * ldb]);
            ComplexAdd(sum, term);
          end;
          pB_input[i + j * ldb] := sum;
        end;
    end;

    for i := 0 to (stride_B * batch_count) - 1 do
    begin
        h_X_expected[i] := h_X_initial[i];
        ComplexMul(h_X_expected[i], alpha);
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B_input, size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_ztrsm_strided_batched(handle, side, uplo, transA, diag, M, N, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, batch_count);
    AssertEquals('rocblas_ztrsm_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_B_result, d_B, size_B, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for i := 0 to (stride_B * batch_count) - 1 do
    begin
      CheckEquals(h_X_expected[i].x, h_B_result[i].x, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (real)');
      CheckEquals(h_X_expected[i].y, h_B_result[i].y, tolerance, 'Mismatch at index ' + IntToStr(i) + ' (imag)');
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_X_initial) then FreeMem(h_X_initial);
    if Assigned(h_B_input) then FreeMem(h_B_input);
    if Assigned(h_B_result) then FreeMem(h_B_result);
    if Assigned(h_X_expected) then FreeMem(h_X_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_SSYRK_Strided_Batched;
var
  h_A, h_C, h_C_expected: PSingle;
  d_A, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, l, b: Integer;
  alpha, beta: Single;
  size_A, size_C: SizeUInt;
  pA, pC, pC_expected: PSingle;
  sum: Single;
const
  N = 64;
  K = 72;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  batch_count = 3;
  lda = N;
  ldc = N;
  stride_A = lda * K;
  stride_C = ldc * N;
  tolerance: Single = 1e-5;
begin
  alpha := 1.25;
  beta := 0.75;
  d_A := nil;
  d_C := nil;
  h_A := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(Single);
  size_C := stride_C * batch_count * SizeOf(Single);

  GetMem(h_A, size_A);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do h_A[i] := (i mod 100) / 100.0;
    for i := 0 to (stride_C * batch_count) - 1 do h_C[i] := (i mod 100) / 100.0;
    Move(h_C^, h_C_expected^, size_C);

    // CPU-based SYRK: C = alpha*A*A^T + beta*C
    for b := 0 to batch_count - 1 do
    begin
      pA := PSingle(PByte(h_A) + b * stride_A * SizeOf(Single));
      pC_expected := PSingle(PByte(h_C_expected) + b * stride_C * SizeOf(Single));
      for j := 0 to N - 1 do
        for i := 0 to j do // Only compute upper triangle
        begin
          sum := 0.0;
          for l := 0 to K - 1 do
            sum := sum + pA[i + l * lda] * pA[j + l * lda];
          pC_expected[i + j * ldc] := alpha * sum + beta * pC_expected[i + j * ldc];
        end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_ssyrk_strided_batched(handle, uplo, transA, N, K, @alpha,
      d_A, lda, stride_A, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_ssyrk_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for b := 0 to batch_count - 1 do
    begin
      pC := PSingle(PByte(h_C) + b * stride_C * SizeOf(Single));
      pC_expected := PSingle(PByte(h_C_expected) + b * stride_C * SizeOf(Single));
      for j := 0 to N - 1 do
        for i := 0 to j do // Compare upper triangle
          CheckEqualsRelative(pC_expected[i + j * ldc], pC[i + j * ldc], tolerance,
            Format('Mismatch at batch %d, index (%d, %d)', [b, i, j]));
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_DSYRK_Strided_Batched;
var
  h_A, h_C, h_C_expected: PDouble;
  d_A, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, l, b: Integer;
  alpha, beta: Double;
  size_A, size_C: SizeUInt;
  pA, pC, pC_expected: PDouble;
  sum: Double;
const
  N = 64;
  K = 72;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  batch_count = 3;
  lda = N;
  ldc = N;
  stride_A = lda * K;
  stride_C = ldc * N;
  tolerance: Double = 1e-12;
begin
  alpha := 1.25;
  beta := 0.75;
  d_A := nil;
  d_C := nil;
  h_A := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(Double);
  size_C := stride_C * batch_count * SizeOf(Double);

  GetMem(h_A, size_A);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do h_A[i] := (i mod 100) / 100.0;
    for i := 0 to (stride_C * batch_count) - 1 do h_C[i] := (i mod 100) / 100.0;
    Move(h_C^, h_C_expected^, size_C);

    for b := 0 to batch_count - 1 do
    begin
      pA := PDouble(PByte(h_A) + b * stride_A * SizeOf(Double));
      pC_expected := PDouble(PByte(h_C_expected) + b * stride_C * SizeOf(Double));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          sum := 0.0;
          for l := 0 to K - 1 do
            sum := sum + pA[i + l * lda] * pA[j + l * lda];
          pC_expected[i + j * ldc] := alpha * sum + beta * pC_expected[i + j * ldc];
        end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_dsyrk_strided_batched(handle, uplo, transA, N, K, @alpha,
      d_A, lda, stride_A, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_dsyrk_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for b := 0 to batch_count - 1 do
    begin
      pC := PDouble(PByte(h_C) + b * stride_C * SizeOf(Double));
      pC_expected := PDouble(PByte(h_C_expected) + b * stride_C * SizeOf(Double));
      for j := 0 to N - 1 do
        for i := 0 to j do
          CheckEquals(pC_expected[i + j * ldc], pC[i + j * ldc], tolerance,
            Format('Mismatch at batch %d, index (%d, %d)', [b, i, j]));
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_CSYRK_Strided_Batched;
var
  h_A, h_C, h_C_expected: Procfloat_complex;
  d_A, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, l, b: Integer;
  alpha, beta: rocblas_float_complex;
  size_A, size_C: SizeUInt;
  pA, pC, pC_expected: Procfloat_complex;
  sum, term, beta_C: rocblas_float_complex;
const
  N = 64;
  K = 72;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  batch_count = 3;
  lda = N;
  ldc = N;
  stride_A = lda * K;
  stride_C = ldc * N;
  tolerance: Single = 1e-5;
begin
  alpha.x := 1.25; alpha.y := -0.5;
  beta.x  := 0.75; beta.y  := 0.25;
  d_A := nil;
  d_C := nil;
  h_A := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_float_complex);
  size_C := stride_C * batch_count * SizeOf(rocblas_float_complex);

  GetMem(h_A, size_A);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do begin h_A[i].x := (i mod 10)/10; h_A[i].y := (i mod 20)/20; end;
    for i := 0 to (stride_C * batch_count) - 1 do begin h_C[i].x := (i mod 30)/30; h_C[i].y := (i mod 40)/40; end;
    Move(h_C^, h_C_expected^, size_C);

    for b := 0 to batch_count - 1 do
    begin
      pA := Procfloat_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_float_complex));
      pC_expected := Procfloat_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_float_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          sum.x := 0.0; sum.y := 0.0;
          for l := 0 to K - 1 do
          begin
            term := pA[i + l * lda];
            ComplexMul(term, pA[j + l * lda]);
            ComplexAdd(sum, term);
          end;
          beta_C := pC_expected[i + j * ldc];
          ComplexMul(beta_C, beta);
          ComplexMul(sum, alpha);
          ComplexAdd(sum, beta_C);
          pC_expected[i + j * ldc] := sum;
        end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_csyrk_strided_batched(handle, uplo, transA, N, K, @alpha,
      d_A, lda, stride_A, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_csyrk_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for b := 0 to batch_count - 1 do
    begin
      pC := Procfloat_complex(PByte(h_C) + b * stride_C * SizeOf(rocblas_float_complex));
      pC_expected := Procfloat_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_float_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          CheckEqualsRelative(pC_expected[i+j*ldc].x, pC[i+j*ldc].x, tolerance, Format('Mismatch at batch %d, index (%d,%d) (real)',[b,i,j]));
          CheckEqualsRelative(pC_expected[i+j*ldc].y, pC[i+j*ldc].y, tolerance, Format('Mismatch at batch %d, index (%d,%d) (imag)',[b,i,j]));
        end;
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_ZSYRK_Strided_Batched;
var
  h_A, h_C, h_C_expected: Procdouble_complex;
  d_A, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, l, b: Integer;
  alpha, beta: rocblas_double_complex;
  size_A, size_C: SizeUInt;
  pA, pC, pC_expected: Procdouble_complex;
  sum, term, beta_C: rocblas_double_complex;
const
  N = 64;
  K = 72;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  batch_count = 3;
  lda = N;
  ldc = N;
  stride_A = lda * K;
  stride_C = ldc * N;
  tolerance: Double = 1e-12;
begin
  alpha.x := 1.25; alpha.y := -0.5;
  beta.x  := 0.75; beta.y  := 0.25;
  d_A := nil;
  d_C := nil;
  h_A := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_double_complex);
  size_C := stride_C * batch_count * SizeOf(rocblas_double_complex);

  GetMem(h_A, size_A);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do begin h_A[i].x := (i mod 10)/10; h_A[i].y := (i mod 20)/20; end;
    for i := 0 to (stride_C * batch_count) - 1 do begin h_C[i].x := (i mod 30)/30; h_C[i].y := (i mod 40)/40; end;
    Move(h_C^, h_C_expected^, size_C);

    for b := 0 to batch_count - 1 do
    begin
      pA := Procdouble_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_double_complex));
      pC_expected := Procdouble_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_double_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          sum.x := 0.0; sum.y := 0.0;
          for l := 0 to K - 1 do
          begin
            term := pA[i + l * lda];
            ComplexMul(term, pA[j + l * lda]);
            ComplexAdd(sum, term);
          end;
          beta_C := pC_expected[i + j * ldc];
          ComplexMul(beta_C, beta);
          ComplexMul(sum, alpha);
          ComplexAdd(sum, beta_C);
          pC_expected[i + j * ldc] := sum;
        end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_zsyrk_strided_batched(handle, uplo, transA, N, K, @alpha,
      d_A, lda, stride_A, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_zsyrk_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for b := 0 to batch_count - 1 do
    begin
      pC := Procdouble_complex(PByte(h_C) + b * stride_C * SizeOf(rocblas_double_complex));
      pC_expected := Procdouble_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_double_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          CheckEquals(pC_expected[i+j*ldc].x, pC[i+j*ldc].x, tolerance, Format('Mismatch at batch %d, index (%d,%d) (real)',[b,i,j]));
          CheckEquals(pC_expected[i+j*ldc].y, pC[i+j*ldc].y, tolerance, Format('Mismatch at batch %d, index (%d,%d) (imag)',[b,i,j]));
        end;
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_CHERK_Strided_Batched;
var
  h_A, h_C, h_C_expected: Procfloat_complex;
  d_A, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, l, b: Integer;
  alpha, beta: Single;
  size_A, size_C: SizeUInt;
  pA, pC, pC_initial, pC_expected: Procfloat_complex;
  sum, term: rocblas_float_complex;
const
  N = 64;
  K = 72;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  batch_count = 3;
  lda = N;
  ldc = N;
  stride_A = lda * K;
  stride_C = ldc * N;
  tolerance: Single = 1e-4; // Relaxed tolerance for single-precision Hermitian operation
begin
  alpha := 1.25;
  beta := 0.75;
  d_A := nil;
  d_C := nil;
  h_A := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_float_complex);
  size_C := stride_C * batch_count * SizeOf(rocblas_float_complex);

  GetMem(h_A, size_A);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do begin h_A[i].x := (i mod 10)/10.0; h_A[i].y := (i mod 20)/20.0; end;
    for i := 0 to (stride_C * batch_count) - 1 do begin h_C[i].x := (i mod 30)/30.0; h_C[i].y := (i mod 40)/40.0; end;
    Move(h_C^, h_C_expected^, size_C);

    // CPU-based HERK: C = alpha*A*A^H + beta*C
    for b := 0 to batch_count - 1 do
    begin
      pA := Procfloat_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_float_complex));
      pC_expected := Procfloat_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_float_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do // Only compute upper triangle
        begin
          sum.x := 0.0; sum.y := 0.0;
          for l := 0 to K - 1 do
          begin
            // term = A(i,l) * conj(A(j,l))
            term.x := pA[i + l*lda].x * pA[j + l*lda].x - pA[i + l*lda].y * (-pA[j + l*lda].y);
            term.y := pA[i + l*lda].x * (-pA[j + l*lda].y) + pA[i + l*lda].y * pA[j + l*lda].x;
            ComplexAdd(sum, term);
          end;
          pC_initial := @pC_expected[i + j * ldc];
          pC_initial^.x := alpha * sum.x + beta * pC_initial^.x;
          pC_initial^.y := alpha * sum.y + beta * pC_initial^.y;
          if i = j then pC_initial^.y := 0.0; // Diagonal of Hermitian matrix is real
        end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_cherk_strided_batched(handle, uplo, transA, N, K, @alpha,
      d_A, lda, stride_A, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_cherk_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for b := 0 to batch_count - 1 do
    begin
      pC := Procfloat_complex(PByte(h_C) + b * stride_C * SizeOf(rocblas_float_complex));
      pC_expected := Procfloat_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_float_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          CheckEqualsRelative(pC_expected[i+j*ldc].x, pC[i+j*ldc].x, tolerance, Format('Mismatch at batch %d, index (%d,%d) (real)',[b,i,j]));
          CheckEqualsRelative(pC_expected[i+j*ldc].y, pC[i+j*ldc].y, tolerance, Format('Mismatch at batch %d, index (%d,%d) (imag)',[b,i,j]));
        end;
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_ZHERK_Strided_Batched;
var
  h_A, h_C, h_C_expected: Procdouble_complex;
  d_A, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, l, b: Integer;
  alpha, beta: Double;
  size_A, size_C: SizeUInt;
  pA, pC, pC_initial, pC_expected: Procdouble_complex;
  sum, term: rocblas_double_complex;
const
  N = 64;
  K = 72;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  batch_count = 3;
  lda = N;
  ldc = N;
  stride_A = lda * K;
  stride_C = ldc * N;
  tolerance: Double = 1e-12;
begin
  alpha := 1.25;
  beta := 0.75;
  d_A := nil;
  d_C := nil;
  h_A := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_double_complex);
  size_C := stride_C * batch_count * SizeOf(rocblas_double_complex);

  GetMem(h_A, size_A);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do begin h_A[i].x := (i mod 10)/10.0; h_A[i].y := (i mod 20)/20.0; end;
    for i := 0 to (stride_C * batch_count) - 1 do begin h_C[i].x := (i mod 30)/30.0; h_C[i].y := (i mod 40)/40.0; end;
    Move(h_C^, h_C_expected^, size_C);

    for b := 0 to batch_count - 1 do
    begin
      pA := Procdouble_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_double_complex));
      pC_expected := Procdouble_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_double_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          sum.x := 0.0; sum.y := 0.0;
          for l := 0 to K - 1 do
          begin
            term.x := pA[i + l*lda].x * pA[j + l*lda].x - pA[i + l*lda].y * (-pA[j + l*lda].y);
            term.y := pA[i + l*lda].x * (-pA[j + l*lda].y) + pA[i + l*lda].y * pA[j + l*lda].x;
            ComplexAdd(sum, term);
          end;
          pC_initial := @pC_expected[i + j * ldc];
          pC_initial^.x := alpha * sum.x + beta * pC_initial^.x;
          pC_initial^.y := alpha * sum.y + beta * pC_initial^.y;
          if i = j then pC_initial^.y := 0.0;
        end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_zherk_strided_batched(handle, uplo, transA, N, K, @alpha,
      d_A, lda, stride_A, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_zherk_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for b := 0 to batch_count - 1 do
    begin
      pC := Procdouble_complex(PByte(h_C) + b * stride_C * SizeOf(rocblas_double_complex));
      pC_expected := Procdouble_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_double_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          CheckEquals(pC_expected[i+j*ldc].x, pC[i+j*ldc].x, tolerance, Format('Mismatch at batch %d, index (%d,%d) (real)',[b,i,j]));
          CheckEquals(pC_expected[i+j*ldc].y, pC[i+j*ldc].y, tolerance, Format('Mismatch at batch %d, index (%d,%d) (imag)',[b,i,j]));
        end;
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_SSyr2k_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: PSingle;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, l, b: Integer;
  alpha, beta: Single;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC, pC_expected: PSingle;
  sum_ABT, sum_BAT: Single;
const
  N = 64;
  K = 72;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  batch_count = 3;
  lda = N;
  ldb = N;
  ldc = N;
  stride_A = lda * K;
  stride_B = ldb * K;
  stride_C = ldc * N;
  tolerance: Single = 1e-5;
begin
  alpha := 1.25;
  beta := 0.75;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(Single);
  size_B := stride_B * batch_count * SizeOf(Single);
  size_C := stride_C * batch_count * SizeOf(Single);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do h_A[i] := (i mod 100) / 100.0;
    for i := 0 to (stride_B * batch_count) - 1 do h_B[i] := (i mod 100) / 100.0;
    for i := 0 to (stride_C * batch_count) - 1 do h_C[i] := (i mod 100) / 100.0;
    Move(h_C^, h_C_expected^, size_C);

    // CPU-based SYR2K: C = alpha*(A*B^T + B*A^T) + beta*C
    for b := 0 to batch_count - 1 do
    begin
      pA := PSingle(PByte(h_A) + b * stride_A * SizeOf(Single));
      pB := PSingle(PByte(h_B) + b * stride_B * SizeOf(Single));
      pC_expected := PSingle(PByte(h_C_expected) + b * stride_C * SizeOf(Single));
      for j := 0 to N - 1 do
        for i := 0 to j do // Only compute upper triangle
        begin
          sum_ABT := 0.0;
          sum_BAT := 0.0;
          for l := 0 to K - 1 do
          begin
            sum_ABT := sum_ABT + pA[i + l * lda] * pB[j + l * ldb];
            sum_BAT := sum_BAT + pB[i + l * ldb] * pA[j + l * lda];
          end;
          pC_expected[i + j * ldc] := alpha * (sum_ABT + sum_BAT) + beta * pC_expected[i + j * ldc];
        end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_ssyr2k_strided_batched(handle, uplo, transA, N, K, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_ssyr2k_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for b := 0 to batch_count - 1 do
    begin
      pC := PSingle(PByte(h_C) + b * stride_C * SizeOf(Single));
      pC_expected := PSingle(PByte(h_C_expected) + b * stride_C * SizeOf(Single));
      for j := 0 to N - 1 do
        for i := 0 to j do
          CheckEqualsRelative(pC_expected[i + j * ldc], pC[i + j * ldc], tolerance,
            Format('Mismatch at batch %d, index (%d, %d)', [b, i, j]));
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_DSyr2k_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: PDouble;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, l, b: Integer;
  alpha, beta: Double;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC, pC_expected: PDouble;
  sum_ABT, sum_BAT: Double;
const
  N = 64;
  K = 72;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  batch_count = 3;
  lda = N;
  ldb = N;
  ldc = N;
  stride_A = lda * K;
  stride_B = ldb * K;
  stride_C = ldc * N;
  tolerance: Double = 1e-12;
begin
  alpha := 1.25;
  beta := 0.75;
  d_A := nil;
  d_B := nil;
  d_C := nil;
  h_A := nil;
  h_B := nil;
  h_C := nil;
  h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(Double);
  size_B := stride_B * batch_count * SizeOf(Double);
  size_C := stride_C * batch_count * SizeOf(Double);

  GetMem(h_A, size_A);
  GetMem(h_B, size_B);
  GetMem(h_C, size_C);
  GetMem(h_C_expected, size_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do h_A[i] := (i mod 100) / 100.0;
    for i := 0 to (stride_B * batch_count) - 1 do h_B[i] := (i mod 100) / 100.0;
    for i := 0 to (stride_C * batch_count) - 1 do h_C[i] := (i mod 100) / 100.0;
    Move(h_C^, h_C_expected^, size_C);

    for b := 0 to batch_count - 1 do
    begin
      pA := PDouble(PByte(h_A) + b * stride_A * SizeOf(Double));
      pB := PDouble(PByte(h_B) + b * stride_B * SizeOf(Double));
      pC_expected := PDouble(PByte(h_C_expected) + b * stride_C * SizeOf(Double));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          sum_ABT := 0.0;
          sum_BAT := 0.0;
          for l := 0 to K - 1 do
          begin
            sum_ABT := sum_ABT + pA[i + l * lda] * pB[j + l * ldb];
            sum_BAT := sum_BAT + pB[i + l * ldb] * pA[j + l * lda];
          end;
          pC_expected[i + j * ldc] := alpha * (sum_ABT + sum_BAT) + beta * pC_expected[i + j * ldc];
        end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H->D C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_dsyr2k_strided_batched(handle, uplo, transA, N, K, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_dsyr2k_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for b := 0 to batch_count - 1 do
    begin
      pC := PDouble(PByte(h_C) + b * stride_C * SizeOf(Double));
      pC_expected := PDouble(PByte(h_C_expected) + b * stride_C * SizeOf(Double));
      for j := 0 to N - 1 do
        for i := 0 to j do
          CheckEquals(pC_expected[i + j * ldc], pC[i + j * ldc], tolerance,
            Format('Mismatch at batch %d, index (%d, %d)', [b, i, j]));
    end;

  finally
    if d_A <> nil then hipFree(d_A);
    if d_B <> nil then hipFree(d_B);
    if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A);
    if Assigned(h_B) then FreeMem(h_B);
    if Assigned(h_C) then FreeMem(h_C);
    if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_CSyr2k_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: Procfloat_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, l, b: Integer;
  alpha, beta: rocblas_float_complex;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC, pC_expected: Procfloat_complex;
  sum, sum_ABT, sum_BAT, term1, term2, beta_C: rocblas_float_complex;
const
  N = 64;
  K = 72;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  batch_count = 3;
  lda = N;
  ldb = N;
  ldc = N;
  stride_A = lda * K;
  stride_B = ldb * K;
  stride_C = ldc * N;
  tolerance: Single = 1e-5;
begin
  alpha.x := 1.25; alpha.y := -0.5;
  beta.x  := 0.75; beta.y  := 0.25;
  d_A := nil; d_B := nil; d_C := nil;
  h_A := nil; h_B := nil; h_C := nil; h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_float_complex);
  size_B := stride_B * batch_count * SizeOf(rocblas_float_complex);
  size_C := stride_C * batch_count * SizeOf(rocblas_float_complex);

  GetMem(h_A, size_A); GetMem(h_B, size_B); GetMem(h_C, size_C); GetMem(h_C_expected, size_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do begin h_A[i].x := (i mod 10)/10.0; h_A[i].y := (i mod 20)/20.0; end;
    for i := 0 to (stride_B * batch_count) - 1 do begin h_B[i].x := (i mod 30)/30.0; h_B[i].y := (i mod 40)/40.0; end;
    for i := 0 to (stride_C * batch_count) - 1 do begin h_C[i].x := (i mod 50)/50.0; h_C[i].y := (i mod 60)/60.0; end;
    Move(h_C^, h_C_expected^, size_C);

    for b := 0 to batch_count - 1 do
    begin
      pA := Procfloat_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_float_complex));
      pB := Procfloat_complex(PByte(h_B) + b * stride_B * SizeOf(rocblas_float_complex));
      pC_expected := Procfloat_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_float_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          sum_ABT.x := 0.0; sum_ABT.y := 0.0;
          sum_BAT.x := 0.0; sum_BAT.y := 0.0;
          for l := 0 to K - 1 do
          begin
            term1 := pA[i + l * lda]; ComplexMul(term1, pB[j + l * ldb]); ComplexAdd(sum_ABT, term1);
            term2 := pB[i + l * ldb]; ComplexMul(term2, pA[j + l * lda]); ComplexAdd(sum_BAT, term2);
          end;
          sum := sum_ABT;
          ComplexAdd(sum, sum_BAT);
          ComplexMul(sum, alpha);
          beta_C := pC_expected[i + j * ldc];
          ComplexMul(beta_C, beta);
          ComplexAdd(sum, beta_C);
          pC_expected[i + j * ldc] := sum;
        end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_csyr2k_strided_batched(handle, uplo, transA, N, K, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_csyr2k_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost); AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for b := 0 to batch_count - 1 do
    begin
      pC := Procfloat_complex(PByte(h_C) + b * stride_C * SizeOf(rocblas_float_complex));
      pC_expected := Procfloat_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_float_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          CheckEqualsRelative(pC_expected[i+j*ldc].x, pC[i+j*ldc].x, tolerance, Format('Mismatch at batch %d, index (%d,%d) (real)',[b,i,j]));
          CheckEqualsRelative(pC_expected[i+j*ldc].y, pC[i+j*ldc].y, tolerance, Format('Mismatch at batch %d, index (%d,%d) (imag)',[b,i,j]));
        end;
    end;

  finally
    if d_A <> nil then hipFree(d_A); if d_B <> nil then hipFree(d_B); if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A); if Assigned(h_B) then FreeMem(h_B); if Assigned(h_C) then FreeMem(h_C); if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_ZSyr2k_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: Procdouble_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, l, b: Integer;
  alpha, beta: rocblas_double_complex;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC, pC_expected: Procdouble_complex;
  sum, sum_ABT, sum_BAT, term1, term2, beta_C: rocblas_double_complex;
const
  N = 64;
  K = 72;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  batch_count = 3;
  lda = N;
  ldb = N;
  ldc = N;
  stride_A = lda * K;
  stride_B = ldb * K;
  stride_C = ldc * N;
  tolerance: Double = 1e-12;
begin
  alpha.x := 1.25; alpha.y := -0.5;
  beta.x  := 0.75; beta.y  := 0.25;
  d_A := nil; d_B := nil; d_C := nil;
  h_A := nil; h_B := nil; h_C := nil; h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_double_complex);
  size_B := stride_B * batch_count * SizeOf(rocblas_double_complex);
  size_C := stride_C * batch_count * SizeOf(rocblas_double_complex);

  GetMem(h_A, size_A); GetMem(h_B, size_B); GetMem(h_C, size_C); GetMem(h_C_expected, size_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do begin h_A[i].x := (i mod 10)/10.0; h_A[i].y := (i mod 20)/20.0; end;
    for i := 0 to (stride_B * batch_count) - 1 do begin h_B[i].x := (i mod 30)/30.0; h_B[i].y := (i mod 40)/40.0; end;
    for i := 0 to (stride_C * batch_count) - 1 do begin h_C[i].x := (i mod 50)/50.0; h_C[i].y := (i mod 60)/60.0; end;
    Move(h_C^, h_C_expected^, size_C);

    for b := 0 to batch_count - 1 do
    begin
      pA := Procdouble_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_double_complex));
      pB := Procdouble_complex(PByte(h_B) + b * stride_B * SizeOf(rocblas_double_complex));
      pC_expected := Procdouble_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_double_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          sum_ABT.x := 0.0; sum_ABT.y := 0.0;
          sum_BAT.x := 0.0; sum_BAT.y := 0.0;
          for l := 0 to K - 1 do
          begin
            term1 := pA[i + l * lda]; ComplexMul(term1, pB[j + l * ldb]); ComplexAdd(sum_ABT, term1);
            term2 := pB[i + l * ldb]; ComplexMul(term2, pA[j + l * lda]); ComplexAdd(sum_BAT, term2);
          end;
          sum := sum_ABT;
          ComplexAdd(sum, sum_BAT);
          ComplexMul(sum, alpha);
          beta_C := pC_expected[i + j * ldc];
          ComplexMul(beta_C, beta);
          ComplexAdd(sum, beta_C);
          pC_expected[i + j * ldc] := sum;
        end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_zsyr2k_strided_batched(handle, uplo, transA, N, K, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_zsyr2k_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost); AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for b := 0 to batch_count - 1 do
    begin
      pC := Procdouble_complex(PByte(h_C) + b * stride_C * SizeOf(rocblas_double_complex));
      pC_expected := Procdouble_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_double_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          CheckEquals(pC_expected[i+j*ldc].x, pC[i+j*ldc].x, tolerance, Format('Mismatch at batch %d, index (%d,%d) (real)',[b,i,j]));
          CheckEquals(pC_expected[i+j*ldc].y, pC[i+j*ldc].y, tolerance, Format('Mismatch at batch %d, index (%d,%d) (imag)',[b,i,j]));
        end;
    end;

  finally
    if d_A <> nil then hipFree(d_A); if d_B <> nil then hipFree(d_B); if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A); if Assigned(h_B) then FreeMem(h_B); if Assigned(h_C) then FreeMem(h_C); if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

procedure TTestRocBLAS_Level3_Strided_Batched.Test_CHER2K_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: Procfloat_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, l, b: Integer;
  alpha, conj_alpha: rocblas_float_complex;
  beta: Single;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC, pC_expected: Procfloat_complex;
  sum_ABH, sum_BAH, term1, term2, final_sum: rocblas_float_complex;
  conjA, conjB: rocblas_float_complex;
  expected_x, actual_x, expected_y, actual_y: Single;
const
  N = 64;
  K = 72;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  batch_count = 3;
  lda = N;
  ldb = N;
  ldc = N;
  stride_A = lda * K;
  stride_B = ldb * K;
  stride_C = ldc * N;
  relative_tolerance: Single = 2e-3;
  absolute_tolerance: Single = 1e-5; // Relaxed for diagonal numerical noise
begin
  alpha.x := 1.25; alpha.y := -0.5;
  beta := 0.75;
  d_A := nil; d_B := nil; d_C := nil;
  h_A := nil; h_B := nil; h_C := nil; h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_float_complex);
  size_B := stride_B * batch_count * SizeOf(rocblas_float_complex);
  size_C := stride_C * batch_count * SizeOf(rocblas_float_complex);

  GetMem(h_A, size_A); GetMem(h_B, size_B); GetMem(h_C, size_C); GetMem(h_C_expected, size_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do begin h_A[i].x := (i mod 10)/10.0; h_A[i].y := (i mod 20)/20.0; end;
    for i := 0 to (stride_B * batch_count) - 1 do begin h_B[i].x := (i mod 30)/30.0; h_B[i].y := (i mod 40)/40.0; end;
    for i := 0 to (stride_C * batch_count) - 1 do begin h_C[i].x := (i mod 50)/50.0; h_C[i].y := (i mod 60)/60.0; end;
    Move(h_C^, h_C_expected^, size_C);

    conj_alpha := alpha;
    conj_alpha.y := -alpha.y;

    for b := 0 to batch_count - 1 do
    begin
      pA := Procfloat_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_float_complex));
      pB := Procfloat_complex(PByte(h_B) + b * stride_B * SizeOf(rocblas_float_complex));
      pC_expected := Procfloat_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_float_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          sum_ABH.x := 0.0; sum_ABH.y := 0.0;
          sum_BAH.x := 0.0; sum_BAH.y := 0.0;
          for l := 0 to K - 1 do
          begin
            term1 := pA[i + l * lda];
            conjB := pB[j + l * ldb];
            conjB.y := -conjB.y;
            ComplexMul(term1, conjB);
            ComplexAdd(sum_ABH, term1);

            term2 := pB[i + l * ldb];
            conjA := pA[j + l * lda];
            conjA.y := -conjA.y;
            ComplexMul(term2, conjA);
            ComplexAdd(sum_BAH, term2);
          end;
          ComplexMul(sum_ABH, alpha);
          ComplexMul(sum_BAH, conj_alpha);
          final_sum := sum_ABH;
          ComplexAdd(final_sum, sum_BAH);
          final_sum.x := final_sum.x + beta * pC_expected[i + j * ldc].x;
          final_sum.y := final_sum.y + beta * pC_expected[i + j * ldc].y;
          if i = j then final_sum.y := 0.0;
          pC_expected[i + j * ldc] := final_sum;
        end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_cher2k_strided_batched(handle, uplo, transA, N, K, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_cher2k_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost); AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for b := 0 to batch_count - 1 do
    begin
      pC := Procfloat_complex(PByte(h_C) + b * stride_C * SizeOf(rocblas_float_complex));
      pC_expected := Procfloat_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_float_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          expected_x := pC_expected[i+j*ldc].x;
          actual_x := pC[i+j*ldc].x;
          expected_y := pC_expected[i+j*ldc].y;
          actual_y := pC[i+j*ldc].y;

          if Abs(expected_x) < absolute_tolerance then
            CheckEquals(expected_x, actual_x, absolute_tolerance, Format('Mismatch at batch %d, index (%d,%d) (real)',[b,i,j]))
          else
            CheckEqualsRelative(expected_x, actual_x, relative_tolerance, Format('Mismatch at batch %d, index (%d,%d) (real)',[b,i,j]));

          if Abs(expected_y) < absolute_tolerance then
            CheckEquals(expected_y, actual_y, absolute_tolerance, Format('Mismatch at batch %d, index (%d,%d) (imag)',[b,i,j]))
          else
            CheckEqualsRelative(expected_y, actual_y, relative_tolerance, Format('Mismatch at batch %d, index (%d,%d) (imag)',[b,i,j]));
        end;
    end;

  finally
    if d_A <> nil then hipFree(d_A); if d_B <> nil then hipFree(d_B); if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A); if Assigned(h_B) then FreeMem(h_B); if Assigned(h_C) then FreeMem(h_C); if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;


procedure TTestRocBLAS_Level3_Strided_Batched.Test_ZHER2K_Strided_Batched;
var
  h_A, h_B, h_C, h_C_expected: Procdouble_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  i, j, l, b: Integer;
  alpha, conj_alpha, conjA, conjB: rocblas_double_complex;
  beta: Double;
  size_A, size_B, size_C: SizeUInt;
  pA, pB, pC, pC_expected: Procdouble_complex;
  sum_ABH, sum_BAH, term1, term2, final_sum: rocblas_double_complex;
const
  N = 64;
  K = 72;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  batch_count = 3;
  lda = N;
  ldb = N;
  ldc = N;
  stride_A = lda * K;
  stride_B = ldb * K;
  stride_C = ldc * N;
  tolerance: Double = 1e-12;
begin
  alpha.x := 1.25; alpha.y := -0.5;
  beta := 0.75;
  d_A := nil; d_B := nil; d_C := nil;
  h_A := nil; h_B := nil; h_C := nil; h_C_expected := nil;

  size_A := stride_A * batch_count * SizeOf(rocblas_double_complex);
  size_B := stride_B * batch_count * SizeOf(rocblas_double_complex);
  size_C := stride_C * batch_count * SizeOf(rocblas_double_complex);

  GetMem(h_A, size_A); GetMem(h_B, size_B); GetMem(h_C, size_C); GetMem(h_C_expected, size_C);

  try
    for i := 0 to (stride_A * batch_count) - 1 do begin h_A[i].x := (i mod 10)/10.0; h_A[i].y := (i mod 20)/20.0; end;
    for i := 0 to (stride_B * batch_count) - 1 do begin h_B[i].x := (i mod 30)/30.0; h_B[i].y := (i mod 40)/40.0; end;
    for i := 0 to (stride_C * batch_count) - 1 do begin h_C[i].x := (i mod 50)/50.0; h_C[i].y := (i mod 60)/60.0; end;
    Move(h_C^, h_C_expected^, size_C);

    conj_alpha := alpha;
    conj_alpha.y := -alpha.y;

    for b := 0 to batch_count - 1 do
    begin
      pA := Procdouble_complex(PByte(h_A) + b * stride_A * SizeOf(rocblas_double_complex));
      pB := Procdouble_complex(PByte(h_B) + b * stride_B * SizeOf(rocblas_double_complex));
      pC_expected := Procdouble_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_double_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          sum_ABH.x := 0.0; sum_ABH.y := 0.0;
          sum_BAH.x := 0.0; sum_BAH.y := 0.0;
          for l := 0 to K - 1 do
          begin
            term1 := pA[i + l * lda];
            conjB := pB[j + l * ldb]; conjB.y := -conjB.y;
            ComplexMul(term1, conjB);
            ComplexAdd(sum_ABH, term1);

            term2 := pB[i + l * ldb];
            conjA := pA[j + l * lda]; conjA.y := -conjA.y;
            ComplexMul(term2, conjA);
            ComplexAdd(sum_BAH, term2);
          end;
          ComplexMul(sum_ABH, alpha);
          ComplexMul(sum_BAH, conj_alpha);
          final_sum := sum_ABH;
          ComplexAdd(final_sum, sum_BAH);
          final_sum.x := final_sum.x + beta * pC_expected[i + j * ldc].x;
          final_sum.y := final_sum.y + beta * pC_expected[i + j * ldc].y;
          if i = j then final_sum.y := 0.0;
          pC_expected[i + j * ldc] := final_sum;
        end;
    end;

    hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc C', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy A', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy B', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy C', Integer(HIP_SUCCESS), Integer(hip_status));

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_zher2k_strided_batched(handle, uplo, transA, N, K, @alpha,
      d_A, lda, stride_A, d_B, ldb, stride_B, @beta, d_C, ldc, stride_C, batch_count);
    AssertEquals('rocblas_zher2k_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    hip_status := hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost); AssertEquals('hipMemcpy D->H C', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipDeviceSynchronize(); AssertEquals('hipDeviceSynchronize failed', Integer(HIP_SUCCESS), Integer(hip_status));

    for b := 0 to batch_count - 1 do
    begin
      pC := Procdouble_complex(PByte(h_C) + b * stride_C * SizeOf(rocblas_double_complex));
      pC_expected := Procdouble_complex(PByte(h_C_expected) + b * stride_C * SizeOf(rocblas_double_complex));
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          CheckEquals(pC_expected[i+j*ldc].x, pC[i+j*ldc].x, tolerance, Format('Mismatch at batch %d, index (%d,%d) (real)',[b,i,j]));
          CheckEquals(pC_expected[i+j*ldc].y, pC[i+j*ldc].y, tolerance, Format('Mismatch at batch %d, index (%d,%d) (imag)',[b,i,j]));
        end;
    end;

  finally
    if d_A <> nil then hipFree(d_A); if d_B <> nil then hipFree(d_B); if d_C <> nil then hipFree(d_C);
    if Assigned(h_A) then FreeMem(h_A); if Assigned(h_B) then FreeMem(h_B); if Assigned(h_C) then FreeMem(h_C); if Assigned(h_C_expected) then FreeMem(h_C_expected);
  end;
end;

initialization
  RegisterTest(TTestRocBLAS_Level3_Strided_Batched);

end.

