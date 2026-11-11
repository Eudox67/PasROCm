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

unit Test_rocblas_lvl3_64;

{$mode objfpc}{$H+}
{$linklib rocblas}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  TTestRocBLAS_Level3_64 = class(TTestRocBLAS_Base)
  published
    procedure Test_rocblas_sgemm_64;
    procedure Test_rocblas_dgemm_64;
    procedure Test_rocblas_cgemm_64;
    procedure Test_rocblas_zgemm_64;
    procedure Test_rocblas_chemm_64;
    procedure Test_rocblas_zhemm_64;
    procedure Test_rocblas_cherk_64;
    procedure Test_rocblas_zherk_64;
    procedure Test_rocblas_cher2k_64;
    procedure Test_rocblas_zher2k_64;
    procedure Test_rocblas_ssymm_64;
    procedure Test_rocblas_dsymm_64;
    procedure Test_rocblas_csymm_64;
    procedure Test_rocblas_zsymm_64;
    procedure Test_rocblas_ssyrk_64;
    procedure Test_rocblas_dsyrk_64;
    procedure Test_rocblas_csyrk_64;
    procedure Test_rocblas_zsyrk_64;
    procedure Test_rocblas_ssyr2k_64;
    procedure Test_rocblas_dsyr2k_64;
    procedure Test_rocblas_csyr2k_64;
    procedure Test_rocblas_zsyr2k_64;
    procedure Test_rocblas_strmm_64;
    procedure Test_rocblas_dtrmm_64;
    procedure Test_rocblas_ctrmm_64;
    procedure Test_rocblas_ztrmm_64;
    procedure Test_rocblas_strsm_64;
    procedure Test_rocblas_dtrsm_64;
    procedure Test_rocblas_ctrsm_64;
    procedure Test_rocblas_ztrsm_64;
  end;

implementation

procedure TTestRocBLAS_Level3_64.Test_rocblas_sgemm_64;
const
  m: Int64 = 4;
  n: Int64 = 3;
  k: Int64 = 5;
  lda: Int64 = 4; // lda >= m
  ldb: Int64 = 5; // ldb >= k
  ldc: Int64 = 4; // ldc >= m
  transA = rocblas_operation_none;
  transB = rocblas_operation_none;
var
  h_A, h_B, h_C, h_C_expected: array of Single;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: Single;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum: Single;
begin
  alpha := 2.0;
  beta := 1.5;

  size_A := lda * k * SizeOf(Single);
  size_B := ldb * n * SizeOf(Single);
  size_C := ldc * n * SizeOf(Single);

  SetLength(h_A, lda * k);
  SetLength(h_B, ldb * n);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);

  for i := 0 to (lda * k) - 1 do h_A[i] := i + 1;
  for i := 0 to (ldb * n) - 1 do h_B[i] := i + 1;
  for i := 0 to (ldc * n) - 1 do h_C[i] := i + 1;

  h_C_expected := Copy(h_C);

  // Reference CPU implementation (Column-Major)
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum := 0;
      for l := 0 to k - 1 do
        sum := sum + h_A[l * lda + i] * h_B[j * ldb + l];
      h_C_expected[j * ldc + i] := alpha * sum + beta * h_C[j * ldc + i];
    end;

  hip_status := hipMalloc(d_A, size_A);
  AssertEquals('hipMalloc d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B);
  AssertEquals('hipMalloc d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C);
  AssertEquals('hipMalloc d_C failed', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A -> d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_B -> d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_C -> d_C failed', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_sgemm_64(handle, transA, transB, m, n, k, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_sgemm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldc * n) - 1 do
    CheckEquals(h_C_expected[i], h_C[i], 1e-4, 'Mismatch at index ' + IntToStr(i));

  hip_status := hipFree(d_A);
  AssertEquals('hipFree d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B);
  AssertEquals('hipFree d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C);
  AssertEquals('hipFree d_C failed', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_dgemm_64;
const
  m: Int64 = 4;
  n: Int64 = 3;
  k: Int64 = 5;
  lda: Int64 = 4;
  ldb: Int64 = 5;
  ldc: Int64 = 4;
  transA = rocblas_operation_none;
  transB = rocblas_operation_none;
var
  h_A, h_B, h_C, h_C_expected: array of Double;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: Double;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum: Double;
begin
  alpha := 2.0;
  beta := 1.5;

  size_A := lda * k * SizeOf(Double);
  size_B := ldb * n * SizeOf(Double);
  size_C := ldc * n * SizeOf(Double);

  SetLength(h_A, lda * k);
  SetLength(h_B, ldb * n);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);

  for i := 0 to (lda * k) - 1 do h_A[i] := i + 1;
  for i := 0 to (ldb * n) - 1 do h_B[i] := i + 1;
  for i := 0 to (ldc * n) - 1 do h_C[i] := i + 1;

  h_C_expected := Copy(h_C);

  // Reference CPU implementation (Column-Major)
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum := 0;
      for l := 0 to k - 1 do
        sum := sum + h_A[l * lda + i] * h_B[j * ldb + l];
      h_C_expected[j * ldc + i] := alpha * sum + beta * h_C[j * ldc + i];
    end;

  hip_status := hipMalloc(d_A, size_A);
  AssertEquals('hipMalloc d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B);
  AssertEquals('hipMalloc d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C);
  AssertEquals('hipMalloc d_C failed', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A -> d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_B -> d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_C -> d_C failed', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_dgemm_64(handle, transA, transB, m, n, k, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_dgemm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldc * n) - 1 do
    CheckEquals(h_C_expected[i], h_C[i], 1e-12, 'Mismatch at index ' + IntToStr(i));

  hip_status := hipFree(d_A);
  AssertEquals('hipFree d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B);
  AssertEquals('hipFree d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C);
  AssertEquals('hipFree d_C failed', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_cgemm_64;
const
  m: Int64 = 4;
  n: Int64 = 3;
  k: Int64 = 5;
  lda: Int64 = 4;
  ldb: Int64 = 5;
  ldc: Int64 = 4;
  transA = rocblas_operation_none;
  transB = rocblas_operation_none;
var
  h_A, h_B, h_C, h_C_expected: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_float_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum, term1, term2: rocblas_float_complex;
begin
  alpha.x := 2.0; alpha.y := 1.0;
  beta.x  := 1.5; beta.y  := -0.5;

  size_A := lda * k * SizeOf(rocblas_float_complex);
  size_B := ldb * n * SizeOf(rocblas_float_complex);
  size_C := ldc * n * SizeOf(rocblas_float_complex);

  SetLength(h_A, lda * k);
  SetLength(h_B, ldb * n);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);

  for i := 0 to (lda * k) - 1 do begin h_A[i].x := i+1; h_A[i].y := i-1; end;
  for i := 0 to (ldb * n) - 1 do begin h_B[i].x := i+1; h_B[i].y := i+1; end;
  for i := 0 to (ldc * n) - 1 do begin h_C[i].x := i-2; h_C[i].y := i+2; end;

  h_C_expected := Copy(h_C);

  // Reference CPU implementation (Column-Major)
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to k - 1 do
      begin
        term1 := h_A[l * lda + i];
        term2 := h_B[j * ldb + l];
        ComplexMul(term1, term2); // term1 = A[i,l]*B[l,j]
        ComplexAdd(sum, term1);
      end;
      term1 := h_C[j * ldc + i];
      ComplexMul(term1, beta);   // term1 = beta * C[i,j]
      ComplexMul(sum, alpha);   // sum = alpha * sum
      ComplexAdd(sum, term1);   // sum = alpha*sum + beta*C
      h_C_expected[j * ldc + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A);
  AssertEquals('hipMalloc d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B);
  AssertEquals('hipMalloc d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C);
  AssertEquals('hipMalloc d_C failed', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A -> d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_B -> d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_C -> d_C failed', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_cgemm_64(handle, transA, transB, m, n, k, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_cgemm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldc * n) - 1 do
  begin
    CheckEquals(h_C_expected[i].x, h_C[i].x, 1e-4, 'Real part mismatch at index ' + IntToStr(i));
    CheckEquals(h_C_expected[i].y, h_C[i].y, 1e-4, 'Imag part mismatch at index ' + IntToStr(i));
  end;

  hip_status := hipFree(d_A);
  AssertEquals('hipFree d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B);
  AssertEquals('hipFree d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C);
  AssertEquals('hipFree d_C failed', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_zgemm_64;
const
  m: Int64 = 4;
  n: Int64 = 3;
  k: Int64 = 5;
  lda: Int64 = 4;
  ldb: Int64 = 5;
  ldc: Int64 = 4;
  transA = rocblas_operation_none;
  transB = rocblas_operation_none;
var
  h_A, h_B, h_C, h_C_expected: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_double_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum, term1, term2: rocblas_double_complex;
begin
  alpha.x := 2.0; alpha.y := 1.0;
  beta.x  := 1.5; beta.y  := -0.5;

  size_A := lda * k * SizeOf(rocblas_double_complex);
  size_B := ldb * n * SizeOf(rocblas_double_complex);
  size_C := ldc * n * SizeOf(rocblas_double_complex);

  SetLength(h_A, lda * k);
  SetLength(h_B, ldb * n);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);

  for i := 0 to (lda * k) - 1 do begin h_A[i].x := i+1; h_A[i].y := i-1; end;
  for i := 0 to (ldb * n) - 1 do begin h_B[i].x := i+1; h_B[i].y := i+1; end;
  for i := 0 to (ldc * n) - 1 do begin h_C[i].x := i-2; h_C[i].y := i+2; end;

  h_C_expected := Copy(h_C);

  // Reference CPU implementation (Column-Major)
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to k - 1 do
      begin
        term1 := h_A[l * lda + i];
        term2 := h_B[j * ldb + l];
        ComplexMul(term1, term2); // term1 = A[i,l]*B[l,j]
        ComplexAdd(sum, term1);
      end;
      term1 := h_C[j * ldc + i];
      ComplexMul(term1, beta);   // term1 = beta * C[i,j]
      ComplexMul(sum, alpha);   // sum = alpha * sum
      ComplexAdd(sum, term1);   // sum = alpha*sum + beta*C
      h_C_expected[j * ldc + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A);
  AssertEquals('hipMalloc d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B);
  AssertEquals('hipMalloc d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C);
  AssertEquals('hipMalloc d_C failed', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A -> d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_B -> d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_C -> d_C failed', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_zgemm_64(handle, transA, transB, m, n, k, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_zgemm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldc * n) - 1 do
  begin
    CheckEquals(h_C_expected[i].x, h_C[i].x, 1e-12, 'Real part mismatch at index ' + IntToStr(i));
    CheckEquals(h_C_expected[i].y, h_C[i].y, 1e-12, 'Imag part mismatch at index ' + IntToStr(i));
  end;

  hip_status := hipFree(d_A);
  AssertEquals('hipFree d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B);
  AssertEquals('hipFree d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C);
  AssertEquals('hipFree d_C failed', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_chemm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4; // if side_left, lda >= m
  ldb: Int64 = 4; // ldb >= m
  ldc: Int64 = 4; // ldc >= m
var
  h_A, h_B, h_C, h_C_expected, h_A_full: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_float_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum, term1, term2: rocblas_float_complex;
begin
  alpha.x := 1.2; alpha.y := -0.5;
  beta.x  := 0.8; beta.y  := 0.3;

  size_A := lda * m * SizeOf(rocblas_float_complex);
  size_B := ldb * n * SizeOf(rocblas_float_complex);
  size_C := ldc * n * SizeOf(rocblas_float_complex);

  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B, ldb * n);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);

  // Initialize a Hermitian matrix A (upper part, column-major)
  for j := 0 to m - 1 do
    for i := 0 to j do
    begin
      h_A[j * lda + i].x := i + j * 0.1;
      if i = j then
        h_A[j * lda + i].y := 0.0 // Diagonal elements must be real
      else
        h_A[j * lda + i].y := j * 0.2 + i;
    end;

  // Create full matrix for CPU calculation (column-major)
  for j := 0 to m - 1 do
    for i := 0 to m - 1 do
      if i <= j then
        h_A_full[j * lda + i] := h_A[j * lda + i]
      else
      begin // Lower part is conjugate transpose of upper part: A(i,j) = conj(A(j,i))
        h_A_full[j * lda + i].x :=  h_A[i * lda + j].x;
        h_A_full[j * lda + i].y := -h_A[i * lda + j].y;
      end;

  for i := 0 to (ldb * n) - 1 do begin h_B[i].x := i+1; h_B[i].y := i-1; end;
  for i := 0 to (ldc * n) - 1 do begin h_C[i].x := i-2; h_C[i].y := i+2; end;
  h_C_expected := Copy(h_C);

  // Reference CPU implementation: C = alpha * A * B + beta * C (Column-Major)
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to m - 1 do // For side_left, inner loop is over m
      begin
        term1 := h_A_full[l * lda + i]; // A(i,l)
        term2 := h_B[j * ldb + l];      // B(l,j)
        ComplexMul(term1, term2);
        ComplexAdd(sum, term1);
      end;
      term1 := h_C[j * ldc + i]; // C(i,j)
      ComplexMul(term1, beta);
      ComplexMul(sum, alpha);
      ComplexAdd(sum, term1);
      h_C_expected[j * ldc + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A);
  AssertEquals('hipMalloc d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B);
  AssertEquals('hipMalloc d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C);
  AssertEquals('hipMalloc d_C failed', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A -> d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_B -> d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_C -> d_C failed', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_chemm_64(handle, side, uplo, m, n, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_chemm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldc * n) - 1 do
  begin
    CheckEquals(h_C_expected[i].x, h_C[i].x, 1e-4, 'Real part mismatch at index ' + IntToStr(i));
    CheckEquals(h_C_expected[i].y, h_C[i].y, 1e-4, 'Imag part mismatch at index ' + IntToStr(i));
  end;

  hip_status := hipFree(d_A);
  AssertEquals('hipFree d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B);
  AssertEquals('hipFree d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C);
  AssertEquals('hipFree d_C failed', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_zhemm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4; // if side_left, lda >= m
  ldb: Int64 = 4; // ldb >= m
  ldc: Int64 = 4; // ldc >= m
var
  h_A, h_B, h_C, h_C_expected, h_A_full: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_double_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum, term1, term2: rocblas_double_complex;
begin
  alpha.x := 1.2; alpha.y := -0.5;
  beta.x  := 0.8; beta.y  := 0.3;

  size_A := lda * m * SizeOf(rocblas_double_complex);
  size_B := ldb * n * SizeOf(rocblas_double_complex);
  size_C := ldc * n * SizeOf(rocblas_double_complex);

  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B, ldb * n);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);

  // Initialize a Hermitian matrix A (upper part, column-major)
  for j := 0 to m - 1 do
    for i := 0 to j do
    begin
      h_A[j * lda + i].x := i + j * 0.1;
      if i = j then
        h_A[j * lda + i].y := 0.0 // Diagonal elements must be real
      else
        h_A[j * lda + i].y := j * 0.2 + i;
    end;

  // Create full matrix for CPU calculation (column-major)
  for j := 0 to m - 1 do
    for i := 0 to m - 1 do
      if i <= j then
        h_A_full[j * lda + i] := h_A[j * lda + i]
      else
      begin // Lower part is conjugate transpose of upper part: A(i,j) = conj(A(j,i))
        h_A_full[j * lda + i].x :=  h_A[i * lda + j].x;
        h_A_full[j * lda + i].y := -h_A[i * lda + j].y;
      end;

  for i := 0 to (ldb * n) - 1 do begin h_B[i].x := i+1; h_B[i].y := i-1; end;
  for i := 0 to (ldc * n) - 1 do begin h_C[i].x := i-2; h_C[i].y := i+2; end;
  h_C_expected := Copy(h_C);

  // Reference CPU implementation: C = alpha * A * B + beta * C (Column-Major)
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to m - 1 do // For side_left, inner loop is over m
      begin
        term1 := h_A_full[l * lda + i]; // A(i,l)
        term2 := h_B[j * ldb + l];      // B(l,j)
        ComplexMul(term1, term2);
        ComplexAdd(sum, term1);
      end;
      term1 := h_C[j * ldc + i]; // C(i,j)
      ComplexMul(term1, beta);
      ComplexMul(sum, alpha);
      ComplexAdd(sum, term1);
      h_C_expected[j * ldc + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A);
  AssertEquals('hipMalloc d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B);
  AssertEquals('hipMalloc d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C);
  AssertEquals('hipMalloc d_C failed', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A -> d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_B -> d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_C -> d_C failed', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_zhemm_64(handle, side, uplo, m, n, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_zhemm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldc * n) - 1 do
  begin
    CheckEquals(h_C_expected[i].x, h_C[i].x, 1e-12, 'Real part mismatch at index ' + IntToStr(i));
    CheckEquals(h_C_expected[i].y, h_C[i].y, 1e-12, 'Imag part mismatch at index ' + IntToStr(i));
  end;

  hip_status := hipFree(d_A);
  AssertEquals('hipFree d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B);
  AssertEquals('hipFree d_B failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C);
  AssertEquals('hipFree d_C failed', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_cherk_64;
const
  n: Int64 = 4;
  k: Int64 = 5;
  transA = rocblas_operation_none;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4; // if transA=none, lda >= n
  ldc: Int64 = 4; // ldc >= n
var
  h_A, h_C, h_C_expected, h_C_initial_full: array of rocblas_float_complex;
  d_A, d_C: hipDeviceptr_t;
  alpha, beta: Single;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_C: SizeUInt;
  i, j, l: Integer;
  sum, term1, term2: rocblas_float_complex;
begin
  alpha := 1.5;
  beta := 1.2;

  size_A := lda * k * SizeOf(rocblas_float_complex);
  size_C := ldc * n * SizeOf(rocblas_float_complex);

  SetLength(h_A, lda * k);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);
  SetLength(h_C_initial_full, ldc * n);

  // Initialize matrices with column-major layout
  for i := 0 to (lda * k) - 1 do begin h_A[i].x := (i+1)*0.1; h_A[i].y := (i-1)*0.1; end;

  // Initialize upper triangle of Hermitian matrix C
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_C[j * ldc + i].x := (i+j+1) * 0.2;
      if i = j then
        h_C[j * ldc + i].y := 0.0 // Diagonal must be real
      else
        h_C[j * ldc + i].y := (i-j) * 0.2;
    end;

  // Create a full initial C matrix for the reference calculation
  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
      if i <= j then
        h_C_initial_full[j * ldc + i] := h_C[j * ldc + i]
      else // Lower part is conjugate transpose of upper part
      begin
        h_C_initial_full[j * ldc + i].x :=  h_C[i * ldc + j].x;
        h_C_initial_full[j * ldc + i].y := -h_C[i * ldc + j].y;
      end;

  // Reference CPU calculation: C := alpha * A * A^H + beta * C
  for j := 0 to n - 1 do
    for i := 0 to j do // Only calculate the upper triangle
    begin
      // sum = (A * A^H)_ij = SUM_l( A_il * conj(A_jl) )
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to k - 1 do
      begin
        term1 := h_A[l * lda + i];         // A(i,l)
        term2.x := h_A[l * lda + j].x;      // conj(A(j,l))
        term2.y := -h_A[l * lda + j].y;
        ComplexMul(term1, term2);
        ComplexAdd(sum, term1);
      end;

      // C_expected = alpha * sum + beta * C_initial
      term1 := h_C_initial_full[j * ldc + i];
      term1.x := beta * term1.x;
      term1.y := beta * term1.y;
      sum.x   := alpha * sum.x;
      sum.y   := alpha * sum.y;
      ComplexAdd(sum, term1);
      h_C_expected[j * ldc + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A);
  AssertEquals('hipMalloc d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C);
  AssertEquals('hipMalloc d_C failed', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A -> d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_C -> d_C failed', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_cherk_64(handle, uplo, transA, n, k, @alpha, d_A, lda, @beta, d_C, ldc);
  AssertEquals('rocblas_cherk_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEquals(h_C_expected[j*ldc+i].x, h_C[j*ldc+i].x, 1e-4, 'Real part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
      CheckEquals(h_C_expected[j*ldc+i].y, h_C[j*ldc+i].y, 1e-4, 'Imag part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
    end;

  hip_status := hipFree(d_A);
  AssertEquals('hipFree d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C);
  AssertEquals('hipFree d_C failed', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_zherk_64;
const
  n: Int64 = 4;
  k: Int64 = 5;
  transA = rocblas_operation_none;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4; // if transA=none, lda >= n
  ldc: Int64 = 4; // ldc >= n
var
  h_A, h_C, h_C_expected, h_C_initial_full: array of rocblas_double_complex;
  d_A, d_C: hipDeviceptr_t;
  alpha, beta: Double;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_C: SizeUInt;
  i, j, l: Integer;
  sum, term1, term2: rocblas_double_complex;
begin
  alpha := 1.5;
  beta := 1.2;

  size_A := lda * k * SizeOf(rocblas_double_complex);
  size_C := ldc * n * SizeOf(rocblas_double_complex);

  SetLength(h_A, lda * k);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);
  SetLength(h_C_initial_full, ldc * n);

  // Initialize matrices with column-major layout
  for i := 0 to (lda * k) - 1 do begin h_A[i].x := (i+1)*0.1; h_A[i].y := (i-1)*0.1; end;

  // Initialize upper triangle of Hermitian matrix C
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_C[j * ldc + i].x := (i+j+1) * 0.2;
      if i = j then
        h_C[j * ldc + i].y := 0.0 // Diagonal must be real
      else
        h_C[j * ldc + i].y := (i-j) * 0.2;
    end;

  // Create a full initial C matrix for the reference calculation
  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
      if i <= j then
        h_C_initial_full[j * ldc + i] := h_C[j * ldc + i]
      else // Lower part is conjugate transpose of upper part
      begin
        h_C_initial_full[j * ldc + i].x :=  h_C[i * ldc + j].x;
        h_C_initial_full[j * ldc + i].y := -h_C[i * ldc + j].y;
      end;

  // Reference CPU calculation: C := alpha * A * A^H + beta * C
  for j := 0 to n - 1 do
    for i := 0 to j do // Only calculate the upper triangle
    begin
      // sum = (A * A^H)_ij = SUM_l( A_il * conj(A_jl) )
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to k - 1 do
      begin
        term1 := h_A[l * lda + i];         // A(i,l)
        term2.x := h_A[l * lda + j].x;      // conj(A(j,l))
        term2.y := -h_A[l * lda + j].y;
        ComplexMul(term1, term2);
        ComplexAdd(sum, term1);
      end;

      // C_expected = alpha * sum + beta * C_initial
      term1 := h_C_initial_full[j * ldc + i];
      term1.x := beta * term1.x;
      term1.y := beta * term1.y;
      sum.x   := alpha * sum.x;
      sum.y   := alpha * sum.y;
      ComplexAdd(sum, term1);
      h_C_expected[j * ldc + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A);
  AssertEquals('hipMalloc d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C);
  AssertEquals('hipMalloc d_C failed', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A -> d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_C -> d_C failed', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_zherk_64(handle, uplo, transA, n, k, @alpha, d_A, lda, @beta, d_C, ldc);
  AssertEquals('rocblas_zherk_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEquals(h_C_expected[j*ldc+i].x, h_C[j*ldc+i].x, 1e-12, 'Real part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
      CheckEquals(h_C_expected[j*ldc+i].y, h_C[j*ldc+i].y, 1e-12, 'Imag part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
    end;

  hip_status := hipFree(d_A);
  AssertEquals('hipFree d_A failed', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C);
  AssertEquals('hipFree d_C failed', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_cher2k_64;
const
  n: Int64 = 4;
  k: Int64 = 5;
  trans = rocblas_operation_none;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4; // if trans=none, lda >= n
  ldb: Int64 = 4; // if trans=none, ldb >= n
  ldc: Int64 = 4; // ldc >= n
var
  h_A, h_B, h_C, h_C_expected, h_C_initial_full: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, conj_alpha: rocblas_float_complex;
  beta: Single;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  term1_sum, term2_sum, C_beta_term, final_sum: rocblas_float_complex;
  prod, conj_A_jl, conj_B_jl : rocblas_float_complex;
begin
  alpha.x := 1.2; alpha.y := 0.5;
  beta := 0.8;

  conj_alpha.x := alpha.x;
  conj_alpha.y := -alpha.y;

  size_A := lda * k * SizeOf(rocblas_float_complex);
  size_B := ldb * k * SizeOf(rocblas_float_complex);
  size_C := ldc * n * SizeOf(rocblas_float_complex);

  SetLength(h_A, lda * k);
  SetLength(h_B, ldb * k);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);
  SetLength(h_C_initial_full, ldc * n);

  // Initialize matrices with column-major layout
  for i := 0 to (lda * k) - 1 do begin h_A[i].x := (i+1)*0.1; h_A[i].y := (i-1)*0.1; end;
  for i := 0 to (ldb * k) - 1 do begin h_B[i].x := (i+1)*0.2; h_B[i].y := (i-2)*0.2; end;

  // Initialize upper triangle of Hermitian matrix C
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_C[j * ldc + i].x := (i+j+1) * 0.3;
      if i = j then h_C[j * ldc + i].y := 0.0
      else h_C[j * ldc + i].y := (i-j) * 0.3;
    end;

  // Create a full initial C matrix for the reference calculation
  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
      if i <= j then h_C_initial_full[j * ldc + i] := h_C[j * ldc + i]
      else begin // Lower part is conjugate transpose of upper part
        h_C_initial_full[j * ldc + i].x :=  h_C[i * ldc + j].x;
        h_C_initial_full[j * ldc + i].y := -h_C[i * ldc + j].y;
      end;

  // Reference CPU calculation
  for j := 0 to n - 1 do
    for i := 0 to j do // Only calculate the upper triangle
    begin
      // Term 1: (A * B^H)_ij = SUM_l( A_il * conj(B_jl) )
      term1_sum.x := 0.0; term1_sum.y := 0.0;
      for l := 0 to k - 1 do
      begin
        prod := h_A[l * lda + i];       // A(i,l)
        conj_B_jl.x :=  h_B[l * ldb + j].x; // B(j,l)
        conj_B_jl.y := -h_B[l * ldb + j].y; // conj(B(j,l))
        ComplexMul(prod, conj_B_jl);
        ComplexAdd(term1_sum, prod);
      end;

      // Term 2: (B * A^H)_ij = SUM_l( B_il * conj(A_jl) )
      term2_sum.x := 0.0; term2_sum.y := 0.0;
      for l := 0 to k - 1 do
      begin
        prod := h_B[l * ldb + i];       // B(i,l)
        conj_A_jl.x :=  h_A[l * lda + j].x; // A(j,l)
        conj_A_jl.y := -h_A[l * lda + j].y; // conj(A(j,l))
        ComplexMul(prod, conj_A_jl);
        ComplexAdd(term2_sum, prod);
      end;

      // final_sum = alpha * term1_sum + conj(alpha) * term2_sum + beta * C_initial
      ComplexMul(term1_sum, alpha);
      ComplexMul(term2_sum, conj_alpha);

      C_beta_term := h_C_initial_full[j * ldc + i];
      C_beta_term.x := C_beta_term.x * beta;
      C_beta_term.y := C_beta_term.y * beta;

      final_sum := term1_sum;
      ComplexAdd(final_sum, term2_sum);
      ComplexAdd(final_sum, C_beta_term);
      h_C_expected[j * ldc + i] := final_sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_cher2k_64(handle, uplo, trans, n, k, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_cher2k_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEquals(h_C_expected[j*ldc+i].x, h_C[j*ldc+i].x, 1e-4, 'Real part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
      CheckEquals(h_C_expected[j*ldc+i].y, h_C[j*ldc+i].y, 1e-4, 'Imag part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
    end;

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_zher2k_64;
const
  n: Int64 = 4;
  k: Int64 = 5;
  trans = rocblas_operation_none;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4; // if trans=none, lda >= n
  ldb: Int64 = 4; // if trans=none, ldb >= n
  ldc: Int64 = 4; // ldc >= n
var
  h_A, h_B, h_C, h_C_expected, h_C_initial_full: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, conj_alpha: rocblas_double_complex;
  beta: Double;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  term1_sum, term2_sum, C_beta_term, final_sum: rocblas_double_complex;
  prod, conj_A_jl, conj_B_jl : rocblas_double_complex;
begin
  alpha.x := 1.2; alpha.y := 0.5;
  beta := 0.8;

  conj_alpha.x := alpha.x;
  conj_alpha.y := -alpha.y;

  size_A := lda * k * SizeOf(rocblas_double_complex);
  size_B := ldb * k * SizeOf(rocblas_double_complex);
  size_C := ldc * n * SizeOf(rocblas_double_complex);

  SetLength(h_A, lda * k);
  SetLength(h_B, ldb * k);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);
  SetLength(h_C_initial_full, ldc * n);

  // Initialize matrices with column-major layout
  for i := 0 to (lda * k) - 1 do begin h_A[i].x := (i+1)*0.1; h_A[i].y := (i-1)*0.1; end;
  for i := 0 to (ldb * k) - 1 do begin h_B[i].x := (i+1)*0.2; h_B[i].y := (i-2)*0.2; end;

  // Initialize upper triangle of Hermitian matrix C
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_C[j * ldc + i].x := (i+j+1) * 0.3;
      if i = j then h_C[j * ldc + i].y := 0.0
      else h_C[j * ldc + i].y := (i-j) * 0.3;
    end;

  // Create a full initial C matrix for the reference calculation
  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
      if i <= j then h_C_initial_full[j * ldc + i] := h_C[j * ldc + i]
      else begin
        h_C_initial_full[j * ldc + i].x :=  h_C[i * ldc + j].x;
        h_C_initial_full[j * ldc + i].y := -h_C[i * ldc + j].y;
      end;

  // Reference CPU calculation
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      // Term 1: (A * B^H)_ij = SUM_l( A_il * conj(B_jl) )
      term1_sum.x := 0.0; term1_sum.y := 0.0;
      for l := 0 to k - 1 do
      begin
        prod := h_A[l * lda + i];
        conj_B_jl.x :=  h_B[l * ldb + j].x;
        conj_B_jl.y := -h_B[l * ldb + j].y;
        ComplexMul(prod, conj_B_jl);
        ComplexAdd(term1_sum, prod);
      end;

      // Term 2: (B * A^H)_ij = SUM_l( B_il * conj(A_jl) )
      term2_sum.x := 0.0; term2_sum.y := 0.0;
      for l := 0 to k - 1 do
      begin
        prod := h_B[l * ldb + i];
        conj_A_jl.x :=  h_A[l * lda + j].x;
        conj_A_jl.y := -h_A[l * lda + j].y;
        ComplexMul(prod, conj_A_jl);
        ComplexAdd(term2_sum, prod);
      end;

      // final_sum = alpha * term1_sum + conj(alpha) * term2_sum + beta * C_initial
      ComplexMul(term1_sum, alpha);
      ComplexMul(term2_sum, conj_alpha);

      C_beta_term := h_C_initial_full[j * ldc + i];
      C_beta_term.x := C_beta_term.x * beta;
      C_beta_term.y := C_beta_term.y * beta;

      final_sum := term1_sum;
      ComplexAdd(final_sum, term2_sum);
      ComplexAdd(final_sum, C_beta_term);
      h_C_expected[j * ldc + i] := final_sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_zher2k_64(handle, uplo, trans, n, k, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_zher2k_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEquals(h_C_expected[j*ldc+i].x, h_C[j*ldc+i].x, 1e-12, 'Real part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
      CheckEquals(h_C_expected[j*ldc+i].y, h_C[j*ldc+i].y, 1e-12, 'Imag part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
    end;

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_ssymm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4; // if side_left, lda >= m
  ldb: Int64 = 4; // ldb >= m
  ldc: Int64 = 4; // ldc >= m
var
  h_A, h_B, h_C, h_C_expected, h_A_full: array of Single;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: Single;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum: Single;
begin
  alpha := 1.5;
  beta := 1.2;

  size_A := lda * m * SizeOf(Single);
  size_B := ldb * n * SizeOf(Single);
  size_C := ldc * n * SizeOf(Single);

  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B, ldb * n);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);

  // Initialize upper triangle of symmetric matrix A (column-major)
  for j := 0 to m - 1 do
    for i := 0 to j do
      h_A[j * lda + i] := (i+1) + (j+1) * 0.1;

  // Create full symmetric matrix for CPU calculation
  for j := 0 to m - 1 do
    for i := 0 to m - 1 do
      if i <= j then
        h_A_full[j * lda + i] := h_A[j * lda + i]
      else
        h_A_full[j * lda + i] := h_A[i * lda + j]; // A(i,j) = A(j,i)

  for i := 0 to (ldb * n) - 1 do h_B[i] := (i+1) * 0.5;
  for i := 0 to (ldc * n) - 1 do h_C[i] := (i+1) * 0.8;
  h_C_expected := Copy(h_C);

  // Reference CPU implementation: C = alpha * A * B + beta * C (Column-Major)
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum := 0.0;
      for l := 0 to m - 1 do // For side_left, inner loop over m
        sum := sum + h_A_full[l * lda + i] * h_B[j * ldb + l];
      h_C_expected[j * ldc + i] := alpha * sum + beta * h_C[j * ldc + i];
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_ssymm_64(handle, side, uplo, m, n, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_ssymm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldc * n) - 1 do
    CheckEquals(h_C_expected[i], h_C[i], 1e-4, 'Mismatch at index ' + IntToStr(i));

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_dsymm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4;
  ldb: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_B, h_C, h_C_expected, h_A_full: array of Double;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: Double;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum: Double;
begin
  alpha := 1.5;
  beta := 1.2;

  size_A := lda * m * SizeOf(Double);
  size_B := ldb * n * SizeOf(Double);
  size_C := ldc * n * SizeOf(Double);

  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B, ldb * n);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);

  // Initialize upper triangle of symmetric matrix A (column-major)
  for j := 0 to m - 1 do
    for i := 0 to j do
      h_A[j * lda + i] := (i+1) + (j+1) * 0.1;

  // Create full symmetric matrix for CPU calculation
  for j := 0 to m - 1 do
    for i := 0 to m - 1 do
      if i <= j then
        h_A_full[j * lda + i] := h_A[j * lda + i]
      else
        h_A_full[j * lda + i] := h_A[i * lda + j];

  for i := 0 to (ldb * n) - 1 do h_B[i] := (i+1) * 0.5;
  for i := 0 to (ldc * n) - 1 do h_C[i] := (i+1) * 0.8;
  h_C_expected := Copy(h_C);

  // Reference CPU implementation: C = alpha * A * B + beta * C (Column-Major)
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum := 0.0;
      for l := 0 to m - 1 do
        sum := sum + h_A_full[l * lda + i] * h_B[j * ldb + l];
      h_C_expected[j * ldc + i] := alpha * sum + beta * h_C[j * ldc + i];
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_dsymm_64(handle, side, uplo, m, n, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_dsymm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldc * n) - 1 do
    CheckEquals(h_C_expected[i], h_C[i], 1e-12, 'Mismatch at index ' + IntToStr(i));

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_csymm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4;
  ldb: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_B, h_C, h_C_expected, h_A_full: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_float_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum, prod: rocblas_float_complex;
begin
  alpha.x := 1.2; alpha.y := -0.5;
  beta.x  := 0.8; beta.y  := 0.3;

  size_A := lda * m * SizeOf(rocblas_float_complex);
  size_B := ldb * n * SizeOf(rocblas_float_complex);
  size_C := ldc * n * SizeOf(rocblas_float_complex);

  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B, ldb * n);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);

  // Initialize upper triangle of symmetric matrix A (column-major)
  for j := 0 to m - 1 do
    for i := 0 to j do
    begin
      h_A[j * lda + i].x := (i+1) + (j+1) * 0.1;
      h_A[j * lda + i].y := (i-j) * 0.2;
    end;

  // Create full symmetric matrix for CPU calculation
  for j := 0 to m - 1 do
    for i := 0 to m - 1 do
      if i <= j then
        h_A_full[j * lda + i] := h_A[j * lda + i]
      else
        h_A_full[j * lda + i] := h_A[i * lda + j]; // Symmetric, not Hermitian

  for i := 0 to (ldb * n) - 1 do begin h_B[i].x := i+1; h_B[i].y := i-1; end;
  for i := 0 to (ldc * n) - 1 do begin h_C[i].x := i-2; h_C[i].y := i+2; end;
  h_C_expected := Copy(h_C);

  // Reference CPU implementation: C = alpha * A * B + beta * C
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to m - 1 do
      begin
        prod := h_A_full[l * lda + i];
        ComplexMul(prod, h_B[j * ldb + l]);
        ComplexAdd(sum, prod);
      end;
      prod := h_C[j * ldc + i];
      ComplexMul(prod, beta);
      ComplexMul(sum, alpha);
      ComplexAdd(sum, prod);
      h_C_expected[j * ldc + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_csymm_64(handle, side, uplo, m, n, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_csymm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldc * n) - 1 do
  begin
    CheckEquals(h_C_expected[i].x, h_C[i].x, 1e-4, 'Real part mismatch at ' + IntToStr(i));
    CheckEquals(h_C_expected[i].y, h_C[i].y, 1e-4, 'Imag part mismatch at ' + IntToStr(i));
  end;

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_zsymm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4;
  ldb: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_B, h_C, h_C_expected, h_A_full: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_double_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum, prod: rocblas_double_complex;
begin
  alpha.x := 1.2; alpha.y := -0.5;
  beta.x  := 0.8; beta.y  := 0.3;

  size_A := lda * m * SizeOf(rocblas_double_complex);
  size_B := ldb * n * SizeOf(rocblas_double_complex);
  size_C := ldc * n * SizeOf(rocblas_double_complex);

  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B, ldb * n);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);

  // Initialize upper triangle of symmetric matrix A (column-major)
  for j := 0 to m - 1 do
    for i := 0 to j do
    begin
      h_A[j * lda + i].x := (i+1) + (j+1) * 0.1;
      h_A[j * lda + i].y := (i-j) * 0.2;
    end;

  // Create full symmetric matrix for CPU calculation
  for j := 0 to m - 1 do
    for i := 0 to m - 1 do
      if i <= j then
        h_A_full[j * lda + i] := h_A[j * lda + i]
      else
        h_A_full[j * lda + i] := h_A[i * lda + j];

  for i := 0 to (ldb * n) - 1 do begin h_B[i].x := i+1; h_B[i].y := i-1; end;
  for i := 0 to (ldc * n) - 1 do begin h_C[i].x := i-2; h_C[i].y := i+2; end;
  h_C_expected := Copy(h_C);

  // Reference CPU implementation: C = alpha * A * B + beta * C
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to m - 1 do
      begin
        prod := h_A_full[l * lda + i];
        ComplexMul(prod, h_B[j * ldb + l]);
        ComplexAdd(sum, prod);
      end;
      prod := h_C[j * ldc + i];
      ComplexMul(prod, beta);
      ComplexMul(sum, alpha);
      ComplexAdd(sum, prod);
      h_C_expected[j * ldc + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_zsymm_64(handle, side, uplo, m, n, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_zsymm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldc * n) - 1 do
  begin
    CheckEquals(h_C_expected[i].x, h_C[i].x, 1e-12, 'Real part mismatch at ' + IntToStr(i));
    CheckEquals(h_C_expected[i].y, h_C[i].y, 1e-12, 'Imag part mismatch at ' + IntToStr(i));
  end;

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_ssyrk_64;
const
  n: Int64 = 4;
  k: Int64 = 5;
  transA = rocblas_operation_none;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4; // if transA=none, lda >= n
  ldc: Int64 = 4; // ldc >= n
var
  h_A, h_C, h_C_expected, h_C_initial_full: array of Single;
  d_A, d_C: hipDeviceptr_t;
  alpha, beta: Single;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_C: SizeUInt;
  i, j, l: Integer;
  sum: Single;
begin
  alpha := 1.5;
  beta := 1.2;

  size_A := lda * k * SizeOf(Single);
  size_C := ldc * n * SizeOf(Single);

  SetLength(h_A, lda * k);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);
  SetLength(h_C_initial_full, ldc * n);

  // Initialize matrices with column-major layout
  for i := 0 to (lda * k) - 1 do h_A[i] := (i+1) * 0.1;

  // Initialize upper triangle of symmetric matrix C
  for j := 0 to n - 1 do
    for i := 0 to j do
      h_C[j * ldc + i] := (i+j+1) * 0.2;

  // Create a full initial C matrix for the reference calculation
  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
      if i <= j then
        h_C_initial_full[j * ldc + i] := h_C[j * ldc + i]
      else // Lower part is same as upper part
        h_C_initial_full[j * ldc + i] := h_C[i * ldc + j];

  // Reference CPU calculation: C := alpha * A * A^T + beta * C
  for j := 0 to n - 1 do
    for i := 0 to j do // Only calculate the upper triangle
    begin
      // sum = (A * A^T)_ij = SUM_l( A_il * A_jl )
      sum := 0.0;
      for l := 0 to k - 1 do
        sum := sum + h_A[l * lda + i] * h_A[l * lda + j];

      h_C_expected[j * ldc + i] := alpha * sum + beta * h_C_initial_full[j * ldc + i];
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_ssyrk_64(handle, uplo, transA, n, k, @alpha, d_A, lda, @beta, d_C, ldc);
  AssertEquals('rocblas_ssyrk_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for j := 0 to n - 1 do
    for i := 0 to j do
      CheckEquals(h_C_expected[j*ldc+i], h_C[j*ldc+i], 1e-4, 'Mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_dsyrk_64;
const
  n: Int64 = 4;
  k: Int64 = 5;
  transA = rocblas_operation_none;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_C, h_C_expected, h_C_initial_full: array of Double;
  d_A, d_C: hipDeviceptr_t;
  alpha, beta: Double;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_C: SizeUInt;
  i, j, l: Integer;
  sum: Double;
begin
  alpha := 1.5;
  beta := 1.2;

  size_A := lda * k * SizeOf(Double);
  size_C := ldc * n * SizeOf(Double);

  SetLength(h_A, lda * k);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);
  SetLength(h_C_initial_full, ldc * n);

  for i := 0 to (lda * k) - 1 do h_A[i] := (i+1) * 0.1;

  for j := 0 to n - 1 do
    for i := 0 to j do
      h_C[j * ldc + i] := (i+j+1) * 0.2;

  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
      if i <= j then
        h_C_initial_full[j * ldc + i] := h_C[j * ldc + i]
      else
        h_C_initial_full[j * ldc + i] := h_C[i * ldc + j];

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      sum := 0.0;
      for l := 0 to k - 1 do
        sum := sum + h_A[l * lda + i] * h_A[l * lda + j];
      h_C_expected[j * ldc + i] := alpha * sum + beta * h_C_initial_full[j * ldc + i];
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_dsyrk_64(handle, uplo, transA, n, k, @alpha, d_A, lda, @beta, d_C, ldc);
  AssertEquals('rocblas_dsyrk_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for j := 0 to n - 1 do
    for i := 0 to j do
      CheckEquals(h_C_expected[j*ldc+i], h_C[j*ldc+i], 1e-12, 'Mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_csyrk_64;
const
  n: Int64 = 4;
  k: Int64 = 5;
  transA = rocblas_operation_none;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_C, h_C_expected, h_C_initial_full: array of rocblas_float_complex;
  d_A, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_float_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_C: SizeUInt;
  i, j, l: Integer;
  sum, term1, term2: rocblas_float_complex;
begin
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x := 1.2; beta.y := 0.8;

  size_A := lda * k * SizeOf(rocblas_float_complex);
  size_C := ldc * n * SizeOf(rocblas_float_complex);

  SetLength(h_A, lda * k);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);
  SetLength(h_C_initial_full, ldc * n);

  for i := 0 to (lda * k) - 1 do begin h_A[i].x := (i+1)*0.1; h_A[i].y := (i-1)*0.1; end;

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_C[j * ldc + i].x := (i+j+1) * 0.2;
      h_C[j * ldc + i].y := (i-j) * 0.2;
    end;

  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
      if i <= j then
        h_C_initial_full[j * ldc + i] := h_C[j * ldc + i]
      else
        h_C_initial_full[j * ldc + i] := h_C[i * ldc + j];

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to k - 1 do
      begin
        term1 := h_A[l * lda + i]; // A(i,l)
        term2 := h_A[l * lda + j]; // A(j,l) - NO CONJUGATE for syrk
        ComplexMul(term1, term2);
        ComplexAdd(sum, term1);
      end;
      term1 := h_C_initial_full[j * ldc + i];
      ComplexMul(term1, beta);
      ComplexMul(sum, alpha);
      ComplexAdd(sum, term1);
      h_C_expected[j * ldc + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_csyrk_64(handle, uplo, transA, n, k, @alpha, d_A, lda, @beta, d_C, ldc);
  AssertEquals('rocblas_csyrk_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEquals(h_C_expected[j*ldc+i].x, h_C[j*ldc+i].x, 1e-4, 'Real part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
      CheckEquals(h_C_expected[j*ldc+i].y, h_C[j*ldc+i].y, 1e-4, 'Imag part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
    end;

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_zsyrk_64;
const
  n: Int64 = 4;
  k: Int64 = 5;
  transA = rocblas_operation_none;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_C, h_C_expected, h_C_initial_full: array of rocblas_double_complex;
  d_A, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_double_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_C: SizeUInt;
  i, j, l: Integer;
  sum, term1, term2: rocblas_double_complex;
begin
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x := 1.2; beta.y := 0.8;

  size_A := lda * k * SizeOf(rocblas_double_complex);
  size_C := ldc * n * SizeOf(rocblas_double_complex);

  SetLength(h_A, lda * k);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);
  SetLength(h_C_initial_full, ldc * n);

  for i := 0 to (lda * k) - 1 do begin h_A[i].x := (i+1)*0.1; h_A[i].y := (i-1)*0.1; end;

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_C[j * ldc + i].x := (i+j+1) * 0.2;
      h_C[j * ldc + i].y := (i-j) * 0.2;
    end;

  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
      if i <= j then
        h_C_initial_full[j * ldc + i] := h_C[j * ldc + i]
      else
        h_C_initial_full[j * ldc + i] := h_C[i * ldc + j];

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to k - 1 do
      begin
        term1 := h_A[l * lda + i];
        term2 := h_A[l * lda + j];
        ComplexMul(term1, term2);
        ComplexAdd(sum, term1);
      end;
      term1 := h_C_initial_full[j * ldc + i];
      ComplexMul(term1, beta);
      ComplexMul(sum, alpha);
      ComplexAdd(sum, term1);
      h_C_expected[j * ldc + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_zsyrk_64(handle, uplo, transA, n, k, @alpha, d_A, lda, @beta, d_C, ldc);
  AssertEquals('rocblas_zsyrk_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEquals(h_C_expected[j*ldc+i].x, h_C[j*ldc+i].x, 1e-12, 'Real part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
      CheckEquals(h_C_expected[j*ldc+i].y, h_C[j*ldc+i].y, 1e-12, 'Imag part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
    end;

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_ssyr2k_64;
const
  n: Int64 = 4;
  k: Int64 = 5;
  trans = rocblas_operation_none;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4; // if trans=none, lda >= n
  ldb: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_B, h_C, h_C_expected, h_C_initial_full: array of Single;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: Single;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum1, sum2: Single;
begin
  alpha := 1.5;
  beta := 1.2;

  size_A := lda * k * SizeOf(Single);
  size_B := ldb * k * SizeOf(Single);
  size_C := ldc * n * SizeOf(Single);

  SetLength(h_A, lda * k);
  SetLength(h_B, ldb * k);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);
  SetLength(h_C_initial_full, ldc * n);

  // Initialize matrices with column-major layout
  for i := 0 to (lda * k) - 1 do h_A[i] := (i+1) * 0.1;
  for i := 0 to (ldb * k) - 1 do h_B[i] := (i+1) * 0.2;
  for j := 0 to n - 1 do
    for i := 0 to j do
      h_C[j * ldc + i] := (i+j+1) * 0.3;

  // Create a full initial C matrix for the reference calculation
  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
      if i <= j then h_C_initial_full[j * ldc + i] := h_C[j * ldc + i]
      else h_C_initial_full[j * ldc + i] := h_C[i * ldc + j];

  // Reference CPU calculation: C := alpha*(A*B^T + B*A^T) + beta*C
  for j := 0 to n - 1 do
    for i := 0 to j do // Only calculate the upper triangle
    begin
      sum1 := 0.0; // for A * B^T
      sum2 := 0.0; // for B * A^T
      for l := 0 to k - 1 do
      begin
        sum1 := sum1 + h_A[l * lda + i] * h_B[l * ldb + j]; // A(i,l) * B(j,l)
        sum2 := sum2 + h_B[l * ldb + i] * h_A[l * lda + j]; // B(i,l) * A(j,l)
      end;
      h_C_expected[j * ldc + i] := alpha * (sum1 + sum2) + beta * h_C_initial_full[j * ldc + i];
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_ssyr2k_64(handle, uplo, trans, n, k, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_ssyr2k_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for j := 0 to n - 1 do
    for i := 0 to j do
      CheckEquals(h_C_expected[j*ldc+i], h_C[j*ldc+i], 1e-4, 'Mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_dsyr2k_64;
const
  n: Int64 = 4;
  k: Int64 = 5;
  trans = rocblas_operation_none;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4;
  ldb: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_B, h_C, h_C_expected, h_C_initial_full: array of Double;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: Double;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum1, sum2: Double;
begin
  alpha := 1.5;
  beta := 1.2;

  size_A := lda * k * SizeOf(Double);
  size_B := ldb * k * SizeOf(Double);
  size_C := ldc * n * SizeOf(Double);

  SetLength(h_A, lda * k);
  SetLength(h_B, ldb * k);
  SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n);
  SetLength(h_C_initial_full, ldc * n);

  for i := 0 to (lda * k) - 1 do h_A[i] := (i+1) * 0.1;
  for i := 0 to (ldb * k) - 1 do h_B[i] := (i+1) * 0.2;
  for j := 0 to n - 1 do
    for i := 0 to j do
      h_C[j * ldc + i] := (i+j+1) * 0.3;

  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
      if i <= j then h_C_initial_full[j * ldc + i] := h_C[j * ldc + i]
      else h_C_initial_full[j * ldc + i] := h_C[i * ldc + j];

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      sum1 := 0.0; sum2 := 0.0;
      for l := 0 to k - 1 do
      begin
        sum1 := sum1 + h_A[l * lda + i] * h_B[l * ldb + j];
        sum2 := sum2 + h_B[l * ldb + i] * h_A[l * lda + j];
      end;
      h_C_expected[j * ldc + i] := alpha * (sum1 + sum2) + beta * h_C_initial_full[j * ldc + i];
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_dsyr2k_64(handle, uplo, trans, n, k, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_dsyr2k_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for j := 0 to n - 1 do
    for i := 0 to j do
      CheckEquals(h_C_expected[j*ldc+i], h_C[j*ldc+i], 1e-12, 'Mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_csyr2k_64;
const
  n: Int64 = 4;
  k: Int64 = 5;
  trans = rocblas_operation_none;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4;
  ldb: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_B, h_C, h_C_expected, h_C_initial_full: array of rocblas_float_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_float_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum1, sum2, temp_prod: rocblas_float_complex;
begin
  alpha.x := 1.2; alpha.y := -0.5;
  beta.x  := 0.8; beta.y  := 0.3;

  size_A := lda * k * SizeOf(rocblas_float_complex);
  size_B := ldb * k * SizeOf(rocblas_float_complex);
  size_C := ldc * n * SizeOf(rocblas_float_complex);

  SetLength(h_A, lda * k); SetLength(h_B, ldb * k); SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n); SetLength(h_C_initial_full, ldc * n);

  for i := 0 to (lda * k) - 1 do
  begin
    h_A[i].x := (i+1)*0.1;
    h_A[i].y := (i-1)*0.1;
  end;
  for i := 0 to (ldb * k) - 1 do
  begin
    h_B[i].x := (i+1)*0.2;
    h_B[i].y := (i-2)*0.2;
  end;

  for j := 0 to n - 1 do
    for i := 0 to j do begin
      h_C[j * ldc + i].x := (i+j+1) * 0.3;
      h_C[j * ldc + i].y := (i-j) * 0.3;
    end;

  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
      if i <= j then h_C_initial_full[j * ldc + i] := h_C[j * ldc + i]
      else h_C_initial_full[j * ldc + i] := h_C[i * ldc + j];

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      sum1.x := 0.0; sum1.y := 0.0;
      sum2.x := 0.0; sum2.y := 0.0;
      for l := 0 to k - 1 do
      begin
        temp_prod := h_A[l * lda + i];
        ComplexMul(temp_prod, h_B[l * ldb + j]);
        ComplexAdd(sum1, temp_prod);
        temp_prod := h_B[l * ldb + i];
        ComplexMul(temp_prod, h_A[l * lda + j]);
        ComplexAdd(sum2, temp_prod);
      end;
      ComplexAdd(sum1, sum2); // sum1 = A*B^T + B*A^T
      ComplexMul(sum1, alpha);
      temp_prod := h_C_initial_full[j * ldc + i];
      ComplexMul(temp_prod, beta);
      ComplexAdd(sum1, temp_prod);
      h_C_expected[j * ldc + i] := sum1;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_csyr2k_64(handle, uplo, trans, n, k, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_csyr2k_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEquals(h_C_expected[j*ldc+i].x, h_C[j*ldc+i].x, 1e-4, 'Real part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
      CheckEquals(h_C_expected[j*ldc+i].y, h_C[j*ldc+i].y, 1e-4, 'Imag part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
    end;

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_zsyr2k_64;
const
  n: Int64 = 4;
  k: Int64 = 5;
  trans = rocblas_operation_none;
  uplo = rocblas_fill_upper;
  lda: Int64 = 4;
  ldb: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_B, h_C, h_C_expected, h_C_initial_full: array of rocblas_double_complex;
  d_A, d_B, d_C: hipDeviceptr_t;
  alpha, beta: rocblas_double_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B, size_C: SizeUInt;
  i, j, l: Integer;
  sum1, sum2, temp_prod: rocblas_double_complex;
begin
  alpha.x := 1.2; alpha.y := -0.5;
  beta.x  := 0.8; beta.y  := 0.3;

  size_A := lda * k * SizeOf(rocblas_double_complex);
  size_B := ldb * k * SizeOf(rocblas_double_complex);
  size_C := ldc * n * SizeOf(rocblas_double_complex);

  SetLength(h_A, lda * k); SetLength(h_B, ldb * k); SetLength(h_C, ldc * n);
  SetLength(h_C_expected, ldc * n); SetLength(h_C_initial_full, ldc * n);

  for i := 0 to (lda * k) - 1 do begin h_A[i].x := (i+1)*0.1; h_A[i].y := (i-1)*0.1; end;
  for i := 0 to (ldb * k) - 1 do begin h_B[i].x := (i+1)*0.2; h_B[i].y := (i-2)*0.2; end;

  for j := 0 to n - 1 do
    for i := 0 to j do begin
      h_C[j * ldc + i].x := (i+j+1) * 0.3;
      h_C[j * ldc + i].y := (i-j) * 0.3;
    end;

  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
      if i <= j then h_C_initial_full[j * ldc + i] := h_C[j * ldc + i]
      else h_C_initial_full[j * ldc + i] := h_C[i * ldc + j];

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      sum1.x := 0.0; sum1.y := 0.0;
      sum2.x := 0.0; sum2.y := 0.0;
      for l := 0 to k - 1 do
      begin
        temp_prod := h_A[l * lda + i]; ComplexMul(temp_prod, h_B[l * ldb + j]); ComplexAdd(sum1, temp_prod);
        temp_prod := h_B[l * ldb + i]; ComplexMul(temp_prod, h_A[l * lda + j]); ComplexAdd(sum2, temp_prod);
      end;
      ComplexAdd(sum1, sum2);
      ComplexMul(sum1, alpha);
      temp_prod := h_C_initial_full[j * ldc + i];
      ComplexMul(temp_prod, beta);
      ComplexAdd(sum1, temp_prod);
      h_C_expected[j * ldc + i] := sum1;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_C, size_C); AssertEquals('hipMalloc d_C', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_C, @h_C[0], size_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_C', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_zsyr2k_64(handle, uplo, trans, n, k, @alpha, d_A, lda, d_B, ldb, @beta, d_C, ldc);
  AssertEquals('rocblas_zsyr2k_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_C[0], d_C, size_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_C -> h_C failed', HIP_SUCCESS, hip_status);

  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEquals(h_C_expected[j*ldc+i].x, h_C[j*ldc+i].x, 1e-12, 'Real part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
      CheckEquals(h_C_expected[j*ldc+i].y, h_C[j*ldc+i].y, 1e-12, 'Imag part mismatch at ('+IntToStr(i)+','+IntToStr(j)+')');
    end;

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_C); AssertEquals('hipFree d_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_strmm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  lda: Int64 = 4;
  ldb: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_A_full, h_B_C, h_B_C_expected: array of Single;
  d_A, d_B_C: hipDeviceptr_t;
  alpha: Single;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B_C: SizeUInt;
  i, j, l: Integer;
  sum: Single;
begin
  alpha := 1.5;
  size_A := lda * m * SizeOf(Single);
  size_B_C := ldc * n * SizeOf(Single);
  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B_C, ldc * n);
  SetLength(h_B_C_expected, ldc * n);

  for j := 0 to m - 1 do
    for i := 0 to j do
      h_A[j * lda + i] := (i+1) + (j+1) * 0.1;
  for i := 0 to (ldc * n) - 1 do h_B_C[i] := (i+1) * 0.5;

  FillChar(h_A_full[0], Length(h_A_full) * SizeOf(Single), 0);
  for j := 0 to m - 1 do
    for i := 0 to j do
      h_A_full[j * lda + i] := h_A[j * lda + i];

  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum := 0.0;
      for l := 0 to m - 1 do
        sum := sum + h_A_full[l * lda + i] * h_B_C[j * ldb + l];
      h_B_C_expected[j * ldc + i] := alpha * sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B_C, size_B_C); AssertEquals('hipMalloc d_B_C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B_C, @h_B_C[0], size_B_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B_C', HIP_SUCCESS, hip_status);
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_strmm_64(handle, side, uplo, transA, diag, m, n, @alpha, d_A, lda, d_B_C, ldb, d_B_C, ldc);
  AssertEquals('rocblas_strmm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_B_C[0], d_B_C, size_B_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_B_C -> h_B_C failed', HIP_SUCCESS, hip_status);
  for i := 0 to (ldc * n) - 1 do
    CheckEquals(h_B_C_expected[i], h_B_C[i], 1e-4, 'Mismatch at index ' + IntToStr(i));

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B_C); AssertEquals('hipFree d_B_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_dtrmm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  lda: Int64 = 4;
  ldb: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_A_full, h_B_C, h_B_C_expected: array of Double;
  d_A, d_B_C: hipDeviceptr_t;
  alpha: Double;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B_C: SizeUInt;
  i, j, l: Integer;
  sum: Double;
begin
  alpha := 1.5;
  size_A := lda * m * SizeOf(Double);
  size_B_C := ldc * n * SizeOf(Double);
  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B_C, ldc * n);
  SetLength(h_B_C_expected, ldc * n);

  for j := 0 to m - 1 do
    for i := 0 to j do
      h_A[j * lda + i] := (i+1) + (j+1) * 0.1;
  for i := 0 to (ldc * n) - 1 do h_B_C[i] := (i+1) * 0.5;

  FillChar(h_A_full[0], Length(h_A_full) * SizeOf(Double), 0);
  for j := 0 to m - 1 do
    for i := 0 to j do
      h_A_full[j * lda + i] := h_A[j * lda + i];

  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum := 0.0;
      for l := 0 to m - 1 do
        sum := sum + h_A_full[l * lda + i] * h_B_C[j * ldb + l];
      h_B_C_expected[j * ldc + i] := alpha * sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B_C, size_B_C); AssertEquals('hipMalloc d_B_C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B_C, @h_B_C[0], size_B_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B_C', HIP_SUCCESS, hip_status);
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_dtrmm_64(handle, side, uplo, transA, diag, m, n, @alpha, d_A, lda, d_B_C, ldb, d_B_C, ldc);
  AssertEquals('rocblas_dtrmm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_B_C[0], d_B_C, size_B_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_B_C -> h_B_C failed', HIP_SUCCESS, hip_status);
  for i := 0 to (ldc * n) - 1 do
    CheckEquals(h_B_C_expected[i], h_B_C[i], 1e-12, 'Mismatch at index ' + IntToStr(i));

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B_C); AssertEquals('hipFree d_B_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_ctrmm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  lda: Int64 = 4;
  ldb: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_A_full, h_B_C, h_B_C_expected: array of rocblas_float_complex;
  d_A, d_B_C: hipDeviceptr_t;
  alpha: rocblas_float_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B_C: SizeUInt;
  i, j, l: Integer;
  sum, prod: rocblas_float_complex;
begin
  alpha.x := 1.2; alpha.y := -0.5;
  size_A := lda * m * SizeOf(rocblas_float_complex);
  size_B_C := ldc * n * SizeOf(rocblas_float_complex);
  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B_C, ldc * n);
  SetLength(h_B_C_expected, ldc * n);

  for j := 0 to m - 1 do
    for i := 0 to j do begin
      h_A[j * lda + i].x := (i+1) * 0.1;
      h_A[j * lda + i].y := (j+1) * 0.2;
    end;
  for i := 0 to (ldc * n) - 1 do begin h_B_C[i].x := (i+1); h_B_C[i].y := -(i+1); end;

  FillChar(h_A_full[0], Length(h_A_full) * SizeOf(rocblas_float_complex), 0);
  for j := 0 to m - 1 do
    for i := 0 to j do
      h_A_full[j * lda + i] := h_A[j * lda + i];

  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to m - 1 do
      begin
        prod := h_A_full[l * lda + i];
        ComplexMul(prod, h_B_C[j * ldb + l]);
        ComplexAdd(sum, prod);
      end;
      ComplexMul(sum, alpha);
      h_B_C_expected[j * ldc + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B_C, size_B_C); AssertEquals('hipMalloc d_B_C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B_C, @h_B_C[0], size_B_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B_C', HIP_SUCCESS, hip_status);
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_ctrmm_64(handle, side, uplo, transA, diag, m, n, @alpha, d_A, lda, d_B_C, ldb, d_B_C, ldc);
  AssertEquals('rocblas_ctrmm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_B_C[0], d_B_C, size_B_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_B_C -> h_B_C failed', HIP_SUCCESS, hip_status);
  for i := 0 to (ldc * n) - 1 do
  begin
    CheckEquals(h_B_C_expected[i].x, h_B_C[i].x, 1e-4, 'Real part mismatch at ' + IntToStr(i));
    CheckEquals(h_B_C_expected[i].y, h_B_C[i].y, 1e-4, 'Imag part mismatch at ' + IntToStr(i));
  end;

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B_C); AssertEquals('hipFree d_B_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_ztrmm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  lda: Int64 = 4;
  ldb: Int64 = 4;
  ldc: Int64 = 4;
var
  h_A, h_A_full, h_B_C, h_B_C_expected: array of rocblas_double_complex;
  d_A, d_B_C: hipDeviceptr_t;
  alpha: rocblas_double_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B_C: SizeUInt;
  i, j, l: Integer;
  sum, prod: rocblas_double_complex;
begin
  alpha.x := 1.2; alpha.y := -0.5;
  size_A := lda * m * SizeOf(rocblas_double_complex);
  size_B_C := ldc * n * SizeOf(rocblas_double_complex);
  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B_C, ldc * n);
  SetLength(h_B_C_expected, ldc * n);

  for j := 0 to m - 1 do
    for i := 0 to j do begin
      h_A[j * lda + i].x := (i+1) * 0.1;
      h_A[j * lda + i].y := (j+1) * 0.2;
    end;
  for i := 0 to (ldc * n) - 1 do begin h_B_C[i].x := (i+1); h_B_C[i].y := -(i+1); end;

  FillChar(h_A_full[0], Length(h_A_full) * SizeOf(rocblas_double_complex), 0);
  for j := 0 to m - 1 do
    for i := 0 to j do
      h_A_full[j * lda + i] := h_A[j * lda + i];

  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to m - 1 do
      begin
        prod := h_A_full[l * lda + i];
        ComplexMul(prod, h_B_C[j * ldb + l]);
        ComplexAdd(sum, prod);
      end;
      ComplexMul(sum, alpha);
      h_B_C_expected[j * ldc + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B_C, size_B_C); AssertEquals('hipMalloc d_B_C', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B_C, @h_B_C[0], size_B_C, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B_C', HIP_SUCCESS, hip_status);
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_ztrmm_64(handle, side, uplo, transA, diag, m, n, @alpha, d_A, lda, d_B_C, ldb, d_B_C, ldc);
  AssertEquals('rocblas_ztrmm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_B_C[0], d_B_C, size_B_C, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_B_C -> h_B_C failed', HIP_SUCCESS, hip_status);
  for i := 0 to (ldc * n) - 1 do
  begin
    CheckEquals(h_B_C_expected[i].x, h_B_C[i].x, 1e-12, 'Real part mismatch at ' + IntToStr(i));
    CheckEquals(h_B_C_expected[i].y, h_B_C[i].y, 1e-12, 'Imag part mismatch at ' + IntToStr(i));
  end;

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B_C); AssertEquals('hipFree d_B_C', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_strsm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  lda: Int64 = 4;
  ldb: Int64 = 4;
var
  h_A, h_A_full, h_B, h_X_expected: array of Single;
  d_A, d_B: hipDeviceptr_t;
  alpha: Single;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B: SizeUInt;
  i, j, l: Integer;
  sum: Single;
begin
  alpha := 1.0; // Use alpha=1.0 for straightforward verification

  size_A := lda * m * SizeOf(Single);
  size_B := ldb * n * SizeOf(Single);

  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B, ldb * n);
  SetLength(h_X_expected, ldb * n);

  // Initialize upper triangular A (column-major) with non-zero diagonal to ensure it's invertible
  for j := 0 to m - 1 do
    for i := 0 to j do
      if i = j then h_A[j * lda + i] := 2.0 + i
      else h_A[j * lda + i] := (i+1) + (j+1) * 0.1;

  // Initialize a known solution matrix X
  for i := 0 to (ldb * n) - 1 do h_X_expected[i] := (i+1) * 0.5;

  // Create full triangular matrix A for CPU calculation
  FillChar(h_A_full[0], Length(h_A_full) * SizeOf(Single), 0);
  for j := 0 to m - 1 do
    for i := 0 to j do
      h_A_full[j * lda + i] := h_A[j * lda + i];

  // Calculate B = A * X. This will be the input to strsm.
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum := 0.0;
      for l := 0 to m - 1 do
        sum := sum + h_A_full[l * lda + i] * h_X_expected[j * ldb + l];
      h_B[j * ldb + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  // This is the clean call that should work with the fixed rocblas.pas
  blas_status := rocblas_strsm_64(handle, side, uplo, transA, diag, m, n, @alpha, d_A, lda, d_B, ldb);
  AssertEquals('rocblas_strsm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_B[0], d_B, size_B, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_B -> h_B failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldb * n) - 1 do
    CheckEquals(h_X_expected[i], h_B[i], 1e-4, 'Mismatch at index ' + IntToStr(i));

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_dtrsm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  lda: Int64 = 4;
  ldb: Int64 = 4;
var
  h_A, h_A_full, h_B, h_X_expected: array of Double;
  d_A, d_B: hipDeviceptr_t;
  alpha: Double;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B: SizeUInt;
  i, j, l: Integer;
  sum: Double;
begin
  alpha := 1.0; // Use alpha=1.0 for straightforward verification

  size_A := lda * m * SizeOf(Double);
  size_B := ldb * n * SizeOf(Double);

  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B, ldb * n);
  SetLength(h_X_expected, ldb * n);

  // Initialize upper triangular A (column-major) with non-zero diagonal to ensure it's invertible
  for j := 0 to m - 1 do
    for i := 0 to j do
      if i = j then h_A[j * lda + i] := 2.0 + i
      else h_A[j * lda + i] := (i+1) + (j+1) * 0.1;

  // Initialize a known solution matrix X
  for i := 0 to (ldb * n) - 1 do h_X_expected[i] := (i+1) * 0.5;

  // Create full triangular matrix A for CPU calculation
  FillChar(h_A_full[0], Length(h_A_full) * SizeOf(Double), 0);
  for j := 0 to m - 1 do
    for i := 0 to j do
      h_A_full[j * lda + i] := h_A[j * lda + i];

  // Calculate B = A * X. This will be the input to strsm.
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum := 0.0;
      for l := 0 to m - 1 do
        sum := sum + h_A_full[l * lda + i] * h_X_expected[j * ldb + l];
      h_B[j * ldb + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  // Solve A*X = alpha*B. The result X overwrites B.
  blas_status := rocblas_dtrsm_64(handle, side, uplo, transA, diag, m, n, @alpha, d_A, lda, d_B, ldb);
  AssertEquals('rocblas_dtrsm_64 failed', rocblas_status_success, blas_status);

  // Copy the result (now in d_B) back to h_B to check against h_X_expected
  hip_status := hipMemcpy(@h_B[0], d_B, size_B, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_B -> h_B failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldb * n) - 1 do
    CheckEquals(h_X_expected[i], h_B[i], 1e-12, 'Mismatch at index ' + IntToStr(i));

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_ctrsm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  lda: Int64 = 4;
  ldb: Int64 = 4;
var
  h_A, h_A_full, h_B, h_X_expected: array of rocblas_float_complex;
  d_A, d_B: hipDeviceptr_t;
  alpha: rocblas_float_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B: SizeUInt;
  i, j, l: Integer;
  sum, prod: rocblas_float_complex;
begin
  alpha.x := 1.0; alpha.y := 0.0;

  size_A := lda * m * SizeOf(rocblas_float_complex);
  size_B := ldb * n * SizeOf(rocblas_float_complex);

  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B, ldb * n);
  SetLength(h_X_expected, ldb * n);

  for j := 0 to m - 1 do
    for i := 0 to j do begin
      if i = j then h_A[j*lda+i].x := 2.0+i else h_A[j*lda+i].x := (i+1)*0.1;
      h_A[j * lda + i].y := (j+1) * 0.2;
    end;

  for i := 0 to (ldb * n) - 1 do begin h_X_expected[i].x := (i+1); h_X_expected[i].y := -(i+1); end;

  FillChar(h_A_full[0], Length(h_A_full) * SizeOf(rocblas_float_complex), 0);
  for j := 0 to m - 1 do
    for i := 0 to j do
      h_A_full[j * lda + i] := h_A[j * lda + i];

  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to m - 1 do
      begin
        prod := h_A_full[l * lda + i];
        ComplexMul(prod, h_X_expected[j * ldb + l]);
        ComplexAdd(sum, prod);
      end;
      h_B[j * ldb + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_ctrsm_64(handle, side, uplo, transA, diag, m, n, @alpha, d_A, lda, d_B, ldb);
  AssertEquals('rocblas_ctrsm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_B[0], d_B, size_B, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_B -> h_B failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldb * n) - 1 do
  begin
    CheckEquals(h_X_expected[i].x, h_B[i].x, 1e-4, 'Real part mismatch at ' + IntToStr(i));
    CheckEquals(h_X_expected[i].y, h_B[i].y, 1e-4, 'Imag part mismatch at ' + IntToStr(i));
  end;

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
end;

procedure TTestRocBLAS_Level3_64.Test_rocblas_ztrsm_64;
const
  m: Int64 = 4;
  n: Int64 = 5;
  side = rocblas_side_left;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  lda: Int64 = 4;
  ldb: Int64 = 4;
var
  h_A, h_A_full, h_B, h_X_expected: array of rocblas_double_complex;
  d_A, d_B: hipDeviceptr_t;
  alpha: rocblas_double_complex;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  size_A, size_B: SizeUInt;
  i, j, l: Integer;
  sum, prod: rocblas_double_complex;
begin
  alpha.x := 1.0; alpha.y := 0.0;

  size_A := lda * m * SizeOf(rocblas_double_complex);
  size_B := ldb * n * SizeOf(rocblas_double_complex);

  SetLength(h_A, lda * m);
  SetLength(h_A_full, lda * m);
  SetLength(h_B, ldb * n);
  SetLength(h_X_expected, ldb * n);

  for j := 0 to m - 1 do
    for i := 0 to j do begin
      if i = j then h_A[j*lda+i].x := 2.0+i else h_A[j*lda+i].x := (i+1)*0.1;
      h_A[j * lda + i].y := (j+1) * 0.2;
    end;

  for i := 0 to (ldb * n) - 1 do begin h_X_expected[i].x := (i+1); h_X_expected[i].y := -(i+1); end;

  FillChar(h_A_full[0], Length(h_A_full) * SizeOf(rocblas_double_complex), 0);
  for j := 0 to m - 1 do
    for i := 0 to j do
      h_A_full[j * lda + i] := h_A[j * lda + i];

  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for l := 0 to m - 1 do
      begin
        prod := h_A_full[l * lda + i];
        ComplexMul(prod, h_X_expected[j * ldb + l]);
        ComplexAdd(sum, prod);
      end;
      h_B[j * ldb + i] := sum;
    end;

  hip_status := hipMalloc(d_A, size_A); AssertEquals('hipMalloc d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_B, size_B); AssertEquals('hipMalloc d_B', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_A, @h_A[0], size_A, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_A', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_B, @h_B[0], size_B, hipMemcpyHostToDevice); AssertEquals('hipMemcpy d_B', HIP_SUCCESS, hip_status);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', rocblas_status_success, blas_status);

  blas_status := rocblas_ztrsm_64(handle, side, uplo, transA, diag, m, n, @alpha, d_A, lda, d_B, ldb);
  AssertEquals('rocblas_ztrsm_64 failed', rocblas_status_success, blas_status);

  hip_status := hipMemcpy(@h_B[0], d_B, size_B, hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_B -> h_B failed', HIP_SUCCESS, hip_status);

  for i := 0 to (ldb * n) - 1 do
  begin
    CheckEquals(h_X_expected[i].x, h_B[i].x, 1e-12, 'Real part mismatch at ' + IntToStr(i));
    CheckEquals(h_X_expected[i].y, h_B[i].y, 1e-12, 'Imag part mismatch at ' + IntToStr(i));
  end;

  hip_status := hipFree(d_A); AssertEquals('hipFree d_A', HIP_SUCCESS, hip_status);
  hip_status := hipFree(d_B); AssertEquals('hipFree d_B', HIP_SUCCESS, hip_status);
end;

initialization
  RegisterTest(TTestRocBLAS_Level3_64);

end.

