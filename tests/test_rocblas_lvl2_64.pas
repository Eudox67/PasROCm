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

  Signature checks for the rocBLAS level-2 (matrix-vector, 64-bit index
  variants) bindings: every call is made and its status is asserted, so a
  changed prototype or a missing symbol fails here first.
------------------------------------------------------------------------------}

unit Test_rocblas_lvl2_64;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  { TTestRocBLAS_Level2_64 }

  TTestRocBLAS_Level2_64 = class(TTestRocBLAS_Base)
  published
    { General Matrix-Vector }
    procedure Test_SGEMV_64;
    procedure Test_DGEMV_64;
    procedure Test_CGEMV_64;
    procedure Test_ZGEMV_64;
    procedure Test_SGBMV_64;
    procedure Test_DGBMV_64;
    procedure Test_CGBMV_64;
    procedure Test_ZGBMV_64;

    { Hermitian }
    procedure Test_CHEMV_64;
    procedure Test_ZHEMV_64;
    procedure Test_CHBMV_64;
    procedure Test_ZHBMV_64;
    procedure Test_CHPMV_64;
    procedure Test_ZHPMV_64;

    { Symmetric }
    procedure Test_SSYMV_64;
    procedure Test_DSYMV_64;
    procedure Test_CSYMV_64;
    procedure Test_ZSYMV_64;
    procedure Test_SSBMV_64;
    procedure Test_DSBMV_64;
    procedure Test_SSPMV_64;
    procedure Test_DSPMV_64;

    { Triangular }
    procedure Test_STRMV_64;
    procedure Test_DTRMV_64;
    procedure Test_CTRMV_64;
    procedure Test_ZTRMV_64;
    procedure Test_STBMV_64;
    procedure Test_DTBMV_64;
    procedure Test_CTBMV_64;
    procedure Test_ZTBMV_64;
    procedure Test_STPMV_64;
    procedure Test_DTPMV_64;
    procedure Test_CTPMV_64;
    procedure Test_ZTPMV_64;

    { Triangular Solvers }
    procedure Test_STRSV_64;
    procedure Test_DTRSV_64;
    procedure Test_CTRSV_64;
    procedure Test_ZTRSV_64;
    procedure Test_STBSV_64;
    procedure Test_DTBSV_64;
    procedure Test_CTBSV_64;
    procedure Test_ZTBSV_64;
    procedure Test_STPSV_64;
    procedure Test_DTPSV_64;
    procedure Test_CTPSV_64;
    procedure Test_ZTPSV_64;

    { Rank-1 Updates }
    procedure Test_SGER_64;
    procedure Test_DGER_64;
    procedure Test_CGERU_64;
    procedure Test_ZGERU_64;
    procedure Test_CGERC_64;
    procedure Test_ZGERC_64;
    procedure Test_SSYR_64;
    procedure Test_DSYR_64;
    procedure Test_CSYR_64;
    procedure Test_ZSYR_64;
    procedure Test_CHER_64;
    procedure Test_ZHER_64;
    procedure Test_SSPR_64;
    procedure Test_DSPR_64;
    procedure Test_CHPR_64;
    procedure Test_ZHPR_64;

    { Rank-2 Updates }
    procedure Test_SSYR2_64;
    procedure Test_DSYR2_64;
    procedure Test_CSYR2_64;
    procedure Test_ZSYR2_64;
    procedure Test_CHER2_64;
    procedure Test_ZHER2_64;
    procedure Test_SSPR2_64;
    procedure Test_DSPR2_64;
    procedure Test_CHPR2_64;
    procedure Test_ZHPR2_64;
  end;

implementation

{ TTestRocBLAS_Level2_64 }

procedure TTestRocBLAS_Level2_64.Test_SGEMV_64;
const
  m: Int64 = 10;
  n: Int64 = 12;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Single = 1.5;
  beta: Single = 2.5;
var
  h_A: array of Single;
  h_x, h_y, h_y_expected: array of Single;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j, y_len, x_len: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  trans: rocblas_operation;
begin
  trans := rocblas_operation_none;
  if trans = rocblas_operation_none then
  begin
    x_len := n;
    y_len := m;
  end
  else
  begin
    x_len := m;
    y_len := n;
  end;

  SetLength(h_A, lda * n);
  SetLength(h_x, x_len * incx);
  SetLength(h_y, y_len * incy);
  SetLength(h_y_expected, y_len * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do h_A[i] := (i mod 10) + 1.2;
  for i := 0 to (x_len * incx) - 1 do h_x[i] := (i mod 5) + 1.1;
  for i := 0 to (y_len * incy) - 1 do
  begin
    h_y[i] := (i mod 3) + 1.3;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation: y := alpha*A*x + beta*y
  for i := 0 to y_len - 1 do
  begin
    h_y_expected[i * incy] := beta * h_y_expected[i * incy];
    for j := 0 to x_len - 1 do
    begin
       h_y_expected[i * incy] := h_y_expected[i * incy] + alpha * h_A[i + j*lda] * h_x[j * incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Single));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, x_len * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, y_len * incy * SizeOf(Single));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], x_len * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], y_len * incy * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sgemv_64(handle, trans, m, n, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_sgemv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, y_len * incy * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to y_len - 1 do
    CheckEqualsRelative(h_y_expected[i * incy], h_y[i * incy], 1e-5, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_DGEMV_64;
const
  m: Int64 = 10;
  n: Int64 = 12;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Double = 1.5;
  beta: Double = 2.5;
var
  h_A: array of Double;
  h_x, h_y, h_y_expected: array of Double;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j, y_len, x_len: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  trans: rocblas_operation;
begin
  trans := rocblas_operation_none;
  if trans = rocblas_operation_none then
  begin
    x_len := n;
    y_len := m;
  end
  else
  begin
    x_len := m;
    y_len := n;
  end;

  SetLength(h_A, lda * n);
  SetLength(h_x, x_len * incx);
  SetLength(h_y, y_len * incy);
  SetLength(h_y_expected, y_len * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do h_A[i] := (i mod 10) + 1.2;
  for i := 0 to (x_len * incx) - 1 do h_x[i] := (i mod 5) + 1.1;
  for i := 0 to (y_len * incy) - 1 do
  begin
    h_y[i] := (i mod 3) + 1.3;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation: y := alpha*A*x + beta*y
  for i := 0 to y_len - 1 do
  begin
    h_y_expected[i * incy] := beta * h_y_expected[i * incy];
    for j := 0 to x_len - 1 do
    begin
       h_y_expected[i * incy] := h_y_expected[i * incy] + alpha * h_A[i + j*lda] * h_x[j * incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Double));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, x_len * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, y_len * incy * SizeOf(Double));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], x_len * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], y_len * incy * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dgemv_64(handle, trans, m, n, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_dgemv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, y_len * incy * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification using relative tolerance
  for i := 0 to y_len - 1 do
    CheckEqualsRelative(h_y_expected[i * incy], h_y[i * incy], 1e-12, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;


procedure TTestRocBLAS_Level2_64.Test_CGEMV_64;
const
  m: Int64 = 10;
  n: Int64 = 12;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha, beta: rocblas_float_complex;
  h_A: array of rocblas_float_complex;
  h_x, h_y, h_y_expected: array of rocblas_float_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j, y_len, x_len: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  trans: rocblas_operation;
  temp: rocblas_float_complex;
begin
  alpha.x := 1.5; alpha.y := 0.5;
  beta.x := 2.5; beta.y := 1.0;
  trans := rocblas_operation_none;

  if trans = rocblas_operation_none then
  begin
    x_len := n;
    y_len := m;
  end
  else
  begin
    x_len := m;
    y_len := n;
  end;

  SetLength(h_A, lda * n);
  SetLength(h_x, x_len * incx);
  SetLength(h_y, y_len * incy);
  SetLength(h_y_expected, y_len * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i].x := i + 1.2; h_A[i].y := i - 0.8; end;
  for i := 0 to (x_len * incx) - 1 do begin h_x[i].x := i + 1.1; h_x[i].y := i - 0.9; end;
  for i := 0 to (y_len * incy) - 1 do
  begin
    h_y[i].x := i + 1.3; h_y[i].y := i - 0.7;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation: y := alpha*A*x + beta*y
  for i := 0 to y_len - 1 do
  begin
    ComplexMul(h_y_expected[i * incy], beta);
    for j := 0 to x_len - 1 do
    begin
       temp := h_A[i + j*lda];
       ComplexMul(temp, h_x[j * incx]);
       ComplexMul(temp, alpha);
       ComplexAdd(h_y_expected[i * incy], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, x_len * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, y_len * incy * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], x_len * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], y_len * incy * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgemv_64(handle, trans, m, n, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_cgemv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, y_len * incy * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to y_len - 1 do
  begin
    CheckEqualsRelative(h_y_expected[i * incy].x, h_y[i * incy].x, 1e-5, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_y_expected[i * incy].y, h_y[i * incy].y, 1e-5, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_ZGEMV_64;
const
  m: Int64 = 10;
  n: Int64 = 12;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha, beta: rocblas_double_complex;
  h_A: array of rocblas_double_complex;
  h_x, h_y, h_y_expected: array of rocblas_double_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j, y_len, x_len: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  trans: rocblas_operation;
  temp: rocblas_double_complex;
begin
  alpha.x := 1.5; alpha.y := 0.5;
  beta.x := 2.5; beta.y := 1.0;
  trans := rocblas_operation_none;

  if trans = rocblas_operation_none then
  begin
    x_len := n;
    y_len := m;
  end
  else
  begin
    x_len := m;
    y_len := n;
  end;

  SetLength(h_A, lda * n);
  SetLength(h_x, x_len * incx);
  SetLength(h_y, y_len * incy);
  SetLength(h_y_expected, y_len * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i].x := i + 1.2; h_A[i].y := i - 0.8; end;
  for i := 0 to (x_len * incx) - 1 do begin h_x[i].x := i + 1.1; h_x[i].y := i - 0.9; end;
  for i := 0 to (y_len * incy) - 1 do
  begin
    h_y[i].x := i + 1.3; h_y[i].y := i - 0.7;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation: y := alpha*A*x + beta*y
  for i := 0 to y_len - 1 do
  begin
    ComplexMul(h_y_expected[i * incy], beta);
    for j := 0 to x_len - 1 do
    begin
       temp := h_A[i + j*lda];
       ComplexMul(temp, h_x[j * incx]);
       ComplexMul(temp, alpha);
       ComplexAdd(h_y_expected[i * incy], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, x_len * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, y_len * incy * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], x_len * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], y_len * incy * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgemv_64(handle, trans, m, n, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_zgemv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, y_len * incy * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification using relative tolerance
  for i := 0 to y_len - 1 do
  begin
    CheckEqualsRelative(h_y_expected[i * incy].x, h_y[i * incy].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_y_expected[i * incy].y, h_y[i * incy].y, 1e-12, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_SGBMV_64;
const
  m: Int64 = 7;
  n: Int64 = 7;
  kl: Int64 = 2;
  ku: Int64 = 2;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Single = 1.5;
  beta: Single = 2.0;
var
  lda: Int64;
  h_A_banded: array of Single;
  h_x, h_y, h_y_expected: array of Single;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  lda := kl + ku + 1;

  SetLength(h_A_banded, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, m * incy);
  SetLength(h_y_expected, m * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do h_A_banded[i] := (i mod 10) + 1.0;
  for i := 0 to (n * incx) - 1 do h_x[i] := (i mod 5) + 1.0;
  for i := 0 to (m * incy) - 1 do
  begin
    h_y[i] := (i mod 3) + 1.0;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation: y := alpha*A*x + beta*y
  for i := 0 to m - 1 do
  begin
    h_y_expected[i*incy] := beta * h_y_expected[i*incy];
    for j := Max(0, i - kl) to Min(n - 1, i + ku) do
    begin
      // A_orig[i,j] is at A_banded[ku + i - j, j]
      h_y_expected[i*incy] := h_y_expected[i*incy] + alpha * h_A_banded[(ku + i - j) + j*lda] * h_x[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Single));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, m * incy * SizeOf(Single));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A_banded to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], m * incy * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sgbmv_64(handle, rocblas_operation_none, m, n, kl, ku, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_sgbmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, m * incy * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to m - 1 do
    CheckEqualsRelative(h_y_expected[i*incy], h_y[i*incy], 1e-4, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_DGBMV_64;
const
  m: Int64 = 7;
  n: Int64 = 7;
  kl: Int64 = 2;
  ku: Int64 = 2;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Double = 1.5;
  beta: Double = 2.0;
var
  lda: Int64;
  h_A_banded: array of Double;
  h_x, h_y, h_y_expected: array of Double;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  lda := kl + ku + 1;

  SetLength(h_A_banded, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, m * incy);
  SetLength(h_y_expected, m * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do h_A_banded[i] := (i mod 10) + 1.0;
  for i := 0 to (n * incx) - 1 do h_x[i] := (i mod 5) + 1.0;
  for i := 0 to (m * incy) - 1 do
  begin
    h_y[i] := (i mod 3) + 1.0;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation: y := alpha*A*x + beta*y
  for i := 0 to m - 1 do
  begin
    h_y_expected[i*incy] := beta * h_y_expected[i*incy];
    for j := Max(0, i - kl) to Min(n - 1, i + ku) do
    begin
      h_y_expected[i*incy] := h_y_expected[i*incy] + alpha * h_A_banded[(ku + i - j) + j*lda] * h_x[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Double));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, m * incy * SizeOf(Double));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A_banded to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], m * incy * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dgbmv_64(handle, rocblas_operation_none, m, n, kl, ku, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_dgbmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, m * incy * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to m - 1 do
    CheckEquals(h_y_expected[i*incy], h_y[i*incy], 1e-12, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_CGBMV_64;
const
  m: Int64 = 7;
  n: Int64 = 7;
  kl: Int64 = 2;
  ku: Int64 = 2;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  lda: Int64;
  alpha, beta, term: rocblas_float_complex;
  h_A_banded: array of rocblas_float_complex;
  h_x, h_y, h_y_expected: array of rocblas_float_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  lda := kl + ku + 1;
  alpha.x := 1.5; alpha.y := 0.5;
  beta.x := 2.0; beta.y := 1.0;

  SetLength(h_A_banded, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, m * incy);
  SetLength(h_y_expected, m * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A_banded[i].x := i+1.1; h_A_banded[i].y := i-1.2; end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.3; h_x[i].y := i-0.4; end;
  for i := 0 to (m * incy) - 1 do
  begin
    h_y[i].x := i+1.5; h_y[i].y := i+2.6;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation
  for i := 0 to m - 1 do
  begin
    ComplexMul(h_y_expected[i*incy], beta);
    for j := Max(0, i - kl) to Min(n - 1, i + ku) do
    begin
      term := h_A_banded[(ku + i - j) + j*lda];
      ComplexMul(term, h_x[j*incx]);
      ComplexMul(term, alpha);
      ComplexAdd(h_y_expected[i*incy], term);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, m * incy * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], m * incy * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgbmv_64(handle, rocblas_operation_none, m, n, kl, ku, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_cgbmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, m * incy * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to m - 1 do
  begin
    CheckEqualsRelative(h_y_expected[i*incy].x, h_y[i*incy].x, 1e-4, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_y_expected[i*incy].y, h_y[i*incy].y, 1e-4, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_ZGBMV_64;
const
  m: Int64 = 7;
  n: Int64 = 7;
  kl: Int64 = 2;
  ku: Int64 = 2;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  lda: Int64;
  alpha, beta, term: rocblas_double_complex;
  h_A_banded: array of rocblas_double_complex;
  h_x, h_y, h_y_expected: array of rocblas_double_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  lda := kl + ku + 1;
  alpha.x := 1.5; alpha.y := 0.5;
  beta.x := 2.0; beta.y := 1.0;

  SetLength(h_A_banded, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, m * incy);
  SetLength(h_y_expected, m * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A_banded[i].x := i+1.1; h_A_banded[i].y := i-1.2; end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.3; h_x[i].y := i-0.4; end;
  for i := 0 to (m * incy) - 1 do
  begin
    h_y[i].x := i+1.5; h_y[i].y := i+2.6;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation
  for i := 0 to m - 1 do
  begin
    ComplexMul(h_y_expected[i*incy], beta);
    for j := Max(0, i - kl) to Min(n - 1, i + ku) do
    begin
      term := h_A_banded[(ku + i - j) + j*lda];
      ComplexMul(term, h_x[j*incx]);
      ComplexMul(term, alpha);
      ComplexAdd(h_y_expected[i*incy], term);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, m * incy * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], m * incy * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgbmv_64(handle, rocblas_operation_none, m, n, kl, ku, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_zgbmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, m * incy * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to m - 1 do
  begin
    CheckEquals(h_y_expected[i*incy].x, h_y[i*incy].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEquals(h_y_expected[i*incy].y, h_y[i*incy].y, 1e-12, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_CHEMV_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha, beta: rocblas_float_complex;
  h_A: array of rocblas_float_complex;
  h_x, h_y, h_y_expected: array of rocblas_float_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  temp: rocblas_float_complex;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x := 2.0; beta.y := 1.0;

  SetLength(h_A, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize a full Hermitian matrix A
  for i := 0 to n - 1 do
  begin
    for j := i to n - 1 do
    begin
      h_A[i + j * lda].x := i + j * 2.0;
      h_A[i + j * lda].y := i - j * 0.5;
      if i = j then
        h_A[i + j * lda].y := 0.0; // Diagonal elements must be real
      h_A[j + i * lda].x := h_A[i + j * lda].x;
      h_A[j + i * lda].y := -h_A[i + j * lda].y; // Conjugate
    end;
  end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i + 1.1; h_x[i].y := i - 0.9; end;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i].x := i + 1.3; h_y[i].y := i - 0.7;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation: y := alpha*A*x + beta*y
  for i := 0 to n - 1 do
  begin
    ComplexMul(h_y_expected[i * incy], beta);
    for j := 0 to n - 1 do
    begin
       temp := h_A[i + j * lda];
       ComplexMul(temp, h_x[j * incx]);
       ComplexMul(temp, alpha);
       ComplexAdd(h_y_expected[i * incy], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_chemv_64(handle, uplo, n, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_chemv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_y_expected[i*incy].x, h_y[i*incy].x, 1e-4, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_y_expected[i*incy].y, h_y[i*incy].y, 1e-4, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_ZHEMV_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha, beta: rocblas_double_complex;
  h_A: array of rocblas_double_complex;
  h_x, h_y, h_y_expected: array of rocblas_double_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  temp: rocblas_double_complex;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x := 2.0; beta.y := 1.0;

  SetLength(h_A, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize a full Hermitian matrix A
  for i := 0 to n - 1 do
  begin
    for j := i to n - 1 do
    begin
      h_A[i + j * lda].x := i + j * 2.0;
      h_A[i + j * lda].y := i - j * 0.5;
      if i = j then
        h_A[i + j * lda].y := 0.0; // Diagonal elements must be real
      h_A[j + i * lda].x := h_A[i + j * lda].x;
      h_A[j + i * lda].y := -h_A[i + j * lda].y; // Conjugate
    end;
  end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i + 1.1; h_x[i].y := i - 0.9; end;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i].x := i + 1.3; h_y[i].y := i - 0.7;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation: y := alpha*A*x + beta*y
  for i := 0 to n - 1 do
  begin
    ComplexMul(h_y_expected[i * incy], beta);
    for j := 0 to n - 1 do
    begin
       temp := h_A[i + j * lda];
       ComplexMul(temp, h_x[j * incx]);
       ComplexMul(temp, alpha);
       ComplexAdd(h_y_expected[i * incy], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zhemv_64(handle, uplo, n, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_zhemv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_y_expected[i*incy].x, h_y[i*incy].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_y_expected[i*incy].y, h_y[i*incy].y, 1e-12, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_CHBMV_64;
const
  n: Int64 = 7;
  k: Int64 = 2;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  lda: Int64;
  uplo: rocblas_fill;
  alpha, beta, term: rocblas_float_complex;
  h_A_banded: array of rocblas_float_complex;
  h_x, h_y, h_y_expected: array of rocblas_float_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  lda := k + 1;
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x := 2.0; beta.y := 1.0;

  SetLength(h_A_banded, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A_banded[i].x := i+1.1; h_A_banded[i].y := i-1.2; end;
  // Ensure main diagonal (row k for upper fill) is real
  for j := 0 to n - 1 do h_A_banded[k + j*lda].y := 0.0;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.3; h_x[i].y := i-0.4; end;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i].x := i+1.5; h_y[i].y := i+2.6;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation
  for i := 0 to n - 1 do
  begin
    ComplexMul(h_y_expected[i*incy], beta);
    // Upper
    for j := i to Min(n - 1, i + k) do
    begin
      term := h_A_banded[(k + i - j) + j*lda]; // A_orig(i,j)
      ComplexMul(term, h_x[j*incx]);
      ComplexMul(term, alpha);
      ComplexAdd(h_y_expected[i*incy], term);
    end;
    // Lower (conjugate part)
    for j := Max(0, i - k) to i - 1 do
    begin
      term := h_A_banded[(k + j - i) + i*lda]; // A_orig(j,i)
      term.y := -term.y; // Conjugate
      ComplexMul(term, h_x[j*incx]);
      ComplexMul(term, alpha);
      ComplexAdd(h_y_expected[i*incy], term);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_chbmv_64(handle, uplo, n, k, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_chbmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_y_expected[i*incy].x, h_y[i*incy].x, 1e-4, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_y_expected[i*incy].y, h_y[i*incy].y, 1e-4, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_ZHBMV_64;
const
  n: Int64 = 7;
  k: Int64 = 2;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  lda: Int64;
  uplo: rocblas_fill;
  alpha, beta, term: rocblas_double_complex;
  h_A_banded: array of rocblas_double_complex;
  h_x, h_y, h_y_expected: array of rocblas_double_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  lda := k + 1;
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x := 2.0; beta.y := 1.0;

  SetLength(h_A_banded, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A_banded[i].x := i+1.1; h_A_banded[i].y := i-1.2; end;
  // Ensure main diagonal (row k for upper fill) is real
  for j := 0 to n - 1 do h_A_banded[k + j*lda].y := 0.0;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.3; h_x[i].y := i-0.4; end;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i].x := i+1.5; h_y[i].y := i+2.6;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation
  for i := 0 to n - 1 do
  begin
    ComplexMul(h_y_expected[i*incy], beta);
    // Upper
    for j := i to Min(n - 1, i + k) do
    begin
      term := h_A_banded[(k + i - j) + j*lda]; // A_orig(i,j)
      ComplexMul(term, h_x[j*incx]);
      ComplexMul(term, alpha);
      ComplexAdd(h_y_expected[i*incy], term);
    end;
    // Lower (conjugate part)
    for j := Max(0, i - k) to i - 1 do
    begin
      term := h_A_banded[(k + j - i) + i*lda]; // A_orig(j,i)
      term.y := -term.y; // Conjugate
      ComplexMul(term, h_x[j*incx]);
      ComplexMul(term, alpha);
      ComplexAdd(h_y_expected[i*incy], term);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zhbmv_64(handle, uplo, n, k, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_zhbmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_y_expected[i*incy].x, h_y[i*incy].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_y_expected[i*incy].y, h_y[i*incy].y, 1e-12, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_CHPMV_64;
const
  n: Int64 = 6;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  uplo: rocblas_fill;
  alpha, beta, temp: rocblas_float_complex;
  h_A_full: array of rocblas_float_complex;
  h_AP: array of rocblas_float_complex;
  h_x, h_y, h_y_expected: array of rocblas_float_complex;
  d_AP, d_x, d_y: hipDeviceptr_t;
  i, j, k: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x := 2.0; beta.y := 1.0;
  ap_size := n * (n + 1) div 2;

  SetLength(h_A_full, n * n);
  SetLength(h_AP, ap_size);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize a full Hermitian matrix A and pack it
  for i := 0 to n - 1 do
    for j := 0 to n - 1 do
    begin
      h_A_full[i + j*n].x := i + j*2.1;
      h_A_full[i + j*n].y := j - i*0.8;
      if i=j then h_A_full[i+j*n].y := 0.0;
    end;
  // Symmetrize
  for i := 0 to n - 1 do
    for j := i+1 to n - 1 do
    begin
      h_A_full[j+i*n].x := h_A_full[i+j*n].x;
      h_A_full[j+i*n].y := -h_A_full[i+j*n].y;
    end;

  // Pack the matrix
  k := 0;
  if uplo = rocblas_fill_upper then
  begin
    for j := 0 to n - 1 do
    begin
      for i := 0 to j do
      begin
        h_AP[k] := h_A_full[i + j*n];
        Inc(k);
      end;
    end;
  end
  else // lower
  begin
    for j := 0 to n - 1 do
    begin
      for i := j to n - 1 do
      begin
        h_AP[k] := h_A_full[i + j*n];
        Inc(k);
      end;
    end;
  end;

  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.3; h_x[i].y := i-0.4; end;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i].x := i+1.5; h_y[i].y := i+2.6;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation using full matrix
  for i := 0 to n - 1 do
  begin
    ComplexMul(h_y_expected[i*incy], beta);
    for j := 0 to n - 1 do
    begin
      temp := h_A_full[i + j*n];
      ComplexMul(temp, h_x[j*incx]);
      ComplexMul(temp, alpha);
      ComplexAdd(h_y_expected[i*incy], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_chpmv_64(handle, uplo, n, @alpha, d_AP, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_chpmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_y_expected[i*incy].x, h_y[i*incy].x, 1e-4, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_y_expected[i*incy].y, h_y[i*incy].y, 1e-4, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_AP);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_ZHPMV_64;
const
  n: Int64 = 6;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  uplo: rocblas_fill;
  alpha, beta, temp: rocblas_double_complex;
  h_A_full: array of rocblas_double_complex;
  h_AP: array of rocblas_double_complex;
  h_x, h_y, h_y_expected: array of rocblas_double_complex;
  d_AP, d_x, d_y: hipDeviceptr_t;
  i, j, k: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x := 2.0; beta.y := 1.0;
  ap_size := n * (n + 1) div 2;

  SetLength(h_A_full, n * n);
  SetLength(h_AP, ap_size);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize a full Hermitian matrix A and pack it
  for i := 0 to n - 1 do
    for j := 0 to n - 1 do
    begin
      h_A_full[i + j*n].x := i + j*2.1;
      h_A_full[i + j*n].y := j - i*0.8;
      if i=j then h_A_full[i+j*n].y := 0.0;
    end;
  // Symmetrize
  for i := 0 to n - 1 do
    for j := i+1 to n - 1 do
    begin
      h_A_full[j+i*n].x := h_A_full[i+j*n].x;
      h_A_full[j+i*n].y := -h_A_full[i+j*n].y;
    end;

  // Pack the matrix
  k := 0;
  if uplo = rocblas_fill_upper then
  begin
    for j := 0 to n - 1 do
    begin
      for i := 0 to j do
      begin
        h_AP[k] := h_A_full[i + j*n];
        Inc(k);
      end;
    end;
  end
  else // lower
  begin
    for j := 0 to n - 1 do
    begin
      for i := j to n - 1 do
      begin
        h_AP[k] := h_A_full[i + j*n];
        Inc(k);
      end;
    end;
  end;

  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.3; h_x[i].y := i-0.4; end;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i].x := i+1.5; h_y[i].y := i+2.6;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation using full matrix
  for i := 0 to n - 1 do
  begin
    ComplexMul(h_y_expected[i*incy], beta);
    for j := 0 to n - 1 do
    begin
      temp := h_A_full[i + j*n];
      ComplexMul(temp, h_x[j*incx]);
      ComplexMul(temp, alpha);
      ComplexAdd(h_y_expected[i*incy], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zhpmv_64(handle, uplo, n, @alpha, d_AP, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_zhpmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_y_expected[i*incy].x, h_y[i*incy].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_y_expected[i*incy].y, h_y[i*incy].y, 1e-12, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_AP);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_SSYMV_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Single = 1.5;
  beta: Single = 2.0;
var
  h_A: array of Single;
  h_x, h_y, h_y_expected: array of Single;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
begin
  uplo := rocblas_fill_upper;

  SetLength(h_A, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize a full symmetric matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
    begin
      h_A[i + j * lda] := i + j * 2.1;
      h_A[j + i * lda] := h_A[i + j * lda]; // Symmetric
    end;
  for i := 0 to (n * incx) - 1 do h_x[i] := i + 1.1;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i] := i + 1.3;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation: y := alpha*A*x + beta*y
  for i := 0 to n - 1 do
  begin
    h_y_expected[i * incy] := beta * h_y_expected[i * incy];
    for j := 0 to n - 1 do
    begin
      h_y_expected[i * incy] := h_y_expected[i * incy] + alpha * h_A[i + j * lda] * h_x[j * incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Single));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(Single));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_ssymv_64(handle, uplo, n, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_ssymv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_y_expected[i*incy], h_y[i*incy], 1e-5, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_DSYMV_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Double = 1.5;
  beta: Double = 2.0;
var
  h_A: array of Double;
  h_x, h_y, h_y_expected: array of Double;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
begin
  uplo := rocblas_fill_upper;

  SetLength(h_A, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize a full symmetric matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
    begin
      h_A[i + j * lda] := i + j * 2.1;
      h_A[j + i * lda] := h_A[i + j * lda]; // Symmetric
    end;
  for i := 0 to (n * incx) - 1 do h_x[i] := i + 1.1;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i] := i + 1.3;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation: y := alpha*A*x + beta*y
  for i := 0 to n - 1 do
  begin
    h_y_expected[i * incy] := beta * h_y_expected[i * incy];
    for j := 0 to n - 1 do
    begin
      h_y_expected[i * incy] := h_y_expected[i * incy] + alpha * h_A[i + j * lda] * h_x[j * incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Double));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(Double));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dsymv_64(handle, uplo, n, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_dsymv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_y_expected[i*incy], h_y[i*incy], 1e-12, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_CSYMV_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha, beta: rocblas_float_complex;
  h_A: array of rocblas_float_complex;
  h_x, h_y, h_y_expected: array of rocblas_float_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  temp: rocblas_float_complex;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x := 2.0; beta.y := 1.0;

  SetLength(h_A, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize a full complex symmetric matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
    begin
      h_A[i + j * lda].x := i + j * 2.1;
      h_A[i + j * lda].y := j - i * 0.8;
      h_A[j + i * lda] := h_A[i + j * lda]; // Symmetric
    end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i + 1.1; h_x[i].y := i - 0.9; end;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i].x := i + 1.3; h_y[i].y := i - 0.7;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation: y := alpha*A*x + beta*y
  for i := 0 to n - 1 do
  begin
    ComplexMul(h_y_expected[i * incy], beta);
    for j := 0 to n - 1 do
    begin
       temp := h_A[i + j * lda];
       ComplexMul(temp, h_x[j * incx]);
       ComplexMul(temp, alpha);
       ComplexAdd(h_y_expected[i * incy], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_csymv_64(handle, uplo, n, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_csymv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_y_expected[i*incy].x, h_y[i*incy].x, 1e-4, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_y_expected[i*incy].y, h_y[i*incy].y, 1e-4, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_ZSYMV_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha, beta: rocblas_double_complex;
  h_A: array of rocblas_double_complex;
  h_x, h_y, h_y_expected: array of rocblas_double_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  temp: rocblas_double_complex;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x := 2.0; beta.y := 1.0;

  SetLength(h_A, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize a full complex symmetric matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
    begin
      h_A[i + j * lda].x := i + j * 2.1;
      h_A[i + j * lda].y := j - i * 0.8;
      h_A[j + i * lda] := h_A[i + j * lda]; // Symmetric
    end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i + 1.1; h_x[i].y := i - 0.9; end;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i].x := i + 1.3; h_y[i].y := i - 0.7;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation: y := alpha*A*x + beta*y
  for i := 0 to n - 1 do
  begin
    ComplexMul(h_y_expected[i * incy], beta);
    for j := 0 to n - 1 do
    begin
       temp := h_A[i + j * lda];
       ComplexMul(temp, h_x[j * incx]);
       ComplexMul(temp, alpha);
       ComplexAdd(h_y_expected[i * incy], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zsymv_64(handle, uplo, n, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_zsymv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_y_expected[i*incy].x, h_y[i*incy].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_y_expected[i*incy].y, h_y[i*incy].y, 1e-12, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_SSBMV_64;
const
  n: Int64 = 7;
  k: Int64 = 2;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Single = 1.5;
  beta: Single = 2.0;
var
  lda: Int64;
  uplo: rocblas_fill;
  h_A_banded: array of Single;
  h_x, h_y, h_y_expected: array of Single;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  lda := k + 1;
  uplo := rocblas_fill_upper;

  SetLength(h_A_banded, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do h_A_banded[i] := (i mod 10) + 1.0;
  for i := 0 to (n * incx) - 1 do h_x[i] := (i mod 5) + 1.0;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i] := (i mod 3) + 1.0;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation
  for i := 0 to n - 1 do
  begin
    h_y_expected[i*incy] := beta * h_y_expected[i*incy];
    // Upper part
    for j := i to Min(n - 1, i + k) do
      h_y_expected[i*incy] := h_y_expected[i*incy] + alpha * h_A_banded[(k + i - j) + j*lda] * h_x[j*incx];
    // Lower (symmetric) part, excluding diagonal
    for j := Max(0, i - k) to i - 1 do
      h_y_expected[i*incy] := h_y_expected[i*incy] + alpha * h_A_banded[(k + j - i) + i*lda] * h_x[j*incx];
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Single));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(Single));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A_banded to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_ssbmv_64(handle, uplo, n, k, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_ssbmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_y_expected[i*incy], h_y[i*incy], 1e-4, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_DSBMV_64;
const
  n: Int64 = 7;
  k: Int64 = 2;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Double = 1.5;
  beta: Double = 2.0;
var
  lda: Int64;
  uplo: rocblas_fill;
  h_A_banded: array of Double;
  h_x, h_y, h_y_expected: array of Double;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  lda := k + 1;
  uplo := rocblas_fill_upper;

  SetLength(h_A_banded, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do h_A_banded[i] := (i mod 10) + 1.0;
  for i := 0 to (n * incx) - 1 do h_x[i] := (i mod 5) + 1.0;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i] := (i mod 3) + 1.0;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation
  for i := 0 to n - 1 do
  begin
    h_y_expected[i*incy] := beta * h_y_expected[i*incy];
    // Upper part
    for j := i to Min(n - 1, i + k) do
      h_y_expected[i*incy] := h_y_expected[i*incy] + alpha * h_A_banded[(k + i - j) + j*lda] * h_x[j*incx];
    // Lower (symmetric) part, excluding diagonal
    for j := Max(0, i - k) to i - 1 do
      h_y_expected[i*incy] := h_y_expected[i*incy] + alpha * h_A_banded[(k + j - i) + i*lda] * h_x[j*incx];
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Double));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(Double));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A_banded to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dsbmv_64(handle, uplo, n, k, @alpha, d_A, lda, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_dsbmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_y_expected[i*incy], h_y[i*incy], 1e-12, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_SSPMV_64;
const
  n: Int64 = 6;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Single = 1.5;
  beta: Single = 2.0;
var
  uplo: rocblas_fill;
  h_A_full: array of Single;
  h_AP: array of Single;
  h_x, h_y, h_y_expected: array of Single;
  d_AP, d_x, d_y: hipDeviceptr_t;
  i, j, pack_idx: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  uplo := rocblas_fill_upper;
  ap_size := n * (n + 1) div 2;

  SetLength(h_A_full, n * n);
  SetLength(h_AP, ap_size);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize a full symmetric matrix A and pack it
  for i := 0 to n - 1 do
    for j := i to n - 1 do
    begin
      h_A_full[i + j*n] := i + j*2.1;
      h_A_full[j + i*n] := h_A_full[i + j*n];
    end;

  // Pack the matrix
  pack_idx := 0;
  if uplo = rocblas_fill_upper then
  begin
    for j := 0 to n - 1 do
      for i := 0 to j do
      begin
        h_AP[pack_idx] := h_A_full[i + j*n];
        Inc(pack_idx);
      end;
  end
  else // lower
  begin
    for j := 0 to n - 1 do
      for i := j to n - 1 do
      begin
        h_AP[pack_idx] := h_A_full[i + j*n];
        Inc(pack_idx);
      end;
  end;

  for i := 0 to (n * incx) - 1 do h_x[i] := i+1.3;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i] := i+1.5;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation using full matrix
  for i := 0 to n - 1 do
  begin
    h_y_expected[i*incy] := beta * h_y_expected[i*incy];
    for j := 0 to n - 1 do
    begin
      h_y_expected[i*incy] := h_y_expected[i*incy] + alpha * h_A_full[i + j*n] * h_x[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(Single));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(Single));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sspmv_64(handle, uplo, n, @alpha, d_AP, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_sspmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_y_expected[i*incy], h_y[i*incy], 1e-4, Format('Mismatch at index %d', [i]));

  hipFree(d_AP);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_DSPMV_64;
const
  n: Int64 = 6;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Double = 1.5;
  beta: Double = 2.0;
var
  uplo: rocblas_fill;
  h_A_full: array of Double;
  h_AP: array of Double;
  h_x, h_y, h_y_expected: array of Double;
  d_AP, d_x, d_y: hipDeviceptr_t;
  i, j, pack_idx: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  uplo := rocblas_fill_upper;
  ap_size := n * (n + 1) div 2;

  SetLength(h_A_full, n * n);
  SetLength(h_AP, ap_size);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);
  SetLength(h_y_expected, n * incy);

  // Initialize a full symmetric matrix A and pack it
  for i := 0 to n - 1 do
    for j := i to n - 1 do
    begin
      h_A_full[i + j*n] := i + j*2.1;
      h_A_full[j + i*n] := h_A_full[i + j*n];
    end;

  // Pack the matrix
  pack_idx := 0;
  if uplo = rocblas_fill_upper then
  begin
    for j := 0 to n - 1 do
      for i := 0 to j do
      begin
        h_AP[pack_idx] := h_A_full[i + j*n];
        Inc(pack_idx);
      end;
  end
  else // lower
  begin
    for j := 0 to n - 1 do
      for i := j to n - 1 do
      begin
        h_AP[pack_idx] := h_A_full[i + j*n];
        Inc(pack_idx);
      end;
  end;

  for i := 0 to (n * incx) - 1 do h_x[i] := i+1.3;
  for i := 0 to (n * incy) - 1 do
  begin
    h_y[i] := i+1.5;
    h_y_expected[i] := h_y[i];
  end;

  // CPU calculation using full matrix
  for i := 0 to n - 1 do
  begin
    h_y_expected[i*incy] := beta * h_y_expected[i*incy];
    for j := 0 to n - 1 do
    begin
      h_y_expected[i*incy] := h_y_expected[i*incy] + alpha * h_A_full[i + j*n] * h_x[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(Double));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(Double));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dspmv_64(handle, uplo, n, @alpha, d_AP, d_x, incx, @beta, d_y, incy);
  AssertEquals('rocblas_dspmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_y[0], d_y, n * incy * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_y to h_y', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_y_expected[i*incy], h_y[i*incy], 1e-12, Format('Mismatch at index %d', [i]));

  hipFree(d_AP);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_STRMV_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
var
  h_A: array of Single;
  h_x, h_x_expected: array of Single;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_x_expected, n * incx);

  // Initialize data
  for i := 0 to (lda * n) - 1 do h_A[i] := (i mod 10) + 1.2;
  for i := 0 to (n * incx) - 1 do
  begin
    h_x[i] := (i mod 5) + 1.1;
    h_x_expected[i] := 0;
  end;

  // CPU calculation: x := A*x
  for i := 0 to n - 1 do
  begin
    for j := 0 to n - 1 do
    begin
      if (uplo = rocblas_fill_upper) and (j >= i) then
        h_x_expected[i*incx] := h_x_expected[i*incx] + h_A[i + j*lda] * h_x[j*incx]
      else if (uplo = rocblas_fill_lower) and (j <= i) then
        h_x_expected[i*incx] := h_x_expected[i*incx] + h_A[i + j*lda] * h_x[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Single));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_strmv_64(handle, uplo, transA, diag, n, d_A, lda, d_x, incx);
  AssertEquals('rocblas_strmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x[0], d_x, n * incx * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_x to h_x', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_x_expected[i*incx], h_x[i*incx], 1e-5, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_DTRMV_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
var
  h_A: array of Double;
  h_x, h_x_expected: array of Double;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_x_expected, n * incx);

  // Initialize data
  for i := 0 to (lda * n) - 1 do h_A[i] := (i mod 10) + 1.2;
  for i := 0 to (n * incx) - 1 do
  begin
    h_x[i] := (i mod 5) + 1.1;
    h_x_expected[i] := 0;
  end;

  // CPU calculation: x := A*x
  for i := 0 to n - 1 do
  begin
    for j := 0 to n - 1 do
    begin
      if (uplo = rocblas_fill_upper) and (j >= i) then
        h_x_expected[i*incx] := h_x_expected[i*incx] + h_A[i + j*lda] * h_x[j*incx]
      else if (uplo = rocblas_fill_lower) and (j <= i) then
        h_x_expected[i*incx] := h_x_expected[i*incx] + h_A[i + j*lda] * h_x[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Double));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_dtrmv_64(handle, uplo, transA, diag, n, d_A, lda, d_x, incx);
  AssertEquals('rocblas_dtrmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x[0], d_x, n * incx * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_x to h_x', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_x_expected[i*incx], h_x[i*incx], 1e-12, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_CTRMV_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
var
  h_A: array of rocblas_float_complex;
  h_x, h_x_expected: array of rocblas_float_complex;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  temp: rocblas_float_complex;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_x_expected, n * incx);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i].x := i + 1.2; h_A[i].y := i - 0.8; end;
  for i := 0 to (n * incx) - 1 do
  begin
    h_x[i].x := i + 1.1; h_x[i].y := i - 0.9;
    h_x_expected[i].x := 0;
    h_x_expected[i].y := 0;
  end;

  // CPU calculation: x := A*x
  for i := 0 to n - 1 do
  begin
    for j := 0 to n - 1 do
    begin
      if (uplo = rocblas_fill_upper) and (j >= i) then
      begin
        temp := h_A[i + j*lda];
        ComplexMul(temp, h_x[j*incx]);
        ComplexAdd(h_x_expected[i*incx], temp);
      end
      else if (uplo = rocblas_fill_lower) and (j <= i) then
      begin
        temp := h_A[i + j*lda];
        ComplexMul(temp, h_x[j*incx]);
        ComplexAdd(h_x_expected[i*incx], temp);
      end;
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ctrmv_64(handle, uplo, transA, diag, n, d_A, lda, d_x, incx);
  AssertEquals('rocblas_ctrmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x[0], d_x, n * incx * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_x to h_x', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_x_expected[i*incx].x, h_x[i*incx].x, 1e-4, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_x_expected[i*incx].y, h_x[i*incx].y, 1e-4, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_ZTRMV_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
var
  h_A: array of rocblas_double_complex;
  h_x, h_x_expected: array of rocblas_double_complex;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  temp: rocblas_double_complex;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_x_expected, n * incx);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i].x := i + 1.2; h_A[i].y := i - 0.8; end;
  for i := 0 to (n * incx) - 1 do
  begin
    h_x[i].x := i + 1.1; h_x[i].y := i - 0.9;
    h_x_expected[i].x := 0;
    h_x_expected[i].y := 0;
  end;

  // CPU calculation: x := A*x
  for i := 0 to n - 1 do
  begin
    for j := 0 to n - 1 do
    begin
      if (uplo = rocblas_fill_upper) and (j >= i) then
      begin
        temp := h_A[i + j*lda];
        ComplexMul(temp, h_x[j*incx]);
        ComplexAdd(h_x_expected[i*incx], temp);
      end
      else if (uplo = rocblas_fill_lower) and (j <= i) then
      begin
        temp := h_A[i + j*lda];
        ComplexMul(temp, h_x[j*incx]);
        ComplexAdd(h_x_expected[i*incx], temp);
      end;
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ztrmv_64(handle, uplo, transA, diag, n, d_A, lda, d_x, incx);
  AssertEquals('rocblas_ztrmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x[0], d_x, n * incx * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_x to h_x', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_x_expected[i*incx].x, h_x[i*incx].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_x_expected[i*incx].y, h_x[i*incx].y, 1e-12, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_STBMV_64;
const
  n: Int64 = 8;
  k: Int64 = 3;
  incx: Int64 = 1;
var
  lda: Int64;
  h_A_banded: array of Single;
  h_x, h_x_expected: array of Single;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
begin
  lda := k + 1;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A_banded, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_x_expected, n * incx);

  // Initialize data
  for i := 0 to (lda * n) - 1 do h_A_banded[i] := (i mod 10) + 1.2;
  for i := 0 to (n * incx) - 1 do
  begin
    h_x[i] := (i mod 5) + 1.1;
    h_x_expected[i] := 0;
  end;

  // CPU calculation: x := A*x
  for i := 0 to n - 1 do
  begin
    for j := i to Min(n - 1, i + k) do
    begin
      // A_orig[i,j] is at A_banded[k + i - j, j]
      h_x_expected[i*incx] := h_x_expected[i*incx] + h_A_banded[(k + i - j) + j*lda] * h_x[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Single));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A_banded to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_stbmv_64(handle, uplo, transA, diag, n, k, d_A, lda, d_x, incx);
  AssertEquals('rocblas_stbmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x[0], d_x, n * incx * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_x to h_x', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_x_expected[i*incx], h_x[i*incx], 1e-5, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_DTBMV_64;
const
  n: Int64 = 8;
  k: Int64 = 3;
  incx: Int64 = 1;
var
  lda: Int64;
  h_A_banded: array of Double;
  h_x, h_x_expected: array of Double;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
begin
  lda := k + 1;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A_banded, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_x_expected, n * incx);

  // Initialize data
  for i := 0 to (lda * n) - 1 do h_A_banded[i] := (i mod 10) + 1.2;
  for i := 0 to (n * incx) - 1 do
  begin
    h_x[i] := (i mod 5) + 1.1;
    h_x_expected[i] := 0;
  end;

  // CPU calculation: x := A*x
  for i := 0 to n - 1 do
  begin
    for j := i to Min(n - 1, i + k) do
    begin
      // A_orig[i,j] is at A_banded[k + i - j, j]
      h_x_expected[i*incx] := h_x_expected[i*incx] + h_A_banded[(k + i - j) + j*lda] * h_x[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Double));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A_banded to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_dtbmv_64(handle, uplo, transA, diag, n, k, d_A, lda, d_x, incx);
  AssertEquals('rocblas_dtbmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x[0], d_x, n * incx * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_x to h_x', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_x_expected[i*incx], h_x[i*incx], 1e-12, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_CTBMV_64;
const
  n: Int64 = 8;
  k: Int64 = 3;
  incx: Int64 = 1;
var
  lda: Int64;
  h_A_banded: array of rocblas_float_complex;
  h_x, h_x_expected: array of rocblas_float_complex;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  temp: rocblas_float_complex;
begin
  lda := k + 1;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A_banded, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_x_expected, n * incx);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A_banded[i].x := i + 1.2; h_A_banded[i].y := i - 0.8; end;
  for i := 0 to (n * incx) - 1 do
  begin
    h_x[i].x := i + 1.1; h_x[i].y := i - 0.9;
    h_x_expected[i].x := 0;
    h_x_expected[i].y := 0;
  end;

  // CPU calculation: x := A*x
  for i := 0 to n - 1 do
  begin
    for j := i to Min(n - 1, i + k) do
    begin
      // A_orig[i,j] is at A_banded[k + i - j, j]
      temp := h_A_banded[(k + i - j) + j*lda];
      ComplexMul(temp, h_x[j*incx]);
      ComplexAdd(h_x_expected[i*incx], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A_banded to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ctbmv_64(handle, uplo, transA, diag, n, k, d_A, lda, d_x, incx);
  AssertEquals('rocblas_ctbmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x[0], d_x, n * incx * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_x to h_x', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_x_expected[i*incx].x, h_x[i*incx].x, 1e-4, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_x_expected[i*incx].y, h_x[i*incx].y, 1e-4, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_ZTBMV_64;
const
  n: Int64 = 8;
  k: Int64 = 3;
  incx: Int64 = 1;
var
  lda: Int64;
  h_A_banded: array of rocblas_double_complex;
  h_x, h_x_expected: array of rocblas_double_complex;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  temp: rocblas_double_complex;
begin
  lda := k + 1;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A_banded, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_x_expected, n * incx);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A_banded[i].x := i + 1.2; h_A_banded[i].y := i - 0.8; end;
  for i := 0 to (n * incx) - 1 do
  begin
    h_x[i].x := i + 1.1; h_x[i].y := i - 0.9;
    h_x_expected[i].x := 0;
    h_x_expected[i].y := 0;
  end;

  // CPU calculation: x := A*x
  for i := 0 to n - 1 do
  begin
    for j := i to Min(n - 1, i + k) do
    begin
      // A_orig[i,j] is at A_banded[k + i - j, j]
      temp := h_A_banded[(k + i - j) + j*lda];
      ComplexMul(temp, h_x[j*incx]);
      ComplexAdd(h_x_expected[i*incx], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A_banded to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ztbmv_64(handle, uplo, transA, diag, n, k, d_A, lda, d_x, incx);
  AssertEquals('rocblas_ztbmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x[0], d_x, n * incx * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_x to h_x', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_x_expected[i*incx].x, h_x[i*incx].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_x_expected[i*incx].y, h_x[i*incx].y, 1e-12, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_STPMV_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
var
  h_A_full: array of Single;
  h_AP: array of Single;
  h_x, h_x_expected: array of Single;
  d_AP, d_x: hipDeviceptr_t;
  i, j, pack_idx: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;
  ap_size := n * (n + 1) div 2;

  SetLength(h_A_full, n * n);
  SetLength(h_AP, ap_size);
  SetLength(h_x, n * incx);
  SetLength(h_x_expected, n * incx);

  // Initialize a full matrix and pack the triangular part
  for i := 0 to (n * n) - 1 do h_A_full[i] := (i mod 10) + 1.2;
  for i := 0 to (n * incx) - 1 do
  begin
    h_x[i] := (i mod 5) + 1.1;
    h_x_expected[i] := 0;
  end;

  // Pack the matrix
  pack_idx := 0;
  if uplo = rocblas_fill_upper then
  begin
    for j := 0 to n - 1 do
      for i := 0 to j do
      begin
        h_AP[pack_idx] := h_A_full[i + j*n];
        Inc(pack_idx);
      end;
  end
  else // lower
  begin
    for j := 0 to n - 1 do
      for i := j to n - 1 do
      begin
        h_AP[pack_idx] := h_A_full[i + j*n];
        Inc(pack_idx);
      end;
  end;


  // CPU calculation using full matrix
  for i := 0 to n - 1 do
  begin
    for j := 0 to n - 1 do
    begin
      if (uplo = rocblas_fill_upper) and (j >= i) then
        h_x_expected[i*incx] := h_x_expected[i*incx] + h_A_full[i + j*n] * h_x[j*incx]
      else if (uplo = rocblas_fill_lower) and (j <= i) then
        h_x_expected[i*incx] := h_x_expected[i*incx] + h_A_full[i + j*n] * h_x[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(Single));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_stpmv_64(handle, uplo, transA, diag, n, d_AP, d_x, incx);
  AssertEquals('rocblas_stpmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x[0], d_x, n * incx * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_x to h_x', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_x_expected[i*incx], h_x[i*incx], 1e-5, Format('Mismatch at index %d', [i]));

  hipFree(d_AP);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_DTPMV_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
var
  h_A_full: array of Double;
  h_AP: array of Double;
  h_x, h_x_expected: array of Double;
  d_AP, d_x: hipDeviceptr_t;
  i, j, pack_idx: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;
  ap_size := n * (n + 1) div 2;

  SetLength(h_A_full, n * n);
  SetLength(h_AP, ap_size);
  SetLength(h_x, n * incx);
  SetLength(h_x_expected, n * incx);

  // Initialize data
  for i := 0 to (n * n) - 1 do h_A_full[i] := (i mod 10) + 1.2;
  for i := 0 to (n * incx) - 1 do
  begin
    h_x[i] := (i mod 5) + 1.1;
    h_x_expected[i] := 0;
  end;

  // Pack the matrix
  pack_idx := 0;
  if uplo = rocblas_fill_upper then
  begin
    for j := 0 to n - 1 do
      for i := 0 to j do
      begin
        h_AP[pack_idx] := h_A_full[i + j*n];
        Inc(pack_idx);
      end;
  end
  else // lower
  begin
    for j := 0 to n - 1 do
      for i := j to n - 1 do
      begin
        h_AP[pack_idx] := h_A_full[i + j*n];
        Inc(pack_idx);
      end;
  end;


  // CPU calculation using full matrix
  for i := 0 to n - 1 do
  begin
    for j := 0 to n - 1 do
    begin
      if (uplo = rocblas_fill_upper) and (j >= i) then
        h_x_expected[i*incx] := h_x_expected[i*incx] + h_A_full[i + j*n] * h_x[j*incx]
      else if (uplo = rocblas_fill_lower) and (j <= i) then
        h_x_expected[i*incx] := h_x_expected[i*incx] + h_A_full[i + j*n] * h_x[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(Double));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_dtpmv_64(handle, uplo, transA, diag, n, d_AP, d_x, incx);
  AssertEquals('rocblas_dtpmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x[0], d_x, n * incx * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_x to h_x', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_x_expected[i*incx], h_x[i*incx], 1e-12, Format('Mismatch at index %d', [i]));

  hipFree(d_AP);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_CTPMV_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
var
  h_A_full: array of rocblas_float_complex;
  h_AP: array of rocblas_float_complex;
  h_x, h_x_expected: array of rocblas_float_complex;
  d_AP, d_x: hipDeviceptr_t;
  i, j, pack_idx: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  temp: rocblas_float_complex;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;
  ap_size := n * (n + 1) div 2;

  SetLength(h_A_full, n * n);
  SetLength(h_AP, ap_size);
  SetLength(h_x, n * incx);
  SetLength(h_x_expected, n * incx);

  // Initialize data
  for i := 0 to (n * n) - 1 do begin h_A_full[i].x := i + 1.2; h_A_full[i].y := i - 0.8; end;
  for i := 0 to (n * incx) - 1 do
  begin
    h_x[i].x := i + 1.1; h_x[i].y := i - 0.9;
    h_x_expected[i].x := 0; h_x_expected[i].y := 0;
  end;

  // Pack the matrix
  pack_idx := 0;
  if uplo = rocblas_fill_upper then
  begin
    for j := 0 to n - 1 do
      for i := 0 to j do
      begin
        h_AP[pack_idx] := h_A_full[i + j*n];
        Inc(pack_idx);
      end;
  end
  else // lower
  begin
    for j := 0 to n - 1 do
      for i := j to n - 1 do
      begin
        h_AP[pack_idx] := h_A_full[i + j*n];
        Inc(pack_idx);
      end;
  end;


  // CPU calculation using full matrix
  for i := 0 to n - 1 do
  begin
    for j := 0 to n - 1 do
    begin
      if (uplo = rocblas_fill_upper) and (j >= i) then
      begin
        temp := h_A_full[i + j*n];
        ComplexMul(temp, h_x[j*incx]);
        ComplexAdd(h_x_expected[i*incx], temp);
      end
      else if (uplo = rocblas_fill_lower) and (j <= i) then
      begin
        temp := h_A_full[i + j*n];
        ComplexMul(temp, h_x[j*incx]);
        ComplexAdd(h_x_expected[i*incx], temp);
      end;
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ctpmv_64(handle, uplo, transA, diag, n, d_AP, d_x, incx);
  AssertEquals('rocblas_ctpmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x[0], d_x, n * incx * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_x to h_x', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_x_expected[i*incx].x, h_x[i*incx].x, 1e-4, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_x_expected[i*incx].y, h_x[i*incx].y, 1e-4, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_AP);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_ZTPMV_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
var
  h_A_full: array of rocblas_double_complex;
  h_AP: array of rocblas_double_complex;
  h_x, h_x_expected: array of rocblas_double_complex;
  d_AP, d_x: hipDeviceptr_t;
  i, j, pack_idx: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  temp: rocblas_double_complex;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;
  ap_size := n * (n + 1) div 2;

  SetLength(h_A_full, n * n);
  SetLength(h_AP, ap_size);
  SetLength(h_x, n * incx);
  SetLength(h_x_expected, n * incx);

  // Initialize data
  for i := 0 to (n * n) - 1 do begin h_A_full[i].x := i + 1.2; h_A_full[i].y := i - 0.8; end;
  for i := 0 to (n * incx) - 1 do
  begin
    h_x[i].x := i + 1.1; h_x[i].y := i - 0.9;
    h_x_expected[i].x := 0; h_x_expected[i].y := 0;
  end;

  // Pack the matrix
  pack_idx := 0;
  if uplo = rocblas_fill_upper then
  begin
    for j := 0 to n - 1 do
      for i := 0 to j do
      begin
        h_AP[pack_idx] := h_A_full[i + j*n];
        Inc(pack_idx);
      end;
  end
  else // lower
  begin
    for j := 0 to n - 1 do
      for i := j to n - 1 do
      begin
        h_AP[pack_idx] := h_A_full[i + j*n];
        Inc(pack_idx);
      end;
  end;


  // CPU calculation using full matrix
  for i := 0 to n - 1 do
  begin
    for j := 0 to n - 1 do
    begin
      if (uplo = rocblas_fill_upper) and (j >= i) then
      begin
        temp := h_A_full[i + j*n];
        ComplexMul(temp, h_x[j*incx]);
        ComplexAdd(h_x_expected[i*incx], temp);
      end
      else if (uplo = rocblas_fill_lower) and (j <= i) then
      begin
        temp := h_A_full[i + j*n];
        ComplexMul(temp, h_x[j*incx]);
        ComplexAdd(h_x_expected[i*incx], temp);
      end;
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ztpmv_64(handle, uplo, transA, diag, n, d_AP, d_x, incx);
  AssertEquals('rocblas_ztpmv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x[0], d_x, n * incx * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_x to h_x', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_x_expected[i*incx].x, h_x[i*incx].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_x_expected[i*incx].y, h_x[i*incx].y, 1e-12, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_AP);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_STRSV_64;
const
  n: Int64 = 8;
  lda: Int64 = 8;
  incx: Int64 = 1;
var
  h_A: array of Single;
  h_x_expected, h_b, h_x_gpu_result: array of Single;
  d_A, d_b: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A, lda * n);
  SetLength(h_x_expected, n * incx);
  SetLength(h_b, n * incx);
  SetLength(h_x_gpu_result, n * incx);

  // Initialize a well-conditioned upper triangular matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
      h_A[i + j * lda] := (i * 1.1) + (j * 0.9) + 1.0;
  for i := 0 to n-1 do h_A[i + i*lda] := n * 2.0; // Diagonally dominant

  // Initialize known solution vector x
  for i := 0 to (n * incx) - 1 do h_x_expected[i] := (i mod 5) + 1.1;

  // Calculate b = A*x on CPU
  for i := 0 to n - 1 do
  begin
    h_b[i*incx] := 0;
    for j := i to n - 1 do
    begin
        h_b[i*incx] := h_b[i*incx] + h_A[i + j*lda] * h_x_expected[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Single));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_b, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_b, @h_b[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_b to d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_strsv_64(handle, uplo, transA, diag, n, d_A, lda, d_b, incx);
  AssertEquals('rocblas_strsv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x_gpu_result[0], d_b, n * incx * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_b to h_x_gpu_result', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_x_expected[i*incx], h_x_gpu_result[i*incx], 1e-5, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_b);
end;

procedure TTestRocBLAS_Level2_64.Test_DTRSV_64;
const
  n: Int64 = 8;
  lda: Int64 = 8;
  incx: Int64 = 1;
var
  h_A: array of Double;
  h_x_expected, h_b, h_x_gpu_result: array of Double;
  d_A, d_b: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A, lda * n);
  SetLength(h_x_expected, n * incx);
  SetLength(h_b, n * incx);
  SetLength(h_x_gpu_result, n * incx);

  // Initialize a well-conditioned upper triangular matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
      h_A[i + j * lda] := (i * 1.1) + (j * 0.9) + 1.0;
  for i := 0 to n-1 do h_A[i + i*lda] := n * 2.0; // Diagonally dominant

  // Initialize known solution vector x
  for i := 0 to (n * incx) - 1 do h_x_expected[i] := (i mod 5) + 1.1;

  // Calculate b = A*x on CPU
  for i := 0 to n - 1 do
  begin
    h_b[i*incx] := 0;
    for j := i to n - 1 do
    begin
        h_b[i*incx] := h_b[i*incx] + h_A[i + j*lda] * h_x_expected[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Double));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_b, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_b, @h_b[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_b to d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_dtrsv_64(handle, uplo, transA, diag, n, d_A, lda, d_b, incx);
  AssertEquals('rocblas_dtrsv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x_gpu_result[0], d_b, n * incx * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_b to h_x_gpu_result', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_x_expected[i*incx], h_x_gpu_result[i*incx], 1e-12, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_b);
end;

procedure TTestRocBLAS_Level2_64.Test_CTRSV_64;
const
  n: Int64 = 8;
  lda: Int64 = 8;
  incx: Int64 = 1;
var
  h_A: array of rocblas_float_complex;
  h_x_expected, h_b, h_x_gpu_result: array of rocblas_float_complex;
  d_A, d_b: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  temp: rocblas_float_complex;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A, lda * n);
  SetLength(h_x_expected, n * incx);
  SetLength(h_b, n * incx);
  SetLength(h_x_gpu_result, n * incx);

  // Initialize a well-conditioned upper triangular matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
    begin
      h_A[i + j * lda].x := (i * 1.1) + (j * 0.9) + 1.0;
      h_A[i + j * lda].y := (i * 0.5) - (j * 0.2);
    end;
  for i := 0 to n-1 do begin h_A[i + i*lda].x := n * 2.0; h_A[i + i*lda].y := 0; end; // Diagonally dominant

  // Initialize known solution vector x
  for i := 0 to (n * incx) - 1 do begin h_x_expected[i].x := (i mod 5) + 1.1; h_x_expected[i].y := (i mod 3) - 0.5; end;

  // Calculate b = A*x on CPU
  for i := 0 to n - 1 do
  begin
    h_b[i*incx].x := 0;
    h_b[i*incx].y := 0;
    for j := i to n - 1 do
    begin
        temp := h_A[i + j*lda];
        ComplexMul(temp, h_x_expected[j*incx]);
        ComplexAdd(h_b[i*incx], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_b, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_b, @h_b[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_b to d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ctrsv_64(handle, uplo, transA, diag, n, d_A, lda, d_b, incx);
  AssertEquals('rocblas_ctrsv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x_gpu_result[0], d_b, n * incx * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_b to h_x_gpu_result', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_x_expected[i*incx].x, h_x_gpu_result[i*incx].x, 1e-4, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_x_expected[i*incx].y, h_x_gpu_result[i*incx].y, 1e-4, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_b);
end;

procedure TTestRocBLAS_Level2_64.Test_ZTRSV_64;
const
  n: Int64 = 8;
  lda: Int64 = 8;
  incx: Int64 = 1;
var
  h_A: array of rocblas_double_complex;
  h_x_expected, h_b, h_x_gpu_result: array of rocblas_double_complex;
  d_A, d_b: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  temp: rocblas_double_complex;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A, lda * n);
  SetLength(h_x_expected, n * incx);
  SetLength(h_b, n * incx);
  SetLength(h_x_gpu_result, n * incx);

  // Initialize a well-conditioned upper triangular matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
    begin
      h_A[i + j * lda].x := (i * 1.1) + (j * 0.9) + 1.0;
      h_A[i + j * lda].y := (i * 0.5) - (j * 0.2);
    end;
  for i := 0 to n-1 do begin h_A[i + i*lda].x := n * 2.0; h_A[i + i*lda].y := 0; end; // Diagonally dominant

  // Initialize known solution vector x
  for i := 0 to (n * incx) - 1 do begin h_x_expected[i].x := (i mod 5) + 1.1; h_x_expected[i].y := (i mod 3) - 0.5; end;

  // Calculate b = A*x on CPU
  for i := 0 to n - 1 do
  begin
    h_b[i*incx].x := 0;
    h_b[i*incx].y := 0;
    for j := i to n - 1 do
    begin
        temp := h_A[i + j*lda];
        ComplexMul(temp, h_x_expected[j*incx]);
        ComplexAdd(h_b[i*incx], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_b, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_b, @h_b[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_b to d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ztrsv_64(handle, uplo, transA, diag, n, d_A, lda, d_b, incx);
  AssertEquals('rocblas_ztrsv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x_gpu_result[0], d_b, n * incx * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_b to h_x_gpu_result', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_x_expected[i*incx].x, h_x_gpu_result[i*incx].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_x_expected[i*incx].y, h_x_gpu_result[i*incx].y, 1e-12, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_b);
end;

procedure TTestRocBLAS_Level2_64.Test_STBSV_64;
const
  n: Int64 = 8;
  k: Int64 = 3;
  incx: Int64 = 1;
var
  lda: Int64;
  h_A_full, h_A_banded: array of Single;
  h_x_expected, h_b, h_x_gpu_result: array of Single;
  d_A, d_b: hipDeviceptr_t;
  i, j, pack_idx_row, pack_idx_col: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
begin
  lda := k + 1;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A_full, n * n);
  SetLength(h_A_banded, lda * n);
  SetLength(h_x_expected, n * incx);
  SetLength(h_b, n * incx);
  SetLength(h_x_gpu_result, n * incx);

  // Initialize a well-conditioned upper triangular matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
      h_A_full[i + j * n] := (i * 1.1) + (j * 0.9) + 1.0;
  for i := 0 to n-1 do h_A_full[i + i*n] := n * 2.0; // Diagonally dominant

  // Pack the matrix into banded format
  for j := 0 to n - 1 do
    for i := Max(0, j - k) to j do
    begin
      pack_idx_row := k + i - j;
      pack_idx_col := j;
      h_A_banded[pack_idx_row + pack_idx_col * lda] := h_A_full[i + j * n];
    end;

  // Initialize known solution vector x
  for i := 0 to (n * incx) - 1 do h_x_expected[i] := (i mod 5) + 1.1;

  // Calculate b = A*x on CPU using the full matrix
  for i := 0 to n - 1 do
  begin
    h_b[i*incx] := 0;
    for j := i to Min(n - 1, i + k) do
    begin
        h_b[i*incx] := h_b[i*incx] + h_A_full[i + j*n] * h_x_expected[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Single));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_b, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A_banded to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_b, @h_b[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_b to d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_stbsv_64(handle, uplo, transA, diag, n, k, d_A, lda, d_b, incx);
  AssertEquals('rocblas_stbsv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x_gpu_result[0], d_b, n * incx * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_b to h_x_gpu_result', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_x_expected[i*incx], h_x_gpu_result[i*incx], 1e-4, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_b);
end;

procedure TTestRocBLAS_Level2_64.Test_DTBSV_64;
const
  n: Int64 = 8;
  k: Int64 = 3;
  incx: Int64 = 1;
var
  lda: Int64;
  h_A_full, h_A_banded: array of Double;
  h_x_expected, h_b, h_x_gpu_result: array of Double;
  d_A, d_b: hipDeviceptr_t;
  i, j, pack_idx_row, pack_idx_col: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
begin
  lda := k + 1;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A_full, n * n);
  SetLength(h_A_banded, lda * n);
  SetLength(h_x_expected, n * incx);
  SetLength(h_b, n * incx);
  SetLength(h_x_gpu_result, n * incx);

  // Initialize a well-conditioned upper triangular matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
      h_A_full[i + j * n] := (i * 1.1) + (j * 0.9) + 1.0;
  for i := 0 to n-1 do h_A_full[i + i*n] := n * 2.0; // Diagonally dominant

  // Pack the matrix into banded format
  for j := 0 to n - 1 do
    for i := Max(0, j - k) to j do
    begin
      pack_idx_row := k + i - j;
      pack_idx_col := j;
      h_A_banded[pack_idx_row + pack_idx_col * lda] := h_A_full[i + j * n];
    end;

  // Initialize known solution vector x
  for i := 0 to (n * incx) - 1 do h_x_expected[i] := (i mod 5) + 1.1;

  // Calculate b = A*x on CPU using the full matrix
  for i := 0 to n - 1 do
  begin
    h_b[i*incx] := 0;
    for j := i to Min(n - 1, i + k) do
    begin
        h_b[i*incx] := h_b[i*incx] + h_A_full[i + j*n] * h_x_expected[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Double));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_b, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A_banded to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_b, @h_b[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_b to d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_dtbsv_64(handle, uplo, transA, diag, n, k, d_A, lda, d_b, incx);
  AssertEquals('rocblas_dtbsv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x_gpu_result[0], d_b, n * incx * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_b to h_x_gpu_result', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_x_expected[i*incx], h_x_gpu_result[i*incx], 1e-12, Format('Mismatch at index %d', [i]));

  hipFree(d_A);
  hipFree(d_b);
end;

procedure TTestRocBLAS_Level2_64.Test_CTBSV_64;
const
  n: Int64 = 8;
  k: Int64 = 3;
  incx: Int64 = 1;
var
  lda: Int64;
  h_A_full, h_A_banded: array of rocblas_float_complex;
  h_x_expected, h_b, h_x_gpu_result: array of rocblas_float_complex;
  d_A, d_b: hipDeviceptr_t;
  i, j, pack_idx_row, pack_idx_col: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  temp: rocblas_float_complex;
begin
  lda := k + 1;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A_full, n * n);
  SetLength(h_A_banded, lda * n);
  SetLength(h_x_expected, n * incx);
  SetLength(h_b, n * incx);
  SetLength(h_x_gpu_result, n * incx);

  // Initialize a well-conditioned upper triangular matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
    begin
      h_A_full[i + j * n].x := (i * 1.1) + (j * 0.9) + 1.0;
      h_A_full[i + j * n].y := (i * 0.5) - (j * 0.2);
    end;
  for i := 0 to n-1 do begin h_A_full[i + i*n].x := n * 2.0; h_A_full[i + i*n].y := 0; end;

  // Pack the matrix into banded format
  for j := 0 to n - 1 do
    for i := Max(0, j - k) to j do
    begin
      pack_idx_row := k + i - j;
      pack_idx_col := j;
      h_A_banded[pack_idx_row + pack_idx_col * lda] := h_A_full[i + j * n];
    end;

  // Initialize known solution vector x
  for i := 0 to (n * incx) - 1 do begin h_x_expected[i].x := (i mod 5)+1.1; h_x_expected[i].y := (i mod 3)-0.5; end;

  // Calculate b = A*x on CPU using the full matrix
  for i := 0 to n - 1 do
  begin
    h_b[i*incx].x := 0; h_b[i*incx].y := 0;
    for j := i to Min(n - 1, i + k) do
    begin
        temp := h_A_full[i + j*n];
        ComplexMul(temp, h_x_expected[j*incx]);
        ComplexAdd(h_b[i*incx], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_b, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A_banded to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_b, @h_b[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_b to d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ctbsv_64(handle, uplo, transA, diag, n, k, d_A, lda, d_b, incx);
  AssertEquals('rocblas_ctbsv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x_gpu_result[0], d_b, n * incx * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_b to h_x_gpu_result', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_x_expected[i*incx].x, h_x_gpu_result[i*incx].x, 1e-4, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_x_expected[i*incx].y, h_x_gpu_result[i*incx].y, 1e-4, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_b);
end;

procedure TTestRocBLAS_Level2_64.Test_ZTBSV_64;
const
  n: Int64 = 8;
  k: Int64 = 3;
  incx: Int64 = 1;
var
  lda: Int64;
  h_A_full, h_A_banded: array of rocblas_double_complex;
  h_x_expected, h_b, h_x_gpu_result: array of rocblas_double_complex;
  d_A, d_b: hipDeviceptr_t;
  i, j, pack_idx_row, pack_idx_col: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  temp: rocblas_double_complex;
begin
  lda := k + 1;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;

  SetLength(h_A_full, n * n);
  SetLength(h_A_banded, lda * n);
  SetLength(h_x_expected, n * incx);
  SetLength(h_b, n * incx);
  SetLength(h_x_gpu_result, n * incx);

  // Initialize a well-conditioned upper triangular matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
    begin
      h_A_full[i + j * n].x := (i * 1.1) + (j * 0.9) + 1.0;
      h_A_full[i + j * n].y := (i * 0.5) - (j * 0.2);
    end;
  for i := 0 to n-1 do begin h_A_full[i + i*n].x := n * 2.0; h_A_full[i + i*n].y := 0; end;

  // Pack the matrix into banded format
  for j := 0 to n - 1 do
    for i := Max(0, j - k) to j do
    begin
      pack_idx_row := k + i - j;
      pack_idx_col := j;
      h_A_banded[pack_idx_row + pack_idx_col * lda] := h_A_full[i + j * n];
    end;

  // Initialize known solution vector x
  for i := 0 to (n * incx) - 1 do begin h_x_expected[i].x := (i mod 5)+1.1; h_x_expected[i].y := (i mod 3)-0.5; end;

  // Calculate b = A*x on CPU using the full matrix
  for i := 0 to n - 1 do
  begin
    h_b[i*incx].x := 0; h_b[i*incx].y := 0;
    for j := i to Min(n - 1, i + k) do
    begin
        temp := h_A_full[i + j*n];
        ComplexMul(temp, h_x_expected[j*incx]);
        ComplexAdd(h_b[i*incx], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_b, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A_banded[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A_banded to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_b, @h_b[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_b to d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ztbsv_64(handle, uplo, transA, diag, n, k, d_A, lda, d_b, incx);
  AssertEquals('rocblas_ztbsv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x_gpu_result[0], d_b, n * incx * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_b to h_x_gpu_result', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_x_expected[i*incx].x, h_x_gpu_result[i*incx].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_x_expected[i*incx].y, h_x_gpu_result[i*incx].y, 1e-12, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_A);
  hipFree(d_b);
end;

procedure TTestRocBLAS_Level2_64.Test_STPSV_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
var
  h_A_full: array of Single;
  h_AP: array of Single;
  h_x_expected, h_b, h_x_gpu_result: array of Single;
  d_AP, d_b: hipDeviceptr_t;
  i, j, pack_idx: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;
  ap_size := n * (n + 1) div 2;

  SetLength(h_A_full, n * n);
  SetLength(h_AP, ap_size);
  SetLength(h_x_expected, n * incx);
  SetLength(h_b, n * incx);
  SetLength(h_x_gpu_result, n * incx);

  // Initialize a well-conditioned upper triangular matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
      h_A_full[i + j * n] := (i * 1.1) + (j * 0.9) + 1.0;
  for i := 0 to n - 1 do h_A_full[i + i * n] := n * 2.0;

  // Pack the upper triangular part
  pack_idx := 0;
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_AP[pack_idx] := h_A_full[i + j * n];
      Inc(pack_idx);
    end;

  // Initialize known solution vector x
  for i := 0 to (n * incx) - 1 do h_x_expected[i] := (i mod 5) + 1.1;

  // Calculate b = A*x on CPU
  for i := 0 to n - 1 do
  begin
    h_b[i*incx] := 0;
    for j := i to n - 1 do
    begin
      h_b[i*incx] := h_b[i*incx] + h_A_full[i + j*n] * h_x_expected[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(Single));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_b, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_b, @h_b[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_b to d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_stpsv_64(handle, uplo, transA, diag, n, d_AP, d_b, incx);
  AssertEquals('rocblas_stpsv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x_gpu_result[0], d_b, n * incx * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_b to h_x_gpu_result', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_x_expected[i*incx], h_x_gpu_result[i*incx], 1e-4, Format('Mismatch at index %d', [i]));

  hipFree(d_AP);
  hipFree(d_b);
end;

procedure TTestRocBLAS_Level2_64.Test_DTPSV_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
var
  h_A_full: array of Double;
  h_AP: array of Double;
  h_x_expected, h_b, h_x_gpu_result: array of Double;
  d_AP, d_b: hipDeviceptr_t;
  i, j, pack_idx: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;
  ap_size := n * (n + 1) div 2;

  SetLength(h_A_full, n * n);
  SetLength(h_AP, ap_size);
  SetLength(h_x_expected, n * incx);
  SetLength(h_b, n * incx);
  SetLength(h_x_gpu_result, n * incx);

  // Initialize a well-conditioned upper triangular matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
      h_A_full[i + j * n] := (i * 1.1) + (j * 0.9) + 1.0;
  for i := 0 to n - 1 do h_A_full[i + i * n] := n * 2.0;

  // Pack the upper triangular part
  pack_idx := 0;
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_AP[pack_idx] := h_A_full[i + j * n];
      Inc(pack_idx);
    end;

  // Initialize known solution vector x
  for i := 0 to (n * incx) - 1 do h_x_expected[i] := (i mod 5) + 1.1;

  // Calculate b = A*x on CPU
  for i := 0 to n - 1 do
  begin
    h_b[i*incx] := 0;
    for j := i to n - 1 do
    begin
      h_b[i*incx] := h_b[i*incx] + h_A_full[i + j*n] * h_x_expected[j*incx];
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(Double));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_b, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_b, @h_b[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_b to d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_dtpsv_64(handle, uplo, transA, diag, n, d_AP, d_b, incx);
  AssertEquals('rocblas_dtpsv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x_gpu_result[0], d_b, n * incx * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_b to h_x_gpu_result', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
    CheckEqualsRelative(h_x_expected[i*incx], h_x_gpu_result[i*incx], 1e-12, Format('Mismatch at index %d', [i]));

  hipFree(d_AP);
  hipFree(d_b);
end;

procedure TTestRocBLAS_Level2_64.Test_CTPSV_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
var
  h_A_full: array of rocblas_float_complex;
  h_AP: array of rocblas_float_complex;
  h_x_expected, h_b, h_x_gpu_result: array of rocblas_float_complex;
  d_AP, d_b: hipDeviceptr_t;
  i, j, pack_idx: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  temp: rocblas_float_complex;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;
  ap_size := n * (n + 1) div 2;

  SetLength(h_A_full, n * n);
  SetLength(h_AP, ap_size);
  SetLength(h_x_expected, n * incx);
  SetLength(h_b, n * incx);
  SetLength(h_x_gpu_result, n * incx);

  // Initialize a well-conditioned upper triangular matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
    begin
      h_A_full[i + j*n].x := (i * 1.1) + (j * 0.9) + 1.0;
      h_A_full[i + j*n].y := (i * 0.5) - (j * 0.2);
    end;
  for i := 0 to n - 1 do begin h_A_full[i + i*n].x := n * 2.0; h_A_full[i + i*n].y := 0; end;

  // Pack the upper triangular part
  pack_idx := 0;
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_AP[pack_idx] := h_A_full[i + j*n];
      Inc(pack_idx);
    end;

  // Initialize known solution vector x
  for i := 0 to (n * incx) - 1 do begin h_x_expected[i].x := (i mod 5)+1.1; h_x_expected[i].y := (i mod 3)-0.5; end;

  // Calculate b = A*x on CPU
  for i := 0 to n - 1 do
  begin
    h_b[i*incx].x := 0; h_b[i*incx].y := 0;
    for j := i to n - 1 do
    begin
        temp := h_A_full[i + j*n];
        ComplexMul(temp, h_x_expected[j*incx]);
        ComplexAdd(h_b[i*incx], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_b, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_b, @h_b[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_b to d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ctpsv_64(handle, uplo, transA, diag, n, d_AP, d_b, incx);
  AssertEquals('rocblas_ctpsv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x_gpu_result[0], d_b, n * incx * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_b to h_x_gpu_result', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_x_expected[i*incx].x, h_x_gpu_result[i*incx].x, 1e-4, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_x_expected[i*incx].y, h_x_gpu_result[i*incx].y, 1e-4, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_AP);
  hipFree(d_b);
end;

procedure TTestRocBLAS_Level2_64.Test_ZTPSV_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
var
  h_A_full: array of rocblas_double_complex;
  h_AP: array of rocblas_double_complex;
  h_x_expected, h_b, h_x_gpu_result: array of rocblas_double_complex;
  d_AP, d_b: hipDeviceptr_t;
  i, j, pack_idx: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  temp: rocblas_double_complex;
begin
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;
  ap_size := n * (n + 1) div 2;

  SetLength(h_A_full, n * n);
  SetLength(h_AP, ap_size);
  SetLength(h_x_expected, n * incx);
  SetLength(h_b, n * incx);
  SetLength(h_x_gpu_result, n * incx);

  // Initialize a well-conditioned upper triangular matrix A
  for i := 0 to n - 1 do
    for j := i to n - 1 do
    begin
      h_A_full[i + j*n].x := (i * 1.1) + (j * 0.9) + 1.0;
      h_A_full[i + j*n].y := (i * 0.5) - (j * 0.2);
    end;
  for i := 0 to n - 1 do begin h_A_full[i + i*n].x := n * 2.0; h_A_full[i + i*n].y := 0; end;

  // Pack the upper triangular part
  pack_idx := 0;
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_AP[pack_idx] := h_A_full[i + j*n];
      Inc(pack_idx);
    end;

  // Initialize known solution vector x
  for i := 0 to (n * incx) - 1 do begin h_x_expected[i].x := (i mod 5)+1.1; h_x_expected[i].y := (i mod 3)-0.5; end;

  // Calculate b = A*x on CPU
  for i := 0 to n - 1 do
  begin
    h_b[i*incx].x := 0; h_b[i*incx].y := 0;
    for j := i to n - 1 do
    begin
        temp := h_A_full[i + j*n];
        ComplexMul(temp, h_x_expected[j*incx]);
        ComplexAdd(h_b[i*incx], temp);
    end;
  end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_b, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_b, @h_b[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_b to d_b', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ztpsv_64(handle, uplo, transA, diag, n, d_AP, d_b, incx);
  AssertEquals('rocblas_ztpsv_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_x_gpu_result[0], d_b, n * incx * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_b to h_x_gpu_result', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to n - 1 do
  begin
    CheckEqualsRelative(h_x_expected[i*incx].x, h_x_gpu_result[i*incx].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_x_expected[i*incx].y, h_x_gpu_result[i*incx].y, 1e-12, Format('Imaginary part mismatch at index %d', [i]));
  end;

  hipFree(d_AP);
  hipFree(d_b);
end;

procedure TTestRocBLAS_Level2_64.Test_SGER_64;
const
  m: Int64 = 10;
  n: Int64 = 12;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Single = 1.5;
var
  h_A, h_A_expected: array of Single;
  h_x, h_y: array of Single;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, m * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do
  begin
    h_A[i] := (i mod 10) + 1.2;
    h_A_expected[i] := h_A[i];
  end;
  for i := 0 to (m * incx) - 1 do h_x[i] := (i mod 5) + 1.1;
  for i := 0 to (n * incy) - 1 do h_y[i] := (i mod 3) + 1.3;

  // CPU calculation: A := alpha*x*y**T + A
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
      h_A_expected[i + j*lda] := h_A_expected[i + j*lda] + alpha * h_x[i*incx] * h_y[j*incy];

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Single));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, m * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(Single));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], m * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sger_64(handle, m, n, @alpha, d_x, incx, d_y, incy, d_A, lda);
  AssertEquals('rocblas_sger_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
      CheckEqualsRelative(h_A_expected[i + j*lda], h_A[i + j*lda], 1e-5, Format('Mismatch at A[%d, %d]', [i, j]));

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_DGER_64;
const
  m: Int64 = 10;
  n: Int64 = 12;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Double = 1.5;
var
  h_A, h_A_expected: array of Double;
  h_x, h_y: array of Double;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, m * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do
  begin
    h_A[i] := (i mod 10) + 1.2;
    h_A_expected[i] := h_A[i];
  end;
  for i := 0 to (m * incx) - 1 do h_x[i] := (i mod 5) + 1.1;
  for i := 0 to (n * incy) - 1 do h_y[i] := (i mod 3) + 1.3;

  // CPU calculation: A := alpha*x*y**T + A
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
      h_A_expected[i + j*lda] := h_A_expected[i + j*lda] + alpha * h_x[i*incx] * h_y[j*incy];

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Double));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, m * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(Double));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], m * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dger_64(handle, m, n, @alpha, d_x, incx, d_y, incy, d_A, lda);
  AssertEquals('rocblas_dger_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
      CheckEqualsRelative(h_A_expected[i + j*lda], h_A[i + j*lda], 1e-12, Format('Mismatch at A[%d, %d]', [i, j]));

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_CGERU_64;
const
  m: Int64 = 10;
  n: Int64 = 12;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha: rocblas_float_complex;
  h_A, h_A_expected: array of rocblas_float_complex;
  h_x, h_y: array of rocblas_float_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  term: rocblas_float_complex;
begin
  alpha.x := 1.5; alpha.y := 0.5;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, m * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i].x := i+1.2; h_A[i].y := i-1.8; h_A_expected[i] := h_A[i]; end;
  for i := 0 to (m * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;
  for i := 0 to (n * incy) - 1 do begin h_y[i].x := i+1.3; h_y[i].y := i-0.7; end;

  // CPU calculation: A := alpha*x*y**T + A
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      term := h_x[i*incx];
      ComplexMul(term, h_y[j*incy]);
      ComplexMul(term, alpha);
      ComplexAdd(h_A_expected[i + j*lda], term);
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, m * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], m * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgeru_64(handle, m, n, @alpha, d_x, incx, d_y, incy, d_A, lda);
  AssertEquals('rocblas_cgeru_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      CheckEqualsRelative(h_A_expected[i + j*lda].x, h_A[i + j*lda].x, 1e-4, Format('Real part mismatch at A[%d,%d]',[i,j]));
      CheckEqualsRelative(h_A_expected[i + j*lda].y, h_A[i + j*lda].y, 1e-4, Format('Imag part mismatch at A[%d,%d]',[i,j]));
    end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_ZGERU_64;
const
  m: Int64 = 10;
  n: Int64 = 12;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha: rocblas_double_complex;
  h_A, h_A_expected: array of rocblas_double_complex;
  h_x, h_y: array of rocblas_double_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  term: rocblas_double_complex;
begin
  alpha.x := 1.5; alpha.y := 0.5;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, m * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i].x := i+1.2; h_A[i].y := i-1.8; h_A_expected[i] := h_A[i]; end;
  for i := 0 to (m * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;
  for i := 0 to (n * incy) - 1 do begin h_y[i].x := i+1.3; h_y[i].y := i-0.7; end;

  // CPU calculation: A := alpha*x*y**T + A
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      term := h_x[i*incx];
      ComplexMul(term, h_y[j*incy]);
      ComplexMul(term, alpha);
      ComplexAdd(h_A_expected[i + j*lda], term);
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, m * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], m * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgeru_64(handle, m, n, @alpha, d_x, incx, d_y, incy, d_A, lda);
  AssertEquals('rocblas_zgeru_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      CheckEqualsRelative(h_A_expected[i + j*lda].x, h_A[i + j*lda].x, 1e-12, Format('Real part mismatch at A[%d,%d]',[i,j]));
      CheckEqualsRelative(h_A_expected[i + j*lda].y, h_A[i + j*lda].y, 1e-12, Format('Imag part mismatch at A[%d,%d]',[i,j]));
    end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_CGERC_64;
const
  m: Int64 = 10;
  n: Int64 = 12;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha: rocblas_float_complex;
  h_A, h_A_expected: array of rocblas_float_complex;
  h_x, h_y: array of rocblas_float_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  term, y_conj: rocblas_float_complex;
begin
  alpha.x := 1.5; alpha.y := 0.5;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, m * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i].x := i+1.2; h_A[i].y := i-1.8; h_A_expected[i] := h_A[i]; end;
  for i := 0 to (m * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;
  for i := 0 to (n * incy) - 1 do begin h_y[i].x := i+1.3; h_y[i].y := i-0.7; end;

  // CPU calculation: A := alpha*x*y**H + A
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      term := h_x[i*incx];
      y_conj := h_y[j*incy];
      y_conj.y := -y_conj.y; // Conjugate y
      ComplexMul(term, y_conj);
      ComplexMul(term, alpha);
      ComplexAdd(h_A_expected[i + j*lda], term);
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, m * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], m * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgerc_64(handle, m, n, @alpha, d_x, incx, d_y, incy, d_A, lda);
  AssertEquals('rocblas_cgerc_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      CheckEqualsRelative(h_A_expected[i + j*lda].x, h_A[i + j*lda].x, 1e-4, Format('Real part mismatch at A[%d,%d]',[i,j]));
      CheckEqualsRelative(h_A_expected[i + j*lda].y, h_A[i + j*lda].y, 1e-4, Format('Imag part mismatch at A[%d,%d]',[i,j]));
    end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_ZGERC_64;
const
  m: Int64 = 10;
  n: Int64 = 12;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha: rocblas_double_complex;
  h_A, h_A_expected: array of rocblas_double_complex;
  h_x, h_y: array of rocblas_double_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  term, y_conj: rocblas_double_complex;
begin
  alpha.x := 1.5; alpha.y := 0.5;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, m * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i].x := i+1.2; h_A[i].y := i-1.8; h_A_expected[i] := h_A[i]; end;
  for i := 0 to (m * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;
  for i := 0 to (n * incy) - 1 do begin h_y[i].x := i+1.3; h_y[i].y := i-0.7; end;

  // CPU calculation: A := alpha*x*y**H + A
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      term := h_x[i*incx];
      y_conj := h_y[j*incy];
      y_conj.y := -y_conj.y; // Conjugate y
      ComplexMul(term, y_conj);
      ComplexMul(term, alpha);
      ComplexAdd(h_A_expected[i + j*lda], term);
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, m * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], m * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgerc_64(handle, m, n, @alpha, d_x, incx, d_y, incy, d_A, lda);
  AssertEquals('rocblas_zgerc_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to m - 1 do
    begin
      CheckEqualsRelative(h_A_expected[i + j*lda].x, h_A[i + j*lda].x, 1e-12, Format('Real part mismatch at A[%d,%d]',[i,j]));
      CheckEqualsRelative(h_A_expected[i + j*lda].y, h_A[i + j*lda].y, 1e-12, Format('Imag part mismatch at A[%d,%d]',[i,j]));
    end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_SSYR_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  alpha: Single = 1.5;
var
  h_A, h_A_expected: array of Single;
  h_x: array of Single;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
begin
  uplo := rocblas_fill_upper;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, n * incx);

  // Initialize data
  for i := 0 to (lda * n) - 1 do
  begin
    h_A[i] := (i mod 10) + 1.2;
    h_A_expected[i] := h_A[i];
  end;
  for i := 0 to (n * incx) - 1 do h_x[i] := (i mod 5) + 1.1;

  // CPU calculation: A := alpha*x*x**T + A (only for the specified triangle)
  for j := 0 to n - 1 do
    for i := 0 to j do // Upper triangle
    begin
      h_A_expected[i + j*lda] := h_A_expected[i + j*lda] + alpha * h_x[i*incx] * h_x[j*incx];
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Single));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_ssyr_64(handle, uplo, n, @alpha, d_x, incx, d_A, lda);
  AssertEquals('rocblas_ssyr_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to j do // Verify only the calculated triangle
      CheckEqualsRelative(h_A_expected[i + j*lda], h_A[i + j*lda], 1e-5, Format('Mismatch at A[%d, %d]', [i, j]));

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_DSYR_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  alpha: Double = 1.5;
var
  h_A, h_A_expected: array of Double;
  h_x: array of Double;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
begin
  uplo := rocblas_fill_upper;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, n * incx);

  // Initialize data
  for i := 0 to (lda * n) - 1 do
  begin
    h_A[i] := (i mod 10) + 1.2;
    h_A_expected[i] := h_A[i];
  end;
  for i := 0 to (n * incx) - 1 do h_x[i] := (i mod 5) + 1.1;

  // CPU calculation: A := alpha*x*x**T + A (only for the specified triangle)
  for j := 0 to n - 1 do
    for i := 0 to j do // Upper triangle
    begin
      h_A_expected[i + j*lda] := h_A_expected[i + j*lda] + alpha * h_x[i*incx] * h_x[j*incx];
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Double));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dsyr_64(handle, uplo, n, @alpha, d_x, incx, d_A, lda);
  AssertEquals('rocblas_dsyr_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to j do // Verify only the calculated triangle
      CheckEqualsRelative(h_A_expected[i + j*lda], h_A[i + j*lda], 1e-12, Format('Mismatch at A[%d, %d]', [i, j]));

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_CSYR_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
var
  alpha: rocblas_float_complex;
  h_A, h_A_expected: array of rocblas_float_complex;
  h_x: array of rocblas_float_complex;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  term: rocblas_float_complex;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := 0.5;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, n * incx);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i].x := i+1.2; h_A[i].y := i-1.8; h_A_expected[i] := h_A[i]; end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;

  // CPU calculation: A := alpha*x*x**T + A
  for j := 0 to n - 1 do
    for i := 0 to j do // Upper triangle
    begin
      term := h_x[i*incx];
      ComplexMul(term, h_x[j*incx]);
      ComplexMul(term, alpha);
      ComplexAdd(h_A_expected[i + j*lda], term);
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_csyr_64(handle, uplo, n, @alpha, d_x, incx, d_A, lda);
  AssertEquals('rocblas_csyr_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(h_A_expected[i + j*lda].x, h_A[i + j*lda].x, 1e-4, Format('Real part mismatch at A[%d,%d]',[i,j]));
      CheckEqualsRelative(h_A_expected[i + j*lda].y, h_A[i + j*lda].y, 1e-4, Format('Imag part mismatch at A[%d,%d]',[i,j]));
    end;

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_ZSYR_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
var
  alpha: rocblas_double_complex;
  h_A, h_A_expected: array of rocblas_double_complex;
  h_x: array of rocblas_double_complex;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  term: rocblas_double_complex;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := 0.5;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, n * incx);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i].x := i+1.2; h_A[i].y := i-1.8; h_A_expected[i] := h_A[i]; end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;

  // CPU calculation: A := alpha*x*x**T + A
  for j := 0 to n - 1 do
    for i := 0 to j do // Upper triangle
    begin
      term := h_x[i*incx];
      ComplexMul(term, h_x[j*incx]);
      ComplexMul(term, alpha);
      ComplexAdd(h_A_expected[i + j*lda], term);
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zsyr_64(handle, uplo, n, @alpha, d_x, incx, d_A, lda);
  AssertEquals('rocblas_zsyr_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(h_A_expected[i + j*lda].x, h_A[i + j*lda].x, 1e-12, Format('Real part mismatch at A[%d,%d]',[i,j]));
      CheckEqualsRelative(h_A_expected[i + j*lda].y, h_A[i + j*lda].y, 1e-12, Format('Imag part mismatch at A[%d,%d]',[i,j]));
    end;

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_CHER_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  alpha: Single = 1.5;
var
  h_A, h_A_expected: array of rocblas_float_complex;
  h_x: array of rocblas_float_complex;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  term, x_conj: rocblas_float_complex;
begin
  uplo := rocblas_fill_upper;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, n * incx);

  // Initialize data
  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
    begin
      h_A[i + j*lda].x := i + j*1.2;
      h_A[i + j*lda].y := i - j*0.8;
      if i = j then h_A[i + j*lda].y := 0.0; // Diagonal of hermitian is real
      h_A_expected[i + j*lda] := h_A[i + j*lda];
    end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;

  // CPU calculation: A := alpha*x*x**H + A
  for j := 0 to n - 1 do
    for i := 0 to j do // Upper triangle
    begin
      x_conj := h_x[j*incx];
      x_conj.y := -x_conj.y; // Conjugate
      term := h_x[i*incx];
      ComplexMul(term, x_conj);
      // Multiply by real alpha
      term.x := term.x * alpha;
      term.y := term.y * alpha;
      ComplexAdd(h_A_expected[i + j*lda], term);
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cher_64(handle, uplo, n, @alpha, d_x, incx, d_A, lda);
  AssertEquals('rocblas_cher_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(h_A_expected[i + j*lda].x, h_A[i + j*lda].x, 1e-4, Format('Real part mismatch at A[%d,%d]',[i,j]));
      // Imaginary part of diagonal should be unchanged (and ideally zero)
      if i <> j then
        CheckEqualsRelative(h_A_expected[i + j*lda].y, h_A[i + j*lda].y, 1e-4, Format('Imag part mismatch at A[%d,%d]',[i,j]));
    end;

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_ZHER_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  alpha: Double = 1.5;
var
  h_A, h_A_expected: array of rocblas_double_complex;
  h_x: array of rocblas_double_complex;
  d_A, d_x: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  term, x_conj: rocblas_double_complex;
begin
  uplo := rocblas_fill_upper;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, n * incx);

  // Initialize data
  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
    begin
      h_A[i + j*lda].x := i + j*1.2;
      h_A[i + j*lda].y := i - j*0.8;
      if i = j then h_A[i + j*lda].y := 0.0;
      h_A_expected[i + j*lda] := h_A[i + j*lda];
    end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;

  // CPU calculation: A := alpha*x*x**H + A
  for j := 0 to n - 1 do
    for i := 0 to j do // Upper triangle
    begin
      x_conj := h_x[j*incx];
      x_conj.y := -x_conj.y; // Conjugate
      term := h_x[i*incx];
      ComplexMul(term, x_conj);
      // Multiply by real alpha
      term.x := term.x * alpha;
      term.y := term.y * alpha;
      ComplexAdd(h_A_expected[i + j*lda], term);
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zher_64(handle, uplo, n, @alpha, d_x, incx, d_A, lda);
  AssertEquals('rocblas_zher_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(h_A_expected[i + j*lda].x, h_A[i + j*lda].x, 1e-12, Format('Real part mismatch at A[%d,%d]',[i,j]));
      if i <> j then
        CheckEqualsRelative(h_A_expected[i + j*lda].y, h_A[i + j*lda].y, 1e-12, Format('Imag part mismatch at A[%d,%d]',[i,j]));
    end;

  hipFree(d_A);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_SSPR_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
  alpha: Single = 1.5;
var
  uplo: rocblas_fill;
  h_AP, h_AP_expected: array of Single;
  h_x: array of Single;
  d_AP, d_x: hipDeviceptr_t;
  i, j, k: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  uplo := rocblas_fill_upper;
  ap_size := n * (n + 1) div 2;

  SetLength(h_AP, ap_size);
  SetLength(h_AP_expected, ap_size);
  SetLength(h_x, n * incx);

  // Initialize data
  for i := 0 to ap_size - 1 do begin h_AP[i] := i + 1.2; h_AP_expected[i] := h_AP[i]; end;
  for i := 0 to (n * incx) - 1 do h_x[i] := i + 1.1;

  // CPU calculation: A := alpha*x*x**T + A
  k := 0;
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_AP_expected[k] := h_AP_expected[k] + alpha * h_x[i*incx] * h_x[j*incx];
      Inc(k);
    end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(Single));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sspr_64(handle, uplo, n, @alpha, d_x, incx, d_AP);
  AssertEquals('rocblas_sspr_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_AP[0], d_AP, ap_size * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_AP to h_AP', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to ap_size - 1 do
    CheckEqualsRelative(h_AP_expected[i], h_AP[i], 1e-5, Format('Mismatch at index %d', [i]));

  hipFree(d_AP);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_DSPR_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
  alpha: Double = 1.5;
var
  uplo: rocblas_fill;
  h_AP, h_AP_expected: array of Double;
  h_x: array of Double;
  d_AP, d_x: hipDeviceptr_t;
  i, j, k: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  uplo := rocblas_fill_upper;
  ap_size := n * (n + 1) div 2;

  SetLength(h_AP, ap_size);
  SetLength(h_AP_expected, ap_size);
  SetLength(h_x, n * incx);

  // Initialize data
  for i := 0 to ap_size - 1 do begin h_AP[i] := i + 1.2; h_AP_expected[i] := h_AP[i]; end;
  for i := 0 to (n * incx) - 1 do h_x[i] := i + 1.1;

  // CPU calculation: A := alpha*x*x**T + A
  k := 0;
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_AP_expected[k] := h_AP_expected[k] + alpha * h_x[i*incx] * h_x[j*incx];
      Inc(k);
    end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(Double));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dspr_64(handle, uplo, n, @alpha, d_x, incx, d_AP);
  AssertEquals('rocblas_dspr_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_AP[0], d_AP, ap_size * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_AP to h_AP', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to ap_size - 1 do
    CheckEqualsRelative(h_AP_expected[i], h_AP[i], 1e-12, Format('Mismatch at index %d', [i]));

  hipFree(d_AP);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_CHPR_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
  alpha: Single = 1.5;
var
  uplo: rocblas_fill;
  h_AP, h_AP_expected: array of rocblas_float_complex;
  h_x: array of rocblas_float_complex;
  d_AP, d_x: hipDeviceptr_t;
  i, j, k: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  term, x_conj: rocblas_float_complex;
begin
  uplo := rocblas_fill_upper;
  ap_size := n * (n + 1) div 2;

  SetLength(h_AP, ap_size);
  SetLength(h_AP_expected, ap_size);
  SetLength(h_x, n * incx);

  // Initialize data
  for i := 0 to ap_size - 1 do begin h_AP[i].x := i+1.2; h_AP[i].y := i-1.8; h_AP_expected[i] := h_AP[i]; end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;

  // CPU calculation: A := alpha*x*x**H + A
  k := 0;
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      x_conj := h_x[j*incx];
      x_conj.y := -x_conj.y; // Conjugate
      term := h_x[i*incx];
      ComplexMul(term, x_conj);
      term.x := term.x * alpha;
      term.y := term.y * alpha;

      if i = j then
      begin
        // Diagonal update must be real. The imaginary part of A is zeroed by rocBLAS.
        h_AP_expected[k].x := h_AP_expected[k].x + term.x;
        h_AP_expected[k].y := 0.0; // <-- CORRECTED
      end
      else
      begin
        ComplexAdd(h_AP_expected[k], term);
      end;
      Inc(k);
    end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_chpr_64(handle, uplo, n, @alpha, d_x, incx, d_AP);
  AssertEquals('rocblas_chpr_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_AP[0], d_AP, ap_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_AP to h_AP', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to ap_size - 1 do
  begin
    CheckEqualsRelative(h_AP_expected[i].x, h_AP[i].x, 1e-4, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_AP_expected[i].y, h_AP[i].y, 1e-4, Format('Imag part mismatch at index %d', [i]));
  end;

  hipFree(d_AP);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_ZHPR_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
  alpha: Double = 1.5;
var
  uplo: rocblas_fill;
  h_AP, h_AP_expected: array of rocblas_double_complex;
  h_x: array of rocblas_double_complex;
  d_AP, d_x: hipDeviceptr_t;
  i, j, k: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  term, x_conj: rocblas_double_complex;
begin
  uplo := rocblas_fill_upper;
  ap_size := n * (n + 1) div 2;

  SetLength(h_AP, ap_size);
  SetLength(h_AP_expected, ap_size);
  SetLength(h_x, n * incx);

  // Initialize data
  for i := 0 to ap_size - 1 do begin h_AP[i].x := i+1.2; h_AP[i].y := i-1.8; h_AP_expected[i] := h_AP[i]; end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;

  // CPU calculation: A := alpha*x*x**H + A
  k := 0;
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      x_conj := h_x[j*incx];
      x_conj.y := -x_conj.y; // Conjugate
      term := h_x[i*incx];
      ComplexMul(term, x_conj);
      term.x := term.x * alpha;
      term.y := term.y * alpha;

      if i = j then
      begin
        // Diagonal update must be real. The imaginary part of A is zeroed by rocBLAS.
        h_AP_expected[k].x := h_AP_expected[k].x + term.x;
        h_AP_expected[k].y := 0.0; // <-- CORRECTED
      end
      else
      begin
        ComplexAdd(h_AP_expected[k], term);
      end;
      Inc(k);
    end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zhpr_64(handle, uplo, n, @alpha, d_x, incx, d_AP);
  AssertEquals('rocblas_zhpr_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_AP[0], d_AP, ap_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_AP to h_AP', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to ap_size - 1 do
  begin
    CheckEqualsRelative(h_AP_expected[i].x, h_AP[i].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_AP_expected[i].y, h_AP[i].y, 1e-12, Format('Imag part mismatch at index %d', [i]));
  end;

  hipFree(d_AP);
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level2_64.Test_SSYR2_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Single = 1.5;
var
  h_A, h_A_expected: array of Single;
  h_x, h_y: array of Single;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
begin
  uplo := rocblas_fill_upper;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i] := (i mod 10) + 1.2; h_A_expected[i] := h_A[i]; end;
  for i := 0 to (n * incx) - 1 do h_x[i] := (i mod 5) + 1.1;
  for i := 0 to (n * incy) - 1 do h_y[i] := (i mod 3) + 1.3;

  // CPU calculation: A := alpha*x*y**T + alpha*y*x**T + A
  for j := 0 to n - 1 do
    for i := 0 to j do // Upper triangle
    begin
      h_A_expected[i + j*lda] := h_A_expected[i + j*lda] +
                                 alpha * h_x[i*incx] * h_y[j*incy] +
                                 alpha * h_y[i*incy] * h_x[j*incx];
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Single));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(Single));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_ssyr2_64(handle, uplo, n, @alpha, d_x, incx, d_y, incy, d_A, lda);
  AssertEquals('rocblas_ssyr2_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to j do
      CheckEqualsRelative(h_A_expected[i + j*lda], h_A[i + j*lda], 1e-5, Format('Mismatch at A[%d, %d]', [i, j]));

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_DSYR2_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Double = 1.5;
var
  h_A, h_A_expected: array of Double;
  h_x, h_y: array of Double;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
begin
  uplo := rocblas_fill_upper;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i] := (i mod 10) + 1.2; h_A_expected[i] := h_A[i]; end;
  for i := 0 to (n * incx) - 1 do h_x[i] := (i mod 5) + 1.1;
  for i := 0 to (n * incy) - 1 do h_y[i] := (i mod 3) + 1.3;

  // CPU calculation: A := alpha*x*y**T + alpha*y*x**T + A
  for j := 0 to n - 1 do
    for i := 0 to j do // Upper triangle
    begin
      h_A_expected[i + j*lda] := h_A_expected[i + j*lda] +
                                 alpha * h_x[i*incx] * h_y[j*incy] +
                                 alpha * h_y[i*incy] * h_x[j*incx];
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(Double));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(Double));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dsyr2_64(handle, uplo, n, @alpha, d_x, incx, d_y, incy, d_A, lda);
  AssertEquals('rocblas_dsyr2_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to j do
      CheckEqualsRelative(h_A_expected[i + j*lda], h_A[i + j*lda], 1e-12, Format('Mismatch at A[%d, %d]', [i, j]));

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_CSYR2_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha: rocblas_float_complex;
  h_A, h_A_expected: array of rocblas_float_complex;
  h_x, h_y: array of rocblas_float_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  term1, term2: rocblas_float_complex;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := 0.5;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i].x := i+1.2; h_A[i].y := i-1.8; h_A_expected[i] := h_A[i]; end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;
  for i := 0 to (n * incy) - 1 do begin h_y[i].x := i+1.3; h_y[i].y := i-0.7; end;

  // CPU calculation: A := alpha*x*y**T + alpha*y*x**T + A
  for j := 0 to n - 1 do
    for i := 0 to j do // Upper triangle
    begin
      // term1 = alpha * x[i] * y[j]
      term1 := h_x[i*incx];
      ComplexMul(term1, h_y[j*incy]);
      ComplexMul(term1, alpha);
      // term2 = alpha * y[i] * x[j]
      term2 := h_y[i*incy];
      ComplexMul(term2, h_x[j*incx]);
      ComplexMul(term2, alpha);

      ComplexAdd(h_A_expected[i + j*lda], term1);
      ComplexAdd(h_A_expected[i + j*lda], term2);
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_csyr2_64(handle, uplo, n, @alpha, d_x, incx, d_y, incy, d_A, lda);
  AssertEquals('rocblas_csyr2_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(h_A_expected[i + j*lda].x, h_A[i + j*lda].x, 1e-4, Format('Real part mismatch at A[%d,%d]',[i,j]));
      CheckEqualsRelative(h_A_expected[i + j*lda].y, h_A[i + j*lda].y, 1e-4, Format('Imag part mismatch at A[%d,%d]',[i,j]));
    end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_ZSYR2_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha: rocblas_double_complex;
  h_A, h_A_expected: array of rocblas_double_complex;
  h_x, h_y: array of rocblas_double_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  term1, term2: rocblas_double_complex;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := 0.5;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to (lda * n) - 1 do begin h_A[i].x := i+1.2; h_A[i].y := i-1.8; h_A_expected[i] := h_A[i]; end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;
  for i := 0 to (n * incy) - 1 do begin h_y[i].x := i+1.3; h_y[i].y := i-0.7; end;

  // CPU calculation: A := alpha*x*y**T + alpha*y*x**T + A
  for j := 0 to n - 1 do
    for i := 0 to j do // Upper triangle
    begin
      // term1 = alpha * x[i] * y[j]
      term1 := h_x[i*incx];
      ComplexMul(term1, h_y[j*incy]);
      ComplexMul(term1, alpha);
      // term2 = alpha * y[i] * x[j]
      term2 := h_y[i*incy];
      ComplexMul(term2, h_x[j*incx]);
      ComplexMul(term2, alpha);

      ComplexAdd(h_A_expected[i + j*lda], term1);
      ComplexAdd(h_A_expected[i + j*lda], term2);
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zsyr2_64(handle, uplo, n, @alpha, d_x, incx, d_y, incy, d_A, lda);
  AssertEquals('rocblas_zsyr2_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(h_A_expected[i + j*lda].x, h_A[i + j*lda].x, 1e-12, Format('Real part mismatch at A[%d,%d]',[i,j]));
      CheckEqualsRelative(h_A_expected[i + j*lda].y, h_A[i + j*lda].y, 1e-12, Format('Imag part mismatch at A[%d,%d]',[i,j]));
    end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_CHER2_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha: rocblas_float_complex;
  h_A, h_A_expected: array of rocblas_float_complex;
  h_x, h_y: array of rocblas_float_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  term1, term2, y_conj, alpha_conj, x_conj: rocblas_float_complex;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := -0.5;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
    begin
      h_A[i + j*lda].x := i + j*1.2;
      h_A[i + j*lda].y := i - j*0.8;
      if i = j then h_A[i + j*lda].y := 0.0;
      h_A_expected[i + j*lda] := h_A[i + j*lda];
    end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;
  for i := 0 to (n * incy) - 1 do begin h_y[i].x := i+1.3; h_y[i].y := i-0.7; end;

  // CPU calculation: A := alpha*x*y**H + conj(alpha)*y*x**H + A
  alpha_conj := alpha;
  alpha_conj.y := -alpha_conj.y;

  for j := 0 to n - 1 do
    for i := 0 to j do // Upper triangle
    begin
      // term1 = alpha * x[i] * y[j]**H
      y_conj := h_y[j*incy]; y_conj.y := -y_conj.y;
      term1 := h_x[i*incx];
      ComplexMul(term1, y_conj);
      ComplexMul(term1, alpha);
      ComplexAdd(h_A_expected[i + j*lda], term1);

      // term2 = conj(alpha) * y[i] * x[j]**H
      x_conj := h_x[j*incx]; x_conj.y := -x_conj.y;
      term2 := h_y[i*incy];
      ComplexMul(term2, x_conj);
      ComplexMul(term2, alpha_conj);
      ComplexAdd(h_A_expected[i + j*lda], term2);

      if i = j then h_A_expected[i+j*lda].y := 0.0; // Enforce real diagonal
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cher2_64(handle, uplo, n, @alpha, d_x, incx, d_y, incy, d_A, lda);
  AssertEquals('rocblas_cher2_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(h_A_expected[i + j*lda].x, h_A[i + j*lda].x, 1e-4, Format('Real part mismatch at A[%d,%d]',[i,j]));
      CheckEqualsRelative(h_A_expected[i + j*lda].y, h_A[i + j*lda].y, 1e-4, Format('Imag part mismatch at A[%d,%d]',[i,j]));
    end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_ZHER2_64;
const
  n: Int64 = 10;
  lda: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  alpha: rocblas_double_complex;
  h_A, h_A_expected: array of rocblas_double_complex;
  h_x, h_y: array of rocblas_double_complex;
  d_A, d_x, d_y: hipDeviceptr_t;
  i, j: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  term1, term2, y_conj, alpha_conj, x_conj: rocblas_double_complex;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := -0.5;

  SetLength(h_A, lda * n);
  SetLength(h_A_expected, lda * n);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for j := 0 to n - 1 do
    for i := 0 to n - 1 do
    begin
      h_A[i + j*lda].x := i + j*1.2;
      h_A[i + j*lda].y := i - j*0.8;
      if i = j then h_A[i + j*lda].y := 0.0;
      h_A_expected[i + j*lda] := h_A[i + j*lda];
    end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;
  for i := 0 to (n * incy) - 1 do begin h_y[i].x := i+1.3; h_y[i].y := i-0.7; end;

  // CPU calculation: A := alpha*x*y**H + conj(alpha)*y*x**H + A
  alpha_conj := alpha;
  alpha_conj.y := -alpha_conj.y;

  for j := 0 to n - 1 do
    for i := 0 to j do // Upper triangle
    begin
      // term1 = alpha * x[i] * y[j]**H
      y_conj := h_y[j*incy]; y_conj.y := -y_conj.y;
      term1 := h_x[i*incx];
      ComplexMul(term1, y_conj);
      ComplexMul(term1, alpha);
      ComplexAdd(h_A_expected[i + j*lda], term1);

      // term2 = conj(alpha) * y[i] * x[j]**H
      x_conj := h_x[j*incx]; x_conj.y := -x_conj.y;
      term2 := h_y[i*incy];
      ComplexMul(term2, x_conj);
      ComplexMul(term2, alpha_conj);
      ComplexAdd(h_A_expected[i + j*lda], term2);

      if i = j then h_A_expected[i+j*lda].y := 0.0; // Enforce real diagonal
    end;

  // GPU execution
  hip_status := hipMalloc(d_A, lda * n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A, @h_A[0], lda * n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_A to d_A', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zher2_64(handle, uplo, n, @alpha, d_x, incx, d_y, incy, d_A, lda);
  AssertEquals('rocblas_zher2_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_A[0], d_A, lda * n * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_A to h_A', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(h_A_expected[i + j*lda].x, h_A[i + j*lda].x, 1e-12, Format('Real part mismatch at A[%d,%d]',[i,j]));
      CheckEqualsRelative(h_A_expected[i + j*lda].y, h_A[i + j*lda].y, 1e-12, Format('Imag part mismatch at A[%d,%d]',[i,j]));
    end;

  hipFree(d_A);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_SSPR2_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Single = 1.5;
var
  uplo: rocblas_fill;
  h_AP, h_AP_expected: array of Single;
  h_x, h_y: array of Single;
  d_AP, d_x, d_y: hipDeviceptr_t;
  i, j, k: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  uplo := rocblas_fill_upper;
  ap_size := n * (n + 1) div 2;

  SetLength(h_AP, ap_size);
  SetLength(h_AP_expected, ap_size);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to ap_size - 1 do begin h_AP[i] := i + 1.2; h_AP_expected[i] := h_AP[i]; end;
  for i := 0 to (n * incx) - 1 do h_x[i] := i + 1.1;
  for i := 0 to (n * incy) - 1 do h_y[i] := i + 1.3;

  // CPU calculation: A := alpha*x*y**T + alpha*y*x**T + A
  k := 0;
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_AP_expected[k] := h_AP_expected[k] +
                          alpha * h_x[i*incx] * h_y[j*incy] +
                          alpha * h_y[i*incy] * h_x[j*incx];
      Inc(k);
    end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(Single));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Single));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(Single));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sspr2_64(handle, uplo, n, @alpha, d_x, incx, d_y, incy, d_AP);
  AssertEquals('rocblas_sspr2_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_AP[0], d_AP, ap_size * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_AP to h_AP', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to ap_size - 1 do
    CheckEqualsRelative(h_AP_expected[i], h_AP[i], 1e-5, Format('Mismatch at index %d', [i]));

  hipFree(d_AP);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_DSPR2_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
  alpha: Double = 1.5;
var
  uplo: rocblas_fill;
  h_AP, h_AP_expected: array of Double;
  h_x, h_y: array of Double;
  d_AP, d_x, d_y: hipDeviceptr_t;
  i, j, k: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  uplo := rocblas_fill_upper;
  ap_size := n * (n + 1) div 2;

  SetLength(h_AP, ap_size);
  SetLength(h_AP_expected, ap_size);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to ap_size - 1 do begin h_AP[i] := i + 1.2; h_AP_expected[i] := h_AP[i]; end;
  for i := 0 to (n * incx) - 1 do h_x[i] := i + 1.1;
  for i := 0 to (n * incy) - 1 do h_y[i] := i + 1.3;

  // CPU calculation: A := alpha*x*y**T + alpha*y*x**T + A
  k := 0;
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      h_AP_expected[k] := h_AP_expected[k] +
                          alpha * h_x[i*incx] * h_y[j*incy] +
                          alpha * h_y[i*incy] * h_x[j*incx];
      Inc(k);
    end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(Double));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(Double));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(Double));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dspr2_64(handle, uplo, n, @alpha, d_x, incx, d_y, incy, d_AP);
  AssertEquals('rocblas_dspr2_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_AP[0], d_AP, ap_size * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_AP to h_AP', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to ap_size - 1 do
    CheckEqualsRelative(h_AP_expected[i], h_AP[i], 1e-12, Format('Mismatch at index %d', [i]));

  hipFree(d_AP);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_CHPR2_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  uplo: rocblas_fill;
  alpha: rocblas_float_complex;
  h_AP, h_AP_expected: array of rocblas_float_complex;
  h_x, h_y: array of rocblas_float_complex;
  d_AP, d_x, d_y: hipDeviceptr_t;
  i, j, k: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  term1, term2, y_conj, alpha_conj, x_conj: rocblas_float_complex;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := -0.5;
  ap_size := n * (n + 1) div 2;

  SetLength(h_AP, ap_size);
  SetLength(h_AP_expected, ap_size);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to ap_size - 1 do begin h_AP[i].x := i+1.2; h_AP[i].y := i-1.8; h_AP_expected[i] := h_AP[i]; end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;
  for i := 0 to (n * incy) - 1 do begin h_y[i].x := i+1.3; h_y[i].y := i-0.7; end;

  // CPU calculation: A := alpha*x*y**H + conj(alpha)*y*x**H + A
  alpha_conj := alpha;
  alpha_conj.y := -alpha_conj.y;
  k := 0;
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      // term1 = alpha * x[i] * y[j]**H
      y_conj := h_y[j*incy]; y_conj.y := -y_conj.y;
      term1 := h_x[i*incx];
      ComplexMul(term1, y_conj);
      ComplexMul(term1, alpha);

      // term2 = conj(alpha) * y[i] * x[j]**H
      x_conj := h_x[j*incx]; x_conj.y := -x_conj.y;
      term2 := h_y[i*incy];
      ComplexMul(term2, x_conj);
      ComplexMul(term2, alpha_conj);

      ComplexAdd(h_AP_expected[k], term1);
      ComplexAdd(h_AP_expected[k], term2);

      if i = j then h_AP_expected[k].y := 0.0;
      Inc(k);
    end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_chpr2_64(handle, uplo, n, @alpha, d_x, incx, d_y, incy, d_AP);
  AssertEquals('rocblas_chpr2_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_AP[0], d_AP, ap_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_AP to h_AP', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to ap_size - 1 do
  begin
    CheckEqualsRelative(h_AP_expected[i].x, h_AP[i].x, 1e-4, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_AP_expected[i].y, h_AP[i].y, 1e-4, Format('Imag part mismatch at index %d', [i]));
  end;

  hipFree(d_AP);
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level2_64.Test_ZHPR2_64;
const
  n: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
var
  uplo: rocblas_fill;
  alpha: rocblas_double_complex;
  h_AP, h_AP_expected: array of rocblas_double_complex;
  h_x, h_y: array of rocblas_double_complex;
  d_AP, d_x, d_y: hipDeviceptr_t;
  i, j, k: Int64;
  ap_size: Int64;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  term1, term2, y_conj, alpha_conj, x_conj: rocblas_double_complex;
begin
  uplo := rocblas_fill_upper;
  alpha.x := 1.5; alpha.y := -0.5;
  ap_size := n * (n + 1) div 2;

  SetLength(h_AP, ap_size);
  SetLength(h_AP_expected, ap_size);
  SetLength(h_x, n * incx);
  SetLength(h_y, n * incy);

  // Initialize data
  for i := 0 to ap_size - 1 do begin h_AP[i].x := i+1.2; h_AP[i].y := i-1.8; h_AP_expected[i] := h_AP[i]; end;
  for i := 0 to (n * incx) - 1 do begin h_x[i].x := i+1.1; h_x[i].y := i-0.9; end;
  for i := 0 to (n * incy) - 1 do begin h_y[i].x := i+1.3; h_y[i].y := i-0.7; end;

  // CPU calculation: A := alpha*x*y**H + conj(alpha)*y*x**H + A
  alpha_conj := alpha;
  alpha_conj.y := -alpha_conj.y;
  k := 0;
  for j := 0 to n - 1 do
    for i := 0 to j do
    begin
      // term1 = alpha * x[i] * y[j]**H
      y_conj := h_y[j*incy]; y_conj.y := -y_conj.y;
      term1 := h_x[i*incx];
      ComplexMul(term1, y_conj);
      ComplexMul(term1, alpha);

      // term2 = conj(alpha) * y[i] * x[j]**H
      x_conj := h_x[j*incx]; x_conj.y := -x_conj.y;
      term2 := h_y[i*incy];
      ComplexMul(term2, x_conj);
      ComplexMul(term2, alpha_conj);

      ComplexAdd(h_AP_expected[k], term1);
      ComplexAdd(h_AP_expected[k], term2);

      if i = j then h_AP_expected[k].y := 0.0;
      Inc(k);
    end;

  // GPU execution
  hip_status := hipMalloc(d_AP, ap_size * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x, n * incx * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, n * incy * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_AP, @h_AP[0], ap_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_AP to d_AP', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x, @h_x[0], n * incx * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_x to d_x', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], n * incy * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy h_y to d_y', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zhpr2_64(handle, uplo, n, @alpha, d_x, incx, d_y, incy, d_AP);
  AssertEquals('rocblas_zhpr2_64', Integer(rocblas_status_success), Integer(blas_status));

  hip_status := hipMemcpy(@h_AP[0], d_AP, ap_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d_AP to h_AP', Integer(HIP_SUCCESS), Integer(hip_status));

  // Verification
  for i := 0 to ap_size - 1 do
  begin
    CheckEqualsRelative(h_AP_expected[i].x, h_AP[i].x, 1e-12, Format('Real part mismatch at index %d', [i]));
    CheckEqualsRelative(h_AP_expected[i].y, h_AP[i].y, 1e-12, Format('Imag part mismatch at index %d', [i]));
  end;

  hipFree(d_AP);
  hipFree(d_x);
  hipFree(d_y);
end;

initialization
  RegisterTest(TTestRocBLAS_Level2_64);

end.

