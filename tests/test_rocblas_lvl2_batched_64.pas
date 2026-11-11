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

unit Test_rocblas_lvl2_batched_64;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  { TTestRocBLAS_Level2_batched_64 }

  TTestRocBLAS_Level2_batched_64 = class(TTestRocBLAS_Base)
  published
    { General Matrix-Vector }
    procedure Test_SGEMV_batched_64;
    procedure Test_DGEMV_batched_64;
    procedure Test_CGEMV_batched_64;
    procedure Test_ZGEMV_batched_64;
    procedure Test_SGBMV_batched_64;
    procedure Test_DGBMV_batched_64;
    procedure Test_CGBMV_batched_64;
    procedure Test_ZGBMV_batched_64;

    { Hermitian }
    procedure Test_CHEMV_batched_64;
    procedure Test_ZHEMV_batched_64;
    //procedure Test_CHBMV_batched_64;
    //procedure Test_ZHBMV_batched_64;
    //procedure Test_CHPMV_batched_64;
    //procedure Test_ZHPMV_batched_64;

    { Symmetric }
    procedure Test_SSYMV_batched_64;
    procedure Test_DSYMV_batched_64;
    //procedure Test_CSYMV_batched_64;
    //procedure Test_ZSYMV_batched_64;
    procedure Test_SSBMV_batched_64;
    procedure Test_DSBMV_batched_64;
    //procedure Test_SSPMV_batched_64;
    //procedure Test_DSPMV_batched_64;

    { Triangular }
    procedure Test_STRMV_batched_64;
    procedure Test_DTRMV_batched_64;
    procedure Test_CTRMV_batched_64;
    procedure Test_ZTRMV_batched_64;
    //procedure Test_STBMV_batched_64;
    //procedure Test_DTBMV_batched_64;
    //procedure Test_CTBMV_batched_64;
    //procedure Test_ZTBMV_batched_64;
    procedure Test_STPMV_batched_64;
    procedure Test_DTPMV_batched_64;
    procedure Test_CTPMV_batched_64;
    procedure Test_ZTPMV_batched_64;

    { Triangular Solvers }
    procedure Test_STRSV_batched_64;
    procedure Test_DTRSV_batched_64;
    procedure Test_CTRSV_batched_64;
    procedure Test_ZTRSV_batched_64;
    //procedure Test_STBSV_batched_64;
    //procedure Test_DTBSV_batched_64;
    //procedure Test_CTBSV_batched_64;
    //procedure Test_ZTBSV_batched_64;
    //procedure Test_STPSV_batched_64;
    //procedure Test_DTPSV_batched_64;
    //procedure Test_CTPSV_batched_64;
    //procedure Test_ZTPSV_batched_64;

    { Rank-1 Updates }
    procedure Test_SGER_batched_64;
    procedure Test_DGER_batched_64;
    procedure Test_CGERU_batched_64;
    procedure Test_ZGERU_batched_64;
    procedure Test_CGERC_batched_64;
    procedure Test_ZGERC_batched_64;
    procedure Test_SSYR_batched_64;
    procedure Test_DSYR_batched_64;
    //procedure Test_CSYR_batched_64;
    //procedure Test_ZSYR_batched_64;
    procedure Test_CHER_batched_64;
    procedure Test_ZHER_batched_64;
    procedure Test_SSPR_batched_64;
    procedure Test_DSPR_batched_64;
    //procedure Test_CHPR_batched_64;
    //procedure Test_ZHPR_batched_64;

    { Rank-2 Updates }
    procedure Test_SSYR2_batched_64;
    procedure Test_DSYR2_batched_64;
    //procedure Test_CSYR2_batched_64;
    //procedure Test_ZSYR2_batched_64;
    procedure Test_CHER2_batched_64;
    procedure Test_ZHER2_batched_64;
    //procedure Test_SSPR2_batched_64;
    //procedure Test_DSPR2_batched_64;
    //procedure Test_CHPR2_batched_64;
    //procedure Test_ZHPR2_batched_64;
  end;

implementation

{ TTestRocBLAS_Level2_batched_64 }

procedure TTestRocBLAS_Level2_batched_64.Test_SGEMV_batched_64;
const
  trans = rocblas_operation_none;
  M: Int64 = 10;
  N: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TSingleVector   = array of Single;
  TSingleMatrix   = array of TSingleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: Single;
  h_A_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TSingleMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product: Single;
begin
  lda := M;
  x_size := N * incx;
  y_size := M * incy;
  A_size := lda * N;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha := 1.5;
  h_beta := 2.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);

    for i := 0 to A_size - 1 do h_A_batch[b][i] := Random;
    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;
    for i := 0 to y_size - 1 do h_y_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Single));
    AssertEquals('hipMalloc A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Single));
    AssertEquals('hipMalloc x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(Single));
    AssertEquals('hipMalloc y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(Single), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Single), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(Single), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      dot_product := 0.0;
      for j := 0 to N - 1 do
      begin
        dot_product := dot_product + h_A_batch[b][i + j * lda] * h_x_batch[b][j * incx];
      end;
      h_y_expected[b][i * incy] := h_alpha * dot_product + h_beta * h_y_expected[b][i * incy];
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sgemv_batched_64(handle, trans, M, N, @h_alpha,
    PPSingle(d_A_ptrs), lda, PPSingle(d_x_ptrs), incx, @h_beta,
    PPSingle(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_sgemv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(Single), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
      CheckEquals(h_y_expected[b][i], h_y_result[b][i], 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs);
  hipFree(d_x_ptrs);
  hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]);
    hipFree(h_x_ptrs[b]);
    hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_DGEMV_batched_64;
const
  trans = rocblas_operation_none;
  M: Int64 = 10;
  N: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TDoubleVector   = array of Double;
  TDoubleMatrix   = array of TDoubleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: Double;
  h_A_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TDoubleMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product: Double;
begin
  lda := M;
  x_size := N * incx;
  y_size := M * incy;
  A_size := lda * N;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha := 1.5;
  h_beta := 2.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);

    for i := 0 to A_size - 1 do h_A_batch[b][i] := Random;
    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;
    for i := 0 to y_size - 1 do h_y_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Double));
    AssertEquals('hipMalloc A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Double));
    AssertEquals('hipMalloc x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(Double));
    AssertEquals('hipMalloc y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(Double), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Double), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(Double), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      dot_product := 0.0;
      for j := 0 to N - 1 do
      begin
        dot_product := dot_product + h_A_batch[b][i + j * lda] * h_x_batch[b][j * incx];
      end;
      h_y_expected[b][i * incy] := h_alpha * dot_product + h_beta * h_y_expected[b][i * incy];
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dgemv_batched_64(handle, trans, M, N, @h_alpha,
    PPDouble(d_A_ptrs), lda, PPDouble(d_x_ptrs), incx, @h_beta,
    PPDouble(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_dgemv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(Double), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
      CheckEquals(h_y_expected[b][i], h_y_result[b][i], 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs);
  hipFree(d_x_ptrs);
  hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]);
    hipFree(h_x_ptrs[b]);
    hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_CGEMV_batched_64;
const
  trans = rocblas_operation_none;
  M: Int64 = 10;
  N: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_float_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: rocblas_float_complex;
  h_A_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TComplexMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product, term, beta_y: rocblas_float_complex;
begin
  lda := M;
  x_size := N * incx;
  y_size := M * incy;
  A_size := lda * N;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha.x := 1.5; h_alpha.y := -0.5;
  h_beta.x  := 2.5; h_beta.y  := 1.0;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);

    for i := 0 to A_size - 1 do begin h_A_batch[b][i].x := Random; h_A_batch[b][i].y := Random; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;
    for i := 0 to y_size - 1 do begin h_y_batch[b][i].x := Random; h_y_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_float_complex));
    AssertEquals('hipMalloc A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_float_complex));
    AssertEquals('hipMalloc x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(rocblas_float_complex));
    AssertEquals('hipMalloc y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      dot_product.x := 0; dot_product.y := 0;
      for j := 0 to N - 1 do
      begin
        term := h_A_batch[b][i + j * lda];
        ComplexMul(term, h_x_batch[b][j * incx]);
        ComplexAdd(dot_product, term);
      end;
      ComplexMul(dot_product, h_alpha);
      beta_y := h_y_expected[b][i * incy];
      ComplexMul(beta_y, h_beta);
      ComplexAdd(dot_product, beta_y);
      h_y_expected[b][i * incy] := dot_product;
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgemv_batched_64(handle, trans, M, N, @h_alpha,
    PPProcfloat_complex(d_A_ptrs), lda, PPProcfloat_complex(d_x_ptrs), incx, @h_beta,
    PPProcfloat_complex(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_cgemv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
    begin
      CheckEquals(h_y_expected[b][i].x, h_y_result[b][i].x, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_y_expected[b][i].y, h_y_result[b][i].y, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs);
  hipFree(d_x_ptrs);
  hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]);
    hipFree(h_x_ptrs[b]);
    hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_ZGEMV_batched_64;
const
  trans = rocblas_operation_none;
  M: Int64 = 10;
  N: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_double_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: rocblas_double_complex;
  h_A_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TComplexMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product, term, beta_y: rocblas_double_complex;
begin
  lda := M;
  x_size := N * incx;
  y_size := M * incy;
  A_size := lda * N;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha.x := 1.5; h_alpha.y := -0.5;
  h_beta.x  := 2.5; h_beta.y  := 1.0;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);

    for i := 0 to A_size - 1 do begin h_A_batch[b][i].x := Random; h_A_batch[b][i].y := Random; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;
    for i := 0 to y_size - 1 do begin h_y_batch[b][i].x := Random; h_y_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_double_complex));
    AssertEquals('hipMalloc A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_double_complex));
    AssertEquals('hipMalloc x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(rocblas_double_complex));
    AssertEquals('hipMalloc y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      dot_product.x := 0; dot_product.y := 0;
      for j := 0 to N - 1 do
      begin
        term := h_A_batch[b][i + j * lda];
        ComplexMul(term, h_x_batch[b][j * incx]);
        ComplexAdd(dot_product, term);
      end;
      ComplexMul(dot_product, h_alpha);
      beta_y := h_y_expected[b][i * incy];
      ComplexMul(beta_y, h_beta);
      ComplexAdd(dot_product, beta_y);
      h_y_expected[b][i * incy] := dot_product;
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgemv_batched_64(handle, trans, M, N, @h_alpha,
    PPProcdouble_complex(d_A_ptrs), lda, PPProcdouble_complex(d_x_ptrs), incx, @h_beta,
    PPProcdouble_complex(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_zgemv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
    begin
      CheckEquals(h_y_expected[b][i].x, h_y_result[b][i].x, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_y_expected[b][i].y, h_y_result[b][i].y, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs);
  hipFree(d_x_ptrs);
  hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]);
    hipFree(h_x_ptrs[b]);
    hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_SGBMV_batched_64;
const
  trans = rocblas_operation_none;
  M: Int64 = 10;
  N: Int64 = 10;
  kl: Int64 = 2;
  ku: Int64 = 2;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TSingleVector   = array of Single;
  TSingleMatrix   = array of TSingleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: Single;
  h_A_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TSingleMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product: Single;
begin
  lda := kl + ku + 1;
  A_size := lda * N;
  x_size := N * incx;
  y_size := M * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha := 1.5;
  h_beta := 2.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);

    for i := 0 to A_size - 1 do h_A_batch[b][i] := Random;
    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;
    for i := 0 to y_size - 1 do h_y_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Single));
    AssertEquals('hipMalloc A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Single));
    AssertEquals('hipMalloc x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(Single));
    AssertEquals('hipMalloc y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(Single), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Single), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(Single), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation for verification
  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      dot_product := 0.0;
      for j := Max(0, i - kl) to Min(N - 1, i + ku) do
      begin
        dot_product := dot_product + h_A_batch[b][(ku + i - j) + j * lda] * h_x_batch[b][j * incx];
      end;
      h_y_expected[b][i * incy] := h_alpha * dot_product + h_beta * h_y_expected[b][i * incy];
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sgbmv_batched_64(handle, trans, M, N, kl, ku, @h_alpha,
    PPSingle(d_A_ptrs), lda, PPSingle(d_x_ptrs), incx, @h_beta,
    PPSingle(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_sgbmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(Single), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
      CheckEquals(h_y_expected[b][i], h_y_result[b][i], 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs);
  hipFree(d_x_ptrs);
  hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]);
    hipFree(h_x_ptrs[b]);
    hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_DGBMV_batched_64;
const
  trans = rocblas_operation_none;
  M: Int64 = 10;
  N: Int64 = 10;
  kl: Int64 = 2;
  ku: Int64 = 2;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TDoubleVector   = array of Double;
  TDoubleMatrix   = array of TDoubleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: Double;
  h_A_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TDoubleMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product: Double;
begin
  lda := kl + ku + 1;
  A_size := lda * N;
  x_size := N * incx;
  y_size := M * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha := 1.5;
  h_beta := 2.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);

    for i := 0 to A_size - 1 do h_A_batch[b][i] := Random;
    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;
    for i := 0 to y_size - 1 do h_y_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Double));
    AssertEquals('hipMalloc A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Double));
    AssertEquals('hipMalloc x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(Double));
    AssertEquals('hipMalloc y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(Double), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Double), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(Double), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation for verification
  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      dot_product := 0.0;
      for j := Max(0, i - kl) to Min(N - 1, i + ku) do
      begin
        dot_product := dot_product + h_A_batch[b][(ku + i - j) + j * lda] * h_x_batch[b][j * incx];
      end;
      h_y_expected[b][i * incy] := h_alpha * dot_product + h_beta * h_y_expected[b][i * incy];
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dgbmv_batched_64(handle, trans, M, N, kl, ku, @h_alpha,
    PPDouble(d_A_ptrs), lda, PPDouble(d_x_ptrs), incx, @h_beta,
    PPDouble(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_dgbmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(Double), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
      CheckEquals(h_y_expected[b][i], h_y_result[b][i], 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs);
  hipFree(d_x_ptrs);
  hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]);
    hipFree(h_x_ptrs[b]);
    hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_CGBMV_batched_64;
const
  trans = rocblas_operation_none;
  M: Int64 = 10;
  N: Int64 = 10;
  kl: Int64 = 2;
  ku: Int64 = 2;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_float_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: rocblas_float_complex;
  h_A_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TComplexMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product, term, beta_y: rocblas_float_complex;
begin
  lda := kl + ku + 1;
  A_size := lda * N;
  x_size := N * incx;
  y_size := M * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha.x := 1.5; h_alpha.y := -0.5;
  h_beta.x  := 2.5; h_beta.y  := 1.0;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);

    for i := 0 to A_size - 1 do begin h_A_batch[b][i].x := Random; h_A_batch[b][i].y := Random; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;
    for i := 0 to y_size - 1 do begin h_y_batch[b][i].x := Random; h_y_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_float_complex));
    AssertEquals('hipMalloc A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_float_complex));
    AssertEquals('hipMalloc x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(rocblas_float_complex));
    AssertEquals('hipMalloc y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation for verification
  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      dot_product.x := 0; dot_product.y := 0;
      for j := Max(0, i - kl) to Min(N - 1, i + ku) do
      begin
        term := h_A_batch[b][(ku + i - j) + j * lda];
        ComplexMul(term, h_x_batch[b][j * incx]);
        ComplexAdd(dot_product, term);
      end;
      ComplexMul(dot_product, h_alpha);
      beta_y := h_y_expected[b][i * incy];
      ComplexMul(beta_y, h_beta);
      ComplexAdd(dot_product, beta_y);
      h_y_expected[b][i * incy] := dot_product;
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgbmv_batched_64(handle, trans, M, N, kl, ku, @h_alpha,
    PPProcfloat_complex(d_A_ptrs), lda, PPProcfloat_complex(d_x_ptrs), incx, @h_beta,
    PPProcfloat_complex(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_cgbmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
    begin
      CheckEquals(h_y_expected[b][i].x, h_y_result[b][i].x, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_y_expected[b][i].y, h_y_result[b][i].y, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs);
  hipFree(d_x_ptrs);
  hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]);
    hipFree(h_x_ptrs[b]);
    hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_ZGBMV_batched_64;
const
  trans = rocblas_operation_none;
  M: Int64 = 10;
  N: Int64 = 10;
  kl: Int64 = 2;
  ku: Int64 = 2;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_double_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: rocblas_double_complex;
  h_A_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TComplexMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product, term, beta_y: rocblas_double_complex;
begin
  lda := kl + ku + 1;
  A_size := lda * N;
  x_size := N * incx;
  y_size := M * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha.x := 1.5; h_alpha.y := -0.5;
  h_beta.x  := 2.5; h_beta.y  := 1.0;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);

    for i := 0 to A_size - 1 do begin h_A_batch[b][i].x := Random; h_A_batch[b][i].y := Random; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;
    for i := 0 to y_size - 1 do begin h_y_batch[b][i].x := Random; h_y_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_double_complex));
    AssertEquals('hipMalloc A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_double_complex));
    AssertEquals('hipMalloc x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(rocblas_double_complex));
    AssertEquals('hipMalloc y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_A_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_x_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D d_y_ptrs failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation for verification
  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      dot_product.x := 0; dot_product.y := 0;
      for j := Max(0, i - kl) to Min(N - 1, i + ku) do
      begin
        term := h_A_batch[b][(ku + i - j) + j * lda];
        ComplexMul(term, h_x_batch[b][j * incx]);
        ComplexAdd(dot_product, term);
      end;
      ComplexMul(dot_product, h_alpha);
      beta_y := h_y_expected[b][i * incy];
      ComplexMul(beta_y, h_beta);
      ComplexAdd(dot_product, beta_y);
      h_y_expected[b][i * incy] := dot_product;
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgbmv_batched_64(handle, trans, M, N, kl, ku, @h_alpha,
    PPProcdouble_complex(d_A_ptrs), lda, PPProcdouble_complex(d_x_ptrs), incx, @h_beta,
    PPProcdouble_complex(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_zgbmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
    begin
      CheckEquals(h_y_expected[b][i].x, h_y_result[b][i].x, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_y_expected[b][i].y, h_y_result[b][i].y, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs);
  hipFree(d_x_ptrs);
  hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]);
    hipFree(h_x_ptrs[b]);
    hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_CHEMV_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_float_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: rocblas_float_complex;
  h_A_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TComplexMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product, term, beta_y: rocblas_float_complex;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;
  y_size := N * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha.x := 1.5; h_alpha.y := -0.5;
  h_beta.x  := 2.5; h_beta.y  := 1.0;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);

    // 1. Fill the upper triangle (and diagonal) which is what rocBLAS reads
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        if i = j then
        begin
          h_A_batch[b][i + j * lda].x := Random;
          h_A_batch[b][i + j * lda].y := 0; // Diagonal must be real
        end
        else
        begin
          h_A_batch[b][i + j * lda].x := Random;
          h_A_batch[b][i + j * lda].y := Random;
        end;
      end;
    // 2. Fill the lower triangle for the CPU reference calculation
    for j := 0 to N - 1 do
      for i := j + 1 to N - 1 do
      begin
          h_A_batch[b][i + j * lda].x :=  h_A_batch[b][j + i * lda].x;
          h_A_batch[b][i + j * lda].y := -h_A_batch[b][j + i * lda].y; // Conjugate
      end;

    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;
    for i := 0 to y_size - 1 do begin h_y_batch[b][i].x := Random; h_y_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_float_complex));
    AssertEquals('hipMalloc A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_float_complex));
    AssertEquals('hipMalloc x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(rocblas_float_complex));
    AssertEquals('hipMalloc y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to N - 1 do
    begin
      dot_product.x := 0; dot_product.y := 0;
      for j := 0 to N - 1 do
      begin
        term := h_A_batch[b][i + j * lda];
        ComplexMul(term, h_x_batch[b][j * incx]);
        ComplexAdd(dot_product, term);
      end;
      ComplexMul(dot_product, h_alpha);
      beta_y := h_y_expected[b][i * incy];
      ComplexMul(beta_y, h_beta);
      ComplexAdd(dot_product, beta_y);
      h_y_expected[b][i * incy] := dot_product;
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_chemv_batched_64(handle, uplo, N, @h_alpha,
    PPProcfloat_complex(d_A_ptrs), lda, PPProcfloat_complex(d_x_ptrs), incx, @h_beta,
    PPProcfloat_complex(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_chemv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
    begin
      CheckEquals(h_y_expected[b][i].x, h_y_result[b][i].x, 1e-3, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_y_expected[b][i].y, h_y_result[b][i].y, 1e-3, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_ZHEMV_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_double_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: rocblas_double_complex;
  h_A_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TComplexMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product, term, beta_y: rocblas_double_complex;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;
  y_size := N * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha.x := 1.5; h_alpha.y := -0.5;
  h_beta.x  := 2.5; h_beta.y  := 1.0;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);

    // 1. Fill the upper triangle (and diagonal) which is what rocBLAS reads
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        if i = j then
        begin
          h_A_batch[b][i + j * lda].x := Random;
          h_A_batch[b][i + j * lda].y := 0; // Diagonal must be real
        end
        else
        begin
          h_A_batch[b][i + j * lda].x := Random;
          h_A_batch[b][i + j * lda].y := Random;
        end;
      end;
    // 2. Fill the lower triangle for the CPU reference calculation
    for j := 0 to N - 1 do
      for i := j + 1 to N - 1 do
      begin
          h_A_batch[b][i + j * lda].x :=  h_A_batch[b][j + i * lda].x;
          h_A_batch[b][i + j * lda].y := -h_A_batch[b][j + i * lda].y; // Conjugate
      end;

    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;
    for i := 0 to y_size - 1 do begin h_y_batch[b][i].x := Random; h_y_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_double_complex));
    AssertEquals('hipMalloc A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_double_complex));
    AssertEquals('hipMalloc x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(rocblas_double_complex));
    AssertEquals('hipMalloc y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to N - 1 do
    begin
      dot_product.x := 0; dot_product.y := 0;
      for j := 0 to N - 1 do
      begin
        term := h_A_batch[b][i + j * lda];
        ComplexMul(term, h_x_batch[b][j * incx]);
        ComplexAdd(dot_product, term);
      end;
      ComplexMul(dot_product, h_alpha);
      beta_y := h_y_expected[b][i * incy];
      ComplexMul(beta_y, h_beta);
      ComplexAdd(dot_product, beta_y);
      h_y_expected[b][i * incy] := dot_product;
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zhemv_batched_64(handle, uplo, N, @h_alpha,
    PPProcdouble_complex(d_A_ptrs), lda, PPProcdouble_complex(d_x_ptrs), incx, @h_beta,
    PPProcdouble_complex(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_zhemv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
    begin
      CheckEquals(h_y_expected[b][i].x, h_y_result[b][i].x, 1e-11, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_y_expected[b][i].y, h_y_result[b][i].y, 1e-11, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

//procedure TTestRocBLAS_Level2_batched_64.Test_CHBMV_batched_64;
//begin
//  WriteLn('Function CHBMV_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_ZHBMV_batched_64;
//begin
//  WriteLn('Function ZHBMV_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_CHPMV_batched_64;
//begin
//  WriteLn('Function CHPMV_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_ZHPMV_batched_64;
//begin
//  WriteLn('Function ZHPMV_batched_64 does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_batched_64.Test_SSYMV_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TSingleVector   = array of Single;
  TSingleMatrix   = array of TSingleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: Single;
  h_A_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TSingleMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product: Single;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;
  y_size := N * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha := 1.5;
  h_beta := 2.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);

    // 1. Fill the upper triangle (and diagonal)
    for j := 0 to N - 1 do
      for i := 0 to j do
        h_A_batch[b][i + j * lda] := Random;
    // 2. Fill the lower triangle by transposing the upper triangle
    for j := 0 to N - 1 do
      for i := j + 1 to N - 1 do
        h_A_batch[b][i + j * lda] := h_A_batch[b][j + i * lda];

    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;
    for i := 0 to y_size - 1 do h_y_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Single));
    AssertEquals('hipMalloc A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Single));
    AssertEquals('hipMalloc x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(Single));
    AssertEquals('hipMalloc y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(Single), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Single), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(Single), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to N - 1 do
    begin
      dot_product := 0.0;
      for j := 0 to N - 1 do
        dot_product := dot_product + h_A_batch[b][i + j * lda] * h_x_batch[b][j * incx];
      h_y_expected[b][i * incy] := h_alpha * dot_product + h_beta * h_y_expected[b][i * incy];
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_ssymv_batched_64(handle, uplo, N, @h_alpha,
    PPSingle(d_A_ptrs), lda, PPSingle(d_x_ptrs), incx, @h_beta,
    PPSingle(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_ssymv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(Single), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
      CheckEquals(h_y_expected[b][i], h_y_result[b][i], 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_DSYMV_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TDoubleVector   = array of Double;
  TDoubleMatrix   = array of TDoubleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: Double;
  h_A_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TDoubleMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product: Double;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;
  y_size := N * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha := 1.5;
  h_beta := 2.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);

    // Generate a full symmetric matrix for the host-side calculation
    for j := 0 to N - 1 do
      for i := 0 to j do
        h_A_batch[b][i + j * lda] := Random;
    for j := 0 to N - 1 do
      for i := j + 1 to N - 1 do
        h_A_batch[b][i + j * lda] := h_A_batch[b][j + i * lda];

    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;
    for i := 0 to y_size - 1 do h_y_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Double));
    AssertEquals('hipMalloc A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Double));
    AssertEquals('hipMalloc x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(Double));
    AssertEquals('hipMalloc y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));

    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(Double), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D A['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Double), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(Double), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to N - 1 do
    begin
      dot_product := 0.0;
      for j := 0 to N - 1 do
        dot_product := dot_product + h_A_batch[b][i + j * lda] * h_x_batch[b][j * incx];
      h_y_expected[b][i * incy] := h_alpha * dot_product + h_beta * h_y_expected[b][i * incy];
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dsymv_batched_64(handle, uplo, N, @h_alpha,
    PPDouble(d_A_ptrs), lda, PPDouble(d_x_ptrs), incx, @h_beta,
    PPDouble(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_dsymv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(Double), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
      CheckEquals(h_y_expected[b][i], h_y_result[b][i], 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

//procedure TTestRocBLAS_Level2_batched_64.Test_CSYMV_batched_64;
//begin
//  WriteLn('Function CSYMV_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_ZSYMV_batched_64;
//begin
//  WriteLn('Function ZSYMV_batched_64 does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_batched_64.Test_SSBMV_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  K: Int64 = 2; // Number of super-diagonals
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TSingleVector   = array of Single;
  TSingleMatrix   = array of TSingleVector;
  TFullMatrix     = array of array of Single;
  TFullMatrixBatch = array of TFullMatrix;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: Single;
  h_A_band_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TSingleMatrix;
  h_A_full_batch: TFullMatrixBatch;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_band_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product: Single;
begin
  lda := K + 1;
  A_band_size := lda * N;
  x_size := N * incx;
  y_size := N * incy;

  SetLength(h_A_band_batch, batch_count);
  SetLength(h_A_full_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha := 1.5;
  h_beta := 2.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_band_batch[b], A_band_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);
    SetLength(h_A_full_batch[b], N, N);

    // 1. Generate a full symmetric matrix for CPU reference
    for j := 0 to N - 1 do
      for i := 0 to j do
        h_A_full_batch[b, i, j] := Random;
    for j := 0 to N - 1 do
      for i := j + 1 to N - 1 do
        h_A_full_batch[b, i, j] := h_A_full_batch[b, j, i];

    // 2. Convert the full matrix to the upper banded format for rocBLAS
    FillChar(h_A_band_batch[b][0], A_band_size * SizeOf(Single), 0);
    for j := 0 to N - 1 do
      for i := Max(0, j - K) to j do
        h_A_band_batch[b][(K + i - j) + j * lda] := h_A_full_batch[b, i, j];

    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;
    for i := 0 to y_size - 1 do h_y_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_band_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_band_batch[b][0], A_band_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation using the full matrix but respecting the band
  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to N - 1 do
    begin
      dot_product := 0.0;
      for j := Max(0, i - K) to Min(N - 1, i + K) do
        dot_product := dot_product + h_A_full_batch[b, i, j] * h_x_batch[b][j * incx];
      h_y_expected[b][i * incy] := h_alpha * dot_product + h_beta * h_y_expected[b][i * incy];
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_ssbmv_batched_64(handle, uplo, N, K, @h_alpha,
    PPSingle(d_A_ptrs), lda, PPSingle(d_x_ptrs), incx, @h_beta,
    PPSingle(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_ssbmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(Single), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
      CheckEquals(h_y_expected[b][i], h_y_result[b][i], 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_DSBMV_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  K: Int64 = 2; // Number of super-diagonals
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TDoubleVector   = array of Double;
  TDoubleMatrix   = array of TDoubleVector;
  TFullMatrix     = array of array of Double;
  TFullMatrixBatch = array of TFullMatrix;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha, h_beta: Double;
  h_A_band_batch, h_x_batch, h_y_batch, h_y_result, h_y_expected: TDoubleMatrix;
  h_A_full_batch: TFullMatrixBatch;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_band_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  dot_product: Double;
begin
  lda := K + 1;
  A_band_size := lda * N;
  x_size := N * incx;
  y_size := N * incy;

  SetLength(h_A_band_batch, batch_count);
  SetLength(h_A_full_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_y_result, batch_count);
  SetLength(h_y_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha := 1.5;
  h_beta := 2.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_band_batch[b], A_band_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_y_result[b], y_size);
    SetLength(h_y_expected[b], y_size);
    SetLength(h_A_full_batch[b], N, N);

    // 1. Generate a full symmetric matrix for CPU reference
    for j := 0 to N - 1 do
      for i := 0 to j do
        h_A_full_batch[b, i, j] := Random;
    for j := 0 to N - 1 do
      for i := j + 1 to N - 1 do
        h_A_full_batch[b, i, j] := h_A_full_batch[b, j, i];

    // 2. Convert the full matrix to the banded format for rocBLAS
    FillChar(h_A_band_batch[b][0], A_band_size * SizeOf(Double), 0);
     if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := Max(0, j - K) to j do
          h_A_band_batch[b][(K + i - j) + j * lda] := h_A_full_batch[b, i, j];
    end
    else // rocblas_fill_lower
    begin
       for j := 0 to N - 1 do
        for i := j to Min(N-1, j + K) do
          h_A_band_batch[b][(i - j) + j * lda] := h_A_full_batch[b, i, j];
    end;

    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;
    for i := 0 to y_size - 1 do h_y_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_band_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_band_batch[b][0], A_band_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  for b := 0 to batch_count - 1 do
  begin
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to N - 1 do
    begin
      dot_product := 0.0;
      for j := Max(0, i - K) to Min(N - 1, i + K) do
        dot_product := dot_product + h_A_full_batch[b, i, j] * h_x_batch[b][j * incx];
      h_y_expected[b][i * incy] := h_alpha * dot_product + h_beta * h_y_expected[b][i * incy];
    end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dsbmv_batched_64(handle, uplo, N, K, @h_alpha,
    PPDouble(d_A_ptrs), lda, PPDouble(d_x_ptrs), incx, @h_beta,
    PPDouble(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_dsbmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_y_ptrs[b], y_size * SizeOf(Double), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H y['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to y_size - 1 do
      CheckEquals(h_y_expected[b][i], h_y_result[b][i], 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

//procedure TTestRocBLAS_Level2_batched_64.Test_SSPMV_batched_64;
//begin
//  WriteLn('Function SSPMV_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_DSPMV_batched_64;
//begin
//  WriteLn('Function DSPMV_batched_64 does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_batched_64.Test_STRMV_batched_64;
const
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TSingleVector   = array of Single;
  TSingleMatrix   = array of TSingleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_A_batch, h_x_batch, h_x_result, h_x_expected: TSingleMatrix;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  temp_x: TSingleVector;
  sum: Single;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_x_result, batch_count);
  SetLength(h_x_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_x_result[b], x_size);
    SetLength(h_x_expected[b], x_size);

    // Generate triangular matrix
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
          h_A_batch[b][i + j * lda] := Random
        else
          h_A_batch[b][i + j * lda] := 0; // Zero out lower part for clarity

    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    temp_x := h_x_batch[b]; // Use a copy for calculation
    for i := 0 to N - 1 do
    begin
      sum := 0.0;
      for j := i to N - 1 do // Upper triangular
        sum := sum + h_A_batch[b][i + j * lda] * temp_x[j * incx];
      h_x_expected[b][i * incx] := sum;
    end;
  end;

  blas_status := rocblas_strmv_batched_64(handle, uplo, transA, diag, N,
    PPSingle(d_A_ptrs), lda, PPSingle(d_x_ptrs), incx, batch_count);
  AssertEquals('rocblas_strmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_x_result[b][0], h_x_ptrs[b], x_size * SizeOf(Single), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to x_size - 1 do
      CheckEquals(h_x_expected[b][i], h_x_result[b][i], 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_DTRMV_batched_64;
const
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TDoubleVector   = array of Double;
  TDoubleMatrix   = array of TDoubleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_A_batch, h_x_batch, h_x_result, h_x_expected: TDoubleMatrix;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  temp_x: TDoubleVector;
  sum: Double;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_x_result, batch_count);
  SetLength(h_x_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_x_result[b], x_size);
    SetLength(h_x_expected[b], x_size);

    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then h_A_batch[b][i + j * lda] := Random else h_A_batch[b][i + j * lda] := 0;
    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    temp_x := h_x_batch[b];
    for i := 0 to N - 1 do
    begin
      sum := 0.0;
      for j := i to N - 1 do
        sum := sum + h_A_batch[b][i + j * lda] * temp_x[j * incx];
      h_x_expected[b][i * incx] := sum;
    end;
  end;

  blas_status := rocblas_dtrmv_batched_64(handle, uplo, transA, diag, N,
    PPDouble(d_A_ptrs), lda, PPDouble(d_x_ptrs), incx, batch_count);
  AssertEquals('rocblas_dtrmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_x_result[b][0], h_x_ptrs[b], x_size * SizeOf(Double), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to x_size - 1 do
      CheckEquals(h_x_expected[b][i], h_x_result[b][i], 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_CTRMV_batched_64;
const
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_float_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_A_batch, h_x_batch, h_x_result, h_x_expected: TComplexMatrix;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  temp_x: TComplexVector;
  sum, term: rocblas_float_complex;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_x_result, batch_count);
  SetLength(h_x_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_x_result[b], x_size);
    SetLength(h_x_expected[b], x_size);

    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then begin h_A_batch[b][i + j * lda].x := Random; h_A_batch[b][i + j * lda].y := Random; end
        else begin h_A_batch[b][i + j * lda].x := 0; h_A_batch[b][i + j * lda].y := 0; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    temp_x := h_x_batch[b];
    for i := 0 to N - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for j := i to N - 1 do
      begin
        term := h_A_batch[b][i + j * lda];
        ComplexMul(term, temp_x[j * incx]);
        ComplexAdd(sum, term);
      end;
      h_x_expected[b][i * incx] := sum;
    end;
  end;

  blas_status := rocblas_ctrmv_batched_64(handle, uplo, transA, diag, N,
    PPProcfloat_complex(d_A_ptrs), lda, PPProcfloat_complex(d_x_ptrs), incx, batch_count);
  AssertEquals('rocblas_ctrmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_x_result[b][0], h_x_ptrs[b], x_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to x_size - 1 do
    begin
      CheckEquals(h_x_expected[b][i].x, h_x_result[b][i].x, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_x_expected[b][i].y, h_x_result[b][i].y, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_ZTRMV_batched_64;
const
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_double_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_A_batch, h_x_batch, h_x_result, h_x_expected: TComplexMatrix;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  temp_x: TComplexVector;
  sum, term: rocblas_double_complex;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_x_result, batch_count);
  SetLength(h_x_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_x_result[b], x_size);
    SetLength(h_x_expected[b], x_size);

    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then begin h_A_batch[b][i + j * lda].x := Random; h_A_batch[b][i + j * lda].y := Random; end
        else begin h_A_batch[b][i + j * lda].x := 0; h_A_batch[b][i + j * lda].y := 0; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    temp_x := h_x_batch[b];
    for i := 0 to N - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for j := i to N - 1 do
      begin
        term := h_A_batch[b][i + j * lda];
        ComplexMul(term, temp_x[j * incx]);
        ComplexAdd(sum, term);
      end;
      h_x_expected[b][i * incx] := sum;
    end;
  end;

  blas_status := rocblas_ztrmv_batched_64(handle, uplo, transA, diag, N,
    PPProcdouble_complex(d_A_ptrs), lda, PPProcdouble_complex(d_x_ptrs), incx, batch_count);
  AssertEquals('rocblas_ztrmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_x_result[b][0], h_x_ptrs[b], x_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H x['+IntToStr(b)+'] failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to x_size - 1 do
    begin
      CheckEquals(h_x_expected[b][i].x, h_x_result[b][i].x, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_x_expected[b][i].y, h_x_result[b][i].y, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

//procedure TTestRocBLAS_Level2_batched_64.Test_STBMV_batched_64;
//begin
//  WriteLn('Function rocblas_stbmv_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_DTBMV_batched_64;
//begin
//  WriteLn('Function rocblas_dtbmv_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_CTBMV_batched_64;
//begin
//  WriteLn('Function rocblas_ctbmv_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_ZTBMV_batched_64;
//begin
//  WriteLn('Function rocblas_ztbmv_batched_64 does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_batched_64.Test_STPMV_batched_64;
const
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TSingleVector   = array of Single;
  TSingleMatrix   = array of TSingleVector;
  TFullMatrix     = array of array of Single;
  TFullMatrixBatch = array of TFullMatrix;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_A_packed_batch, h_x_batch, h_x_result, h_x_expected: TSingleMatrix;
  h_A_full_batch: TFullMatrixBatch;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  A_packed_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  temp_x: TSingleVector;
  sum: Single;
begin
  A_packed_size := (N * (N + 1)) div 2;
  x_size := N * incx;

  SetLength(h_A_packed_batch, batch_count);
  SetLength(h_A_full_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_x_result, batch_count);
  SetLength(h_x_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_packed_batch[b], A_packed_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_x_result[b], x_size);
    SetLength(h_x_expected[b], x_size);
    SetLength(h_A_full_batch[b], N, N);

    // 1. Generate a full upper triangular matrix for reference
    for j := 0 to N - 1 do
      for i := 0 to j do
        h_A_full_batch[b, i, j] := Random;

    // 2. Convert to packed format for GPU (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        h_A_packed_batch[b][k] := h_A_full_batch[b, i, j];
        Inc(k);
      end;

    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_packed_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_packed_batch[b][0], A_packed_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation using full matrix
  for b := 0 to batch_count - 1 do
  begin
    temp_x := h_x_batch[b];
    for i := 0 to N - 1 do
    begin
      sum := 0.0;
      for j := i to N - 1 do // Upper triangular
        sum := sum + h_A_full_batch[b, i, j] * temp_x[j * incx];
      h_x_expected[b][i * incx] := sum;
    end;
  end;

  blas_status := rocblas_stpmv_batched_64(handle, uplo, transA, diag, N,
    PPSingle(d_A_ptrs), PPSingle(d_x_ptrs), incx, batch_count);
  AssertEquals('rocblas_stpmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_x_result[b][0], h_x_ptrs[b], x_size * SizeOf(Single), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to x_size - 1 do
      CheckEquals(h_x_expected[b][i], h_x_result[b][i], 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_DTPMV_batched_64;
const
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TDoubleVector   = array of Double;
  TDoubleMatrix   = array of TDoubleVector;
  TFullMatrix     = array of array of Double;
  TFullMatrixBatch = array of TFullMatrix;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_A_packed_batch, h_x_batch, h_x_result, h_x_expected: TDoubleMatrix;
  h_A_full_batch: TFullMatrixBatch;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  A_packed_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  temp_x: TDoubleVector;
  sum: Double;
begin
  A_packed_size := (N * (N + 1)) div 2;
  x_size := N * incx;

  SetLength(h_A_packed_batch, batch_count);
  SetLength(h_A_full_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_x_result, batch_count);
  SetLength(h_x_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_packed_batch[b], A_packed_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_x_result[b], x_size);
    SetLength(h_x_expected[b], x_size);
    SetLength(h_A_full_batch[b], N, N);

    for j := 0 to N - 1 do
      for i := 0 to j do
        h_A_full_batch[b, i, j] := Random;

    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        h_A_packed_batch[b][k] := h_A_full_batch[b, i, j];
        Inc(k);
      end;

    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_packed_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_packed_batch[b][0], A_packed_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    temp_x := h_x_batch[b];
    for i := 0 to N - 1 do
    begin
      sum := 0.0;
      for j := i to N - 1 do
        sum := sum + h_A_full_batch[b, i, j] * temp_x[j * incx];
      h_x_expected[b][i * incx] := sum;
    end;
  end;

  blas_status := rocblas_dtpmv_batched_64(handle, uplo, transA, diag, N,
    PPDouble(d_A_ptrs), PPDouble(d_x_ptrs), incx, batch_count);
  AssertEquals('rocblas_dtpmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_x_result[b][0], h_x_ptrs[b], x_size * SizeOf(Double), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to x_size - 1 do
      CheckEquals(h_x_expected[b][i], h_x_result[b][i], 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_CTPMV_batched_64;
const
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_float_complex;
  TComplexMatrix  = array of TComplexVector;
  TFullMatrix     = array of array of rocblas_float_complex;
  TFullMatrixBatch = array of TFullMatrix;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_A_packed_batch, h_x_batch, h_x_result, h_x_expected: TComplexMatrix;
  h_A_full_batch: TFullMatrixBatch;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  A_packed_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  temp_x: TComplexVector;
  sum, term: rocblas_float_complex;
begin
  A_packed_size := (N * (N + 1)) div 2;
  x_size := N * incx;

  SetLength(h_A_packed_batch, batch_count);
  SetLength(h_A_full_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_x_result, batch_count);
  SetLength(h_x_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_packed_batch[b], A_packed_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_x_result[b], x_size);
    SetLength(h_x_expected[b], x_size);
    SetLength(h_A_full_batch[b], N, N);

    for j := 0 to N - 1 do
      for i := 0 to j do begin h_A_full_batch[b, i, j].x := Random; h_A_full_batch[b, i, j].y := Random; end;

    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        h_A_packed_batch[b][k] := h_A_full_batch[b, i, j];
        Inc(k);
      end;

    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_packed_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_packed_batch[b][0], A_packed_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    temp_x := h_x_batch[b];
    for i := 0 to N - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for j := i to N - 1 do
      begin
        term := h_A_full_batch[b, i, j];
        ComplexMul(term, temp_x[j * incx]);
        ComplexAdd(sum, term);
      end;
      h_x_expected[b][i * incx] := sum;
    end;
  end;

  blas_status := rocblas_ctpmv_batched_64(handle, uplo, transA, diag, N,
    PPProcfloat_complex(d_A_ptrs), PPProcfloat_complex(d_x_ptrs), incx, batch_count);
  AssertEquals('rocblas_ctpmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_x_result[b][0], h_x_ptrs[b], x_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to x_size - 1 do
    begin
      CheckEquals(h_x_expected[b][i].x, h_x_result[b][i].x, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_x_expected[b][i].y, h_x_result[b][i].y, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_ZTPMV_batched_64;
const
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_double_complex;
  TComplexMatrix  = array of TComplexVector;
  TFullMatrix     = array of array of rocblas_double_complex;
  TFullMatrixBatch = array of TFullMatrix;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_A_packed_batch, h_x_batch, h_x_result, h_x_expected: TComplexMatrix;
  h_A_full_batch: TFullMatrixBatch;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  A_packed_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  temp_x: TComplexVector;
  sum, term: rocblas_double_complex;
begin
  A_packed_size := (N * (N + 1)) div 2;
  x_size := N * incx;

  SetLength(h_A_packed_batch, batch_count);
  SetLength(h_A_full_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_x_result, batch_count);
  SetLength(h_x_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_packed_batch[b], A_packed_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_x_result[b], x_size);
    SetLength(h_x_expected[b], x_size);
    SetLength(h_A_full_batch[b], N, N);

    for j := 0 to N - 1 do
      for i := 0 to j do begin h_A_full_batch[b, i, j].x := Random; h_A_full_batch[b, i, j].y := Random; end;

    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        h_A_packed_batch[b][k] := h_A_full_batch[b, i, j];
        Inc(k);
      end;

    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_packed_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_packed_batch[b][0], A_packed_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    temp_x := h_x_batch[b];
    for i := 0 to N - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for j := i to N - 1 do
      begin
        term := h_A_full_batch[b, i, j];
        ComplexMul(term, temp_x[j * incx]);
        ComplexAdd(sum, term);
      end;
      h_x_expected[b][i * incx] := sum;
    end;
  end;

  blas_status := rocblas_ztpmv_batched_64(handle, uplo, transA, diag, N,
    PPProcdouble_complex(d_A_ptrs), PPProcdouble_complex(d_x_ptrs), incx, batch_count);
  AssertEquals('rocblas_ztpmv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_x_result[b][0], h_x_ptrs[b], x_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to x_size - 1 do
    begin
      CheckEquals(h_x_expected[b][i].x, h_x_result[b][i].x, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_x_expected[b][i].y, h_x_result[b][i].y, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_STRSV_batched_64;
const
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TSingleVector   = array of Single;
  TSingleMatrix   = array of TSingleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_A_batch, h_x_original_batch, h_b_input_batch, h_x_result_batch: TSingleMatrix;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  sum: Single;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_original_batch, batch_count);
  SetLength(h_b_input_batch, batch_count);
  SetLength(h_x_result_batch, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_original_batch[b], x_size);
    SetLength(h_b_input_batch[b], x_size);
    SetLength(h_x_result_batch[b], x_size);

    // Generate non-singular triangular matrix A and known solution x
    for j := 0 to N - 1 do
      for i := 0 to j do
        if i = j then h_A_batch[b][i + j*lda] := Random + 1.0 // Avoid zero on diagonal
        else h_A_batch[b][i + j*lda] := Random;
    for i := 0 to x_size - 1 do h_x_original_batch[b][i] := Random;

    // Calculate b = A*x on CPU
    for i := 0 to N - 1 do
    begin
      sum := 0.0;
      for j := i to N - 1 do
        sum := sum + h_A_batch[b][i + j * lda] * h_x_original_batch[b][j * incx];
      h_b_input_batch[b][i * incx] := sum;
    end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_b_input_batch[b][0], x_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // Solve A*x = b for x
  blas_status := rocblas_strsv_batched_64(handle, uplo, transA, diag, N,
    PPSingle(d_A_ptrs), lda, PPSingle(d_x_ptrs), incx, batch_count);
  AssertEquals('rocblas_strsv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  // Verify result x against original x
  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_x_result_batch[b][0], h_x_ptrs[b], x_size * SizeOf(Single), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to x_size - 1 do
      CheckEquals(h_x_original_batch[b][i], h_x_result_batch[b][i], 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_DTRSV_batched_64;
const
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TDoubleVector   = array of Double;
  TDoubleMatrix   = array of TDoubleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_A_batch, h_x_original_batch, h_b_input_batch, h_x_result_batch: TDoubleMatrix;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  sum: Double;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_original_batch, batch_count);
  SetLength(h_b_input_batch, batch_count);
  SetLength(h_x_result_batch, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_original_batch[b], x_size);
    SetLength(h_b_input_batch[b], x_size);
    SetLength(h_x_result_batch[b], x_size);

    for j := 0 to N - 1 do
      for i := 0 to j do
        if i = j then h_A_batch[b][i + j*lda] := Random + 1.0
        else h_A_batch[b][i + j*lda] := Random;
    for i := 0 to x_size - 1 do h_x_original_batch[b][i] := Random;

    for i := 0 to N - 1 do
    begin
      sum := 0.0;
      for j := i to N - 1 do
        sum := sum + h_A_batch[b][i + j * lda] * h_x_original_batch[b][j * incx];
      h_b_input_batch[b][i * incx] := sum;
    end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_b_input_batch[b][0], x_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_dtrsv_batched_64(handle, uplo, transA, diag, N,
    PPDouble(d_A_ptrs), lda, PPDouble(d_x_ptrs), incx, batch_count);
  AssertEquals('rocblas_dtrsv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_x_result_batch[b][0], h_x_ptrs[b], x_size * SizeOf(Double), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to x_size - 1 do
      CheckEquals(h_x_original_batch[b][i], h_x_result_batch[b][i], 1e-11, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_CTRSV_batched_64;
const
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_float_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_A_batch, h_x_original_batch, h_b_input_batch, h_x_result_batch: TComplexMatrix;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  sum, term: rocblas_float_complex;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_original_batch, batch_count);
  SetLength(h_b_input_batch, batch_count);
  SetLength(h_x_result_batch, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_original_batch[b], x_size);
    SetLength(h_b_input_batch[b], x_size);
    SetLength(h_x_result_batch[b], x_size);

    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        if i = j then begin h_A_batch[b][i + j*lda].x := Random + 1.0; h_A_batch[b][i + j*lda].y := Random + 1.0; end
        else begin h_A_batch[b][i + j*lda].x := Random; h_A_batch[b][i + j*lda].y := Random; end;
      end;
    for i := 0 to x_size - 1 do begin h_x_original_batch[b][i].x := Random; h_x_original_batch[b][i].y := Random; end;

    for i := 0 to N - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for j := i to N - 1 do
      begin
        term := h_A_batch[b][i + j * lda];
        ComplexMul(term, h_x_original_batch[b][j * incx]);
        ComplexAdd(sum, term);
      end;
      h_b_input_batch[b][i * incx] := sum;
    end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_b_input_batch[b][0], x_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ctrsv_batched_64(handle, uplo, transA, diag, N,
    PPProcfloat_complex(d_A_ptrs), lda, PPProcfloat_complex(d_x_ptrs), incx, batch_count);
  AssertEquals('rocblas_ctrsv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_x_result_batch[b][0], h_x_ptrs[b], x_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to x_size - 1 do
    begin
      CheckEquals(h_x_original_batch[b][i].x, h_x_result_batch[b][i].x, 1e-3, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_x_original_batch[b][i].y, h_x_result_batch[b][i].y, 1e-3, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_ZTRSV_batched_64;
const
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_double_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_A_batch, h_x_original_batch, h_b_input_batch, h_x_result_batch: TComplexMatrix;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  sum, term: rocblas_double_complex;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_original_batch, batch_count);
  SetLength(h_b_input_batch, batch_count);
  SetLength(h_x_result_batch, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_original_batch[b], x_size);
    SetLength(h_b_input_batch[b], x_size);
    SetLength(h_x_result_batch[b], x_size);

    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        if i = j then begin h_A_batch[b][i + j*lda].x := Random + 1.0; h_A_batch[b][i + j*lda].y := Random + 1.0; end
        else begin h_A_batch[b][i + j*lda].x := Random; h_A_batch[b][i + j*lda].y := Random; end;
      end;
    for i := 0 to x_size - 1 do begin h_x_original_batch[b][i].x := Random; h_x_original_batch[b][i].y := Random; end;

    for i := 0 to N - 1 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for j := i to N - 1 do
      begin
        term := h_A_batch[b][i + j * lda];
        ComplexMul(term, h_x_original_batch[b][j * incx]);
        ComplexAdd(sum, term);
      end;
      h_b_input_batch[b][i * incx] := sum;
    end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_b_input_batch[b][0], x_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  blas_status := rocblas_ztrsv_batched_64(handle, uplo, transA, diag, N,
    PPProcdouble_complex(d_A_ptrs), lda, PPProcdouble_complex(d_x_ptrs), incx, batch_count);
  AssertEquals('rocblas_ztrsv_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_x_result_batch[b][0], h_x_ptrs[b], x_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to x_size - 1 do
    begin
      CheckEquals(h_x_original_batch[b][i].x, h_x_result_batch[b][i].x, 1e-11, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_x_original_batch[b][i].y, h_x_result_batch[b][i].y, 1e-11, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

//procedure TTestRocBLAS_Level2_batched_64.Test_STBSV_batched_64;
//begin
//  WriteLn('Function rocblas_stbsv_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_DTBSV_batched_64;
//begin
//  WriteLn('Function rocblas_dtbsv_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_CTBSV_batched_64;
//begin
//  WriteLn('Function rocblas_ctbsv_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_ZTBSV_batched_64;
//begin
//  WriteLn('Function rocblas_ztbsv_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_STPSV_batched_64;
//begin
//  WriteLn('Function rocblas_stpsv_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_DTPSV_batched_64;
//begin
//  WriteLn('Function rocblas_dtpsv_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_CTPSV_batched_64;
//begin
//  WriteLn('Function rocblas_ctpsv_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_ZTPSV_batched_64;
//begin
//  WriteLn('Function rocblas_ztpsv_batched_64 does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_batched_64.Test_SGER_batched_64;
const
  M: Int64 = 10;
  N: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TSingleVector   = array of Single;
  TSingleMatrix   = array of TSingleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: Single;
  h_A_batch, h_x_batch, h_y_batch, h_A_result, h_A_expected: TSingleMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  lda := M;
  A_size := lda * N;
  x_size := M * incx;
  y_size := N * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_A_result, batch_count);
  SetLength(h_A_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha := 1.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_A_result[b], A_size);
    SetLength(h_A_expected[b], A_size);

    for i := 0 to A_size - 1 do h_A_batch[b][i] := Random;
    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;
    for i := 0 to y_size - 1 do h_y_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected[b] := h_A_batch[b];
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
        h_A_expected[b][i + j * lda] := h_A_expected[b][i + j * lda] +
          h_alpha * h_x_batch[b][i * incx] * h_y_batch[b][j * incy];
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sger_batched_64(handle, M, N, @h_alpha,
    PPSingle(d_x_ptrs), incx, PPSingle(d_y_ptrs), incy,
    PPSingle(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_sger_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result[b][0], h_A_ptrs[b], A_size * SizeOf(Single), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to A_size - 1 do
      CheckEquals(h_A_expected[b][i], h_A_result[b][i], 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_DGER_batched_64;
const
  M: Int64 = 10;
  N: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TDoubleVector   = array of Double;
  TDoubleMatrix   = array of TDoubleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: Double;
  h_A_batch, h_x_batch, h_y_batch, h_A_result, h_A_expected: TDoubleMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  lda := M;
  A_size := lda * N;
  x_size := M * incx;
  y_size := N * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_A_result, batch_count);
  SetLength(h_A_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha := 1.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_A_result[b], A_size);
    SetLength(h_A_expected[b], A_size);

    for i := 0 to A_size - 1 do h_A_batch[b][i] := Random;
    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;
    for i := 0 to y_size - 1 do h_y_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected[b] := h_A_batch[b];
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
        h_A_expected[b][i + j * lda] := h_A_expected[b][i + j * lda] +
          h_alpha * h_x_batch[b][i * incx] * h_y_batch[b][j * incy];
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dger_batched_64(handle, M, N, @h_alpha,
    PPDouble(d_x_ptrs), incx, PPDouble(d_y_ptrs), incy,
    PPDouble(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_dger_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result[b][0], h_A_ptrs[b], A_size * SizeOf(Double), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to A_size - 1 do
      CheckEquals(h_A_expected[b][i], h_A_result[b][i], 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_CGERU_batched_64;
const
  M: Int64 = 10;
  N: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_float_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: rocblas_float_complex;
  h_A_batch, h_x_batch, h_y_batch, h_A_result, h_A_expected: TComplexMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  term: rocblas_float_complex;
begin
  lda := M;
  A_size := lda * N;
  x_size := M * incx;
  y_size := N * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_A_result, batch_count);
  SetLength(h_A_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha.x := 1.5; h_alpha.y := -0.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_A_result[b], A_size);
    SetLength(h_A_expected[b], A_size);

    for i := 0 to A_size - 1 do begin h_A_batch[b][i].x := Random; h_A_batch[b][i].y := Random; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;
    for i := 0 to y_size - 1 do begin h_y_batch[b][i].x := Random; h_y_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected[b] := h_A_batch[b];
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        term := h_alpha;
        ComplexMul(term, h_x_batch[b][i * incx]);
        ComplexMul(term, h_y_batch[b][j * incy]); // GERU uses transpose, not conjugate
        ComplexAdd(h_A_expected[b][i + j * lda], term);
      end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgeru_batched_64(handle, M, N, @h_alpha,
    PPProcfloat_complex(d_x_ptrs), incx, PPProcfloat_complex(d_y_ptrs), incy,
    PPProcfloat_complex(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_cgeru_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result[b][0], h_A_ptrs[b], A_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to A_size - 1 do
    begin
      CheckEquals(h_A_expected[b][i].x, h_A_result[b][i].x, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_A_expected[b][i].y, h_A_result[b][i].y, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_ZGERU_batched_64;
const
  M: Int64 = 10;
  N: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_double_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: rocblas_double_complex;
  h_A_batch, h_x_batch, h_y_batch, h_A_result, h_A_expected: TComplexMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  term: rocblas_double_complex;
begin
  lda := M;
  A_size := lda * N;
  x_size := M * incx;
  y_size := N * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_A_result, batch_count);
  SetLength(h_A_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha.x := 1.5; h_alpha.y := -0.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_A_result[b], A_size);
    SetLength(h_A_expected[b], A_size);

    for i := 0 to A_size - 1 do begin h_A_batch[b][i].x := Random; h_A_batch[b][i].y := Random; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;
    for i := 0 to y_size - 1 do begin h_y_batch[b][i].x := Random; h_y_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected[b] := h_A_batch[b];
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        term := h_alpha;
        ComplexMul(term, h_x_batch[b][i * incx]);
        ComplexMul(term, h_y_batch[b][j * incy]);
        ComplexAdd(h_A_expected[b][i + j * lda], term);
      end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgeru_batched_64(handle, M, N, @h_alpha,
    PPProcdouble_complex(d_x_ptrs), incx, PPProcdouble_complex(d_y_ptrs), incy,
    PPProcdouble_complex(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_zgeru_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result[b][0], h_A_ptrs[b], A_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to A_size - 1 do
    begin
      CheckEquals(h_A_expected[b][i].x, h_A_result[b][i].x, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_A_expected[b][i].y, h_A_result[b][i].y, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_CGERC_batched_64;
const
  M: Int64 = 10;
  N: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_float_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: rocblas_float_complex;
  h_A_batch, h_x_batch, h_y_batch, h_A_result, h_A_expected: TComplexMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  term, y_conj: rocblas_float_complex;
begin
  lda := M;
  A_size := lda * N;
  x_size := M * incx;
  y_size := N * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_A_result, batch_count);
  SetLength(h_A_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha.x := 1.5; h_alpha.y := -0.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_A_result[b], A_size);
    SetLength(h_A_expected[b], A_size);

    for i := 0 to A_size - 1 do begin h_A_batch[b][i].x := Random; h_A_batch[b][i].y := Random; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;
    for i := 0 to y_size - 1 do begin h_y_batch[b][i].x := Random; h_y_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected[b] := h_A_batch[b];
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        y_conj.x :=  h_y_batch[b][j * incy].x;
        y_conj.y := -h_y_batch[b][j * incy].y;
        term := h_alpha;
        ComplexMul(term, h_x_batch[b][i * incx]);
        ComplexMul(term, y_conj);
        ComplexAdd(h_A_expected[b][i + j * lda], term);
      end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgerc_batched_64(handle, M, N, @h_alpha,
    PPProcfloat_complex(d_x_ptrs), incx, PPProcfloat_complex(d_y_ptrs), incy,
    PPProcfloat_complex(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_cgerc_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result[b][0], h_A_ptrs[b], A_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to A_size - 1 do
    begin
      CheckEquals(h_A_expected[b][i].x, h_A_result[b][i].x, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_A_expected[b][i].y, h_A_result[b][i].y, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_ZGERC_batched_64;
const
  M: Int64 = 10;
  N: Int64 = 8;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_double_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: rocblas_double_complex;
  h_A_batch, h_x_batch, h_y_batch, h_A_result, h_A_expected: TComplexMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  term, y_conj: rocblas_double_complex;
begin
  lda := M;
  A_size := lda * N;
  x_size := M * incx;
  y_size := N * incy;

  SetLength(h_A_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_A_result, batch_count);
  SetLength(h_A_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha.x := 1.5; h_alpha.y := -0.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_A_result[b], A_size);
    SetLength(h_A_expected[b], A_size);

    for i := 0 to A_size - 1 do begin h_A_batch[b][i].x := Random; h_A_batch[b][i].y := Random; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;
    for i := 0 to y_size - 1 do begin h_y_batch[b][i].x := Random; h_y_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_batch[b][0], A_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected[b] := h_A_batch[b];
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        y_conj.x :=  h_y_batch[b][j * incy].x;
        y_conj.y := -h_y_batch[b][j * incy].y;
        term := h_alpha;
        ComplexMul(term, h_x_batch[b][i * incx]);
        ComplexMul(term, y_conj);
        ComplexAdd(h_A_expected[b][i + j * lda], term);
      end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgerc_batched_64(handle, M, N, @h_alpha,
    PPProcdouble_complex(d_x_ptrs), incx, PPProcdouble_complex(d_y_ptrs), incy,
    PPProcdouble_complex(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_zgerc_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result[b][0], h_A_ptrs[b], A_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to A_size - 1 do
    begin
      CheckEquals(h_A_expected[b][i].x, h_A_result[b][i].x, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.x');
      CheckEquals(h_A_expected[b][i].y, h_A_result[b][i].y, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i) + '.y');
    end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_SSYR_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TSingleVector   = array of Single;
  TSingleMatrix   = array of TSingleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: Single;
  h_A_initial_batch, h_x_batch, h_A_result_batch, h_A_expected_batch: TSingleMatrix;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;

  SetLength(h_A_initial_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_A_result_batch, batch_count);
  SetLength(h_A_expected_batch, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  h_alpha := 1.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_initial_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_A_result_batch[b], A_size);
    SetLength(h_A_expected_batch[b], A_size);

    for i := 0 to A_size - 1 do h_A_initial_batch[b][i] := Random;
    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_initial_batch[b][0], A_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected_batch[b] := h_A_initial_batch[b];
    for j := 0 to N - 1 do
      for i := 0 to j do // Since uplo is upper, only calculate this triangle
        h_A_expected_batch[b][i + j * lda] := h_A_expected_batch[b][i + j * lda] +
          h_alpha * h_x_batch[b][i * incx] * h_x_batch[b][j * incx];
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_ssyr_batched_64(handle, uplo, N, @h_alpha,
    PPSingle(d_x_ptrs), incx, PPSingle(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_ssyr_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result_batch[b][0], h_A_ptrs[b], A_size * SizeOf(Single), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    // Corrected Verification Loop
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then // Upper triangle should match the calculation
          CheckEquals(h_A_expected_batch[b][i + j * lda], h_A_result_batch[b][i + j * lda], 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index (' + IntToStr(i) + ',' + IntToStr(j) + ')')
        else // Lower triangle should be untouched
          CheckEquals(h_A_initial_batch[b][i + j * lda], h_A_result_batch[b][i + j * lda], 1e-4, 'Untouched element modified at batch ' + IntToStr(b) + ' index (' + IntToStr(i) + ',' + IntToStr(j) + ')');
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_DSYR_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TDoubleVector   = array of Double;
  TDoubleMatrix   = array of TDoubleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: Double;
  h_A_initial_batch, h_x_batch, h_A_result_batch, h_A_expected_batch: TDoubleMatrix;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;

  SetLength(h_A_initial_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_A_result_batch, batch_count);
  SetLength(h_A_expected_batch, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  h_alpha := 1.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_initial_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_A_result_batch[b], A_size);
    SetLength(h_A_expected_batch[b], A_size);

    for i := 0 to A_size - 1 do h_A_initial_batch[b][i] := Random;
    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_initial_batch[b][0], A_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected_batch[b] := h_A_initial_batch[b];
    for j := 0 to N - 1 do
      for i := 0 to j do // Since uplo is upper, only calculate this triangle
        h_A_expected_batch[b][i + j * lda] := h_A_expected_batch[b][i + j * lda] +
          h_alpha * h_x_batch[b][i * incx] * h_x_batch[b][j * incx];
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dsyr_batched_64(handle, uplo, N, @h_alpha,
    PPDouble(d_x_ptrs), incx, PPDouble(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_dsyr_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result_batch[b][0], h_A_ptrs[b], A_size * SizeOf(Double), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    // Corrected Verification Loop
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then // Upper triangle should match the calculation
          CheckEquals(h_A_expected_batch[b][i + j * lda], h_A_result_batch[b][i + j * lda], 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index (' + IntToStr(i) + ',' + IntToStr(j) + ')')
        else // Lower triangle should be untouched
          CheckEquals(h_A_initial_batch[b][i + j * lda], h_A_result_batch[b][i + j * lda], 1e-12, 'Untouched element modified at batch ' + IntToStr(b) + ' index (' + IntToStr(i) + ',' + IntToStr(j) + ')');
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

//procedure TTestRocBLAS_Level2_batched_64.Test_DSYR_batched_64;
//const
//  uplo = rocblas_fill_upper;
//  N: Int64 = 10;
//  incx: Int64 = 1;
//  batch_count: Int64 = 3;
//type
//  TDoubleVector   = array of Double;
//  TDoubleMatrix   = array of TDoubleVector;
//  TDevicePtrArray = array of hipDeviceptr_t;
//var
//  h_alpha: Double;
//  h_A_initial_batch, h_x_batch, h_A_result_batch, h_A_expected_batch: TDoubleMatrix;
//  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
//  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
//  i, j, b: Integer;
//  lda: Int64;
//  A_size, x_size: SizeUInt;
//  blas_status: rocblas_status;
//  hip_status: hipError_t;
//begin
//  lda := N;
//  A_size := lda * N;
//  x_size := N * incx;
//
//  SetLength(h_A_initial_batch, batch_count);
//  SetLength(h_x_batch, batch_count);
//  SetLength(h_A_result_batch, batch_count);
//  SetLength(h_A_expected_batch, batch_count);
//  SetLength(h_A_ptrs, batch_count);
//  SetLength(h_x_ptrs, batch_count);
//
//  h_alpha := 1.5;
//
//  Randomize;
//  for b := 0 to batch_count - 1 do
//  begin
//    SetLength(h_A_initial_batch[b], A_size);
//    SetLength(h_x_batch[b], x_size);
//    SetLength(h_A_result_batch[b], A_size);
//    SetLength(h_A_expected_batch[b], A_size);
//
//    for i := 0 to A_size - 1 do h_A_initial_batch[b][i] := Random;
//    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;
//
//    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
//    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
//    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_initial_batch[b][0], A_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
//    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
//  end;
//
//  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
//  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
//  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
//  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
//
//  // CPU calculation
//  for b := 0 to batch_count - 1 do
//  begin
//    h_A_expected_batch[b] := h_A_initial_batch[b];
//    for j := 0 to N - 1 do
//      for i := 0 to N - 1 do
//        if (uplo = rocblas_fill_upper) and (i <= j) then
//           h_A_expected_batch[b][i + j * lda] := h_A_expected_batch[b][i + j * lda] + h_alpha * h_x_batch[b][i * incx] * h_x_batch[b][j * incx]
//        else if (uplo = rocblas_fill_lower) and (i >= j) then
//           h_A_expected_batch[b][i + j * lda] := h_A_expected_batch[b][i + j * lda] + h_alpha * h_x_batch[b][i * incx] * h_x_batch[b][j * incx];
//  end;
//
//  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
//  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));
//
//  blas_status := rocblas_dsyr_batched_64(handle, uplo, N, @h_alpha,
//    PPDouble(d_x_ptrs), incx, PPDouble(d_A_ptrs), lda, batch_count);
//  AssertEquals('rocblas_dsyr_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
//
//  for b := 0 to batch_count - 1 do
//  begin
//    hip_status := hipMemcpy(@h_A_result_batch[b][0], h_A_ptrs[b], A_size * SizeOf(Double), hipMemcpyDeviceToHost);
//    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
//    for i := 0 to A_size - 1 do
//      CheckEquals(h_A_expected_batch[b][i], h_A_result_batch[b][i], 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
//  end;
//
//  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
//  for b := 0 to batch_count - 1 do
//  begin
//    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
//  end;
//end;

//procedure TTestRocBLAS_Level2_batched_64.Test_CSYR_batched_64;
//begin
//  WriteLn('Function CSYR_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_ZSYR_batched_64;
//begin
//  WriteLn('Function ZSYR_batched_64 does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_batched_64.Test_CHER_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_float_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: Single; // NOTE: alpha is REAL for HER
  h_A_initial_batch, h_x_batch, h_A_result_batch, h_A_expected_batch: TComplexMatrix;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  x_conj, term: rocblas_float_complex;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;

  SetLength(h_A_initial_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_A_result_batch, batch_count);
  SetLength(h_A_expected_batch, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  h_alpha := 1.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_initial_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_A_result_batch[b], A_size);
    SetLength(h_A_expected_batch[b], A_size);

    for i := 0 to A_size - 1 do begin h_A_initial_batch[b][i].x := Random; h_A_initial_batch[b][i].y := Random; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_initial_batch[b][0], A_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation: A := alpha * x * x^H + A
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected_batch[b] := h_A_initial_batch[b];
    for j := 0 to N - 1 do
      for i := 0 to j do // Only calculate for the upper triangle
      begin
        x_conj.x :=  h_x_batch[b][j * incx].x;
        x_conj.y := -h_x_batch[b][j * incx].y; // Conjugate of x[j]
        term := h_x_batch[b][i * incx];
        ComplexMul(term, x_conj);
        term.x := h_alpha * term.x;
        term.y := h_alpha * term.y;
        ComplexAdd(h_A_expected_batch[b][i + j * lda], term);
        if i = j then // Diagonal of a Hermitian matrix must be real
          h_A_expected_batch[b][i + j * lda].y := 0;
      end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cher_batched_64(handle, uplo, N, @h_alpha,
    PPProcfloat_complex(d_x_ptrs), incx, PPProcfloat_complex(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_cher_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result_batch[b][0], h_A_ptrs[b], A_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if i <= j then // Upper triangle: Compare against calculated value
        begin
           CheckEquals(h_A_expected_batch[b][i + j * lda].x, h_A_result_batch[b][i + j * lda].x, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').x');
           CheckEquals(h_A_expected_batch[b][i + j * lda].y, h_A_result_batch[b][i + j * lda].y, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').y');
        end
        else // Lower triangle: Compare against original value (should be untouched)
        begin
           CheckEquals(h_A_initial_batch[b][i + j * lda].x, h_A_result_batch[b][i + j * lda].x, 1e-4, 'Untouched element modified at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').x');
           CheckEquals(h_A_initial_batch[b][i + j * lda].y, h_A_result_batch[b][i + j * lda].y, 1e-4, 'Untouched element modified at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').y');
        end;
      end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_ZHER_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_double_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: Double; // NOTE: alpha is REAL for HER
  h_A_initial_batch, h_x_batch, h_A_result_batch, h_A_expected_batch: TComplexMatrix;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  x_conj, term: rocblas_double_complex;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;

  SetLength(h_A_initial_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_A_result_batch, batch_count);
  SetLength(h_A_expected_batch, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  h_alpha := 1.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_initial_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_A_result_batch[b], A_size);
    SetLength(h_A_expected_batch[b], A_size);

    for i := 0 to A_size - 1 do begin h_A_initial_batch[b][i].x := Random; h_A_initial_batch[b][i].y := Random; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_initial_batch[b][0], A_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation: A := alpha * x * x^H + A
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected_batch[b] := h_A_initial_batch[b];
    for j := 0 to N - 1 do
      for i := 0 to j do // Only calculate for the upper triangle
      begin
        x_conj.x :=  h_x_batch[b][j * incx].x;
        x_conj.y := -h_x_batch[b][j * incx].y;
        term := h_x_batch[b][i * incx];
        ComplexMul(term, x_conj);
        term.x := h_alpha * term.x;
        term.y := h_alpha * term.y;
        ComplexAdd(h_A_expected_batch[b][i + j * lda], term);
        if i = j then // Diagonal of a Hermitian matrix must be real
          h_A_expected_batch[b][i + j * lda].y := 0;
      end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zher_batched_64(handle, uplo, N, @h_alpha,
    PPProcdouble_complex(d_x_ptrs), incx, PPProcdouble_complex(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_zher_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result_batch[b][0], h_A_ptrs[b], A_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if i <= j then // Upper triangle: Compare against calculated value
        begin
           CheckEquals(h_A_expected_batch[b][i + j * lda].x, h_A_result_batch[b][i + j * lda].x, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').x');
           CheckEquals(h_A_expected_batch[b][i + j * lda].y, h_A_result_batch[b][i + j * lda].y, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').y');
        end
        else // Lower triangle: Compare against original value (should be untouched)
        begin
           CheckEquals(h_A_initial_batch[b][i + j * lda].x, h_A_result_batch[b][i + j * lda].x, 1e-12, 'Untouched element modified at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').x');
           CheckEquals(h_A_initial_batch[b][i + j * lda].y, h_A_result_batch[b][i + j * lda].y, 1e-12, 'Untouched element modified at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').y');
        end;
      end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_SSPR_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TSingleVector   = array of Single;
  TSingleMatrix   = array of TSingleVector;
  TFullMatrix     = array of array of Single;
  TFullMatrixBatch = array of TFullMatrix;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: Single;
  h_AP_initial, h_x_batch, h_AP_result, h_AP_expected: TSingleMatrix;
  h_A_full_initial, h_A_full_expected: TFullMatrixBatch;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  A_packed_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  A_packed_size := (N * (N + 1)) div 2;
  x_size := N * incx;

  SetLength(h_A_full_initial, batch_count);
  SetLength(h_A_full_expected, batch_count);
  SetLength(h_AP_initial, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_AP_result, batch_count);
  SetLength(h_AP_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  h_alpha := 1.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_full_initial[b], N, N);
    SetLength(h_A_full_expected[b], N, N);
    SetLength(h_AP_initial[b], A_packed_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_AP_result[b], A_packed_size);
    SetLength(h_AP_expected[b], A_packed_size);

    for j := 0 to N-1 do for i := 0 to j do h_A_full_initial[b,i,j] := Random;
    for j := 0 to N-1 do for i := j + 1 to N-1 do h_A_full_initial[b,i,j] := h_A_full_initial[b,j,i];
    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;

    k := 0;
    for j := 0 to N-1 do for i := 0 to j do begin h_AP_initial[b,k] := h_A_full_initial[b,i,j]; Inc(k); end;

    hip_status := hipMalloc(h_A_ptrs[b], A_packed_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_AP_initial[b][0], A_packed_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_full_expected[b] := h_A_full_initial[b];
    for j := 0 to N - 1 do
      for i := 0 to j do
        h_A_full_expected[b, i, j] := h_A_full_expected[b, i, j] + h_alpha * h_x_batch[b][i*incx] * h_x_batch[b][j*incx];
    k := 0;
    for j := 0 to N-1 do for i := 0 to j do begin h_AP_expected[b,k] := h_A_full_expected[b,i,j]; Inc(k); end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sspr_batched_64(handle, uplo, N, @h_alpha,
    PPSingle(d_x_ptrs), incx, PPSingle(d_A_ptrs), batch_count);
  AssertEquals('rocblas_sspr_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_AP_result[b][0], h_A_ptrs[b], A_packed_size * SizeOf(Single), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to A_packed_size - 1 do
      CheckEquals(h_AP_expected[b][i], h_AP_result[b][i], 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do begin hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_DSPR_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  batch_count: Int64 = 3;
type
  TDoubleVector   = array of Double;
  TDoubleMatrix   = array of TDoubleVector;
  TFullMatrix     = array of array of Double;
  TFullMatrixBatch = array of TFullMatrix;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: Double;
  h_AP_initial, h_x_batch, h_AP_result, h_AP_expected: TDoubleMatrix;
  h_A_full_initial, h_A_full_expected: TFullMatrixBatch;
  h_A_ptrs, h_x_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  A_packed_size, x_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  A_packed_size := (N * (N + 1)) div 2;
  x_size := N * incx;

  SetLength(h_A_full_initial, batch_count);
  SetLength(h_A_full_expected, batch_count);
  SetLength(h_AP_initial, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_AP_result, batch_count);
  SetLength(h_AP_expected, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);

  h_alpha := 1.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_full_initial[b], N, N);
    SetLength(h_A_full_expected[b], N, N);
    SetLength(h_AP_initial[b], A_packed_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_AP_result[b], A_packed_size);
    SetLength(h_AP_expected[b], A_packed_size);

    for j := 0 to N-1 do for i := 0 to j do h_A_full_initial[b,i,j] := Random;
    for j := 0 to N-1 do for i := j + 1 to N-1 do h_A_full_initial[b,i,j] := h_A_full_initial[b,j,i];
    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;

    k := 0;
    for j := 0 to N-1 do for i := 0 to j do begin h_AP_initial[b,k] := h_A_full_initial[b,i,j]; Inc(k); end;

    hip_status := hipMalloc(h_A_ptrs[b], A_packed_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_AP_initial[b][0], A_packed_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_full_expected[b] := h_A_full_initial[b];
    for j := 0 to N - 1 do
      for i := 0 to j do
        h_A_full_expected[b, i, j] := h_A_full_expected[b, i, j] + h_alpha * h_x_batch[b][i*incx] * h_x_batch[b][j*incx];
    k := 0;
    for j := 0 to N-1 do for i := 0 to j do begin h_AP_expected[b,k] := h_A_full_expected[b,i,j]; Inc(k); end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dspr_batched_64(handle, uplo, N, @h_alpha,
    PPDouble(d_x_ptrs), incx, PPDouble(d_A_ptrs), batch_count);
  AssertEquals('rocblas_dspr_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_AP_result[b][0], h_A_ptrs[b], A_packed_size * SizeOf(Double), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for i := 0 to A_packed_size - 1 do
      CheckEquals(h_AP_expected[b][i], h_AP_result[b][i], 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ' + IntToStr(i));
  end;

  hipFree(d_A_ptrs);
  hipFree(d_x_ptrs);
  for b := 0 to batch_count - 1 do begin hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); end;
end;

//procedure TTestRocBLAS_Level2_batched_64.Test_CHPR_batched_64;
//begin
//  WriteLn('Function CHPR_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_ZHPR_batched_64;
//begin
//  WriteLn('Function ZHPR_batched_64 does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_batched_64.Test_SSYR2_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TSingleVector   = array of Single;
  TSingleMatrix   = array of TSingleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: Single;
  h_A_initial_batch, h_x_batch, h_y_batch, h_A_result_batch, h_A_expected_batch: TSingleMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;
  y_size := N * incy;

  SetLength(h_A_initial_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_A_result_batch, batch_count);
  SetLength(h_A_expected_batch, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha := 1.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_initial_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_A_result_batch[b], A_size);
    SetLength(h_A_expected_batch[b], A_size);

    for i := 0 to A_size - 1 do h_A_initial_batch[b][i] := Random;
    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;
    for i := 0 to y_size - 1 do h_y_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(Single)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_initial_batch[b][0], A_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected_batch[b] := h_A_initial_batch[b];
    for j := 0 to N - 1 do
      for i := 0 to j do // uplo = rocblas_fill_upper
        h_A_expected_batch[b][i + j * lda] := h_A_expected_batch[b][i + j * lda] +
          h_alpha * h_x_batch[b][i * incx] * h_y_batch[b][j * incy] +
          h_alpha * h_y_batch[b][i * incy] * h_x_batch[b][j * incx];
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_ssyr2_batched_64(handle, uplo, N, @h_alpha,
    PPSingle(d_x_ptrs), incx, PPSingle(d_y_ptrs), incy, PPSingle(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_ssyr2_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result_batch[b][0], h_A_ptrs[b], A_size * SizeOf(Single), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if i <= j then // Upper triangle
          CheckEquals(h_A_expected_batch[b][i + j * lda], h_A_result_batch[b][i + j * lda], 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+')')
        else // Lower triangle
          CheckEquals(h_A_initial_batch[b][i + j * lda], h_A_result_batch[b][i + j * lda], 1e-4, 'Untouched element modified at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+')');
      end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_DSYR2_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TDoubleVector   = array of Double;
  TDoubleMatrix   = array of TDoubleVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: Double;
  h_A_initial_batch, h_x_batch, h_y_batch, h_A_result_batch, h_A_expected_batch: TDoubleMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;
  y_size := N * incy;

  SetLength(h_A_initial_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_A_result_batch, batch_count);
  SetLength(h_A_expected_batch, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha := 1.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_initial_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_A_result_batch[b], A_size);
    SetLength(h_A_expected_batch[b], A_size);

    for i := 0 to A_size - 1 do h_A_initial_batch[b][i] := Random;
    for i := 0 to x_size - 1 do h_x_batch[b][i] := Random;
    for i := 0 to y_size - 1 do h_y_batch[b][i] := Random;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(Double)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_initial_batch[b][0], A_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected_batch[b] := h_A_initial_batch[b];
    for j := 0 to N - 1 do
      for i := 0 to j do // uplo = rocblas_fill_upper
        h_A_expected_batch[b][i + j * lda] := h_A_expected_batch[b][i + j * lda] +
          h_alpha * h_x_batch[b][i * incx] * h_y_batch[b][j * incy] +
          h_alpha * h_y_batch[b][i * incy] * h_x_batch[b][j * incx];
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dsyr2_batched_64(handle, uplo, N, @h_alpha,
    PPDouble(d_x_ptrs), incx, PPDouble(d_y_ptrs), incy, PPDouble(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_dsyr2_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result_batch[b][0], h_A_ptrs[b], A_size * SizeOf(Double), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if i <= j then // Upper triangle
          CheckEquals(h_A_expected_batch[b][i + j * lda], h_A_result_batch[b][i + j * lda], 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+')')
        else // Lower triangle
          CheckEquals(h_A_initial_batch[b][i + j * lda], h_A_result_batch[b][i + j * lda], 1e-12, 'Untouched element modified at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+')');
      end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

//procedure TTestRocBLAS_Level2_batched_64.Test_CSYR2_batched_64;
//begin
//  WriteLn('Function CSYR2_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_ZSYR2_batched_64;
//begin
//  WriteLn('Function ZSYR2_batched_64does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_batched_64.Test_CHER2_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_float_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: rocblas_float_complex;
  h_A_initial_batch, h_x_batch, h_y_batch, h_A_result_batch, h_A_expected_batch: TComplexMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  alpha_conj, x_conj, y_conj, term1, term2: rocblas_float_complex;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;
  y_size := N * incy;

  SetLength(h_A_initial_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_A_result_batch, batch_count);
  SetLength(h_A_expected_batch, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha.x := 1.5; h_alpha.y := -0.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_initial_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_A_result_batch[b], A_size);
    SetLength(h_A_expected_batch[b], A_size);

    for i := 0 to A_size - 1 do begin h_A_initial_batch[b][i].x := Random; h_A_initial_batch[b][i].y := Random; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;
    for i := 0 to y_size - 1 do begin h_y_batch[b][i].x := Random; h_y_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(rocblas_float_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_initial_batch[b][0], A_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected_batch[b] := h_A_initial_batch[b];
    alpha_conj.x :=  h_alpha.x;
    alpha_conj.y := -h_alpha.y;
    for j := 0 to N - 1 do
      for i := 0 to j do // uplo = upper
      begin
        x_conj.x :=  h_x_batch[b][j * incx].x;
        x_conj.y := -h_x_batch[b][j * incx].y;
        y_conj.x :=  h_y_batch[b][j * incy].x;
        y_conj.y := -h_y_batch[b][j * incy].y;

        term1 := h_alpha;
        ComplexMul(term1, h_x_batch[b][i*incx]);
        ComplexMul(term1, y_conj);

        term2 := alpha_conj;
        ComplexMul(term2, h_y_batch[b][i*incy]);
        ComplexMul(term2, x_conj);

        ComplexAdd(h_A_expected_batch[b][i + j * lda], term1);
        ComplexAdd(h_A_expected_batch[b][i + j * lda], term2);

        if i = j then
          h_A_expected_batch[b][i + j * lda].y := 0;
      end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cher2_batched_64(handle, uplo, N, @h_alpha,
    PPProcfloat_complex(d_x_ptrs), incx, PPProcfloat_complex(d_y_ptrs), incy,
    PPProcfloat_complex(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_cher2_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result_batch[b][0], h_A_ptrs[b], A_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then // Upper triangle
        begin
          CheckEquals(h_A_expected_batch[b][i+j*lda].x, h_A_result_batch[b][i+j*lda].x, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').x');
          CheckEquals(h_A_expected_batch[b][i+j*lda].y, h_A_result_batch[b][i+j*lda].y, 1e-4, 'Mismatch at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').y');
        end
        else // Lower triangle
        begin
          CheckEquals(h_A_initial_batch[b][i+j*lda].x, h_A_result_batch[b][i+j*lda].x, 1e-4, 'Untouched element modified at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').x');
          CheckEquals(h_A_initial_batch[b][i+j*lda].y, h_A_result_batch[b][i+j*lda].y, 1e-4, 'Untouched element modified at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').y');
        end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

procedure TTestRocBLAS_Level2_batched_64.Test_ZHER2_batched_64;
const
  uplo = rocblas_fill_upper;
  N: Int64 = 10;
  incx: Int64 = 1;
  incy: Int64 = 1;
  batch_count: Int64 = 3;
type
  TComplexVector  = array of rocblas_double_complex;
  TComplexMatrix  = array of TComplexVector;
  TDevicePtrArray = array of hipDeviceptr_t;
var
  h_alpha: rocblas_double_complex;
  h_A_initial_batch, h_x_batch, h_y_batch, h_A_result_batch, h_A_expected_batch: TComplexMatrix;
  h_A_ptrs, h_x_ptrs, h_y_ptrs: TDevicePtrArray;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  lda: Int64;
  A_size, x_size, y_size: SizeUInt;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  alpha_conj, x_conj, y_conj, term1, term2: rocblas_double_complex;
begin
  lda := N;
  A_size := lda * N;
  x_size := N * incx;
  y_size := N * incy;

  SetLength(h_A_initial_batch, batch_count);
  SetLength(h_x_batch, batch_count);
  SetLength(h_y_batch, batch_count);
  SetLength(h_A_result_batch, batch_count);
  SetLength(h_A_expected_batch, batch_count);
  SetLength(h_A_ptrs, batch_count);
  SetLength(h_x_ptrs, batch_count);
  SetLength(h_y_ptrs, batch_count);

  h_alpha.x := 1.5; h_alpha.y := -0.5;

  Randomize;
  for b := 0 to batch_count - 1 do
  begin
    SetLength(h_A_initial_batch[b], A_size);
    SetLength(h_x_batch[b], x_size);
    SetLength(h_y_batch[b], y_size);
    SetLength(h_A_result_batch[b], A_size);
    SetLength(h_A_expected_batch[b], A_size);

    for i := 0 to A_size - 1 do begin h_A_initial_batch[b][i].x := Random; h_A_initial_batch[b][i].y := Random; end;
    for i := 0 to x_size - 1 do begin h_x_batch[b][i].x := Random; h_x_batch[b][i].y := Random; end;
    for i := 0 to y_size - 1 do begin h_y_batch[b][i].x := Random; h_y_batch[b][i].y := Random; end;

    hip_status := hipMalloc(h_A_ptrs[b], A_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_x_ptrs[b], x_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_y_ptrs[b], y_size * SizeOf(rocblas_double_complex)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_A_ptrs[b], @h_A_initial_batch[b][0], A_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_x_ptrs[b], @h_x_batch[b][0], x_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_y_ptrs[b], @h_y_batch[b][0], y_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  hip_status := hipMalloc(d_A_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, batch_count * SizeOf(hipDeviceptr_t)); AssertEquals('hipMalloc failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_A_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice); AssertEquals('hipMemcpy failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // CPU calculation
  for b := 0 to batch_count - 1 do
  begin
    h_A_expected_batch[b] := h_A_initial_batch[b];
    alpha_conj.x :=  h_alpha.x;
    alpha_conj.y := -h_alpha.y;
    for j := 0 to N - 1 do
      for i := 0 to j do // uplo = upper
      begin
        x_conj.x :=  h_x_batch[b][j * incx].x;
        x_conj.y := -h_x_batch[b][j * incx].y;
        y_conj.x :=  h_y_batch[b][j * incy].x;
        y_conj.y := -h_y_batch[b][j * incy].y;

        term1 := h_alpha;
        ComplexMul(term1, h_x_batch[b][i*incx]);
        ComplexMul(term1, y_conj);

        term2 := alpha_conj;
        ComplexMul(term2, h_y_batch[b][i*incy]);
        ComplexMul(term2, x_conj);

        ComplexAdd(h_A_expected_batch[b][i + j * lda], term1);
        ComplexAdd(h_A_expected_batch[b][i + j * lda], term2);

        if i = j then
          h_A_expected_batch[b][i + j * lda].y := 0;
      end;
  end;

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zher2_batched_64(handle, uplo, N, @h_alpha,
    PPProcdouble_complex(d_x_ptrs), incx, PPProcdouble_complex(d_y_ptrs), incy,
    PPProcdouble_complex(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_zher2_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_A_result_batch[b][0], h_A_ptrs[b], A_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H failed', Integer(HIP_SUCCESS), Integer(hip_status));
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then // Upper triangle
        begin
          CheckEquals(h_A_expected_batch[b][i+j*lda].x, h_A_result_batch[b][i+j*lda].x, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').x');
          CheckEquals(h_A_expected_batch[b][i+j*lda].y, h_A_result_batch[b][i+j*lda].y, 1e-12, 'Mismatch at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').y');
        end
        else // Lower triangle
        begin
          CheckEquals(h_A_initial_batch[b][i+j*lda].x, h_A_result_batch[b][i+j*lda].x, 1e-12, 'Untouched element modified at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').x');
          CheckEquals(h_A_initial_batch[b][i+j*lda].y, h_A_result_batch[b][i+j*lda].y, 1e-12, 'Untouched element modified at batch ' + IntToStr(b) + ' index ('+IntToStr(i)+','+IntToStr(j)+').y');
        end;
  end;

  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_A_ptrs[b]); hipFree(h_x_ptrs[b]); hipFree(h_y_ptrs[b]);
  end;
end;

//procedure TTestRocBLAS_Level2_batched_64.Test_SSPR2_batched_64;
//begin
//  WriteLn('Function rocblas_sspr2_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_DSPR2_batched_64;
//begin
//  WriteLn('Function rocblas_dspr2_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_CHPR2_batched_64;
//begin
//  WriteLn('Function rocblas_chpr2_batched_64 does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched_64.Test_ZHPR2_batched_64;
//begin
//  WriteLn('Function rocblas_zhpr2_batched_64 does not exist in the rocBLAS API');
//end;

initialization
  RegisterTest(TTestRocBLAS_Level2_batched_64);

end.

