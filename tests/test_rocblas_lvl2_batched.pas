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

unit Test_rocblas_lvl2_batched;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  { TTestRocBLAS_Level2_batched }

  TTestRocBLAS_Level2_batched = class(TTestRocBLAS_Base)
  published
    { General Matrix-Vector }
    procedure Test_SGEMV_batched;
    procedure Test_DGEMV_batched;
    procedure Test_CGEMV_batched;
    procedure Test_ZGEMV_batched;
    procedure Test_SGBMV_batched;
    procedure Test_DGBMV_batched;
    procedure Test_CGBMV_batched;
    procedure Test_ZGBMV_batched;

    { Hermitian }
    procedure Test_CHEMV_batched;
    procedure Test_ZHEMV_batched;
    //procedure Test_CHBMV_batched;
    //procedure Test_ZHBMV_batched;
    //procedure Test_CHPMV_batched;
    //procedure Test_ZHPMV_batched;

    { Symmetric }
    procedure Test_SSYMV_batched;
    procedure Test_DSYMV_batched;
    //procedure Test_CSYMV_batched;
    //procedure Test_ZSYMV_batched;
    //procedure Test_SSBMV_batched;
    //procedure Test_DSBMV_batched;
    procedure Test_SSPMV_batched;
    procedure Test_DSPMV_batched;

    { Triangular }
    //procedure Test_STRMV_batched;
    //procedure Test_DTRMV_batched;
    //procedure Test_CTRMV_batched;
    //procedure Test_ZTRMV_batched;
    //procedure Test_STBMV_batched;
    procedure Test_DTBMV_batched;
    procedure Test_CTBMV_batched;
    procedure Test_ZTBMV_batched;
    procedure Test_STPMV_batched;
    procedure Test_DTPMV_batched;
    procedure Test_CTPMV_batched;
    procedure Test_ZTPMV_batched;

    { Triangular Solvers }
    procedure Test_STRSV_batched;
    procedure Test_DTRSV_batched;
    procedure Test_CTRSV_batched;
    procedure Test_ZTRSV_batched;
    procedure Test_STBSV_batched;
    procedure Test_DTBSV_batched;
    procedure Test_CTBSV_batched;
    procedure Test_ZTBSV_batched;
    procedure Test_STPSV_batched;
    procedure Test_DTPSV_batched;
    procedure Test_CTPSV_batched;
    procedure Test_ZTPSV_batched;

    { Rank-1 Updates }
    procedure Test_SGER_batched;
    procedure Test_DGER_batched;
    procedure Test_CGERU_batched;
    procedure Test_ZGERU_batched;
    procedure Test_CGERC_batched;
    procedure Test_ZGERC_batched;
    procedure Test_SSYR_batched;
    procedure Test_DSYR_batched;
    procedure Test_CSYR_batched;
    procedure Test_ZSYR_batched;
    procedure Test_CHER_batched;
    procedure Test_ZHER_batched;
    procedure Test_SSPR_batched;
    procedure Test_DSPR_batched;
    procedure Test_CHPR_batched;
    procedure Test_ZHPR_batched;

    { Rank-2 Updates }
    procedure Test_SSYR2_batched;
    procedure Test_DSYR2_batched;
    procedure Test_CSYR2_batched;
    procedure Test_ZSYR2_batched;
    procedure Test_CHER2_batched;
    procedure Test_ZHER2_batched;
    procedure Test_SSPR2_batched;
    procedure Test_DSPR2_batched;
    procedure Test_CHPR2_batched;
    procedure Test_ZHPR2_batched;
  end;

implementation

{ TTestRocBLAS_Level2_batched }

procedure TTestRocBLAS_Level2_batched.Test_SGEMV_batched;
const
  M = 5;
  N = 10;
  lda = M; // For column-major, lda >= M
  incx = 1;
  incy = 1;
  batch_count = 3;
  trans = rocblas_operation_none;
type
  TSingleMatrix = array[0..lda*N-1] of Single;
  TSingleVectorX = array[0..N-1] of Single;
  TSingleVectorY = array[0..M-1] of Single;
var
  h_A_batch: array[0..batch_count-1] of TSingleMatrix;
  h_x_batch: array[0..batch_count-1] of TSingleVectorX;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TSingleVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: Single;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // Constants
  alpha := 1.5;
  beta := 2.5;
  size_A := lda * N * SizeOf(Single);
  size_x := N * incx * SizeOf(Single);
  size_y := M * incy * SizeOf(Single);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil;
  d_x_ptrs := nil;
  d_y_ptrs := nil;

  // Set up host data (Column-Major) and calculate expected result
  for b := 0 to batch_count - 1 do
  begin
    // Initialize host data in column-major format
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
        h_A_batch[b][i + j * lda] := 1.0 + b + i + j; // A(i,j)

    for i := 0 to M - 1 do h_y_batch[b][i] := 1.0 + b + i;
    for i := 0 to N - 1 do h_x_batch[b][i] := 1.0 + b + i;

    // Calculate expected result on CPU (y = beta*y + alpha*A*x)
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      h_y_expected[b][i] := h_y_expected[b][i] * beta;
      for j := 0 to N - 1 do
        h_y_expected[b][i] := h_y_expected[b][i] + alpha * h_A_batch[b][i + j * lda] * h_x_batch[b][j];
    end;
  end;

  // Allocate device memory and copy data for each batch item
  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMalloc(h_dev_A_ptrs[b], size_A);
    AssertEquals('hipMalloc for A batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));
    hip_status := hipMalloc(h_dev_x_ptrs[b], size_x);
    AssertEquals('hipMalloc for x batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));
    hip_status := hipMalloc(h_dev_y_ptrs[b], size_y);
    AssertEquals('hipMalloc for y batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));

    hip_status := hipMemcpy(h_dev_A_ptrs[b], @h_A_batch[b][0], size_A, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D for A batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));
    hip_status := hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D for x batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));
    hip_status := hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D for y batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));
  end;

  // Allocate device memory for pointer arrays and copy them
  hip_status := hipMalloc(d_A_ptrs, size_ptrs); AssertEquals('hipMalloc for d_A_ptrs', HIP_SUCCESS, Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, size_ptrs); AssertEquals('hipMalloc for d_x_ptrs', HIP_SUCCESS, Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, size_ptrs); AssertEquals('hipMalloc for d_y_ptrs', HIP_SUCCESS, Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H2D for pointer array A', HIP_SUCCESS, Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H2D for pointer array X', HIP_SUCCESS, Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H2D for pointer array Y', HIP_SUCCESS, Integer(hip_status));

  // Set pointer mode and call rocBLAS
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode to host', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sgemv_batched(handle, trans, M, N, @alpha, PPSingle(d_A_ptrs), lda, PPSingle(d_x_ptrs), incx, @beta, PPSingle(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_sgemv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy results back and verify
  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H for result batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));
    for i := 0 to M - 1 do
      CheckEqualsRelative(h_y_expected[b][i], h_y_result[b][i], 1e-5, 'Batch ' + IntToStr(b) + ', Element ' + IntToStr(i) + ' mismatch');
  end;

  // Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_DGEMV_batched;
const
  M = 5;
  N = 10;
  lda = M; // For column-major, lda >= M
  incx = 1;
  incy = 1;
  batch_count = 3;
  trans = rocblas_operation_none;
type
  TDoubleMatrix = array[0..lda*N-1] of Double;
  TDoubleVectorX = array[0..N-1] of Double;
  TDoubleVectorY = array[0..M-1] of Double;
var
  h_A_batch: array[0..batch_count-1] of TDoubleMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleVectorX;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TDoubleVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: Double;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // Constants
  alpha := 1.5;
  beta := 2.5;
  size_A := lda * N * SizeOf(Double);
  size_x := N * incx * SizeOf(Double);
  size_y := M * incy * SizeOf(Double);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil;
  d_x_ptrs := nil;
  d_y_ptrs := nil;

  // Set up host data (Column-Major) and calculate expected result
  for b := 0 to batch_count - 1 do
  begin
    // Initialize host data in column-major format
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
        h_A_batch[b][i + j * lda] := 1.0 + b + i + j; // A(i,j)

    for i := 0 to M - 1 do h_y_batch[b][i] := 1.0 + b + i;
    for i := 0 to N - 1 do h_x_batch[b][i] := 1.0 + b + i;

    // Calculate expected result on CPU (y = beta*y + alpha*A*x)
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      h_y_expected[b][i] := h_y_expected[b][i] * beta;
      for j := 0 to N - 1 do
        h_y_expected[b][i] := h_y_expected[b][i] + alpha * h_A_batch[b][i + j * lda] * h_x_batch[b][j];
    end;
  end;

  // Allocate device memory and copy data for each batch item
  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMalloc(h_dev_A_ptrs[b], size_A);
    AssertEquals('hipMalloc for A batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));
    hip_status := hipMalloc(h_dev_x_ptrs[b], size_x);
    AssertEquals('hipMalloc for x batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));
    hip_status := hipMalloc(h_dev_y_ptrs[b], size_y);
    AssertEquals('hipMalloc for y batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));

    hip_status := hipMemcpy(h_dev_A_ptrs[b], @h_A_batch[b][0], size_A, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D for A batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));
    hip_status := hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D for x batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));
    hip_status := hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D for y batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));
  end;

  // Allocate device memory for pointer arrays and copy them
  hip_status := hipMalloc(d_A_ptrs, size_ptrs); AssertEquals('hipMalloc for d_A_ptrs', HIP_SUCCESS, Integer(hip_status));
  hip_status := hipMalloc(d_x_ptrs, size_ptrs); AssertEquals('hipMalloc for d_x_ptrs', HIP_SUCCESS, Integer(hip_status));
  hip_status := hipMalloc(d_y_ptrs, size_ptrs); AssertEquals('hipMalloc for d_y_ptrs', HIP_SUCCESS, Integer(hip_status));
  hip_status := hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H2D for pointer array A', HIP_SUCCESS, Integer(hip_status));
  hip_status := hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H2D for pointer array X', HIP_SUCCESS, Integer(hip_status));
  hip_status := hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice); AssertEquals('hipMemcpy H2D for pointer array Y', HIP_SUCCESS, Integer(hip_status));

  // Set pointer mode and call rocBLAS
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode to host', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dgemv_batched(handle, trans, M, N, @alpha, PPDouble(d_A_ptrs), lda, PPDouble(d_x_ptrs), incx, @beta, PPDouble(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_dgemv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy results back and verify
  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H for result batch ' + IntToStr(b), HIP_SUCCESS, Integer(hip_status));
    for i := 0 to M - 1 do
      AssertEquals('Batch ' + IntToStr(b) + ', Element ' + IntToStr(i) + ' mismatch', h_y_expected[b][i], h_y_result[b][i], 1e-12);
  end;

  // Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_CGEMV_batched;
const
  M = 5;
  N = 10;
  lda = M; // For column-major, lda >= M
  incx = 1;
  incy = 1;
  batch_count = 3;
  trans = rocblas_operation_none;
type
  TComplexMatrix = array[0..lda*N-1] of rocblas_float_complex;
  TComplexVectorX = array[0..N-1] of rocblas_float_complex;
  TComplexVectorY = array[0..M-1] of rocblas_float_complex;
var
  h_A_batch: array[0..batch_count-1] of TComplexMatrix;
  h_x_batch: array[0..batch_count-1] of TComplexVectorX;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TComplexVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: rocblas_float_complex;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // Constants
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x := 2.5; beta.y := 1.0;
  size_A := lda * N * SizeOf(rocblas_float_complex);
  size_x := N * incx * SizeOf(rocblas_float_complex);
  size_y := M * incy * SizeOf(rocblas_float_complex);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil;
  d_x_ptrs := nil;
  d_y_ptrs := nil;

  // Set up host data (Column-Major) and calculate expected result
  for b := 0 to batch_count - 1 do
  begin
    // Initialize host data in column-major format
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        h_A_batch[b][i + j * lda].x := 1.0 + b + i; // A(i,j).x
        h_A_batch[b][i + j * lda].y := 0.5 + j;   // A(i,j).y
      end;
    for i := 0 to M - 1 do begin h_y_batch[b][i].x := 1.0 + b; h_y_batch[b][i].y := 0.5 + i; end;
    for i := 0 to N - 1 do begin h_x_batch[b][i].x := 1.0 + i; h_x_batch[b][i].y := 0.5 + b; end;

    // Calculate expected result on CPU
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      ComplexMul(h_y_expected[b][i], beta);
      for j := 0 to N - 1 do
        ComplexFMA(h_y_expected[b][i], h_A_batch[b][i + j * lda], h_x_batch[b][j], alpha);
    end;
  end;

  // Allocate device memory and copy data
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_A_ptrs[b], size_A);
    hipMalloc(h_dev_x_ptrs[b], size_x);
    hipMalloc(h_dev_y_ptrs[b], size_y);
    hipMemcpy(h_dev_A_ptrs[b], @h_A_batch[b][0], size_A, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
  end;

  // Allocate and copy pointer arrays
  hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
  hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

  // Set pointer mode and call rocBLAS
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode to host', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgemv_batched(handle, trans, M, N, @alpha, PPProcfloat_complex(d_A_ptrs), lda, PPProcfloat_complex(d_x_ptrs), incx, @beta, PPProcfloat_complex(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_cgemv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy results back and verify
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
    for i := 0 to M - 1 do
    begin
      CheckEqualsRelative(h_y_expected[b][i].x, h_y_result[b][i].x, 1e-5, 'Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + ' .x mismatch');
      CheckEqualsRelative(h_y_expected[b][i].y, h_y_result[b][i].y, 1e-5, 'Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + ' .y mismatch');
    end;
  end;

  // Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_ZGEMV_batched;
const
  M = 5;
  N = 10;
  lda = M; // For column-major, lda >= M
  incx = 1;
  incy = 1;
  batch_count = 3;
  trans = rocblas_operation_none;
type
  TDoubleComplexMatrix = array[0..lda*N-1] of rocblas_double_complex;
  TDoubleComplexVectorX = array[0..N-1] of rocblas_double_complex;
  TDoubleComplexVectorY = array[0..M-1] of rocblas_double_complex;
var
  h_A_batch: array[0..batch_count-1] of TDoubleComplexMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleComplexVectorX;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TDoubleComplexVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: rocblas_double_complex;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // Constants
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x := 2.5; beta.y := 1.0;
  size_A := lda * N * SizeOf(rocblas_double_complex);
  size_x := N * incx * SizeOf(rocblas_double_complex);
  size_y := M * incy * SizeOf(rocblas_double_complex);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil;
  d_x_ptrs := nil;
  d_y_ptrs := nil;

  // Set up host data (Column-Major) and calculate expected result
  for b := 0 to batch_count - 1 do
  begin
    // Initialize host data in column-major format
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        h_A_batch[b][i + j * lda].x := 1.0 + b + i; // A(i,j).x
        h_A_batch[b][i + j * lda].y := 0.5 + j;   // A(i,j).y
      end;
    for i := 0 to M - 1 do begin h_y_batch[b][i].x := 1.0 + b; h_y_batch[b][i].y := 0.5 + i; end;
    for i := 0 to N - 1 do begin h_x_batch[b][i].x := 1.0 + i; h_x_batch[b][i].y := 0.5 + b; end;

    // Calculate expected result on CPU
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      ComplexMul(h_y_expected[b][i], beta);
      for j := 0 to N - 1 do
        ComplexFMA(h_y_expected[b][i], h_A_batch[b][i + j * lda], h_x_batch[b][j], alpha);
    end;
  end;

  // Allocate device memory and copy data
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_A_ptrs[b], size_A);
    hipMalloc(h_dev_x_ptrs[b], size_x);
    hipMalloc(h_dev_y_ptrs[b], size_y);
    hipMemcpy(h_dev_A_ptrs[b], @h_A_batch[b][0], size_A, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
  end;

  // Allocate and copy pointer arrays
  hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
  hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

  // Set pointer mode and call rocBLAS
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode to host', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgemv_batched(handle, trans, M, N, @alpha, PPProcdouble_complex(d_A_ptrs), lda, PPProcdouble_complex(d_x_ptrs), incx, @beta, PPProcdouble_complex(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_zgemv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy results back and verify
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
    for i := 0 to M - 1 do
    begin
      AssertEquals('Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + ' .x mismatch', h_y_expected[b][i].x, h_y_result[b][i].x, 1e-12);
      AssertEquals('Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + ' .y mismatch', h_y_expected[b][i].y, h_y_result[b][i].y, 1e-12);
    end;
  end;

  // Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_SGBMV_batched;
const
  M = 10;
  N = 10;
  kl = 2;
  ku = 2;
  lda = kl + ku + 1;
  incx = 1;
  incy = 1;
  batch_count = 3;
  trans = rocblas_operation_none;
type
  TBandMatrix = array[0..lda*N-1] of Single;
  TVectorX = array[0..N-1] of Single;
  TVectorY = array[0..M-1] of Single;
var
  h_A_band_batch: array[0..batch_count-1] of TBandMatrix;
  h_x_batch: array[0..batch_count-1] of TVectorX;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: Single;
  i, j, b, band_row: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // Constants
  alpha := 1.5;
  beta := 2.5;
  size_A := lda * N * SizeOf(Single);
  size_x := N * SizeOf(Single);
  size_y := M * SizeOf(Single);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil;
  d_x_ptrs := nil;
  d_y_ptrs := nil;

  // Set up host data and calculate expected result
  for b := 0 to batch_count - 1 do
  begin
    // Initialize host band matrix in column-major format
    for j := 0 to N - 1 do
      for i := 0 to lda - 1 do
        h_A_band_batch[b][i + j * lda] := 1.0 + b + i + j;

    for i := 0 to N-1 do h_x_batch[b][i] := 1.0 + b + i;
    for i := 0 to M-1 do h_y_batch[b][i] := 1.0 + b + i;

    // CPU calculation from band matrix format
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      h_y_expected[b][i] := beta * h_y_expected[b][i];
      for j := Max(0, i - kl) to Min(N - 1, i + ku) do
      begin
        band_row := ku + i - j;
        h_y_expected[b][i] := h_y_expected[b][i] + alpha * h_A_band_batch[b][band_row + j * lda] * h_x_batch[b][j];
      end;
    end;
  end;

  // Allocate device memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_A_ptrs[b], size_A);
    hipMalloc(h_dev_x_ptrs[b], size_x);
    hipMalloc(h_dev_y_ptrs[b], size_y);
    hipMemcpy(h_dev_A_ptrs[b], @h_A_band_batch[b][0], size_A, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
  end;

  hipMalloc(d_A_ptrs, size_ptrs);
  hipMalloc(d_x_ptrs, size_ptrs);
  hipMalloc(d_y_ptrs, size_ptrs);
  hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

  // Call rocBLAS
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode to host', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sgbmv_batched(handle, trans, M, N, kl, ku, @alpha, PPSingle(d_A_ptrs), lda, PPSingle(d_x_ptrs), incx, @beta, PPSingle(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_sgbmv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy results and verify
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
    for i := 0 to M - 1 do
      CheckEqualsRelative(h_y_expected[b][i], h_y_result[b][i], 1e-4, 'Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + ' mismatch');
  end;

  // Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_DGBMV_batched;
const
  M = 10;
  N = 10;
  kl = 2;
  ku = 2;
  lda = kl + ku + 1;
  incx = 1;
  incy = 1;
  batch_count = 3;
  trans = rocblas_operation_none;
type
  TBandMatrix = array[0..lda*N-1] of Double;
  TVectorX = array[0..N-1] of Double;
  TVectorY = array[0..M-1] of Double;
var
  h_A_band_batch: array[0..batch_count-1] of TBandMatrix;
  h_x_batch: array[0..batch_count-1] of TVectorX;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: Double;
  i, j, b, band_row: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // Constants
  alpha := 1.5;
  beta := 2.5;
  size_A := lda * N * SizeOf(Double);
  size_x := N * SizeOf(Double);
  size_y := M * SizeOf(Double);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil;
  d_x_ptrs := nil;
  d_y_ptrs := nil;

  // Set up host data and calculate expected result
  for b := 0 to batch_count - 1 do
  begin
    // Initialize host band matrix in column-major format
    for j := 0 to N - 1 do
      for i := 0 to lda - 1 do
        h_A_band_batch[b][i + j * lda] := 1.0 + b + i + j;

    for i := 0 to N-1 do h_x_batch[b][i] := 1.0 + b + i;
    for i := 0 to M-1 do h_y_batch[b][i] := 1.0 + b + i;

    // CPU calculation from band matrix format
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      h_y_expected[b][i] := beta * h_y_expected[b][i];
      for j := Max(0, i - kl) to Min(N - 1, i + ku) do
      begin
        band_row := ku + i - j;
        h_y_expected[b][i] := h_y_expected[b][i] + alpha * h_A_band_batch[b][band_row + j * lda] * h_x_batch[b][j];
      end;
    end;
  end;

  // Allocate device memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_A_ptrs[b], size_A);
    hipMalloc(h_dev_x_ptrs[b], size_x);
    hipMalloc(h_dev_y_ptrs[b], size_y);
    hipMemcpy(h_dev_A_ptrs[b], @h_A_band_batch[b][0], size_A, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
  end;

  hipMalloc(d_A_ptrs, size_ptrs);
  hipMalloc(d_x_ptrs, size_ptrs);
  hipMalloc(d_y_ptrs, size_ptrs);
  hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

  // Call rocBLAS
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode to host', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dgbmv_batched(handle, trans, M, N, kl, ku, @alpha, PPDouble(d_A_ptrs), lda, PPDouble(d_x_ptrs), incx, @beta, PPDouble(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_dgbmv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy results and verify
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
    for i := 0 to M - 1 do
      AssertEquals('Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + ' mismatch', h_y_expected[b][i], h_y_result[b][i], 1e-12);
  end;

  // Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_CGBMV_batched;
const
  M = 10;
  N = 10;
  kl = 2;
  ku = 2;
  lda = kl + ku + 1;
  incx = 1;
  incy = 1;
  batch_count = 3;
  trans = rocblas_operation_none;
type
  TComplexBandMatrix = array[0..lda*N-1] of rocblas_float_complex;
  TComplexVectorX = array[0..N-1] of rocblas_float_complex;
  TComplexVectorY = array[0..M-1] of rocblas_float_complex;
var
  h_A_band_batch: array[0..batch_count-1] of TComplexBandMatrix;
  h_x_batch: array[0..batch_count-1] of TComplexVectorX;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TComplexVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: rocblas_float_complex;
  i, j, b, band_row: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // Constants
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x  := 2.5; beta.y  := 1.0;
  size_A := lda * N * SizeOf(rocblas_float_complex);
  size_x := N * SizeOf(rocblas_float_complex);
  size_y := M * SizeOf(rocblas_float_complex);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

  // Set up host data and calculate expected result
  for b := 0 to batch_count - 1 do
  begin
    // Initialize host band matrix in column-major format
    for j := 0 to N - 1 do
      for i := 0 to lda - 1 do
      begin
        h_A_band_batch[b][i + j * lda].x := 1.0 + b + i;
        h_A_band_batch[b][i + j * lda].y := 0.5 + j;
      end;
    for i := 0 to N-1 do begin h_x_batch[b][i].x := 1.0 + b; h_x_batch[b][i].y := 0.5 + i; end;
    for i := 0 to M-1 do begin h_y_batch[b][i].x := 1.0 + i; h_y_batch[b][i].y := 0.5 + b; end;

    // CPU calculation
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      ComplexMul(h_y_expected[b][i], beta);
      for j := Max(0, i - kl) to Min(N - 1, i + ku) do
      begin
        band_row := ku + i - j;
        ComplexFMA(h_y_expected[b][i], h_A_band_batch[b][band_row + j * lda], h_x_batch[b][j], alpha);
      end;
    end;
  end;

  // Allocate device memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
    hipMemcpy(h_dev_A_ptrs[b], @h_A_band_batch[b][0], size_A, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
  end;

  hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
  hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

  // Call rocBLAS
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode to host', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgbmv_batched(handle, trans, M, N, kl, ku, @alpha, PPProcfloat_complex(d_A_ptrs), lda, PPProcfloat_complex(d_x_ptrs), incx, @beta, PPProcfloat_complex(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_cgbmv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy results and verify
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
    for i := 0 to M - 1 do
    begin
      CheckEqualsRelative(h_y_expected[b][i].x, h_y_result[b][i].x, 1e-4, 'Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + ' .x mismatch');
      CheckEqualsRelative(h_y_expected[b][i].y, h_y_result[b][i].y, 1e-4, 'Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + ' .y mismatch');
    end;
  end;

  // Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_ZGBMV_batched;
const
  M = 10;
  N = 10;
  kl = 2;
  ku = 2;
  lda = kl + ku + 1;
  incx = 1;
  incy = 1;
  batch_count = 3;
  trans = rocblas_operation_none;
type
  TDoubleComplexBandMatrix = array[0..lda*N-1] of rocblas_double_complex;
  TDoubleComplexVectorX = array[0..N-1] of rocblas_double_complex;
  TDoubleComplexVectorY = array[0..M-1] of rocblas_double_complex;
var
  h_A_band_batch: array[0..batch_count-1] of TDoubleComplexBandMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleComplexVectorX;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TDoubleComplexVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: rocblas_double_complex;
  i, j, b, band_row: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // Constants
  alpha.x := 1.5; alpha.y := -0.5;
  beta.x  := 2.5; beta.y  := 1.0;
  size_A := lda * N * SizeOf(rocblas_double_complex);
  size_x := N * SizeOf(rocblas_double_complex);
  size_y := M * SizeOf(rocblas_double_complex);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

  // Set up host data and calculate expected result
  for b := 0 to batch_count - 1 do
  begin
    // Initialize host band matrix in column-major format
    for j := 0 to N - 1 do
      for i := 0 to lda - 1 do
      begin
        h_A_band_batch[b][i + j * lda].x := 1.0 + b + i;
        h_A_band_batch[b][i + j * lda].y := 0.5 + j;
      end;
    for i := 0 to N-1 do begin h_x_batch[b][i].x := 1.0 + b; h_x_batch[b][i].y := 0.5 + i; end;
    for i := 0 to M-1 do begin h_y_batch[b][i].x := 1.0 + i; h_y_batch[b][i].y := 0.5 + b; end;

    // CPU calculation
    h_y_expected[b] := h_y_batch[b];
    for i := 0 to M - 1 do
    begin
      ComplexMul(h_y_expected[b][i], beta);
      for j := Max(0, i - kl) to Min(N - 1, i + ku) do
      begin
        band_row := ku + i - j;
        ComplexFMA(h_y_expected[b][i], h_A_band_batch[b][band_row + j * lda], h_x_batch[b][j], alpha);
      end;
    end;
  end;

  // Allocate device memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
    hipMemcpy(h_dev_A_ptrs[b], @h_A_band_batch[b][0], size_A, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
  end;

  hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
  hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

  // Call rocBLAS
  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode to host', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgbmv_batched(handle, trans, M, N, kl, ku, @alpha, PPProcdouble_complex(d_A_ptrs), lda, PPProcdouble_complex(d_x_ptrs), incx, @beta, PPProcdouble_complex(d_y_ptrs), incy, batch_count);
  AssertEquals('rocblas_zgbmv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // Copy results and verify
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
    for i := 0 to M - 1 do
    begin
      AssertEquals('Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + ' .x mismatch', h_y_expected[b][i].x, h_y_result[b][i].x, 1e-12);
      AssertEquals('Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + ' .y mismatch', h_y_expected[b][i].y, h_y_result[b][i].y, 1e-12);
    end;
  end;

  // Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_CHEMV_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TComplexMatrix = array[0..lda*N-1] of rocblas_float_complex;
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_A_batch, h_A_full: array[0..batch_count-1] of TComplexMatrix;
  h_x_batch: array[0..batch_count-1] of TComplexVector;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: rocblas_float_complex;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    // Constants
    alpha.x := 1.5; alpha.y := -0.5;
    beta.x := 2.5; beta.y := 1.0;
    size_A := lda * N * SizeOf(rocblas_float_complex);
    size_x := N * SizeOf(rocblas_float_complex);
    size_y := N * SizeOf(rocblas_float_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    // Set up host data and calculate expected result
    for b := 0 to batch_count - 1 do
    begin
      // Initialize full Hermitian matrix for CPU calculation
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
        begin
          if i = j then
          begin
            h_A_full[b][i + j * lda].x := 2.0 + b + i; // Real diagonal
            h_A_full[b][i + j * lda].y := 0.0;
          end
          else if i < j then
          begin
            h_A_full[b][i + j * lda].x := 1.0 + i;
            h_A_full[b][i + j * lda].y := 0.5 + j;
            h_A_full[b][j + i * lda].x := h_A_full[b][i + j * lda].x; // Conjugate
            h_A_full[b][j + i * lda].y := -h_A_full[b][i + j * lda].y;
          end;
        end;
      h_A_batch[b] := h_A_full[b];

      for i := 0 to N - 1 do begin
        h_x_batch[b][i].x := 1.0 + b + i;
        h_x_batch[b][i].y := 0.5 * i;
        h_y_batch[b][i].x := 1.0 + b;
        h_y_batch[b][i].y := 0.5 + i;
      end;

      // CPU reference calculation using the full matrix
      h_y_expected[b] := h_y_batch[b];
      for i := 0 to N - 1 do
      begin
        ComplexMul(h_y_expected[b][i], beta);
        for j := 0 to N - 1 do
          ComplexFMA(h_y_expected[b][i], h_A_full[b][i + j * lda], h_x_batch[b][j], alpha);
      end;
    end;

    // Allocate device memory and copy data
    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A);
      hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;

    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode to host', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_chemv_batched(handle, uplo, N, @alpha, PPProcfloat_complex(d_A_ptrs), lda, PPProcfloat_complex(d_x_ptrs), incx, @beta, PPProcfloat_complex(d_y_ptrs), incy, batch_count);
    AssertEquals('rocblas_chemv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
      begin
        CheckEqualsRelative(h_y_expected[b][i].x, h_y_result[b][i].x, 1e-5, uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x');
        CheckEqualsRelative(h_y_expected[b][i].y, h_y_result[b][i].y, 1e-5, uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y');
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]);
    end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_ZHEMV_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TDoubleComplexMatrix = array[0..lda*N-1] of rocblas_double_complex;
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_A_batch, h_A_full: array[0..batch_count-1] of TDoubleComplexMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleComplexVector;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: rocblas_double_complex;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    // Constants
    alpha.x := 1.5; alpha.y := -0.5;
    beta.x := 2.5; beta.y := 1.0;
    size_A := lda * N * SizeOf(rocblas_double_complex);
    size_x := N * SizeOf(rocblas_double_complex);
    size_y := N * SizeOf(rocblas_double_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    // Set up host data and calculate expected result
    for b := 0 to batch_count - 1 do
    begin
      // Initialize full Hermitian matrix for CPU calculation
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
        begin
          if i = j then
          begin
            h_A_full[b][i + j * lda].x := 2.0 + b + i; // Real diagonal
            h_A_full[b][i + j * lda].y := 0.0;
          end
          else if i < j then
          begin
            h_A_full[b][i + j * lda].x := 1.0 + i;
            h_A_full[b][i + j * lda].y := 0.5 + j;
            h_A_full[b][j + i * lda].x := h_A_full[b][i + j * lda].x; // Conjugate
            h_A_full[b][j + i * lda].y := -h_A_full[b][i + j * lda].y;
          end;
        end;
      h_A_batch[b] := h_A_full[b];

      for i := 0 to N - 1 do begin
        h_x_batch[b][i].x := 1.0 + b + i;
        h_x_batch[b][i].y := 0.5 * i;
        h_y_batch[b][i].x := 1.0 + b;
        h_y_batch[b][i].y := 0.5 + i;
      end;

      // CPU reference calculation
      h_y_expected[b] := h_y_batch[b];
      for i := 0 to N - 1 do
      begin
        ComplexMul(h_y_expected[b][i], beta);
        for j := 0 to N - 1 do
          ComplexFMA(h_y_expected[b][i], h_A_full[b][i + j * lda], h_x_batch[b][j], alpha);
      end;
    end;

    // Allocate device memory and copy data
    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A);
      hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;

    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode to host', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_zhemv_batched(handle, uplo, N, @alpha, PPProcdouble_complex(d_A_ptrs), lda, PPProcdouble_complex(d_x_ptrs), incx, @beta, PPProcdouble_complex(d_y_ptrs), incy, batch_count);
    AssertEquals('rocblas_zhemv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
      begin
        AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x', h_y_expected[b][i].x, h_y_result[b][i].x, 1e-12);
        AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y', h_y_expected[b][i].y, h_y_result[b][i].y, 1e-12);
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]);
    end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

//procedure TTestRocBLAS_Level2_batched.Test_CHBMV_batched;
//begin
//  WriteLn('Function CHBMV does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched.Test_ZHBMV_batched;
//begin
//  WriteLn('Function ZHBMV does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched.Test_CHPMV_batched;
//begin
//  WriteLn('Function CHPMV does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched.Test_ZHPMV_batched;
//begin
//  WriteLn('Function ZHPMV does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_batched.Test_SSYMV_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TSingleMatrix = array[0..lda*N-1] of Single;
  TSingleVector = array[0..N-1] of Single;
var
  h_A_batch, h_A_full: array[0..batch_count-1] of TSingleMatrix;
  h_x_batch: array[0..batch_count-1] of TSingleVector;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TSingleVector;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: Single;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    // Constants
    alpha := 1.5;
    beta := 2.5;
    size_A := lda * N * SizeOf(Single);
    size_x := N * incx * SizeOf(Single);
    size_y := N * incy * SizeOf(Single);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    // Set up host data and calculate expected result
    for b := 0 to batch_count - 1 do
    begin
      // Initialize a full symmetric matrix for CPU calculation
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          h_A_full[b][i + j * lda] := 1.0 + b + i + j;
          h_A_full[b][j + i * lda] := h_A_full[b][i + j * lda]; // Mirror for symmetry
        end;
      h_A_batch[b] := h_A_full[b]; // Copy full matrix for GPU input

      for i := 0 to N - 1 do
      begin
        h_x_batch[b][i] := 1.0 + b + i;
        h_y_batch[b][i] := 0.5 + b + i;
      end;

      // CPU reference calculation using the full symmetric matrix
      h_y_expected[b] := h_y_batch[b];
      for i := 0 to N - 1 do
      begin
        h_y_expected[b][i] := h_y_expected[b][i] * beta;
        for j := 0 to N - 1 do
          h_y_expected[b][i] := h_y_expected[b][i] + alpha * h_A_full[b][i + j * lda] * h_x_batch[b][j];
      end;
    end;

    // Allocate and manage device memory
    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;

    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    // Call rocBLAS function
    blas_status := rocblas_ssymv_batched(handle, uplo, N, @alpha, PPSingle(d_A_ptrs), lda, PPSingle(d_x_ptrs), incx, @beta, PPSingle(d_y_ptrs), incy, batch_count);
    AssertEquals('rocblas_ssymv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    // Copy results back and verify
    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
        CheckEqualsRelative(h_y_expected[b][i], h_y_result[b][i], 1e-5, uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i));
    end;

    // Cleanup
    for b := 0 to batch_count - 1 do
    begin
      hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]);
    end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_DSYMV_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TDoubleMatrix = array[0..lda*N-1] of Double;
  TDoubleVector = array[0..N-1] of Double;
var
  h_A_batch, h_A_full: array[0..batch_count-1] of TDoubleMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleVector;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TDoubleVector;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: Double;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    // Constants
    alpha := 1.5;
    beta := 2.5;
    size_A := lda * N * SizeOf(Double);
    size_x := N * incx * SizeOf(Double);
    size_y := N * incy * SizeOf(Double);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    // Set up host data and calculate expected result
    for b := 0 to batch_count - 1 do
    begin
      // Initialize a full symmetric matrix for CPU calculation
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          h_A_full[b][i + j * lda] := 1.0 + b + i + j;
          h_A_full[b][j + i * lda] := h_A_full[b][i + j * lda];
        end;
      h_A_batch[b] := h_A_full[b];

      for i := 0 to N - 1 do
      begin
        h_x_batch[b][i] := 1.0 + b + i;
        h_y_batch[b][i] := 0.5 + b + i;
      end;

      // CPU reference calculation
      h_y_expected[b] := h_y_batch[b];
      for i := 0 to N - 1 do
      begin
        h_y_expected[b][i] := h_y_expected[b][i] * beta;
        for j := 0 to N - 1 do
          h_y_expected[b][i] := h_y_expected[b][i] + alpha * h_A_full[b][i + j * lda] * h_x_batch[b][j];
      end;
    end;

    // Allocate and manage device memory
    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;

    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    // Call rocBLAS function
    blas_status := rocblas_dsymv_batched(handle, uplo, N, @alpha, PPDouble(d_A_ptrs), lda, PPDouble(d_x_ptrs), incx, @beta, PPDouble(d_y_ptrs), incy, batch_count);
    AssertEquals('rocblas_dsymv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    // Copy results back and verify
    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
        AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i), h_y_expected[b][i], h_y_result[b][i], 1e-12);
    end;

    // Cleanup
    for b := 0 to batch_count - 1 do
    begin
      hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]);
    end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

//procedure TTestRocBLAS_Level2_batched.Test_CSYMV_batched;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched.Test_ZSYMV_batched;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched.Test_SSBMV_batched;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched.Test_DSBMV_batched;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_batched.Test_SSPMV_batched;
const
  N = 10;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TSinglePackedMatrix = array[0..N*(N+1) div 2 - 1] of Single;
  TSingleVector = array[0..N-1] of Single;
  TSingleFullMatrix = array[0..N*N-1] of Single;
var
  h_AP_batch: array[0..batch_count-1] of TSinglePackedMatrix;
  h_A_full: array[0..batch_count-1] of TSingleFullMatrix;
  h_x_batch: array[0..batch_count-1] of TSingleVector;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TSingleVector;
  h_dev_AP_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: Single;
  i, j, b, k: Integer;
  size_AP, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    // Constants
    alpha := 1.5;
    beta := 2.5;
    size_AP := (N * (N + 1) div 2) * SizeOf(Single);
    size_x := N * incx * SizeOf(Single);
    size_y := N * incy * SizeOf(Single);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    // Set up host data and calculate expected result
    for b := 0 to batch_count - 1 do
    begin
      // Initialize full symmetric matrix for reference and create packed version
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := 0 to N - 1 do
        begin
          if i >= j then // Lower triangle including diagonal
          begin
            h_A_full[b][i + j * N] := 1.0 + b + i + j;
            h_A_full[b][j + i * N] := h_A_full[b][i + j * N];
          end;
        end;

        if uplo = rocblas_fill_upper then
          for i := 0 to j do
          begin
            h_AP_batch[b][k] := h_A_full[b][i + j * N];
            Inc(k);
          end
        else // rocblas_fill_lower
          for i := j to N - 1 do
          begin
            h_AP_batch[b][k] := h_A_full[b][i + j * N];
            Inc(k);
          end;
      end;

      for i := 0 to N - 1 do begin
        h_x_batch[b][i] := 1.0 + b + i;
        h_y_batch[b][i] := 0.5 + b + i;
      end;

      // CPU reference calculation using the full matrix
      h_y_expected[b] := h_y_batch[b];
      for i := 0 to N - 1 do
      begin
        h_y_expected[b][i] := beta * h_y_expected[b][i];
        for j := 0 to N - 1 do
          h_y_expected[b][i] := h_y_expected[b][i] + alpha * h_A_full[b][i + j * N] * h_x_batch[b][j];
      end;
    end;

    // Allocate and manage device memory
    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_batch[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;

    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_sspmv_batched(handle, uplo, N, @alpha, PPSingle(d_AP_ptrs), PPSingle(d_x_ptrs), incx, @beta, PPSingle(d_y_ptrs), incy, batch_count);
    AssertEquals('rocblas_sspmv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
        CheckEqualsRelative(h_y_expected[b][i], h_y_result[b][i], 1e-5, uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i));
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_DSPMV_batched;
const
  N = 10;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TDoublePackedMatrix = array[0..N*(N+1) div 2 - 1] of Double;
  TDoubleVector = array[0..N-1] of Double;
  TDoubleFullMatrix = array[0..N*N-1] of Double;
var
  h_AP_batch: array[0..batch_count-1] of TDoublePackedMatrix;
  h_A_full: array[0..batch_count-1] of TDoubleFullMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleVector;
  h_y_batch, h_y_expected, h_y_result: array[0..batch_count-1] of TDoubleVector;
  h_dev_AP_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, beta: Double;
  i, j, b, k: Integer;
  size_AP, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    // Constants
    alpha := 1.5;
    beta := 2.5;
    size_AP := (N * (N + 1) div 2) * SizeOf(Double);
    size_x := N * incx * SizeOf(Double);
    size_y := N * incy * SizeOf(Double);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    // Set up host data
    for b := 0 to batch_count - 1 do
    begin
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := 0 to N - 1 do
        begin
          if i >= j then
          begin
            h_A_full[b][i + j * N] := 1.0 + b + i + j;
            h_A_full[b][j + i * N] := h_A_full[b][i + j * N];
          end;
        end;
        if uplo = rocblas_fill_upper then
          for i := 0 to j do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else
          for i := j to N - 1 do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
      for i := 0 to N - 1 do begin h_x_batch[b][i] := 1.0 + b + i; h_y_batch[b][i] := 0.5 + b + i; end;
      h_y_expected[b] := h_y_batch[b];
      for i := 0 to N - 1 do
      begin
        h_y_expected[b][i] := beta * h_y_expected[b][i];
        for j := 0 to N - 1 do
          h_y_expected[b][i] := h_y_expected[b][i] + alpha * h_A_full[b][i + j * N] * h_x_batch[b][j];
      end;
    end;

    // Allocate and manage device memory
    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_batch[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_dspmv_batched(handle, uplo, N, @alpha, PPDouble(d_AP_ptrs), PPDouble(d_x_ptrs), incx, @beta, PPDouble(d_y_ptrs), incy, batch_count);
    AssertEquals('rocblas_dspmv_batched failed', Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_y_result[b][0], h_dev_y_ptrs[b], size_y, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
        AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i), h_y_expected[b][i], h_y_result[b][i], 1e-12);
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

//procedure TTestRocBLAS_Level2_batched.Test_STRMV_batched;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched.Test_DTRMV_batched;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched.Test_CTRMV_batched;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched.Test_ZTRMV_batched;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_batched.Test_STBMV_batched;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_batched.Test_DTBMV_batched;
const
  N = 10;
  K = 2; // Number of super/sub-diagonals
  lda = K + 1;
  incx = 1;
  batch_count = 2;
type
  TDoubleBandMatrix = array[0..lda*N-1] of Double;
  TDoubleVector = array[0..N-1] of Double;
  TDoubleFullMatrix = array[0..N*N-1] of Double;
var
  h_A_band_batch: array[0..batch_count-1] of TDoubleBandMatrix;
  h_A_full: array[0..batch_count-1] of TDoubleFullMatrix;
  h_x_batch, h_x_expected, h_x_result: array[0..batch_count-1] of TDoubleVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_A := lda * N * SizeOf(Double);
    size_x := N * incx * SizeOf(Double);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      FillChar(h_A_band_batch[b], SizeOf(h_A_band_batch[b]), 0);

      // Create a full triangular matrix for reference
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (j >= i) and (j - i <= K) then // CORRECTED: Parentheses
            h_A_full[b][i + j * N] := 1.0 + i + j + b
          else if (uplo = rocblas_fill_lower) and (i >= j) and (i - j <= K) then // CORRECTED: Parentheses
            h_A_full[b][i + j * N] := 1.0 + i + j + b;

      if diag = rocblas_diagonal_unit then
        for i := 0 to N - 1 do h_A_full[b][i + i * N] := 1.0;

      // Convert full matrix to rocBLAS banded format
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (j-i <= K) and (i <= j) then // CORRECTED: Parentheses
            h_A_band_batch[b][(K + i - j) + j * lda] := h_A_full[b][i + j * N]
          else if (uplo = rocblas_fill_lower) and (i-j <= K) and (j <= i) then // CORRECTED: Parentheses
            h_A_band_batch[b][(i - j) + j * lda] := h_A_full[b][i + j * N];

      for i := 0 to N - 1 do h_x_batch[b][i] := 1.0 + b + i;

      // CPU reference calculation
      h_x_expected[b] := h_x_batch[b];
      FillChar(h_x_result[b], SizeOf(h_x_result[b]), 0);
      if transA = rocblas_operation_none then
        for i := 0 to N - 1 do for j := 0 to N - 1 do
          h_x_result[b][i] := h_x_result[b][i] + h_A_full[b][i + j * N] * h_x_expected[b][j]
      else
        for i := 0 to N - 1 do for j := 0 to N - 1 do
          h_x_result[b][i] := h_x_result[b][i] + h_A_full[b][j + i * N] * h_x_expected[b][j];
      h_x_expected[b] := h_x_result[b];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_band_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_dtbmv_batched(handle, uplo, transA, diag, N, K, PPDouble(d_A_ptrs), lda, PPDouble(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_dtbmv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i), h_x_expected[b][i], h_x_result[b][i], 1e-12);
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_CTBMV_batched;
const
  N = 10;
  K = 2;
  lda = K + 1;
  incx = 1;
  batch_count = 2;
  one_complex: rocblas_float_complex = (x: 1.0; y: 0.0);
type
  TComplexBandMatrix = array[0..lda*N-1] of rocblas_float_complex;
  TComplexVector = array[0..N-1] of rocblas_float_complex;
  TComplexFullMatrix = array[0..N*N-1] of rocblas_float_complex;
var
  h_A_band_batch: array[0..batch_count-1] of TComplexBandMatrix;
  h_A_full: array[0..batch_count-1] of TComplexFullMatrix;
  h_x_batch, h_x_expected, h_x_result: array[0..batch_count-1] of TComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
  temp: rocblas_float_complex;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose, rocblas_operation_conjugate_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_A := lda * N * SizeOf(rocblas_float_complex);
    size_x := N * incx * SizeOf(rocblas_float_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      FillChar(h_A_band_batch[b], SizeOf(h_A_band_batch[b]), 0);

      // Create a full triangular matrix for reference
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
        begin
          if (uplo = rocblas_fill_upper) and (j >= i) and (j - i <= K) then
          begin
            h_A_full[b][i + j * N].x := 1.0 + i + j + b;
            h_A_full[b][i + j * N].y := 0.5 + i - j;
          end
          else if (uplo = rocblas_fill_lower) and (i >= j) and (i - j <= K) then
          begin
            h_A_full[b][i + j * N].x := 1.0 + i + j + b;
            h_A_full[b][i + j * N].y := 0.5 + i - j;
          end;
        end;

      if diag = rocblas_diagonal_unit then
        for i := 0 to N - 1 do
        begin
          h_A_full[b][i + i * N].x := 1.0;
          h_A_full[b][i + i * N].y := 0.0;
        end;

      // Convert full matrix to rocBLAS banded format
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (j-i <= K) and (i <= j) then
            h_A_band_batch[b][(K + i - j) + j * lda] := h_A_full[b][i + j * N]
          else if (uplo = rocblas_fill_lower) and (i-j <= K) and (j <= i) then
            h_A_band_batch[b][(i - j) + j * lda] := h_A_full[b][i + j * N];

      for i := 0 to N - 1 do
      begin
        h_x_batch[b][i].x := 1.0 + b + i;
        h_x_batch[b][i].y := 0.5 - i;
      end;

      // CPU reference calculation
      h_x_expected[b] := h_x_batch[b];
      FillChar(h_x_result[b], SizeOf(h_x_result[b]), 0);
      for i := 0 to N - 1 do for j := 0 to N - 1 do
      begin
        case transA of
          rocblas_operation_none: temp := h_A_full[b][i + j * N];
          rocblas_operation_transpose: temp := h_A_full[b][j + i * N];
          rocblas_operation_conjugate_transpose: begin temp := h_A_full[b][j + i * N]; temp.y := -temp.y; end;
        end;
        ComplexFMA(h_x_result[b][i], temp, h_x_expected[b][j], one_complex);
      end;
      h_x_expected[b] := h_x_result[b];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_band_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_ctbmv_batched(handle, uplo, transA, diag, N, K, PPProcfloat_complex(d_A_ptrs), lda, PPProcfloat_complex(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_ctbmv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
      begin
        CheckEqualsRelative(h_x_expected[b][i].x, h_x_result[b][i].x, 1e-5, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x');
        CheckEqualsRelative(h_x_expected[b][i].y, h_x_result[b][i].y, 1e-5, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y');
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_ZTBMV_batched;
const
  N = 10;
  K = 2;
  lda = K + 1;
  incx = 1;
  batch_count = 2;
  one_complex: rocblas_double_complex = (x: 1.0; y: 0.0);
type
  TDoubleComplexBandMatrix = array[0..lda*N-1] of rocblas_double_complex;
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
  TDoubleComplexFullMatrix = array[0..N*N-1] of rocblas_double_complex;
var
  h_A_band_batch: array[0..batch_count-1] of TDoubleComplexBandMatrix;
  h_A_full: array[0..batch_count-1] of TDoubleComplexFullMatrix;
  h_x_batch, h_x_expected, h_x_result: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
  temp: rocblas_double_complex;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose, rocblas_operation_conjugate_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_A := lda * N * SizeOf(rocblas_double_complex);
    size_x := N * incx * SizeOf(rocblas_double_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      FillChar(h_A_band_batch[b], SizeOf(h_A_band_batch[b]), 0);

      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
        begin
          if (uplo = rocblas_fill_upper) and (j >= i) and (j - i <= K) then
          begin h_A_full[b][i + j * N].x := 1.0 + i + j + b; h_A_full[b][i + j * N].y := 0.5 + i - j; end
          else if (uplo = rocblas_fill_lower) and (i >= j) and (i - j <= K) then
          begin h_A_full[b][i + j * N].x := 1.0 + i + j + b; h_A_full[b][i + j * N].y := 0.5 + i - j; end;
        end;

      if diag = rocblas_diagonal_unit then
        for i := 0 to N - 1 do begin h_A_full[b][i + i * N].x := 1.0; h_A_full[b][i + i * N].y := 0.0; end;

      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (j-i <= K) and (i <= j) then h_A_band_batch[b][(K + i - j) + j * lda] := h_A_full[b][i + j * N]
          else if (uplo = rocblas_fill_lower) and (i-j <= K) and (j <= i) then h_A_band_batch[b][(i - j) + j * lda] := h_A_full[b][i + j * N];

      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 1.0 + b + i; h_x_batch[b][i].y := 0.5 - i; end;

      h_x_expected[b] := h_x_batch[b];
      FillChar(h_x_result[b], SizeOf(h_x_result[b]), 0);
      for i := 0 to N - 1 do for j := 0 to N - 1 do
      begin
        case transA of
          rocblas_operation_none: temp := h_A_full[b][i + j * N];
          rocblas_operation_transpose: temp := h_A_full[b][j + i * N];
          rocblas_operation_conjugate_transpose: begin temp := h_A_full[b][j + i * N]; temp.y := -temp.y; end;
        end;
        ComplexFMA(h_x_result[b][i], temp, h_x_expected[b][j], one_complex);
      end;
      h_x_expected[b] := h_x_result[b];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_band_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_ztbmv_batched(handle, uplo, transA, diag, N, K, PPProcdouble_complex(d_A_ptrs), lda, PPProcdouble_complex(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_ztbmv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
      begin
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x', h_x_expected[b][i].x, h_x_result[b][i].x, 1e-12);
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y', h_x_expected[b][i].y, h_x_result[b][i].y, 1e-12);
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_STPMV_batched;
const
  N = 10;
  incx = 1;
  batch_count = 2;
type
  TSinglePackedMatrix = array[0..N*(N+1) div 2 - 1] of Single;
  TSingleVector = array[0..N-1] of Single;
  TSingleFullMatrix = array[0..N*N-1] of Single;
var
  h_AP_batch: array[0..batch_count-1] of TSinglePackedMatrix;
  h_A_full: array[0..batch_count-1] of TSingleFullMatrix;
  h_x_batch, h_x_expected, h_x_result: array[0..batch_count-1] of TSingleVector;
  h_dev_AP_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  elem_val: Single;
  size_AP, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_AP := (N * (N + 1) div 2) * SizeOf(Single);
    size_x := N * incx * SizeOf(Single);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := 0 to N - 1 do
        begin
          if i = j then elem_val := 1.0 + b + i
          else if i < j then elem_val := 2.0 + i - j
          else elem_val := 2.0 + j - i;

          if (uplo = rocblas_fill_upper) and (i <= j) then       // CORRECTED: Parentheses
            h_A_full[b][i + j * N] := elem_val
          else if (uplo = rocblas_fill_lower) and (i >= j) then // CORRECTED: Parentheses
            h_A_full[b][i + j * N] := elem_val;
        end;

        if diag = rocblas_diagonal_unit then
          for i := 0 to N - 1 do h_A_full[b][i + i * N] := 1.0;

        if uplo = rocblas_fill_upper then
          for i := 0 to j do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else
          for i := j to N - 1 do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
      for i := 0 to N - 1 do h_x_batch[b][i] := 1.0 + b + i;

      h_x_expected[b] := h_x_batch[b];
      FillChar(h_x_result[b], SizeOf(h_x_result[b]), 0);
      if transA = rocblas_operation_none then
        for i := 0 to N - 1 do for j := 0 to N - 1 do
          h_x_result[b][i] := h_x_result[b][i] + h_A_full[b][i + j * N] * h_x_expected[b][j]
      else
        for i := 0 to N - 1 do for j := 0 to N - 1 do
          h_x_result[b][i] := h_x_result[b][i] + h_A_full[b][j + i * N] * h_x_expected[b][j];
      h_x_expected[b] := h_x_result[b];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_batch[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_stpmv_batched(handle, uplo, transA, diag, N, PPSingle(d_AP_ptrs), PPSingle(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_stpmv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
        CheckEqualsRelative(h_x_expected[b][i], h_x_result[b][i], 1e-5, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i));
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_DTPMV_batched;
const
  N = 10;
  incx = 1;
  batch_count = 2;
type
  TDoublePackedMatrix = array[0..N*(N+1) div 2 - 1] of Double;
  TDoubleVector = array[0..N-1] of Double;
  TDoubleFullMatrix = array[0..N*N-1] of Double;
var
  h_AP_batch: array[0..batch_count-1] of TDoublePackedMatrix;
  h_A_full: array[0..batch_count-1] of TDoubleFullMatrix;
  h_x_batch, h_x_expected, h_x_result: array[0..batch_count-1] of TDoubleVector;
  h_dev_AP_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  elem_val: Double;
  size_AP, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_AP := (N * (N + 1) div 2) * SizeOf(Double);
    size_x := N * incx * SizeOf(Double);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := 0 to N - 1 do
        begin
          if i = j then elem_val := 1.0 + b + i
          else if i < j then elem_val := 2.0 + i - j
          else elem_val := 2.0 + j - i;
          if (uplo = rocblas_fill_upper) and (i <= j) then h_A_full[b][i + j * N] := elem_val
          else if (uplo = rocblas_fill_lower) and (i >= j) then h_A_full[b][i + j * N] := elem_val;
        end;

        if diag = rocblas_diagonal_unit then for i := 0 to N - 1 do h_A_full[b][i + i * N] := 1.0;

        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
      for i := 0 to N - 1 do h_x_batch[b][i] := 1.0 + b + i;

      h_x_expected[b] := h_x_batch[b];
      FillChar(h_x_result[b], SizeOf(h_x_result[b]), 0);
      if transA = rocblas_operation_none then for i := 0 to N - 1 do for j := 0 to N - 1 do h_x_result[b][i] := h_x_result[b][i] + h_A_full[b][i + j * N] * h_x_expected[b][j]
      else for i := 0 to N - 1 do for j := 0 to N - 1 do h_x_result[b][i] := h_x_result[b][i] + h_A_full[b][j + i * N] * h_x_expected[b][j];
      h_x_expected[b] := h_x_result[b];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_batch[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_dtpmv_batched(handle, uplo, transA, diag, N, PPDouble(d_AP_ptrs), PPDouble(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_dtpmv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i), h_x_expected[b][i], h_x_result[b][i], 1e-12);
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_CTPMV_batched;
const
  N = 10;
  incx = 1;
  batch_count = 2;
  one_complex: rocblas_float_complex = (x: 1.0; y: 0.0);
type
  TComplexPackedMatrix = array[0..N*(N+1) div 2 - 1] of rocblas_float_complex;
  TComplexVector = array[0..N-1] of rocblas_float_complex;
  TComplexFullMatrix = array[0..N*N-1] of rocblas_float_complex;
var
  h_AP_batch: array[0..batch_count-1] of TComplexPackedMatrix;
  h_A_full: array[0..batch_count-1] of TComplexFullMatrix;
  h_x_batch, h_x_expected, h_x_result: array[0..batch_count-1] of TComplexVector;
  h_dev_AP_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  size_AP, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
  temp: rocblas_float_complex;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose, rocblas_operation_conjugate_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_AP := (N * (N + 1) div 2) * SizeOf(rocblas_float_complex);
    size_x := N * incx * SizeOf(rocblas_float_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (i <= j) then
          begin
            h_A_full[b][i + j*N].x := 1+i+j+b; h_A_full[b][i + j*N].y := 0.5+i-j;
          end
          else if (uplo = rocblas_fill_lower) and (i >= j) then
          begin
            h_A_full[b][i + j*N].x := 1+i+j+b; h_A_full[b][i + j*N].y := 0.5+i-j;
          end;

        if diag = rocblas_diagonal_unit then
          for i := 0 to N - 1 do begin h_A_full[b][i + i * N].x := 1.0; h_A_full[b][i + i*N].y := 0.0; end;

        if uplo = rocblas_fill_upper then
          for i := 0 to j do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else
          for i := j to N - 1 do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 1+b+i; h_x_batch[b][i].y := 0.5-i; end;

      h_x_expected[b] := h_x_batch[b];
      FillChar(h_x_result[b], SizeOf(h_x_result[b]), 0);
      for i := 0 to N - 1 do for j := 0 to N - 1 do
      begin
        case transA of
          rocblas_operation_none:
            temp := h_A_full[b][i + j * N];
          rocblas_operation_transpose:
            temp := h_A_full[b][j + i * N];
          rocblas_operation_conjugate_transpose:
            begin temp := h_A_full[b][j + i * N]; temp.y := -temp.y; end;
        end;
        ComplexFMA(h_x_result[b][i], temp, h_x_expected[b][j], one_complex);
      end;
      h_x_expected[b] := h_x_result[b];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_batch[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_ctpmv_batched(handle, uplo, transA, diag, N, PPProcfloat_complex(d_AP_ptrs), PPProcfloat_complex(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_ctpmv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
      begin
        CheckEqualsRelative(h_x_expected[b][i].x, h_x_result[b][i].x, 1e-5, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x');
        CheckEqualsRelative(h_x_expected[b][i].y, h_x_result[b][i].y, 1e-5, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y');
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_ZTPMV_batched;
const
  N = 10;
  incx = 1;
  batch_count = 2;
  one_complex: rocblas_double_complex = (x: 1.0; y: 0.0);
type
  TDoubleComplexPackedMatrix = array[0..N*(N+1) div 2 - 1] of rocblas_double_complex;
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
  TDoubleComplexFullMatrix = array[0..N*N-1] of rocblas_double_complex;
var
  h_AP_batch: array[0..batch_count-1] of TDoubleComplexPackedMatrix;
  h_A_full: array[0..batch_count-1] of TDoubleComplexFullMatrix;
  h_x_batch, h_x_expected, h_x_result: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_AP_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  size_AP, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
  temp: rocblas_double_complex;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose, rocblas_operation_conjugate_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_AP := (N * (N + 1) div 2) * SizeOf(rocblas_double_complex);
    size_x := N * incx * SizeOf(rocblas_double_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (i <= j) then
          begin h_A_full[b][i + j*N].x := 1+i+j+b; h_A_full[b][i + j*N].y := 0.5+i-j; end
          else if (uplo = rocblas_fill_lower) and (i >= j) then
          begin h_A_full[b][i + j*N].x := 1+i+j+b; h_A_full[b][i + j*N].y := 0.5+i-j; end;
        if diag = rocblas_diagonal_unit then for i := 0 to N - 1 do begin h_A_full[b][i + i * N].x := 1.0; h_A_full[b][i + i*N].y := 0.0; end;
        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 1+b+i; h_x_batch[b][i].y := 0.5-i; end;

      h_x_expected[b] := h_x_batch[b];
      FillChar(h_x_result[b], SizeOf(h_x_result[b]), 0);
      for i := 0 to N - 1 do for j := 0 to N - 1 do
      begin
        case transA of
          rocblas_operation_none: temp := h_A_full[b][i + j * N];
          rocblas_operation_transpose: temp := h_A_full[b][j + i * N];
          rocblas_operation_conjugate_transpose: begin temp := h_A_full[b][j + i * N]; temp.y := -temp.y; end;
        end;
        ComplexFMA(h_x_result[b][i], temp, h_x_expected[b][j], one_complex);
      end;
      h_x_expected[b] := h_x_result[b];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_batch[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_ztpmv_batched(handle, uplo, transA, diag, N, PPProcdouble_complex(d_AP_ptrs), PPProcdouble_complex(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_ztpmv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
      begin
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x', h_x_expected[b][i].x, h_x_result[b][i].x, 1e-12);
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y', h_x_expected[b][i].y, h_x_result[b][i].y, 1e-12);
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_STRSV_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  batch_count = 2;
type
  TSingleMatrix = array[0..lda*N-1] of Single;
  TSingleVector = array[0..N-1] of Single;
var
  h_A_batch: array[0..batch_count-1] of TSingleMatrix;
  h_x_sol_batch, h_b_batch, h_x_result: array[0..batch_count-1] of TSingleVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_A := lda * N * SizeOf(Single);
    size_x := N * incx * SizeOf(Single);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_batch[b], SizeOf(h_A_batch[b]), 0);
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
        begin
          if (uplo = rocblas_fill_upper) and (i <= j) then
            h_A_batch[b][i + j * lda] := 1.0 + i + j + b
          else if (uplo = rocblas_fill_lower) and (i >= j) then
            h_A_batch[b][i + j * lda] := 1.0 + i + j + b;
        end;

      if diag = rocblas_diagonal_unit then
        for i := 0 to N - 1 do h_A_batch[b][i + i * lda] := 1.0;

      for i := 0 to N - 1 do h_x_sol_batch[b][i] := 1.0 + b + i;

      // CPU calculation of b = op(A) * x_sol
      FillChar(h_b_batch[b], SizeOf(h_b_batch[b]), 0);
      if transA = rocblas_operation_none then
        for i := 0 to N - 1 do for j := 0 to N - 1 do
          h_b_batch[b][i] := h_b_batch[b][i] + h_A_batch[b][i + j * lda] * h_x_sol_batch[b][j]
      else
        for i := 0 to N - 1 do for j := 0 to N - 1 do
          h_b_batch[b][i] := h_b_batch[b][i] + h_A_batch[b][j + i * lda] * h_x_sol_batch[b][j];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_b_batch[b][0], size_x, hipMemcpyHostToDevice); // Copy b to device x
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_strsv_batched(handle, uplo, transA, diag, N, PPSingle(d_A_ptrs), lda, PPSingle(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_strsv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
        CheckEqualsRelative(h_x_sol_batch[b][i], h_x_result[b][i], 1e-4, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i));
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_DTRSV_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  batch_count = 2;
type
  TDoubleMatrix = array[0..lda*N-1] of Double;
  TDoubleVector = array[0..N-1] of Double;
var
  h_A_batch: array[0..batch_count-1] of TDoubleMatrix;
  h_x_sol_batch, h_b_batch, h_x_result: array[0..batch_count-1] of TDoubleVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_A := lda * N * SizeOf(Double);
    size_x := N * incx * SizeOf(Double);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_batch[b], SizeOf(h_A_batch[b]), 0);
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
        begin
          if (uplo = rocblas_fill_upper) and (i <= j) then
            h_A_batch[b][i + j * lda] := 1.0 + i + j + b
          else if (uplo = rocblas_fill_lower) and (i >= j) then
            h_A_batch[b][i + j * lda] := 1.0 + i + j + b;
        end;

      if diag = rocblas_diagonal_unit then
        for i := 0 to N - 1 do h_A_batch[b][i + i * lda] := 1.0;

      for i := 0 to N - 1 do h_x_sol_batch[b][i] := 1.0 + b + i;

      // CPU calculation of b = op(A) * x_sol
      FillChar(h_b_batch[b], SizeOf(h_b_batch[b]), 0);
      if transA = rocblas_operation_none then
        for i := 0 to N - 1 do for j := 0 to N - 1 do
          h_b_batch[b][i] := h_b_batch[b][i] + h_A_batch[b][i + j * lda] * h_x_sol_batch[b][j]
      else
        for i := 0 to N - 1 do for j := 0 to N - 1 do
          h_b_batch[b][i] := h_b_batch[b][i] + h_A_batch[b][j + i * lda] * h_x_sol_batch[b][j];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_b_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_dtrsv_batched(handle, uplo, transA, diag, N, PPDouble(d_A_ptrs), lda, PPDouble(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_dtrsv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i), h_x_sol_batch[b][i], h_x_result[b][i], 1e-12);
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_CTRSV_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  batch_count = 2;
  one_complex: rocblas_float_complex = (x: 1.0; y: 0.0);
type
  TComplexMatrix = array[0..lda*N-1] of rocblas_float_complex;
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_A_batch: array[0..batch_count-1] of TComplexMatrix;
  h_x_sol_batch, h_b_batch, h_x_result: array[0..batch_count-1] of TComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
  temp: rocblas_float_complex;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose, rocblas_operation_conjugate_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_A := lda * N * SizeOf(rocblas_float_complex);
    size_x := N * incx * SizeOf(rocblas_float_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_batch[b], SizeOf(h_A_batch[b]), 0);
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
        begin
          if (uplo = rocblas_fill_upper) and (i <= j) then
          begin h_A_batch[b][i + j * lda].x := 2.0 + i + j + b; h_A_batch[b][i + j * lda].y := 0.5 + i - j; end
          else if (uplo = rocblas_fill_lower) and (i >= j) then
          begin h_A_batch[b][i + j * lda].x := 2.0 + i + j + b; h_A_batch[b][i + j * lda].y := 0.5 + i - j; end;
        end;

      if diag = rocblas_diagonal_unit then
        for i := 0 to N - 1 do begin h_A_batch[b][i + i * lda].x := 1.0; h_A_batch[b][i + i * lda].y := 0.0; end;

      for i := 0 to N - 1 do begin h_x_sol_batch[b][i].x := 1.0 + b + i; h_x_sol_batch[b][i].y := 0.5 - i; end;

      FillChar(h_b_batch[b], SizeOf(h_b_batch[b]), 0);
      for i := 0 to N - 1 do for j := 0 to N - 1 do
      begin
        case transA of
          rocblas_operation_none: temp := h_A_batch[b][i + j * lda];
          rocblas_operation_transpose: temp := h_A_batch[b][j + i * lda];
          rocblas_operation_conjugate_transpose: begin temp := h_A_batch[b][j + i * lda]; temp.y := -temp.y; end;
        end;
        ComplexFMA(h_b_batch[b][i], temp, h_x_sol_batch[b][j], one_complex);
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_b_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_ctrsv_batched(handle, uplo, transA, diag, N, PPProcfloat_complex(d_A_ptrs), lda, PPProcfloat_complex(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_ctrsv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
      begin
        CheckEqualsRelative(h_x_sol_batch[b][i].x, h_x_result[b][i].x, 1e-4, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x');
        CheckEqualsRelative(h_x_sol_batch[b][i].y, h_x_result[b][i].y, 1e-4, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y');
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_ZTRSV_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  batch_count = 2;
  one_complex: rocblas_double_complex = (x: 1.0; y: 0.0);
type
  TDoubleComplexMatrix = array[0..lda*N-1] of rocblas_double_complex;
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_A_batch: array[0..batch_count-1] of TDoubleComplexMatrix;
  h_x_sol_batch, h_b_batch, h_x_result: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
  temp: rocblas_double_complex;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose, rocblas_operation_conjugate_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_A := lda * N * SizeOf(rocblas_double_complex);
    size_x := N * incx * SizeOf(rocblas_double_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_batch[b], SizeOf(h_A_batch[b]), 0);
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
        begin
          if (uplo = rocblas_fill_upper) and (i <= j) then
          begin h_A_batch[b][i + j * lda].x := 2.0 + i + j + b; h_A_batch[b][i + j * lda].y := 0.5 + i - j; end
          else if (uplo = rocblas_fill_lower) and (i >= j) then
          begin h_A_batch[b][i + j * lda].x := 2.0 + i + j + b; h_A_batch[b][i + j * lda].y := 0.5 + i - j; end;
        end;

      if diag = rocblas_diagonal_unit then
        for i := 0 to N - 1 do begin h_A_batch[b][i + i * lda].x := 1.0; h_A_batch[b][i + i * lda].y := 0.0; end;

      for i := 0 to N - 1 do begin h_x_sol_batch[b][i].x := 1.0 + b + i; h_x_sol_batch[b][i].y := 0.5 - i; end;

      FillChar(h_b_batch[b], SizeOf(h_b_batch[b]), 0);
      for i := 0 to N - 1 do for j := 0 to N - 1 do
      begin
        case transA of
          rocblas_operation_none: temp := h_A_batch[b][i + j * lda];
          rocblas_operation_transpose: temp := h_A_batch[b][j + i * lda];
          rocblas_operation_conjugate_transpose: begin temp := h_A_batch[b][j + i * lda]; temp.y := -temp.y; end;
        end;
        ComplexFMA(h_b_batch[b][i], temp, h_x_sol_batch[b][j], one_complex);
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_b_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_ztrsv_batched(handle, uplo, transA, diag, N, PPProcdouble_complex(d_A_ptrs), lda, PPProcdouble_complex(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_ztrsv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
      begin
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x', h_x_sol_batch[b][i].x, h_x_result[b][i].x, 1e-12);
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y', h_x_sol_batch[b][i].y, h_x_result[b][i].y, 1e-12);
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_STBSV_batched;
const
  N = 10;
  K = 2;
  lda = K + 1;
  incx = 1;
  batch_count = 2;
type
  TSingleBandMatrix = array[0..lda*N-1] of Single;
  TSingleVector = array[0..N-1] of Single;
  TSingleFullMatrix = array[0..N*N-1] of Single;
var
  h_A_band_batch: array[0..batch_count-1] of TSingleBandMatrix;
  h_A_full: array[0..batch_count-1] of TSingleFullMatrix;
  h_x_sol_batch, h_b_batch, h_x_result: array[0..batch_count-1] of TSingleVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_A := lda * N * SizeOf(Single);
    size_x := N * incx * SizeOf(Single);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      FillChar(h_A_band_batch[b], SizeOf(h_A_band_batch[b]), 0);

      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (j >= i) and (j - i <= K) then
            h_A_full[b][i + j * N] := 2.0 + i + j + b
          else if (uplo = rocblas_fill_lower) and (i >= j) and (i - j <= K) then
            h_A_full[b][i + j * N] := 2.0 + i + j + b;

      if diag = rocblas_diagonal_unit then
        for i := 0 to N - 1 do h_A_full[b][i + i * N] := 1.0;

      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (j-i <= K) and (i <= j) then
            h_A_band_batch[b][(K + i - j) + j * lda] := h_A_full[b][i + j * N]
          else if (uplo = rocblas_fill_lower) and (i-j <= K) and (j <= i) then
            h_A_band_batch[b][(i - j) + j * lda] := h_A_full[b][i + j * N];

      for i := 0 to N - 1 do h_x_sol_batch[b][i] := 1.0 + b + i;

      FillChar(h_b_batch[b], SizeOf(h_b_batch[b]), 0);
      if transA = rocblas_operation_none then
        for i := 0 to N - 1 do for j := 0 to N - 1 do
          h_b_batch[b][i] := h_b_batch[b][i] + h_A_full[b][i + j * N] * h_x_sol_batch[b][j]
      else
        for i := 0 to N - 1 do for j := 0 to N - 1 do
          h_b_batch[b][i] := h_b_batch[b][i] + h_A_full[b][j + i * N] * h_x_sol_batch[b][j];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_band_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_b_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_stbsv_batched(handle, uplo, transA, diag, N, K, PPSingle(d_A_ptrs), lda, PPSingle(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_stbsv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
        CheckEqualsRelative(h_x_sol_batch[b][i], h_x_result[b][i], 1e-4, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i));
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_DTBSV_batched;
const
  N = 10;
  K = 2;
  lda = K + 1;
  incx = 1;
  batch_count = 2;
type
  TDoubleBandMatrix = array[0..lda*N-1] of Double;
  TDoubleVector = array[0..N-1] of Double;
  TDoubleFullMatrix = array[0..N*N-1] of Double;
var
  h_A_band_batch: array[0..batch_count-1] of TDoubleBandMatrix;
  h_A_full: array[0..batch_count-1] of TDoubleFullMatrix;
  h_x_sol_batch, h_b_batch, h_x_result: array[0..batch_count-1] of TDoubleVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_A := lda * N * SizeOf(Double);
    size_x := N * incx * SizeOf(Double);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      FillChar(h_A_band_batch[b], SizeOf(h_A_band_batch[b]), 0);

      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (j >= i) and (j - i <= K) then
            h_A_full[b][i + j * N] := 2.0 + i + j + b
          else if (uplo = rocblas_fill_lower) and (i >= j) and (i - j <= K) then
            h_A_full[b][i + j * N] := 2.0 + i + j + b;

      if diag = rocblas_diagonal_unit then
        for i := 0 to N - 1 do h_A_full[b][i + i * N] := 1.0;

      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (j-i <= K) and (i <= j) then
            h_A_band_batch[b][(K + i - j) + j * lda] := h_A_full[b][i + j * N]
          else if (uplo = rocblas_fill_lower) and (i-j <= K) and (j <= i) then
            h_A_band_batch[b][(i - j) + j * lda] := h_A_full[b][i + j * N];

      for i := 0 to N - 1 do h_x_sol_batch[b][i] := 1.0 + b + i;

      FillChar(h_b_batch[b], SizeOf(h_b_batch[b]), 0);
      if transA = rocblas_operation_none then
        for i := 0 to N - 1 do for j := 0 to N - 1 do
          h_b_batch[b][i] := h_b_batch[b][i] + h_A_full[b][i + j * N] * h_x_sol_batch[b][j]
      else
        for i := 0 to N - 1 do for j := 0 to N - 1 do
          h_b_batch[b][i] := h_b_batch[b][i] + h_A_full[b][j + i * N] * h_x_sol_batch[b][j];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_band_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_b_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_dtbsv_batched(handle, uplo, transA, diag, N, K, PPDouble(d_A_ptrs), lda, PPDouble(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_dtbsv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i), h_x_sol_batch[b][i], h_x_result[b][i], 1e-12);
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_CTBSV_batched;
const
  N = 10;
  K = 2;
  lda = K + 1;
  incx = 1;
  batch_count = 2;
  one_complex: rocblas_float_complex = (x: 1.0; y: 0.0);
type
  TComplexBandMatrix = array[0..lda*N-1] of rocblas_float_complex;
  TComplexVector = array[0..N-1] of rocblas_float_complex;
  TComplexFullMatrix = array[0..N*N-1] of rocblas_float_complex;
var
  h_A_band_batch: array[0..batch_count-1] of TComplexBandMatrix;
  h_A_full: array[0..batch_count-1] of TComplexFullMatrix;
  h_x_sol_batch, h_b_batch, h_x_result: array[0..batch_count-1] of TComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
  temp: rocblas_float_complex;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose, rocblas_operation_conjugate_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_A := lda * N * SizeOf(rocblas_float_complex);
    size_x := N * incx * SizeOf(rocblas_float_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      FillChar(h_A_band_batch[b], SizeOf(h_A_band_batch[b]), 0);

      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
        begin
          if (uplo = rocblas_fill_upper) and (j >= i) and (j - i <= K) then
          begin h_A_full[b][i + j * N].x := 2.0 + i + j + b; h_A_full[b][i + j * N].y := 0.5 + i - j; end
          else if (uplo = rocblas_fill_lower) and (i >= j) and (i - j <= K) then
          begin h_A_full[b][i + j * N].x := 2.0 + i + j + b; h_A_full[b][i + j * N].y := 0.5 + i - j; end;
        end;

      if diag = rocblas_diagonal_unit then
        for i := 0 to N - 1 do begin h_A_full[b][i + i * N].x := 1.0; h_A_full[b][i + i * N].y := 0.0; end;

      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (j-i <= K) and (i <= j) then
            h_A_band_batch[b][(K + i - j) + j * lda] := h_A_full[b][i + j * N]
          else if (uplo = rocblas_fill_lower) and (i-j <= K) and (j <= i) then
            h_A_band_batch[b][(i - j) + j * lda] := h_A_full[b][i + j * N];

      for i := 0 to N - 1 do begin h_x_sol_batch[b][i].x := 1.0 + b + i; h_x_sol_batch[b][i].y := 0.5 - i; end;

      FillChar(h_b_batch[b], SizeOf(h_b_batch[b]), 0);
      for i := 0 to N - 1 do for j := 0 to N - 1 do
      begin
        case transA of
          rocblas_operation_none: temp := h_A_full[b][i + j * N];
          rocblas_operation_transpose: temp := h_A_full[b][j + i * N];
          rocblas_operation_conjugate_transpose: begin temp := h_A_full[b][j + i * N]; temp.y := -temp.y; end;
        end;
        ComplexFMA(h_b_batch[b][i], temp, h_x_sol_batch[b][j], one_complex);
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_band_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_b_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_ctbsv_batched(handle, uplo, transA, diag, N, K, PPProcfloat_complex(d_A_ptrs), lda, PPProcfloat_complex(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_ctbsv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
      begin
        CheckEqualsRelative(h_x_sol_batch[b][i].x, h_x_result[b][i].x, 1e-4, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x');
        CheckEqualsRelative(h_x_sol_batch[b][i].y, h_x_result[b][i].y, 1e-4, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y');
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_ZTBSV_batched;
const
  N = 10;
  K = 2;
  lda = K + 1;
  incx = 1;
  batch_count = 2;
  one_complex: rocblas_double_complex = (x: 1.0; y: 0.0);
type
  TDoubleComplexBandMatrix = array[0..lda*N-1] of rocblas_double_complex;
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
  TDoubleComplexFullMatrix = array[0..N*N-1] of rocblas_double_complex;
var
  h_A_band_batch: array[0..batch_count-1] of TDoubleComplexBandMatrix;
  h_A_full: array[0..batch_count-1] of TDoubleComplexFullMatrix;
  h_x_sol_batch, h_b_batch, h_x_result: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
  temp: rocblas_double_complex;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose, rocblas_operation_conjugate_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_A := lda * N * SizeOf(rocblas_double_complex);
    size_x := N * incx * SizeOf(rocblas_double_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      FillChar(h_A_band_batch[b], SizeOf(h_A_band_batch[b]), 0);

      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
        begin
          if (uplo = rocblas_fill_upper) and (j >= i) and (j - i <= K) then
          begin h_A_full[b][i + j * N].x := 2.0 + i + j + b; h_A_full[b][i + j * N].y := 0.5 + i - j; end
          else if (uplo = rocblas_fill_lower) and (i >= j) and (i - j <= K) then
          begin h_A_full[b][i + j * N].x := 2.0 + i + j + b; h_A_full[b][i + j * N].y := 0.5 + i - j; end;
        end;

      if diag = rocblas_diagonal_unit then
        for i := 0 to N - 1 do begin h_A_full[b][i + i * N].x := 1.0; h_A_full[b][i + i * N].y := 0.0; end;

      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (j-i <= K) and (i <= j) then
            h_A_band_batch[b][(K + i - j) + j * lda] := h_A_full[b][i + j * N]
          else if (uplo = rocblas_fill_lower) and (i-j <= K) and (j <= i) then
            h_A_band_batch[b][(i - j) + j * lda] := h_A_full[b][i + j * N];

      for i := 0 to N - 1 do begin h_x_sol_batch[b][i].x := 1.0 + b + i; h_x_sol_batch[b][i].y := 0.5 - i; end;

      FillChar(h_b_batch[b], SizeOf(h_b_batch[b]), 0);
      for i := 0 to N - 1 do for j := 0 to N - 1 do
      begin
        case transA of
          rocblas_operation_none: temp := h_A_full[b][i + j * N];
          rocblas_operation_transpose: temp := h_A_full[b][j + i * N];
          rocblas_operation_conjugate_transpose: begin temp := h_A_full[b][j + i * N]; temp.y := -temp.y; end;
        end;
        ComplexFMA(h_b_batch[b][i], temp, h_x_sol_batch[b][j], one_complex);
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_band_batch[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_b_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_ztbsv_batched(handle, uplo, transA, diag, N, K, PPProcdouble_complex(d_A_ptrs), lda, PPProcdouble_complex(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_ztbsv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
      begin
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x', h_x_sol_batch[b][i].x, h_x_result[b][i].x, 1e-12);
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y', h_x_sol_batch[b][i].y, h_x_result[b][i].y, 1e-12);
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_STPSV_batched;
const
  N = 10;
  incx = 1;
  batch_count = 2;
type
  TSinglePackedMatrix = array[0..N*(N+1) div 2 - 1] of Single;
  TSingleVector = array[0..N-1] of Single;
  TSingleFullMatrix = array[0..N*N-1] of Single;
var
  h_AP_batch: array[0..batch_count-1] of TSinglePackedMatrix;
  h_A_full: array[0..batch_count-1] of TSingleFullMatrix;
  h_x_sol_batch, h_b_batch, h_x_result: array[0..batch_count-1] of TSingleVector;
  h_dev_AP_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  size_AP, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_AP := (N * (N + 1) div 2) * SizeOf(Single);
    size_x := N * incx * SizeOf(Single);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (i <= j) then h_A_full[b][i + j * N] := 2.0 + i + j + b
          else if (uplo = rocblas_fill_lower) and (i >= j) then h_A_full[b][i + j * N] := 2.0 + i + j + b;

        if diag = rocblas_diagonal_unit then h_A_full[b][j + j * N] := 1.0;

        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;

      for i := 0 to N - 1 do h_x_sol_batch[b][i] := 1.0 + b + i;

      FillChar(h_b_batch[b], SizeOf(h_b_batch[b]), 0);
      if transA = rocblas_operation_none then for i := 0 to N - 1 do for j := 0 to N - 1 do h_b_batch[b][i] := h_b_batch[b][i] + h_A_full[b][i + j * N] * h_x_sol_batch[b][j]
      else for i := 0 to N - 1 do for j := 0 to N - 1 do h_b_batch[b][i] := h_b_batch[b][i] + h_A_full[b][j + i * N] * h_x_sol_batch[b][j];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_batch[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_b_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_stpsv_batched(handle, uplo, transA, diag, N, PPSingle(d_AP_ptrs), PPSingle(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_stpsv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
        CheckEqualsRelative(h_x_sol_batch[b][i], h_x_result[b][i], 1e-4, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i));
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_DTPSV_batched;
const
  N = 10;
  incx = 1;
  batch_count = 2;
type
  TDoublePackedMatrix = array[0..N*(N+1) div 2 - 1] of Double;
  TDoubleVector = array[0..N-1] of Double;
  TDoubleFullMatrix = array[0..N*N-1] of Double;
var
  h_AP_batch: array[0..batch_count-1] of TDoublePackedMatrix;
  h_A_full: array[0..batch_count-1] of TDoubleFullMatrix;
  h_x_sol_batch, h_b_batch, h_x_result: array[0..batch_count-1] of TDoubleVector;
  h_dev_AP_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  size_AP, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_AP := (N * (N + 1) div 2) * SizeOf(Double);
    size_x := N * incx * SizeOf(Double);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (i <= j) then h_A_full[b][i + j * N] := 2.0 + i + j + b
          else if (uplo = rocblas_fill_lower) and (i >= j) then h_A_full[b][i + j * N] := 2.0 + i + j + b;

        if diag = rocblas_diagonal_unit then h_A_full[b][j + j * N] := 1.0;

        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;

      for i := 0 to N - 1 do h_x_sol_batch[b][i] := 1.0 + b + i;

      FillChar(h_b_batch[b], SizeOf(h_b_batch[b]), 0);
      if transA = rocblas_operation_none then for i := 0 to N - 1 do for j := 0 to N - 1 do h_b_batch[b][i] := h_b_batch[b][i] + h_A_full[b][i + j * N] * h_x_sol_batch[b][j]
      else for i := 0 to N - 1 do for j := 0 to N - 1 do h_b_batch[b][i] := h_b_batch[b][i] + h_A_full[b][j + i * N] * h_x_sol_batch[b][j];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_batch[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_b_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_dtpsv_batched(handle, uplo, transA, diag, N, PPDouble(d_AP_ptrs), PPDouble(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_dtpsv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i), h_x_sol_batch[b][i], h_x_result[b][i], 1e-12);
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_CTPSV_batched;
const
  N = 10;
  incx = 1;
  batch_count = 2;
  one_complex: rocblas_float_complex = (x: 1.0; y: 0.0);
type
  TComplexPackedMatrix = array[0..N*(N+1) div 2 - 1] of rocblas_float_complex;
  TComplexVector = array[0..N-1] of rocblas_float_complex;
  TComplexFullMatrix = array[0..N*N-1] of rocblas_float_complex;
var
  h_AP_batch: array[0..batch_count-1] of TComplexPackedMatrix;
  h_A_full: array[0..batch_count-1] of TComplexFullMatrix;
  h_x_sol_batch, h_b_batch, h_x_result: array[0..batch_count-1] of TComplexVector;
  h_dev_AP_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  size_AP, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
  temp: rocblas_float_complex;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose, rocblas_operation_conjugate_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_AP := (N * (N + 1) div 2) * SizeOf(rocblas_float_complex);
    size_x := N * incx * SizeOf(rocblas_float_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (i <= j) then
          begin h_A_full[b][i + j*N].x := 2.0+i+j+b; h_A_full[b][i + j*N].y := 0.5+i-j; end
          else if (uplo = rocblas_fill_lower) and (i >= j) then
          begin h_A_full[b][i + j*N].x := 2.0+i+j+b; h_A_full[b][i + j*N].y := 0.5+i-j; end;
        if diag = rocblas_diagonal_unit then begin h_A_full[b][j + j * N].x := 1.0; h_A_full[b][j + j*N].y := 0.0; end;

        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
      for i := 0 to N - 1 do begin h_x_sol_batch[b][i].x := 1+b+i; h_x_sol_batch[b][i].y := 0.5-i; end;

      FillChar(h_b_batch[b], SizeOf(h_b_batch[b]), 0);
      for i := 0 to N - 1 do for j := 0 to N - 1 do
      begin
        case transA of
          rocblas_operation_none: temp := h_A_full[b][i + j * N];
          rocblas_operation_transpose: temp := h_A_full[b][j + i * N];
          rocblas_operation_conjugate_transpose: begin temp := h_A_full[b][j + i * N]; temp.y := -temp.y; end;
        end;
        ComplexFMA(h_b_batch[b][i], temp, h_x_sol_batch[b][j], one_complex);
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_batch[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_b_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_ctpsv_batched(handle, uplo, transA, diag, N, PPProcfloat_complex(d_AP_ptrs), PPProcfloat_complex(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_ctpsv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
      begin
        CheckEqualsRelative(h_x_sol_batch[b][i].x, h_x_result[b][i].x, 1e-4, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x');
        CheckEqualsRelative(h_x_sol_batch[b][i].y, h_x_result[b][i].y, 1e-4, msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y');
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_ZTPSV_batched;
const
  N = 10;
  incx = 1;
  batch_count = 2;
  one_complex: rocblas_double_complex = (x: 1.0; y: 0.0);
type
  TDoubleComplexPackedMatrix = array[0..N*(N+1) div 2 - 1] of rocblas_double_complex;
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
  TDoubleComplexFullMatrix = array[0..N*N-1] of rocblas_double_complex;
var
  h_AP_batch: array[0..batch_count-1] of TDoubleComplexPackedMatrix;
  h_A_full: array[0..batch_count-1] of TDoubleComplexFullMatrix;
  h_x_sol_batch, h_b_batch, h_x_result: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_AP_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs: hipDeviceptr_t;
  i, j, b, k: Integer;
  size_AP, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  msg: string;
  temp: rocblas_double_complex;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  for transA in [rocblas_operation_none, rocblas_operation_transpose, rocblas_operation_conjugate_transpose] do
  for diag in [rocblas_diagonal_non_unit, rocblas_diagonal_unit] do
  begin
    msg := Format('uplo=%d, transA=%d, diag=%d', [Ord(uplo), Ord(transA), Ord(diag)]);
    size_AP := (N * (N + 1) div 2) * SizeOf(rocblas_double_complex);
    size_x := N * incx * SizeOf(rocblas_double_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      FillChar(h_A_full[b], SizeOf(h_A_full[b]), 0);
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := 0 to N - 1 do
          if (uplo = rocblas_fill_upper) and (i <= j) then
          begin h_A_full[b][i + j*N].x := 2.0+i+j+b; h_A_full[b][i + j*N].y := 0.5+i-j; end
          else if (uplo = rocblas_fill_lower) and (i >= j) then
          begin h_A_full[b][i + j*N].x := 2.0+i+j+b; h_A_full[b][i + j*N].y := 0.5+i-j; end;
        if diag = rocblas_diagonal_unit then begin h_A_full[b][j + j * N].x := 1.0; h_A_full[b][j + j*N].y := 0.0; end;

        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_batch[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
      for i := 0 to N - 1 do begin h_x_sol_batch[b][i].x := 1+b+i; h_x_sol_batch[b][i].y := 0.5-i; end;

      FillChar(h_b_batch[b], SizeOf(h_b_batch[b]), 0);
      for i := 0 to N - 1 do for j := 0 to N - 1 do
      begin
        case transA of
          rocblas_operation_none: temp := h_A_full[b][i + j * N];
          rocblas_operation_transpose: temp := h_A_full[b][j + i * N];
          rocblas_operation_conjugate_transpose: begin temp := h_A_full[b][j + i * N]; temp.y := -temp.y; end;
        end;
        ComplexFMA(h_b_batch[b][i], temp, h_x_sol_batch[b][j], one_complex);
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_batch[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_b_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_ztpsv_batched(handle, uplo, transA, diag, N, PPProcdouble_complex(d_AP_ptrs), PPProcdouble_complex(d_x_ptrs), incx, batch_count);
    AssertEquals('rocblas_ztpsv_batched failed: ' + msg, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_x_result[b][0], h_dev_x_ptrs[b], size_x, hipMemcpyDeviceToHost);
      for i := 0 to N - 1 do
      begin
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x', h_x_sol_batch[b][i].x, h_x_result[b][i].x, 1e-12);
        AssertEquals(msg + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y', h_x_sol_batch[b][i].y, h_x_result[b][i].y, 1e-12);
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_SGER_batched;
const
  M = 5;
  N = 10;
  lda = M;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TSingleMatrix = array[0..lda*N-1] of Single;
  TSingleVectorX = array[0..M-1] of Single;
  TSingleVectorY = array[0..N-1] of Single;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TSingleMatrix;
  h_x_batch: array[0..batch_count-1] of TSingleVectorX;
  h_y_batch: array[0..batch_count-1] of TSingleVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha: Single;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  alpha := 1.5;
  size_A := lda * N * SizeOf(Single);
  size_x := M * incx * SizeOf(Single);
  size_y := N * incy * SizeOf(Single);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
        h_A_initial[b][i + j * lda] := 1.0 + b + i + j;
    for i := 0 to M - 1 do h_x_batch[b][i] := 0.5 + b + i;
    for i := 0 to N - 1 do h_y_batch[b][i] := 0.5 + b + i;

    h_A_expected[b] := h_A_initial[b];
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
        h_A_expected[b][i + j * lda] := h_A_expected[b][i + j * lda] + alpha * h_x_batch[b][i] * h_y_batch[b][j];
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
    hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
  end;
  hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
  hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_sger_batched(handle, M, N, @alpha, PPSingle(d_x_ptrs), incx, PPSingle(d_y_ptrs), incy, PPSingle(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_sger_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
    for i := 0 to lda * N - 1 do
      CheckEqualsRelative(h_A_expected[b][i], h_A_result[b][i], 1e-5, 'Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i));
  end;

  for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_DGER_batched;
const
  M = 5;
  N = 10;
  lda = M;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TDoubleMatrix = array[0..lda*N-1] of Double;
  TDoubleVectorX = array[0..M-1] of Double;
  TDoubleVectorY = array[0..N-1] of Double;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TDoubleMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleVectorX;
  h_y_batch: array[0..batch_count-1] of TDoubleVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha: Double;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  alpha := 1.5;
  size_A := lda * N * SizeOf(Double);
  size_x := M * incx * SizeOf(Double);
  size_y := N * incy * SizeOf(Double);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do for i := 0 to M - 1 do h_A_initial[b][i + j * lda] := 1.0 + b + i + j;
    for i := 0 to M - 1 do h_x_batch[b][i] := 0.5 + b + i;
    for i := 0 to N - 1 do h_y_batch[b][i] := 0.5 + b + i;
    h_A_expected[b] := h_A_initial[b];
    for j := 0 to N - 1 do for i := 0 to M - 1 do
        h_A_expected[b][i + j * lda] := h_A_expected[b][i + j * lda] + alpha * h_x_batch[b][i] * h_y_batch[b][j];
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
    hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
  end;
  hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
  hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_dger_batched(handle, M, N, @alpha, PPDouble(d_x_ptrs), incx, PPDouble(d_y_ptrs), incy, PPDouble(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_dger_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
    for i := 0 to lda * N - 1 do
      AssertEquals('Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i), h_A_expected[b][i], h_A_result[b][i], 1e-12);
  end;

  for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_CGERU_batched;
const
  M = 5;
  N = 10;
  lda = M;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TComplexMatrix = array[0..lda*N-1] of rocblas_float_complex;
  TComplexVectorX = array[0..M-1] of rocblas_float_complex;
  TComplexVectorY = array[0..N-1] of rocblas_float_complex;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TComplexMatrix;
  h_x_batch: array[0..batch_count-1] of TComplexVectorX;
  h_y_batch: array[0..batch_count-1] of TComplexVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, product: rocblas_float_complex;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  alpha.x := 1.5; alpha.y := -0.5;
  size_A := lda * N * SizeOf(rocblas_float_complex);
  size_x := M * incx * SizeOf(rocblas_float_complex);
  size_y := N * incy * SizeOf(rocblas_float_complex);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do for i := 0 to M - 1 do begin h_A_initial[b][i+j*lda].x := 1+i+j+b; h_A_initial[b][i+j*lda].y := 0.5+i-j; end;
    for i := 0 to M - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;
    for i := 0 to N - 1 do begin h_y_batch[b][i].x := 0.5+i; h_y_batch[b][i].y := 0.2*b-i; end;

    h_A_expected[b] := h_A_initial[b];
    for j := 0 to N - 1 do for i := 0 to M - 1 do
    begin
      product := h_x_batch[b][i];
      ComplexMul(product, h_y_batch[b][j]);
      ComplexMul(product, alpha);
      ComplexAdd(h_A_expected[b][i+j*lda], product);
    end;
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
    hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
  end;
  hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
  hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgeru_batched(handle, M, N, @alpha, PPProcfloat_complex(d_x_ptrs), incx, PPProcfloat_complex(d_y_ptrs), incy, PPProcfloat_complex(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_cgeru_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
    for i := 0 to lda * N - 1 do
    begin
      CheckEqualsRelative(h_A_expected[b][i].x, h_A_result[b][i].x, 1e-5, 'Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + '.x');
      CheckEqualsRelative(h_A_expected[b][i].y, h_A_result[b][i].y, 1e-5, 'Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + '.y');
    end;
  end;

  for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_ZGERU_batched;
const
  M = 5;
  N = 10;
  lda = M;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TDoubleComplexMatrix = array[0..lda*N-1] of rocblas_double_complex;
  TDoubleComplexVectorX = array[0..M-1] of rocblas_double_complex;
  TDoubleComplexVectorY = array[0..N-1] of rocblas_double_complex;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TDoubleComplexMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleComplexVectorX;
  h_y_batch: array[0..batch_count-1] of TDoubleComplexVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, product: rocblas_double_complex;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  alpha.x := 1.5; alpha.y := -0.5;
  size_A := lda * N * SizeOf(rocblas_double_complex);
  size_x := M * incx * SizeOf(rocblas_double_complex);
  size_y := N * incy * SizeOf(rocblas_double_complex);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do for i := 0 to M - 1 do begin h_A_initial[b][i+j*lda].x := 1+i+j+b; h_A_initial[b][i+j*lda].y := 0.5+i-j; end;
    for i := 0 to M - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;
    for i := 0 to N - 1 do begin h_y_batch[b][i].x := 0.5+i; h_y_batch[b][i].y := 0.2*b-i; end;

    h_A_expected[b] := h_A_initial[b];
    for j := 0 to N - 1 do for i := 0 to M - 1 do
    begin
      product := h_x_batch[b][i];
      ComplexMul(product, h_y_batch[b][j]);
      ComplexMul(product, alpha);
      ComplexAdd(h_A_expected[b][i+j*lda], product);
    end;
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
    hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
  end;
  hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
  hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgeru_batched(handle, M, N, @alpha, PPProcdouble_complex(d_x_ptrs), incx, PPProcdouble_complex(d_y_ptrs), incy, PPProcdouble_complex(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_zgeru_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
    for i := 0 to lda * N - 1 do
    begin
      AssertEquals('Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + '.x', h_A_expected[b][i].x, h_A_result[b][i].x, 1e-12);
      AssertEquals('Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + '.y', h_A_expected[b][i].y, h_A_result[b][i].y, 1e-12);
    end;
  end;

  for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_CGERC_batched;
const
  M = 5;
  N = 10;
  lda = M;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TComplexMatrix = array[0..lda*N-1] of rocblas_float_complex;
  TComplexVectorX = array[0..M-1] of rocblas_float_complex;
  TComplexVectorY = array[0..N-1] of rocblas_float_complex;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TComplexMatrix;
  h_x_batch: array[0..batch_count-1] of TComplexVectorX;
  h_y_batch, h_y_conj: array[0..batch_count-1] of TComplexVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, product: rocblas_float_complex;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  alpha.x := 1.5; alpha.y := -0.5;
  size_A := lda * N * SizeOf(rocblas_float_complex);
  size_x := M * incx * SizeOf(rocblas_float_complex);
  size_y := N * incy * SizeOf(rocblas_float_complex);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do for i := 0 to M - 1 do begin h_A_initial[b][i+j*lda].x := 1+i+j+b; h_A_initial[b][i+j*lda].y := 0.5+i-j; end;
    for i := 0 to M - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;
    for i := 0 to N - 1 do begin h_y_batch[b][i].x := 0.5+i; h_y_batch[b][i].y := 0.2*b-i; end;

    h_A_expected[b] := h_A_initial[b];
    h_y_conj[b] := h_y_batch[b];
    for i := 0 to N - 1 do h_y_conj[b][i].y := -h_y_conj[b][i].y; // Create conjugate of y

    for j := 0 to N - 1 do for i := 0 to M - 1 do
    begin
      product := h_x_batch[b][i];
      ComplexMul(product, h_y_conj[b][j]);
      ComplexMul(product, alpha);
      ComplexAdd(h_A_expected[b][i+j*lda], product);
    end;
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
    hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
  end;
  hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
  hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_cgerc_batched(handle, M, N, @alpha, PPProcfloat_complex(d_x_ptrs), incx, PPProcfloat_complex(d_y_ptrs), incy, PPProcfloat_complex(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_cgerc_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
    for i := 0 to lda * N - 1 do
    begin
      CheckEqualsRelative(h_A_expected[b][i].x, h_A_result[b][i].x, 1e-5, 'Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + '.x');
      CheckEqualsRelative(h_A_expected[b][i].y, h_A_result[b][i].y, 1e-5, 'Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + '.y');
    end;
  end;

  for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_ZGERC_batched;
const
  M = 5;
  N = 10;
  lda = M;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TDoubleComplexMatrix = array[0..lda*N-1] of rocblas_double_complex;
  TDoubleComplexVectorX = array[0..M-1] of rocblas_double_complex;
  TDoubleComplexVectorY = array[0..N-1] of rocblas_double_complex;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TDoubleComplexMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleComplexVectorX;
  h_y_batch, h_y_conj: array[0..batch_count-1] of TDoubleComplexVectorY;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, product: rocblas_double_complex;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  alpha.x := 1.5; alpha.y := -0.5;
  size_A := lda * N * SizeOf(rocblas_double_complex);
  size_x := M * incx * SizeOf(rocblas_double_complex);
  size_y := N * incy * SizeOf(rocblas_double_complex);
  size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
  d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do for i := 0 to M - 1 do begin h_A_initial[b][i+j*lda].x := 1+i+j+b; h_A_initial[b][i+j*lda].y := 0.5+i-j; end;
    for i := 0 to M - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;
    for i := 0 to N - 1 do begin h_y_batch[b][i].x := 0.5+i; h_y_batch[b][i].y := 0.2*b-i; end;

    h_A_expected[b] := h_A_initial[b];
    h_y_conj[b] := h_y_batch[b];
    for i := 0 to N - 1 do h_y_conj[b][i].y := -h_y_conj[b][i].y;

    for j := 0 to N - 1 do for i := 0 to M - 1 do
    begin
      product := h_x_batch[b][i];
      ComplexMul(product, h_y_conj[b][j]);
      ComplexMul(product, alpha);
      ComplexAdd(h_A_expected[b][i+j*lda], product);
    end;
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
    hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
  end;
  hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
  hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
  hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

  blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

  blas_status := rocblas_zgerc_batched(handle, M, N, @alpha, PPProcdouble_complex(d_x_ptrs), incx, PPProcdouble_complex(d_y_ptrs), incy, PPProcdouble_complex(d_A_ptrs), lda, batch_count);
  AssertEquals('rocblas_zgerc_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
    for i := 0 to lda * N - 1 do
    begin
      AssertEquals('Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + '.x', h_A_expected[b][i].x, h_A_result[b][i].x, 1e-12);
      AssertEquals('Batch ' + IntToStr(b) + ', Elem ' + IntToStr(i) + '.y', h_A_expected[b][i].y, h_A_result[b][i].y, 1e-12);
    end;
  end;

  for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
  hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
end;

procedure TTestRocBLAS_Level2_batched.Test_SSYR_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  batch_count = 2;
type
  TSingleMatrix = array[0..lda*N-1] of Single;
  TSingleVector = array[0..N-1] of Single;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TSingleMatrix;
  h_x_batch: array[0..batch_count-1] of TSingleVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  alpha: Single;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha := 1.5;
    size_A := lda * N * SizeOf(Single);
    size_x := N * incx * SizeOf(Single);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          h_A_initial[b][i + j * lda] := 1.0 + b + i + j;
          h_A_initial[b][j + i * lda] := h_A_initial[b][i + j * lda]; // Ensure symmetry
        end;

      for i := 0 to N - 1 do h_x_batch[b][i] := 0.5 + b + i;

      h_A_expected[b] := h_A_initial[b];
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          h_A_expected[b][i + j * lda] := h_A_expected[b][i + j * lda] + alpha * h_x_batch[b][i] * h_x_batch[b][j];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_ssyr_batched(handle, uplo, N, @alpha, PPSingle(d_x_ptrs), incx, PPSingle(d_A_ptrs), lda, batch_count);
    AssertEquals('rocblas_ssyr_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if ((uplo = rocblas_fill_lower) and (i >= j)) or ((uplo = rocblas_fill_upper) and (i <= j)) then
            CheckEqualsRelative(h_A_expected[b][i+j*lda], h_A_result[b][i+j*lda], 1e-5, uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ')');
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_DSYR_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  batch_count = 2;
type
  TDoubleMatrix = array[0..lda*N-1] of Double;
  TDoubleVector = array[0..N-1] of Double;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TDoubleMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  alpha: Double;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha := 1.5;
    size_A := lda * N * SizeOf(Double);
    size_x := N * incx * SizeOf(Double);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      for j := 0 to N - 1 do for i := j to N - 1 do
      begin h_A_initial[b][i + j * lda] := 1.0 + b + i + j; h_A_initial[b][j + i * lda] := h_A_initial[b][i + j * lda]; end;
      for i := 0 to N - 1 do h_x_batch[b][i] := 0.5 + b + i;
      h_A_expected[b] := h_A_initial[b];
      for j := 0 to N - 1 do for i := 0 to N - 1 do
        h_A_expected[b][i + j * lda] := h_A_expected[b][i + j * lda] + alpha * h_x_batch[b][i] * h_x_batch[b][j];
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_dsyr_batched(handle, uplo, N, @alpha, PPDouble(d_x_ptrs), incx, PPDouble(d_A_ptrs), lda, batch_count);
    AssertEquals('rocblas_dsyr_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
      for j := 0 to N - 1 do for i := 0 to N - 1 do
        if ((uplo = rocblas_fill_lower) and (i >= j)) or ((uplo = rocblas_fill_upper) and (i <= j)) then
          AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ')', h_A_expected[b][i+j*lda], h_A_result[b][i+j*lda], 1e-12);
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_CSYR_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  batch_count = 2;
type
  TComplexMatrix = array[0..lda*N-1] of rocblas_float_complex;
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TComplexMatrix;
  h_x_batch: array[0..batch_count-1] of TComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  alpha, product: rocblas_float_complex;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha.x := 1.5; alpha.y := -0.5;
    size_A := lda * N * SizeOf(rocblas_float_complex);
    size_x := N * incx * SizeOf(rocblas_float_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      for j := 0 to N - 1 do for i := j to N - 1 do
      begin
        h_A_initial[b][i+j*lda].x := 1+i+j+b; h_A_initial[b][i+j*lda].y := 0.5+i-j;
        h_A_initial[b][j+i*lda] := h_A_initial[b][i+j*lda];
      end;
      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;

      h_A_expected[b] := h_A_initial[b];
      for j := 0 to N - 1 do for i := 0 to N - 1 do
      begin
        product := h_x_batch[b][i];
        ComplexMul(product, h_x_batch[b][j]);
        ComplexMul(product, alpha);
        ComplexAdd(h_A_expected[b][i+j*lda], product);
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_csyr_batched(handle, uplo, N, @alpha, PPProcfloat_complex(d_x_ptrs), incx, PPProcfloat_complex(d_A_ptrs), lda, batch_count);
    AssertEquals('rocblas_csyr_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
      for j := 0 to N - 1 do for i := 0 to N - 1 do
        if ((uplo = rocblas_fill_lower) and (i >= j)) or ((uplo = rocblas_fill_upper) and (i <= j)) then
        begin
          CheckEqualsRelative(h_A_expected[b][i+j*lda].x, h_A_result[b][i+j*lda].x, 1e-5, uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').x');
          CheckEqualsRelative(h_A_expected[b][i+j*lda].y, h_A_result[b][i+j*lda].y, 1e-5, uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').y');
        end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_ZSYR_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  batch_count = 2;
type
  TDoubleComplexMatrix = array[0..lda*N-1] of rocblas_double_complex;
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TDoubleComplexMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  alpha, product: rocblas_double_complex;
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha.x := 1.5; alpha.y := -0.5;
    size_A := lda * N * SizeOf(rocblas_double_complex);
    size_x := N * incx * SizeOf(rocblas_double_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      for j := 0 to N - 1 do for i := j to N - 1 do
      begin
        h_A_initial[b][i+j*lda].x := 1+i+j+b; h_A_initial[b][i+j*lda].y := 0.5+i-j;
        h_A_initial[b][j+i*lda] := h_A_initial[b][i+j*lda];
      end;
      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;

      h_A_expected[b] := h_A_initial[b];
      for j := 0 to N - 1 do for i := 0 to N - 1 do
      begin
        product := h_x_batch[b][i];
        ComplexMul(product, h_x_batch[b][j]);
        ComplexMul(product, alpha);
        ComplexAdd(h_A_expected[b][i+j*lda], product);
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_zsyr_batched(handle, uplo, N, @alpha, PPProcdouble_complex(d_x_ptrs), incx, PPProcdouble_complex(d_A_ptrs), lda, batch_count);
    AssertEquals('rocblas_zsyr_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
      for j := 0 to N - 1 do for i := 0 to N - 1 do
        if ((uplo = rocblas_fill_lower) and (i >= j)) or ((uplo = rocblas_fill_upper) and (i <= j)) then
        begin
          AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').x', h_A_expected[b][i+j*lda].x, h_A_result[b][i+j*lda].x, 1e-12);
          AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').y', h_A_expected[b][i+j*lda].y, h_A_result[b][i+j*lda].y, 1e-12);
        end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_CHER_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  batch_count = 2;
type
  TComplexMatrix = array[0..lda*N-1] of rocblas_float_complex;
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TComplexMatrix;
  h_x_batch: array[0..batch_count-1] of TComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  alpha: Single; // NOTE: alpha is REAL for HER
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha := 1.5;
    size_A := lda * N * SizeOf(rocblas_float_complex);
    size_x := N * incx * SizeOf(rocblas_float_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      // Initialize a full Hermitian matrix for reference
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          if i = j then
          begin
            h_A_initial[b][i + j * lda].x := 2.0 + b + i; // Real diagonal
            h_A_initial[b][i + j * lda].y := 0.0;
          end
          else
          begin
            h_A_initial[b][i + j * lda].x := 1.0 + i;
            h_A_initial[b][i + j * lda].y := 0.5 + j;
            h_A_initial[b][j + i * lda].x := h_A_initial[b][i + j * lda].x; // Conjugate transpose
            h_A_initial[b][j + i * lda].y := -h_A_initial[b][i + j * lda].y;
          end;
        end;

      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;

      // CPU Calculation: A_expected := alpha*x*x_conj**T + A
      h_A_expected[b] := h_A_initial[b];
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
        begin
          // A(i,j) += alpha * x(i) * conj(x(j))
          h_A_expected[b][i+j*lda].x := h_A_expected[b][i+j*lda].x + alpha * (h_x_batch[b][i].x * h_x_batch[b][j].x + h_x_batch[b][i].y * h_x_batch[b][j].y);
          h_A_expected[b][i+j*lda].y := h_A_expected[b][i+j*lda].y + alpha * (h_x_batch[b][i].y * h_x_batch[b][j].x - h_x_batch[b][i].x * h_x_batch[b][j].y);
        end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_cher_batched(handle, uplo, N, @alpha, PPProcfloat_complex(d_x_ptrs), incx, PPProcfloat_complex(d_A_ptrs), lda, batch_count);
    AssertEquals('rocblas_cher_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if ((uplo = rocblas_fill_lower) and (i >= j)) or ((uplo = rocblas_fill_upper) and (i <= j)) then
          begin
            CheckEqualsRelative(h_A_expected[b][i+j*lda].x, h_A_result[b][i+j*lda].x, 1e-4, uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').x');
            CheckEqualsRelative(h_A_expected[b][i+j*lda].y, h_A_result[b][i+j*lda].y, 1e-4, uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').y');
          end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_ZHER_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  batch_count = 2;
type
  TDoubleComplexMatrix = array[0..lda*N-1] of rocblas_double_complex;
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TDoubleComplexMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs: hipDeviceptr_t;
  alpha: Double; // NOTE: alpha is REAL for HER
  i, j, b: Integer;
  size_A, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha := 1.5;
    size_A := lda * N * SizeOf(rocblas_double_complex);
    size_x := N * incx * SizeOf(rocblas_double_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          if i = j then
          begin
            h_A_initial[b][i + j * lda].x := 2.0 + b + i;
            h_A_initial[b][i + j * lda].y := 0.0;
          end
          else
          begin
            h_A_initial[b][i + j * lda].x := 1.0 + i;
            h_A_initial[b][i + j * lda].y := 0.5 + j;
            h_A_initial[b][j + i * lda].x := h_A_initial[b][i + j * lda].x;
            h_A_initial[b][j + i * lda].y := -h_A_initial[b][i + j * lda].y;
          end;
        end;

      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;

      h_A_expected[b] := h_A_initial[b];
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
        begin
          h_A_expected[b][i+j*lda].x := h_A_expected[b][i+j*lda].x + alpha * (h_x_batch[b][i].x * h_x_batch[b][j].x + h_x_batch[b][i].y * h_x_batch[b][j].y);
          h_A_expected[b][i+j*lda].y := h_A_expected[b][i+j*lda].y + alpha * (h_x_batch[b][i].y * h_x_batch[b][j].x - h_x_batch[b][i].x * h_x_batch[b][j].y);
        end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_zher_batched(handle, uplo, N, @alpha, PPProcdouble_complex(d_x_ptrs), incx, PPProcdouble_complex(d_A_ptrs), lda, batch_count);
    AssertEquals('rocblas_zher_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if ((uplo = rocblas_fill_lower) and (i >= j)) or ((uplo = rocblas_fill_upper) and (i <= j)) then
          begin
            AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').x', h_A_expected[b][i+j*lda].x, h_A_result[b][i+j*lda].x, 1e-12);
            AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').y', h_A_expected[b][i+j*lda].y, h_A_result[b][i+j*lda].y, 1e-12);
          end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_SSPR_batched;
const
  N = 10;
  incx = 1;
  batch_count = 2;
type
  TSinglePackedMatrix = array[0..N*(N+1) div 2 - 1] of Single;
  TSingleVector = array[0..N-1] of Single;
  TSingleFullMatrix = array[0..N*N-1] of Single;
var
  h_AP_initial, h_AP_expected, h_AP_result: array[0..batch_count-1] of TSinglePackedMatrix;
  h_A_full: array[0..batch_count-1] of TSingleFullMatrix;
  h_x_batch: array[0..batch_count-1] of TSingleVector;
  h_dev_AP_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs: hipDeviceptr_t;
  alpha: Single;
  i, j, b, k: Integer;
  size_AP, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha := 1.5;
    size_AP := (N * (N + 1) div 2) * SizeOf(Single);
    size_x := N * incx * SizeOf(Single);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      // Initialize full symmetric matrix for reference and create initial packed version
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := j to N - 1 do
        begin
          h_A_full[b][i + j * N] := 1.0 + b + i + j;
          h_A_full[b][j + i * N] := h_A_full[b][i + j * N];
        end;

        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;

      for i := 0 to N - 1 do h_x_batch[b][i] := 0.5 + b + i;

      // Calculate expected full result matrix
      for j := 0 to N - 1 do for i := 0 to N - 1 do
          h_A_full[b][i + j * N] := h_A_full[b][i + j * N] + alpha * h_x_batch[b][i] * h_x_batch[b][j];

      // Convert expected full matrix to packed format for verification
      k := 0;
      for j := 0 to N - 1 do
      begin
        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_initial[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_sspr_batched(handle, uplo, N, @alpha, PPSingle(d_x_ptrs), incx, PPSingle(d_AP_ptrs), batch_count);
    AssertEquals('rocblas_sspr_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_AP_result[b][0], h_dev_AP_ptrs[b], size_AP, hipMemcpyDeviceToHost);
      for i := 0 to size_AP div SizeOf(Single) - 1 do
        CheckEqualsRelative(h_AP_expected[b][i], h_AP_result[b][i], 1e-5, uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i));
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_DSPR_batched;
const
  N = 10;
  incx = 1;
  batch_count = 2;
type
  TDoublePackedMatrix = array[0..N*(N+1) div 2 - 1] of Double;
  TDoubleVector = array[0..N-1] of Double;
  TDoubleFullMatrix = array[0..N*N-1] of Double;
var
  h_AP_initial, h_AP_expected, h_AP_result: array[0..batch_count-1] of TDoublePackedMatrix;
  h_A_full: array[0..batch_count-1] of TDoubleFullMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleVector;
  h_dev_AP_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs: hipDeviceptr_t;
  alpha: Double;
  i, j, b, k: Integer;
  size_AP, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha := 1.5;
    size_AP := (N * (N + 1) div 2) * SizeOf(Double);
    size_x := N * incx * SizeOf(Double);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := j to N - 1 do begin h_A_full[b][i + j * N] := 1.0 + b + i + j; h_A_full[b][j + i * N] := h_A_full[b][i + j * N]; end;
        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
      for i := 0 to N - 1 do h_x_batch[b][i] := 0.5 + b + i;
      for j := 0 to N - 1 do for i := 0 to N - 1 do
        h_A_full[b][i + j * N] := h_A_full[b][i + j * N] + alpha * h_x_batch[b][i] * h_x_batch[b][j];
      k := 0;
      for j := 0 to N - 1 do
      begin
        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_initial[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_dspr_batched(handle, uplo, N, @alpha, PPDouble(d_x_ptrs), incx, PPDouble(d_AP_ptrs), batch_count);
    AssertEquals('rocblas_dspr_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_AP_result[b][0], h_dev_AP_ptrs[b], size_AP, hipMemcpyDeviceToHost);
      for i := 0 to size_AP div SizeOf(Double) - 1 do
        AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i), h_AP_expected[b][i], h_AP_result[b][i], 1e-12);
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_CHPR_batched;
const
  N = 10;
  incx = 1;
  batch_count = 2;
type
  TComplexPackedMatrix = array[0..N*(N+1) div 2 - 1] of rocblas_float_complex;
  TComplexVector = array[0..N-1] of rocblas_float_complex;
  TComplexFullMatrix = array[0..N*N-1] of rocblas_float_complex;
var
  h_AP_initial, h_AP_expected, h_AP_result: array[0..batch_count-1] of TComplexPackedMatrix;
  h_A_full: array[0..batch_count-1] of TComplexFullMatrix;
  h_x_batch: array[0..batch_count-1] of TComplexVector;
  h_dev_AP_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs: hipDeviceptr_t;
  alpha: Single; // NOTE: alpha is REAL for HPR
  i, j, b, k: Integer;
  size_AP, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha := 1.5;
    size_AP := (N * (N + 1) div 2) * SizeOf(rocblas_float_complex);
    size_x := N * incx * SizeOf(rocblas_float_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := j to N - 1 do
        begin
          if i = j then begin h_A_full[b][i + j*N].x := 2+b+i; h_A_full[b][i + j*N].y := 0.0; end
          else
          begin
            h_A_full[b][i + j*N].x := 1+i; h_A_full[b][i + j*N].y := 0.5+j;
            h_A_full[b][j + i*N].x := h_A_full[b][i + j*N].x;
            h_A_full[b][j + i*N].y := -h_A_full[b][i + j*N].y;
          end;
        end;

        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;

      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;

      for j := 0 to N - 1 do for i := 0 to N - 1 do
      begin
        h_A_full[b][i+j*N].x := h_A_full[b][i+j*N].x + alpha * (h_x_batch[b][i].x * h_x_batch[b][j].x + h_x_batch[b][i].y * h_x_batch[b][j].y);
        h_A_full[b][i+j*N].y := h_A_full[b][i+j*N].y + alpha * (h_x_batch[b][i].y * h_x_batch[b][j].x - h_x_batch[b][i].x * h_x_batch[b][j].y);
      end;
      k := 0;
      for j := 0 to N - 1 do
      begin
        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_initial[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_chpr_batched(handle, uplo, N, @alpha, PPProcfloat_complex(d_x_ptrs), incx, PPProcfloat_complex(d_AP_ptrs), batch_count);
    AssertEquals('rocblas_chpr_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_AP_result[b][0], h_dev_AP_ptrs[b], size_AP, hipMemcpyDeviceToHost);
      for i := 0 to size_AP div SizeOf(rocblas_float_complex) - 1 do
      begin
        CheckEqualsRelative(h_AP_expected[b][i].x, h_AP_result[b][i].x, 1e-4, uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x');
        CheckEqualsRelative(h_AP_expected[b][i].y, h_AP_result[b][i].y, 1e-4, uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y');
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_ZHPR_batched;
const
  N = 10;
  incx = 1;
  batch_count = 2;
type
  TDoubleComplexPackedMatrix = array[0..N*(N+1) div 2 - 1] of rocblas_double_complex;
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
  TDoubleComplexFullMatrix = array[0..N*N-1] of rocblas_double_complex;
var
  h_AP_initial, h_AP_expected, h_AP_result: array[0..batch_count-1] of TDoubleComplexPackedMatrix;
  h_A_full: array[0..batch_count-1] of TDoubleComplexFullMatrix;
  h_x_batch: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_AP_ptrs, h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs: hipDeviceptr_t;
  alpha: Double; // NOTE: alpha is REAL for HPR
  i, j, b, k: Integer;
  size_AP, size_x, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha := 1.5;
    size_AP := (N * (N + 1) div 2) * SizeOf(rocblas_double_complex);
    size_x := N * incx * SizeOf(rocblas_double_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := j to N - 1 do
        begin
          if i = j then begin h_A_full[b][i + j*N].x := 2+b+i; h_A_full[b][i + j*N].y := 0.0; end
          else
          begin
            h_A_full[b][i + j*N].x := 1+i; h_A_full[b][i + j*N].y := 0.5+j;
            h_A_full[b][j + i*N].x := h_A_full[b][i + j*N].x;
            h_A_full[b][j + i*N].y := -h_A_full[b][i + j*N].y;
          end;
        end;

        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;

      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;

      for j := 0 to N - 1 do for i := 0 to N - 1 do
      begin
        h_A_full[b][i+j*N].x := h_A_full[b][i+j*N].x + alpha * (h_x_batch[b][i].x * h_x_batch[b][j].x + h_x_batch[b][i].y * h_x_batch[b][j].y);
        h_A_full[b][i+j*N].y := h_A_full[b][i+j*N].y + alpha * (h_x_batch[b][i].y * h_x_batch[b][j].x - h_x_batch[b][i].x * h_x_batch[b][j].y);
      end;
      k := 0;
      for j := 0 to N - 1 do
      begin
        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_initial[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_zhpr_batched(handle, uplo, N, @alpha, PPProcdouble_complex(d_x_ptrs), incx, PPProcdouble_complex(d_AP_ptrs), batch_count);
    AssertEquals('rocblas_zhpr_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_AP_result[b][0], h_dev_AP_ptrs[b], size_AP, hipMemcpyDeviceToHost);
      for i := 0 to size_AP div SizeOf(rocblas_double_complex) - 1 do
      begin
        AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x', h_AP_expected[b][i].x, h_AP_result[b][i].x, 1e-12);
        AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y', h_AP_expected[b][i].y, h_AP_result[b][i].y, 1e-12);
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_SSYR2_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TSingleMatrix = array[0..lda*N-1] of Single;
  TSingleVector = array[0..N-1] of Single;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TSingleMatrix;
  h_x_batch, h_y_batch: array[0..batch_count-1] of TSingleVector;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha: Single;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha := 1.5;
    size_A := lda * N * SizeOf(Single);
    size_x := N * incx * SizeOf(Single);
    size_y := N * incy * SizeOf(Single);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          h_A_initial[b][i + j * lda] := 1.0 + b + i + j;
          h_A_initial[b][j + i * lda] := h_A_initial[b][i + j * lda];
        end;

      for i := 0 to N - 1 do
      begin
        h_x_batch[b][i] := 0.5 + b + i;
        h_y_batch[b][i] := 0.5 - b - i;
      end;

      h_A_expected[b] := h_A_initial[b];
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          h_A_expected[b][i + j * lda] := h_A_expected[b][i + j * lda] +
            alpha * (h_x_batch[b][i] * h_y_batch[b][j] + h_y_batch[b][i] * h_x_batch[b][j]);
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_ssyr2_batched(handle, uplo, N, @alpha, PPSingle(d_x_ptrs), incx, PPSingle(d_y_ptrs), incy, PPSingle(d_A_ptrs), lda, batch_count);
    AssertEquals('rocblas_ssyr2_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
      for j := 0 to N - 1 do
        for i := 0 to N - 1 do
          if ((uplo = rocblas_fill_lower) and (i >= j)) or ((uplo = rocblas_fill_upper) and (i <= j)) then
            CheckEqualsRelative(h_A_expected[b][i+j*lda], h_A_result[b][i+j*lda], 1e-5, uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ')');
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_DSYR2_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TDoubleMatrix = array[0..lda*N-1] of Double;
  TDoubleVector = array[0..N-1] of Double;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TDoubleMatrix;
  h_x_batch, h_y_batch: array[0..batch_count-1] of TDoubleVector;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha: Double;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha := 1.5;
    size_A := lda * N * SizeOf(Double);
    size_x := N * incx * SizeOf(Double);
    size_y := N * incy * SizeOf(Double);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      for j := 0 to N - 1 do for i := j to N - 1 do
      begin h_A_initial[b][i + j * lda] := 1.0 + b + i + j; h_A_initial[b][j + i * lda] := h_A_initial[b][i + j * lda]; end;
      for i := 0 to N - 1 do begin h_x_batch[b][i] := 0.5 + b + i; h_y_batch[b][i] := 0.5 - b - i; end;
      h_A_expected[b] := h_A_initial[b];
      for j := 0 to N - 1 do for i := 0 to N - 1 do
        h_A_expected[b][i + j * lda] := h_A_expected[b][i + j * lda] +
          alpha * (h_x_batch[b][i] * h_y_batch[b][j] + h_y_batch[b][i] * h_x_batch[b][j]);
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_dsyr2_batched(handle, uplo, N, @alpha, PPDouble(d_x_ptrs), incx, PPDouble(d_y_ptrs), incy, PPDouble(d_A_ptrs), lda, batch_count);
    AssertEquals('rocblas_dsyr2_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
      for j := 0 to N - 1 do for i := 0 to N - 1 do
        if ((uplo = rocblas_fill_lower) and (i >= j)) or ((uplo = rocblas_fill_upper) and (i <= j)) then
          AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ')', h_A_expected[b][i+j*lda], h_A_result[b][i+j*lda], 1e-12);
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_CSYR2_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TComplexMatrix = array[0..lda*N-1] of rocblas_float_complex;
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TComplexMatrix;
  h_x_batch, h_y_batch: array[0..batch_count-1] of TComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, term1, term2, update: rocblas_float_complex;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha.x := 1.5; alpha.y := -0.5;
    size_A := lda * N * SizeOf(rocblas_float_complex);
    size_x := N * incx * SizeOf(rocblas_float_complex);
    size_y := N * incy * SizeOf(rocblas_float_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      for j := 0 to N - 1 do for i := j to N - 1 do
      begin
        h_A_initial[b][i+j*lda].x := 1+i+j+b; h_A_initial[b][i+j*lda].y := 0.5+i-j;
        h_A_initial[b][j+i*lda] := h_A_initial[b][i+j*lda];
      end;
      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;
      for i := 0 to N - 1 do begin h_y_batch[b][i].x := 0.5-i; h_y_batch[b][i].y := 0.2*b+i; end;

      h_A_expected[b] := h_A_initial[b];
      for j := 0 to N - 1 do for i := 0 to N - 1 do
      begin
        term1 := h_x_batch[b][i]; ComplexMul(term1, h_y_batch[b][j]);
        term2 := h_y_batch[b][i]; ComplexMul(term2, h_x_batch[b][j]);
        update.x := term1.x + term2.x; update.y := term1.y + term2.y;
        ComplexMul(update, alpha);
        ComplexAdd(h_A_expected[b][i+j*lda], update);
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_csyr2_batched(handle, uplo, N, @alpha, PPProcfloat_complex(d_x_ptrs), incx, PPProcfloat_complex(d_y_ptrs), incy, PPProcfloat_complex(d_A_ptrs), lda, batch_count);
    AssertEquals('rocblas_csyr2_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
      for j := 0 to N - 1 do for i := 0 to N - 1 do
        if ((uplo = rocblas_fill_lower) and (i >= j)) or ((uplo = rocblas_fill_upper) and (i <= j)) then
        begin
          CheckEqualsRelative(h_A_expected[b][i+j*lda].x, h_A_result[b][i+j*lda].x, 1e-5, uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').x');
          CheckEqualsRelative(h_A_expected[b][i+j*lda].y, h_A_result[b][i+j*lda].y, 1e-5, uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').y');
        end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_ZSYR2_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TDoubleComplexMatrix = array[0..lda*N-1] of rocblas_double_complex;
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TDoubleComplexMatrix;
  h_x_batch, h_y_batch: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, term1, term2, update: rocblas_double_complex;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha.x := 1.5; alpha.y := -0.5;
    size_A := lda * N * SizeOf(rocblas_double_complex);
    size_x := N * incx * SizeOf(rocblas_double_complex);
    size_y := N * incy * SizeOf(rocblas_double_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      for j := 0 to N - 1 do for i := j to N - 1 do
      begin
        h_A_initial[b][i+j*lda].x := 1+i+j+b; h_A_initial[b][i+j*lda].y := 0.5+i-j;
        h_A_initial[b][j+i*lda] := h_A_initial[b][i+j*lda];
      end;
      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;
      for i := 0 to N - 1 do begin h_y_batch[b][i].x := 0.5-i; h_y_batch[b][i].y := 0.2*b+i; end;

      h_A_expected[b] := h_A_initial[b];
      for j := 0 to N - 1 do for i := 0 to N - 1 do
      begin
        term1 := h_x_batch[b][i]; ComplexMul(term1, h_y_batch[b][j]);
        term2 := h_y_batch[b][i]; ComplexMul(term2, h_x_batch[b][j]);
        update.x := term1.x + term2.x; update.y := term1.y + term2.y;
        ComplexMul(update, alpha);
        ComplexAdd(h_A_expected[b][i+j*lda], update);
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_zsyr2_batched(handle, uplo, N, @alpha, PPProcdouble_complex(d_x_ptrs), incx, PPProcdouble_complex(d_y_ptrs), incy, PPProcdouble_complex(d_A_ptrs), lda, batch_count);
    AssertEquals('rocblas_zsyr2_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
      for j := 0 to N - 1 do for i := 0 to N - 1 do
        if ((uplo = rocblas_fill_lower) and (i >= j)) or ((uplo = rocblas_fill_upper) and (i <= j)) then
        begin
          AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').x', h_A_expected[b][i+j*lda].x, h_A_result[b][i+j*lda].x, 1e-12);
          AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').y', h_A_expected[b][i+j*lda].y, h_A_result[b][i+j*lda].y, 1e-12);
        end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_CHER2_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TComplexMatrix = array[0..lda*N-1] of rocblas_float_complex;
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TComplexMatrix;
  h_x_batch, h_y_batch: array[0..batch_count-1] of TComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, alpha_conj, term1, term2, update, temp: rocblas_float_complex;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha.x := 1.5; alpha.y := -0.5;
    alpha_conj.x := alpha.x; alpha_conj.y := -alpha.y;
    size_A := lda * N * SizeOf(rocblas_float_complex);
    size_x := N * incx * SizeOf(rocblas_float_complex);
    size_y := N * incy * SizeOf(rocblas_float_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      for j := 0 to N - 1 do for i := j to N - 1 do
      begin
        if i = j then begin h_A_initial[b][i+j*lda].x := 2+b+i; h_A_initial[b][i+j*lda].y := 0.0; end
        else
        begin
          h_A_initial[b][i+j*lda].x := 1+i; h_A_initial[b][i+j*lda].y := 0.5+j;
          h_A_initial[b][j+i*lda].x := h_A_initial[b][i+j*lda].x;
          h_A_initial[b][j+i*lda].y := -h_A_initial[b][i+j*lda].y;
        end;
      end;
      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;
      for i := 0 to N - 1 do begin h_y_batch[b][i].x := 0.5-i; h_y_batch[b][i].y := 0.2*b+i; end;

      h_A_expected[b] := h_A_initial[b];
      for j := 0 to N - 1 do for i := 0 to N - 1 do
      begin
        term1 := h_x_batch[b][i]; temp := h_y_batch[b][j]; temp.y := -temp.y; // x * conj(y)
        ComplexMul(term1, temp);
        ComplexMul(term1, alpha);

        term2 := h_y_batch[b][i]; temp := h_x_batch[b][j]; temp.y := -temp.y; // y * conj(x)
        ComplexMul(term2, temp);
        ComplexMul(term2, alpha_conj);

        update.x := term1.x + term2.x; update.y := term1.y + term2.y;
        ComplexAdd(h_A_expected[b][i+j*lda], update);
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_cher2_batched(handle, uplo, N, @alpha, PPProcfloat_complex(d_x_ptrs), incx, PPProcfloat_complex(d_y_ptrs), incy, PPProcfloat_complex(d_A_ptrs), lda, batch_count);
    AssertEquals('rocblas_cher2_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
      for j := 0 to N - 1 do for i := 0 to N - 1 do
        if ((uplo = rocblas_fill_lower) and (i >= j)) or ((uplo = rocblas_fill_upper) and (i <= j)) then
        begin
          // CORRECTED: Use hybrid absolute/relative tolerance check
          if Abs(h_A_expected[b][i+j*lda].x) < 1e-4 then
             AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').x', h_A_expected[b][i+j*lda].x, h_A_result[b][i+j*lda].x, 1e-4)
          else
             CheckEqualsRelative(h_A_expected[b][i+j*lda].x, h_A_result[b][i+j*lda].x, 1e-4, uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').x');

          if Abs(h_A_expected[b][i+j*lda].y) < 1e-4 then
             AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').y', h_A_expected[b][i+j*lda].y, h_A_result[b][i+j*lda].y, 1e-4)
          else
             CheckEqualsRelative(h_A_expected[b][i+j*lda].y, h_A_result[b][i+j*lda].y, 1e-4, uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').y');
        end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_ZHER2_batched;
const
  N = 10;
  lda = N;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TDoubleComplexMatrix = array[0..lda*N-1] of rocblas_double_complex;
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_A_initial, h_A_expected, h_A_result: array[0..batch_count-1] of TDoubleComplexMatrix;
  h_x_batch, h_y_batch: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_A_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_A_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, alpha_conj, term1, term2, update, temp: rocblas_double_complex;
  i, j, b: Integer;
  size_A, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha.x := 1.5; alpha.y := -0.5;
    alpha_conj.x := alpha.x; alpha_conj.y := -alpha.y;
    size_A := lda * N * SizeOf(rocblas_double_complex);
    size_x := N * incx * SizeOf(rocblas_double_complex);
    size_y := N * incy * SizeOf(rocblas_double_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_A_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      for j := 0 to N - 1 do for i := j to N - 1 do
      begin
        if i = j then begin h_A_initial[b][i+j*lda].x := 2+b+i; h_A_initial[b][i+j*lda].y := 0.0; end
        else
        begin
          h_A_initial[b][i+j*lda].x := 1+i; h_A_initial[b][i+j*lda].y := 0.5+j;
          h_A_initial[b][j+i*lda].x := h_A_initial[b][i+j*lda].x;
          h_A_initial[b][j+i*lda].y := -h_A_initial[b][i+j*lda].y;
        end;
      end;
      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;
      for i := 0 to N - 1 do begin h_y_batch[b][i].x := 0.5-i; h_y_batch[b][i].y := 0.2*b+i; end;

      h_A_expected[b] := h_A_initial[b];
      for j := 0 to N - 1 do for i := 0 to N - 1 do
      begin
        term1 := h_x_batch[b][i]; temp := h_y_batch[b][j]; temp.y := -temp.y;
        ComplexMul(term1, temp);
        ComplexMul(term1, alpha);
        term2 := h_y_batch[b][i]; temp := h_x_batch[b][j]; temp.y := -temp.y;
        ComplexMul(term2, temp);
        ComplexMul(term2, alpha_conj);
        update.x := term1.x + term2.x; update.y := term1.y + term2.y;
        ComplexAdd(h_A_expected[b][i+j*lda], update);
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_A_ptrs[b], size_A); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_A_ptrs[b], @h_A_initial[b][0], size_A, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_A_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_A_ptrs, @h_dev_A_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_zher2_batched(handle, uplo, N, @alpha, PPProcdouble_complex(d_x_ptrs), incx, PPProcdouble_complex(d_y_ptrs), incy, PPProcdouble_complex(d_A_ptrs), lda, batch_count);
    AssertEquals('rocblas_zher2_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_A_result[b][0], h_dev_A_ptrs[b], size_A, hipMemcpyDeviceToHost);
      for j := 0 to N - 1 do for i := 0 to N - 1 do
        if ((uplo = rocblas_fill_lower) and (i >= j)) or ((uplo = rocblas_fill_upper) and (i <= j)) then
        begin
          // For double precision, a direct absolute tolerance check is robust and simpler.
          AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').x', h_A_expected[b][i+j*lda].x, h_A_result[b][i+j*lda].x, 1e-12);
          AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem(' + IntToStr(i) + ',' + IntToStr(j) + ').y', h_A_expected[b][i+j*lda].y, h_A_result[b][i+j*lda].y, 1e-12);
        end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_A_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
    hipFree(d_A_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_SSPR2_batched;
const
  N = 10;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TSinglePackedMatrix = array[0..N*(N+1) div 2 - 1] of Single;
  TSingleVector = array[0..N-1] of Single;
  TSingleFullMatrix = array[0..N*N-1] of Single;
var
  h_AP_initial, h_AP_expected, h_AP_result: array[0..batch_count-1] of TSinglePackedMatrix;
  h_A_full: array[0..batch_count-1] of TSingleFullMatrix;
  h_x_batch, h_y_batch: array[0..batch_count-1] of TSingleVector;
  h_dev_AP_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha: Single;
  i, j, b, k: Integer;
  size_AP, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha := 1.5;
    size_AP := (N * (N + 1) div 2) * SizeOf(Single);
    size_x := N * incx * SizeOf(Single);
    size_y := N * incy * SizeOf(Single);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := j to N - 1 do begin h_A_full[b][i + j * N] := 1.0 + b + i + j; h_A_full[b][j + i * N] := h_A_full[b][i + j * N]; end;
        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;

      for i := 0 to N - 1 do begin h_x_batch[b][i] := 0.5 + b + i; h_y_batch[b][i] := 0.5 - b - i; end;

      for j := 0 to N - 1 do for i := 0 to N - 1 do
          h_A_full[b][i + j * N] := h_A_full[b][i + j * N] + alpha * (h_x_batch[b][i] * h_y_batch[b][j] + h_y_batch[b][i] * h_x_batch[b][j]);
      k := 0;
      for j := 0 to N - 1 do
      begin
        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_initial[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_sspr2_batched(handle, uplo, N, @alpha, PPSingle(d_x_ptrs), incx, PPSingle(d_y_ptrs), incy, PPSingle(d_AP_ptrs), batch_count);
    AssertEquals('rocblas_sspr2_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_AP_result[b][0], h_dev_AP_ptrs[b], size_AP, hipMemcpyDeviceToHost);
      for i := 0 to size_AP div SizeOf(Single) - 1 do
        CheckEqualsRelative(h_AP_expected[b][i], h_AP_result[b][i], 1e-5, uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i));
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_DSPR2_batched;
const
  N = 10;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TDoublePackedMatrix = array[0..N*(N+1) div 2 - 1] of Double;
  TDoubleVector = array[0..N-1] of Double;
  TDoubleFullMatrix = array[0..N*N-1] of Double;
var
  h_AP_initial, h_AP_expected, h_AP_result: array[0..batch_count-1] of TDoublePackedMatrix;
  h_A_full: array[0..batch_count-1] of TDoubleFullMatrix;
  h_x_batch, h_y_batch: array[0..batch_count-1] of TDoubleVector;
  h_dev_AP_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha: Double;
  i, j, b, k: Integer;
  size_AP, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha := 1.5;
    size_AP := (N * (N + 1) div 2) * SizeOf(Double);
    size_x := N * incx * SizeOf(Double);
    size_y := N * incy * SizeOf(Double);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := j to N - 1 do begin h_A_full[b][i + j * N] := 1.0 + b + i + j; h_A_full[b][j + i * N] := h_A_full[b][i + j * N]; end;
        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
      for i := 0 to N - 1 do begin h_x_batch[b][i] := 0.5 + b + i; h_y_batch[b][i] := 0.5 - b - i; end;
      for j := 0 to N - 1 do for i := 0 to N - 1 do
        h_A_full[b][i + j * N] := h_A_full[b][i + j * N] + alpha * (h_x_batch[b][i] * h_y_batch[b][j] + h_y_batch[b][i] * h_x_batch[b][j]);
      k := 0;
      for j := 0 to N - 1 do
      begin
        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_initial[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_dspr2_batched(handle, uplo, N, @alpha, PPDouble(d_x_ptrs), incx, PPDouble(d_y_ptrs), incy, PPDouble(d_AP_ptrs), batch_count);
    AssertEquals('rocblas_dspr2_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_AP_result[b][0], h_dev_AP_ptrs[b], size_AP, hipMemcpyDeviceToHost);
      for i := 0 to size_AP div SizeOf(Double) - 1 do
        AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i), h_AP_expected[b][i], h_AP_result[b][i], 1e-12);
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_CHPR2_batched;
const
  N = 10;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TComplexPackedMatrix = array[0..N*(N+1) div 2 - 1] of rocblas_float_complex;
  TComplexVector = array[0..N-1] of rocblas_float_complex;
  TComplexFullMatrix = array[0..N*N-1] of rocblas_float_complex;
var
  h_AP_initial, h_AP_expected, h_AP_result: array[0..batch_count-1] of TComplexPackedMatrix;
  h_A_full: array[0..batch_count-1] of TComplexFullMatrix;
  h_x_batch, h_y_batch: array[0..batch_count-1] of TComplexVector;
  h_dev_AP_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, alpha_conj, term1, term2, update, temp: rocblas_float_complex;
  i, j, b, k: Integer;
  size_AP, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha.x := 1.5; alpha.y := -0.5;
    alpha_conj.x := alpha.x; alpha_conj.y := -alpha.y;
    size_AP := (N * (N + 1) div 2) * SizeOf(rocblas_float_complex);
    size_x := N * incx * SizeOf(rocblas_float_complex);
    size_y := N * incy * SizeOf(rocblas_float_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := j to N - 1 do
        begin
          if i = j then begin h_A_full[b][i + j*N].x := 2+b+i; h_A_full[b][i + j*N].y := 0.0; end
          else
          begin
            h_A_full[b][i + j*N].x := 1+i; h_A_full[b][i + j*N].y := 0.5+j;
            h_A_full[b][j + i*N].x := h_A_full[b][i + j*N].x;
            h_A_full[b][j + i*N].y := -h_A_full[b][i + j*N].y;
          end;
        end;

        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;
      for i := 0 to N - 1 do begin h_y_batch[b][i].x := 0.5-i; h_y_batch[b][i].y := 0.2*b+i; end;

      for j := 0 to N - 1 do for i := 0 to N - 1 do
      begin
        term1 := h_x_batch[b][i]; temp := h_y_batch[b][j]; temp.y := -temp.y;
        ComplexMul(term1, temp); ComplexMul(term1, alpha);
        term2 := h_y_batch[b][i]; temp := h_x_batch[b][j]; temp.y := -temp.y;
        ComplexMul(term2, temp); ComplexMul(term2, alpha_conj);
        update.x := term1.x + term2.x; update.y := term1.y + term2.y;
        ComplexAdd(h_A_full[b][i+j*N], update);
      end;
      k := 0;
      for j := 0 to N - 1 do
      begin
        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_initial[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_chpr2_batched(handle, uplo, N, @alpha, PPProcfloat_complex(d_x_ptrs), incx, PPProcfloat_complex(d_y_ptrs), incy, PPProcfloat_complex(d_AP_ptrs), batch_count);
    AssertEquals('rocblas_chpr2_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_AP_result[b][0], h_dev_AP_ptrs[b], size_AP, hipMemcpyDeviceToHost);
      for i := 0 to size_AP div SizeOf(rocblas_float_complex) - 1 do
      begin
        if Abs(h_AP_expected[b][i].x) < 1e-4 then
           AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x', h_AP_expected[b][i].x, h_AP_result[b][i].x, 1e-4)
        else
           CheckEqualsRelative(h_AP_expected[b][i].x, h_AP_result[b][i].x, 1e-4, uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x');
        if Abs(h_AP_expected[b][i].y) < 1e-4 then
           AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y', h_AP_expected[b][i].y, h_AP_result[b][i].y, 1e-4)
        else
           CheckEqualsRelative(h_AP_expected[b][i].y, h_AP_result[b][i].y, 1e-4, uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y');
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

procedure TTestRocBLAS_Level2_batched.Test_ZHPR2_batched;
const
  N = 10;
  incx = 1;
  incy = 1;
  batch_count = 2;
type
  TDoubleComplexPackedMatrix = array[0..N*(N+1) div 2 - 1] of rocblas_double_complex;
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
  TDoubleComplexFullMatrix = array[0..N*N-1] of rocblas_double_complex;
var
  h_AP_initial, h_AP_expected, h_AP_result: array[0..batch_count-1] of TDoubleComplexPackedMatrix;
  h_A_full: array[0..batch_count-1] of TDoubleComplexFullMatrix;
  h_x_batch, h_y_batch: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_AP_ptrs, h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_AP_ptrs, d_x_ptrs, d_y_ptrs: hipDeviceptr_t;
  alpha, alpha_conj, term1, term2, update, temp: rocblas_double_complex;
  i, j, b, k: Integer;
  size_AP, size_x, size_y, size_ptrs: SizeUInt;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  uplo: rocblas_fill;
  uplo_str: string;
begin
  for uplo in [rocblas_fill_lower, rocblas_fill_upper] do
  begin
    case uplo of
      rocblas_fill_lower: uplo_str := 'rocblas_fill_lower';
      rocblas_fill_upper: uplo_str := 'rocblas_fill_upper';
      else uplo_str := 'unknown';
    end;

    alpha.x := 1.5; alpha.y := -0.5;
    alpha_conj.x := alpha.x; alpha_conj.y := -alpha.y;
    size_AP := (N * (N + 1) div 2) * SizeOf(rocblas_double_complex);
    size_x := N * incx * SizeOf(rocblas_double_complex);
    size_y := N * incy * SizeOf(rocblas_double_complex);
    size_ptrs := batch_count * SizeOf(hipDeviceptr_t);
    d_AP_ptrs := nil; d_x_ptrs := nil; d_y_ptrs := nil;

    for b := 0 to batch_count - 1 do
    begin
      k := 0;
      for j := 0 to N - 1 do
      begin
        for i := j to N - 1 do
        begin
          if i = j then begin h_A_full[b][i + j*N].x := 2+b+i; h_A_full[b][i + j*N].y := 0.0; end
          else
          begin
            h_A_full[b][i + j*N].x := 1+i; h_A_full[b][i + j*N].y := 0.5+j;
            h_A_full[b][j + i*N].x := h_A_full[b][i + j*N].x;
            h_A_full[b][j + i*N].y := -h_A_full[b][i + j*N].y;
          end;
        end;

        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_initial[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
      for i := 0 to N - 1 do begin h_x_batch[b][i].x := 0.5+i; h_x_batch[b][i].y := 0.2*b-i; end;
      for i := 0 to N - 1 do begin h_y_batch[b][i].x := 0.5-i; h_y_batch[b][i].y := 0.2*b+i; end;

      for j := 0 to N - 1 do for i := 0 to N - 1 do
      begin
        term1 := h_x_batch[b][i]; temp := h_y_batch[b][j]; temp.y := -temp.y;
        ComplexMul(term1, temp); ComplexMul(term1, alpha);
        term2 := h_y_batch[b][i]; temp := h_x_batch[b][j]; temp.y := -temp.y;
        ComplexMul(term2, temp); ComplexMul(term2, alpha_conj);
        update.x := term1.x + term2.x; update.y := term1.y + term2.y;
        ComplexAdd(h_A_full[b][i+j*N], update);
      end;
      k := 0;
      for j := 0 to N - 1 do
      begin
        if uplo = rocblas_fill_upper then for i := 0 to j do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end
        else for i := j to N - 1 do begin h_AP_expected[b][k] := h_A_full[b][i + j * N]; Inc(k); end;
      end;
    end;

    for b := 0 to batch_count - 1 do
    begin
      hipMalloc(h_dev_AP_ptrs[b], size_AP); hipMalloc(h_dev_x_ptrs[b], size_x); hipMalloc(h_dev_y_ptrs[b], size_y);
      hipMemcpy(h_dev_AP_ptrs[b], @h_AP_initial[b][0], size_AP, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_x_ptrs[b], @h_x_batch[b][0], size_x, hipMemcpyHostToDevice);
      hipMemcpy(h_dev_y_ptrs[b], @h_y_batch[b][0], size_y, hipMemcpyHostToDevice);
    end;
    hipMalloc(d_AP_ptrs, size_ptrs); hipMalloc(d_x_ptrs, size_ptrs); hipMalloc(d_y_ptrs, size_ptrs);
    hipMemcpy(d_AP_ptrs, @h_dev_AP_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_x_ptrs, @h_dev_x_ptrs[0], size_ptrs, hipMemcpyHostToDevice);
    hipMemcpy(d_y_ptrs, @h_dev_y_ptrs[0], size_ptrs, hipMemcpyHostToDevice);

    blas_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(blas_status));

    blas_status := rocblas_zhpr2_batched(handle, uplo, N, @alpha, PPProcdouble_complex(d_x_ptrs), incx, PPProcdouble_complex(d_y_ptrs), incy, PPProcdouble_complex(d_AP_ptrs), batch_count);
    AssertEquals('rocblas_zhpr2_batched failed for uplo=' + uplo_str, Integer(rocblas_status_success), Integer(blas_status));

    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_AP_result[b][0], h_dev_AP_ptrs[b], size_AP, hipMemcpyDeviceToHost);
      for i := 0 to size_AP div SizeOf(rocblas_double_complex) - 1 do
      begin
        AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.x', h_AP_expected[b][i].x, h_AP_result[b][i].x, 1e-12);
        AssertEquals(uplo_str + ', B=' + IntToStr(b) + ', Elem=' + IntToStr(i) + '.y', h_AP_expected[b][i].y, h_AP_result[b][i].y, 1e-12);
      end;
    end;

    for b := 0 to batch_count - 1 do begin hipFree(h_dev_AP_ptrs[b]); hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
    hipFree(d_AP_ptrs); hipFree(d_x_ptrs); hipFree(d_y_ptrs);
  end;
end;

initialization
  RegisterTest(TTestRocBLAS_Level2_batched);

end.

