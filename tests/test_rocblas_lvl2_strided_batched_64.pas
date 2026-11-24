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

  Signature checks for the rocBLAS level-2 (matrix-vector, strided-batched,
  64-bit index variants) bindings: every call is made and its status is
  asserted, so a changed prototype or a missing symbol fails here first.
------------------------------------------------------------------------------}

unit Test_rocblas_lvl2_strided_batched_64;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  { TTestRocBLAS_Level2_strided_batched_64 }

  TTestRocBLAS_Level2_strided_batched_64 = class(TTestRocBLAS_Base)
  published
    { General Matrix-Vector }
    procedure Test_SGEMV_strided_batched_64;
    procedure Test_DGEMV_strided_batched_64;
    procedure Test_CGEMV_strided_batched_64;
    procedure Test_ZGEMV_strided_batched_64;
    procedure Test_SGBMV_strided_batched_64;
    procedure Test_DGBMV_strided_batched_64;
    procedure Test_CGBMV_strided_batched_64;
    procedure Test_ZGBMV_strided_batched_64;

    { Hermitian }
    procedure Test_CHEMV_strided_batched_64;
    procedure Test_ZHEMV_strided_batched_64;
    //procedure Test_CHBMV_strided_batched_64;
    //procedure Test_ZHBMV_strided_batched_64;
    procedure Test_CHPMV_strided_batched_64;
    procedure Test_ZHPMV_strided_batched_64;

    { Symmetric }
    procedure Test_SSYMV_strided_batched_64;
    procedure Test_DSYMV_strided_batched_64;
    //procedure Test_CSYMV_strided_batched_64;
    //procedure Test_ZSYMV_strided_batched_64;
    procedure Test_SSBMV_strided_batched_64;
    procedure Test_DSBMV_strided_batched_64;
    procedure Test_SSPMV_strided_batched_64;
    procedure Test_DSPMV_strided_batched_64;

    { Triangular }
    procedure Test_STRMV_strided_batched_64;
    procedure Test_DTRMV_strided_batched_64;
    procedure Test_CTRMV_strided_batched_64;
    procedure Test_ZTRMV_strided_batched_64;
    //procedure Test_STBMV_strided_batched_64;
    //procedure Test_DTBMV_strided_batched_64;
    //procedure Test_CTBMV_strided_batched_64;
    //procedure Test_ZTBMV_strided_batched_64;
    procedure Test_STPMV_strided_batched_64;
    procedure Test_DTPMV_strided_batched_64;
    procedure Test_CTPMV_strided_batched_64;
    procedure Test_ZTPMV_strided_batched_64;

    { Triangular Solvers }
    procedure Test_STRSV_strided_batched_64;
    procedure Test_DTRSV_strided_batched_64;
    procedure Test_CTRSV_strided_batched_64;
    procedure Test_ZTRSV_strided_batched_64;
    procedure Test_STBSV_strided_batched_64;
    procedure Test_DTBSV_strided_batched_64;
    procedure Test_CTBSV_strided_batched_64;
    procedure Test_ZTBSV_strided_batched_64;
    procedure Test_STPSV_strided_batched_64;
    procedure Test_DTPSV_strided_batched_64;
    procedure Test_CTPSV_strided_batched_64;
    procedure Test_ZTPSV_strided_batched_64;

    { Rank-1 Updates }
    procedure Test_SGER_strided_batched_64;
    procedure Test_DGER_strided_batched_64;
    procedure Test_CGERU_strided_batched_64;
    procedure Test_ZGERU_strided_batched_64;
    procedure Test_CGERC_strided_batched_64;
    procedure Test_ZGERC_strided_batched_64;
    procedure Test_SSYR_strided_batched_64;
    procedure Test_DSYR_strided_batched_64;
    //procedure Test_CSYR_strided_batched_64;
    //procedure Test_ZSYR_strided_batched_64;
    procedure Test_CHER_strided_batched_64;
    procedure Test_ZHER_strided_batched_64;
    procedure Test_SSPR_strided_batched_64;
    procedure Test_DSPR_strided_batched_64;
    procedure Test_CHPR_strided_batched_64;
    procedure Test_ZHPR_strided_batched_64;

    { Rank-2 Updates }
    procedure Test_SSYR2_strided_batched_64;
    procedure Test_DSYR2_strided_batched_64;
    //procedure Test_CSYR2_strided_batched_64;
    //procedure Test_ZSYR2_strided_batched_64;
    procedure Test_CHER2_strided_batched_64;
    procedure Test_ZHER2_strided_batched_64;
    procedure Test_SSPR2_strided_batched_64;
    procedure Test_DSPR2_strided_batched_64;
    procedure Test_CHPR2_strided_batched_64;
    procedure Test_ZHPR2_strided_batched_64;
  end;

implementation

{ TTestRocBLAS_Level2_strided_batched_64 }

procedure TTestRocBLAS_Level2_strided_batched_64.Test_SGEMV_strided_batched_64;
const
  M = 3; N = 4; batch_count = 2;
  alpha: Single = 2.0; beta: Single = 1.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of Single;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
begin
  strideA := M * N;
  strideX := N;
  strideY := M;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
        hostA[b*strideA + j*M + i] := (b+1)*100 + (j+1)*10 + (i+1);
    for i := 0 to N - 1 do hostX[b*strideX + i] := i + 1.0;
    for i := 0 to M - 1 do hostY[b*strideY + i] := i * 2.0;
  end;

  for b := 0 to batch_count - 1 do
    for i := 0 to M - 1 do
    begin
      expectedY[b*strideY + i] := beta * hostY[b*strideY + i];
      for j := 0 to N - 1 do
        expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * hostA[b*strideA + j*M + i] * hostX[b*strideX + j];
    end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_sgemv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_sgemv_strided_batched_64(handle, rocblas_operation_none, M, N,
                                          @alpha, devA, M, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-5, Format('SGEMV_strided_batched_64 Y[%d]',[i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DGEMV_strided_batched_64;
const
  M = 3; N = 4; batch_count = 2;
  alpha: Double = 2.0; beta: Double = 1.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of Double;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
begin
  strideA := M * N;
  strideX := N;
  strideY := M;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
        hostA[b*strideA + j*M + i] := (b+1)*100 + (j+1)*10 + (i+1);
    for i := 0 to N - 1 do hostX[b*strideX + i] := i + 1.0;
    for i := 0 to M - 1 do hostY[b*strideY + i] := i * 2.0;
  end;

  for b := 0 to batch_count - 1 do
    for i := 0 to M - 1 do
    begin
      expectedY[b*strideY + i] := beta * hostY[b*strideY + i];
      for j := 0 to N - 1 do
        expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * hostA[b*strideA + j*M + i] * hostX[b*strideX + j];
    end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dgemv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dgemv_strided_batched_64(handle, rocblas_operation_none, M, N,
                                          @alpha, devA, M, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-12, Format('DGEMV_strided_batched_64 Y[%d]',[i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CGEMV_strided_batched_64;
const
  M = 3; N = 4; batch_count = 2;
  alpha: rocblas_float_complex = (x: 2.0; y: 1.0); // 2 + i
  beta: rocblas_float_complex = (x: 1.5; y: -0.5); // 1.5 - 0.5i
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of rocblas_float_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  temp_sum: rocblas_float_complex;
begin
  strideA := M * N;
  strideX := N;
  strideY := M;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        hostA[b*strideA + j*M + i].x := (b+1)*100 + (j+1)*10 + (i+1);
        hostA[b*strideA + j*M + i].y := (b+1)*10 + (j+1) + (i+1)*0.1;
      end;
    for i := 0 to N - 1 do
    begin
       hostX[b*strideX + i].x := i + 1.0;
       hostX[b*strideX + i].y := (i + 1) * 0.5;
    end;
    for i := 0 to M - 1 do
    begin
       hostY[b*strideY + i].x := i * 2.0;
       hostY[b*strideY + i].y := i * 1.5;
    end;
  end;

  // Compute expected result: Y = alpha * A * X + beta * Y
  for b := 0 to batch_count - 1 do
    for i := 0 to M - 1 do
    begin
      // expectedY[b*strideY + i] = beta * hostY[b*strideY + i];
      expectedY[b*strideY + i].x := beta.x * hostY[b*strideY + i].x - beta.y * hostY[b*strideY + i].y;
      expectedY[b*strideY + i].y := beta.x * hostY[b*strideY + i].y + beta.y * hostY[b*strideY + i].x;

      temp_sum.x := 0.0; temp_sum.y := 0.0; // Initialize sum for A * X
      for j := 0 to N - 1 do
      begin
        // temp_sum += A[i,j] * X[j]
        temp_sum.x := temp_sum.x + (hostA[b*strideA + j*M + i].x * hostX[b*strideX + j].x - hostA[b*strideA + j*M + i].y * hostX[b*strideX + j].y);
        temp_sum.y := temp_sum.y + (hostA[b*strideA + j*M + i].x * hostX[b*strideX + j].y + hostA[b*strideA + j*M + i].y * hostX[b*strideX + j].x);
      end;
      // expectedY[b*strideY + i] += alpha * temp_sum;
      expectedY[b*strideY + i].x := expectedY[b*strideY + i].x + (alpha.x * temp_sum.x - alpha.y * temp_sum.y);
      expectedY[b*strideY + i].y := expectedY[b*strideY + i].y + (alpha.x * temp_sum.y + alpha.y * temp_sum.x);
    end;


  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_cgemv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_cgemv_strided_batched_64(handle, rocblas_operation_none, M, N,
                                          @alpha, devA, M, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-4, Format('CGEMV_strided_batched_64 Y[%d].x',[i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-4, Format('CGEMV_strided_batched_64 Y[%d].y',[i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZGEMV_strided_batched_64;
const
  M = 3; N = 4; batch_count = 2;
  alpha: rocblas_double_complex = (x: 2.0; y: 1.0); // 2 + i
  beta: rocblas_double_complex = (x: 1.5; y: -0.5); // 1.5 - 0.5i
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of rocblas_double_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  temp_sum: rocblas_double_complex;
begin
  strideA := M * N;
  strideX := N;
  strideY := M;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        hostA[b*strideA + j*M + i].x := (b+1)*100 + (j+1)*10 + (i+1);
        hostA[b*strideA + j*M + i].y := (b+1)*10 + (j+1) + (i+1)*0.1;
      end;
    for i := 0 to N - 1 do
    begin
       hostX[b*strideX + i].x := i + 1.0;
       hostX[b*strideX + i].y := (i + 1) * 0.5;
    end;
    for i := 0 to M - 1 do
    begin
       hostY[b*strideY + i].x := i * 2.0;
       hostY[b*strideY + i].y := i * 1.5;
    end;
  end;

  // Compute expected result: Y = alpha * A * X + beta * Y
  for b := 0 to batch_count - 1 do
    for i := 0 to M - 1 do
    begin
      // expectedY[b*strideY + i] = beta * hostY[b*strideY + i];
      expectedY[b*strideY + i].x := beta.x * hostY[b*strideY + i].x - beta.y * hostY[b*strideY + i].y;
      expectedY[b*strideY + i].y := beta.x * hostY[b*strideY + i].y + beta.y * hostY[b*strideY + i].x;

      temp_sum.x := 0.0; temp_sum.y := 0.0; // Initialize sum for A * X
      for j := 0 to N - 1 do
      begin
        // temp_sum += A[i,j] * X[j]
        temp_sum.x := temp_sum.x + (hostA[b*strideA + j*M + i].x * hostX[b*strideX + j].x - hostA[b*strideA + j*M + i].y * hostX[b*strideX + j].y);
        temp_sum.y := temp_sum.y + (hostA[b*strideA + j*M + i].x * hostX[b*strideX + j].y + hostA[b*strideA + j*M + i].y * hostX[b*strideX + j].x);
      end;
      // expectedY[b*strideY + i] += alpha * temp_sum;
      expectedY[b*strideY + i].x := expectedY[b*strideY + i].x + (alpha.x * temp_sum.x - alpha.y * temp_sum.y);
      expectedY[b*strideY + i].y := expectedY[b*strideY + i].y + (alpha.x * temp_sum.y + alpha.y * temp_sum.x);
    end;


  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_zgemv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_zgemv_strided_batched_64(handle, rocblas_operation_none, M, N,
                                          @alpha, devA, M, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-12, Format('ZGEMV_strided_batched_64 Y[%d].x',[i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-12, Format('ZGEMV_strided_batched_64 Y[%d].y',[i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_SGBMV_strided_batched_64;
const
  M = 4; N = 5; KL = 1; KU = 2; // Example: 4x5 matrix, 1 sub-diagonal, 2 super-diagonals
  batch_count = 2;
  alpha: Single = 2.0;
  beta: Single = 1.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of Single;
  strideA, strideX, strideY: Int64;
  i, j, b, k, row, col: Integer;
  A_ld, A_size_per_batch: Int64;
begin
  // Leading dimension for banded storage (KL + KU + 1)
  A_ld := KL + KU + 1;
  // Stride for matrix A: number of elements per batch (columns * leading dimension)
  strideA := N * A_ld;
  strideX := N; // Stride for vector X
  strideY := M; // Stride for vector Y

  A_size_per_batch := strideA;

  SetLength(hostA, A_size_per_batch * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Initialize A (banded storage - Fortran style, but accessed column-wise in our flat array)
    // A[ku+1+i-j, j] = matrix[i, j] if max(0, j-ku) <= i <= min(m-1, j+kl)
    for j := 0 to N - 1 do // Iterate through columns
    begin
      for k := 0 to A_ld - 1 do // Iterate through rows in banded storage column
      begin
         hostA[b * strideA + j * A_ld + k] := 0.0; // Zero out first
      end;
      // Fill relevant bands
      for i := Max(0, j - KU) to Min(M - 1, j + KL) do
      begin
        // k is the row index within the banded storage column (j * A_ld)
        k := (KU + 1 + i - j) - 1; // Convert 1-based Fortran index to 0-based C index
        // Example simple value based on position and batch
        hostA[b * strideA + j * A_ld + k] := (b + 1) * 100 + (i + 1) * 10 + (j + 1);
      end;
    end;

    // Initialize X
    for i := 0 to N - 1 do
      hostX[b * strideX + i] := (i + 1) * 1.0;

    // Initialize Y
    for i := 0 to M - 1 do
      hostY[b * strideY + i] := i * 2.0;
  end;

  // Calculate expected result: Y = alpha * A * X + beta * Y
  // Note: A is MxN, X is Nx1, Y is Mx1
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M - 1 do // For each row of result vector Y
    begin
      expectedY[b * strideY + i] := beta * hostY[b * strideY + i]; // beta * Y_part
      for j := 0 to N - 1 do // For dot product A_row * X
      begin
        // Check if A[i, j] is within the band
        if ((j >= i - KL) and (j <= i + KU)) then
        begin
          // Find the stored index in the banded format
          k := (KU + 1 + i - j) - 1; // Convert 1-based Fortran index to 0-based
          expectedY[b * strideY + i] := expectedY[b * strideY + i] + alpha * hostA[b * strideA + j * A_ld + k] * hostX[b * strideX + j];
        end;
      end;
    end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, A_size_per_batch * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], A_size_per_batch * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_sgbmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_sgbmv_strided_batched_64(handle, rocblas_operation_none, M, N, KL, KU,
                                          @alpha, devA, A_ld, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-4, Format('SGBMV_strided_batched_64 Y[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DGBMV_strided_batched_64;
const
  M = 4; N = 5; KL = 1; KU = 2; // Example: 4x5 matrix, 1 sub-diagonal, 2 super-diagonals
  batch_count = 2;
  alpha: Double = 2.0;
  beta: Double = 1.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of Double;
  strideA, strideX, strideY: Int64;
  i, j, b, k, row, col: Integer;
  A_ld, A_size_per_batch: Int64;
begin
  // Leading dimension for banded storage (KL + KU + 1)
  A_ld := KL + KU + 1;
  // Stride for matrix A: number of elements per batch (columns * leading dimension)
  strideA := N * A_ld;
  strideX := N; // Stride for vector X
  strideY := M; // Stride for vector Y

  A_size_per_batch := strideA;

  SetLength(hostA, A_size_per_batch * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Initialize A (banded storage - Fortran style, but accessed column-wise in our flat array)
    // A[ku+1+i-j, j] = matrix[i, j] if max(0, j-ku) <= i <= min(m-1, j+kl)
    for j := 0 to N - 1 do // Iterate through columns
    begin
      for k := 0 to A_ld - 1 do // Iterate through rows in banded storage column
      begin
         hostA[b * strideA + j * A_ld + k] := 0.0; // Zero out first
      end;
      // Fill relevant bands
      for i := Max(0, j - KU) to Min(M - 1, j + KL) do
      begin
        // k is the row index within the banded storage column (j * A_ld)
        k := (KU + 1 + i - j) - 1; // Convert 1-based Fortran index to 0-based C index
        // Example simple value based on position and batch
        hostA[b * strideA + j * A_ld + k] := (b + 1) * 100 + (i + 1) * 10 + (j + 1);
      end;
    end;

    // Initialize X
    for i := 0 to N - 1 do
      hostX[b * strideX + i] := (i + 1) * 1.0;

    // Initialize Y
    for i := 0 to M - 1 do
      hostY[b * strideY + i] := i * 2.0;
  end;

  // Calculate expected result: Y = alpha * A * X + beta * Y
  // Note: A is MxN, X is Nx1, Y is Mx1
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M - 1 do // For each row of result vector Y
    begin
      expectedY[b * strideY + i] := beta * hostY[b * strideY + i]; // beta * Y_part
      for j := 0 to N - 1 do // For dot product A_row * X
      begin
        // Check if A[i, j] is within the band
        if ((j >= i - KL) and (j <= i + KU)) then
        begin
          // Find the stored index in the banded format
          k := (KU + 1 + i - j) - 1; // Convert 1-based Fortran index to 0-based
          expectedY[b * strideY + i] := expectedY[b * strideY + i] + alpha * hostA[b * strideA + j * A_ld + k] * hostX[b * strideX + j];
        end;
      end;
    end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, A_size_per_batch * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], A_size_per_batch * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dgbmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dgbmv_strided_batched_64(handle, rocblas_operation_none, M, N, KL, KU,
                                          @alpha, devA, A_ld, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-12, Format('DGBMV_strided_batched_64 Y[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CGBMV_strided_batched_64;
const
  M = 4; N = 5; KL = 1; KU = 2; // Example: 4x5 matrix, 1 sub-diagonal, 2 super-diagonals
  batch_count = 2;
  alpha: rocblas_float_complex = (x: 2.0; y: 1.0); // 2 + i
  beta: rocblas_float_complex = (x: 1.5; y: -0.5); // 1.5 - 0.5i
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of rocblas_float_complex;
  strideA, strideX, strideY: Int64;
  i, j, b, k: Integer;
  A_ld, A_size_per_batch: Int64;
  temp_val: Single;
begin
  // Leading dimension for banded storage (KL + KU + 1)
  A_ld := KL + KU + 1;
  // Stride for matrix A: number of elements per batch (columns * leading dimension)
  strideA := N * A_ld;
  strideX := N; // Stride for vector X
  strideY := M; // Stride for vector Y

  A_size_per_batch := strideA;

  SetLength(hostA, A_size_per_batch * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Initialize A (banded storage)
    for j := 0 to N - 1 do // Iterate through columns
    begin
      for k := 0 to A_ld - 1 do // Iterate through rows in banded storage column
      begin
         hostA[b * strideA + j * A_ld + k].x := 0.0;
         hostA[b * strideA + j * A_ld + k].y := 0.0;
      end;
      // Fill relevant bands
      for i := Max(0, j - KU) to Min(M - 1, j + KL) do
      begin
        k := (KU + 1 + i - j) - 1; // Convert 1-based Fortran index to 0-based C index
        // Real part based on position and batch
        hostA[b * strideA + j * A_ld + k].x := (b + 1) * 100 + (i + 1) * 10 + (j + 1);
        // Imaginary part based on position and batch
        hostA[b * strideA + j * A_ld + k].y := (b + 1) * 10 + (i + 1) + (j + 1) * 0.1;
      end;
    end;

    // Initialize X
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := (i + 1) * 1.0;
      hostX[b * strideX + i].y := (i + 1) * 0.5;
    end;

    // Initialize Y
    for i := 0 to M - 1 do
    begin
      hostY[b * strideY + i].x := i * 2.0;
      hostY[b * strideY + i].y := i * 1.5;
    end;
  end;

  // Calculate expected result: Y = alpha * A * X + beta * Y
  // Complex multiply-add: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M - 1 do // For each row of result vector Y
    begin
      // expectedY[b * strideY + i] = beta * hostY[b * strideY + i];
      expectedY[b * strideY + i].x := beta.x * hostY[b * strideY + i].x - beta.y * hostY[b * strideY + i].y;
      expectedY[b * strideY + i].y := beta.x * hostY[b * strideY + i].y + beta.y * hostY[b * strideY + i].x;

      for j := 0 to N - 1 do // For dot product A_row * X
      begin
        // Check if A[i, j] is within the band
        if ((j >= i - KL) and (j <= i + KU)) then
        begin
          k := (KU + 1 + i - j) - 1; // Convert 1-based Fortran index to 0-based
          // temp_val = alpha * A[i,j] * X[j]
          // (alpha.x + alpha.y*i) * (A.x + A.y*i) * (X.x + X.y*i)
          // First: alpha * A
          temp_val := alpha.x * hostA[b * strideA + j * A_ld + k].x - alpha.y * hostA[b * strideA + j * A_ld + k].y; // Real part
          expectedY[b * strideY + i].x := expectedY[b * strideY + i].x + (temp_val * hostX[b * strideX + j].x - (alpha.x * hostA[b * strideA + j * A_ld + k].y + alpha.y * hostA[b * strideA + j * A_ld + k].x) * hostX[b * strideX + j].y);

          temp_val := alpha.x * hostA[b * strideA + j * A_ld + k].y + alpha.y * hostA[b * strideA + j * A_ld + k].x; // Imaginary part
          expectedY[b * strideY + i].y := expectedY[b * strideY + i].y + (temp_val * hostX[b * strideX + j].x + (alpha.x * hostA[b * strideA + j * A_ld + k].x - alpha.y * hostA[b * strideA + j * A_ld + k].y) * hostX[b * strideX + j].y);
        end;
      end;
    end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, A_size_per_batch * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], A_size_per_batch * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_cgbmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_cgbmv_strided_batched_64(handle, rocblas_operation_none, M, N, KL, KU,
                                          @alpha, devA, A_ld, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-4, Format('CGBMV_strided_batched_64 Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-4, Format('CGBMV_strided_batched_64 Y[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZGBMV_strided_batched_64;
const
  M = 4; N = 5; KL = 1; KU = 2; // Example: 4x5 matrix, 1 sub-diagonal, 2 super-diagonals
  batch_count = 2;
  alpha: rocblas_double_complex = (x: 2.0; y: 1.0); // 2 + i
  beta: rocblas_double_complex = (x: 1.5; y: -0.5); // 1.5 - 0.5i
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of rocblas_double_complex;
  strideA, strideX, strideY: Int64;
  i, j, b, k: Integer;
  A_ld, A_size_per_batch: Int64;
  temp_val: Double;
begin
  // Leading dimension for banded storage (KL + KU + 1)
  A_ld := KL + KU + 1;
  // Stride for matrix A: number of elements per batch (columns * leading dimension)
  strideA := N * A_ld;
  strideX := N; // Stride for vector X
  strideY := M; // Stride for vector Y

  A_size_per_batch := strideA;

  SetLength(hostA, A_size_per_batch * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Initialize A (banded storage)
    for j := 0 to N - 1 do // Iterate through columns
    begin
      for k := 0 to A_ld - 1 do // Iterate through rows in banded storage column
      begin
         hostA[b * strideA + j * A_ld + k].x := 0.0;
         hostA[b * strideA + j * A_ld + k].y := 0.0;
      end;
      // Fill relevant bands
      for i := Max(0, j - KU) to Min(M - 1, j + KL) do
      begin
        k := (KU + 1 + i - j) - 1; // Convert 1-based Fortran index to 0-based C index
        // Real part based on position and batch
        hostA[b * strideA + j * A_ld + k].x := (b + 1) * 100 + (i + 1) * 10 + (j + 1);
        // Imaginary part based on position and batch
        hostA[b * strideA + j * A_ld + k].y := (b + 1) * 10 + (i + 1) + (j + 1) * 0.1;
      end;
    end;

    // Initialize X
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := (i + 1) * 1.0;
      hostX[b * strideX + i].y := (i + 1) * 0.5;
    end;

    // Initialize Y
    for i := 0 to M - 1 do
    begin
      hostY[b * strideY + i].x := i * 2.0;
      hostY[b * strideY + i].y := i * 1.5;
    end;
  end;

  // Calculate expected result: Y = alpha * A * X + beta * Y
  // Complex multiply-add: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M - 1 do // For each row of result vector Y
    begin
      // expectedY[b * strideY + i] = beta * hostY[b * strideY + i];
      expectedY[b * strideY + i].x := beta.x * hostY[b * strideY + i].x - beta.y * hostY[b * strideY + i].y;
      expectedY[b * strideY + i].y := beta.x * hostY[b * strideY + i].y + beta.y * hostY[b * strideY + i].x;

      for j := 0 to N - 1 do // For dot product A_row * X
      begin
        // Check if A[i, j] is within the band
        if ((j >= i - KL) and (j <= i + KU)) then
        begin
          k := (KU + 1 + i - j) - 1; // Convert 1-based Fortran index to 0-based
          // temp_val = alpha * A[i,j] * X[j]
          // (alpha.x + alpha.y*i) * (A.x + A.y*i) * (X.x + X.y*i)
          // First: alpha * A
          temp_val := alpha.x * hostA[b * strideA + j * A_ld + k].x - alpha.y * hostA[b * strideA + j * A_ld + k].y; // Real part
          expectedY[b * strideY + i].x := expectedY[b * strideY + i].x + (temp_val * hostX[b * strideX + j].x - (alpha.x * hostA[b * strideA + j * A_ld + k].y + alpha.y * hostA[b * strideA + j * A_ld + k].x) * hostX[b * strideX + j].y);

          temp_val := alpha.x * hostA[b * strideA + j * A_ld + k].y + alpha.y * hostA[b * strideA + j * A_ld + k].x; // Imaginary part
          expectedY[b * strideY + i].y := expectedY[b * strideY + i].y + (temp_val * hostX[b * strideX + j].x + (alpha.x * hostA[b * strideA + j * A_ld + k].x - alpha.y * hostA[b * strideA + j * A_ld + k].y) * hostX[b * strideX + j].y);
        end;
      end;
    end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, A_size_per_batch * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], A_size_per_batch * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_zgbmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_zgbmv_strided_batched_64(handle, rocblas_operation_none, M, N, KL, KU,
                                          @alpha, devA, A_ld, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-12, Format('ZGBMV_strided_batched_64 Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-12, Format('ZGBMV_strided_batched_64 Y[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CHEMV_strided_batched_64;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: rocblas_float_complex = (x: 1.5; y: -1.0);
  beta:  rocblas_float_complex = (x: 0.5; y: 2.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of rocblas_float_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  temp, sum_prod: rocblas_float_complex;
  A_ij: rocblas_float_complex;
begin
  strideA := N * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  // Initialize host matrices and vectors
  for b := 0 to batch_count - 1 do
  begin
    // Initialize Hermitian matrix A (upper triangular part)
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if i <= j then
        begin
          hostA[b*strideA + j*N + i].x := (b+1)*10 + i + j + 2;
          if i = j then
            hostA[b*strideA + j*N + i].y := 0.0 // Diagonal elements must be real
          else
            hostA[b*strideA + j*N + i].y := (b+1) + i - j;
        end
        else
        begin
          hostA[b*strideA + j*N + i].x := 0.0; // Lower part is not referenced
          hostA[b*strideA + j*N + i].y := 0.0;
        end;
      end;
    // Initialize vectors x and y
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.5;
      hostX[b*strideX + i].y := i - 0.5;
      hostY[b*strideY + i].x := i * 2.0;
      hostY[b*strideY + i].y := i * -1.5;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      // expectedY = beta * hostY
      expectedY[b*strideY + i] := hostY[b*strideY + i];
      ComplexMul(expectedY[b*strideY + i], beta);

      sum_prod.x := 0; sum_prod.y := 0;
      for j := 0 to N - 1 do
      begin
        // Reconstruct full Hermitian matrix for calculation
        if i <= j then // Upper part
          A_ij := hostA[b*strideA + j*N + i]
        else // Lower part is conjugate of upper part
        begin
          A_ij := hostA[b*strideA + i*N + j];
          A_ij.y := -A_ij.y; // Conjugate
        end;

        temp := hostX[b*strideX + j];
        ComplexMul(temp, A_ij);
        ComplexAdd(sum_prod, temp);
      end;
      // expectedY += alpha * sum_prod
      ComplexMul(sum_prod, alpha);
      ComplexAdd(expectedY[b*strideY + i], sum_prod);
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX H2D', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY H2D', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_chemv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_chemv_strided_batched_64(handle, uplo, N, @alpha, devA, N, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devY D2H', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-3, Format('CHEMV_strided_batched_64 Y[%d].x', [i])); // Relaxed tolerance
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-3, Format('CHEMV_strided_batched_64 Y[%d].y', [i])); // Relaxed tolerance
  end;


  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZHEMV_strided_batched_64;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: rocblas_double_complex = (x: 1.5; y: -1.0);
  beta:  rocblas_double_complex = (x: 0.5; y: 2.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of rocblas_double_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  temp, sum_prod: rocblas_double_complex;
  A_ij: rocblas_double_complex;
begin
  strideA := N * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  // Initialize host matrices and vectors
  for b := 0 to batch_count - 1 do
  begin
    // Initialize Hermitian matrix A (upper triangular part)
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if i <= j then
        begin
          hostA[b*strideA + j*N + i].x := (b+1)*10 + i + j + 2;
          if i = j then
            hostA[b*strideA + j*N + i].y := 0.0 // Diagonal elements must be real
          else
            hostA[b*strideA + j*N + i].y := (b+1) + i - j;
        end
        else
        begin
          hostA[b*strideA + j*N + i].x := 0.0; // Lower part is not referenced
          hostA[b*strideA + j*N + i].y := 0.0;
        end;
      end;
    // Initialize vectors x and y
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.5;
      hostX[b*strideX + i].y := i - 0.5;
      hostY[b*strideY + i].x := i * 2.0;
      hostY[b*strideY + i].y := i * -1.5;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      // expectedY = beta * hostY
      expectedY[b*strideY + i] := hostY[b*strideY + i];
      ComplexMul(expectedY[b*strideY + i], beta);

      sum_prod.x := 0; sum_prod.y := 0;
      for j := 0 to N - 1 do
      begin
        // Reconstruct full Hermitian matrix for calculation
        if i <= j then // Upper part
          A_ij := hostA[b*strideA + j*N + i]
        else // Lower part is conjugate of upper part
        begin
          A_ij := hostA[b*strideA + i*N + j];
          A_ij.y := -A_ij.y; // Conjugate
        end;

        temp := hostX[b*strideX + j];
        ComplexMul(temp, A_ij);
        ComplexAdd(sum_prod, temp);
      end;
      // expectedY += alpha * sum_prod
      ComplexMul(sum_prod, alpha);
      ComplexAdd(expectedY[b*strideY + i], sum_prod);
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX H2D', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY H2D', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_zhemv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_zhemv_strided_batched_64(handle, uplo, N, @alpha, devA, N, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devY D2H', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-10, Format('ZHEMV_strided_batched_64 Y[%d].x', [i])); // Relaxed tolerance
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-10, Format('ZHEMV_strided_batched_64 Y[%d].y', [i])); // Relaxed tolerance
  end;


  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

//procedure TTestRocBLAS_Level2_strided_batched_64.Test_CHBMV_strided_batched_64;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZHBMV_strided_batched_64;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CHPMV_strided_batched_64;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: rocblas_float_complex = (x: 1.5; y: -1.0);
  beta:  rocblas_float_complex = (x: 0.5; y: 2.0);
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostX, hostY, hostY_result, expectedY: array of rocblas_float_complex;
  strideAP, strideX, strideY: Int64;
  i, j, b, idx_flat, idx_packed: Integer;
  temp, sum_prod: rocblas_float_complex;
  A_ij: rocblas_float_complex;
  elements_packed: Int64;
begin
  // For packed format, the number of elements for an NxN matrix is N*(N+1)/2
  elements_packed := (N * (N + 1)) div 2;
  strideAP := elements_packed;
  strideX := N;
  strideY := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  // Initialize host packed matrices and vectors
  for b := 0 to batch_count - 1 do
  begin
    // Initialize Hermitian matrix A in packed format (column-major upper triangular storage)
    // Packed index for column j, row i (i <= j): idx_packed = j*(j+1)/2 + i
    idx_packed := 0;
    for j := 0 to N - 1 do // Iterate through columns
    begin
      for i := 0 to j do // Iterate through rows within the triangular part
      begin
        if i = j then
        begin
          // Diagonal elements are real
          hostAP[b * strideAP + idx_packed].x := (b + 1) * 10 + i + j + 2;
          hostAP[b * strideAP + idx_packed].y := 0.0;
        end
        else
        begin
          // Off-diagonal elements
          hostAP[b * strideAP + idx_packed].x := (b + 1) + i - j;
          hostAP[b * strideAP + idx_packed].y := (b + 1) * 0.5 + i * 0.2 + j * 0.1;
        end;
        Inc(idx_packed);
      end;
    end;

    // Initialize vectors x and y
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := i + 1.5;
      hostX[b * strideX + i].y := i - 0.5;
      hostY[b * strideY + i].x := i * 2.0;
      hostY[b * strideY + i].y := i * -1.5;
    end;
  end;

  // CPU reference calculation (reconstruct full matrix for simplicity)
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      // expectedY = beta * hostY
      expectedY[b * strideY + i] := hostY[b * strideY + i];
      ComplexMul(expectedY[b * strideY + i], beta);

      sum_prod.x := 0; sum_prod.y := 0;
      for j := 0 to N - 1 do
      begin
        // Reconstruct full Hermitian matrix element A[i,j] for calculation
        if i <= j then // Upper part (including diagonal)
        begin
          // Packed index for column j, row i: idx_packed = j*(j+1)/2 + i
          idx_packed := (j * (j + 1)) div 2 + i;
          A_ij := hostAP[b * strideAP + idx_packed];
        end
        else // Lower part is conjugate of upper part
        begin
          // Packed index for column i, row j: idx_packed = i*(i+1)/2 + j
          idx_packed := (i * (i + 1)) div 2 + j;
          A_ij := hostAP[b * strideAP + idx_packed];
          A_ij.y := -A_ij.y; // Conjugate
        end;

        temp := hostX[b * strideX + j];
        ComplexMul(temp, A_ij);
        ComplexAdd(sum_prod, temp);
      end;
      // expectedY += alpha * sum_prod
      ComplexMul(sum_prod, alpha);
      ComplexAdd(expectedY[b * strideY + i], sum_prod);
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devAP H2D', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX H2D', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY H2D', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_chpmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_chpmv_strided_batched_64(handle, uplo, N, @alpha, devAP, strideAP, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devY D2H', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 5e-3, Format('CHPMV_strided_batched_64 Y[%d].x', [i])); // Relaxed tolerance
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 5e-3, Format('CHPMV_strided_batched_64 Y[%d].y', [i])); // Relaxed tolerance
  end;


  hipFree(devAP);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZHPMV_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: rocblas_double_complex = (x: 1.5; y: -1.0);
  beta:  rocblas_double_complex = (x: 0.5; y: 2.0);
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostX, hostY, hostY_result, expectedY: array of rocblas_double_complex;
  fullA: array of rocblas_double_complex; // For reference calculation
  packed_size, strideAP, strideX, strideY: Int64;
  i, j, b, k: Integer;
  temp, sum_prod: rocblas_double_complex;
  A_ij: rocblas_double_complex;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;
  strideY := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);
  SetLength(fullA, N * N * batch_count); // Host full matrix for easy reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full Hermitian matrix (correctly)
    // First, fill the upper triangle and diagonal (elements A(i,j) where i <= j)
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        if i = j then // Diagonal (must be real)
        begin
          fullA[b*N*N + j*N + i].x := (b+1)*20 + i;
          fullA[b*N*N + j*N + i].y := 0.0;
        end
        else // Upper part (A(i,j) where i < j)
        begin
          fullA[b*N*N + j*N + i].x := (b+1)*10 + i + j;
          fullA[b*N*N + j*N + i].y := (b+1) + i - j;
        end;
      end;
    // Now, fill the lower triangle as the conjugate of the upper (A(i,j) = conj(A(j,i)))
    for j := 0 to N - 1 do
      for i := j + 1 to N - 1 do
      begin
         fullA[b*N*N + j*N + i].x :=  fullA[b*N*N + i*N + j].x;
         fullA[b*N*N + j*N + i].y := -fullA[b*N*N + i*N + j].y;
      end;

    // 2. Pack the upper part of the full matrix into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vectors
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.5;
      hostX[b*strideX + i].y := i - 0.5;
      hostY[b*strideY + i].x := i * 2.0;
      hostY[b*strideY + i].y := i * -1.5;
    end;
  end;

  // CPU reference calculation using the correctly built full matrix
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      // expectedY = beta * hostY
      expectedY[b*strideY + i].x := hostY[b*strideY + i].x * beta.x - hostY[b*strideY + i].y * beta.y;
      expectedY[b*strideY + i].y := hostY[b*strideY + i].x * beta.y + hostY[b*strideY + i].y * beta.x;

      sum_prod.x := 0; sum_prod.y := 0;
      for j := 0 to N - 1 do
      begin
        A_ij := fullA[b*N*N + j*N + i]; // A(i,j)
        // temp = A(i,j) * x(j)
        temp.x := A_ij.x * hostX[b*strideX + j].x - A_ij.y * hostX[b*strideX + j].y;
        temp.y := A_ij.x * hostX[b*strideX + j].y + A_ij.y * hostX[b*strideX + j].x;
        // sum += temp
        sum_prod.x := sum_prod.x + temp.x;
        sum_prod.y := sum_prod.y + temp.y;
      end;

      // sum_prod = alpha * sum_prod
      temp.x := alpha.x * sum_prod.x - alpha.y * sum_prod.y;
      temp.y := alpha.x * sum_prod.y + alpha.y * sum_prod.x;
      // expectedY += alpha * sum_prod
      expectedY[b*strideY + i].x := expectedY[b*strideY + i].x + temp.x;
      expectedY[b*strideY + i].y := expectedY[b*strideY + i].y + temp.y;
    end;

  // Allocate and copy to device
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy devAP H2D', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX H2D', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY H2D', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_zhpmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_zhpmv_strided_batched_64(handle, uplo, N, @alpha, devAP, strideAP, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy devY D2H', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-10, Format('ZHPMV_strided_batched_64 Y[%d].x', [i])); // Relaxed tolerance
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-10, Format('ZHPMV_strided_batched_64 Y[%d].y', [i])); // Relaxed tolerance
  end;

  hipFree(devAP);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_SSYMV_strided_batched_64;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Single = 1.5;
  beta:  Single = 2.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of Single;
  fullA: array of Single; // For CPU reference calculation
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
begin
  strideA := N * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full symmetric matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j;
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i]; // A(j,i) = A(i,j)
      end;

    // 2. Copy the referenced part (upper) to the matrix that will be sent to the GPU
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
          hostA[b*strideA + j*N + i] := fullA[b*N*N + j*N + i]
        else
          hostA[b*strideA + j*N + i] := 0.0; // This part is not referenced by the GPU call

    // 3. Initialize vectors
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
      hostY[b*strideY + i] := i * 0.5;
    end;
  end;

  // CPU reference calculation using the full symmetric matrix
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      expectedY[b*strideY + i] := beta * hostY[b*strideY + i];
      for j := 0 to N - 1 do
        expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
    end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_ssymv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ssymv_strided_batched_64(handle, uplo, N, @alpha, devA, N, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-5, Format('SSYMV_strided_batched_64 Y[%d]', [i]));


  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DSYMV_strided_batched_64;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Double = 1.5;
  beta:  Double = 2.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of Double;
  fullA: array of Double; // For CPU reference calculation
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
begin
  strideA := N * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full symmetric matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j;
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i]; // A(j,i) = A(i,j)
      end;

    // 2. Copy the referenced part (upper) to the matrix that will be sent to the GPU
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
          hostA[b*strideA + j*N + i] := fullA[b*N*N + j*N + i]
        else
          hostA[b*strideA + j*N + i] := 0.0; // This part is not referenced by the GPU call

    // 3. Initialize vectors
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
      hostY[b*strideY + i] := i * 0.5;
    end;
  end;

  // CPU reference calculation using the full symmetric matrix
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      expectedY[b*strideY + i] := beta * hostY[b*strideY + i];
      for j := 0 to N - 1 do
        expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
    end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_dsymv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dsymv_strided_batched_64(handle, uplo, N, @alpha, devA, N, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-12, Format('DSYMV_strided_batched_64 Y[%d]', [i]));


  hipFree(devA); hipFree(devX); hipFree(devY);
end;

//procedure TTestRocBLAS_Level2_strided_batched_64.Test_CSYMV_strided_batched_64;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZSYMV_strided_batched_64;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_SSBMV_strided_batched_64;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  LDA = K + 1; // Leading dimension for banded storage (K+1 for upper)
  alpha: Single = 1.5;
  beta:  Single = -0.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of Single;
  fullA: array of Single; // For reference calculation
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
begin
  strideA := LDA * N; // Number of rows in banded format (LDA) * number of columns (N)
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);
  SetLength(fullA, N * N * batch_count); // Full NxN matrix for easy reference calc

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full symmetric banded matrix for easy reference calculation
    // First, zero out the full matrix
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        fullA[b*N*N + j*N + i] := 0.0;

    // Then, fill only the banded part (including diagonal)
    // For upper triangular band storage with K super-diagonals:
    // Elements A[i,j] where j-K <= i <= j are non-zero.
    for j := 0 to N - 1 do // Column index
      for i := Max(0, j - K) to j do // Row index within band
      begin
        // Assign a value based on position and batch
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j + 2;
        // Ensure symmetry: A[i,j] = A[j,i]
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i];
      end;

    // 2. Pack the upper band into hostA format (LAPACK/rocBLAS banded storage)
    // Zero out the banded storage array first
    FillChar(hostA[b*strideA], strideA * SizeOf(Single), 0);
    // Fill the banded storage array
    // For rocblas_fill_upper and column-major storage:
    // Element A[i,j] is stored in hostA[b*strideA + j*LDA + (K + i - j)]
    // where (K + i - j) is the row index within the banded storage column j.
    for j := 0 to N-1 do // Iterate through columns
      for i := Max(0, j-K) to j do // Iterate through rows within the band for column j
      begin
        // Calculate the row index within the banded storage for column j
        // The formula is: row_in_band = K + (i - j)
        // Since i >= j-K, row_in_band >= 0.
        // Since i <= j, row_in_band <= K.
        hostA[b*strideA + j*LDA + (K + i - j)] := fullA[b*N*N + j*N + i];
      end;

    // 3. Initialize vectors X and Y
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
      hostY[b*strideY + i] := i * 2.0;
    end;
  end;

  // CPU reference calculation using the full symmetric banded matrix
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do // For each element of the result vector Y
    begin
      // Start with beta * Y[i]
      expectedY[b*strideY + i] := beta * hostY[b*strideY + i];
      // Add alpha * (sum over j of A[i,j] * X[j])
      for j := 0 to N - 1 do // Iterate through all columns
      begin
        // A[i,j] is stored in fullA[b*N*N + j*N + i]
        expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
      end;
    end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ssbmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ssbmv_strided_batched_64(handle, uplo, N, K, @alpha, devA, LDA, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-5, Format('SSBMV_strided_batched_64 Y[%d]', [i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DSBMV_strided_batched_64;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  LDA = K + 1; // Leading dimension for banded storage (K+1 for upper)
  alpha: Double = 1.5;
  beta:  Double = -0.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of Double;
  fullA: array of Double; // For reference calculation
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
begin
  strideA := LDA * N; // Number of rows in banded format (LDA) * number of columns (N)
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);
  SetLength(fullA, N * N * batch_count); // Full NxN matrix for easy reference calc

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full symmetric banded matrix for easy reference calculation
    // First, zero out the full matrix
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        fullA[b*N*N + j*N + i] := 0.0;

    // Then, fill only the banded part (including diagonal)
    // For upper triangular band storage with K super-diagonals:
    // Elements A[i,j] where j-K <= i <= j are non-zero.
    for j := 0 to N - 1 do // Column index
      for i := Max(0, j - K) to j do // Row index within band
      begin
        // Assign a value based on position and batch
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j + 2;
        // Ensure symmetry: A[i,j] = A[j,i]
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i];
      end;

    // 2. Pack the upper band into hostA format (LAPACK/rocBLAS banded storage)
    // Zero out the banded storage array first
    FillChar(hostA[b*strideA], strideA * SizeOf(Double), 0);
    // Fill the banded storage array
    // For rocblas_fill_upper and column-major storage:
    // Element A[i,j] is stored in hostA[b*strideA + j*LDA + (K + i - j)]
    for j := 0 to N-1 do // Iterate through columns
      for i := Max(0, j-K) to j do // Iterate through rows within the band for column j
      begin
        hostA[b*strideA + j*LDA + (K + i - j)] := fullA[b*N*N + j*N + i];
      end;

    // 3. Initialize vectors X and Y
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
      hostY[b*strideY + i] := i * 2.0;
    end;
  end;

  // CPU reference calculation using the full symmetric banded matrix
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do // For each element of the result vector Y
    begin
      // Start with beta * Y[i]
      expectedY[b*strideY + i] := beta * hostY[b*strideY + i];
      // Add alpha * (sum over j of A[i,j] * X[j])
      for j := 0 to N - 1 do // Iterate through all columns
      begin
        // A[i,j] is stored in fullA[b*N*N + j*N + i]
        expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
      end;
    end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dsbmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dsbmv_strided_batched_64(handle, uplo, N, K, @alpha, devA, LDA, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-12, Format('DSBMV_strided_batched_64 Y[%d]', [i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_SSPMV_strided_batched_64;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Single = 1.5;
  beta: Single = 2.5;
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostX, hostY, hostY_result, expectedY: array of Single;
  fullA: array of Single; // For CPU reference calculation
  packed_size, strideAP, strideX, strideY: Int64;
  i, j, b, k: Integer;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;
  strideY := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full symmetric matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j;
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i]; // A(j,i) = A(i,j)
      end;

    // 2. Pack the referenced part (upper) of the full matrix into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vectors
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
      hostY[b*strideY + i] := i * 0.5;
    end;
  end;

  // CPU reference calculation using the full symmetric matrix
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      expectedY[b*strideY + i] := beta * hostY[b*strideY + i];
      for j := 0 to N - 1 do
        expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
    end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_sspmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_sspmv_strided_batched_64(handle, uplo, N, @alpha, devAP, strideAP, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-5, Format('SSPMV_strided_batched_64 Y[%d]', [i]));


  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DSPMV_strided_batched_64;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Double = 1.5;
  beta: Double = 2.5;
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostX, hostY, hostY_result, expectedY: array of Double;
  fullA: array of Double; // For CPU reference calculation
  packed_size, strideAP, strideX, strideY: Int64;
  i, j, b, k: Integer;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;
  strideY := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full symmetric matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j;
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i]; // A(j,i) = A(i,j)
      end;

    // 2. Pack the referenced part (upper) of the full matrix into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vectors
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
      hostY[b*strideY + i] := i * 0.5;
    end;
  end;

  // CPU reference calculation using the full symmetric matrix
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      expectedY[b*strideY + i] := beta * hostY[b*strideY + i];
      for j := 0 to N - 1 do
        expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
    end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_dspmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dspmv_strided_batched_64(handle, uplo, N, @alpha, devAP, strideAP, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-12, Format('DSPMV_strided_batched_64 Y[%d]', [i]));


  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_STRMV_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Single = 1.5; // Used only for initializing matrix A for better numerical values
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostX_result, expectedX: array of Single;
  strideA, strideX: Int64;
  i, j, b: Integer;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize triangular matrix A
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          // Use alpha to scale values for better numerics
          hostA[b * strideA + j * N + i] := alpha * ((b + 1) * 10 + i + j + 1);
          // Make diagonal elements slightly larger for stability
          if i = j then
            hostA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i] + 10;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          hostA[b * strideA + j * N + i] := 0.0;
        end;
      end;
    end;

    // 2. Initialize vector X (input)
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := i + 1.0;
    end;
  end;

  // 3. CPU reference calculation: X := A * X (or A^T * X)
  // This performs matrix-vector multiplication for each batch
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do // Row of result vector
    begin
      expectedX[b * strideX + i] := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // X[i] += A[i,j] * X[j]
        expectedX[b * strideX + i] := expectedX[b * strideX + i] + hostA[b * strideA + j * N + i] * hostX[b * strideX + j];
      end;
    end;
  end;
  // Note: This simple loop handles transA=none.
  // For transA=transpose, the loop structure for accessing A would change to A[j*N+i].

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX H2D', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_strmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_strmv_strided_batched_64(handle, uplo, transA, diag, N,
      devA, N, strideA,
      devX, 1, strideX,
      batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devX D2H', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-5, Format('STRMV_strided_batched_64 X[%d]', [i]));


  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DTRMV_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Double = 1.5; // Used only for initializing matrix A for better numerical values
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostX_result, expectedX: array of Double;
  strideA, strideX: Int64;
  i, j, b: Integer;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize triangular matrix A
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          hostA[b * strideA + j * N + i] := alpha * ((b + 1) * 10 + i + j + 1);
          // Make diagonal elements slightly larger for stability
          if i = j then
            hostA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i] + 10;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          hostA[b * strideA + j * N + i] := 0.0;
        end;
      end;
    end;

    // 2. Initialize vector X (input)
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := i + 1.0;
    end;
  end;

  // 3. CPU reference calculation: X := A * X (or A^T * X)
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do // Row of result vector
    begin
      expectedX[b * strideX + i] := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // X[i] += A[i,j] * X[j]
        expectedX[b * strideX + i] := expectedX[b * strideX + i] + hostA[b * strideA + j * N + i] * hostX[b * strideX + j];
      end;
    end;
  end;
  // Note: This simple loop handles transA=none.
  // For transA=transpose, the loop structure for accessing A would change to A[j*N+i].

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX H2D', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_dtrmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dtrmv_strided_batched_64(handle, uplo, transA, diag, N,
      devA, N, strideA,
      devX, 1, strideX,
      batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devX D2H', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTRMV_strided_batched_64 X[%d]', [i]));


  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CTRMV_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Single = 1.5; // Scaling factor for matrix elements
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostX_result, expectedX: array of rocblas_float_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
  temp_val: rocblas_float_complex;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize triangular matrix A
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          hostA[b * strideA + j * N + i].x := alpha * ((b + 1) * 10 + i + j + 1);
          hostA[b * strideA + j * N + i].y := alpha * ((b + 1) * 0.5 + Abs(i - j) * 0.1);
          // Make diagonal elements real and larger for stability
          if i = j then
          begin
            hostA[b * strideA + j * N + i].y := 0.0;
            hostA[b * strideA + j * N + i].x := hostA[b * strideA + j * N + i].x + 10;
          end;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          hostA[b * strideA + j * N + i].x := 0.0;
          hostA[b * strideA + j * N + i].y := 0.0;
        end;
      end;
    end;

    // 2. Initialize vector X (input)
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := i + 1.0;
      hostX[b * strideX + i].y := (i + 1) * 0.5;
    end;
  end;

  // 3. CPU reference calculation: X := A * X (or A^T * X or A^H * X)
  // This performs complex matrix-vector multiplication for each batch
  // Note: This implementation assumes transA = rocblas_operation_none.
  // For other transA values, the access pattern for A would change.
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do // Row of result vector
    begin
      expectedX[b * strideX + i].x := 0.0;
      expectedX[b * strideX + i].y := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // temp_val = A[i,j] * X[j]
        // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        temp_val.x := hostA[b * strideA + j * N + i].x * hostX[b * strideX + j].x -
                      hostA[b * strideA + j * N + i].y * hostX[b * strideX + j].y;
        temp_val.y := hostA[b * strideA + j * N + i].x * hostX[b * strideX + j].y +
                      hostA[b * strideA + j * N + i].y * hostX[b * strideX + j].x;
        // expectedX[i] += temp_val
        expectedX[b * strideX + i].x := expectedX[b * strideX + i].x + temp_val.x;
        expectedX[b * strideX + i].y := expectedX[b * strideX + i].y + temp_val.y;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX H2D', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_ctrmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ctrmv_strided_batched_64(handle, uplo, transA, diag, N,
      devA, N, strideA,
      devX, 1, strideX,
      batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devX D2H', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-4, Format('CTRMV_strided_batched_64 X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-4, Format('CTRMV_strided_batched_64 X[%d].y', [i]));
  end;


  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZTRMV_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Double = 1.5; // Scaling factor for matrix elements
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostX_result, expectedX: array of rocblas_double_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
  temp_val: rocblas_double_complex;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize triangular matrix A
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          hostA[b * strideA + j * N + i].x := alpha * ((b + 1) * 10 + i + j + 1);
          hostA[b * strideA + j * N + i].y := alpha * ((b + 1) * 0.5 + Abs(i - j) * 0.1);
          // Make diagonal elements real and larger for stability
          if i = j then
          begin
            hostA[b * strideA + j * N + i].y := 0.0;
            hostA[b * strideA + j * N + i].x := hostA[b * strideA + j * N + i].x + 10;
          end;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          hostA[b * strideA + j * N + i].x := 0.0;
          hostA[b * strideA + j * N + i].y := 0.0;
        end;
      end;
    end;

    // 2. Initialize vector X (input)
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := i + 1.0;
      hostX[b * strideX + i].y := (i + 1) * 0.5;
    end;
  end;

  // 3. CPU reference calculation: X := A * X (or A^T * X or A^H * X)
  // Note: This implementation assumes transA = rocblas_operation_none.
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do // Row of result vector
    begin
      expectedX[b * strideX + i].x := 0.0;
      expectedX[b * strideX + i].y := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // temp_val = A[i,j] * X[j]
        // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        temp_val.x := hostA[b * strideA + j * N + i].x * hostX[b * strideX + j].x -
                      hostA[b * strideA + j * N + i].y * hostX[b * strideX + j].y;
        temp_val.y := hostA[b * strideA + j * N + i].x * hostX[b * strideX + j].y +
                      hostA[b * strideA + j * N + i].y * hostX[b * strideX + j].x;
        // expectedX[i] += temp_val
        expectedX[b * strideX + i].x := expectedX[b * strideX + i].x + temp_val.x;
        expectedX[b * strideX + i].y := expectedX[b * strideX + i].y + temp_val.y;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX H2D', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_ztrmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ztrmv_strided_batched_64(handle, uplo, transA, diag, N,
      devA, N, strideA,
      devX, 1, strideX,
      batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devX D2H', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTRMV_strided_batched_64 X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTRMV_strided_batched_64 X[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
end;

//procedure TTestRocBLAS_Level2_strided_batched_64.Test_STBMV_strided_batched_64;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_strided_batched_64.Test_DTBMV_strided_batched_64;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_strided_batched_64.Test_CTBMV_strided_batched_64;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZTBMV_strided_batched_64;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_STPMV_strided_batched_64;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Single = 1.5; // Scaling factor for matrix elements
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostX, hostX_result, expectedX: array of Single;
  fullA: array of Single; // For CPU reference calculation
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full triangular matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          fullA[b*N*N + j*N + i] := alpha * ((b+1)*10 + i + j + 1);
          // Make diagonal elements slightly larger for stability
          if i = j then
            fullA[b*N*N + j*N + i] := fullA[b*N*N + j*N + i] + 5;
          // Handle unit diagonal: if diag = rocblas_diagonal_unit, set diagonal to 1
          if diag = rocblas_diagonal_unit then
             if i = j then fullA[b*N*N + j*N + i] := 1.0;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          fullA[b*N*N + j*N + i] := 0.0;
        end;
      end;

    // 2. Pack the referenced part (upper or lower) of the full matrix into hostAP (column-major)
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;

    // 3. Initialize vector X (input)
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
    end;
  end;

  // CPU reference calculation using the full triangular matrix
  // Compute X := A * X (or A^T * X depending on transA)
  // This implementation assumes transA = rocblas_operation_none.
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do // For each element of result vector X
    begin
      expectedX[b*strideX + i] := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // X[i] += A[i,j] * X[j]
        expectedX[b*strideX + i] := expectedX[b*strideX + i] + fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
      end;
    end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_stpmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_stpmv_strided_batched_64(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-5, Format('STPMV_strided_batched_64 X[%d]', [i]));


  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DTPMV_strided_batched_64;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Double = 1.5; // Scaling factor for matrix elements
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostX, hostX_result, expectedX: array of Double;
  fullA: array of Double; // For CPU reference calculation
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full triangular matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          fullA[b*N*N + j*N + i] := alpha * ((b+1)*10 + i + j + 1);
          // Make diagonal elements slightly larger for stability
          if i = j then
            fullA[b*N*N + j*N + i] := fullA[b*N*N + j*N + i] + 5;
          // Handle unit diagonal
          if diag = rocblas_diagonal_unit then
             if i = j then fullA[b*N*N + j*N + i] := 1.0;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          fullA[b*N*N + j*N + i] := 0.0;
        end;
      end;

    // 2. Pack the referenced part (upper or lower) of the full matrix into hostAP (column-major)
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;

    // 3. Initialize vector X (input)
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
    end;
  end;

  // CPU reference calculation using the full triangular matrix
  // Compute X := A * X (or A^T * X depending on transA)
  // This implementation assumes transA = rocblas_operation_none.
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do // For each element of result vector X
    begin
      expectedX[b*strideX + i] := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // X[i] += A[i,j] * X[j]
        expectedX[b*strideX + i] := expectedX[b*strideX + i] + fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
      end;
    end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_dtpmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dtpmv_strided_batched_64(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTPMV_strided_batched_64 X[%d]', [i]));


  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CTPMV_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Single = 1.5; // Scaling factor for matrix elements
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostX, hostX_result, expectedX: array of rocblas_float_complex;
  fullA: array of rocblas_float_complex; // For CPU reference calculation
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
  temp_val: rocblas_float_complex;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full triangular matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          fullA[b*N*N + j*N + i].x := alpha * ((b+1)*10 + i + j + 1);
          fullA[b*N*N + j*N + i].y := alpha * ((b+1)*0.5 + Abs(i-j)*0.1);
          // Make diagonal elements larger for stability and optionally real if needed
          if i = j then
          begin
            fullA[b*N*N + j*N + i].x := fullA[b*N*N + j*N + i].x + 5;
            // Handle unit diagonal
            if diag = rocblas_diagonal_unit then
            begin
              fullA[b*N*N + j*N + i].x := 1.0;
              fullA[b*N*N + j*N + i].y := 0.0;
            end;
          end;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          fullA[b*N*N + j*N + i].x := 0.0;
          fullA[b*N*N + j*N + i].y := 0.0;
        end;
      end;

    // 2. Pack the referenced part (upper or lower) of the full matrix into hostAP (column-major)
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;

    // 3. Initialize vector X (input)
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0;
      hostX[b*strideX + i].y := (i + 1) * 0.5;
    end;
  end;

  // CPU reference calculation using the full triangular matrix
  // Compute X := A * X (or A^T * X or A^H * X depending on transA)
  // This implementation assumes transA = rocblas_operation_none.
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do // For each element of result vector X
    begin
      expectedX[b*strideX + i].x := 0.0;
      expectedX[b*strideX + i].y := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // temp_val = A[i,j] * X[j]
        // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        temp_val.x := fullA[b*N*N + j*N + i].x * hostX[b*strideX + j].x -
                      fullA[b*N*N + j*N + i].y * hostX[b*strideX + j].y;
        temp_val.y := fullA[b*N*N + j*N + i].x * hostX[b*strideX + j].y +
                      fullA[b*N*N + j*N + i].y * hostX[b*strideX + j].x;
        // expectedX[i] += temp_val
        expectedX[b*strideX + i].x := expectedX[b*strideX + i].x + temp_val.x;
        expectedX[b*strideX + i].y := expectedX[b*strideX + i].y + temp_val.y;
      end;
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_ctpmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ctpmv_strided_batched_64(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-4, Format('CTPMV_strided_batched_64 X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-4, Format('CTPMV_strided_batched_64 X[%d].y', [i]));
  end;


  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZTPMV_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Double = 1.5; // Scaling factor for matrix elements
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostX, hostX_result, expectedX: array of rocblas_double_complex;
  fullA: array of rocblas_double_complex; // For CPU reference calculation
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
  temp_val: rocblas_double_complex;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full triangular matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          fullA[b*N*N + j*N + i].x := alpha * ((b+1)*10 + i + j + 1);
          fullA[b*N*N + j*N + i].y := alpha * ((b+1)*0.5 + Abs(i-j)*0.1);
          // Make diagonal elements larger for stability and optionally real if needed
          if i = j then
          begin
            fullA[b*N*N + j*N + i].x := fullA[b*N*N + j*N + i].x + 5;
            // Handle unit diagonal
            if diag = rocblas_diagonal_unit then
            begin
              fullA[b*N*N + j*N + i].x := 1.0;
              fullA[b*N*N + j*N + i].y := 0.0;
            end;
          end;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          fullA[b*N*N + j*N + i].x := 0.0;
          fullA[b*N*N + j*N + i].y := 0.0;
        end;
      end;

    // 2. Pack the referenced part (upper or lower) of the full matrix into hostAP (column-major)
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;

    // 3. Initialize vector X (input)
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0;
      hostX[b*strideX + i].y := (i + 1) * 0.5;
    end;
  end;

  // CPU reference calculation using the full triangular matrix
  // Compute X := A * X (or A^T * X or A^H * X depending on transA)
  // This implementation assumes transA = rocblas_operation_none.
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do // For each element of result vector X
    begin
      expectedX[b*strideX + i].x := 0.0;
      expectedX[b*strideX + i].y := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // temp_val = A[i,j] * X[j]
        // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        temp_val.x := fullA[b*N*N + j*N + i].x * hostX[b*strideX + j].x -
                      fullA[b*N*N + j*N + i].y * hostX[b*strideX + j].y;
        temp_val.y := fullA[b*N*N + j*N + i].x * hostX[b*strideX + j].y +
                      fullA[b*N*N + j*N + i].y * hostX[b*strideX + j].x;
        // expectedX[i] += temp_val
        expectedX[b*strideX + i].x := expectedX[b*strideX + i].x + temp_val.x;
        expectedX[b*strideX + i].y := expectedX[b*strideX + i].y + temp_val.y;
      end;
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_ztpmv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ztpmv_strided_batched_64(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTPMV_strided_batched_64 X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTPMV_strided_batched_64 X[%d].y', [i]));
  end;


  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_STRSV_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Single = 2.0; // Scaling factor for matrix elements
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX, hostX_result, expectedX: array of Single;
  strideA, strideX: Int64;
  i, j, b, k: Integer;
  // For simple reference solution, we'll use a basic forward/backward substitution
  // or solve a small system manually.
  // A simple approach is to create a diagonally dominant triangular matrix
  // and compute B = A * known_X, then solve A * X = B and check X against known_X.
  knownX: array of Single; // The true solution we will use to generate B
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count); // B vector for RHS
  SetLength(hostX, strideX * batch_count);  // X vector for initial guess / solution
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(knownX, strideX * batch_count); // True solution

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize triangular matrix A (make it diagonally dominant for stability)
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          hostA[b * strideA + j * N + i] := alpha * ((b + 1) * 10 + i + j + 1);
          // Make diagonal elements significantly larger for diagonal dominance and stability
          if i = j then
            hostA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i] + N * 10 * alpha;
          // Handle unit diagonal: if diag = rocblas_diagonal_unit, set diagonal to 1 in the matrix used for computation
          // Note: rocBLAS handles unit diagonal internally, so we store 1s but rocBLAS assumes diag=1.
          if diag = rocblas_diagonal_unit then
             if i = j then hostA[b * strideA + j * N + i] := 1.0;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          hostA[b * strideA + j * N + i] := 0.0;
        end;
      end;
    end;

    // 2. Initialize the true solution vector (knownX)
    for i := 0 to N - 1 do
    begin
      knownX[b * strideX + i] := (b + 1) * 3 + i + 1; // Simple pattern
    end;

    // 3. Compute RHS vector B = A * knownX for each batch
    // This is used as input to the solver
    for i := 0 to N - 1 do // Row of result vector B
    begin
      hostB[b * strideX + i] := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // B[i] += A[i,j] * knownX[j]
        hostB[b * strideX + i] := hostB[b * strideX + i] + hostA[b * strideA + j * N + i] * knownX[b * strideX + j];
      end;
    end;

    // 4. Initialize vector X (initial guess for the solver) - can be anything, often zero or B
    // For testing solver, we can initialize X to B (right-hand side) or zero.
    // The solver should overwrite X with the solution.
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := hostB[b * strideX + i]; // Initialize with RHS
      // hostX[b * strideX + i] := 0.0; // Or initialize with zero
    end;

    // 5. Set expectedX to knownX for comparison
    for i := 0 to N - 1 do
    begin
      expectedX[b * strideX + i] := knownX[b * strideX + i];
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX (B) H2D', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice)); // Copy B to X initially

  // Call rocBLAS function: Solve A * X = B, where B is in X on input, and solution X is in X on output.
  AssertEquals('rocblas_strsv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_strsv_strided_batched_64(handle, uplo, transA, diag, N,
      devA, N, strideA,
      devX, 1, strideX,
      batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devX D2H', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verify results: hostX_result should be close to expectedX (which is knownX)
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-3, Format('STRSV_strided_batched_64 X[%d]', [i])); // Relaxed tolerance for single precision


  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DTRSV_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Double = 2.0; // Scaling factor for matrix elements
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX, hostX_result, expectedX: array of Double;
  strideA, strideX: Int64;
  i, j, b: Integer;
  knownX: array of Double; // The true solution we will use to generate B
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count); // B vector for RHS
  SetLength(hostX, strideX * batch_count);  // X vector for initial guess / solution
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(knownX, strideX * batch_count); // True solution

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize triangular matrix A (make it diagonally dominant for stability)
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          hostA[b * strideA + j * N + i] := alpha * ((b + 1) * 10 + i + j + 1);
          // Make diagonal elements significantly larger for diagonal dominance and stability
          if i = j then
            hostA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i] + N * 10 * alpha;
          // Handle unit diagonal
          if diag = rocblas_diagonal_unit then
             if i = j then hostA[b * strideA + j * N + i] := 1.0;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          hostA[b * strideA + j * N + i] := 0.0;
        end;
      end;
    end;

    // 2. Initialize the true solution vector (knownX)
    for i := 0 to N - 1 do
    begin
      knownX[b * strideX + i] := (b + 1) * 3 + i + 1; // Simple pattern
    end;

    // 3. Compute RHS vector B = A * knownX for each batch
    for i := 0 to N - 1 do // Row of result vector B
    begin
      hostB[b * strideX + i] := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // B[i] += A[i,j] * knownX[j]
        hostB[b * strideX + i] := hostB[b * strideX + i] + hostA[b * strideA + j * N + i] * knownX[b * strideX + j];
      end;
    end;

    // 4. Initialize vector X (initial guess for the solver)
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := hostB[b * strideX + i]; // Initialize with RHS
    end;

    // 5. Set expectedX to knownX for comparison
    for i := 0 to N - 1 do
    begin
      expectedX[b * strideX + i] := knownX[b * strideX + i];
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX (B) H2D', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice)); // Copy B to X initially

  // Call rocBLAS function: Solve A * X = B
  AssertEquals('rocblas_dtrsv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dtrsv_strided_batched_64(handle, uplo, transA, diag, N,
      devA, N, strideA,
      devX, 1, strideX,
      batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devX D2H', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTRSV_strided_batched_64 X[%d]', [i]));


  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CTRSV_strided_batched_64;
const
  N = 3;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Single = 2.0; // Scaling factor for matrix elements
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX, hostX_result, expectedX: array of rocblas_float_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
  knownX: array of rocblas_float_complex; // The true solution we will use to generate B
  temp_val: rocblas_float_complex;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count); // B vector for RHS
  SetLength(hostX, strideX * batch_count);  // X vector for initial guess / solution
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(knownX, strideX * batch_count); // True solution

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize triangular matrix A (complex, make it diagonally dominant)
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          hostA[b * strideA + j * N + i].x := alpha * ((b + 1) * 10 + i + j + 1);
          hostA[b * strideA + j * N + i].y := alpha * ((b + 1) * 0.5 + Abs(i-j) * 0.1);
          // Make diagonal elements significantly larger for diagonal dominance and stability
          if i = j then
          begin
            hostA[b * strideA + j * N + i].x := hostA[b * strideA + j * N + i].x + N * 10 * alpha;
            // Optionally make diagonal real if needed for better conditioning
            // hostA[b * strideA + j * N + i].y := 0.0;
          end;
          // Handle unit diagonal
          if diag = rocblas_diagonal_unit then
          begin
             if i = j then
             begin
               hostA[b * strideA + j * N + i].x := 1.0;
               hostA[b * strideA + j * N + i].y := 0.0;
             end;
          end;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          hostA[b * strideA + j * N + i].x := 0.0;
          hostA[b * strideA + j * N + i].y := 0.0;
        end;
      end;
    end;

    // 2. Initialize the true solution vector (knownX)
    for i := 0 to N - 1 do
    begin
      knownX[b * strideX + i].x := (b + 1) * 3 + i + 1; // Real part
      knownX[b * strideX + i].y := (b + 1) * 0.5 + i * 0.2; // Imaginary part
    end;

    // 3. Compute RHS vector B = A * knownX for each batch (complex matrix-vector multiply)
    for i := 0 to N - 1 do // Row of result vector B
    begin
      hostB[b * strideX + i].x := 0.0;
      hostB[b * strideX + i].y := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // temp_val = A[i,j] * knownX[j]
        // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        temp_val.x := hostA[b * strideA + j * N + i].x * knownX[b * strideX + j].x -
                      hostA[b * strideA + j * N + i].y * knownX[b * strideX + j].y;
        temp_val.y := hostA[b * strideA + j * N + i].x * knownX[b * strideX + j].y +
                      hostA[b * strideA + j * N + i].y * knownX[b * strideX + j].x;
        // B[i] += temp_val
        hostB[b * strideX + i].x := hostB[b * strideX + i].x + temp_val.x;
        hostB[b * strideX + i].y := hostB[b * strideX + i].y + temp_val.y;
      end;
    end;

    // 4. Initialize vector X (initial guess for the solver)
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := hostB[b * strideX + i]; // Initialize with RHS
    end;

    // 5. Set expectedX to knownX for comparison
    for i := 0 to N - 1 do
    begin
      expectedX[b * strideX + i] := knownX[b * strideX + i];
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX (B) H2D', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice)); // Copy B to X initially

  // Call rocBLAS function: Solve A * X = B
  AssertEquals('rocblas_ctrsv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ctrsv_strided_batched_64(handle, uplo, transA, diag, N,
      devA, N, strideA,
      devX, 1, strideX,
      batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devX D2H', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-3, Format('CTRSV_strided_batched_64 X[%d].x', [i])); // Relaxed tolerance for single precision complex
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-3, Format('CTRSV_strided_batched_64 X[%d].y', [i]));
  end;


  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZTRSV_strided_batched_64;
const
  N = 3;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Double = 2.0; // Scaling factor for matrix elements
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX, hostX_result, expectedX: array of rocblas_double_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
  knownX: array of rocblas_double_complex; // The true solution we will use to generate B
  temp_val: rocblas_double_complex;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count); // B vector for RHS
  SetLength(hostX, strideX * batch_count);  // X vector for initial guess / solution
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(knownX, strideX * batch_count); // True solution

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize triangular matrix A (complex, make it diagonally dominant)
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          hostA[b * strideA + j * N + i].x := alpha * ((b + 1) * 10 + i + j + 1);
          hostA[b * strideA + j * N + i].y := alpha * ((b + 1) * 0.5 + Abs(i-j) * 0.1);
          // Make diagonal elements significantly larger for diagonal dominance and stability
          if i = j then
          begin
            hostA[b * strideA + j * N + i].x := hostA[b * strideA + j * N + i].x + N * 10 * alpha;
          end;
          // Handle unit diagonal
          if diag = rocblas_diagonal_unit then
          begin
             if i = j then
             begin
               hostA[b * strideA + j * N + i].x := 1.0;
               hostA[b * strideA + j * N + i].y := 0.0;
             end;
          end;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          hostA[b * strideA + j * N + i].x := 0.0;
          hostA[b * strideA + j * N + i].y := 0.0;
        end;
      end;
    end;

    // 2. Initialize the true solution vector (knownX)
    for i := 0 to N - 1 do
    begin
      knownX[b * strideX + i].x := (b + 1) * 3 + i + 1; // Real part
      knownX[b * strideX + i].y := (b + 1) * 0.5 + i * 0.2; // Imaginary part
    end;

    // 3. Compute RHS vector B = A * knownX for each batch (complex matrix-vector multiply)
    for i := 0 to N - 1 do // Row of result vector B
    begin
      hostB[b * strideX + i].x := 0.0;
      hostB[b * strideX + i].y := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // temp_val = A[i,j] * knownX[j]
        temp_val.x := hostA[b * strideA + j * N + i].x * knownX[b * strideX + j].x -
                      hostA[b * strideA + j * N + i].y * knownX[b * strideX + j].y;
        temp_val.y := hostA[b * strideA + j * N + i].x * knownX[b * strideX + j].y +
                      hostA[b * strideA + j * N + i].y * knownX[b * strideX + j].x;
        // B[i] += temp_val
        hostB[b * strideX + i].x := hostB[b * strideX + i].x + temp_val.x;
        hostB[b * strideX + i].y := hostB[b * strideX + i].y + temp_val.y;
      end;
    end;

    // 4. Initialize vector X (initial guess for the solver)
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := hostB[b * strideX + i]; // Initialize with RHS
    end;

    // 5. Set expectedX to knownX for comparison
    for i := 0 to N - 1 do
    begin
      expectedX[b * strideX + i] := knownX[b * strideX + i];
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX (B) H2D', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice)); // Copy B to X initially

  // Call rocBLAS function: Solve A * X = B
  AssertEquals('rocblas_ztrsv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ztrsv_strided_batched_64(handle, uplo, transA, diag, N,
      devA, N, strideA,
      devX, 1, strideX,
      batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devX D2H', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTRSV_strided_batched_64 X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTRSV_strided_batched_64 X[%d].y', [i]));
  end;


  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_STBSV_strided_batched_64;
const
  N = 5;
  K = 2; // Number of super-diagonals
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Single = 2.0; // Scaling factor for matrix elements
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX, hostX_result, expectedX: array of Single;
  strideA, strideX: Int64;
  i, j, b, k_band: Integer;
  lda: Int64; // Leading dimension for banded storage
  knownX: array of Single; // The true solution we will use to generate B
begin
  // For banded storage with K super-diagonals and uplo=upper,
  // the leading dimension is K+1 (number of rows in the banded storage format)
  lda := K + 1;
  strideA := N * lda; // Stride is number of columns * leading dimension
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count); // B vector for RHS
  SetLength(hostX, strideX * batch_count);  // X vector for initial guess / solution
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(knownX, strideX * batch_count); // True solution

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize banded triangular matrix A in the correct banded storage format
    // Make it diagonally dominant for stability
    for j := 0 to N - 1 do // Iterate through columns
    begin
      for k_band := 0 to lda - 1 do // Iterate through rows in banded storage column
      begin
        hostA[b * strideA + j * lda + k_band] := 0.0; // Zero out first
      end;
      // Fill relevant bands (diagonal and super-diagonals)
      for i := Max(0, j - K) to j do // For upper band, i ranges from j-K (or 0) to j
      begin
        k_band := K + i - j; // Row index in banded storage
        if (k_band >= 0) and (k_band < lda) then
        begin
          hostA[b * strideA + j * lda + k_band] := alpha * ((b + 1) * 10 + (i + 1) + (j + 1) * 0.1);
          // Make diagonal elements significantly larger for diagonal dominance
          if i = j then
            hostA[b * strideA + j * lda + k_band] := hostA[b * strideA + j * lda + k_band] + N * 5 * alpha;
          // Handle unit diagonal
          if diag = rocblas_diagonal_unit then
            if i = j then hostA[b * strideA + j * lda + k_band] := 1.0;
        end;
      end;
    end;

    // 2. Initialize the true solution vector (knownX)
    for i := 0 to N - 1 do
    begin
      knownX[b * strideX + i] := (b + 1) * 3 + i + 1; // Simple pattern
    end;

    // 3. Compute RHS vector B = A * knownX for each batch
    // Compute expected result: B = A * knownX (assuming transA = none)
    for i := 0 to N - 1 do // For each row of result vector B
    begin
      hostB[b * strideX + i] := 0.0;
      for j := 0 to N - 1 do // For dot product A_row * knownX
      begin
        // Check if A[i, j] is within the band (j - K <= i <= j for upper band)
        if ((i >= j - K) and (i <= j)) then
        begin
          k_band := K + i - j; // Row index in banded storage
          if (k_band >= 0) and (k_band < lda) then
          begin
            hostB[b * strideX + i] := hostB[b * strideX + i] + hostA[b * strideA + j * lda + k_band] * knownX[b * strideX + j];
          end;
        end;
      end;
    end;

    // 4. Initialize vector X (initial guess for the solver)
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := hostB[b * strideX + i]; // Initialize with RHS
    end;

    // 5. Set expectedX to knownX for comparison
    for i := 0 to N - 1 do
    begin
      expectedX[b * strideX + i] := knownX[b * strideX + i];
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX (B) H2D', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice)); // Copy B to X initially

  // Call rocBLAS function: Solve A * X = B
  AssertEquals('rocblas_stbsv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_stbsv_strided_batched_64(handle, uplo, transA, diag, N, K,
      devA, lda, strideA,
      devX, 1, strideX,
      batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devX D2H', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-3, Format('STBSV_strided_batched_64 X[%d]', [i])); // Relaxed tolerance


  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DTBSV_strided_batched_64;
const
  N = 5;
  K = 2; // Number of super-diagonals
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Double = 2.0; // Scaling factor for matrix elements
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX, hostX_result, expectedX: array of Double;
  strideA, strideX: Int64;
  i, j, b, k_band: Integer;
  lda: Int64; // Leading dimension for banded storage
  knownX: array of Double; // The true solution we will use to generate B
begin
  // For banded storage with K super-diagonals and uplo=upper,
  // the leading dimension is K+1
  lda := K + 1;
  strideA := N * lda;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count); // B vector for RHS
  SetLength(hostX, strideX * batch_count);  // X vector for initial guess / solution
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(knownX, strideX * batch_count); // True solution

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize banded triangular matrix A
    for j := 0 to N - 1 do // Iterate through columns
    begin
      for k_band := 0 to lda - 1 do
      begin
        hostA[b * strideA + j * lda + k_band] := 0.0;
      end;
      // Fill relevant bands (diagonal and super-diagonals)
      for i := Max(0, j - K) to j do
      begin
        k_band := K + i - j;
        if (k_band >= 0) and (k_band < lda) then
        begin
          hostA[b * strideA + j * lda + k_band] := alpha * ((b + 1) * 10 + (i + 1) + (j + 1) * 0.1);
          // Make diagonal elements significantly larger for diagonal dominance
          if i = j then
            hostA[b * strideA + j * lda + k_band] := hostA[b * strideA + j * lda + k_band] + N * 5 * alpha;
          // Handle unit diagonal
          if diag = rocblas_diagonal_unit then
            if i = j then hostA[b * strideA + j * lda + k_band] := 1.0;
        end;
      end;
    end;

    // 2. Initialize the true solution vector (knownX)
    for i := 0 to N - 1 do
    begin
      knownX[b * strideX + i] := (b + 1) * 3 + i + 1;
    end;

    // 3. Compute RHS vector B = A * knownX for each batch
    for i := 0 to N - 1 do
    begin
      hostB[b * strideX + i] := 0.0;
      for j := 0 to N - 1 do
      begin
        // Check if A[i, j] is within the band (j - K <= i <= j for upper band)
        if ((i >= j - K) and (i <= j)) then
        begin
          k_band := K + i - j;
          if (k_band >= 0) and (k_band < lda) then
          begin
            hostB[b * strideX + i] := hostB[b * strideX + i] + hostA[b * strideA + j * lda + k_band] * knownX[b * strideX + j];
          end;
        end;
      end;
    end;

    // 4. Initialize vector X (initial guess for the solver)
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := hostB[b * strideX + i]; // Initialize with RHS
    end;

    // 5. Set expectedX to knownX for comparison
    for i := 0 to N - 1 do
    begin
      expectedX[b * strideX + i] := knownX[b * strideX + i];
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX (B) H2D', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  // Call rocBLAS function: Solve A * X = B
  AssertEquals('rocblas_dtbsv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dtbsv_strided_batched_64(handle, uplo, transA, diag, N, K,
      devA, lda, strideA,
      devX, 1, strideX,
      batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devX D2H', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTBSV_strided_batched_64 X[%d]', [i]));


  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CTBSV_strided_batched_64;
const
  N = 4;
  K = 1; // Number of super-diagonals
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Single = 2.0; // Scaling factor for matrix elements
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX, hostX_result, expectedX: array of rocblas_float_complex;
  strideA, strideX: Int64;
  i, j, b, k_band: Integer;
  lda: Int64; // Leading dimension for banded storage
  knownX: array of rocblas_float_complex; // The true solution
  temp_val: rocblas_float_complex;
begin
  lda := K + 1;
  strideA := N * lda;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count); // B vector for RHS
  SetLength(hostX, strideX * batch_count);  // X vector for initial guess / solution
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(knownX, strideX * batch_count); // True solution

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize banded triangular matrix A (complex)
    for j := 0 to N - 1 do // Iterate through columns
    begin
      for k_band := 0 to lda - 1 do
      begin
        hostA[b * strideA + j * lda + k_band].x := 0.0;
        hostA[b * strideA + j * lda + k_band].y := 0.0;
      end;
      // Fill relevant bands
      for i := Max(0, j - K) to j do
      begin
        k_band := K + i - j;
        if (k_band >= 0) and (k_band < lda) then
        begin
          hostA[b * strideA + j * lda + k_band].x := alpha * ((b + 1) * 10 + (i + 1) + (j + 1) * 0.1);
          hostA[b * strideA + j * lda + k_band].y := alpha * ((b + 1) * 0.5 + Abs(i-j) * 0.1);
          // Make diagonal elements significantly larger for diagonal dominance
          if i = j then
          begin
            hostA[b * strideA + j * lda + k_band].x := hostA[b * strideA + j * lda + k_band].x + N * 5 * alpha;
            // Optionally make diagonal real
            // hostA[b * strideA + j * lda + k_band].y := 0.0;
          end;
          // Handle unit diagonal
          if diag = rocblas_diagonal_unit then
          begin
            if i = j then
            begin
              hostA[b * strideA + j * lda + k_band].x := 1.0;
              hostA[b * strideA + j * lda + k_band].y := 0.0;
            end;
          end;
        end;
      end;
    end;

    // 2. Initialize the true solution vector (knownX)
    for i := 0 to N - 1 do
    begin
      knownX[b * strideX + i].x := (b + 1) * 3 + i + 1; // Real part
      knownX[b * strideX + i].y := (b + 1) * 0.5 + i * 0.2; // Imaginary part
    end;

    // 3. Compute RHS vector B = A * knownX for each batch (complex banded mat-vec)
    for i := 0 to N - 1 do
    begin
      hostB[b * strideX + i].x := 0.0;
      hostB[b * strideX + i].y := 0.0;
      for j := 0 to N - 1 do
      begin
        // Check if A[i, j] is within the band (j - K <= i <= j for upper band)
        if ((i >= j - K) and (i <= j)) then
        begin
          k_band := K + i - j;
          if (k_band >= 0) and (k_band < lda) then
          begin
            // temp_val = A[i,j] * knownX[j]
            temp_val.x := hostA[b * strideA + j * lda + k_band].x * knownX[b * strideX + j].x -
                          hostA[b * strideA + j * lda + k_band].y * knownX[b * strideX + j].y;
            temp_val.y := hostA[b * strideA + j * lda + k_band].x * knownX[b * strideX + j].y +
                          hostA[b * strideA + j * lda + k_band].y * knownX[b * strideX + j].x;
            // B[i] += temp_val
            hostB[b * strideX + i].x := hostB[b * strideX + i].x + temp_val.x;
            hostB[b * strideX + i].y := hostB[b * strideX + i].y + temp_val.y;
          end;
        end;
      end;
    end;

    // 4. Initialize vector X (initial guess for the solver)
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := hostB[b * strideX + i]; // Initialize with RHS
    end;

    // 5. Set expectedX to knownX for comparison
    for i := 0 to N - 1 do
    begin
      expectedX[b * strideX + i] := knownX[b * strideX + i];
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX (B) H2D', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function: Solve A * X = B
  AssertEquals('rocblas_ctbsv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ctbsv_strided_batched_64(handle, uplo, transA, diag, N, K,
      devA, lda, strideA,
      devX, 1, strideX,
      batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devX D2H', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-3, Format('CTBSV_strided_batched_64 X[%d].x', [i])); // Relaxed tolerance
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-3, Format('CTBSV_strided_batched_64 X[%d].y', [i]));
  end;


  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZTBSV_strided_batched_64;
const
  N = 4;
  K = 1; // Number of super-diagonals
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Double = 2.0; // Scaling factor for matrix elements
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX, hostX_result, expectedX: array of rocblas_double_complex;
  strideA, strideX: Int64;
  i, j, b, k_band: Integer;
  lda: Int64; // Leading dimension for banded storage
  knownX: array of rocblas_double_complex; // The true solution
  temp_val: rocblas_double_complex;
begin
  lda := K + 1;
  strideA := N * lda;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count); // B vector for RHS
  SetLength(hostX, strideX * batch_count);  // X vector for initial guess / solution
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(knownX, strideX * batch_count); // True solution

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize banded triangular matrix A (complex)
    for j := 0 to N - 1 do // Iterate through columns
    begin
      for k_band := 0 to lda - 1 do
      begin
        hostA[b * strideA + j * lda + k_band].x := 0.0;
        hostA[b * strideA + j * lda + k_band].y := 0.0;
      end;
      // Fill relevant bands
      for i := Max(0, j - K) to j do
      begin
        k_band := K + i - j;
        if (k_band >= 0) and (k_band < lda) then
        begin
          hostA[b * strideA + j * lda + k_band].x := alpha * ((b + 1) * 10 + (i + 1) + (j + 1) * 0.1);
          hostA[b * strideA + j * lda + k_band].y := alpha * ((b + 1) * 0.5 + Abs(i-j) * 0.1);
          // Make diagonal elements significantly larger for diagonal dominance
          if i = j then
          begin
            hostA[b * strideA + j * lda + k_band].x := hostA[b * strideA + j * lda + k_band].x + N * 5 * alpha;
          end;
          // Handle unit diagonal
          if diag = rocblas_diagonal_unit then
          begin
            if i = j then
            begin
              hostA[b * strideA + j * lda + k_band].x := 1.0;
              hostA[b * strideA + j * lda + k_band].y := 0.0;
            end;
          end;
        end;
      end;
    end;

    // 2. Initialize the true solution vector (knownX)
    for i := 0 to N - 1 do
    begin
      knownX[b * strideX + i].x := (b + 1) * 3 + i + 1; // Real part
      knownX[b * strideX + i].y := (b + 1) * 0.5 + i * 0.2; // Imaginary part
    end;

    // 3. Compute RHS vector B = A * knownX for each batch (complex banded mat-vec)
    for i := 0 to N - 1 do
    begin
      hostB[b * strideX + i].x := 0.0;
      hostB[b * strideX + i].y := 0.0;
      for j := 0 to N - 1 do
      begin
        // Check if A[i, j] is within the band (j - K <= i <= j for upper band)
        if ((i >= j - K) and (i <= j)) then
        begin
          k_band := K + i - j;
          if (k_band >= 0) and (k_band < lda) then
          begin
            // temp_val = A[i,j] * knownX[j]
            temp_val.x := hostA[b * strideA + j * lda + k_band].x * knownX[b * strideX + j].x -
                          hostA[b * strideA + j * lda + k_band].y * knownX[b * strideX + j].y;
            temp_val.y := hostA[b * strideA + j * lda + k_band].x * knownX[b * strideX + j].y +
                          hostA[b * strideA + j * lda + k_band].y * knownX[b * strideX + j].x;
            // B[i] += temp_val
            hostB[b * strideX + i].x := hostB[b * strideX + i].x + temp_val.x;
            hostB[b * strideX + i].y := hostB[b * strideX + i].y + temp_val.y;
          end;
        end;
      end;
    end;

    // 4. Initialize vector X (initial guess for the solver)
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := hostB[b * strideX + i]; // Initialize with RHS
    end;

    // 5. Set expectedX to knownX for comparison
    for i := 0 to N - 1 do
    begin
      expectedX[b * strideX + i] := knownX[b * strideX + i];
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  // Copy data from host to device
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX (B) H2D', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function: Solve A * X = B
  AssertEquals('rocblas_ztbsv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ztbsv_strided_batched_64(handle, uplo, transA, diag, N, K,
      devA, lda, strideA,
      devX, 1, strideX,
      batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devX D2H', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTBSV_strided_batched_64 X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTBSV_strided_batched_64 X[%d].y', [i]));
  end;


  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_STPSV_strided_batched_64;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Single = 2.0; // Scaling factor for matrix elements
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostB, hostX, hostX_result, expectedX: array of Single;
  fullA: array of Single; // For CPU reference calculation
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
  knownX: array of Single; // The true solution we will use to generate B
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostB, strideX * batch_count); // B vector for RHS
  SetLength(hostX, strideX * batch_count);  // X vector for initial guess / solution
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(knownX, strideX * batch_count); // True solution
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full triangular matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          fullA[b*N*N + j*N + i] := alpha * ((b+1)*10 + i + j + 1);
          // Make diagonal elements significantly larger for stability
          if i = j then
          begin
            fullA[b*N*N + j*N + i] := fullA[b*N*N + j*N + i] + N * 5 * alpha;
            // Handle unit diagonal
            if diag = rocblas_diagonal_unit then
               fullA[b*N*N + j*N + i] := 1.0;
          end;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          fullA[b*N*N + j*N + i] := 0.0;
        end;
      end;

    // 2. Pack the referenced part (upper or lower) of the full matrix into hostAP (column-major)
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;

    // 3. Initialize the true solution vector (knownX)
    for i := 0 to N - 1 do
    begin
      knownX[b*strideX + i] := (b + 1) * 3 + i + 1; // Simple pattern
    end;

    // 4. Compute RHS vector B = A * knownX for each batch
    for i := 0 to N - 1 do // For each element of result vector B
    begin
      hostB[b*strideX + i] := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // B[i] += A[i,j] * knownX[j]
        hostB[b*strideX + i] := hostB[b*strideX + i] + fullA[b*N*N + j*N + i] * knownX[b*strideX + j];
      end;
    end;

    // 5. Initialize vector X (initial guess for the solver)
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := hostB[b*strideX + i]; // Initialize with RHS
    end;

    // 6. Set expectedX to knownX for comparison
    for i := 0 to N - 1 do
    begin
      expectedX[b*strideX + i] := knownX[b*strideX + i];
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X (B)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice)); // Copy B to X initially

  // Call rocBLAS function: Solve A * X = B
  AssertEquals('rocblas_stpsv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_stpsv_strided_batched_64(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-3, Format('STPSV_strided_batched_64 X[%d]', [i])); // Relaxed tolerance


  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DTPSV_strided_batched_64;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Double = 2.0; // Scaling factor for matrix elements
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostB, hostX, hostX_result, expectedX: array of Double;
  fullA: array of Double; // For CPU reference calculation
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
  knownX: array of Double; // The true solution we will use to generate B
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostB, strideX * batch_count); // B vector for RHS
  SetLength(hostX, strideX * batch_count);  // X vector for initial guess / solution
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(knownX, strideX * batch_count); // True solution
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full triangular matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          fullA[b*N*N + j*N + i] := alpha * ((b+1)*10 + i + j + 1);
          // Make diagonal elements significantly larger for stability
          if i = j then
          begin
            fullA[b*N*N + j*N + i] := fullA[b*N*N + j*N + i] + N * 5 * alpha;
            // Handle unit diagonal
            if diag = rocblas_diagonal_unit then
               fullA[b*N*N + j*N + i] := 1.0;
          end;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          fullA[b*N*N + j*N + i] := 0.0;
        end;
      end;

    // 2. Pack the referenced part (upper or lower) of the full matrix into hostAP (column-major)
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;

    // 3. Initialize the true solution vector (knownX)
    for i := 0 to N - 1 do
    begin
      knownX[b*strideX + i] := (b + 1) * 3 + i + 1; // Simple pattern
    end;

    // 4. Compute RHS vector B = A * knownX for each batch
    for i := 0 to N - 1 do // For each element of result vector B
    begin
      hostB[b*strideX + i] := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // B[i] += A[i,j] * knownX[j]
        hostB[b*strideX + i] := hostB[b*strideX + i] + fullA[b*N*N + j*N + i] * knownX[b*strideX + j];
      end;
    end;

    // 5. Initialize vector X (initial guess for the solver)
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := hostB[b*strideX + i]; // Initialize with RHS
    end;

    // 6. Set expectedX to knownX for comparison
    for i := 0 to N - 1 do
    begin
      expectedX[b*strideX + i] := knownX[b*strideX + i];
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X (B)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice)); // Copy B to X initially

  // Call rocBLAS function: Solve A * X = B
  AssertEquals('rocblas_dtpsv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dtpsv_strided_batched_64(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTPSV_strided_batched_64 X[%d]', [i]));


  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CTPSV_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Single = 2.0; // Scaling factor for matrix elements
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostB, hostX, hostX_result, expectedX: array of rocblas_float_complex;
  fullA: array of rocblas_float_complex; // For CPU reference calculation
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
  knownX: array of rocblas_float_complex; // The true solution
  temp_val: rocblas_float_complex;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostB, strideX * batch_count); // B vector for RHS
  SetLength(hostX, strideX * batch_count);  // X vector for initial guess / solution
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(knownX, strideX * batch_count); // True solution
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full triangular matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          fullA[b*N*N + j*N + i].x := alpha * ((b+1)*10 + i + j + 1);
          fullA[b*N*N + j*N + i].y := alpha * ((b+1)*0.5 + Abs(i-j)*0.1);
          // Make diagonal elements significantly larger for stability
          if i = j then
          begin
            fullA[b*N*N + j*N + i].x := fullA[b*N*N + j*N + i].x + N * 5 * alpha;
            // Optionally make diagonal real if needed
            // fullA[b*N*N + j*N + i].y := 0.0;
            // Handle unit diagonal
            if diag = rocblas_diagonal_unit then
            begin
               fullA[b*N*N + j*N + i].x := 1.0;
               fullA[b*N*N + j*N + i].y := 0.0;
            end;
          end;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          fullA[b*N*N + j*N + i].x := 0.0;
          fullA[b*N*N + j*N + i].y := 0.0;
        end;
      end;

    // 2. Pack the referenced part (upper or lower) of the full matrix into hostAP (column-major)
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;

    // 3. Initialize the true solution vector (knownX)
    for i := 0 to N - 1 do
    begin
      knownX[b*strideX + i].x := (b + 1) * 3 + i + 1; // Real part
      knownX[b*strideX + i].y := (b + 1) * 0.5 + i * 0.2; // Imaginary part
    end;

    // 4. Compute RHS vector B = A * knownX for each batch (complex mat-vec)
    for i := 0 to N - 1 do // For each element of result vector B
    begin
      hostB[b*strideX + i].x := 0.0;
      hostB[b*strideX + i].y := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // temp_val = A[i,j] * knownX[j]
        temp_val.x := fullA[b*N*N + j*N + i].x * knownX[b*strideX + j].x -
                      fullA[b*N*N + j*N + i].y * knownX[b*strideX + j].y;
        temp_val.y := fullA[b*N*N + j*N + i].x * knownX[b*strideX + j].y +
                      fullA[b*N*N + j*N + i].y * knownX[b*strideX + j].x;
        // B[i] += temp_val
        hostB[b*strideX + i].x := hostB[b*strideX + i].x + temp_val.x;
        hostB[b*strideX + i].y := hostB[b*strideX + i].y + temp_val.y;
      end;
    end;

    // 5. Initialize vector X (initial guess for the solver)
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := hostB[b*strideX + i]; // Initialize with RHS
    end;

    // 6. Set expectedX to knownX for comparison
    for i := 0 to N - 1 do
    begin
      expectedX[b*strideX + i] := knownX[b*strideX + i];
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X (B)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice)); // Copy B to X initially

  // Call rocBLAS function: Solve A * X = B
  AssertEquals('rocblas_ctpsv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ctpsv_strided_batched_64(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-3, Format('CTPSV_strided_batched_64 X[%d].x', [i])); // Relaxed tolerance
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-3, Format('CTPSV_strided_batched_64 X[%d].y', [i]));
  end;


  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZTPSV_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  alpha: Double = 2.0; // Scaling factor for matrix elements
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostB, hostX, hostX_result, expectedX: array of rocblas_double_complex;
  fullA: array of rocblas_double_complex; // For CPU reference calculation
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
  knownX: array of rocblas_double_complex; // The true solution
  temp_val: rocblas_double_complex;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostB, strideX * batch_count); // B vector for RHS
  SetLength(hostX, strideX * batch_count);  // X vector for initial guess / solution
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(knownX, strideX * batch_count); // True solution
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full triangular matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Part of the triangle (including diagonal)
          fullA[b*N*N + j*N + i].x := alpha * ((b+1)*10 + i + j + 1);
          fullA[b*N*N + j*N + i].y := alpha * ((b+1)*0.5 + Abs(i-j)*0.1);
          // Make diagonal elements significantly larger for stability
          if i = j then
          begin
            fullA[b*N*N + j*N + i].x := fullA[b*N*N + j*N + i].x + N * 5 * alpha;
            // Handle unit diagonal
            if diag = rocblas_diagonal_unit then
            begin
               fullA[b*N*N + j*N + i].x := 1.0;
               fullA[b*N*N + j*N + i].y := 0.0;
            end;
          end;
        end
        else
        begin
          // Part not part of the triangle (set to zero)
          fullA[b*N*N + j*N + i].x := 0.0;
          fullA[b*N*N + j*N + i].y := 0.0;
        end;
      end;

    // 2. Pack the referenced part (upper or lower) of the full matrix into hostAP (column-major)
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;

    // 3. Initialize the true solution vector (knownX)
    for i := 0 to N - 1 do
    begin
      knownX[b*strideX + i].x := (b + 1) * 3 + i + 1; // Real part
      knownX[b*strideX + i].y := (b + 1) * 0.5 + i * 0.2; // Imaginary part
    end;

    // 4. Compute RHS vector B = A * knownX for each batch (complex mat-vec)
    for i := 0 to N - 1 do // For each element of result vector B
    begin
      hostB[b*strideX + i].x := 0.0;
      hostB[b*strideX + i].y := 0.0;
      for j := 0 to N - 1 do // Column of matrix / Row of vector
      begin
        // temp_val = A[i,j] * knownX[j]
        temp_val.x := fullA[b*N*N + j*N + i].x * knownX[b*strideX + j].x -
                      fullA[b*N*N + j*N + i].y * knownX[b*strideX + j].y;
        temp_val.y := fullA[b*N*N + j*N + i].x * knownX[b*strideX + j].y +
                      fullA[b*N*N + j*N + i].y * knownX[b*strideX + j].x;
        // B[i] += temp_val
        hostB[b*strideX + i].x := hostB[b*strideX + i].x + temp_val.x;
        hostB[b*strideX + i].y := hostB[b*strideX + i].y + temp_val.y;
      end;
    end;

    // 5. Initialize vector X (initial guess for the solver)
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := hostB[b*strideX + i]; // Initialize with RHS
    end;

    // 6. Set expectedX to knownX for comparison
    for i := 0 to N - 1 do
    begin
      expectedX[b*strideX + i] := knownX[b*strideX + i];
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X (B)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice)); // Copy B to X initially

  // Call rocBLAS function: Solve A * X = B
  AssertEquals('rocblas_ztpsv_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ztpsv_strided_batched_64(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTPSV_strided_batched_64 X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTPSV_strided_batched_64 X[%d].y', [i]));
  end;

  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_SGER_strided_batched_64;
const
  M = 3; N = 4;
  batch_count = 2;
  alpha: Single = 2.0;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostA_result, expectedA: array of Single;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
begin
  strideA := M * N;
  strideX := M;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    // Initialize A
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
        hostA[b*strideA + j*M + i] := (b+1)*100 + (j+1)*10 + (i+1);

    // Initialize X
    for i := 0 to M - 1 do
      hostX[b*strideX + i] := (i + 1) * 1.0;

    // Initialize Y
    for i := 0 to N - 1 do
      hostY[b*strideY + i] := (i + 1) * 2.0;
  end;

  // Calculate expected result: A = alpha * x * y^T + A
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        expectedA[b*strideA + j*M + i] := hostA[b*strideA + j*M + i] + alpha * hostX[b*strideX + i] * hostY[b*strideY + j];
      end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_sger_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_sger_strided_batched_64(handle, M, N, @alpha,
                                          devX, 1, strideX,
                                          devY, 1, strideY,
                                          devA, M, strideA,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  for i := 0 to (strideA * batch_count) - 1 do
    CheckEqualsRelative(expectedA[i], hostA_result[i], 1e-5, Format('SGER_strided_batched_64 A[%d]',[i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DGER_strided_batched_64;
const
  M = 3; N = 4;
  batch_count = 2;
  alpha: Double = 2.0;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostA_result, expectedA: array of Double;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
begin
  strideA := M * N;
  strideX := M;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    // Initialize A
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
        hostA[b*strideA + j*M + i] := (b+1)*100 + (j+1)*10 + (i+1);

    // Initialize X
    for i := 0 to M - 1 do
      hostX[b*strideX + i] := (i + 1) * 1.0;

    // Initialize Y
    for i := 0 to N - 1 do
      hostY[b*strideY + i] := (i + 1) * 2.0;
  end;

  // Calculate expected result: A = alpha * x * y^T + A
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        expectedA[b*strideA + j*M + i] := hostA[b*strideA + j*M + i] + alpha * hostX[b*strideX + i] * hostY[b*strideY + j];
      end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dger_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dger_strided_batched_64(handle, M, N, @alpha,
                                          devX, 1, strideX,
                                          devY, 1, strideY,
                                          devA, M, strideA,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  for i := 0 to (strideA * batch_count) - 1 do
    CheckEqualsRelative(expectedA[i], hostA_result[i], 1e-12, Format('DGER_strided_batched_64 A[%d]',[i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CGERU_strided_batched_64;
const
  M = 4;
  N = 5;
  batch_count = 2;
  alpha: rocblas_float_complex = (x: 1.5; y: -1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostA_result, expectedA: array of rocblas_float_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  temp1, prod: rocblas_float_complex;
begin
  strideA := M * N;
  strideX := M;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M-1 do begin hostX[b*strideX+i].x:=(b+1)+i*0.5; hostX[b*strideX+i].y:=i*0.2; end;
    for i := 0 to N-1 do begin hostY[b*strideY+i].x:=(b+1)+i*-0.25; hostY[b*strideY+i].y:=i*-0.3; end;
    for j:=0 to N-1 do for i:=0 to M-1 do begin hostA[b*strideA+j*M+i].x:=(b+1)*10+i+j; hostA[b*strideA+j*M+i].y:=(b+1)*5-i-j; end;
  end;

  // CPU reference: A = alpha*x*y^T + A (GERU uses transpose, not conjugate transpose)
  for i := 0 to High(hostA) do expectedA[i] := hostA[i];

  for b := 0 to batch_count - 1 do
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        // prod = alpha * x[i] * y[j]
        // Step 1: temp1 = alpha * x[i]
        // (alpha.x + alpha.y*i) * (hostX.x + hostX.y*i)
        // = (alpha.x*hostX.x - alpha.y*hostX.y) + (alpha.x*hostX.y + alpha.y*hostX.x)*i
        temp1.x := alpha.x * hostX[b*strideX + i].x - alpha.y * hostX[b*strideX + i].y;
        temp1.y := alpha.x * hostX[b*strideX + i].y + alpha.y * hostX[b*strideX + i].x;

        // Step 2: prod = temp1 * y[j]
        // (temp1.x + temp1.y*i) * (hostY.x + hostY.y*i)
        // = (temp1.x*hostY.x - temp1.y*hostY.y) + (temp1.x*hostY.y + temp1.y*hostY.x)*i
        prod.x := temp1.x * hostY[b*strideY + j].x - temp1.y * hostY[b*strideY + j].y;
        prod.y := temp1.x * hostY[b*strideY + j].y + temp1.y * hostY[b*strideY + j].x;

        expectedA[b*strideA + j*M + i].x := expectedA[b*strideA + j*M + i].x + prod.x;
        expectedA[b*strideA + j*M + i].y := expectedA[b*strideA + j*M + i].y + prod.y;
      end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_cgeru_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_cgeru_strided_batched_64(handle, M, N, @alpha, devX, 1, strideX, devY, 1, strideY, devA, M, strideA, batch_count)));

  AssertEquals('hipMemcpy D2H A', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideA * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA_result[i].x, 1e-4, Format('CGERU_strided_batched_64 A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA_result[i].y, 1e-4, Format('CGERU_strided_batched_64 A[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZGERU_strided_batched_64;
const
  M = 4;
  N = 5;
  batch_count = 2;
  alpha: rocblas_double_complex = (x: 1.5; y: -1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostA_result, expectedA: array of rocblas_double_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  temp1, prod: rocblas_double_complex;
begin
  strideA := M * N;
  strideX := M;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M-1 do begin hostX[b*strideX+i].x:=(b+1)+i*0.5; hostX[b*strideX+i].y:=i*0.2; end;
    for i := 0 to N-1 do begin hostY[b*strideY+i].x:=(b+1)+i*-0.25; hostY[b*strideY+i].y:=i*-0.3; end;
    for j:=0 to N-1 do for i:=0 to M-1 do begin hostA[b*strideA+j*M+i].x:=(b+1)*10+i+j; hostA[b*strideA+j*M+i].y:=(b+1)*5-i-j; end;
  end;

  // CPU reference: A = alpha*x*y^T + A (GERU uses transpose, not conjugate transpose)
  for i := 0 to High(hostA) do expectedA[i] := hostA[i];

  for b := 0 to batch_count - 1 do
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        // prod = alpha * x[i] * y[j]
        // Step 1: temp1 = alpha * x[i]
        temp1.x := alpha.x * hostX[b*strideX + i].x - alpha.y * hostX[b*strideX + i].y;
        temp1.y := alpha.x * hostX[b*strideX + i].y + alpha.y * hostX[b*strideX + i].x;

        // Step 2: prod = temp1 * y[j]
        prod.x := temp1.x * hostY[b*strideY + j].x - temp1.y * hostY[b*strideY + j].y;
        prod.y := temp1.x * hostY[b*strideY + j].y + temp1.y * hostY[b*strideY + j].x;

        expectedA[b*strideA + j*M + i].x := expectedA[b*strideA + j*M + i].x + prod.x;
        expectedA[b*strideA + j*M + i].y := expectedA[b*strideA + j*M + i].y + prod.y;
      end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_zgeru_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_zgeru_strided_batched_64(handle, M, N, @alpha, devX, 1, strideX, devY, 1, strideY, devA, M, strideA, batch_count)));

  AssertEquals('hipMemcpy D2H A', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideA * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA_result[i].x, 1e-12, Format('ZGERU_strided_batched_64 A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA_result[i].y, 1e-12, Format('ZGERU_strided_batched_64 A[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CGERC_strided_batched_64;
const
  M = 4;
  N = 5;
  batch_count = 2;
  alpha: rocblas_float_complex = (x: 1.5; y: -1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostA_result, expectedA: array of rocblas_float_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  y_conj, temp1, prod: rocblas_float_complex;
begin
  strideA := M * N;
  strideX := M;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M-1 do begin hostX[b*strideX+i].x:=(b+1)+i*0.5; hostX[b*strideX+i].y:=i*0.2; end;
    for i := 0 to N-1 do begin hostY[b*strideY+i].x:=(b+1)+i*-0.25; hostY[b*strideY+i].y:=i*-0.3; end;
    for j:=0 to N-1 do for i:=0 to M-1 do begin hostA[b*strideA+j*M+i].x:=(b+1)*10+i+j; hostA[b*strideA+j*M+i].y:=(b+1)*5-i-j; end;
  end;

  // CPU reference: A = alpha*x*y^H + A (GERC uses conjugate transpose)
  for i := 0 to High(hostA) do expectedA[i] := hostA[i];

  for b := 0 to batch_count - 1 do
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        y_conj := hostY[b*strideY + j];
        y_conj.y := -y_conj.y; // Conjugate y

        // prod = alpha * x[i] * conj(y[j])
        // Step 1: temp1 = alpha * x[i]
        temp1.x := alpha.x * hostX[b*strideX + i].x - alpha.y * hostX[b*strideX + i].y;
        temp1.y := alpha.x * hostX[b*strideX + i].y + alpha.y * hostX[b*strideX + i].x;

        // Step 2: prod = temp1 * y_conj
        prod.x := temp1.x * y_conj.x - temp1.y * y_conj.y;
        prod.y := temp1.x * y_conj.y + temp1.y * y_conj.x;

        expectedA[b*strideA + j*M + i].x := expectedA[b*strideA + j*M + i].x + prod.x;
        expectedA[b*strideA + j*M + i].y := expectedA[b*strideA + j*M + i].y + prod.y;
      end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_cgerc_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_cgerc_strided_batched_64(handle, M, N, @alpha, devX, 1, strideX, devY, 1, strideY, devA, M, strideA, batch_count)));

  AssertEquals('hipMemcpy D2H A', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideA * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA_result[i].x, 1e-4, Format('CGERC_strided_batched_64 A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA_result[i].y, 1e-4, Format('CGERC_strided_batched_64 A[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZGERC_strided_batched_64;
const
  M = 4;
  N = 5;
  batch_count = 2;
  alpha: rocblas_double_complex = (x: 1.5; y: -1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostA_result, expectedA: array of rocblas_double_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  y_conj, temp1, prod: rocblas_double_complex;
begin
  strideA := M * N;
  strideX := M;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M-1 do begin hostX[b*strideX+i].x:=(b+1)+i*0.5; hostX[b*strideX+i].y:=i*0.2; end;
    for i := 0 to N-1 do begin hostY[b*strideY+i].x:=(b+1)+i*-0.25; hostY[b*strideY+i].y:=i*-0.3; end;
    for j:=0 to N-1 do for i:=0 to M-1 do begin hostA[b*strideA+j*M+i].x:=(b+1)*10+i+j; hostA[b*strideA+j*M+i].y:=(b+1)*5-i-j; end;
  end;

  // CPU reference: A = alpha*x*y^H + A (GERC uses conjugate transpose)
  for i := 0 to High(hostA) do expectedA[i] := hostA[i];

  for b := 0 to batch_count - 1 do
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        y_conj := hostY[b*strideY + j];
        y_conj.y := -y_conj.y; // Conjugate y

        // prod = alpha * x[i] * conj(y[j])
        // Step 1: temp1 = alpha * x[i]
        temp1.x := alpha.x * hostX[b*strideX + i].x - alpha.y * hostX[b*strideX + i].y;
        temp1.y := alpha.x * hostX[b*strideX + i].y + alpha.y * hostX[b*strideX + i].x;

        // Step 2: prod = temp1 * y_conj
        prod.x := temp1.x * y_conj.x - temp1.y * y_conj.y;
        prod.y := temp1.x * y_conj.y + temp1.y * y_conj.x;

        expectedA[b*strideA + j*M + i].x := expectedA[b*strideA + j*M + i].x + prod.x;
        expectedA[b*strideA + j*M + i].y := expectedA[b*strideA + j*M + i].y + prod.y;
      end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_zgerc_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_zgerc_strided_batched_64(handle, M, N, @alpha, devX, 1, strideX, devY, 1, strideY, devA, M, strideA, batch_count)));

  AssertEquals('hipMemcpy D2H A', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideA * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA_result[i].x, 1e-12, Format('ZGERC_strided_batched_64 A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA_result[i].y, 1e-12, Format('ZGERC_strided_batched_64 A[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_SSYR_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Single = 2.0;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostA_result, expectedA: array of Single;
  strideA, strideX: Int64;
  i, j, b: Integer;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    // Initialize symmetric matrix A (upper triangular part explicitly, but fill full for simplicity)
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        if i <= j then
        begin
          // Upper part (including diagonal)
          hostA[b * strideA + j * N + i] := (b + 1) * 100 + (i + 1) * 10 + (j + 1);
        end
        else
        begin
          // Lower part is copy of upper part for symmetric matrix
          hostA[b * strideA + j * N + i] := hostA[b * strideA + i * N + j];
        end;
      end;
    end;

    // Initialize X
    for i := 0 to N - 1 do
      hostX[b * strideX + i] := i + 1.0;
  end;

  // Compute expected result: A = alpha * x * x^T + A
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update the triangle specified by 'uplo'
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          expectedA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i] + alpha * hostX[b * strideX + i] * hostX[b * strideX + j];
        end
        else
        begin
          // Elements not in the specified triangle remain unchanged
          expectedA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i];
        end;
      end;
    end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ssyr_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ssyr_strided_batched_64(handle, uplo, N, @alpha,
                                          devX, 1, strideX,
                                          devA, N, strideA,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  for i := 0 to (strideA * batch_count) - 1 do
    CheckEqualsRelative(expectedA[i], hostA_result[i], 1e-5, Format('SSYR_strided_batched_64 A[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DSYR_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Double = 2.0;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostA_result, expectedA: array of Double;
  strideA, strideX: Int64;
  i, j, b: Integer;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    // Initialize symmetric matrix A (upper triangular part explicitly, but fill full for simplicity)
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        if i <= j then
        begin
          // Upper part (including diagonal)
          hostA[b * strideA + j * N + i] := (b + 1) * 100 + (i + 1) * 10 + (j + 1);
        end
        else
        begin
          // Lower part is copy of upper part for symmetric matrix
          hostA[b * strideA + j * N + i] := hostA[b * strideA + i * N + j];
        end;
      end;
    end;

    // Initialize X
    for i := 0 to N - 1 do
      hostX[b * strideX + i] := i + 1.0;
  end;

  // Compute expected result: A = alpha * x * x^T + A
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update the triangle specified by 'uplo'
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          expectedA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i] + alpha * hostX[b * strideX + i] * hostX[b * strideX + j];
        end
        else
        begin
          // Elements not in the specified triangle remain unchanged
          expectedA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i];
        end;
      end;
    end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dsyr_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dsyr_strided_batched_64(handle, uplo, N, @alpha,
                                          devX, 1, strideX,
                                          devA, N, strideA,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  for i := 0 to (strideA * batch_count) - 1 do
    CheckEqualsRelative(expectedA[i], hostA_result[i], 1e-12, Format('DSYR_strided_batched_64 A[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
end;

//procedure TTestRocBLAS_Level2_strided_batched_64.Test_CSYR_strided_batched_64;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZSYR_strided_batched_64;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CHER_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Single = 2.0; // Real alpha for Hermitian rank-1 update
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostA_result, expectedA: array of rocblas_float_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    // Initialize Hermitian matrix A (upper triangular part explicitly, but fill full for simplicity)
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        if i <= j then
        begin
          // Upper part (including diagonal)
          hostA[b * strideA + j * N + i].x := (b + 1) * 100 + (i + 1) * 10 + (j + 1);
          if i = j then
            hostA[b * strideA + j * N + i].y := 0.0 // Diagonal must be real
          else
            hostA[b * strideA + j * N + i].y := (b + 1) * 10 + (i + 1) + (j + 1) * 0.5;
        end
        else
        begin
          // Lower part is conjugate of upper part for Hermitian matrix
          hostA[b * strideA + j * N + i].x := hostA[b * strideA + i * N + j].x;
          hostA[b * strideA + j * N + i].y := -hostA[b * strideA + i * N + j].y;
        end;
      end;
    end;

    // Initialize X
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := (i + 1) * 1.0;
      hostX[b * strideX + i].y := (i + 1) * 0.5;
    end;
  end;

  // Compute expected result: A = alpha * x * x^H + A
  // For Hermitian: A[i,j] = A[i,j] + alpha * x[i] * conj(x[j])
  // Note: alpha is real, so conj(alpha) = alpha
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update the triangle specified by 'uplo'
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Calculate alpha * x[i] * conj(x[j])
          // x[i] = (x_real + x_imag * i)
          // conj(x[j]) = (xj_real - xj_imag * i)
          // Product = (x_real * xj_real + x_imag * xj_imag) + (x_imag * xj_real - x_real * xj_imag) * i
          expectedA[b * strideA + j * N + i].x := hostA[b * strideA + j * N + i].x +
            alpha * (hostX[b * strideX + i].x * hostX[b * strideX + j].x + hostX[b * strideX + i].y * hostX[b * strideX + j].y);
          if i = j then
          begin
            // Diagonal element remains real
            expectedA[b * strideA + j * N + i].y := 0.0;
          end
          else
          begin
            expectedA[b * strideA + j * N + i].y := hostA[b * strideA + j * N + i].y +
              alpha * (hostX[b * strideX + i].y * hostX[b * strideX + j].x - hostX[b * strideX + i].x * hostX[b * strideX + j].y);
          end;
        end
        else
        begin
          // Elements not in the specified triangle remain unchanged
          expectedA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i];
        end;
      end;
    end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_cher_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_cher_strided_batched_64(handle, uplo, N, @alpha,
                                          devX, 1, strideX,
                                          devA, N, strideA,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideA * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA_result[i].x, 1e-4, Format('CHER_strided_batched_64 A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA_result[i].y, 1e-4, Format('CHER_strided_batched_64 A[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZHER_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Double = 2.0; // Real alpha for Hermitian rank-1 update
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostA_result, expectedA: array of rocblas_double_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    // Initialize Hermitian matrix A (upper triangular part explicitly, but fill full for simplicity)
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        if i <= j then
        begin
          // Upper part (including diagonal)
          hostA[b * strideA + j * N + i].x := (b + 1) * 100 + (i + 1) * 10 + (j + 1);
          if i = j then
            hostA[b * strideA + j * N + i].y := 0.0 // Diagonal must be real
          else
            hostA[b * strideA + j * N + i].y := (b + 1) * 10 + (i + 1) + (j + 1) * 0.5;
        end
        else
        begin
          // Lower part is conjugate of upper part for Hermitian matrix
          hostA[b * strideA + j * N + i].x := hostA[b * strideA + i * N + j].x;
          hostA[b * strideA + j * N + i].y := -hostA[b * strideA + i * N + j].y;
        end;
      end;
    end;

    // Initialize X
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := (i + 1) * 1.0;
      hostX[b * strideX + i].y := (i + 1) * 0.5;
    end;
  end;

  // Compute expected result: A = alpha * x * x^H + A
  // For Hermitian: A[i,j] = A[i,j] + alpha * x[i] * conj(x[j])
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update the triangle specified by 'uplo'
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Calculate alpha * x[i] * conj(x[j])
          expectedA[b * strideA + j * N + i].x := hostA[b * strideA + j * N + i].x +
            alpha * (hostX[b * strideX + i].x * hostX[b * strideX + j].x + hostX[b * strideX + i].y * hostX[b * strideX + j].y);
          if i = j then
          begin
            // Diagonal element remains real
            expectedA[b * strideA + j * N + i].y := 0.0;
          end
          else
          begin
            expectedA[b * strideA + j * N + i].y := hostA[b * strideA + j * N + i].y +
              alpha * (hostX[b * strideX + i].y * hostX[b * strideX + j].x - hostX[b * strideX + i].x * hostX[b * strideX + j].y);
          end;
        end
        else
        begin
          // Elements not in the specified triangle remain unchanged
          expectedA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i];
        end;
      end;
    end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_zher_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_zher_strided_batched_64(handle, uplo, N, @alpha,
                                          devX, 1, strideX,
                                          devA, N, strideA,
                                          batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideA * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA_result[i].x, 1e-12, Format('ZHER_strided_batched_64 A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA_result[i].y, 1e-12, Format('ZHER_strided_batched_64 A[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_SSPR_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Single = 2.0;
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostX, hostAP_result, expectedAP: array of Single;
  fullA: array of Single; // For CPU reference calculation
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full symmetric matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j;
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i]; // A(j,i) = A(i,j)
      end;

    // 2. Pack the referenced part (upper) of the full matrix into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vector X
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
    end;
  end;

  // CPU reference calculation: AP = alpha * x * x^T + AP
  // Using the full matrix for easier computation, then repack
  for b := 0 to batch_count - 1 do
  begin
    // Update full matrix
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        fullA[b*N*N + j*N + i] := fullA[b*N*N + j*N + i] + alpha * hostX[b*strideX + i] * hostX[b*strideX + j];

    // Repack the updated full matrix into expectedAP (only the triangle specified by 'uplo')
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_sspr_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_sspr_strided_batched_64(handle, uplo, N, @alpha, devX, 1, strideX, devAP, strideAP, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H AP', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  for i := 0 to (strideAP * batch_count) - 1 do
    CheckEqualsRelative(expectedAP[i], hostAP_result[i], 1e-5, Format('SSPR_strided_batched_64 AP[%d]', [i]));


  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DSPR_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Double = 2.0;
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostX, hostAP_result, expectedAP: array of Double;
  fullA: array of Double; // For CPU reference calculation
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full symmetric matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j;
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i]; // A(j,i) = A(i,j)
      end;

    // 2. Pack the referenced part (upper) of the full matrix into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vector X
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
    end;
  end;

  // CPU reference calculation: AP = alpha * x * x^T + AP
  // Using the full matrix for easier computation, then repack
  for b := 0 to batch_count - 1 do
  begin
    // Update full matrix
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        fullA[b*N*N + j*N + i] := fullA[b*N*N + j*N + i] + alpha * hostX[b*strideX + i] * hostX[b*strideX + j];

    // Repack the updated full matrix into expectedAP (only the triangle specified by 'uplo')
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_dspr_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dspr_strided_batched_64(handle, uplo, N, @alpha, devX, 1, strideX, devAP, strideAP, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H AP', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  for i := 0 to (strideAP * batch_count) - 1 do
    CheckEqualsRelative(expectedAP[i], hostAP_result[i], 1e-12, Format('DSPR_strided_batched_64 AP[%d]', [i]));


  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CHPR_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Single = 2.0; // Real alpha for Hermitian packed rank-1 update
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostX, hostAP_result, expectedAP: array of rocblas_float_complex;
  fullA: array of rocblas_float_complex; // For CPU reference calculation
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full Hermitian matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        if i = j then
        begin
          // Diagonal elements are real
          fullA[b*N*N + j*N + i].x := (b+1)*20 + i;
          fullA[b*N*N + j*N + i].y := 0.0;
        end
        else // Upper off-diagonal elements
        begin
          fullA[b*N*N + j*N + i].x := (b+1)*10 + i + j;
          fullA[b*N*N + j*N + i].y := (b+1) + i - j;
        end;
        // Lower part is conjugate of upper part
        fullA[b*N*N + i*N + j].x := fullA[b*N*N + j*N + i].x;
        fullA[b*N*N + i*N + j].y := -fullA[b*N*N + j*N + i].y;
      end;

    // 2. Pack the referenced part (upper) of the full matrix into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vector X
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.5;
      hostX[b*strideX + i].y := i - 0.5;
    end;
  end;

  // CPU reference calculation: AP = alpha * x * x^H + AP
  // Using the full matrix for easier computation, then repack
  for b := 0 to batch_count - 1 do
  begin
    // Update full matrix: A[i,j] = A[i,j] + alpha * x[i] * conj(x[j])
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        // Calculate alpha * x[i] * conj(x[j])
        // alpha is real
        // temp_real = alpha * (x[i].real * x[j].real + x[i].imag * x[j].imag)
        // temp_imag = alpha * (x[i].imag * x[j].real - x[i].real * x[j].imag)
        fullA[b*N*N + j*N + i].x := fullA[b*N*N + j*N + i].x +
          alpha * (hostX[b*strideX + i].x * hostX[b*strideX + j].x + hostX[b*strideX + i].y * hostX[b*strideX + j].y);
        if i = j then
        begin
          // Diagonal element remains real
          fullA[b*N*N + j*N + i].y := 0.0;
        end
        else
        begin
          fullA[b*N*N + j*N + i].y := fullA[b*N*N + j*N + i].y +
            alpha * (hostX[b*strideX + i].y * hostX[b*strideX + j].x - hostX[b*strideX + i].x * hostX[b*strideX + j].y);
        end;
      end;

    // Repack the updated full matrix into expectedAP (only the triangle specified by 'uplo')
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_chpr_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_chpr_strided_batched_64(handle, uplo, N, @alpha, devX, 1, strideX, devAP, strideAP, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H AP', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideAP * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedAP[i].x, hostAP_result[i].x, 1e-4, Format('CHPR_strided_batched_64 AP[%d].x', [i]));
    CheckEqualsRelative(expectedAP[i].y, hostAP_result[i].y, 1e-4, Format('CHPR_strided_batched_64 AP[%d].y', [i]));
  end;


  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZHPR_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Double = 2.0; // Real alpha for Hermitian packed rank-1 update
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostX, hostAP_result, expectedAP: array of rocblas_double_complex;
  fullA: array of rocblas_double_complex; // For CPU reference calculation
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full Hermitian matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        if i = j then
        begin
          // Diagonal elements are real
          fullA[b*N*N + j*N + i].x := (b+1)*20 + i;
          fullA[b*N*N + j*N + i].y := 0.0;
        end
        else // Upper off-diagonal elements
        begin
          fullA[b*N*N + j*N + i].x := (b+1)*10 + i + j;
          fullA[b*N*N + j*N + i].y := (b+1) + i - j;
        end;
        // Lower part is conjugate of upper part
        fullA[b*N*N + i*N + j].x := fullA[b*N*N + j*N + i].x;
        fullA[b*N*N + i*N + j].y := -fullA[b*N*N + j*N + i].y;
      end;

    // 2. Pack the referenced part (upper) of the full matrix into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vector X
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.5;
      hostX[b*strideX + i].y := i - 0.5;
    end;
  end;

  // CPU reference calculation: AP = alpha * x * x^H + AP
  // Using the full matrix for easier computation, then repack
  for b := 0 to batch_count - 1 do
  begin
    // Update full matrix: A[i,j] = A[i,j] + alpha * x[i] * conj(x[j])
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        // Calculate alpha * x[i] * conj(x[j])
        fullA[b*N*N + j*N + i].x := fullA[b*N*N + j*N + i].x +
          alpha * (hostX[b*strideX + i].x * hostX[b*strideX + j].x + hostX[b*strideX + i].y * hostX[b*strideX + j].y);
        if i = j then
        begin
          // Diagonal element remains real
          fullA[b*N*N + j*N + i].y := 0.0;
        end
        else
        begin
          fullA[b*N*N + j*N + i].y := fullA[b*N*N + j*N + i].y +
            alpha * (hostX[b*strideX + i].y * hostX[b*strideX + j].x - hostX[b*strideX + i].x * hostX[b*strideX + j].y);
        end;
      end;

    // Repack the updated full matrix into expectedAP (only the triangle specified by 'uplo')
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_zhpr_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_zhpr_strided_batched_64(handle, uplo, N, @alpha, devX, 1, strideX, devAP, strideAP, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H AP', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideAP * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedAP[i].x, hostAP_result[i].x, 1e-12, Format('ZHPR_strided_batched_64 AP[%d].x', [i]));
    CheckEqualsRelative(expectedAP[i].y, hostAP_result[i].y, 1e-12, Format('ZHPR_strided_batched_64 AP[%d].y', [i]));
  end;


  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_SSYR2_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Single = 2.0;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostA_result, expectedA: array of Single;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
begin
  strideA := N * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    // Initialize symmetric matrix A (upper triangular part explicitly, but fill full for simplicity)
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        if i <= j then
        begin
          // Upper part (including diagonal)
          hostA[b * strideA + j * N + i] := (b + 1) * 100 + (i + 1) * 10 + (j + 1);
        end
        else
        begin
          // Lower part is copy of upper part for symmetric matrix
          hostA[b * strideA + j * N + i] := hostA[b * strideA + i * N + j];
        end;
      end;
    end;

    // Initialize X
    for i := 0 to N - 1 do
      hostX[b * strideX + i] := i + 1.0;

    // Initialize Y
    for i := 0 to N - 1 do
      hostY[b * strideY + i] := (i + 1) * 2.0;
  end;

  // Compute expected result: A = alpha * x * y^T + alpha * y * x^T + A
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update the triangle specified by 'uplo'
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          expectedA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i] +
            alpha * hostX[b * strideX + i] * hostY[b * strideY + j] +
            alpha * hostY[b * strideY + i] * hostX[b * strideX + j];
        end
        else
        begin
          // Elements not in the specified triangle remain unchanged
          expectedA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i];
        end;
      end;
    end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ssyr2_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_ssyr2_strided_batched_64(handle, uplo, N, @alpha,
                                           devX, 1, strideX,
                                           devY, 1, strideY,
                                           devA, N, strideA,
                                           batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  for i := 0 to (strideA * batch_count) - 1 do
    CheckEqualsRelative(expectedA[i], hostA_result[i], 1e-5, Format('SSYR2_strided_batched_64 A[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DSYR2_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Double = 2.0;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostA_result, expectedA: array of Double;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
begin
  strideA := N * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    // Initialize symmetric matrix A (upper triangular part explicitly, but fill full for simplicity)
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        if i <= j then
        begin
          // Upper part (including diagonal)
          hostA[b * strideA + j * N + i] := (b + 1) * 100 + (i + 1) * 10 + (j + 1);
        end
        else
        begin
          // Lower part is copy of upper part for symmetric matrix
          hostA[b * strideA + j * N + i] := hostA[b * strideA + i * N + j];
        end;
      end;
    end;

    // Initialize X
    for i := 0 to N - 1 do
      hostX[b * strideX + i] := i + 1.0;

    // Initialize Y
    for i := 0 to N - 1 do
      hostY[b * strideY + i] := (i + 1) * 2.0;
  end;

  // Compute expected result: A = alpha * x * y^T + alpha * y * x^T + A
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update the triangle specified by 'uplo'
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          expectedA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i] +
            alpha * hostX[b * strideX + i] * hostY[b * strideY + j] +
            alpha * hostY[b * strideY + i] * hostX[b * strideX + j];
        end
        else
        begin
          // Elements not in the specified triangle remain unchanged
          expectedA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i];
        end;
      end;
    end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dsyr2_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dsyr2_strided_batched_64(handle, uplo, N, @alpha,
                                           devX, 1, strideX,
                                           devY, 1, strideY,
                                           devA, N, strideA,
                                           batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  for i := 0 to (strideA * batch_count) - 1 do
    CheckEqualsRelative(expectedA[i], hostA_result[i], 1e-12, Format('DSYR2_strided_batched_64 A[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

//procedure TTestRocBLAS_Level2_strided_batched_64.Test_CSYR2_strided_batched_64;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZSYR2_strided_batched_64;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CHER2_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: rocblas_float_complex = (x: 1.5; y: -0.5); // Complex alpha
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostA_result, expectedA: array of rocblas_float_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  temp1, temp2, prod1, prod2: rocblas_float_complex;
begin
  strideA := N * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    // Initialize Hermitian matrix A (upper triangular part explicitly, but fill full for simplicity)
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        if i <= j then
        begin
          // Upper part (including diagonal)
          hostA[b * strideA + j * N + i].x := (b + 1) * 100 + (i + 1) * 10 + (j + 1);
          if i = j then
            hostA[b * strideA + j * N + i].y := 0.0 // Diagonal must be real
          else
            hostA[b * strideA + j * N + i].y := (b + 1) * 10 + (i + 1) + (j + 1) * 0.5;
        end
        else
        begin
          // Lower part is conjugate of upper part for Hermitian matrix
          hostA[b * strideA + j * N + i].x := hostA[b * strideA + i * N + j].x;
          hostA[b * strideA + j * N + i].y := -hostA[b * strideA + i * N + j].y;
        end;
      end;
    end;

    // Initialize X
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := (i + 1) * 1.0;
      hostX[b * strideX + i].y := (i + 1) * 0.5;
    end;

    // Initialize Y
    for i := 0 to N - 1 do
    begin
      hostY[b * strideY + i].x := (i + 1) * 2.0;
      hostY[b * strideY + i].y := (i + 1) * 1.5;
    end;
  end;

  // Compute expected result: A = alpha * x * y^H + conj(alpha) * y * x^H + A
  // For Hermitian: A[i,j] = A[i,j] + alpha * x[i] * conj(y[j]) + conj(alpha) * y[i] * conj(x[j])
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update the triangle specified by 'uplo'
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Calculate alpha * x[i] * conj(y[j])
          // conj(y[j]) = (y_real - y_imag * i)
          // (alpha.x + alpha.y*i) * (x_real + x_imag*i) * (y_real - y_imag*i)
          // Step 1: temp1 = alpha * x[i]
          temp1.x := alpha.x * hostX[b * strideX + i].x - alpha.y * hostX[b * strideX + i].y;
          temp1.y := alpha.x * hostX[b * strideX + i].y + alpha.y * hostX[b * strideX + i].x;
          // Step 2: prod1 = temp1 * conj(y[j])
          prod1.x := temp1.x * hostY[b * strideY + j].x + temp1.y * hostY[b * strideY + j].y;
          prod1.y := temp1.y * hostY[b * strideY + j].x - temp1.x * hostY[b * strideY + j].y;

          // Calculate conj(alpha) * y[i] * conj(x[j])
          // conj(alpha) = (alpha.x - alpha.y*i)
          // Step 1: temp2 = conj(alpha) * y[i]
          temp2.x := alpha.x * hostY[b * strideY + i].x + alpha.y * hostY[b * strideY + i].y;
          temp2.y := alpha.x * hostY[b * strideY + i].y - alpha.y * hostY[b * strideY + i].x;
          // Step 2: prod2 = temp2 * conj(x[j])
          prod2.x := temp2.x * hostX[b * strideX + j].x + temp2.y * hostX[b * strideX + j].y;
          prod2.y := temp2.y * hostX[b * strideX + j].x - temp2.x * hostX[b * strideX + j].y;

          expectedA[b * strideA + j * N + i].x := hostA[b * strideA + j * N + i].x + prod1.x + prod2.x;
          if i = j then
          begin
            // Diagonal element remains real
            expectedA[b * strideA + j * N + i].y := 0.0;
          end
          else
          begin
            expectedA[b * strideA + j * N + i].y := hostA[b * strideA + j * N + i].y + prod1.y + prod2.y;
          end;
        end
        else
        begin
          // Elements not in the specified triangle remain unchanged
          expectedA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i];
        end;
      end;
    end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_cher2_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_cher2_strided_batched_64(handle, uplo, N, @alpha,
                                           devX, 1, strideX,
                                           devY, 1, strideY,
                                           devA, N, strideA,
                                           batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideA * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA_result[i].x, 1e-4, Format('CHER2_strided_batched_64 A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA_result[i].y, 1e-4, Format('CHER2_strided_batched_64 A[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZHER2_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: rocblas_double_complex = (x: 1.5; y: -0.5); // Complex alpha
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostA_result, expectedA: array of rocblas_double_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  temp1, temp2, prod1, prod2: rocblas_double_complex;
begin
  strideA := N * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostA_result, strideA * batch_count);
  SetLength(expectedA, strideA * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    // Initialize Hermitian matrix A (upper triangular part explicitly, but fill full for simplicity)
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        if i <= j then
        begin
          // Upper part (including diagonal)
          hostA[b * strideA + j * N + i].x := (b + 1) * 100 + (i + 1) * 10 + (j + 1);
          if i = j then
            hostA[b * strideA + j * N + i].y := 0.0 // Diagonal must be real
          else
            hostA[b * strideA + j * N + i].y := (b + 1) * 10 + (i + 1) + (j + 1) * 0.5;
        end
        else
        begin
          // Lower part is conjugate of upper part for Hermitian matrix
          hostA[b * strideA + j * N + i].x := hostA[b * strideA + i * N + j].x;
          hostA[b * strideA + j * N + i].y := -hostA[b * strideA + i * N + j].y;
        end;
      end;
    end;

    // Initialize X
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := (i + 1) * 1.0;
      hostX[b * strideX + i].y := (i + 1) * 0.5;
    end;

    // Initialize Y
    for i := 0 to N - 1 do
    begin
      hostY[b * strideY + i].x := (i + 1) * 2.0;
      hostY[b * strideY + i].y := (i + 1) * 1.5;
    end;
  end;

  // Compute expected result: A = alpha * x * y^H + conj(alpha) * y * x^H + A
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update the triangle specified by 'uplo'
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Calculate alpha * x[i] * conj(y[j])
          // Step 1: temp1 = alpha * x[i]
          temp1.x := alpha.x * hostX[b * strideX + i].x - alpha.y * hostX[b * strideX + i].y;
          temp1.y := alpha.x * hostX[b * strideX + i].y + alpha.y * hostX[b * strideX + i].x;
          // Step 2: prod1 = temp1 * conj(y[j])
          prod1.x := temp1.x * hostY[b * strideY + j].x + temp1.y * hostY[b * strideY + j].y;
          prod1.y := temp1.y * hostY[b * strideY + j].x - temp1.x * hostY[b * strideY + j].y;

          // Calculate conj(alpha) * y[i] * conj(x[j])
          // Step 1: temp2 = conj(alpha) * y[i]
          temp2.x := alpha.x * hostY[b * strideY + i].x + alpha.y * hostY[b * strideY + i].y;
          temp2.y := alpha.x * hostY[b * strideY + i].y - alpha.y * hostY[b * strideY + i].x;
          // Step 2: prod2 = temp2 * conj(x[j])
          prod2.x := temp2.x * hostX[b * strideX + j].x + temp2.y * hostX[b * strideX + j].y;
          prod2.y := temp2.y * hostX[b * strideX + j].x - temp2.x * hostX[b * strideX + j].y;

          expectedA[b * strideA + j * N + i].x := hostA[b * strideA + j * N + i].x + prod1.x + prod2.x;
          if i = j then
          begin
            // Diagonal element remains real
            expectedA[b * strideA + j * N + i].y := 0.0;
          end
          else
          begin
            expectedA[b * strideA + j * N + i].y := hostA[b * strideA + j * N + i].y + prod1.y + prod2.y;
          end;
        end
        else
        begin
          // Elements not in the specified triangle remain unchanged
          expectedA[b * strideA + j * N + i] := hostA[b * strideA + j * N + i];
        end;
      end;
    end;
  end;

  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_zher2_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_zher2_strided_batched_64(handle, uplo, N, @alpha,
                                           devX, 1, strideX,
                                           devY, 1, strideY,
                                           devA, N, strideA,
                                           batch_count)));

  AssertEquals('hipMemcpy result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideA * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA_result[i].x, 1e-12, Format('ZHER2_strided_batched_64 A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA_result[i].y, 1e-12, Format('ZHER2_strided_batched_64 A[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_SSPR2_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Single = 2.0;
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostX, hostY, hostAP_result, expectedAP: array of Single;
  fullA: array of Single; // For CPU reference calculation
  packed_size, strideAP, strideX, strideY: Int64;
  i, j, b, k: Integer;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;
  strideY := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full symmetric matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j;
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i]; // A(j,i) = A(i,j)
      end;

    // 2. Pack the referenced part (upper) of the full matrix into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vectors X and Y
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
      hostY[b*strideY + i] := (i + 1) * 2.0;
    end;
  end;

  // CPU reference calculation: AP = alpha * x * y^T + alpha * y * x^T + AP
  // Using the full matrix for easier computation, then repack
  for b := 0 to batch_count - 1 do
  begin
    // Update full matrix
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        fullA[b*N*N + j*N + i] := fullA[b*N*N + j*N + i] +
          alpha * hostX[b*strideX + i] * hostY[b*strideY + j] +
          alpha * hostY[b*strideY + i] * hostX[b*strideX + j];

    // Repack the updated full matrix into expectedAP (only the triangle specified by 'uplo')
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_sspr2_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_sspr2_strided_batched_64(handle, uplo, N, @alpha,
      devX, 1, strideX,
      devY, 1, strideY,
      devAP, strideAP,
      batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H AP', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  for i := 0 to (strideAP * batch_count) - 1 do
    CheckEqualsRelative(expectedAP[i], hostAP_result[i], 1e-5, Format('SSPR2_strided_batched_64 AP[%d]', [i]));


  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_DSPR2_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Double = 2.0;
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostX, hostY, hostAP_result, expectedAP: array of Double;
  fullA: array of Double; // For CPU reference calculation
  packed_size, strideAP, strideX, strideY: Int64;
  i, j, b, k: Integer;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;
  strideY := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full symmetric matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j;
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i]; // A(j,i) = A(i,j)
      end;

    // 2. Pack the referenced part (upper) of the full matrix into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vectors X and Y
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
      hostY[b*strideY + i] := (i + 1) * 2.0;
    end;
  end;

  // CPU reference calculation: AP = alpha * x * y^T + alpha * y * x^T + AP
  // Using the full matrix for easier computation, then repack
  for b := 0 to batch_count - 1 do
  begin
    // Update full matrix
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        fullA[b*N*N + j*N + i] := fullA[b*N*N + j*N + i] +
          alpha * hostX[b*strideX + i] * hostY[b*strideY + j] +
          alpha * hostY[b*strideY + i] * hostX[b*strideX + j];

    // Repack the updated full matrix into expectedAP (only the triangle specified by 'uplo')
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_dspr2_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_dspr2_strided_batched_64(handle, uplo, N, @alpha,
      devX, 1, strideX,
      devY, 1, strideY,
      devAP, strideAP,
      batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H AP', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  for i := 0 to (strideAP * batch_count) - 1 do
    CheckEqualsRelative(expectedAP[i], hostAP_result[i], 1e-12, Format('DSPR2_strided_batched_64 AP[%d]', [i]));


  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_CHPR2_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: rocblas_float_complex = (x: 1.5; y: -0.5); // Complex alpha
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostX, hostY, hostAP_result, expectedAP: array of rocblas_float_complex;
  fullA: array of rocblas_float_complex; // For CPU reference calculation
  packed_size, strideAP, strideX, strideY: Int64;
  i, j, b, k: Integer;
  temp1, temp2, x_conj, y_conj: rocblas_float_complex;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;
  strideY := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full Hermitian matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        if i = j then
        begin
          // Diagonal elements are real
          fullA[b*N*N + j*N + i].x := (b+1)*20 + i;
          fullA[b*N*N + j*N + i].y := 0.0;
        end
        else // Upper off-diagonal elements
        begin
          fullA[b*N*N + j*N + i].x := (b+1)*10 + i + j;
          fullA[b*N*N + j*N + i].y := (b+1) + i - j;
        end;
        // Lower part is conjugate of upper part
        fullA[b*N*N + i*N + j].x := fullA[b*N*N + j*N + i].x;
        fullA[b*N*N + i*N + j].y := -fullA[b*N*N + j*N + i].y;
      end;

    // 2. Pack the referenced part (upper) of the full matrix into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vectors X and Y
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.5;
      hostX[b*strideX + i].y := i - 0.5;
      hostY[b*strideY + i].x := (i + 1) * 2.0;
      hostY[b*strideY + i].y := (i + 1) * 1.5;
    end;
  end;

  // CPU reference calculation: AP = alpha * x * y^H + conj(alpha) * y * x^H + AP
  // Using the full matrix for easier computation, then repack
  for b := 0 to batch_count - 1 do
  begin
    // Update full matrix: A[i,j] = A[i,j] + alpha * x[i] * conj(y[j]) + conj(alpha) * y[i] * conj(x[j])
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        // Calculate alpha * x[i] * conj(y[j])
        y_conj := hostY[b*strideY + j];
        y_conj.y := -y_conj.y; // conj(y[j])
        temp1 := alpha;
        ComplexMul(temp1, hostX[b*strideX + i]);
        ComplexMul(temp1, y_conj);

        // Calculate conj(alpha) * y[i] * conj(x[j])
        x_conj := hostX[b*strideX + j];
        x_conj.y := -x_conj.y; // conj(x[j])
        temp2 := alpha;
        temp2.y := -temp2.y; // conj(alpha)
        ComplexMul(temp2, hostY[b*strideY + i]);
        ComplexMul(temp2, x_conj);

        // Add both terms to A[i,j]
        ComplexAdd(fullA[b*N*N + j*N + i], temp1);
        ComplexAdd(fullA[b*N*N + j*N + i], temp2);

        // Ensure diagonal remains real
        if i = j then
          fullA[b*N*N + j*N + i].y := 0.0;
      end;

    // Repack the updated full matrix into expectedAP (only the triangle specified by 'uplo')
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      // For upper triangular packed storage (column-major):
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      // For lower triangular packed storage (column-major):
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_chpr2_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_chpr2_strided_batched_64(handle, uplo, N, @alpha,
      devX, 1, strideX,
      devY, 1, strideY,
      devAP, strideAP,
      batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H AP', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideAP * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedAP[i].x, hostAP_result[i].x, 1e-4, Format('CHPR2_strided_batched_64 AP[%d].x', [i]));
    CheckEqualsRelative(expectedAP[i].y, hostAP_result[i].y, 1e-4, Format('CHPR2_strided_batched_64 AP[%d].y', [i]));
  end;


  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched_64.Test_ZHPR2_strided_batched_64;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: rocblas_double_complex = (x: 1.5; y: -0.5); // Complex alpha
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostX, hostY, hostAP_result, expectedAP: array of rocblas_double_complex;
  fullA: array of rocblas_double_complex; // For CPU reference calculation
  packed_size, strideAP, strideX, strideY: Int64;
  i, j, b, k: Integer;
  temp1, temp2, x_conj, y_conj: rocblas_double_complex;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;
  strideY := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(fullA, N * N * batch_count); // Helper for reference calculation

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full Hermitian matrix for easy reference calculation
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        if i = j then
        begin
          // Diagonal elements are real
          fullA[b*N*N + j*N + i].x := (b+1)*20 + i;
          fullA[b*N*N + j*N + i].y := 0.0;
        end
        else // Upper off-diagonal elements
        begin
          fullA[b*N*N + j*N + i].x := (b+1)*10 + i + j;
          fullA[b*N*N + j*N + i].y := (b+1) + i - j;
        end;
        // Lower part is conjugate of upper part
        fullA[b*N*N + i*N + j].x := fullA[b*N*N + j*N + i].x;
        fullA[b*N*N + i*N + j].y := -fullA[b*N*N + j*N + i].y;
      end;

    // 2. Pack the referenced part (upper) of the full matrix into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vectors X and Y
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.5;
      hostX[b*strideX + i].y := i - 0.5;
      hostY[b*strideY + i].x := (i + 1) * 2.0;
      hostY[b*strideY + i].y := (i + 1) * 1.5;
    end;
  end;

  // CPU reference calculation: AP = alpha * x * y^H + conj(alpha) * y * x^H + AP
  // Using the full matrix for easier computation, then repack
  for b := 0 to batch_count - 1 do
  begin
    // Update full matrix: A[i,j] = A[i,j] + alpha * x[i] * conj(y[j]) + conj(alpha) * y[i] * conj(x[j])
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
      begin
        // Calculate alpha * x[i] * conj(y[j])
        y_conj := hostY[b*strideY + j];
        y_conj.y := -y_conj.y; // conj(y[j])
        temp1 := alpha;
        ComplexMul(temp1, hostX[b*strideX + i]);
        ComplexMul(temp1, y_conj);

        // Calculate conj(alpha) * y[i] * conj(x[j])
        x_conj := hostX[b*strideX + j];
        x_conj.y := -x_conj.y; // conj(x[j])
        temp2 := alpha;
        temp2.y := -temp2.y; // conj(alpha)
        ComplexMul(temp2, hostY[b*strideY + i]);
        ComplexMul(temp2, x_conj);

        // Add both terms to A[i,j]
        ComplexAdd(fullA[b*N*N + j*N + i], temp1);
        ComplexAdd(fullA[b*N*N + j*N + i], temp2);

        // Ensure diagonal remains real
        if i = j then
          fullA[b*N*N + j*N + i].y := 0.0;
      end;

    // Repack the updated full matrix into expectedAP (only the triangle specified by 'uplo')
    k := 0;
    if uplo = rocblas_fill_upper then
    begin
      // For upper triangular packed storage (column-major):
      for j := 0 to N - 1 do
        for i := 0 to j do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end
    else // rocblas_fill_lower
    begin
      // For lower triangular packed storage (column-major):
      for j := 0 to N - 1 do
        for i := j to N - 1 do
        begin
          expectedAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
          Inc(k);
        end;
    end;
  end;

  // Allocate and manage GPU memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_zhpr2_strided_batched_64 OK', Integer(rocblas_status_success),
    Integer(rocblas_zhpr2_strided_batched_64(handle, uplo, N, @alpha,
      devX, 1, strideX,
      devY, 1, strideY,
      devAP, strideAP,
      batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H AP', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideAP * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedAP[i].x, hostAP_result[i].x, 1e-12, Format('ZHPR2_strided_batched_64 AP[%d].x', [i]));
    CheckEqualsRelative(expectedAP[i].y, hostAP_result[i].y, 1e-12, Format('ZHPR2_strided_batched_64 AP[%d].y', [i]));
  end;


  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

initialization
  RegisterTest(TTestRocBLAS_Level2_strided_batched_64);

end.

