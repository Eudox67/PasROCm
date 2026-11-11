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

unit Test_rocblas_lvl2_strided_batched;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  { TTestRocBLAS_Level2_strided_batched }

  TTestRocBLAS_Level2_strided_batched = class(TTestRocBLAS_Base)
  published
    { General Matrix-Vector }
    procedure Test_SGEMV_strided_batched;
    procedure Test_DGEMV_strided_batched;
    procedure Test_CGEMV_strided_batched;
    procedure Test_ZGEMV_strided_batched;
    procedure Test_SGBMV_strided_batched;
    procedure Test_DGBMV_strided_batched;
    procedure Test_CGBMV_strided_batched;
    procedure Test_ZGBMV_strided_batched;

    { Hermitian }
    procedure Test_CHEMV_strided_batched;
    procedure Test_ZHEMV_strided_batched;
    procedure Test_CHBMV_strided_batched;
    procedure Test_ZHBMV_strided_batched;
    procedure Test_CHPMV_strided_batched;
    procedure Test_ZHPMV_strided_batched;

    { Symmetric }
    procedure Test_SSYMV_strided_batched;
    procedure Test_DSYMV_strided_batched;
    //procedure Test_CSYMV_strided_batched;
    //procedure Test_ZSYMV_strided_batched;
    procedure Test_SSBMV_strided_batched;
    procedure Test_DSBMV_strided_batched;
    procedure Test_SSPMV_strided_batched;
    procedure Test_DSPMV_strided_batched;

    { Triangular }
    procedure Test_STRMV_strided_batched;
    procedure Test_DTRMV_strided_batched;
    procedure Test_CTRMV_strided_batched;
    procedure Test_ZTRMV_strided_batched;
    procedure Test_STBMV_strided_batched;
    procedure Test_DTBMV_strided_batched;
    procedure Test_CTBMV_strided_batched;
    procedure Test_ZTBMV_strided_batched;
    procedure Test_STPMV_strided_batched;
    procedure Test_DTPMV_strided_batched;
    procedure Test_CTPMV_strided_batched;
    procedure Test_ZTPMV_strided_batched;

    { Triangular Solvers }
    procedure Test_STRSV_strided_batched;
    procedure Test_DTRSV_strided_batched;
    procedure Test_CTRSV_strided_batched;
    procedure Test_ZTRSV_strided_batched;
    procedure Test_STBSV_strided_batched;
    procedure Test_DTBSV_strided_batched;
    procedure Test_CTBSV_strided_batched;
    procedure Test_ZTBSV_strided_batched;
    procedure Test_STPSV_strided_batched;
    procedure Test_DTPSV_strided_batched;
    procedure Test_CTPSV_strided_batched;
    procedure Test_ZTPSV_strided_batched;

    { Rank-1 Updates }
    procedure Test_SGER_strided_batched;
    procedure Test_DGER_strided_batched;
    procedure Test_CGERU_strided_batched;
    procedure Test_ZGERU_strided_batched;
    procedure Test_CGERC_strided_batched;
    procedure Test_ZGERC_strided_batched;
    procedure Test_SSYR_strided_batched;
    procedure Test_DSYR_strided_batched;
    //procedure Test_CSYR_strided_batched;
    //procedure Test_ZSYR_strided_batched;
    procedure Test_CHER_strided_batched;
    procedure Test_ZHER_strided_batched;
    procedure Test_SSPR_strided_batched;
    procedure Test_DSPR_strided_batched;
    procedure Test_CHPR_strided_batched;
    procedure Test_ZHPR_strided_batched;

    { Rank-2 Updates }
    procedure Test_SSYR2_strided_batched;
    procedure Test_DSYR2_strided_batched;
    //procedure Test_CSYR2_strided_batched;
    //procedure Test_ZSYR2_strided_batched;
    procedure Test_CHER2_strided_batched;
    procedure Test_ZHER2_strided_batched;
    procedure Test_SSPR2_strided_batched;
    procedure Test_DSPR2_strided_batched;
    procedure Test_CHPR2_strided_batched;
    procedure Test_ZHPR2_strided_batched;
  end;

implementation

{ TTestRocBLAS_Level2_strided_batched }

procedure TTestRocBLAS_Level2_strided_batched.Test_SGEMV_strided_batched;
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

  hipMalloc(devA, strideA * batch_count * SizeOf(Single));
  hipMalloc(devX, strideX * batch_count * SizeOf(Single));
  hipMalloc(devY, strideY * batch_count * SizeOf(Single));

  hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_sgemv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_sgemv_strided_batched(handle, rocblas_operation_none, M, N,
                                          @alpha, devA, M, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-5, Format('SGEMV_strided_batched Y[%d]',[i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_DGEMV_strided_batched;
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

  hipMalloc(devA, strideA * batch_count * SizeOf(Double));
  hipMalloc(devX, strideX * batch_count * SizeOf(Double));
  hipMalloc(devY, strideY * batch_count * SizeOf(Double));

  hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dgemv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_dgemv_strided_batched(handle, rocblas_operation_none, M, N,
                                          @alpha, devA, M, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-12, Format('DGEMV_strided_batched Y[%d]',[i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CGEMV_strided_batched;
const
  M = 3; N = 4; batch_count = 2;
  alpha: rocblas_float_complex = (x: 1.5; y: -0.5);
  beta:  rocblas_float_complex = (x: -0.5; y: 1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of rocblas_float_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  temp: rocblas_float_complex;
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
    for j := 0 to N - 1 do for i := 0 to M - 1 do
    begin
      hostA[b*strideA + j*M + i].x := (b+1)*100 + (j+1)*10 + (i+1);
      hostA[b*strideA + j*M + i].y := -((b+1)*100 + (j+1)*10 + (i+1));
    end;
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0;
      hostX[b*strideX + i].y := -(i + 1.0);
    end;
    for i := 0 to M - 1 do
    begin
      hostY[b*strideY + i].x := i * 2.0;
      hostY[b*strideY + i].y := -(i * 2.0);
    end;
  end;

  for b := 0 to batch_count - 1 do
    for i := 0 to M - 1 do
    begin
      expectedY[b*strideY + i] := hostY[b*strideY + i];
      ComplexMul(expectedY[b*strideY + i], beta);
      temp.x := 0; temp.y := 0;
      for j := 0 to N - 1 do
        ComplexFMA(temp, hostA[b*strideA + j*M + i], hostX[b*strideX + j], alpha);
      ComplexAdd(expectedY[b*strideY + i], temp);
    end;

  hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex));
  hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex));
  hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex));

  hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_cgemv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_cgemv_strided_batched(handle, rocblas_operation_none, M, N,
                                          @alpha, devA, M, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-5, Format('CGEMV_strided_batched Y[%d].x',[i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-5, Format('CGEMV_strided_batched Y[%d].y',[i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_ZGEMV_strided_batched;
const
  M = 3; N = 4; batch_count = 2;
  alpha: rocblas_double_complex = (x: 1.5; y: -0.5);
  beta:  rocblas_double_complex = (x: -0.5; y: 1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of rocblas_double_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  temp: rocblas_double_complex;
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
    for j := 0 to N - 1 do for i := 0 to M - 1 do
    begin
      hostA[b*strideA + j*M + i].x := (b+1)*100 + (j+1)*10 + (i+1);
      hostA[b*strideA + j*M + i].y := -((b+1)*100 + (j+1)*10 + (i+1));
    end;
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0;
      hostX[b*strideX + i].y := -(i + 1.0);
    end;
    for i := 0 to M - 1 do
    begin
      hostY[b*strideY + i].x := i * 2.0;
      hostY[b*strideY + i].y := -(i * 2.0);
    end;
  end;

  for b := 0 to batch_count - 1 do
    for i := 0 to M - 1 do
    begin
      expectedY[b*strideY + i] := hostY[b*strideY + i];
      ComplexMul(expectedY[b*strideY + i], beta);
      temp.x := 0; temp.y := 0;
      for j := 0 to N - 1 do
        ComplexFMA(temp, hostA[b*strideA + j*M + i], hostX[b*strideX + j], alpha);
      ComplexAdd(expectedY[b*strideY + i], temp);
    end;

  hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex));
  hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex));
  hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex));

  hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zgemv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_zgemv_strided_batched(handle, rocblas_operation_none, M, N,
                                          @alpha, devA, M, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-12, Format('ZGEMV_strided_batched Y[%d].x',[i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-12, Format('ZGEMV_strided_batched Y[%d].y',[i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_SGBMV_strided_batched;
const
  M = 5; N = 6; KL = 2; KU = 1; batch_count = 2;
  alpha: Single = 2.0; beta: Single = -0.5;
  LDA = 4; // KL + KU + 1
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of Single;
  strideA, strideX, strideY: Int64;
  i, j, b, diag, row: Integer;
begin
  strideA := LDA * N;
  strideX := N;
  strideY := M;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    FillChar(hostA[b*strideA], strideA * SizeOf(Single), 0);
    for j := 0 to N - 1 do
      for diag := -KL to KU do
      begin
        row := j + diag;
        if (row >= 0) and (row < M) then
          hostA[b*strideA + (KU+diag) + j*LDA] := (b+1)*100 + (row+1)*10 + (j+1);
      end;
    for i := 0 to N - 1 do hostX[b*strideX + i] := i + 1.0;
    for i := 0 to M - 1 do hostY[b*strideY + i] := i * 3.0;
  end;

  for b := 0 to batch_count - 1 do
    for i := 0 to M - 1 do
    begin
      expectedY[b*strideY + i] := hostY[b*strideY + i] * beta;
      for j := 0 to N - 1 do
        if (i >= j-KU) and (i <= j+KL) then
          expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * hostA[b*strideA + (KU+(i-j)) + j*LDA] * hostX[b*strideX + j];
    end;

  hipMalloc(devA, strideA * batch_count * SizeOf(Single));
  hipMalloc(devX, strideX * batch_count * SizeOf(Single));
  hipMalloc(devY, strideY * batch_count * SizeOf(Single));

  hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_sgbmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_sgbmv_strided_batched(handle, rocblas_operation_none, M, N, KL, KU,
                                          @alpha, devA, LDA, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-5, Format('SGBMV_strided_batched Y[%d]',[i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_DGBMV_strided_batched;
const
  M = 5; N = 6; KL = 2; KU = 1; batch_count = 2;
  alpha: Double = 2.0; beta: Double = -0.5;
  LDA = 4; // KL + KU + 1
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of Double;
  strideA, strideX, strideY: Int64;
  i, j, b, diag, row: Integer;
begin
  strideA := LDA * N;
  strideX := N;
  strideY := M;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    FillChar(hostA[b*strideA], strideA * SizeOf(Double), 0);
    for j := 0 to N - 1 do
      for diag := -KL to KU do
      begin
        row := j + diag;
        if (row >= 0) and (row < M) then
          hostA[b*strideA + (KU+diag) + j*LDA] := (b+1)*100 + (row+1)*10 + (j+1);
      end;
    for i := 0 to N - 1 do hostX[b*strideX + i] := i + 1.0;
    for i := 0 to M - 1 do hostY[b*strideY + i] := i * 3.0;
  end;

  for b := 0 to batch_count - 1 do
    for i := 0 to M - 1 do
    begin
      expectedY[b*strideY + i] := hostY[b*strideY + i] * beta;
      for j := 0 to N - 1 do
        if (i >= j-KU) and (i <= j+KL) then
          expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * hostA[b*strideA + (KU+(i-j)) + j*LDA] * hostX[b*strideX + j];
    end;

  hipMalloc(devA, strideA * batch_count * SizeOf(Double));
  hipMalloc(devX, strideX * batch_count * SizeOf(Double));
  hipMalloc(devY, strideY * batch_count * SizeOf(Double));

  hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dgbmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_dgbmv_strided_batched(handle, rocblas_operation_none, M, N, KL, KU,
                                          @alpha, devA, LDA, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-12, Format('DGBMV_strided_batched Y[%d]',[i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CGBMV_strided_batched;
const
  M = 5; N = 6; KL = 2; KU = 1; batch_count = 2;
  alpha: rocblas_float_complex = (x: 1.0; y: -0.5);
  beta:  rocblas_float_complex = (x: -0.5; y: 1.0);
  LDA = 4; // KL + KU + 1
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of rocblas_float_complex;
  strideA, strideX, strideY: Int64;
  i, j, b, diag, row: Integer;
  temp: rocblas_float_complex;
begin
  strideA := LDA * N;
  strideX := N;
  strideY := M;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    FillChar(hostA[b*strideA], strideA * SizeOf(rocblas_float_complex), 0);
    for j := 0 to N - 1 do for diag := -KL to KU do
    begin
      row := j + diag;
      if (row >= 0) and (row < M) then
      begin
        hostA[b*strideA + (KU+diag) + j*LDA].x := (b+1)*100 + (row+1)*10 + (j+1);
        hostA[b*strideA + (KU+diag) + j*LDA].y := -((b+1)*100 + (row+1)*10 + (j+1));
      end;
    end;
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0;
      hostX[b*strideX + i].y := -(i + 1.0);
    end;
    for i := 0 to M - 1 do
    begin
      hostY[b*strideY + i].x := i * 3.0;
      hostY[b*strideY + i].y := -(i * 3.0);
    end;
  end;

  for b := 0 to batch_count - 1 do
    for i := 0 to M - 1 do
    begin
      expectedY[b*strideY + i] := hostY[b*strideY + i];
      ComplexMul(expectedY[b*strideY + i], beta);
      temp.x := 0; temp.y := 0;
      for j := 0 to N - 1 do
        if (i >= j-KU) and (i <= j+KL) then
          ComplexFMA(temp, hostA[b*strideA + (KU+(i-j)) + j*LDA], hostX[b*strideX + j], alpha);
      ComplexAdd(expectedY[b*strideY + i], temp);
    end;

  hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex));
  hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex));
  hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex));

  hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_cgbmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_cgbmv_strided_batched(handle, rocblas_operation_none, M, N, KL, KU,
                                          @alpha, devA, LDA, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-5, Format('CGBMV_strided_batched Y[%d].x',[i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-5, Format('CGBMV_strided_batched Y[%d].y',[i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_ZGBMV_strided_batched;
const
  M = 5; N = 6; KL = 2; KU = 1; batch_count = 2;
  alpha: rocblas_double_complex = (x: 1.0; y: -0.5);
  beta:  rocblas_double_complex = (x: -0.5; y: 1.0);
  LDA = 4; // KL + KU + 1
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of rocblas_double_complex;
  strideA, strideX, strideY: Int64;
  i, j, b, diag, row: Integer;
  temp: rocblas_double_complex;
begin
  strideA := LDA * N;
  strideX := N;
  strideY := M;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  for b := 0 to batch_count - 1 do
  begin
    FillChar(hostA[b*strideA], strideA * SizeOf(rocblas_double_complex), 0);
    for j := 0 to N - 1 do for diag := -KL to KU do
    begin
      row := j + diag;
      if (row >= 0) and (row < M) then
      begin
        hostA[b*strideA + (KU+diag) + j*LDA].x := (b+1)*100 + (row+1)*10 + (j+1);
        hostA[b*strideA + (KU+diag) + j*LDA].y := -((b+1)*100 + (row+1)*10 + (j+1));
      end;
    end;
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0;
      hostX[b*strideX + i].y := -(i + 1.0);
    end;
    for i := 0 to M - 1 do
    begin
      hostY[b*strideY + i].x := i * 3.0;
      hostY[b*strideY + i].y := -(i * 3.0);
    end;
  end;

  for b := 0 to batch_count - 1 do
    for i := 0 to M - 1 do
    begin
      expectedY[b*strideY + i] := hostY[b*strideY + i];
      ComplexMul(expectedY[b*strideY + i], beta);
      temp.x := 0; temp.y := 0;
      for j := 0 to N - 1 do
        if (i >= j-KU) and (i <= j+KL) then
          ComplexFMA(temp, hostA[b*strideA + (KU+(i-j)) + j*LDA], hostX[b*strideX + j], alpha);
      ComplexAdd(expectedY[b*strideY + i], temp);
    end;

  hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex));
  hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex));
  hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex));

  hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zgbmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_zgbmv_strided_batched(handle, rocblas_operation_none, M, N, KL, KU,
                                          @alpha, devA, LDA, strideA,
                                          devX, 1, strideX,
                                          @beta, devY, 1, strideY,
                                          batch_count)));

  hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-12, Format('ZGBMV_strided_batched Y[%d].x',[i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-12, Format('ZGBMV_strided_batched Y[%d].y',[i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;


procedure TTestRocBLAS_Level2_strided_batched.Test_CHEMV_strided_batched;
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
  AssertEquals('rocblas_chemv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_chemv_strided_batched(handle, uplo, N, @alpha, devA, N, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devY D2H', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-5, Format('CHEMV_strided_batched Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-5, Format('CHEMV_strided_batched Y[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_ZHEMV_strided_batched;
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
  AssertEquals('rocblas_zhemv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_zhemv_strided_batched(handle, uplo, N, @alpha, devA, N, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back to host
  AssertEquals('hipMemcpy devY D2H', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-12, Format('ZHEMV_strided_batched Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-12, Format('ZHEMV_strided_batched Y[%d].y', [i]));
  end;

  // Cleanup
  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CHBMV_strided_batched;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  LDA = K + 1;
  alpha: rocblas_float_complex = (x: 2.0; y: -1.0);
  beta:  rocblas_float_complex = (x: -1.0; y: 1.5);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of rocblas_float_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  temp, sum_prod: rocblas_float_complex;
  A_ij: rocblas_float_complex;
begin
  strideA := LDA * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Initialize Hermitian Banded Matrix A (Upper)
    for j := 0 to N - 1 do
      for i := 0 to LDA - 1 do
        hostA[b*strideA + j*LDA + i].x := 0; // zero out
    for j := 0 to N - 1 do
    begin
      for i := Max(0, j - K) to j do
      begin
        if (K + i - j) < LDA then
        begin
          hostA[b*strideA + j*LDA + (K + i - j)].x := (b+1)*10 + i + j + 2;
          if i = j then
             hostA[b*strideA + j*LDA + (K + i - j)].y := 0.0 // Real diagonal
          else
             hostA[b*strideA + j*LDA + (K + i - j)].y := (b+1) + i - j;
        end;
      end;
    end;
    // Initialize vectors
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0; hostX[b*strideX + i].y := 2.0 - i;
      hostY[b*strideY + i].x := i * 1.5; hostY[b*strideY + i].y := i * -1.0;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      expectedY[b*strideY + i] := hostY[b*strideY + i];
      ComplexMul(expectedY[b*strideY + i], beta);
      sum_prod.x := 0; sum_prod.y := 0;

      for j := Max(0, i-K) to Min(N-1, i+K) do
      begin
         if j >= i then // Upper triangular part A(i,j)
         begin
            A_ij := hostA[b*strideA + j*LDA + (K + i - j)];
         end
         else // Lower triangular part A(i,j) = conj(A(j,i))
         begin
            A_ij := hostA[b*strideA + i*LDA + (K + j - i)];
            A_ij.y := -A_ij.y; // Conjugate
         end;
         temp := hostX[b*strideX + j];
         ComplexMul(temp, A_ij);
         ComplexAdd(sum_prod, temp);
      end;
      ComplexMul(sum_prod, alpha);
      ComplexAdd(expectedY[b*strideY + i], sum_prod);
    end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_chbmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_chbmv_strided_batched(handle, uplo, N, K, @alpha, devA, LDA, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-5, Format('CHBMV_strided_batched Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-5, Format('CHBMV_strided_batched Y[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_ZHBMV_strided_batched;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  LDA = K + 1;
  alpha: rocblas_double_complex = (x: 2.0; y: -1.0);
  beta:  rocblas_double_complex = (x: -1.0; y: 1.5);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of rocblas_double_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  temp, sum_prod: rocblas_double_complex;
  A_ij: rocblas_double_complex;
begin
  strideA := LDA * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Initialize Hermitian Banded Matrix A (Upper)
    for j := 0 to N - 1 do
      for i := 0 to LDA - 1 do
        hostA[b*strideA + j*LDA + i].x := 0; // zero out
    for j := 0 to N - 1 do
    begin
      for i := Max(0, j - K) to j do
      begin
        if (K + i - j) < LDA then
        begin
          hostA[b*strideA + j*LDA + (K + i - j)].x := (b+1)*10 + i + j + 2;
          if i = j then
             hostA[b*strideA + j*LDA + (K + i - j)].y := 0.0 // Real diagonal
          else
             hostA[b*strideA + j*LDA + (K + i - j)].y := (b+1) + i - j;
        end;
      end;
    end;
    // Initialize vectors
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0; hostX[b*strideX + i].y := 2.0 - i;
      hostY[b*strideY + i].x := i * 1.5; hostY[b*strideY + i].y := i * -1.0;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      expectedY[b*strideY + i] := hostY[b*strideY + i];
      ComplexMul(expectedY[b*strideY + i], beta);
      sum_prod.x := 0; sum_prod.y := 0;

      for j := Max(0, i-K) to Min(N-1, i+K) do
      begin
         if j >= i then // Upper triangular part A(i,j)
         begin
            A_ij := hostA[b*strideA + j*LDA + (K + i - j)];
         end
         else // Lower triangular part A(i,j) = conj(A(j,i))
         begin
            A_ij := hostA[b*strideA + i*LDA + (K + j - i)];
            A_ij.y := -A_ij.y; // Conjugate
         end;
         temp := hostX[b*strideX + j];
         ComplexMul(temp, A_ij);
         ComplexAdd(sum_prod, temp);
      end;
      ComplexMul(sum_prod, alpha);
      ComplexAdd(expectedY[b*strideY + i], sum_prod);
    end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_zhbmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_zhbmv_strided_batched(handle, uplo, N, K, @alpha, devA, LDA, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-12, Format('ZHBMV_strided_batched Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-12, Format('ZHBMV_strided_batched Y[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CHPMV_strided_batched;
const
  N = 4;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: rocblas_float_complex = (x: 1.5; y: -1.0);
  beta:  rocblas_float_complex = (x: 0.5; y: 2.0);
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostX, hostY, hostY_result, expectedY: array of rocblas_float_complex;
  fullA: array of rocblas_float_complex; // For reference calculation
  packed_size, strideAP, strideX, strideY: Int64;
  i, j, b, k: Integer;
  temp, sum_prod: rocblas_float_complex;
  A_ij: rocblas_float_complex;
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
      expectedY[b*strideY + i] := hostY[b*strideY + i];
      ComplexMul(expectedY[b*strideY + i], beta);

      sum_prod.x := 0; sum_prod.y := 0;
      for j := 0 to N - 1 do
      begin
        A_ij := fullA[b*N*N + j*N + i]; // A(i,j)
        temp := hostX[b*strideX + j];   // x(j)
        ComplexMul(temp, A_ij);         // temp = A(i,j) * x(j)
        ComplexAdd(sum_prod, temp);     // sum += temp
      end;

      ComplexMul(sum_prod, alpha);
      ComplexAdd(expectedY[b*strideY + i], sum_prod);
    end;

  // Allocate and copy to device
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy devAP H2D', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX H2D', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY H2D', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_chpmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_chpmv_strided_batched(handle, uplo, N, @alpha, devAP, strideAP, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy devY D2H', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-5, Format('CHPMV_strided_batched Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-5, Format('CHPMV_strided_batched Y[%d].y', [i]));
  end;

  // Cleanup
  hipFree(devAP);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_ZHPMV_strided_batched;
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
      expectedY[b*strideY + i] := hostY[b*strideY + i];
      ComplexMul(expectedY[b*strideY + i], beta);

      sum_prod.x := 0; sum_prod.y := 0;
      for j := 0 to N - 1 do
      begin
        A_ij := fullA[b*N*N + j*N + i]; // A(i,j)
        temp := hostX[b*strideX + j];   // x(j)
        ComplexMul(temp, A_ij);         // temp = A(i,j) * x(j)
        ComplexAdd(sum_prod, temp);     // sum += temp
      end;

      ComplexMul(sum_prod, alpha);
      ComplexAdd(expectedY[b*strideY + i], sum_prod);
    end;

  // Allocate and copy to device
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy devAP H2D', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devX H2D', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devY H2D', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_zhpmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_zhpmv_strided_batched(handle, uplo, N, @alpha, devAP, strideAP, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy devY D2H', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY_result[i].x, 1e-12, Format('ZHPMV_strided_batched Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY_result[i].y, 1e-12, Format('ZHPMV_strided_batched Y[%d].y', [i]));
  end;

  hipFree(devAP);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_SSYMV_strided_batched;
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
  AssertEquals('rocblas_ssymv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ssymv_strided_batched(handle, uplo, N, @alpha, devA, N, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-5, Format('SSYMV_strided_batched Y[%d]', [i]));

  // Cleanup
  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_DSYMV_strided_batched;
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
          hostA[b*strideA + j*N + i] := 0.0; // This part is not referenced

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
  AssertEquals('rocblas_dsymv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_dsymv_strided_batched(handle, uplo, N, @alpha, devA, N, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  // Copy result back and verify
  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-12, Format('DSYMV_strided_batched Y[%d]', [i]));

  // Cleanup
  hipFree(devA); hipFree(devX); hipFree(devY);
end;

//procedure TTestRocBLAS_Level2_strided_batched.Test_CSYMV_strided_batched;
//begin
//  WriteLn('Function CSYMV_strided_batched does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_strided_batched.Test_ZSYMV_strided_batched;
//begin
//  WriteLn('Function ZSYMV_strided_batched does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_strided_batched.Test_SSBMV_strided_batched;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  LDA = K + 1;
  alpha: Single = 1.5;
  beta:  Single = -0.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of Single;
  fullA: array of Single; // For reference calculation
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
begin
  strideA := LDA * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);
  SetLength(fullA, N * N * batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full symmetric matrix (banded part only) for easy reference
    for j := 0 to N - 1 do for i := 0 to N - 1 do fullA[b*N*N + j*N + i] := 0.0;
    for j := 0 to N - 1 do
      for i := Max(0, j - K) to j do
      begin
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j + 2;
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i]; // Symmetric
      end;

    // 2. Pack the upper band into hostA format
    FillChar(hostA[b*strideA], strideA * SizeOf(Single), 0);
    for j := 0 to N-1 do
      for i := Max(0, j-K) to j do
        hostA[b*strideA + j*LDA + (K + i - j)] := fullA[b*N*N + j*N + i];

    // 3. Initialize vectors
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
      hostY[b*strideY + i] := i * 2.0;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      expectedY[b*strideY + i] := beta * hostY[b*strideY + i];
      for j := 0 to N - 1 do
        expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
    end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ssbmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ssbmv_strided_batched(handle, uplo, N, K, @alpha, devA, LDA, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-5, Format('SSBMV_strided_batched Y[%d]', [i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_DSBMV_strided_batched;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  LDA = K + 1;
  alpha: Double = 1.5;
  beta:  Double = -0.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostX, hostY, hostY_result, expectedY: array of Double;
  fullA: array of Double; // For reference calculation
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
begin
  strideA := LDA * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(hostY_result, strideY * batch_count);
  SetLength(expectedY, strideY * batch_count);
  SetLength(fullA, N * N * batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full symmetric matrix (banded part only) for easy reference
    for j := 0 to N - 1 do for i := 0 to N - 1 do fullA[b*N*N + j*N + i] := 0.0;
    for j := 0 to N - 1 do
      for i := Max(0, j - K) to j do
      begin
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j + 2;
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i]; // Symmetric
      end;

    // 2. Pack the upper band into hostA format
    FillChar(hostA[b*strideA], strideA * SizeOf(Double), 0);
    for j := 0 to N-1 do
      for i := Max(0, j-K) to j do
        hostA[b*strideA + j*LDA + (K + i - j)] := fullA[b*N*N + j*N + i];

    // 3. Initialize vectors
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 1.0;
      hostY[b*strideY + i] := i * 2.0;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      expectedY[b*strideY + i] := beta * hostY[b*strideY + i];
      for j := 0 to N - 1 do
        expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
    end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dsbmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_dsbmv_strided_batched(handle, uplo, N, K, @alpha, devA, LDA, strideA, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-12, Format('DSBMV_strided_batched Y[%d]', [i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_SSPMV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Single = 1.5;
  beta:  Single = 2.5;
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostX, hostY, hostY_result, expectedY: array of Single;
  fullA: array of Single; // For reference calculation
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
  SetLength(fullA, N * N * batch_count);

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Create full symmetric matrix for reference
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j;
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i];
      end;

    // 2. Pack upper part into hostAP
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        inc(k);
      end;

    // 3. Initialize vectors
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 0.5;
      hostY[b*strideY + i] := i * 1.5;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      expectedY[b*strideY + i] := beta * hostY[b*strideY + i];
      for j := 0 to N - 1 do
        expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
    end;

  // GPU operations
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_sspmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_sspmv_strided_batched(handle, uplo, N, @alpha, devAP, strideAP, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-5, Format('SSPMV_strided_batched Y[%d]', [i]));

  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_DSPMV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  alpha: Double = 1.5;
  beta:  Double = 2.5;
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostX, hostY, hostY_result, expectedY: array of Double;
  fullA: array of Double; // For reference calculation
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
  SetLength(fullA, N * N * batch_count);

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Create full symmetric matrix for reference
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j;
        fullA[b*N*N + i*N + j] := fullA[b*N*N + j*N + i];
      end;

    // 2. Pack upper part into hostAP
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        inc(k);
      end;

    // 3. Initialize vectors
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i] := i + 0.5;
      hostY[b*strideY + i] := i * 1.5;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      expectedY[b*strideY + i] := beta * hostY[b*strideY + i];
      for j := 0 to N - 1 do
        expectedY[b*strideY + i] := expectedY[b*strideY + i] + alpha * fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
    end;

  // GPU operations
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dspmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_dspmv_strided_batched(handle, uplo, N, @alpha, devAP, strideAP, devX, 1, strideX, @beta, devY, 1, strideY, batch_count)));

  AssertEquals('hipMemcpy D2H Y', HIP_SUCCESS, hipMemcpy(@hostY_result[0], devY, strideY * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideY * batch_count) - 1 do
    CheckEqualsRelative(expectedY[i], hostY_result[i], 1e-12, Format('DSPMV_strided_batched Y[%d]', [i]));

  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_STRMV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostX_result, expectedX: array of Single;
  tempX: array of Single; // for reference calculation
  strideA, strideX: Int64;
  i, j, b: Integer;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Initialize upper triangular matrix A
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
          hostA[b*strideA + j*N + i] := (b+1)*10 + i + j + 1.0
        else
          hostA[b*strideA + j*N + i] := 0.0; // Not referenced

    // Initialize vector x
    for i := 0 to N - 1 do
      hostX[b*strideX + i] := i + 1.0;
  end;

  // CPU reference calculation: x_new = A * x_old
  for b := 0 to batch_count - 1 do
  begin
    // Calculate one batch result into tempX
    for i := 0 to N - 1 do
    begin
      tempX[i] := 0.0;
      for j := i to N - 1 do // Upper triangular
        tempX[i] := tempX[i] + hostA[b*strideA + j*N + i] * hostX[b*strideX + j];
    end;
    // Copy the result to the final expected array
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_strmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_strmv_strided_batched(handle, uplo, transA, diag, N, devA, N, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-5, Format('STRMV_strided_batched X[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_DTRMV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostX_result, expectedX: array of Double;
  tempX: array of Double; // for reference calculation
  strideA, strideX: Int64;
  i, j, b: Integer;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Initialize upper triangular matrix A
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
          hostA[b*strideA + j*N + i] := (b+1)*10 + i + j + 1.0
        else
          hostA[b*strideA + j*N + i] := 0.0; // Not referenced

    // Initialize vector x
    for i := 0 to N - 1 do
      hostX[b*strideX + i] := i + 1.0;
  end;

  // CPU reference calculation: x_new = A * x_old
  for b := 0 to batch_count - 1 do
  begin
    // Calculate one batch result into tempX
    for i := 0 to N - 1 do
    begin
      tempX[i] := 0.0;
      for j := i to N - 1 do // Upper triangular
        tempX[i] := tempX[i] + hostA[b*strideA + j*N + i] * hostX[b*strideX + j];
    end;
    // Copy the result to the final expected array
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dtrmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_dtrmv_strided_batched(handle, uplo, transA, diag, N, devA, N, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTRMV_strided_batched X[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CTRMV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostX_result, expectedX: array of rocblas_float_complex;
  tempX: array of rocblas_float_complex; // for reference calculation
  strideA, strideX: Int64;
  i, j, b: Integer;
  prod: rocblas_float_complex;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Initialize upper triangular matrix A
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
        begin
          hostA[b*strideA + j*N + i].x := (b+1)*10 + i + j + 1.0;
          hostA[b*strideA + j*N + i].y := (b+1) + i - j - 0.5;
        end
        else
        begin
          hostA[b*strideA + j*N + i].x := 0.0; hostA[b*strideA + j*N + i].y := 0.0;
        end;
    // Initialize vector x
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0;
      hostX[b*strideX + i].y := 1.0 - i;
    end;
  end;

  // CPU reference calculation: x_new = A * x_old
  for b := 0 to batch_count - 1 do
  begin
    // Calculate one batch result into tempX
    for i := 0 to N - 1 do
    begin
      tempX[i].x := 0.0; tempX[i].y := 0.0;
      for j := i to N - 1 do // Upper triangular
      begin
        prod := hostA[b*strideA + j*N + i];
        ComplexMul(prod, hostX[b*strideX + j]);
        ComplexAdd(tempX[i], prod);
      end;
    end;
    // Copy the result to the final expected array
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ctrmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ctrmv_strided_batched(handle, uplo, transA, diag, N, devA, N, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-5, Format('CTRMV_strided_batched X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-5, Format('CTRMV_strided_batched X[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_ZTRMV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostX_result, expectedX: array of rocblas_double_complex;
  tempX: array of rocblas_double_complex; // for reference calculation
  strideA, strideX: Int64;
  i, j, b: Integer;
  prod: rocblas_double_complex;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Initialize upper triangular matrix A
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
        begin
          hostA[b*strideA + j*N + i].x := (b+1)*10 + i + j + 1.0;
          hostA[b*strideA + j*N + i].y := (b+1) + i - j - 0.5;
        end
        else
        begin
          hostA[b*strideA + j*N + i].x := 0.0; hostA[b*strideA + j*N + i].y := 0.0;
        end;
    // Initialize vector x
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0;
      hostX[b*strideX + i].y := 1.0 - i;
    end;
  end;

  // CPU reference calculation: x_new = A * x_old
  for b := 0 to batch_count - 1 do
  begin
    // Calculate one batch result into tempX
    for i := 0 to N - 1 do
    begin
      tempX[i].x := 0.0; tempX[i].y := 0.0;
      for j := i to N - 1 do // Upper triangular
      begin
        prod := hostA[b*strideA + j*N + i];
        ComplexMul(prod, hostX[b*strideX + j]);
        ComplexAdd(tempX[i], prod);
      end;
    end;
    // Copy the result to the final expected array
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ztrmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ztrmv_strided_batched(handle, uplo, transA, diag, N, devA, N, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTRMV_strided_batched X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTRMV_strided_batched X[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_STBMV_strided_batched;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  LDA = K + 1;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostX_result, expectedX, tempX: array of Single;
  fullA: array of Single; // For reference calculation
  strideA, strideX: Int64;
  i, j, b: Integer;
begin
  strideA := LDA * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full upper triangular banded matrix for reference
    for j := 0 to N - 1 do for i := 0 to N - 1 do fullA[b*N*N + j*N + i] := 0.0;
    for j := 0 to N - 1 do
      for i := Max(0, j - K) to j do
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j + 1.0;

    // 2. Pack the upper band into hostA format
    for j := 0 to N - 1 do
      for i := Max(0, j-K) to j do
        hostA[b*strideA + j*LDA + (K + i - j)] := fullA[b*N*N + j*N + i];

    // 3. Initialize vector x
    for i := 0 to N - 1 do
      hostX[b*strideX + i] := i + 1.0;
  end;

  // CPU reference calculation: x_new = A * x_old
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      tempX[i] := 0.0;
      for j := i to Min(N - 1, i + K) do
        tempX[i] := tempX[i] + fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
    end;
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_stbmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_stbmv_strided_batched(handle, uplo, transA, diag, N, K, devA, LDA, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-5, Format('STBMV_strided_batched X[%d]', [i]));

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_DTBMV_strided_batched;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  LDA = K + 1;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostX_result, expectedX, tempX: array of Double;
  fullA: array of Double; // For reference calculation
  strideA, strideX: Int64;
  i, j, b: Integer;
begin
  strideA := LDA * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize a full upper triangular banded matrix for reference
    for j := 0 to N - 1 do for i := 0 to N - 1 do fullA[b*N*N + j*N + i] := 0.0;
    for j := 0 to N - 1 do
      for i := Max(0, j - K) to j do
        fullA[b*N*N + j*N + i] := (b+1)*10 + i + j + 1.0;

    // 2. Pack the upper band into hostA format
    for j := 0 to N - 1 do
      for i := Max(0, j-K) to j do
        hostA[b*strideA + j*LDA + (K + i - j)] := fullA[b*N*N + j*N + i];

    // 3. Initialize vector x
    for i := 0 to N - 1 do
      hostX[b*strideX + i] := i + 1.0;
  end;

  // CPU reference calculation: x_new = A * x_old
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      tempX[i] := 0.0;
      for j := i to Min(N - 1, i + K) do
        tempX[i] := tempX[i] + fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
    end;
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dtbmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_dtbmv_strided_batched(handle, uplo, transA, diag, N, K, devA, LDA, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTBMV_strided_batched X[%d]', [i]));

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CTBMV_strided_batched;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  LDA = K + 1;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostX_result, expectedX, tempX: array of rocblas_float_complex;
  fullA: array of rocblas_float_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
  prod: rocblas_float_complex;
begin
  strideA := LDA * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do for i := 0 to N - 1 do begin fullA[b*N*N + j*N + i].x:=0; fullA[b*N*N + j*N + i].y:=0; end;
    for j := 0 to N - 1 do
      for i := Max(0, j - K) to j do
      begin
        fullA[b*N*N + j*N + i].x := (b+1)*10 + i + j + 1.0;
        fullA[b*N*N + j*N + i].y := (b+1) + i - j;
      end;
    for j := 0 to N - 1 do
      for i := Max(0, j-K) to j do
        hostA[b*strideA + j*LDA + (K + i - j)] := fullA[b*N*N + j*N + i];
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0; hostX[b*strideX + i].y := 1.0 - i;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      tempX[i].x := 0.0; tempX[i].y := 0.0;
      for j := i to Min(N - 1, i + K) do
      begin
        prod := fullA[b*N*N + j*N + i];
        ComplexMul(prod, hostX[b*strideX + j]);
        ComplexAdd(tempX[i], prod);
      end;
    end;
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ctbmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ctbmv_strided_batched(handle, uplo, transA, diag, N, K, devA, LDA, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-5, Format('CTBMV_strided_batched X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-5, Format('CTBMV_strided_batched X[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_ZTBMV_strided_batched;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  LDA = K + 1;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostX, hostX_result, expectedX, tempX: array of rocblas_double_complex;
  fullA: array of rocblas_double_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
  prod: rocblas_double_complex;
begin
  strideA := LDA * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do for i := 0 to N - 1 do begin fullA[b*N*N + j*N + i].x:=0; fullA[b*N*N + j*N + i].y:=0; end;
    for j := 0 to N - 1 do
      for i := Max(0, j - K) to j do
      begin
        fullA[b*N*N + j*N + i].x := (b+1)*10 + i + j + 1.0;
        fullA[b*N*N + j*N + i].y := (b+1) + i - j;
      end;
    for j := 0 to N - 1 do
      for i := Max(0, j-K) to j do
        hostA[b*strideA + j*LDA + (K + i - j)] := fullA[b*N*N + j*N + i];
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0; hostX[b*strideX + i].y := 1.0 - i;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      tempX[i].x := 0.0; tempX[i].y := 0.0;
      for j := i to Min(N - 1, i + K) do
      begin
        prod := fullA[b*N*N + j*N + i];
        ComplexMul(prod, hostX[b*strideX + j]);
        ComplexAdd(tempX[i], prod);
      end;
    end;
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ztbmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ztbmv_strided_batched(handle, uplo, transA, diag, N, K, devA, LDA, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTBMV_strided_batched X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTBMV_strided_batched X[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_STPMV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostX, hostX_result, expectedX: array of Single;
  fullA: array of Single;
  tempX: array of Single;
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
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Create a full upper triangular matrix for reference
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
          fullA[b*N*N + j*N + i] := (b+1)*5 + i + j + 1.0
        else
          fullA[b*N*N + j*N + i] := 0.0;

    // 2. Pack the upper part into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vector x
    for i := 0 to N - 1 do
      hostX[b*strideX + i] := i + 1.0;
  end;

  // CPU reference calculation: x_new = A * x_old
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      tempX[i] := 0.0;
      for j := 0 to N - 1 do
        tempX[i] := tempX[i] + fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
    end;
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_stpmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_stpmv_strided_batched(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-5, Format('STPMV_strided_batched X[%d]', [i]));

  hipFree(devAP);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_DTPMV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostX, hostX_result, expectedX: array of Double;
  fullA: array of Double;
  tempX: array of Double;
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
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Create a full upper triangular matrix for reference
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
          fullA[b*N*N + j*N + i] := (b+1)*5 + i + j + 1.0
        else
          fullA[b*N*N + j*N + i] := 0.0;

    // 2. Pack the upper part into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vector x
    for i := 0 to N - 1 do
      hostX[b*strideX + i] := i + 1.0;
  end;

  // CPU reference calculation: x_new = A * x_old
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      tempX[i] := 0.0;
      for j := 0 to N - 1 do
        tempX[i] := tempX[i] + fullA[b*N*N + j*N + i] * hostX[b*strideX + j];
    end;
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dtpmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_dtpmv_strided_batched(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTPMV_strided_batched X[%d]', [i]));

  hipFree(devAP);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CTPMV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostX, hostX_result, expectedX: array of rocblas_float_complex;
  fullA: array of rocblas_float_complex;
  tempX: array of rocblas_float_complex;
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
  prod: rocblas_float_complex;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Create a full upper triangular matrix for reference
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
        begin
          fullA[b*N*N + j*N + i].x := (b+1)*5 + i + j + 1.0;
          fullA[b*N*N + j*N + i].y := (b+1) + i - j;
        end
        else
        begin
          fullA[b*N*N + j*N + i].x := 0.0; fullA[b*N*N + j*N + i].y := 0.0;
        end;
    // 2. Pack it
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;
    // 3. Init vector
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0; hostX[b*strideX + i].y := 1.0 - i;
    end;
  end;

  // CPU reference
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      tempX[i].x := 0.0; tempX[i].y := 0.0;
      for j := 0 to N - 1 do
      begin
        prod := fullA[b*N*N + j*N + i];
        ComplexMul(prod, hostX[b*strideX + j]);
        ComplexAdd(tempX[i], prod);
      end;
    end;
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ctpmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ctpmv_strided_batched(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-5, Format('CTPMV_strided_batched X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-5, Format('CTPMV_strided_batched X[%d].y', [i]));
  end;

  hipFree(devAP);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_ZTPMV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostX, hostX_result, expectedX: array of rocblas_double_complex;
  fullA: array of rocblas_double_complex;
  tempX: array of rocblas_double_complex;
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
  prod: rocblas_double_complex;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Create a full upper triangular matrix for reference
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
        begin
          fullA[b*N*N + j*N + i].x := (b+1)*5 + i + j + 1.0;
          fullA[b*N*N + j*N + i].y := (b+1) + i - j;
        end
        else
        begin
          fullA[b*N*N + j*N + i].x := 0.0; fullA[b*N*N + j*N + i].y := 0.0;
        end;
    // 2. Pack it
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;
    // 3. Init vector
    for i := 0 to N - 1 do
    begin
      hostX[b*strideX + i].x := i + 1.0; hostX[b*strideX + i].y := 1.0 - i;
    end;
  end;

  // CPU reference
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      tempX[i].x := 0.0; tempX[i].y := 0.0;
      for j := 0 to N - 1 do
      begin
        prod := fullA[b*N*N + j*N + i];
        ComplexMul(prod, hostX[b*strideX + j]);
        ComplexAdd(tempX[i], prod);
      end;
    end;
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ztpmv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ztpmv_strided_batched(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTPMV_strided_batched X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTPMV_strided_batched X[%d].y', [i]));
  end;

  hipFree(devAP);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_STRSV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX_result, expectedX: array of Single;
  tempX: array of Single; // for reference calculation
  strideA, strideX: Int64;
  i, j, b: Integer;
  sum: Single;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count); // This is vector 'b'
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Initialize upper triangular matrix A (non-singular)
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
          hostA[b*strideA + j*N + i] := (b+1)*2 + i + j + 1.0
        else
          hostA[b*strideA + j*N + i] := 0.0;

    // Initialize vector b
    for i := 0 to N - 1 do
      hostB[b*strideX + i] := (b+1)*10 + i * 2.0;
  end;

  // CPU reference calculation: solve A*x = b for x using back-substitution
  for b := 0 to batch_count - 1 do
  begin
    for i := N - 1 downto 0 do
    begin
      sum := 0.0;
      for j := i + 1 to N - 1 do
      begin
        sum := sum + hostA[b*strideA + j*N + i] * tempX[j];
      end;
      tempX[i] := (hostB[b*strideX + i] - sum) / hostA[b*strideA + i*N + i];
    end;
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X (b)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_strsv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_strsv_strided_batched(handle, uplo, transA, diag, N, devA, N, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-4, Format('STRSV_strided_batched X[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_DTRSV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX_result, expectedX: array of Double;
  tempX: array of Double;
  strideA, strideX: Int64;
  i, j, b: Integer;
  sum: Double;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
          hostA[b*strideA + j*N + i] := (b+1)*2 + i + j + 1.0
        else
          hostA[b*strideA + j*N + i] := 0.0;
    for i := 0 to N - 1 do
      hostB[b*strideX + i] := (b+1)*10 + i * 2.0;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    for i := N - 1 downto 0 do
    begin
      sum := 0.0;
      for j := i + 1 to N - 1 do
      begin
        sum := sum + hostA[b*strideA + j*N + i] * tempX[j];
      end;
      tempX[i] := (hostB[b*strideX + i] - sum) / hostA[b*strideA + i*N + i];
    end;
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X (b)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dtrsv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_dtrsv_strided_batched(handle, uplo, transA, diag, N, devA, N, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTRSV_strided_batched X[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CTRSV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX_result, expectedX: array of rocblas_float_complex;
  tempX: array of rocblas_float_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
  sum, prod, num, den: rocblas_float_complex;

begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(tempX, N);

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do for i := 0 to N-1 do
      if i <= j then
      begin
        hostA[b*strideA + j*N + i].x := (b+1)*2 + i + j + 1;
        hostA[b*strideA + j*N + i].y := (i-j)*0.5;
      end else
      begin
        hostA[b*strideA + j*N + i].x := 0; hostA[b*strideA + j*N + i].y := 0;
      end;
    for i := 0 to N-1 do
    begin
      hostB[b*strideX + i].x := (b+1)*10 + i*2; hostB[b*strideX + i].y := (b+1)*5 - i;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    for i := N - 1 downto 0 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for j := i + 1 to N - 1 do
      begin
        prod := hostA[b*strideA + j*N + i];
        ComplexMul(prod, tempX[j]);
        ComplexAdd(sum, prod);
      end;
      num := hostB[b*strideX + i];
      ComplexSub(num, sum);
      den := hostA[b*strideA + i*N + i];
      ComplexDiv(num, den);
      tempX[i] := num;
    end;
    for i := 0 to N-1 do expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X (b)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ctrsv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ctrsv_strided_batched(handle, uplo, transA, diag, N, devA, N, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-5, Format('CTRSV_strided_batched X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-5, Format('CTRSV_strided_batched X[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_ZTRSV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX_result, expectedX: array of rocblas_double_complex;
  tempX: array of rocblas_double_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
  sum, prod, num, den: rocblas_double_complex;
begin
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(tempX, N);

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do for i := 0 to N-1 do
      if i <= j then
      begin
        hostA[b*strideA + j*N + i].x := (b+1)*2 + i + j + 1;
        hostA[b*strideA + j*N + i].y := (i-j)*0.5;
      end else
      begin
        hostA[b*strideA + j*N + i].x := 0; hostA[b*strideA + j*N + i].y := 0;
      end;
    for i := 0 to N-1 do
    begin
      hostB[b*strideX + i].x := (b+1)*10 + i*2; hostB[b*strideX + i].y := (b+1)*5 - i;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    for i := N - 1 downto 0 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for j := i + 1 to N - 1 do
      begin
        prod := hostA[b*strideA + j*N + i];
        ComplexMul(prod, tempX[j]);
        ComplexAdd(sum, prod);
      end;
      num := hostB[b*strideX + i];
      ComplexSub(num, sum);
      den := hostA[b*strideA + i*N + i];
      ComplexDiv(num, den);
      tempX[i] := num;
    end;
    for i := 0 to N-1 do expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X (b)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ztrsv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ztrsv_strided_batched(handle, uplo, transA, diag, N, devA, N, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTRSV_strided_batched X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTRSV_strided_batched X[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_STBSV_strided_batched;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  LDA = K + 1;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX_result, expectedX: array of Single;
  fullA: array of Single; // For reference calculation
  tempX: array of Single;
  strideA, strideX: Int64;
  i, j, b: Integer;
  sum: Single;
begin
  strideA := LDA * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Initialize full upper triangular banded matrix for reference
    for j := 0 to N-1 do for i:= 0 to N-1 do fullA[b*N*N + j*N + i] := 0.0;
    for j := 0 to N - 1 do
      for i := Max(0, j - K) to j do
        fullA[b*N*N + j*N + i] := (b+1)*5 + i + j + 1.0;

    // 2. Pack the upper band into hostA format
    for j := 0 to N - 1 do
      for i := Max(0, j-K) to j do
        hostA[b*strideA + j*LDA + (K + i - j)] := fullA[b*N*N + j*N + i];

    // 3. Initialize vector b
    for i := 0 to N-1 do
      hostB[b*strideX + i] := (b+1)*20 + i*3.0;
  end;

  // CPU reference calculation: solve A*x = b via back-substitution
  for b := 0 to batch_count - 1 do
  begin
    for i := N - 1 downto 0 do
    begin
      sum := 0.0;
      for j := i + 1 to Min(N-1, i+K) do
        sum := sum + fullA[b*N*N + j*N + i] * tempX[j];
      tempX[i] := (hostB[b*strideX + i] - sum) / fullA[b*N*N + i*N + i];
    end;
    for i := 0 to N-1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X(b)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_stbsv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_stbsv_strided_batched(handle, uplo, transA, diag, N, K, devA, LDA, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-4, Format('STBSV_strided_batched X[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_DTBSV_strided_batched;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  LDA = K + 1;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX_result, expectedX: array of Double;
  fullA: array of Double;
  tempX: array of Double;
  strideA, strideX: Int64;
  i, j, b: Integer;
  sum: Double;
begin
  strideA := LDA * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N-1 do for i:= 0 to N-1 do fullA[b*N*N + j*N + i] := 0.0;
    for j := 0 to N - 1 do
      for i := Max(0, j - K) to j do
        fullA[b*N*N + j*N + i] := (b+1)*5 + i + j + 1.0;
    for j := 0 to N - 1 do
      for i := Max(0, j-K) to j do
        hostA[b*strideA + j*LDA + (K + i - j)] := fullA[b*N*N + j*N + i];
    for i := 0 to N-1 do
      hostB[b*strideX + i] := (b+1)*20 + i*3.0;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    for i := N - 1 downto 0 do
    begin
      sum := 0.0;
      for j := i + 1 to Min(N-1, i+K) do
        sum := sum + fullA[b*N*N + j*N + i] * tempX[j];
      tempX[i] := (hostB[b*strideX + i] - sum) / fullA[b*N*N + i*N + i];
    end;
    for i := 0 to N-1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X(b)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dtbsv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_dtbsv_strided_batched(handle, uplo, transA, diag, N, K, devA, LDA, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTBSV_strided_batched X[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CTBSV_strided_batched;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  LDA = K + 1;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX_result, expectedX: array of rocblas_float_complex;
  fullA: array of rocblas_float_complex;
  tempX: array of rocblas_float_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
  sum, prod, num, den: rocblas_float_complex;
begin
  strideA := LDA * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N-1 do for i:= 0 to N-1 do begin fullA[b*N*N + j*N + i].x:=0; fullA[b*N*N + j*N + i].y:=0; end;
    for j := 0 to N - 1 do
      for i := Max(0, j - K) to j do
      begin
        fullA[b*N*N + j*N + i].x := (b+1)*5 + i + j + 1.0;
        fullA[b*N*N + j*N + i].y := (i-j)*0.2;
      end;
    for j := 0 to N - 1 do
      for i := Max(0, j-K) to j do
        hostA[b*strideA + j*LDA + (K + i - j)] := fullA[b*N*N + j*N + i];
    for i := 0 to N-1 do
    begin
      hostB[b*strideX + i].x := (b+1)*20 + i*3.0;
      hostB[b*strideX + i].y := (b+1)*10 - i*2.0;
    end;
  end;

  // CPU reference
  for b := 0 to batch_count - 1 do
  begin
    for i := N - 1 downto 0 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for j := i + 1 to Min(N-1, i+K) do
      begin
        prod := fullA[b*N*N + j*N + i];
        ComplexMul(prod, tempX[j]);
        ComplexAdd(sum, prod);
      end;
      num := hostB[b*strideX + i];
      num.x := num.x - sum.x; num.y := num.y - sum.y;
      den := fullA[b*N*N + i*N + i];
      ComplexDiv(num, den);
      tempX[i] := num;
    end;
    for i := 0 to N-1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X(b)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ctbsv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ctbsv_strided_batched(handle, uplo, transA, diag, N, K, devA, LDA, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-4, Format('CTBSV_strided_batched X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-4, Format('CTBSV_strided_batched X[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_ZTBSV_strided_batched;
const
  N = 6;
  K = 2;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
  LDA = K + 1;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostB, hostX_result, expectedX: array of rocblas_double_complex;
  fullA: array of rocblas_double_complex;
  tempX: array of rocblas_double_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
  sum, prod, num, den: rocblas_double_complex;
begin
  strideA := LDA * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostB, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N-1 do for i:= 0 to N-1 do begin fullA[b*N*N + j*N + i].x:=0; fullA[b*N*N + j*N + i].y:=0; end;
    for j := 0 to N - 1 do
      for i := Max(0, j - K) to j do
      begin
        fullA[b*N*N + j*N + i].x := (b+1)*5 + i + j + 1.0;
        fullA[b*N*N + j*N + i].y := (i-j)*0.2;
      end;
    for j := 0 to N - 1 do
      for i := Max(0, j-K) to j do
        hostA[b*strideA + j*LDA + (K + i - j)] := fullA[b*N*N + j*N + i];
    for i := 0 to N-1 do
    begin
      hostB[b*strideX + i].x := (b+1)*20 + i*3.0;
      hostB[b*strideX + i].y := (b+1)*10 - i*2.0;
    end;
  end;

  // CPU reference
  for b := 0 to batch_count - 1 do
  begin
    for i := N - 1 downto 0 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for j := i + 1 to Min(N-1, i+K) do
      begin
        prod := fullA[b*N*N + j*N + i];
        ComplexMul(prod, tempX[j]);
        ComplexAdd(sum, prod);
      end;
      num := hostB[b*strideX + i];
      num.x := num.x - sum.x; num.y := num.y - sum.y;
      den := fullA[b*N*N + i*N + i];
      ComplexDiv(num, den);
      tempX[i] := num;
    end;
    for i := 0 to N-1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X(b)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ztbsv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ztbsv_strided_batched(handle, uplo, transA, diag, N, K, devA, LDA, strideA, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTBSV_strided_batched X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTBSV_strided_batched X[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_STPSV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostB, hostX_result, expectedX: array of Single;
  fullA: array of Single;
  tempX: array of Single;
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
  sum: Single;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostB, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Create a full upper triangular matrix for reference
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
          fullA[b*N*N + j*N + i] := (b+1)*3 + i + j + 1.0
        else
          fullA[b*N*N + j*N + i] := 0.0;

    // 2. Pack the upper part into hostAP (column-major)
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vector b
    for i := 0 to N - 1 do
      hostB[b*strideX + i] := (b+1)*15 + i * 2.5;
  end;

  // CPU reference calculation: solve A*x = b via back-substitution
  for b := 0 to batch_count - 1 do
  begin
    for i := N - 1 downto 0 do
    begin
      sum := 0.0;
      for j := i + 1 to N - 1 do
        sum := sum + fullA[b*N*N + j*N + i] * tempX[j];
      tempX[i] := (hostB[b*strideX + i] - sum) / fullA[b*N*N + i*N + i];
    end;
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X (b)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_stpsv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_stpsv_strided_batched(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-4, Format('STPSV_strided_batched X[%d]', [i]));

  hipFree(devAP);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_DTPSV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostB, hostX_result, expectedX: array of Double;
  fullA: array of Double;
  tempX: array of Double;
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
  sum: Double;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostB, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    // 1. Create a full upper triangular matrix for reference
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
          fullA[b*N*N + j*N + i] := (b+1)*3 + i + j + 1.0
        else
          fullA[b*N*N + j*N + i] := 0.0;

    // 2. Pack the upper part into hostAP
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;

    // 3. Initialize vector b
    for i := 0 to N - 1 do
      hostB[b*strideX + i] := (b+1)*15 + i * 2.5;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    for i := N - 1 downto 0 do
    begin
      sum := 0.0;
      for j := i + 1 to N - 1 do
        sum := sum + fullA[b*N*N + j*N + i] * tempX[j];
      tempX[i] := (hostB[b*strideX + i] - sum) / fullA[b*N*N + i*N + i];
    end;
    for i := 0 to N - 1 do
      expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X (b)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dtpsv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_dtpsv_strided_batched(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTPSV_strided_batched X[%d]', [i]));

  hipFree(devAP);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CTPSV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostB, hostX_result, expectedX: array of rocblas_float_complex;
  fullA: array of rocblas_float_complex;
  tempX: array of rocblas_float_complex;
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
  sum, prod, num, den: rocblas_float_complex;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostB, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
        begin
          fullA[b*N*N + j*N + i].x := (b+1)*3 + i + j + 1;
          fullA[b*N*N + j*N + i].y := (i-j)*0.3;
        end else
        begin
          fullA[b*N*N + j*N + i].x := 0; fullA[b*N*N + j*N + i].y := 0;
        end;
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;
    for i := 0 to N - 1 do
    begin
      hostB[b*strideX + i].x := (b+1)*15 + i * 2.5;
      hostB[b*strideX + i].y := (b+1)*8 - i * 1.5;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    for i := N - 1 downto 0 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for j := i + 1 to N - 1 do
      begin
        prod := fullA[b*N*N + j*N + i];
        ComplexMul(prod, tempX[j]);
        ComplexAdd(sum, prod);
      end;
      num := hostB[b*strideX + i];
      num.x := num.x - sum.x; num.y := num.y - sum.y;
      den := fullA[b*N*N + i*N + i];
      ComplexDiv(num, den);
      tempX[i] := num;
    end;
    for i := 0 to N-1 do expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X(b)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ctpsv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ctpsv_strided_batched(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 2e-4, Format('CTPSV_strided_batched X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 2e-4, Format('CTPSV_strided_batched X[%d].y', [i]));
  end;

  hipFree(devAP);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_ZTPSV_strided_batched;
const
  N = 5;
  batch_count = 2;
  uplo = rocblas_fill_upper;
  transA = rocblas_operation_none;
  diag = rocblas_diagonal_non_unit;
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostB, hostX_result, expectedX: array of rocblas_double_complex;
  fullA: array of rocblas_double_complex;
  tempX: array of rocblas_double_complex;
  packed_size, strideAP, strideX: Int64;
  i, j, b, k: Integer;
  sum, prod, num, den: rocblas_double_complex;
begin
  packed_size := N * (N + 1) div 2;
  strideAP := packed_size;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostB, strideX * batch_count);
  SetLength(hostX_result, strideX * batch_count);
  SetLength(expectedX, strideX * batch_count);
  SetLength(fullA, N * N * batch_count);
  SetLength(tempX, N);

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do
      for i := 0 to N - 1 do
        if i <= j then
        begin
          fullA[b*N*N + j*N + i].x := (b+1)*3 + i + j + 1;
          fullA[b*N*N + j*N + i].y := (i-j)*0.3;
        end else
        begin
          fullA[b*N*N + j*N + i].x := 0; fullA[b*N*N + j*N + i].y := 0;
        end;
    k := 0;
    for j := 0 to N - 1 do
      for i := 0 to j do
      begin
        hostAP[b*strideAP + k] := fullA[b*N*N + j*N + i];
        Inc(k);
      end;
    for i := 0 to N - 1 do
    begin
      hostB[b*strideX + i].x := (b+1)*15 + i * 2.5;
      hostB[b*strideX + i].y := (b+1)*8 - i * 1.5;
    end;
  end;

  // CPU reference calculation
  for b := 0 to batch_count - 1 do
  begin
    for i := N - 1 downto 0 do
    begin
      sum.x := 0.0; sum.y := 0.0;
      for j := i + 1 to N - 1 do
      begin
        prod := fullA[b*N*N + j*N + i];
        ComplexMul(prod, tempX[j]);
        ComplexAdd(sum, prod);
      end;
      num := hostB[b*strideX + i];
      num.x := num.x - sum.x; num.y := num.y - sum.y;
      den := fullA[b*N*N + i*N + i];
      ComplexDiv(num, den);
      tempX[i] := num;
    end;
    for i := 0 to N-1 do expectedX[b*strideX + i] := tempX[i];
  end;

  // GPU operations
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X(b)', HIP_SUCCESS, hipMemcpy(devX, @hostB[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_ztpsv_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_ztpsv_strided_batched(handle, uplo, transA, diag, N, devAP, strideAP, devX, 1, strideX, batch_count)));

  AssertEquals('hipMemcpy D2H X', HIP_SUCCESS, hipMemcpy(@hostX_result[0], devX, strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideX * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTPSV_strided_batched X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTPSV_strided_batched X[%d].y', [i]));
  end;

  hipFree(devAP);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_SGER_strided_batched;
const
  M = 4;
  N = 5;
  batch_count = 2;
  alpha: Single = 2.5;
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

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M - 1 do hostX[b*strideX + i] := (b+1) + (i * 0.5);
    for i := 0 to N - 1 do hostY[b*strideY + i] := (b+1) + (i * -0.25);
    for i := 0 to strideA - 1 do hostA[b*strideA + i] := (b+1)*10 + i;
  end;

  // CPU reference calculation: A = alpha*x*y' + A
  for i := 0 to High(hostA) do expectedA[i] := hostA[i];

  for b := 0 to batch_count - 1 do
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
        expectedA[b*strideA + j*M + i] := expectedA[b*strideA + j*M + i] +
          alpha * hostX[b*strideX + i] * hostY[b*strideY + j];

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Single)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  AssertEquals('rocblas_sger_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_sger_strided_batched(handle, M, N, @alpha, devX, 1, strideX, devY, 1, strideY, devA, M, strideA, batch_count)));

  AssertEquals('hipMemcpy D2H A', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideA * batch_count) - 1 do
    CheckEqualsRelative(expectedA[i], hostA_result[i], 1e-5, Format('SGER_strided_batched A[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_DGER_strided_batched;
const
  M = 4;
  N = 5;
  batch_count = 2;
  alpha: Double = 2.5;
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

  // Initialize data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M - 1 do hostX[b*strideX + i] := (b+1) + (i * 0.5);
    for i := 0 to N - 1 do hostY[b*strideY + i] := (b+1) + (i * -0.25);
    for i := 0 to strideA - 1 do hostA[b*strideA + i] := (b+1)*10 + i;
  end;

  // CPU reference calculation: A = alpha*x*y' + A
  for i := 0 to High(hostA) do expectedA[i] := hostA[i];

  for b := 0 to batch_count - 1 do
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
        expectedA[b*strideA + j*M + i] := expectedA[b*strideA + j*M + i] +
          alpha * hostX[b*strideX + i] * hostY[b*strideY + j];

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Double)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  AssertEquals('rocblas_dger_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_dger_strided_batched(handle, M, N, @alpha, devX, 1, strideX, devY, 1, strideY, devA, M, strideA, batch_count)));

  AssertEquals('hipMemcpy D2H A', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideA * batch_count) - 1 do
    CheckEqualsRelative(expectedA[i], hostA_result[i], 1e-12, Format('DGER_strided_batched A[%d]', [i]));

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CGERU_strided_batched;
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
  prod: rocblas_float_complex;
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

  // CPU reference: A = alpha*x*y' + A
  for i := 0 to High(hostA) do expectedA[i] := hostA[i];

  for b := 0 to batch_count - 1 do
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        prod := hostX[b*strideX + i];
        ComplexMul(prod, hostY[b*strideY + j]);
        ComplexMul(prod, alpha);
        ComplexAdd(expectedA[b*strideA + j*M + i], prod);
      end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_cgeru_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_cgeru_strided_batched(handle, M, N, @alpha, devX, 1, strideX, devY, 1, strideY, devA, M, strideA, batch_count)));

  AssertEquals('hipMemcpy D2H A', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideA * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA_result[i].x, 1e-5, Format('CGERU_strided_batched A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA_result[i].y, 1e-5, Format('CGERU_strided_batched A[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_ZGERU_strided_batched;
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
  prod: rocblas_double_complex;
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

  // CPU reference: A = alpha*x*y' + A
  for i := 0 to High(hostA) do expectedA[i] := hostA[i];

  for b := 0 to batch_count - 1 do
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        prod := hostX[b*strideX + i];
        ComplexMul(prod, hostY[b*strideY + j]);
        ComplexMul(prod, alpha);
        ComplexAdd(expectedA[b*strideA + j*M + i], prod);
      end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_zgeru_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_zgeru_strided_batched(handle, M, N, @alpha, devX, 1, strideX, devY, 1, strideY, devA, M, strideA, batch_count)));

  AssertEquals('hipMemcpy D2H A', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideA * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA_result[i].x, 1e-12, Format('ZGERU_strided_batched A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA_result[i].y, 1e-12, Format('ZGERU_strided_batched A[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CGERC_strided_batched;
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
  prod, y_conj: rocblas_float_complex;
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

  // CPU reference: A = alpha*x*y^H + A
  for i := 0 to High(hostA) do expectedA[i] := hostA[i];

  for b := 0 to batch_count - 1 do
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        y_conj := hostY[b*strideY + j];
        y_conj.y := -y_conj.y; // Conjugate y
        prod := hostX[b*strideX + i];
        ComplexMul(prod, y_conj);
        ComplexMul(prod, alpha);
        ComplexAdd(expectedA[b*strideA + j*M + i], prod);
      end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_cgerc_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_cgerc_strided_batched(handle, M, N, @alpha, devX, 1, strideX, devY, 1, strideY, devA, M, strideA, batch_count)));

  AssertEquals('hipMemcpy D2H A', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideA * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA_result[i].x, 1e-5, Format('CGERC_strided_batched A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA_result[i].y, 1e-5, Format('CGERC_strided_batched A[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_ZGERC_strided_batched;
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
  prod, y_conj: rocblas_double_complex;
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

  // CPU reference: A = alpha*x*y^H + A
  for i := 0 to High(hostA) do expectedA[i] := hostA[i];

  for b := 0 to batch_count - 1 do
    for j := 0 to N - 1 do
      for i := 0 to M - 1 do
      begin
        y_conj := hostY[b*strideY + j];
        y_conj.y := -y_conj.y; // Conjugate y
        prod := hostX[b*strideX + i];
        ComplexMul(prod, y_conj);
        ComplexMul(prod, alpha);
        ComplexAdd(expectedA[b*strideA + j*M + i], prod);
      end;

  // GPU operations
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  AssertEquals('hipMemcpy H2D A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy H2D Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  AssertEquals('rocblas_zgerc_strided_batched OK', Integer(rocblas_status_success),
    Integer(rocblas_zgerc_strided_batched(handle, M, N, @alpha, devX, 1, strideX, devY, 1, strideY, devA, M, strideA, batch_count)));

  AssertEquals('hipMemcpy D2H A', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verification
  for i := 0 to (strideA * batch_count) - 1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA_result[i].x, 1e-12, Format('ZGERC_strided_batched A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA_result[i].y, 1e-12, Format('ZGERC_strided_batched A[%d].y', [i]));
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_SSYR_strided_batched;
const
  N = 2;
  batch_count = 1;
  incx = 1;
  uplo = rocblas_fill_upper;
var
  alpha: Single;
  devA, devX: hipDeviceptr_t;
  hostA, hostA_result, expectedA: array of Single;
  hostX: array of Single;
  strideA, strideX: Int64;
  i, j, b: Integer;
  idx_linear: Integer;
begin
  alpha := 1.0;
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(expectedA, strideA * batch_count);
  SetLength(hostA_result, strideA * batch_count);

  // Initialize A with simple values (column-major indexing)
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        hostA[idx_linear] := 0.0; // All elements start at 0.0
      end;
    end;
  end;

  // Initialize X with simple, predictable values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := 1.0 + i; // [1.0, 2.0] for batch 0
    end;
  end;

  // Calculate expected result manually for each batch (using column-major indexing)
  // First, copy the initial A matrix
  for i := 0 to High(expectedA) do
    expectedA[i] := hostA[i];

  // Then apply the SYR update: A := A + alpha * x * x^T
  // For column-major, we still compute the same mathematical operation,
  // but we store/access the elements in column-major order
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Only update elements in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          expectedA[idx_linear] := expectedA[idx_linear] +
                                   alpha * hostX[b * strideX + i] * hostX[b * strideX + j];
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  // Copy data to device
  AssertEquals('hipMemcpy A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_ssyr_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_ssyr_strided_batched(
      handle, uplo, N, @alpha,
      devX, incx, strideX, // x vector and its stride
      devA, N, strideA, // A matrix and its stride (lda = N)
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy A_result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verify results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Only check the part of the matrix that was supposed to be updated
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
             CheckEqualsRelative(expectedA[idx_linear], hostA_result[idx_linear], 1e-5,
              Format('SSYR_strided_batched A[%d][%d][%d]', [b, i, j]));
        end
        else
        begin
            // Parts not updated should remain the same as the original input A
            CheckEqualsRelative(hostA[idx_linear], hostA_result[idx_linear], 1e-5,
              Format('SSYR_strided_batched (Unchanged) A[%d][%d][%d]', [b, i, j]));
        end;
      end;
    end;
  end;

  hipFree(devA);
  hipFree(devX);
end;


procedure TTestRocBLAS_Level2_strided_batched.Test_DSYR_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  uplo = rocblas_fill_lower; // Test the lower triangle (opposite of SSYR test)
var
  alpha: Double;
  devA, devX: hipDeviceptr_t;
  hostA, hostA_result, expectedA: array of Double;
  hostX: array of Double; // Make sure this is declared in the var section
  strideA, strideX: Int64;
  i, j, b: Integer;
  idx_linear: Integer;
begin
  alpha := -1.5; // Use a different alpha value for variety
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(expectedA, strideA * batch_count);
  SetLength(hostA_result, strideA * batch_count);

  // Initialize A with distinct values (column-major indexing)
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Example initialization: base value + row contribution + column contribution + batch contribution
        hostA[idx_linear] := 2.0 + (i * 0.1) + (j * 0.01) + b;
      end;
    end;
  end;

  // Initialize X with simple, predictable values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      // X elements are stored linearly: X[b][i] at index b * strideX + i
      hostX[b * strideX + i] := (i + 1) * 0.5 + b; // e.g., [0.5+b, 1.0+b, 1.5+b] for batch b
    end;
  end;

  // Calculate expected result manually for each batch (using column-major indexing)
  // First, copy the initial A matrix
  for i := 0 to High(expectedA) do
    expectedA[i] := hostA[i];

  // Then apply the SYR update: A := A + alpha * x * x^T
  // Only elements within the specified triangle (uplo) should be updated
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Apply update only if (i,j) is in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          expectedA[idx_linear] := expectedA[idx_linear] +
                                   alpha * hostX[b * strideX + i] * hostX[b * strideX + j];
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  // Copy data to device
  AssertEquals('hipMemcpy A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_dsyr_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_dsyr_strided_batched(
      handle, uplo, N, @alpha,
      devX, incx, strideX, // x vector and its stride
      devA, N, strideA,    // A matrix (lda=N) and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy A_result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verify results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Check elements based on whether they should have been updated
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
             CheckEqualsRelative(expectedA[idx_linear], hostA_result[idx_linear], 1e-12,
              Format('DSYR_strided_batched A[%d][%d][%d]', [b, i, j]));
        end
        else
        begin
            // Parts not updated should remain the same as the original input A
            CheckEqualsRelative(hostA[idx_linear], hostA_result[idx_linear], 1e-12,
              Format('DSYR_strided_batched (Unchanged) A[%d][%d][%d]', [b, i, j]));
        end;
      end;
    end;
  end;

  hipFree(devA);
  hipFree(devX);
end;

//procedure TTestRocBLAS_Level2_strided_batched.Test_CSYR_strided_batched;
//begin
//  WriteLn('Function CSYR_strided_batched does not exist in the rocBLAS API');
//end;
//
//procedure TTestRocBLAS_Level2_strided_batched.Test_ZSYR_strided_batched;
//begin
//  WriteLn('Function ZSYR_strided_batched does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CHER_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  uplo = rocblas_fill_upper; // Test upper triangle
var
  alpha: Single; // alpha is real for HER
  devA, devX: hipDeviceptr_t;
  hostA, hostA_result, expectedA: array of rocblas_float_complex;
  hostX: array of rocblas_float_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
  idx_linear: Integer;
  temp_complex: rocblas_float_complex;
begin
  alpha := 2.0; // Real scalar alpha
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(expectedA, strideA * batch_count);
  SetLength(hostA_result, strideA * batch_count);

  // Initialize A with Hermitian values (column-major indexing)
  // For a Hermitian matrix, A[i][j] = conj(A[j][i])
  // Diagonal elements must be real.
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do // Column index first for column-major
    begin
      for i := 0 to N - 1 do // Row index
      begin
        idx_linear := b * strideA + j * N + i;
        if i = j then
        begin
          // Diagonal elements are real
          hostA[idx_linear].x := 1.0 + b + i; // Real part
          hostA[idx_linear].y := 0.0;         // Imaginary part is 0
        end
        else
        begin
          // Off-diagonal elements
          hostA[idx_linear].x := (1.0 + b + i + j) * 0.1;     // Real part
          hostA[idx_linear].y := (1.0 + b + Abs(i - j)) * 0.05; // Imaginary part
          // Ensure Hermitian property by setting A[j][i] = conj(A[i][j])
          // This initialization loop handles this implicitly by setting all elements,
          // but we need to be careful about the relationship.
          // Actually, let's just initialize the full matrix and ensure diagonals are real.
          // The HER operation will maintain the Hermitian property.
        end;
      end;
    end;
  end;

  // Initialize X with complex values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := 1.0 + i + b;      // Real part
      hostX[b * strideX + i].y := 0.5 * (i + b);    // Imaginary part
    end;
  end;

  // Calculate expected result manually for each batch (using column-major indexing)
  // First, copy the initial A matrix
  for i := 0 to High(expectedA) do
    expectedA[i] := hostA[i];

  // Then apply the HEMV update: A := A + alpha * x * x^H
  // Note: x^H is the conjugate transpose (row vector where each element is conjugated)
  // The operation is A[i][j] += alpha * x[i] * conj(x[j])
  // Only elements within the specified triangle (uplo) are referenced/updated by rocBLAS
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Apply update only if (i,j) is in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // alpha * x[i] * conj(x[j])
          // conj(x[j]) = x[j].x - i*x[j].y
          temp_complex.x := hostX[b * strideX + i].x * hostX[b * strideX + j].x +
                            hostX[b * strideX + i].y * hostX[b * strideX + j].y;
          temp_complex.y := hostX[b * strideX + i].y * hostX[b * strideX + j].x -
                            hostX[b * strideX + i].x * hostX[b * strideX + j].y;
          // Scale by alpha (real)
          temp_complex.x := alpha * temp_complex.x;
          temp_complex.y := alpha * temp_complex.y;

          expectedA[idx_linear].x := expectedA[idx_linear].x + temp_complex.x;
          expectedA[idx_linear].y := expectedA[idx_linear].y + temp_complex.y;
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  // Copy data to device
  AssertEquals('hipMemcpy A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_cher_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_cher_strided_batched(
      handle, uplo, N, @alpha,        // alpha is a real pointer for cher
      devX, incx, strideX,           // x vector and its stride
      devA, N, strideA,              // A matrix (lda=N) and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy A_result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verify results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Check elements based on whether they should have been updated
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
             CheckEqualsRelative(expectedA[idx_linear].x, hostA_result[idx_linear].x, 1e-5,
              Format('CHER_strided_batched A[%d][%d][%d].x', [b, i, j]));
             CheckEqualsRelative(expectedA[idx_linear].y, hostA_result[idx_linear].y, 1e-5,
              Format('CHER_strided_batched A[%d][%d][%d].y', [b, i, j]));
        end
        else
        begin
            // Parts not referenced should remain the same as the original input A
            CheckEqualsRelative(hostA[idx_linear].x, hostA_result[idx_linear].x, 1e-5,
              Format('CHER_strided_batched (Unchanged) A[%d][%d][%d].x', [b, i, j]));
            CheckEqualsRelative(hostA[idx_linear].y, hostA_result[idx_linear].y, 1e-5,
              Format('CHER_strided_batched (Unchanged) A[%d][%d][%d].y', [b, i, j]));
        end;
      end;
    end;
  end;

  hipFree(devA);
  hipFree(devX);
end;


procedure TTestRocBLAS_Level2_strided_batched.Test_ZHER_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  uplo = rocblas_fill_lower; // Test lower triangle (opposite of CHER test)
var
  alpha: Double; // alpha is real for HER
  devA, devX: hipDeviceptr_t;
  hostA, hostA_result, expectedA: array of rocblas_double_complex;
  hostX: array of rocblas_double_complex;
  strideA, strideX: Int64;
  i, j, b: Integer;
  idx_linear: Integer;
  temp_complex: rocblas_double_complex;
begin
  alpha := -1.5; // Real scalar alpha, different value and sign
  strideA := N * N;
  strideX := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(expectedA, strideA * batch_count);
  SetLength(hostA_result, strideA * batch_count);

  // Initialize A with Hermitian values (column-major indexing)
  // Diagonal elements must be real.
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do // Column index first for column-major
    begin
      for i := 0 to N - 1 do // Row index
      begin
        idx_linear := b * strideA + j * N + i;
        if i = j then
        begin
          // Diagonal elements are real
          hostA[idx_linear].x := 2.0 + b + i; // Real part
          hostA[idx_linear].y := 0.0;         // Imaginary part is 0
        end
        else
        begin
          // Off-diagonal elements
          hostA[idx_linear].x := (2.0 + b + i + j) * 0.15;    // Real part
          hostA[idx_linear].y := (2.0 + b + Abs(i - j)) * 0.07; // Imaginary part
        end;
      end;
    end;
  end;

  // Initialize X with complex values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := (i + 1) * 0.5 + b;  // Real part
      hostX[b * strideX + i].y := (i + 1) * 0.25 - b; // Imaginary part
    end;
  end;

  // Calculate expected result manually for each batch (using column-major indexing)
  // First, copy the initial A matrix
  for i := 0 to High(expectedA) do
    expectedA[i] := hostA[i];

  // Then apply the HEMV update: A := A + alpha * x * x^H
  // The operation is A[i][j] += alpha * x[i] * conj(x[j])
  // Only elements within the specified triangle (uplo) are referenced/updated by rocBLAS
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Apply update only if (i,j) is in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // alpha * x[i] * conj(x[j])
          // conj(x[j]) = x[j].x - i*x[j].y
          temp_complex.x := hostX[b * strideX + i].x * hostX[b * strideX + j].x +
                            hostX[b * strideX + i].y * hostX[b * strideX + j].y;
          temp_complex.y := hostX[b * strideX + i].y * hostX[b * strideX + j].x -
                            hostX[b * strideX + i].x * hostX[b * strideX + j].y;
          // Scale by alpha (real)
          temp_complex.x := alpha * temp_complex.x;
          temp_complex.y := alpha * temp_complex.y;

          expectedA[idx_linear].x := expectedA[idx_linear].x + temp_complex.x;
          expectedA[idx_linear].y := expectedA[idx_linear].y + temp_complex.y;
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  // Copy data to device
  AssertEquals('hipMemcpy A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_zher_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_zher_strided_batched(
      handle, uplo, N, @alpha,        // alpha is a real pointer for zher
      devX, incx, strideX,           // x vector and its stride
      devA, N, strideA,              // A matrix (lda=N) and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy A_result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verify results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Check elements based on whether they should have been updated
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
             CheckEqualsRelative(expectedA[idx_linear].x, hostA_result[idx_linear].x, 1e-12,
              Format('ZHER_strided_batched A[%d][%d][%d].x', [b, i, j]));
             CheckEqualsRelative(expectedA[idx_linear].y, hostA_result[idx_linear].y, 1e-12,
              Format('ZHER_strided_batched A[%d][%d][%d].y', [b, i, j]));
        end
        else
        begin
            // Parts not referenced should remain the same as the original input A
            CheckEqualsRelative(hostA[idx_linear].x, hostA_result[idx_linear].x, 1e-12,
              Format('ZHER_strided_batched (Unchanged) A[%d][%d][%d].x', [b, i, j]));
            CheckEqualsRelative(hostA[idx_linear].y, hostA_result[idx_linear].y, 1e-12,
              Format('ZHER_strided_batched (Unchanged) A[%d][%d][%d].y', [b, i, j]));
        end;
      end;
    end;
  end;

  hipFree(devA);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_SSPR_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  uplo = rocblas_fill_upper; // Test upper triangle
var
  alpha: Single;
  devAP, devX: hipDeviceptr_t;
  hostAP, hostAP_result, expectedAP: array of Single;
  hostX: array of Single;
  strideAP, strideX: Int64;
  i, j, b, k: Integer;
  idx_linear, idx_packed: Int64;
begin
  alpha := 2.0;
  // For packed format, the size is N*(N+1)/2 elements per matrix
  strideAP := (N * (N + 1)) div 2;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);

  // Initialize AP (packed symmetric matrix) with simple values
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Packed index
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        // Only store the relevant triangle (including diagonal) in packed format
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          idx_packed := b * strideAP + k;
          hostAP[idx_packed] := 1.0 + b + i * 0.1 + j * 0.01; // Distinct initial values
          Inc(k);
        end;
      end;
    end;
  end;

  // Initialize X with simple, predictable values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := 1.0 + i + b; // [1.0+b, 2.0+b, 3.0+b] for batch b
    end;
  end;

  // Calculate expected result manually for each batch
  // First, copy the initial AP matrix
  for i := 0 to High(expectedAP) do
    expectedAP[i] := hostAP[i];

  // Then apply the SPR update: AP := AP + alpha * x * x^T
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Reset packed index for this batch
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update elements in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          idx_packed := b * strideAP + k;
          expectedAP[idx_packed] := expectedAP[idx_packed] +
                                    alpha * hostX[b * strideX + i] * hostX[b * strideX + j];
          Inc(k);
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));

  // Copy data to device
  AssertEquals('hipMemcpy AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_sspr_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_sspr_strided_batched(
      handle, uplo, N, @alpha,
      devX, incx, strideX, // x vector and its stride
      devAP, strideAP,     // AP matrix and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy AP_result', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to High(expectedAP) do
  begin
    CheckEqualsRelative(expectedAP[i], hostAP_result[i], 1e-5,
      Format('SSPR_strided_batched AP[%d]', [i]));
  end;

  hipFree(devAP);
  hipFree(devX);
end;


procedure TTestRocBLAS_Level2_strided_batched.Test_DSPR_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  uplo = rocblas_fill_lower; // Test lower triangle (opposite of SSPR)
var
  alpha: Double;
  devAP, devX: hipDeviceptr_t;
  hostAP, hostAP_result, expectedAP: array of Double;
  hostX: array of Double;
  strideAP, strideX: Int64;
  i, j, b, k: Integer;
  idx_packed: Int64;
begin
  alpha := -1.5; // Different alpha value and sign
  // For packed format, the size is N*(N+1)/2 elements per matrix
  strideAP := (N * (N + 1)) div 2;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);

  // Initialize AP (packed symmetric matrix) with simple values
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Packed index
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        // Only store the lower triangle (including diagonal) in packed format
        if (uplo = rocblas_fill_lower) and (i >= j) then
        begin
          idx_packed := b * strideAP + k;
          hostAP[idx_packed] := 2.0 + b + i * 0.15 + j * 0.02; // Distinct initial values
          Inc(k);
        end
        else if (uplo = rocblas_fill_upper) and (i <= j) then
        begin
          idx_packed := b * strideAP + k;
          hostAP[idx_packed] := 2.0 + b + i * 0.15 + j * 0.02;
          Inc(k);
        end;
      end;
    end;
  end;

  // Initialize X with simple, predictable values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := (i + 1) * 0.5 - b; // [0.5-b, 1.0-b, 1.5-b] for batch b
    end;
  end;

  // Calculate expected result manually for each batch
  // First, copy the initial AP matrix
  for i := 0 to High(expectedAP) do
    expectedAP[i] := hostAP[i];

  // Then apply the SPR update: AP := AP + alpha * x * x^T
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Reset packed index for this batch
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update elements in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          idx_packed := b * strideAP + k;
          expectedAP[idx_packed] := expectedAP[idx_packed] +
                                    alpha * hostX[b * strideX + i] * hostX[b * strideX + j];
          Inc(k);
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));

  // Copy data to device
  AssertEquals('hipMemcpy AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_dspr_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_dspr_strided_batched(
      handle, uplo, N, @alpha,
      devX, incx, strideX, // x vector and its stride
      devAP, strideAP,     // AP matrix and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy AP_result', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to High(expectedAP) do
  begin
    CheckEqualsRelative(expectedAP[i], hostAP_result[i], 1e-12,
      Format('DSPR_strided_batched AP[%d]', [i]));
  end;

  hipFree(devAP);
  hipFree(devX);
end;


procedure TTestRocBLAS_Level2_strided_batched.Test_CHPR_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  uplo = rocblas_fill_upper; // Test upper triangle
var
  alpha: Single; // alpha is real for HPR
  devAP, devX: hipDeviceptr_t;
  hostAP, hostAP_result, expectedAP: array of rocblas_float_complex;
  hostX: array of rocblas_float_complex;
  strideAP, strideX: Int64;
  i, j, b, k: Integer;
  idx_packed: Int64;
  temp_complex: rocblas_float_complex;
begin
  alpha := 2.0; // Real scalar alpha
  // For packed format, the size is N*(N+1)/2 elements per matrix
  strideAP := (N * (N + 1)) div 2;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);

  // Initialize AP (packed Hermitian matrix) with values
  // Diagonal elements must be real.
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Packed index
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        // Only store the relevant triangle (including diagonal) in packed format
        if ((uplo = rocblas_fill_lower) and (i >= j)) or
           ((uplo = rocblas_fill_upper) and (i <= j)) then
        begin
          idx_packed := b * strideAP + k;
          if i = j then
          begin
            // Diagonal elements are real
            hostAP[idx_packed].x := 2.0 + b + i; // Real part
            hostAP[idx_packed].y := 0.0;         // Imaginary part is 0
          end
          else
          begin
            // Off-diagonal elements
            hostAP[idx_packed].x := (2.0 + b + i + j) * 0.15;    // Real part
            hostAP[idx_packed].y := (2.0 + b + Abs(i - j)) * 0.07; // Imaginary part
          end;
          Inc(k);
        end;
      end;
    end;
  end;

  // Initialize X with complex values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := 1.0 + i + b;      // Real part
      hostX[b * strideX + i].y := 0.5 * (i + b);    // Imaginary part
    end;
  end;

  // Calculate expected result manually for each batch
  // First, copy the initial AP matrix
  for i := 0 to High(expectedAP) do
    expectedAP[i] := hostAP[i];

  // Then apply the HPR update: AP := AP + alpha * x * x^H
  // Note: x^H is the conjugate transpose (row vector where each element is conjugated)
  // The operation is AP[i][j] += alpha * x[i] * conj(x[j])
  // Only elements within the specified triangle (uplo) are referenced/updated by rocBLAS
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Reset packed index for this batch
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update elements in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          idx_packed := b * strideAP + k;
          // alpha * x[i] * conj(x[j])
          // conj(x[j]) = x[j].x - i*x[j].y
          temp_complex.x := hostX[b * strideX + i].x * hostX[b * strideX + j].x +
                            hostX[b * strideX + i].y * hostX[b * strideX + j].y;
          temp_complex.y := hostX[b * strideX + i].y * hostX[b * strideX + j].x -
                            hostX[b * strideX + i].x * hostX[b * strideX + j].y;
          // Scale by alpha (real)
          temp_complex.x := alpha * temp_complex.x;
          temp_complex.y := alpha * temp_complex.y;

          expectedAP[idx_packed].x := expectedAP[idx_packed].x + temp_complex.x;
          expectedAP[idx_packed].y := expectedAP[idx_packed].y + temp_complex.y;
          Inc(k);
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));

  // Copy data to device
  AssertEquals('hipMemcpy AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_chpr_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_chpr_strided_batched(
      handle, uplo, N, @alpha,        // alpha is a real pointer for chpr
      devX, incx, strideX,           // x vector and its stride
      devAP, strideAP,               // AP matrix and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy AP_result', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to High(expectedAP) do
  begin
    CheckEqualsRelative(expectedAP[i].x, hostAP_result[i].x, 1e-5,
      Format('CHPR_strided_batched AP[%d].x', [i]));
    CheckEqualsRelative(expectedAP[i].y, hostAP_result[i].y, 1e-5,
      Format('CHPR_strided_batched AP[%d].y', [i]));
  end;

  hipFree(devAP);
  hipFree(devX);
end;


procedure TTestRocBLAS_Level2_strided_batched.Test_ZHPR_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  uplo = rocblas_fill_lower; // Test lower triangle (opposite of CHPR)
var
  alpha: Double; // alpha is real for HPR
  devAP, devX: hipDeviceptr_t;
  hostAP, hostAP_result, expectedAP: array of rocblas_double_complex;
  hostX: array of rocblas_double_complex;
  strideAP, strideX: Int64;
  i, j, b, k: Integer;
  idx_packed: Int64;
  temp_complex: rocblas_double_complex;
begin
  alpha := -1.5; // Real scalar alpha, different value and sign
  // For packed format, the size is N*(N+1)/2 elements per matrix
  strideAP := (N * (N + 1)) div 2;
  strideX := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);

  // Initialize AP (packed Hermitian matrix) with values
  // Diagonal elements must be real.
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Packed index
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        // Only store the relevant triangle (including diagonal) in packed format
        if ((uplo = rocblas_fill_lower) and (i >= j)) or
           ((uplo = rocblas_fill_upper) and (i <= j)) then
        begin
          idx_packed := b * strideAP + k;
          if i = j then
          begin
            // Diagonal elements are real
            hostAP[idx_packed].x := 2.0 + b + i; // Real part
            hostAP[idx_packed].y := 0.0;         // Imaginary part is 0
          end
          else
          begin
            // Off-diagonal elements
            hostAP[idx_packed].x := (2.0 + b + i + j) * 0.15;    // Real part
            hostAP[idx_packed].y := (2.0 + b + Abs(i - j)) * 0.07; // Imaginary part
          end;
          Inc(k);
        end;      end;
    end;
  end;

  // Initialize X with complex values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := (i + 1) * 0.5 + b;  // Real part
      hostX[b * strideX + i].y := (i + 1) * 0.25 - b; // Imaginary part
    end;
  end;

  // Calculate expected result manually for each batch
  // First, copy the initial AP matrix
  for i := 0 to High(expectedAP) do
    expectedAP[i] := hostAP[i];

  // Then apply the HPR update: AP := AP + alpha * x * x^H
  // The operation is AP[i][j] += alpha * x[i] * conj(x[j])
  // Only elements within the specified triangle (uplo) are referenced/updated by rocBLAS
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Reset packed index for this batch
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update elements in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          idx_packed := b * strideAP + k;
          // alpha * x[i] * conj(x[j])
          // conj(x[j]) = x[j].x - i*x[j].y
          temp_complex.x := hostX[b * strideX + i].x * hostX[b * strideX + j].x +
                            hostX[b * strideX + i].y * hostX[b * strideX + j].y;
          temp_complex.y := hostX[b * strideX + i].y * hostX[b * strideX + j].x -
                            hostX[b * strideX + i].x * hostX[b * strideX + j].y;
          // Scale by alpha (real)
          temp_complex.x := alpha * temp_complex.x;
          temp_complex.y := alpha * temp_complex.y;

          expectedAP[idx_packed].x := expectedAP[idx_packed].x + temp_complex.x;
          expectedAP[idx_packed].y := expectedAP[idx_packed].y + temp_complex.y;
          Inc(k);
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));

  // Copy data to device
  AssertEquals('hipMemcpy AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_zhpr_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_zhpr_strided_batched(
      handle, uplo, N, @alpha,        // alpha is a real pointer for zhpr
      devX, incx, strideX,           // x vector and its stride
      devAP, strideAP,               // AP matrix and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy AP_result', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to High(expectedAP) do
  begin
    CheckEqualsRelative(expectedAP[i].x, hostAP_result[i].x, 1e-12,
      Format('ZHPR_strided_batched AP[%d].x', [i]));
    CheckEqualsRelative(expectedAP[i].y, hostAP_result[i].y, 1e-12,
      Format('ZHPR_strided_batched AP[%d].y', [i]));
  end;

  hipFree(devAP);
  hipFree(devX);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_SSYR2_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  incy = 1;
  uplo = rocblas_fill_upper; // Test upper triangle
var
  alpha: Single;
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_result, expectedA: array of Single;
  hostX, hostY: array of Single;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  idx_linear: Integer;
begin
  alpha := 2.0;
  strideA := N * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(expectedA, strideA * batch_count);
  SetLength(hostA_result, strideA * batch_count);

  // Initialize A with symmetric values (column-major indexing)
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do // Column index first for column-major
    begin
      for i := 0 to N - 1 do // Row index
      begin
        idx_linear := b * strideA + j * N + i;
        // Create a symmetric matrix base
        if i = j then
          hostA[idx_linear] := 1.0 + b + i // Diagonal
        else
          hostA[idx_linear] := 0.5 * (Abs(i - j)) + b; // Off-diagonal
      end;
    end;
  end;

  // Initialize X and Y with simple, predictable values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := 1.0 + i + b; // [1.0+b, 2.0+b, 3.0+b] for batch b
      hostY[b * strideY + i] := 0.5 + i * 0.5 + b; // [0.5+b, 1.0+b, 1.5+b] for batch b
    end;
  end;

  // Calculate expected result manually for each batch (using column-major indexing)
  // First, copy the initial A matrix
  for i := 0 to High(expectedA) do
    expectedA[i] := hostA[i];

  // Then apply the SYR2 update: A := A + alpha * x * y^T + alpha * y * x^T
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Apply update only if (i,j) is in the specified triangle
        // Note: The operation affects the entire matrix mathematically,
        // but rocBLAS only updates the specified triangle.
        // For verification, we compute the full update but only check the relevant parts.
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          expectedA[idx_linear] := expectedA[idx_linear] +
                                   alpha * hostX[b * strideX + i] * hostY[b * strideY + j] +
                                   alpha * hostY[b * strideY + i] * hostX[b * strideX + j];
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Single)));

  // Copy data to device
  AssertEquals('hipMemcpy A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_ssyr2_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_ssyr2_strided_batched(
      handle, uplo, N, @alpha,
      devX, incx, strideX, // x vector and its stride
      devY, incy, strideY, // y vector and its stride
      devA, N, strideA,    // A matrix (lda=N) and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy A_result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verify results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Check elements based on whether they should have been updated
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
             CheckEqualsRelative(expectedA[idx_linear], hostA_result[idx_linear], 1e-5,
              Format('SSYR2_strided_batched A[%d][%d][%d]', [b, i, j]));
        end
        else
        begin
            // Parts not updated should remain the same as the original input A
            CheckEqualsRelative(hostA[idx_linear], hostA_result[idx_linear], 1e-5,
              Format('SSYR2_strided_batched (Unchanged) A[%d][%d][%d]', [b, i, j]));
        end;
      end;
    end;
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;


procedure TTestRocBLAS_Level2_strided_batched.Test_DSYR2_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  incy = 1;
  uplo = rocblas_fill_lower; // Test lower triangle (opposite of SSYR2)
var
  alpha: Double;
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_result, expectedA: array of Double;
  hostX, hostY: array of Double;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  idx_linear: Integer;
begin
  alpha := -1.5; // Different alpha value and sign
  strideA := N * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(expectedA, strideA * batch_count);
  SetLength(hostA_result, strideA * batch_count);

  // Initialize A with symmetric values (column-major indexing)
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do // Column index first for column-major
    begin
      for i := 0 to N - 1 do // Row index
      begin
        idx_linear := b * strideA + j * N + i;
        // Create a symmetric matrix base
        if i = j then
          hostA[idx_linear] := 2.0 + b + i // Diagonal
        else
          hostA[idx_linear] := 0.3 * (Abs(i - j)) + b; // Off-diagonal
      end;
    end;
  end;

  // Initialize X and Y with simple, predictable values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := (i + 1) * 0.5 - b; // [0.5-b, 1.0-b, 1.5-b] for batch b
      hostY[b * strideY + i] := 1.0 + i * 0.25 + b; // [1.0+b, 1.25+b, 1.5+b] for batch b
    end;
  end;

  // Calculate expected result manually for each batch (using column-major indexing)
  // First, copy the initial A matrix
  for i := 0 to High(expectedA) do
    expectedA[i] := hostA[i];

  // Then apply the SYR2 update: A := A + alpha * x * y^T + alpha * y * x^T
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Apply update only if (i,j) is in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          expectedA[idx_linear] := expectedA[idx_linear] +
                                   alpha * hostX[b * strideX + i] * hostY[b * strideY + j] +
                                   alpha * hostY[b * strideY + i] * hostX[b * strideX + j];
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Double)));

  // Copy data to device
  AssertEquals('hipMemcpy A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_dsyr2_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_dsyr2_strided_batched(
      handle, uplo, N, @alpha,
      devX, incx, strideX, // x vector and its stride
      devY, incy, strideY, // y vector and its stride
      devA, N, strideA,    // A matrix (lda=N) and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy A_result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verify results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Check elements based on whether they should have been updated
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
             CheckEqualsRelative(expectedA[idx_linear], hostA_result[idx_linear], 1e-12,
              Format('DSYR2_strided_batched A[%d][%d][%d]', [b, i, j]));
        end
        else
        begin
            // Parts not updated should remain the same as the original input A
            CheckEqualsRelative(hostA[idx_linear], hostA_result[idx_linear], 1e-12,
              Format('DSYR2_strided_batched (Unchanged) A[%d][%d][%d]', [b, i, j]));
        end;
      end;
    end;
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

//procedure TTestRocBLAS_Level2_strided_batched.Test_CSYR2_strided_batched;
//begin
//  WriteLn('Function CSYR2_strided_batched does not exist in the rocBLAS API');
//end;
//
//
//procedure TTestRocBLAS_Level2_strided_batched.Test_ZSYR2_strided_batched;
//begin
//  WriteLn('Function ZSYR2_strided_batched does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level2_strided_batched.Test_CHER2_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  incy = 1;
  uplo = rocblas_fill_upper; // Test upper triangle
var
  alpha: rocblas_float_complex; // alpha is complex for HER2
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_result, expectedA: array of rocblas_float_complex;
  hostX, hostY: array of rocblas_float_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  idx_linear: Integer;
  temp1, temp2, temp3, temp4: rocblas_float_complex; // Working variables for complex calculations
  conj_alpha: rocblas_float_complex; // conjugate of alpha
  temp1_real, temp1_imag, temp2_real, temp2_imag, temp3_real, temp3_imag, temp4_real, temp4_imag: Single; // For intermediate complex mult results
begin
  // Initialize alpha
  alpha.x := 1.5; // Real part
  alpha.y := -0.5; // Imaginary part

  // Calculate conjugate of alpha
  conj_alpha.x := alpha.x;
  conj_alpha.y := -alpha.y; // conj(a + bi) = a - bi

  strideA := N * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(expectedA, strideA * batch_count);
  SetLength(hostA_result, strideA * batch_count);

  // Initialize A with Hermitian values (column-major indexing)
  // Diagonal elements must be real.
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do // Column index first for column-major
    begin
      for i := 0 to N - 1 do // Row index
      begin
        idx_linear := b * strideA + j * N + i;
        if i = j then
        begin
          // Diagonal elements are real
          hostA[idx_linear].x := 1.0 + b + i; // Real part
          hostA[idx_linear].y := 0.0;         // Imaginary part is 0
        end
        else
        begin
          // Off-diagonal elements
          hostA[idx_linear].x := (1.0 + b + i + j) * 0.1;     // Real part
          hostA[idx_linear].y := (1.0 + b + Abs(i - j)) * 0.05; // Imaginary part
        end;
      end;
    end;
  end;

  // Initialize X and Y with complex values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := 1.0 + i + b;      // Real part
      hostX[b * strideX + i].y := 0.5 * (i + b);    // Imaginary part

      hostY[b * strideY + i].x := 0.5 + i * 0.5 + b; // Real part
      hostY[b * strideY + i].y := 0.25 * (i + b);    // Imaginary part
    end;
  end;

  // Calculate expected result manually for each batch (using column-major indexing)
  // First, copy the initial A matrix
  for i := 0 to High(expectedA) do
    expectedA[i] := hostA[i];

  // Then apply the HER2 update: A := A + alpha * x * y^H + conj(alpha) * y * x^H
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Apply update only if (i,j) is in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Calculate alpha * x[i] * conj(y[j])
          // Step 1: alpha * x[i]
          // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
          temp1_real := alpha.x * hostX[b * strideX + i].x - alpha.y * hostX[b * strideX + i].y;
          temp1_imag := alpha.x * hostX[b * strideX + i].y + alpha.y * hostX[b * strideX + i].x;
          temp1.x := temp1_real;
          temp1.y := temp1_imag;

          // Step 2: (alpha * x[i]) * conj(y[j])
          // conj(y[j]) = (hostY[b * strideY + j].x, -hostY[b * strideY + j].y)
          temp2_real := temp1.x * hostY[b * strideY + j].x - temp1.y * (-hostY[b * strideY + j].y);
          temp2_imag := temp1.x * (-hostY[b * strideY + j].y) + temp1.y * hostY[b * strideY + j].x;
          temp2.x := temp2_real;
          temp2.y := temp2_imag;

          // Calculate conj(alpha) * y[i] * conj(x[j])
          // Step 1: conj(alpha) * y[i]
          // conj(alpha) = (alpha.x, -alpha.y)
          temp3_real := conj_alpha.x * hostY[b * strideY + i].x - conj_alpha.y * hostY[b * strideY + i].y;
          temp3_imag := conj_alpha.x * hostY[b * strideY + i].y + conj_alpha.y * hostY[b * strideY + i].x;
          temp3.x := temp3_real;
          temp3.y := temp3_imag;

          // Step 2: (conj(alpha) * y[i]) * conj(x[j])
          // conj(x[j]) = (hostX[b * strideX + j].x, -hostX[b * strideX + j].y)
          temp4_real := temp3.x * hostX[b * strideX + j].x - temp3.y * (-hostX[b * strideX + j].y);
          temp4_imag := temp3.x * (-hostX[b * strideX + j].y) + temp3.y * hostX[b * strideX + j].x;
          temp4.x := temp4_real;
          temp4.y := temp4_imag;

          expectedA[idx_linear].x := expectedA[idx_linear].x + temp2.x + temp4.x;
          expectedA[idx_linear].y := expectedA[idx_linear].y + temp2.y + temp4.y;
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  // Copy data to device
  AssertEquals('hipMemcpy A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_cher2_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_cher2_strided_batched(
      handle, uplo, N, @alpha,
      devX, incx, strideX, // x vector and its stride
      devY, incy, strideY, // y vector and its stride
      devA, N, strideA,    // A matrix (lda=N) and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy A_result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verify results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Check elements based on whether they should have been updated
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
             CheckEqualsRelative(expectedA[idx_linear].x, hostA_result[idx_linear].x, 1e-5,
              Format('CHER2_strided_batched A[%d][%d][%d].x', [b, i, j]));
             CheckEqualsRelative(expectedA[idx_linear].y, hostA_result[idx_linear].y, 1e-5,
              Format('CHER2_strided_batched A[%d][%d][%d].y', [b, i, j]));
        end
        else
        begin
            // Parts not updated should remain the same as the original input A
            CheckEqualsRelative(hostA[idx_linear].x, hostA_result[idx_linear].x, 1e-5,
              Format('CHER2_strided_batched (Unchanged) A[%d][%d][%d].x', [b, i, j]));
            CheckEqualsRelative(hostA[idx_linear].y, hostA_result[idx_linear].y, 1e-5,
              Format('CHER2_strided_batched (Unchanged) A[%d][%d][%d].y', [b, i, j]));
        end;
      end;
    end;
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;


procedure TTestRocBLAS_Level2_strided_batched.Test_ZHER2_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  incy = 1;
  uplo = rocblas_fill_lower; // Test lower triangle (opposite of CHER2)
var
  alpha: rocblas_double_complex; // alpha is complex for HER2
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_result, expectedA: array of rocblas_double_complex;
  hostX, hostY: array of rocblas_double_complex;
  strideA, strideX, strideY: Int64;
  i, j, b: Integer;
  idx_linear: Integer;
  temp1, temp2, temp3, temp4: rocblas_double_complex; // Working variables for complex calculations
  conj_alpha: rocblas_double_complex; // conjugate of alpha
  temp1_real, temp1_imag, temp2_real, temp2_imag: Double; // For intermediate complex mult results
  temp3_real, temp3_imag, temp4_real, temp4_imag: Double; // For intermediate complex mult results
begin
  // Initialize alpha
  alpha.x := -1.0; // Real part
  alpha.y := 0.75; // Imaginary part (different value and sign from CHER2)

  // Calculate conjugate of alpha
  conj_alpha.x := alpha.x;
  conj_alpha.y := -alpha.y; // conj(a + bi) = a - bi

  strideA := N * N;
  strideX := N;
  strideY := N;

  SetLength(hostA, strideA * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(expectedA, strideA * batch_count);
  SetLength(hostA_result, strideA * batch_count);

  // Initialize A with Hermitian values (column-major indexing)
  // Diagonal elements must be real.
  for b := 0 to batch_count - 1 do
  begin
    for j := 0 to N - 1 do // Column index first for column-major
    begin
      for i := 0 to N - 1 do // Row index
      begin
        idx_linear := b * strideA + j * N + i;
        if i = j then
        begin
          // Diagonal elements are real
          hostA[idx_linear].x := 2.0 + b + i; // Real part
          hostA[idx_linear].y := 0.0;         // Imaginary part is 0
        end
        else
        begin
          // Off-diagonal elements
          hostA[idx_linear].x := (2.0 + b + i + j) * 0.15;    // Real part
          hostA[idx_linear].y := (2.0 + b + Abs(i - j)) * 0.07; // Imaginary part
        end;
      end;
    end;
  end;

  // Initialize X and Y with complex values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := (i + 1) * 0.5 - b;  // Real part
      hostX[b * strideX + i].y := (i + 1) * 0.25 + b; // Imaginary part

      hostY[b * strideY + i].x := 1.0 + i * 0.25 - b; // Real part
      hostY[b * strideY + i].y := 1.0 + i * 0.125 + b; // Imaginary part
    end;
  end;

  // Calculate expected result manually for each batch (using column-major indexing)
  // First, copy the initial A matrix
  for i := 0 to High(expectedA) do
    expectedA[i] := hostA[i];

  // Then apply the HER2 update: A := A + alpha * x * y^H + conj(alpha) * y * x^H
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Apply update only if (i,j) is in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          // Calculate alpha * x[i] * conj(y[j])
          // Step 1: alpha * x[i]
          // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
          temp1_real := alpha.x * hostX[b * strideX + i].x - alpha.y * hostX[b * strideX + i].y;
          temp1_imag := alpha.x * hostX[b * strideX + i].y + alpha.y * hostX[b * strideX + i].x;
          temp1.x := temp1_real;
          temp1.y := temp1_imag;

          // Step 2: (alpha * x[i]) * conj(y[j])
          // conj(y[j]) = (hostY[b * strideY + j].x, -hostY[b * strideY + j].y)
          temp2_real := temp1.x * hostY[b * strideY + j].x - temp1.y * (-hostY[b * strideY + j].y);
          temp2_imag := temp1.x * (-hostY[b * strideY + j].y) + temp1.y * hostY[b * strideY + j].x;
          temp2.x := temp2_real;
          temp2.y := temp2_imag;

          // Calculate conj(alpha) * y[i] * conj(x[j])
          // Step 1: conj(alpha) * y[i]
          // conj(alpha) = (alpha.x, -alpha.y)
          temp3_real := conj_alpha.x * hostY[b * strideY + i].x - conj_alpha.y * hostY[b * strideY + i].y;
          temp3_imag := conj_alpha.x * hostY[b * strideY + i].y + conj_alpha.y * hostY[b * strideY + i].x;
          temp3.x := temp3_real;
          temp3.y := temp3_imag;

          // Step 2: (conj(alpha) * y[i]) * conj(x[j])
          // conj(x[j]) = (hostX[b * strideX + j].x, -hostX[b * strideX + j].y)
          temp4_real := temp3.x * hostX[b * strideX + j].x - temp3.y * (-hostX[b * strideX + j].y);
          temp4_imag := temp3.x * (-hostX[b * strideX + j].y) + temp3.y * hostX[b * strideX + j].x;
          temp4.x := temp4_real;
          temp4.y := temp4_imag;

          expectedA[idx_linear].x := expectedA[idx_linear].x + temp2.x + temp4.x;
          expectedA[idx_linear].y := expectedA[idx_linear].y + temp2.y + temp4.y;
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, strideA * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  // Copy data to device
  AssertEquals('hipMemcpy A', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_zher2_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_zher2_strided_batched(
      handle, uplo, N, @alpha,
      devX, incx, strideX, // x vector and its stride
      devY, incy, strideY, // y vector and its stride
      devA, N, strideA,    // A matrix (lda=N) and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy A_result', HIP_SUCCESS, hipMemcpy(@hostA_result[0], devA, strideA * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verify results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      for j := 0 to N - 1 do
      begin
        // Column-major: element (i,j) is at index j * N + i
        idx_linear := b * strideA + j * N + i;
        // Check elements based on whether they should have been updated
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
             CheckEqualsRelative(expectedA[idx_linear].x, hostA_result[idx_linear].x, 1e-12,
              Format('ZHER2_strided_batched A[%d][%d][%d].x', [b, i, j]));
             CheckEqualsRelative(expectedA[idx_linear].y, hostA_result[idx_linear].y, 1e-12,
              Format('ZHER2_strided_batched A[%d][%d][%d].y', [b, i, j]));
        end
        else
        begin
            // Parts not updated should remain the same as the original input A
            CheckEqualsRelative(hostA[idx_linear].x, hostA_result[idx_linear].x, 1e-12,
              Format('ZHER2_strided_batched (Unchanged) A[%d][%d][%d].x', [b, i, j]));
            CheckEqualsRelative(hostA[idx_linear].y, hostA_result[idx_linear].y, 1e-12,
              Format('ZHER2_strided_batched (Unchanged) A[%d][%d][%d].y', [b, i, j]));
        end;
      end;
    end;
  end;

  hipFree(devA);
  hipFree(devX);
  hipFree(devY);
end;

procedure TTestRocBLAS_Level2_strided_batched.Test_SSPR2_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  incy = 1;
  uplo = rocblas_fill_upper; // Test upper triangle
var
  alpha: Single;
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostAP_result, expectedAP: array of Single;
  hostX, hostY: array of Single;
  strideAP, strideX, strideY: Int64;
  i, j, b, k: Integer;
  idx_packed: Int64;
begin
  alpha := 2.0;
  // For packed format, the size is N*(N+1)/2 elements per matrix
  strideAP := (N * (N + 1)) div 2;
  strideX := N;
  strideY := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);

  // Initialize AP (packed symmetric matrix) with simple values
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Packed index
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        // Only store elements in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          idx_packed := b * strideAP + k;
          hostAP[idx_packed] := 1.0 + b + i * 0.1 + j * 0.01; // Distinct initial values
          Inc(k);
        end;
      end;
    end;
  end;

  // Initialize X and Y with simple, predictable values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := 1.0 + i + b; // [1.0+b, 2.0+b, 3.0+b] for batch b
      hostY[b * strideY + i] := 0.5 + i * 0.5 + b; // [0.5+b, 1.0+b, 1.5+b] for batch b
    end;
  end;

  // Calculate expected result manually for each batch
  // First, copy the initial AP matrix
  for i := 0 to High(expectedAP) do
    expectedAP[i] := hostAP[i];

  // Then apply the SPR2 update: AP := AP + alpha * x * y^T + alpha * y * x^T
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Reset packed index for this batch
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update elements in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          idx_packed := b * strideAP + k;
          expectedAP[idx_packed] := expectedAP[idx_packed] +
                                    alpha * hostX[b * strideX + i] * hostY[b * strideY + j] +
                                    alpha * hostY[b * strideY + i] * hostX[b * strideX + j];
          Inc(k);
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Single)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Single)));

  // Copy data to device
  AssertEquals('hipMemcpy AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Single), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Single), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_sspr2_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_sspr2_strided_batched(
      handle, uplo, N, @alpha,
      devX, incx, strideX, // x vector and its stride
      devY, incy, strideY, // y vector and its stride
      devAP, strideAP,     // AP matrix and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy AP_result', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(Single), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to High(expectedAP) do
  begin
    CheckEqualsRelative(expectedAP[i], hostAP_result[i], 1e-5,
      Format('SSPR2_strided_batched AP[%d]', [i]));
  end;

  hipFree(devAP);
  hipFree(devX);
  hipFree(devY);
end;


procedure TTestRocBLAS_Level2_strided_batched.Test_DSPR2_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  incy = 1;
  uplo = rocblas_fill_lower; // Test lower triangle (opposite of SSPR2)
var
  alpha: Double;
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostAP_result, expectedAP: array of Double;
  hostX, hostY: array of Double;
  strideAP, strideX, strideY: Int64;
  i, j, b, k: Integer;
  idx_packed: Int64;
begin
  alpha := -1.5; // Different alpha value and sign
  // For packed format, the size is N*(N+1)/2 elements per matrix
  strideAP := (N * (N + 1)) div 2;
  strideX := N;
  strideY := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);

  // Initialize AP (packed symmetric matrix) with simple values
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Packed index
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        // Only store elements in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          idx_packed := b * strideAP + k;
          hostAP[idx_packed] := 2.0 + b + i * 0.15 + j * 0.02; // Distinct initial values
          Inc(k);
        end;
      end;
    end;
  end;

  // Initialize X and Y with simple, predictable values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i] := (i + 1) * 0.5 - b; // [0.5-b, 1.0-b, 1.5-b] for batch b
      hostY[b * strideY + i] := 1.0 + i * 0.25 + b; // [1.0+b, 1.25+b, 1.5+b] for batch b
    end;
  end;

  // Calculate expected result manually for each batch
  // First, copy the initial AP matrix
  for i := 0 to High(expectedAP) do
    expectedAP[i] := hostAP[i];

  // Then apply the SPR2 update: AP := AP + alpha * x * y^T + alpha * y * x^T
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Reset packed index for this batch
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update elements in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          idx_packed := b * strideAP + k;
          expectedAP[idx_packed] := expectedAP[idx_packed] +
                                    alpha * hostX[b * strideX + i] * hostY[b * strideY + j] +
                                    alpha * hostY[b * strideY + i] * hostX[b * strideX + j];
          Inc(k);
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(Double)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(Double)));

  // Copy data to device
  AssertEquals('hipMemcpy AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(Double), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(Double), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_dspr2_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_dspr2_strided_batched(
      handle, uplo, N, @alpha,
      devX, incx, strideX, // x vector and its stride
      devY, incy, strideY, // y vector and its stride
      devAP, strideAP,     // AP matrix and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy AP_result', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(Double), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to High(expectedAP) do
  begin
    CheckEqualsRelative(expectedAP[i], hostAP_result[i], 1e-12,
      Format('DSPR2_strided_batched AP[%d]', [i]));
  end;

  hipFree(devAP);
  hipFree(devX);
  hipFree(devY);
end;


procedure TTestRocBLAS_Level2_strided_batched.Test_CHPR2_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  incy = 1;
  uplo = rocblas_fill_upper; // Test upper triangle
var
  alpha: rocblas_float_complex; // alpha is complex for HPR2
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostAP_result, expectedAP: array of rocblas_float_complex;
  hostX, hostY: array of rocblas_float_complex;
  strideAP, strideX, strideY: Int64;
  i, j, b, k: Integer;
  idx_packed: Int64;
  temp1, temp2: rocblas_float_complex; // temp1 = alpha*x[i]*conj(y[j]), temp2 = conj(alpha)*y[i]*conj(x[j])
  conj_alpha: rocblas_float_complex; // conjugate of alpha
  temp1_real, temp1_imag, temp2_real, temp2_imag: Single; // For intermediate complex mult results
begin
  // Initialize alpha
  alpha.x := 1.5; // Real part
  alpha.y := -0.5; // Imaginary part

  // Calculate conjugate of alpha
  conj_alpha.x := alpha.x;
  conj_alpha.y := -alpha.y; // conj(a + bi) = a - bi

  // For packed format, the size is N*(N+1)/2 elements per matrix
  strideAP := (N * (N + 1)) div 2;
  strideX := N;
  strideY := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);

  // Initialize AP (packed Hermitian matrix) with values
  // Diagonal elements must be real.
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Packed index
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        // Only store elements in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          idx_packed := b * strideAP + k;
          if i = j then
          begin
            // Diagonal elements are real
            hostAP[idx_packed].x := 1.0 + b + i; // Real part
            hostAP[idx_packed].y := 0.0;         // Imaginary part is 0
          end
          else
          begin
            // Off-diagonal elements
            hostAP[idx_packed].x := (1.0 + b + i + j) * 0.1;     // Real part
            hostAP[idx_packed].y := (1.0 + b + Abs(i - j)) * 0.05; // Imaginary part
          end;
          Inc(k);
        end;
      end;
    end;
  end;

  // Initialize X and Y with complex values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := 1.0 + i + b;      // Real part
      hostX[b * strideX + i].y := 0.5 * (i + b);    // Imaginary part

      hostY[b * strideY + i].x := 0.5 + i * 0.5 + b; // Real part
      hostY[b * strideY + i].y := 0.25 * (i + b);    // Imaginary part
    end;
  end;

  // Calculate expected result manually for each batch
  // First, copy the initial AP matrix
  for i := 0 to High(expectedAP) do
    expectedAP[i] := hostAP[i];

  // Then apply the HPR2 update: AP := AP + alpha * x * y^H + conj(alpha) * y * x^H
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Reset packed index for this batch
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update elements in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          idx_packed := b * strideAP + k;

          // Calculate alpha * x[i] * conj(y[j])
          // Step 1: alpha * x[i]
          temp1_real := alpha.x * hostX[b * strideX + i].x - alpha.y * hostX[b * strideX + i].y;
          temp1_imag := alpha.x * hostX[b * strideX + i].y + alpha.y * hostX[b * strideX + i].x;
          temp1.x := temp1_real;
          temp1.y := temp1_imag;

          // Step 2: (alpha * x[i]) * conj(y[j])
          // conj(y[j]) = (hostY[b * strideY + j].x, -hostY[b * strideY + j].y)
          temp1_real := temp1.x * hostY[b * strideY + j].x - temp1.y * (-hostY[b * strideY + j].y);
          temp1_imag := temp1.x * (-hostY[b * strideY + j].y) + temp1.y * hostY[b * strideY + j].x;
          temp1.x := temp1_real;
          temp1.y := temp1_imag;

          // Calculate conj(alpha) * y[i] * conj(x[j])
          // Step 1: conj(alpha) * y[i]
          temp2_real := conj_alpha.x * hostY[b * strideY + i].x - conj_alpha.y * hostY[b * strideY + i].y;
          temp2_imag := conj_alpha.x * hostY[b * strideY + i].y + conj_alpha.y * hostY[b * strideY + i].x;
          temp2.x := temp2_real;
          temp2.y := temp2_imag;

          // Step 2: (conj(alpha) * y[i]) * conj(x[j])
          // conj(x[j]) = (hostX[b * strideX + j].x, -hostX[b * strideX + j].y)
          temp2_real := temp2.x * hostX[b * strideX + j].x - temp2.y * (-hostX[b * strideX + j].y);
          temp2_imag := temp2.x * (-hostX[b * strideX + j].y) + temp2.y * hostX[b * strideX + j].x;
          temp2.x := temp2_real;
          temp2.y := temp2_imag;

          expectedAP[idx_packed].x := expectedAP[idx_packed].x + temp1.x + temp2.x;
          expectedAP[idx_packed].y := expectedAP[idx_packed].y + temp1.y + temp2.y;

          Inc(k);
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_float_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_float_complex)));

  // Copy data to device
  AssertEquals('hipMemcpy AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_chpr2_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_chpr2_strided_batched(
      handle, uplo, N, @alpha,
      devX, incx, strideX, // x vector and its stride
      devY, incy, strideY, // y vector and its stride
      devAP, strideAP,     // AP matrix and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy AP_result', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to High(expectedAP) do
  begin
    CheckEqualsRelative(expectedAP[i].x, hostAP_result[i].x, 1e-5,
      Format('CHPR2_strided_batched AP[%d].x', [i]));
    CheckEqualsRelative(expectedAP[i].y, hostAP_result[i].y, 1e-5,
      Format('CHPR2_strided_batched AP[%d].y', [i]));
  end;

  hipFree(devAP);
  hipFree(devX);
  hipFree(devY);
end;


procedure TTestRocBLAS_Level2_strided_batched.Test_ZHPR2_strided_batched;
const
  N = 3;
  batch_count = 2;
  incx = 1;
  incy = 1;
  uplo = rocblas_fill_lower; // Test lower triangle (opposite of CHPR2)
var
  alpha: rocblas_double_complex; // alpha is complex for HPR2
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostAP_result, expectedAP: array of rocblas_double_complex;
  hostX, hostY: array of rocblas_double_complex;
  strideAP, strideX, strideY: Int64;
  i, j, b, k: Integer;
  idx_packed: Int64;
  temp1, temp2: rocblas_double_complex; // temp1 = alpha*x[i]*conj(y[j]), temp2 = conj(alpha)*y[i]*conj(x[j])
  conj_alpha: rocblas_double_complex; // conjugate of alpha
  temp1_real, temp1_imag, temp2_real, temp2_imag: Double; // For intermediate complex mult results
begin
  // Initialize alpha
  alpha.x := -1.0; // Real part
  alpha.y := 0.75; // Imaginary part (different value and sign from CHPR2)

  // Calculate conjugate of alpha
  conj_alpha.x := alpha.x;
  conj_alpha.y := -alpha.y; // conj(a + bi) = a - bi

  // For packed format, the size is N*(N+1)/2 elements per matrix
  strideAP := (N * (N + 1)) div 2;
  strideX := N;
  strideY := N;

  SetLength(hostAP, strideAP * batch_count);
  SetLength(hostX, strideX * batch_count);
  SetLength(hostY, strideY * batch_count);
  SetLength(expectedAP, strideAP * batch_count);
  SetLength(hostAP_result, strideAP * batch_count);

  // Initialize AP (packed Hermitian matrix) with values
  // Diagonal elements must be real.
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Packed index
    for j := 0 to N - 1 do // Column index
    begin
      for i := 0 to N - 1 do // Row index
      begin
        // Only store elements in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          idx_packed := b * strideAP + k;
          if i = j then
          begin
            // Diagonal elements are real
            hostAP[idx_packed].x := 2.0 + b + i; // Real part
            hostAP[idx_packed].y := 0.0;         // Imaginary part is 0
          end
          else
          begin
            // Off-diagonal elements
            hostAP[idx_packed].x := (2.0 + b + i + j) * 0.15;    // Real part
            hostAP[idx_packed].y := (2.0 + b + Abs(i - j)) * 0.07; // Imaginary part
          end;
          Inc(k);
        end;
      end;
    end;
  end;

  // Initialize X and Y with complex values
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      hostX[b * strideX + i].x := (i + 1) * 0.5 - b;  // Real part
      hostX[b * strideX + i].y := (i + 1) * 0.25 + b; // Imaginary part

      hostY[b * strideY + i].x := 1.0 + i * 0.25 - b; // Real part
      hostY[b * strideY + i].y := 1.0 + i * 0.125 + b; // Imaginary part
    end;
  end;

  // Calculate expected result manually for each batch
  // First, copy the initial AP matrix
  for i := 0 to High(expectedAP) do
    expectedAP[i] := hostAP[i];

  // Then apply the HPR2 update: AP := AP + alpha * x * y^H + conj(alpha) * y * x^H
  for b := 0 to batch_count - 1 do
  begin
    k := 0; // Reset packed index for this batch
    for j := 0 to N - 1 do
    begin
      for i := 0 to N - 1 do
      begin
        // Only update elements in the specified triangle
        if ((uplo = rocblas_fill_upper) and (i <= j)) or
           ((uplo = rocblas_fill_lower) and (i >= j)) then
        begin
          idx_packed := b * strideAP + k;

          // Calculate alpha * x[i] * conj(y[j])
          // Step 1: alpha * x[i]
          temp1_real := alpha.x * hostX[b * strideX + i].x - alpha.y * hostX[b * strideX + i].y;
          temp1_imag := alpha.x * hostX[b * strideX + i].y + alpha.y * hostX[b * strideX + i].x;
          temp1.x := temp1_real;
          temp1.y := temp1_imag;

          // Step 2: (alpha * x[i]) * conj(y[j])
          // conj(y[j]) = (hostY[b * strideY + j].x, -hostY[b * strideY + j].y)
          temp1_real := temp1.x * hostY[b * strideY + j].x - temp1.y * (-hostY[b * strideY + j].y);
          temp1_imag := temp1.x * (-hostY[b * strideY + j].y) + temp1.y * hostY[b * strideY + j].x;
          temp1.x := temp1_real;
          temp1.y := temp1_imag;

          // Calculate conj(alpha) * y[i] * conj(x[j])
          // Step 1: conj(alpha) * y[i]
          temp2_real := conj_alpha.x * hostY[b * strideY + i].x - conj_alpha.y * hostY[b * strideY + i].y;
          temp2_imag := conj_alpha.x * hostY[b * strideY + i].y + conj_alpha.y * hostY[b * strideY + i].x;
          temp2.x := temp2_real;
          temp2.y := temp2_imag;

          // Step 2: (conj(alpha) * y[i]) * conj(x[j])
          // conj(x[j]) = (hostX[b * strideX + j].x, -hostX[b * strideX + j].y)
          temp2_real := temp2.x * hostX[b * strideX + j].x - temp2.y * (-hostX[b * strideX + j].y);
          temp2_imag := temp2.x * (-hostX[b * strideX + j].y) + temp2.y * hostX[b * strideX + j].x;
          temp2.x := temp2_real;
          temp2.y := temp2_imag;

          expectedAP[idx_packed].x := expectedAP[idx_packed].x + temp1.x + temp2.x;
          expectedAP[idx_packed].y := expectedAP[idx_packed].y + temp1.y + temp2.y;

          Inc(k);
        end;
      end;
    end;
  end;

  // Allocate device memory
  AssertEquals('hipMalloc devAP', HIP_SUCCESS, hipMalloc(devAP, strideAP * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devX', HIP_SUCCESS, hipMalloc(devX, strideX * batch_count * SizeOf(rocblas_double_complex)));
  AssertEquals('hipMalloc devY', HIP_SUCCESS, hipMalloc(devY, strideY * batch_count * SizeOf(rocblas_double_complex)));

  // Copy data to device
  AssertEquals('hipMemcpy AP', HIP_SUCCESS, hipMemcpy(devAP, @hostAP[0], strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy X', HIP_SUCCESS, hipMemcpy(devX, @hostX[0], strideX * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy Y', HIP_SUCCESS, hipMemcpy(devY, @hostY[0], strideY * batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice));

  // Call rocBLAS function
  AssertEquals('rocblas_zhpr2_strided_batched OK',
    Integer(rocblas_status_success),
    Integer(rocblas_zhpr2_strided_batched(
      handle, uplo, N, @alpha,
      devX, incx, strideX, // x vector and its stride
      devY, incy, strideY, // y vector and its stride
      devAP, strideAP,     // AP matrix and its stride
      batch_count
    ))
  );

  // Copy result back to host
  AssertEquals('hipMemcpy AP_result', HIP_SUCCESS, hipMemcpy(@hostAP_result[0], devAP, strideAP * batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost));

  // Verify results
  for i := 0 to High(expectedAP) do
  begin
    CheckEqualsRelative(expectedAP[i].x, hostAP_result[i].x, 1e-12,
      Format('ZHPR2_strided_batched AP[%d].x', [i]));
    CheckEqualsRelative(expectedAP[i].y, hostAP_result[i].y, 1e-12,
      Format('ZHPR2_strided_batched AP[%d].y', [i]));
  end;

  hipFree(devAP);
  hipFree(devX);
  hipFree(devY);
end;

initialization
  RegisterTest(TTestRocBLAS_Level2_strided_batched);

end.
