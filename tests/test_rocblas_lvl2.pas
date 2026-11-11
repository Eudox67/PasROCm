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

unit Test_rocblas_lvl2;

{$mode objfpc}{$H+}
{$linklib rocblas}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  { TTestRocBLAS_Level2 }

  TTestRocBLAS_Level2 = class(TTestRocBLAS_Base)
  published
    procedure Test_SGEMV;
    procedure Test_DGEMV;
    procedure Test_CGEMV;
    procedure Test_ZGEMV;

    procedure Test_SGBMV;
    procedure Test_DGBMV;
    procedure Test_CGBMV;
    procedure Test_ZGBMV;

    procedure Test_CHEMV;
    procedure Test_ZHEMV;
    procedure Test_CHBMV;
    procedure Test_ZHBMV;
    procedure Test_CHPMV;
    procedure Test_ZHPMV;
    procedure Test_SSYMV;

    procedure Test_DSYMV;
    procedure Test_CSYMV;
    procedure Test_ZSYMV;
    procedure Test_SSBMV;
    procedure Test_DSBMV;
    procedure Test_SSPMV;
    procedure Test_DSPMV;
    procedure Test_STRMV;
    procedure Test_DTRMV;
    procedure Test_CTRMV;
    procedure Test_ZTRMV;
    procedure Test_STBMV;
    procedure Test_DTBMV;
    procedure Test_CTBMV;
    procedure Test_ZTBMV;
    procedure Test_STPMV;
    procedure Test_DTPMV;
    procedure Test_CTPMV;
    procedure Test_ZTPMV;
    procedure Test_STRSV;
    procedure Test_DTRSV;
    procedure Test_CTRSV;
    procedure Test_ZTRSV;
    procedure Test_STPSV;
    procedure Test_DTPSV;
    procedure Test_CTPSV;
    procedure Test_ZTPSV;
    procedure Test_STBSV;
    procedure Test_DTBSV;
    procedure Test_CTBSV;
    procedure Test_ZTBSV;
    procedure Test_SGER;
    procedure Test_DGER;
    procedure Test_CGERU;
    procedure Test_ZGERU;
    procedure Test_CGERC;
    procedure Test_ZGERC;
    procedure Test_SSYR;
    procedure Test_DSYR;
    procedure Test_CSYR;
    procedure Test_ZSYR;
    procedure Test_CHER;
    procedure Test_ZHER;
    procedure Test_SSPR;
    procedure Test_DSPR;
    procedure Test_CHPR;
    procedure Test_ZHPR;
    procedure Test_SSYR2;
    procedure Test_DSYR2;
    procedure Test_CSYR2;
    procedure Test_ZSYR2;
    procedure Test_CHER2;
    procedure Test_ZHER2;
    procedure Test_SSPR2;
    procedure Test_DSPR2;
    procedure Test_CHPR2;
    procedure Test_ZHPR2;
  end;


implementation

procedure TTestRocBLAS_Level2.Test_SGEMV;
const
  M = 3; N = 4;
  alpha: Single = 2.0; beta: Single = 1.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array of Single;
  hostX: array of Single;
  hostY, expectedY: array of Single;
  i, j: Integer;
begin
  SetLength(hostA, M*N); SetLength(hostX, N); SetLength(hostY, M); SetLength(expectedY, M);

  for j := 0 to N - 1 do for i := 0 to M - 1 do hostA[j*M + i] := (j+1) * 10 + (i+1);
  for i := 0 to N - 1 do hostX[i] := i + 1.0;
  for i := 0 to M - 1 do hostY[i] := i * 2.0;

  for i := 0 to M - 1 do
  begin
    expectedY[i] := beta * hostY[i];
    for j := 0 to N - 1 do
      expectedY[i] := expectedY[i] + alpha * hostA[j*M + i] * hostX[j];
  end;

  hipMalloc(devA, M*N*SizeOf(Single)); hipMalloc(devX, N*SizeOf(Single)); hipMalloc(devY, M*SizeOf(Single));
  hipMemcpy(devA, @hostA[0], M*N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], M*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_sgemv should return success', Integer(rocblas_status_success), Integer(rocblas_sgemv(handle, rocblas_operation_none, M, N, @alpha, devA, M, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, M*SizeOf(Single), hipMemcpyDeviceToHost);
  for i := 0 to M - 1 do AssertEquals('SGEMV Y[' + IntToStr(i) + ']', expectedY[i], hostY[i], 1e-5);

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_DGEMV;
const
  M = 3; N = 4;
  alpha: Double = 2.0; beta: Double = 1.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array of Double;
  hostX: array of Double;
  hostY, expectedY: array of Double;
  i, j: Integer;
begin
  SetLength(hostA, M*N); SetLength(hostX, N); SetLength(hostY, M); SetLength(expectedY, M);

  for j := 0 to N - 1 do for i := 0 to M - 1 do hostA[j*M + i] := (j+1) * 10 + (i+1);
  for i := 0 to N - 1 do hostX[i] := i + 1.0;
  for i := 0 to M - 1 do hostY[i] := i * 2.0;

  for i := 0 to M - 1 do
  begin
    expectedY[i] := beta * hostY[i];
    for j := 0 to N - 1 do
      expectedY[i] := expectedY[i] + alpha * hostA[j*M + i] * hostX[j];
  end;

  hipMalloc(devA, M*N*SizeOf(Double)); hipMalloc(devX, N*SizeOf(Double)); hipMalloc(devY, M*SizeOf(Double));
  hipMemcpy(devA, @hostA[0], M*N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], M*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dgemv should return success', Integer(rocblas_status_success), Integer(rocblas_dgemv(handle, rocblas_operation_none, M, N, @alpha, devA, M, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, M*SizeOf(Double), hipMemcpyDeviceToHost);
  for i := 0 to M - 1 do AssertEquals('DGEMV Y[' + IntToStr(i) + ']', expectedY[i], hostY[i], 1e-12);

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_CGEMV;
const
  M = 3;
  N = 4;
  alpha: rocblas_float_complex  = (x:  1.5; y: -0.5);
  beta : rocblas_float_complex  = (x: -0.5; y:  1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array[0..11] of rocblas_float_complex;
  hostX: array[0..3]  of rocblas_float_complex;
  hostY, expectedY: array[0..2] of rocblas_float_complex;
  i, j: Integer;
  temp: rocblas_float_complex;
begin
  for j := 0 to N-1 do
    for i := 0 to M-1 do
    begin
      hostA[j*M + i].x :=  (j+1)*10 + (i+1);
      hostA[j*M + i].y := -(j+1)*10 - (i+1);
    end;

  for i := 0 to N-1 do
  begin
    hostX[i].x :=  i + 1.0;
    hostX[i].y := -(i + 1.0);
  end;

  for i := 0 to M-1 do
  begin
    hostY[i].x :=  i * 2.0;
    hostY[i].y := -i * 2.0;
  end;

  for i := 0 to M-1 do
  begin
    expectedY[i] := hostY[i];
    ComplexMul(expectedY[i], beta);
    temp.x := 0; temp.y := 0;
    for j := 0 to N-1 do
      ComplexFMA(temp, hostA[j*M + i], hostX[j], alpha);
    ComplexAdd(expectedY[i], temp);
  end;

  hipMalloc(devA, M*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMalloc(devY, M*SizeOf(rocblas_float_complex));

  hipMemcpy(devA, @hostA[0], M*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], M*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_cgemv OK',
               Integer(rocblas_status_success),
               Integer(rocblas_cgemv(handle,
                                     rocblas_operation_none,
                                     M, N,
                                     @alpha, devA, M,
                                     devX, 1,
                                     @beta,  devY, 1)));

  hipMemcpy(@hostY[0], devY, M*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  for i := 0 to M-1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY[i].x, 1e-5, Format('CGEMV Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY[i].y, 1e-5, Format('CGEMV Y[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;


procedure TTestRocBLAS_Level2.Test_ZGEMV;
const
  M = 3;
  N = 4;
  alpha: rocblas_double_complex = (x:  1.5; y: -0.5);
  beta : rocblas_double_complex = (x: -0.5; y:  1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array[0..11] of rocblas_double_complex;
  hostX: array[0..3]  of rocblas_double_complex;
  hostY, expectedY: array[0..2] of rocblas_double_complex;
  i, j: Integer;
  temp: rocblas_double_complex;
begin
  for j := 0 to N-1 do
    for i := 0 to M-1 do
    begin
      hostA[j*M + i].x :=  (j+1)*10 + (i+1);
      hostA[j*M + i].y := -(j+1)*10 - (i+1);
    end;

  for i := 0 to N-1 do
  begin
    hostX[i].x :=  i + 1.0;
    hostX[i].y := -(i + 1.0);
  end;

  for i := 0 to M-1 do
  begin
    hostY[i].x :=  i * 2.0;
    hostY[i].y := -i * 2.0;
  end;

  for i := 0 to M-1 do
  begin
    expectedY[i] := hostY[i];
    ComplexMul(expectedY[i], beta);
    temp.x := 0; temp.y := 0;
    for j := 0 to N-1 do
      ComplexFMA(temp, hostA[j*M + i], hostX[j], alpha);
    ComplexAdd(expectedY[i], temp);
  end;

  hipMalloc(devA, M*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMalloc(devY, M*SizeOf(rocblas_double_complex));

  hipMemcpy(devA, @hostA[0], M*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], M*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zgemv OK',
               Integer(rocblas_status_success),
               Integer(rocblas_zgemv(handle,
                                     rocblas_operation_none,
                                     M, N,
                                     @alpha, devA, M,
                                     devX, 1,
                                     @beta,  devY, 1)));

  hipMemcpy(@hostY[0], devY, M*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  for i := 0 to M-1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY[i].x, 1e-12, Format('ZGEMV Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY[i].y, 1e-12, Format('ZGEMV Y[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_SGBMV;
const
  M = 5; N = 6; KL = 2; KU = 1;
  alpha: Single =  2.0;
  beta : Single = -0.5;
  LDA = 4; // KL + KU + 1
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array[0..23] of Single;
  hostX: array[0..5]  of Single;
  hostY, expectedY: array[0..4] of Single;
  i, j, diag, row: Integer;
begin
  FillChar(hostA, SizeOf(hostA), 0);
  for j := 0 to N-1 do
    for diag := -KL to KU do
    begin
      row := j + diag;
      if (row >= 0) and (row < M) then
        hostA[(KU+diag) + j*LDA] := (row+1)*10 + (j+1);
    end;

  for i := 0 to N-1 do hostX[i] :=  i + 1.0;
  for i := 0 to M-1 do hostY[i] :=  i * 3.0;

  for i := 0 to M-1 do
  begin
    expectedY[i] := hostY[i] * beta;
    for j := 0 to N-1 do
      if (i >= j-KU) and (i <= j+KL) then
        expectedY[i] := expectedY[i] + alpha * hostA[(KU+(i-j)) + j*LDA] * hostX[j];
  end;

  hipMalloc(devA, LDA*N*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));
  hipMalloc(devY, M*SizeOf(Single));

  hipMemcpy(devA, @hostA[0], LDA*N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], M*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_sgbmv OK',
               Integer(rocblas_status_success),
               Integer(rocblas_sgbmv(handle,
                                     rocblas_operation_none,
                                     M, N, KL, KU,
                                     @alpha, devA, LDA,
                                     devX, 1,
                                     @beta,  devY, 1)));

  hipMemcpy(@hostY[0], devY, M*SizeOf(Single), hipMemcpyDeviceToHost);
  for i := 0 to M-1 do
    CheckEqualsRelative(expectedY[i], hostY[i], 1e-5, Format('SGBMV Y[%d]', [i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;


procedure TTestRocBLAS_Level2.Test_DGBMV;
const
  M = 5; N = 6; KL = 2; KU = 1;
  alpha: Double =  2.0;
  beta : Double = -0.5;
  LDA = 4;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array[0..23] of Double;
  hostX: array[0..5]   of Double;
  hostY, expectedY       : array[0..4] of Double;
  i, j, diag, row: Integer;
begin
  FillChar(hostA, SizeOf(hostA), 0);
  for j := 0 to N-1 do
    for diag := -KL to KU do
    begin
      row := j + diag;
      if (row >= 0) and (row < M) then
        hostA[(KU+diag) + j*LDA] := (row+1)*10 + (j+1);
    end;

  for i := 0 to N-1 do hostX[i] :=  i + 1.0;
  for i := 0 to M-1 do hostY[i] :=  i * 3.0;

  for i := 0 to M-1 do
  begin
    expectedY[i] := hostY[i] * beta;
    for j := 0 to N-1 do
      if (i >= j-KU) and (i <= j+KL) then
        expectedY[i] := expectedY[i] + alpha * hostA[(KU+(i-j)) + j*LDA] * hostX[j];
  end;

  hipMalloc(devA, LDA*N*SizeOf(Double));
  hipMalloc(devX, N    *SizeOf(Double));
  hipMalloc(devY, M    *SizeOf(Double));

  hipMemcpy(devA, @hostA[0], LDA*N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N    *SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], M    *SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dgbmv OK',
               Integer(rocblas_status_success),
               Integer(rocblas_dgbmv(handle,
                                     rocblas_operation_none,
                                     M, N, KL, KU,
                                     @alpha, devA, LDA,
                                     devX, 1,
                                     @beta,  devY, 1)));

  hipMemcpy(@hostY[0], devY, M*SizeOf(Double), hipMemcpyDeviceToHost);
  for i := 0 to M-1 do
    CheckEqualsRelative(expectedY[i], hostY[i], 1e-12, Format('DGBMV Y[%d]', [i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;


procedure TTestRocBLAS_Level2.Test_CGBMV;
const
  M = 5; N = 6; KL = 2; KU = 1;
  alpha: rocblas_float_complex  = (x:  1.0; y: -0.5);
  beta : rocblas_float_complex  = (x: -0.5; y:  1.0);
  LDA = 4; // KL + KU + 1
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array[0..23] of rocblas_float_complex;
  hostX: array[0..5]  of rocblas_float_complex;
  hostY, expectedY: array[0..4] of rocblas_float_complex;
  i, j, diag, row: Integer;
  temp: rocblas_float_complex;
begin
  FillChar(hostA, SizeOf(hostA), 0);
  for j := 0 to N-1 do
    for diag := -KL to KU do
    begin
      row := j + diag;
      if (row >= 0) and (row < M) then
      begin
        hostA[(KU + diag) + j*LDA].x :=  (row+1)*10 + (j+1);
        hostA[(KU + diag) + j*LDA].y := -(row+1)*10 - (j+1);
      end;
    end;

  for i := 0 to N-1 do
  begin
    hostX[i].x :=  i + 1.0;
    hostX[i].y := -(i + 1.0);
  end;

  for i := 0 to M-1 do
  begin
    hostY[i].x :=  i * 3.0;
    hostY[i].y := -i * 3.0;
  end;

  for i := 0 to M-1 do
  begin
    expectedY[i] := hostY[i];
    ComplexMul(expectedY[i], beta);

    temp.x := 0; temp.y := 0;
    for j := 0 to N-1 do
      if (i >= j-KU) and (i <= j+KL) then
        ComplexFMA(temp, hostA[(KU+(i-j)) + j*LDA], hostX[j], alpha);
    ComplexAdd(expectedY[i], temp);
  end;

  hipMalloc(devA, LDA*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMalloc(devY, M*SizeOf(rocblas_float_complex));

  hipMemcpy(devA, @hostA[0], LDA*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], M*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_cgbmv OK',
             Integer(rocblas_status_success),
             Integer(rocblas_cgbmv(handle,
                                   rocblas_operation_none,
                                   M, N, KL, KU, // Corrected call with KU
                                   @alpha, devA, LDA,
                                   devX, 1,
                                   @beta,  devY, 1)));

  hipMemcpy(@hostY[0], devY, M*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to M-1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY[i].x, 1e-5, Format('CGBMV Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY[i].y, 1e-5, Format('CGBMV Y[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_ZGBMV;
const
  M = 5; N = 6; KL = 2; KU = 1;
  alpha: rocblas_double_complex = (x:  1.0; y: -0.5);
  beta : rocblas_double_complex = (x: -0.5; y:  1.0);
  LDA = 4;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array[0..23] of rocblas_double_complex;
  hostX: array[0..5]   of rocblas_double_complex;
  hostY, expectedY       : array[0..4] of rocblas_double_complex;
  i, j, diag, row: Integer;
  temp: rocblas_double_complex;
begin
  FillChar(hostA, SizeOf(hostA), 0);
  for j := 0 to N-1 do
    for diag := -KL to KU do
    begin
      row := j + diag;
      if (row >= 0) and (row < M) then
      begin
        hostA[(KU + diag) + j*LDA].x :=  (row+1)*10 + (j+1);
        hostA[(KU + diag) + j*LDA].y := -(row+1)*10 - (j+1);
      end;
    end;

  for i := 0 to N-1 do
  begin
    hostX[i].x :=  i + 1.0;
    hostX[i].y := -(i + 1.0);
  end;

  for i := 0 to M-1 do
  begin
    hostY[i].x :=  i * 3.0;
    hostY[i].y := -i * 3.0;
  end;

  for i := 0 to M-1 do
  begin
    expectedY[i] := hostY[i];
    ComplexMul(expectedY[i], beta);

    temp.x := 0; temp.y := 0;
    for j := 0 to N-1 do
      if (i >= j-KU) and (i <= j+KL) then
        ComplexFMA(temp, hostA[(KU+(i-j)) + j*LDA], hostX[j], alpha);
    ComplexAdd(expectedY[i], temp);
  end;

  hipMalloc(devA, LDA*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N    *SizeOf(rocblas_double_complex));
  hipMalloc(devY, M    *SizeOf(rocblas_double_complex));

  hipMemcpy(devA, @hostA[0], LDA*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N    *SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], M    *SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zgbmv OK',
               Integer(rocblas_status_success),
               Integer(rocblas_zgbmv(handle,
                                     rocblas_operation_none,
                                     M, N, KL, KU, // Corrected call with KU
                                     @alpha, devA, LDA,
                                     devX, 1,
                                     @beta,  devY, 1)));

  hipMemcpy(@hostY[0], devY, M*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to M-1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY[i].x, 1e-12, Format('ZGBMV Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY[i].y, 1e-12, Format('ZGBMV Y[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_CHEMV;
const
  N = 5;
  alpha: rocblas_float_complex = (x: 1.5; y: -0.5);
  beta:  rocblas_float_complex = (x: -0.5; y: 1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array[0..N*N-1] of rocblas_float_complex;
  hostX: array[0..N-1]   of rocblas_float_complex;
  hostY, expectedY: array[0..N-1] of rocblas_float_complex;
  i, j: Integer;
  temp: rocblas_float_complex;
begin
  // Initialize Hermitian matrix (storing full matrix for easy CPU reference)
  for j := 0 to N-1 do
    for i := 0 to N-1 do
    begin
      hostA[j*N + i].x := (i+1) + (j+1)*10;
      hostA[j*N + i].y := -((i+1) + (j+1)*5);
    end;
  for i := 0 to N-1 do hostA[i*N + i].y := 0; // Diag is real
  for j := 0 to N-1 do
    for i := j + 1 to N-1 do
    begin
      hostA[j*N + i].x :=  hostA[i*N + j].x; // Conjugate transpose
      hostA[j*N + i].y := -hostA[i*N + j].y;
    end;

  for i := 0 to N-1 do begin hostX[i].x := i + 1.0; hostX[i].y := -(i + 1.0); end;
  for i := 0 to N-1 do begin hostY[i].x := i * 2.0; hostY[i].y := -(i * 2.0); end;

  // CPU reference y = alpha*A*x + beta*y
  for i := 0 to N-1 do
  begin
    expectedY[i] := hostY[i];
    ComplexMul(expectedY[i], beta);
    temp.x := 0; temp.y := 0;
    for j := 0 to N-1 do
      ComplexFMA(temp, hostA[j*N+i], hostX[j], alpha);
    ComplexAdd(expectedY[i], temp);
  end;

  hipMalloc(devA, N*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMalloc(devY, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_chemv OK', Integer(rocblas_status_success),
    Integer(rocblas_chemv(handle, rocblas_fill_upper, N, @alpha, devA, N, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY[i].x, 1e-5, Format('CHEMV Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY[i].y, 1e-5, Format('CHEMV Y[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_ZHEMV;
const
  N = 5;
  alpha: rocblas_double_complex = (x: 1.5; y: -0.5);
  beta:  rocblas_double_complex = (x: -0.5; y: 1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array[0..N*N-1] of rocblas_double_complex;
  hostX: array[0..N-1]   of rocblas_double_complex;
  hostY, expectedY: array[0..N-1] of rocblas_double_complex;
  i, j: Integer;
  temp: rocblas_double_complex;
begin
  // Initialize Hermitian matrix (storing full matrix for easy CPU reference)
  for j := 0 to N-1 do
    for i := 0 to N-1 do
    begin
      hostA[j*N + i].x := (i+1) + (j+1)*10;
      hostA[j*N + i].y := -((i+1) + (j+1)*5);
    end;
  for i := 0 to N-1 do hostA[i*N + i].y := 0; // Diag is real
  for j := 0 to N-1 do
    for i := j + 1 to N-1 do
    begin
      hostA[j*N + i].x :=  hostA[i*N + j].x; // Conjugate transpose
      hostA[j*N + i].y := -hostA[i*N + j].y;
    end;

  for i := 0 to N-1 do begin hostX[i].x := i + 1.0; hostX[i].y := -(i + 1.0); end;
  for i := 0 to N-1 do begin hostY[i].x := i * 2.0; hostY[i].y := -(i * 2.0); end;

  // CPU reference y = alpha*A*x + beta*y
  for i := 0 to N-1 do
  begin
    expectedY[i] := hostY[i];
    ComplexMul(expectedY[i], beta);
    temp.x := 0; temp.y := 0;
    for j := 0 to N-1 do
      ComplexFMA(temp, hostA[j*N+i], hostX[j], alpha);
    ComplexAdd(expectedY[i], temp);
  end;

  hipMalloc(devA, N*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMalloc(devY, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zhemv OK', Integer(rocblas_status_success),
    Integer(rocblas_zhemv(handle, rocblas_fill_upper, N, @alpha, devA, N, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY[i].x, 1e-12, Format('ZHEMV Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY[i].y, 1e-12, Format('ZHEMV Y[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_CHBMV;
const
  N = 6; K = 2; // N=Order, K=super-diagonals
  LDA = 3;     // LDA >= K+1
  alpha: rocblas_float_complex = (x: 1.5; y: -0.5);
  beta:  rocblas_float_complex = (x: -0.5; y: 1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA_full: array of rocblas_float_complex;
  hostA_band: array[0..LDA*N-1] of rocblas_float_complex;
  hostX: array[0..N-1] of rocblas_float_complex;
  hostY, expectedY: array[0..N-1] of rocblas_float_complex;
  i, j: Integer;
  temp: rocblas_float_complex;
begin
  SetLength(hostA_full, N*N);

  // 1. Create a full reference Hermitian matrix
  for j := 0 to N-1 do
    for i := 0 to N-1 do
      if abs(i-j) <= K then
      begin
        hostA_full[j*N+i].x := i+j+1;
        hostA_full[j*N+i].y := i-j-1;
      end else
      begin
        hostA_full[j*N+i].x := 0;
        hostA_full[j*N+i].y := 0;
      end;
  for i := 0 to N-1 do hostA_full[i*N+i].y := 0; // Real diagonal
  for j := 0 to N-1 do for i := j+1 to N-1 do
  begin
      hostA_full[j*N+i].x := hostA_full[i*N+j].x;
      hostA_full[j*N+i].y := -hostA_full[i*N+j].y;
  end;

  // 2. Convert the full matrix to the required band storage format (upper)
  FillChar(hostA_band, SizeOf(hostA_band), 0);
  for j := 0 to N-1 do
    for i := max(0, j-K) to j do
      hostA_band[(K+i-j) + j*LDA] := hostA_full[j*N+i];

  // 3. Initialize vectors
  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+1); end;
  for i := 0 to N-1 do begin hostY[i].x := i*2; hostY[i].y := -(i*2); end;

  // 4. Calculate CPU reference using the full matrix
  for i := 0 to N-1 do
  begin
    expectedY[i] := hostY[i];
    ComplexMul(expectedY[i], beta);
    temp.x := 0; temp.y := 0;
    for j := 0 to N-1 do
      ComplexFMA(temp, hostA_full[j*N+i], hostX[j], alpha);
    ComplexAdd(expectedY[i], temp);
  end;

  // 5. ROCm execution
  hipMalloc(devA, LDA*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMalloc(devY, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devA, @hostA_band[0], LDA*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_chbmv OK', Integer(rocblas_status_success),
    Integer(rocblas_chbmv(handle, rocblas_fill_upper, N, K, @alpha, devA, LDA, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY[i].x, 1e-5, Format('CHBMV Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY[i].y, 1e-5, Format('CHBMV Y[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_ZHBMV;
const
  N = 6; K = 2; // N=Order, K=super-diagonals
  LDA = 3;     // LDA >= K+1
  alpha: rocblas_double_complex = (x: 1.5; y: -0.5);
  beta:  rocblas_double_complex = (x: -0.5; y: 1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA_full: array of rocblas_double_complex;
  hostA_band: array[0..LDA*N-1] of rocblas_double_complex;
  hostX: array[0..N-1] of rocblas_double_complex;
  hostY, expectedY: array[0..N-1] of rocblas_double_complex;
  i, j: Integer;
  temp: rocblas_double_complex;
begin
  SetLength(hostA_full, N*N);

  for j := 0 to N-1 do
    for i := 0 to N-1 do
      if abs(i-j) <= K then
      begin
        hostA_full[j*N+i].x := i+j+1;
        hostA_full[j*N+i].y := i-j-1;
      end else
      begin
        hostA_full[j*N+i].x := 0;
        hostA_full[j*N+i].y := 0;
      end;
  for i := 0 to N-1 do hostA_full[i*N+i].y := 0;
  for j := 0 to N-1 do for i := j+1 to N-1 do
  begin
      hostA_full[j*N+i].x := hostA_full[i*N+j].x;
      hostA_full[j*N+i].y := -hostA_full[i*N+j].y;
  end;

  FillChar(hostA_band, SizeOf(hostA_band), 0);
  for j := 0 to N-1 do
    for i := max(0, j-K) to j do
      hostA_band[(K+i-j) + j*LDA] := hostA_full[j*N+i];

  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+1); end;
  for i := 0 to N-1 do begin hostY[i].x := i*2; hostY[i].y := -(i*2); end;

  for i := 0 to N-1 do
  begin
    expectedY[i] := hostY[i];
    ComplexMul(expectedY[i], beta);
    temp.x := 0; temp.y := 0;
    for j := 0 to N-1 do
      ComplexFMA(temp, hostA_full[j*N+i], hostX[j], alpha);
    ComplexAdd(expectedY[i], temp);
  end;

  hipMalloc(devA, LDA*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMalloc(devY, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devA, @hostA_band[0], LDA*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zhbmv OK', Integer(rocblas_status_success),
    Integer(rocblas_zhbmv(handle, rocblas_fill_upper, N, K, @alpha, devA, LDA, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY[i].x, 1e-12, Format('ZHBMV Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY[i].y, 1e-12, Format('ZHBMV Y[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_CHPMV;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
  alpha: rocblas_float_complex = (x: 1.5; y: -0.5);
  beta:  rocblas_float_complex = (x: -0.5; y: 1.0);
var
  devAP, devX, devY: hipDeviceptr_t;
  hostA_full: array of rocblas_float_complex;
  hostAP: array[0..PACKED_SIZE-1] of rocblas_float_complex;
  hostX: array[0..N-1] of rocblas_float_complex;
  hostY, expectedY: array[0..N-1] of rocblas_float_complex;
  i, j, k: Integer;
  temp: rocblas_float_complex;
begin
  SetLength(hostA_full, N*N);

  // 1. Create a full reference Hermitian matrix
  for j := 0 to N-1 do for i := 0 to N-1 do
  begin
    hostA_full[j*N+i].x := i+j+1;
    hostA_full[j*N+i].y := i-j-1;
  end;
  for i := 0 to N-1 do hostA_full[i*N+i].y := 0; // Real diagonal
  for j := 0 to N-1 do for i := j+1 to N-1 do
  begin
      hostA_full[j*N+i].x := hostA_full[i*N+j].x;
      hostA_full[j*N+i].y := -hostA_full[i*N+j].y;
  end;

  // 2. Convert to packed storage (upper)
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := hostA_full[j*N+i];
      Inc(k);
    end;

  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+1); end;
  for i := 0 to N-1 do begin hostY[i].x := i*2; hostY[i].y := -(i*2); end;

  // 3. Calculate CPU reference
  for i := 0 to N-1 do
  begin
    expectedY[i] := hostY[i];
    ComplexMul(expectedY[i], beta);
    temp.x := 0; temp.y := 0;
    for j := 0 to N-1 do
      ComplexFMA(temp, hostA_full[j*N+i], hostX[j], alpha);
    ComplexAdd(expectedY[i], temp);
  end;

  // 4. ROCm execution
  hipMalloc(devAP, PACKED_SIZE*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMalloc(devY, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_chpmv OK', Integer(rocblas_status_success),
    Integer(rocblas_chpmv(handle, rocblas_fill_upper, N, @alpha, devAP, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY[i].x, 1e-5, Format('CHPMV Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY[i].y, 1e-5, Format('CHPMV Y[%d].y', [i]));
  end;

  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_ZHPMV;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
  alpha: rocblas_double_complex = (x: 1.5; y: -0.5);
  beta:  rocblas_double_complex = (x: -0.5; y: 1.0);
var
  devAP, devX, devY: hipDeviceptr_t;
  hostA_full: array of rocblas_double_complex;
  hostAP: array[0..PACKED_SIZE-1] of rocblas_double_complex;
  hostX: array[0..N-1] of rocblas_double_complex;
  hostY, expectedY: array[0..N-1] of rocblas_double_complex;
  i, j, k: Integer;
  temp: rocblas_double_complex;
begin
  SetLength(hostA_full, N*N);

  for j := 0 to N-1 do for i := 0 to N-1 do
  begin
    hostA_full[j*N+i].x := i+j+1;
    hostA_full[j*N+i].y := i-j-1;
  end;
  for i := 0 to N-1 do hostA_full[i*N+i].y := 0;
  for j := 0 to N-1 do for i := j+1 to N-1 do
  begin
      hostA_full[j*N+i].x := hostA_full[i*N+j].x;
      hostA_full[j*N+i].y := -hostA_full[i*N+j].y;
  end;

  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := hostA_full[j*N+i];
      Inc(k);
    end;

  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+1); end;
  for i := 0 to N-1 do begin hostY[i].x := i*2; hostY[i].y := -(i*2); end;

  for i := 0 to N-1 do
  begin
    expectedY[i] := hostY[i];
    ComplexMul(expectedY[i], beta);
    temp.x := 0; temp.y := 0;
    for j := 0 to N-1 do
      ComplexFMA(temp, hostA_full[j*N+i], hostX[j], alpha);
    ComplexAdd(expectedY[i], temp);
  end;

  hipMalloc(devAP, PACKED_SIZE*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMalloc(devY, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zhpmv OK', Integer(rocblas_status_success),
    Integer(rocblas_zhpmv(handle, rocblas_fill_upper, N, @alpha, devAP, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY[i].x, 1e-12, Format('ZHPMV Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY[i].y, 1e-12, Format('ZHPMV Y[%d].y', [i]));
  end;

  hipFree(devAP); hipFree(devX); hipFree(devY);
end;
procedure TTestRocBLAS_Level2.Test_SSYMV;
const
  N = 5;
  alpha: Single = 1.5;
  beta:  Single = -0.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array[0..N*N-1] of Single;
  hostX: array[0..N-1]   of Single;
  hostY, expectedY: array[0..N-1] of Single;
  i, j: Integer;
begin
  // Initialize symmetric matrix (storing full matrix for easy CPU reference)
  for j := 0 to N-1 do
    for i := 0 to N-1 do
      hostA[j*N + i] := (i+1) + (j+1)*10;
  for i := 0 to N-1 do hostA[i*N + i] := i + 1; // Diagonal
  for j := 0 to N-1 do
    for i := j + 1 to N-1 do
      hostA[j*N + i] := hostA[i*N + j]; // Make symmetric

  for i := 0 to N-1 do hostX[i] := i + 1.0;
  for i := 0 to N-1 do hostY[i] := i * 2.0;

  // CPU reference y = alpha*A*x + beta*y
  for i := 0 to N-1 do
  begin
    expectedY[i] := beta * hostY[i];
    for j := 0 to N-1 do
      expectedY[i] := expectedY[i] + alpha * hostA[j*N + i] * hostX[j];
  end;

  hipMalloc(devA, N*N*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));
  hipMalloc(devY, N*SizeOf(Single));

  hipMemcpy(devA, @hostA[0], N*N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ssymv OK', Integer(rocblas_status_success),
    Integer(rocblas_ssymv(handle, rocblas_fill_upper, N, @alpha, devA, N, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(Single), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(expectedY[i], hostY[i], 1e-5, Format('SSYMV Y[%d]',[i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_DSYMV;
const
  N = 5;
  alpha: Double = 1.5;
  beta:  Double = -0.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array[0..N*N-1] of Double;
  hostX: array[0..N-1]   of Double;
  hostY, expectedY: array[0..N-1] of Double;
  i, j: Integer;
begin
  // Initialize symmetric matrix (storing full matrix for easy CPU reference)
  for j := 0 to N-1 do
    for i := 0 to N-1 do
      hostA[j*N + i] := (i+1) + (j+1)*10;
  for i := 0 to N-1 do hostA[i*N + i] := i + 1; // Diagonal
  for j := 0 to N-1 do
    for i := j + 1 to N-1 do
      hostA[j*N + i] := hostA[i*N + j]; // Make symmetric

  for i := 0 to N-1 do hostX[i] := i + 1.0;
  for i := 0 to N-1 do hostY[i] := i * 2.0;

  // CPU reference y = alpha*A*x + beta*y
  for i := 0 to N-1 do
  begin
    expectedY[i] := beta * hostY[i];
    for j := 0 to N-1 do
      expectedY[i] := expectedY[i] + alpha * hostA[j*N + i] * hostX[j];
  end;

  hipMalloc(devA, N*N*SizeOf(Double));
  hipMalloc(devX, N*SizeOf(Double));
  hipMalloc(devY, N*SizeOf(Double));

  hipMemcpy(devA, @hostA[0], N*N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dsymv OK', Integer(rocblas_status_success),
    Integer(rocblas_dsymv(handle, rocblas_fill_upper, N, @alpha, devA, N, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(Double), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(expectedY[i], hostY[i], 1e-12, Format('DSYMV Y[%d]',[i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_CSYMV;
const
  N = 5;
  alpha: rocblas_float_complex = (x: 1.5; y: -0.5);
  beta:  rocblas_float_complex = (x: -0.5; y: 1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array[0..N*N-1] of rocblas_float_complex;
  hostX: array[0..N-1]   of rocblas_float_complex;
  hostY, expectedY: array[0..N-1] of rocblas_float_complex;
  i, j: Integer;
  temp: rocblas_float_complex;
begin
  // Initialize symmetric matrix (A = A**T)
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostA[j*N + i].x := (i+1) + (j+1)*10;
      hostA[j*N + i].y := -((i+1) + (j+1)*5);
      hostA[i*N + j] := hostA[j*N + i]; // Symmetry
    end;

  for i := 0 to N-1 do begin hostX[i].x := i + 1.0; hostX[i].y := -(i + 1.0); end;
  for i := 0 to N-1 do begin hostY[i].x := i * 2.0; hostY[i].y := -(i * 2.0); end;

  // CPU reference y = alpha*A*x + beta*y
  for i := 0 to N-1 do
  begin
    expectedY[i] := hostY[i];
    ComplexMul(expectedY[i], beta);
    temp.x := 0; temp.y := 0;
    for j := 0 to N-1 do
      ComplexFMA(temp, hostA[j*N+i], hostX[j], alpha);
    ComplexAdd(expectedY[i], temp);
  end;

  hipMalloc(devA, N*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMalloc(devY, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_csymv OK', Integer(rocblas_status_success),
    Integer(rocblas_csymv(handle, rocblas_fill_upper, N, @alpha, devA, N, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY[i].x, 1e-5, Format('CSYMV Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY[i].y, 1e-5, Format('CSYMV Y[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_ZSYMV;
const
  N = 5;
  alpha: rocblas_double_complex = (x: 1.5; y: -0.5);
  beta:  rocblas_double_complex = (x: -0.5; y: 1.0);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA: array[0..N*N-1] of rocblas_double_complex;
  hostX: array[0..N-1]   of rocblas_double_complex;
  hostY, expectedY: array[0..N-1] of rocblas_double_complex;
  i, j: Integer;
  temp: rocblas_double_complex;
begin
  // Initialize symmetric matrix (A = A**T)
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostA[j*N + i].x := (i+1) + (j+1)*10;
      hostA[j*N + i].y := -((i+1) + (j+1)*5);
      hostA[i*N + j] := hostA[j*N + i]; // Symmetry
    end;

  for i := 0 to N-1 do begin hostX[i].x := i + 1.0; hostX[i].y := -(i + 1.0); end;
  for i := 0 to N-1 do begin hostY[i].x := i * 2.0; hostY[i].y := -(i * 2.0); end;

  // CPU reference y = alpha*A*x + beta*y
  for i := 0 to N-1 do
  begin
    expectedY[i] := hostY[i];
    ComplexMul(expectedY[i], beta);
    temp.x := 0; temp.y := 0;
    for j := 0 to N-1 do
      ComplexFMA(temp, hostA[j*N+i], hostX[j], alpha);
    ComplexAdd(expectedY[i], temp);
  end;

  hipMalloc(devA, N*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMalloc(devY, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zsymv OK', Integer(rocblas_status_success),
    Integer(rocblas_zsymv(handle, rocblas_fill_upper, N, @alpha, devA, N, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedY[i].x, hostY[i].x, 1e-12, Format('ZSYMV Y[%d].x', [i]));
    CheckEqualsRelative(expectedY[i].y, hostY[i].y, 1e-12, Format('ZSYMV Y[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_SSBMV;
const
  N = 6; K = 2; // N=Order, K=super/sub-diagonals
  LDA = 3;     // LDA >= K+1
  alpha: Single = 1.5;
  beta:  Single = -0.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA_full: array of Single;
  hostA_band: array[0..LDA*N-1] of Single;
  hostX: array[0..N-1] of Single;
  hostY, expectedY: array[0..N-1] of Single;
  i, j: Integer;
begin
  SetLength(hostA_full, N*N);

  // 1. Create a full reference symmetric banded matrix
  for j := 0 to N-1 do
    for i := 0 to j do
      if (j-i) <= K then
      begin
        hostA_full[j*N+i] := i+j+2;
        hostA_full[i*N+j] := i+j+2; // Symmetry
      end else
      begin
        hostA_full[j*N+i] := 0;
        hostA_full[i*N+j] := 0;
      end;

  // 2. Convert to symmetric band storage (upper)
  FillChar(hostA_band, SizeOf(hostA_band), 0);
  for j := 0 to N-1 do
    for i := max(0, j-K) to j do
      hostA_band[(K+i-j) + j*LDA] := hostA_full[j*N+i];

  // 3. Initialize vectors
  for i := 0 to N-1 do hostX[i] := i + 1.0;
  for i := 0 to N-1 do hostY[i] := i * 2.0;

  // 4. Calculate CPU reference
  for i := 0 to N-1 do
  begin
    expectedY[i] := beta * hostY[i];
    for j := 0 to N-1 do
      expectedY[i] := expectedY[i] + alpha * hostA_full[j*N + i] * hostX[j];
  end;

  // 5. ROCm execution
  hipMalloc(devA, LDA*N*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));
  hipMalloc(devY, N*SizeOf(Single));
  hipMemcpy(devA, @hostA_band[0], LDA*N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ssbmv OK', Integer(rocblas_status_success),
    Integer(rocblas_ssbmv(handle, rocblas_fill_upper, N, K, @alpha, devA, LDA, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(Single), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(expectedY[i], hostY[i], 1e-5, Format('SSBMV Y[%d]', [i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_DSBMV;
const
  N = 6; K = 2;
  LDA = 3;
  alpha: Double = 1.5;
  beta:  Double = -0.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA_full: array of Double;
  hostA_band: array[0..LDA*N-1] of Double;
  hostX: array[0..N-1] of Double;
  hostY, expectedY: array[0..N-1] of Double;
  i, j: Integer;
begin
  SetLength(hostA_full, N*N);

  for j := 0 to N-1 do
    for i := 0 to j do
      if (j-i) <= K then
      begin
        hostA_full[j*N+i] := i+j+2;
        hostA_full[i*N+j] := i+j+2;
      end else
      begin
        hostA_full[j*N+i] := 0;
        hostA_full[i*N+j] := 0;
      end;

  FillChar(hostA_band, SizeOf(hostA_band), 0);
  for j := 0 to N-1 do
    for i := max(0, j-K) to j do
      hostA_band[(K+i-j) + j*LDA] := hostA_full[j*N+i];

  for i := 0 to N-1 do hostX[i] := i + 1.0;
  for i := 0 to N-1 do hostY[i] := i * 2.0;

  for i := 0 to N-1 do
  begin
    expectedY[i] := beta * hostY[i];
    for j := 0 to N-1 do
      expectedY[i] := expectedY[i] + alpha * hostA_full[j*N + i] * hostX[j];
  end;

  hipMalloc(devA, LDA*N*SizeOf(Double));
  hipMalloc(devX, N*SizeOf(Double));
  hipMalloc(devY, N*SizeOf(Double));
  hipMemcpy(devA, @hostA_band[0], LDA*N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dsbmv OK', Integer(rocblas_status_success),
    Integer(rocblas_dsbmv(handle, rocblas_fill_upper, N, K, @alpha, devA, LDA, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(Double), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(expectedY[i], hostY[i], 1e-12, Format('DSBMV Y[%d]', [i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_SSPMV;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
  alpha: Single = 1.5;
  beta:  Single = -0.5;
var
  devAP, devX, devY: hipDeviceptr_t;
  hostA_full: array of Single;
  hostAP: array[0..PACKED_SIZE-1] of Single;
  hostX: array[0..N-1] of Single;
  hostY, expectedY: array[0..N-1] of Single;
  i, j, k: Integer;
begin
  SetLength(hostA_full, N*N);

  // 1. Create a full symmetric matrix
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostA_full[j*N+i] := i+j+2;
      hostA_full[i*N+j] := i+j+2;
    end;

  // 2. Convert to packed storage (upper)
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := hostA_full[j*N+i];
      Inc(k);
    end;

  // 3. Initialize vectors
  for i := 0 to N-1 do hostX[i] := i + 1.0;
  for i := 0 to N-1 do hostY[i] := i * 2.0;

  // 4. CPU reference
  for i := 0 to N-1 do
  begin
    expectedY[i] := beta * hostY[i];
    for j := 0 to N-1 do
      expectedY[i] := expectedY[i] + alpha * hostA_full[j*N + i] * hostX[j];
  end;

  // 5. ROCm execution
  hipMalloc(devAP, PACKED_SIZE*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));
  hipMalloc(devY, N*SizeOf(Single));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_sspmv OK', Integer(rocblas_status_success),
    Integer(rocblas_sspmv(handle, rocblas_fill_upper, N, @alpha, devAP, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(Single), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(expectedY[i], hostY[i], 1e-5, Format('SSPMV Y[%d]', [i]));

  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_DSPMV;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
  alpha: Double = 1.5;
  beta:  Double = -0.5;
var
  devAP, devX, devY: hipDeviceptr_t;
  hostA_full: array of Double;
  hostAP: array[0..PACKED_SIZE-1] of Double;
  hostX: array[0..N-1] of Double;
  hostY, expectedY: array[0..N-1] of Double;
  i, j, k: Integer;
begin
  SetLength(hostA_full, N*N);

  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostA_full[j*N+i] := i+j+2;
      hostA_full[i*N+j] := i+j+2;
    end;

  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := hostA_full[j*N+i];
      Inc(k);
    end;

  for i := 0 to N-1 do hostX[i] := i + 1.0;
  for i := 0 to N-1 do hostY[i] := i * 2.0;

  for i := 0 to N-1 do
  begin
    expectedY[i] := beta * hostY[i];
    for j := 0 to N-1 do
      expectedY[i] := expectedY[i] + alpha * hostA_full[j*N + i] * hostX[j];
  end;

  hipMalloc(devAP, PACKED_SIZE*SizeOf(Double));
  hipMalloc(devX, N*SizeOf(Double));
  hipMalloc(devY, N*SizeOf(Double));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dspmv OK', Integer(rocblas_status_success),
    Integer(rocblas_dspmv(handle, rocblas_fill_upper, N, @alpha, devAP, devX, 1, @beta, devY, 1)));

  hipMemcpy(@hostY[0], devY, N*SizeOf(Double), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(expectedY[i], hostY[i], 1e-12, Format('DSPMV Y[%d]', [i]));

  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_STRMV;
const
  N = 5;
var
  devA, devX: hipDeviceptr_t;
  hostA: array[0..N*N-1] of Single;
  hostX_orig, expectedX, hostX_result: array[0..N-1] of Single;
  i, j: Integer;
begin
  // 1. Initialize upper triangular matrix A and original vector x
  FillChar(hostA, SizeOf(hostA), 0);
  for j := 0 to N-1 do
    for i := 0 to j do
      hostA[j*N + i] := i + j + 2;
  for i := 0 to N-1 do hostX_orig[i] := i + 1.0;

  // 2. Calculate CPU reference: expectedX = A * hostX_orig
  FillChar(expectedX, SizeOf(expectedX), 0);
  for j := 0 to N-1 do
    for i := 0 to j do // Upper triangular part
      expectedX[i] := expectedX[i] + hostA[j*N + i] * hostX_orig[j];

  // 3. ROCm execution
  hipMalloc(devA, N*N*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX_orig[0], N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_strmv OK', Integer(rocblas_status_success),
    Integer(rocblas_strmv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devA, N, devX, 1)));

  // 4. Verify result
  hipMemcpy(@hostX_result[0], devX, N*SizeOf(Single), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-5, Format('STRMV X[%d]',[i]));

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_DTRMV;
const
  N = 5;
var
  devA, devX: hipDeviceptr_t;
  hostA: array[0..N*N-1] of Double;
  hostX_orig, expectedX, hostX_result: array[0..N-1] of Double;
  i, j: Integer;
begin
  FillChar(hostA, SizeOf(hostA), 0);
  for j := 0 to N-1 do
    for i := 0 to j do
      hostA[j*N + i] := i + j + 2;
  for i := 0 to N-1 do hostX_orig[i] := i + 1.0;

  FillChar(expectedX, SizeOf(expectedX), 0);
  for j := 0 to N-1 do
    for i := 0 to j do
      expectedX[i] := expectedX[i] + hostA[j*N + i] * hostX_orig[j];

  hipMalloc(devA, N*N*SizeOf(Double));
  hipMalloc(devX, N*SizeOf(Double));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX_orig[0], N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dtrmv OK', Integer(rocblas_status_success),
    Integer(rocblas_dtrmv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devA, N, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(Double), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTRMV X[%d]',[i]));

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_CTRMV;
const
  N = 5;
var
  devA, devX: hipDeviceptr_t;
  hostA: array[0..N*N-1] of rocblas_float_complex;
  hostX_orig, expectedX, hostX_result: array[0..N-1] of rocblas_float_complex;
  i, j: Integer;
  term: rocblas_float_complex;
begin
  FillChar(hostA, SizeOf(hostA), 0);
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostA[j*N + i].x := i + j + 2;
      hostA[j*N + i].y := (i-j)/2.0;
    end;
  for i := 0 to N-1 do begin hostX_orig[i].x := i+1; hostX_orig[i].y := -(i+1); end;

  for i := 0 to N-1 do begin expectedX[i].x := 0; expectedX[i].y := 0; end;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      term := hostA[j*N+i];
      ComplexMul(term, hostX_orig[j]);
      ComplexAdd(expectedX[i], term);
    end;

  hipMalloc(devA, N*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX_orig[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ctrmv OK', Integer(rocblas_status_success),
    Integer(rocblas_ctrmv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devA, N, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-5, Format('CTRMV X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-5, Format('CTRMV X[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_ZTRMV;
const
  N = 5;
var
  devA, devX: hipDeviceptr_t;
  hostA: array[0..N*N-1] of rocblas_double_complex;
  hostX_orig, expectedX, hostX_result: array[0..N-1] of rocblas_double_complex;
  i, j: Integer;
  term: rocblas_double_complex;
begin
  FillChar(hostA, SizeOf(hostA), 0);
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostA[j*N + i].x := i + j + 2;
      hostA[j*N + i].y := (i-j)/2.0;
    end;
  for i := 0 to N-1 do begin hostX_orig[i].x := i+1; hostX_orig[i].y := -(i+1); end;

  for i := 0 to N-1 do begin expectedX[i].x := 0; expectedX[i].y := 0; end;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      term := hostA[j*N+i];
      ComplexMul(term, hostX_orig[j]);
      ComplexAdd(expectedX[i], term);
    end;

  hipMalloc(devA, N*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX_orig[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ztrmv OK', Integer(rocblas_status_success),
    Integer(rocblas_ztrmv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devA, N, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTRMV X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTRMV X[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_STBMV;
const
  N = 6; K = 2; // N=Order, K=super-diagonals
  LDA = 3;     // LDA >= K+1
var
  devA, devX: hipDeviceptr_t;
  hostA_full: array of Single;
  hostA_band: array[0..LDA*N-1] of Single;
  hostX_orig, expectedX, hostX_result: array[0..N-1] of Single;
  i, j: Integer;
begin
  SetLength(hostA_full, N*N);

  // 1. Create a full upper triangular banded matrix for reference
  FillChar(hostA_full[0], N*N*SizeOf(Single), 0);
  for j := 0 to N-1 do
    for i := Max(0, j-K) to j do
      hostA_full[j*N+i] := i+j+2.5;

  // 2. Convert to band storage format
  FillChar(hostA_band, SizeOf(hostA_band), 0);
  for j := 0 to N-1 do
    for i := max(0, j-K) to j do
      hostA_band[(K+i-j) + j*LDA] := hostA_full[j*N+i];

  // 3. Initialize original vector and calculate CPU reference result
  for i := 0 to N-1 do hostX_orig[i] := i + 1.25;

  FillChar(expectedX, SizeOf(expectedX), 0);
  for j := 0 to N-1 do
    for i := 0 to N-1 do
      expectedX[i] := expectedX[i] + hostA_full[j*N+i] * hostX_orig[j];

  // 4. ROCm execution
  hipMalloc(devA, LDA*N*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));
  hipMemcpy(devA, @hostA_band[0], LDA*N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX_orig[0], N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_stbmv OK', Integer(rocblas_status_success),
    Integer(rocblas_stbmv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, K, devA, LDA, devX, 1)));

  // 5. Verify result
  hipMemcpy(@hostX_result[0], devX, N*SizeOf(Single), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-5, Format('STBMV X[%d]', [i]));

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_DTBMV;
const
  N = 6; K = 2;
  LDA = 3;
var
  devA, devX: hipDeviceptr_t;
  hostA_full: array of Double;
  hostA_band: array[0..LDA*N-1] of Double;
  hostX_orig, expectedX, hostX_result: array[0..N-1] of Double;
  i, j: Integer;
begin
  SetLength(hostA_full, N*N);

  FillChar(hostA_full[0], N*N*SizeOf(Double), 0);
  for j := 0 to N-1 do
    for i := Max(0, j-K) to j do
      hostA_full[j*N+i] := i+j+2.5;

  FillChar(hostA_band, SizeOf(hostA_band), 0);
  for j := 0 to N-1 do
    for i := max(0, j-K) to j do
      hostA_band[(K+i-j) + j*LDA] := hostA_full[j*N+i];

  for i := 0 to N-1 do hostX_orig[i] := i + 1.25;

  FillChar(expectedX, SizeOf(expectedX), 0);
  for j := 0 to N-1 do
    for i := 0 to N-1 do
      expectedX[i] := expectedX[i] + hostA_full[j*N+i] * hostX_orig[j];

  hipMalloc(devA, LDA*N*SizeOf(Double));
  hipMalloc(devX, N*SizeOf(Double));
  hipMemcpy(devA, @hostA_band[0], LDA*N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX_orig[0], N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dtbmv OK', Integer(rocblas_status_success),
    Integer(rocblas_dtbmv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, K, devA, LDA, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(Double), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTBMV X[%d]', [i]));

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_CTBMV;
const
  N = 6; K = 2;
  LDA = 3;
var
  devA, devX: hipDeviceptr_t;
  hostA_full: array of rocblas_float_complex;
  hostA_band: array[0..LDA*N-1] of rocblas_float_complex;
  hostX_orig, expectedX, hostX_result: array[0..N-1] of rocblas_float_complex;
  i, j: Integer;
  term: rocblas_float_complex;
begin
  SetLength(hostA_full, N*N);

  FillChar(hostA_full[0], N*N*SizeOf(rocblas_float_complex), 0);
  for j := 0 to N-1 do
    for i := Max(0, j-K) to j do
    begin
      hostA_full[j*N+i].x := i+j+2.5;
      hostA_full[j*N+i].y := (i-j)/2.0;
    end;

  FillChar(hostA_band, SizeOf(hostA_band), 0);
  for j := 0 to N-1 do
    for i := max(0, j-K) to j do
      hostA_band[(K+i-j) + j*LDA] := hostA_full[j*N+i];

  for i := 0 to N-1 do begin hostX_orig[i].x := i+1.25; hostX_orig[i].y := -(i+1.25); end;

  for i := 0 to N-1 do begin expectedX[i].x := 0; expectedX[i].y := 0; end;
  for j := 0 to N-1 do
    for i := 0 to N-1 do
    begin
      term := hostA_full[j*N+i];
      ComplexMul(term, hostX_orig[j]);
      ComplexAdd(expectedX[i], term);
    end;

  hipMalloc(devA, LDA*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devA, @hostA_band[0], LDA*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX_orig[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ctbmv OK', Integer(rocblas_status_success),
    Integer(rocblas_ctbmv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, K, devA, LDA, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-5, Format('CTBMV X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-5, Format('CTBMV X[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_ZTBMV;
const
  N = 6; K = 2;
  LDA = 3;
var
  devA, devX: hipDeviceptr_t;
  hostA_full: array of rocblas_double_complex;
  hostA_band: array[0..LDA*N-1] of rocblas_double_complex;
  hostX_orig, expectedX, hostX_result: array[0..N-1] of rocblas_double_complex;
  i, j: Integer;
  term: rocblas_double_complex;
begin
  SetLength(hostA_full, N*N);

  FillChar(hostA_full[0], N*N*SizeOf(rocblas_double_complex), 0);
  for j := 0 to N-1 do
    for i := Max(0, j-K) to j do
    begin
      hostA_full[j*N+i].x := i+j+2.5;
      hostA_full[j*N+i].y := (i-j)/2.0;
    end;

  FillChar(hostA_band, SizeOf(hostA_band), 0);
  for j := 0 to N-1 do
    for i := max(0, j-K) to j do
      hostA_band[(K+i-j) + j*LDA] := hostA_full[j*N+i];

  for i := 0 to N-1 do begin hostX_orig[i].x := i+1.25; hostX_orig[i].y := -(i+1.25); end;

  for i := 0 to N-1 do begin expectedX[i].x := 0; expectedX[i].y := 0; end;
  for j := 0 to N-1 do
    for i := 0 to N-1 do
    begin
      term := hostA_full[j*N+i];
      ComplexMul(term, hostX_orig[j]);
      ComplexAdd(expectedX[i], term);
    end;

  hipMalloc(devA, LDA*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devA, @hostA_band[0], LDA*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX_orig[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ztbmv OK', Integer(rocblas_status_success),
    Integer(rocblas_ztbmv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, K, devA, LDA, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTBMV X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTBMV X[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_STPMV;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
var
  devAP, devX: hipDeviceptr_t;
  hostA_full: array of Single;
  hostAP: array[0..PACKED_SIZE-1] of Single;
  hostX_orig, expectedX, hostX_result: array[0..N-1] of Single;
  i, j, k: Integer;
begin
  SetLength(hostA_full, N*N);

  // 1. Create a full upper triangular matrix for reference
  FillChar(hostA_full[0], N*N*SizeOf(Single), 0);
  for j := 0 to N-1 do
    for i := 0 to j do
      hostA_full[j*N+i] := i + j + 2.5;

  // 2. Convert to packed storage format (upper, column-major)
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := hostA_full[j*N+i];
      Inc(k);
    end;

  // 3. Initialize original vector and calculate CPU reference result
  for i := 0 to N-1 do hostX_orig[i] := i + 1.25;

  FillChar(expectedX, SizeOf(expectedX), 0);
  for j := 0 to N-1 do
    for i := 0 to N-1 do
      expectedX[i] := expectedX[i] + hostA_full[j*N+i] * hostX_orig[j];

  // 4. ROCm execution
  hipMalloc(devAP, PACKED_SIZE*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX_orig[0], N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_stpmv OK', Integer(rocblas_status_success),
    Integer(rocblas_stpmv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devAP, devX, 1)));

  // 5. Verify result
  hipMemcpy(@hostX_result[0], devX, N*SizeOf(Single), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-5, Format('STPMV X[%d]', [i]));

  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_DTPMV;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
var
  devAP, devX: hipDeviceptr_t;
  hostA_full: array of Double;
  hostAP: array[0..PACKED_SIZE-1] of Double;
  hostX_orig, expectedX, hostX_result: array[0..N-1] of Double;
  i, j, k: Integer;
begin
  SetLength(hostA_full, N*N);

  FillChar(hostA_full[0], N*N*SizeOf(Double), 0);
  for j := 0 to N-1 do
    for i := 0 to j do
      hostA_full[j*N+i] := i + j + 2.5;

  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := hostA_full[j*N+i];
      Inc(k);
    end;

  for i := 0 to N-1 do hostX_orig[i] := i + 1.25;

  FillChar(expectedX, SizeOf(expectedX), 0);
  for j := 0 to N-1 do
    for i := 0 to N-1 do
      expectedX[i] := expectedX[i] + hostA_full[j*N+i] * hostX_orig[j];

  hipMalloc(devAP, PACKED_SIZE*SizeOf(Double));
  hipMalloc(devX, N*SizeOf(Double));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX_orig[0], N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dtpmv OK', Integer(rocblas_status_success),
    Integer(rocblas_dtpmv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devAP, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(Double), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(expectedX[i], hostX_result[i], 1e-12, Format('DTPMV X[%d]', [i]));

  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_CTPMV;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
var
  devAP, devX: hipDeviceptr_t;
  hostA_full: array of rocblas_float_complex;
  hostAP: array[0..PACKED_SIZE-1] of rocblas_float_complex;
  hostX_orig, expectedX, hostX_result: array[0..N-1] of rocblas_float_complex;
  i, j, k: Integer;
  term: rocblas_float_complex;
begin
  SetLength(hostA_full, N*N);

  FillChar(hostA_full[0], N*N*SizeOf(rocblas_float_complex), 0);
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostA_full[j*N+i].x := i+j+2.5;
      hostA_full[j*N+i].y := (i-j)/2.0;
    end;

  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := hostA_full[j*N+i];
      Inc(k);
    end;

  for i := 0 to N-1 do begin hostX_orig[i].x := i+1.25; hostX_orig[i].y := -(i+1.25); end;

  for i := 0 to N-1 do begin expectedX[i].x := 0; expectedX[i].y := 0; end;
  for j := 0 to N-1 do
    for i := 0 to N-1 do
    begin
      term := hostA_full[j*N+i];
      ComplexMul(term, hostX_orig[j]);
      ComplexAdd(expectedX[i], term);
    end;

  hipMalloc(devAP, PACKED_SIZE*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX_orig[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ctpmv OK', Integer(rocblas_status_success),
    Integer(rocblas_ctpmv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devAP, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-5, Format('CTPMV X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-5, Format('CTPMV X[%d].y', [i]));
  end;

  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_ZTPMV;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
var
  devAP, devX: hipDeviceptr_t;
  hostA_full: array of rocblas_double_complex;
  hostAP: array[0..PACKED_SIZE-1] of rocblas_double_complex;
  hostX_orig, expectedX, hostX_result: array[0..N-1] of rocblas_double_complex;
  i, j, k: Integer;
  term: rocblas_double_complex;
begin
  SetLength(hostA_full, N*N);

  FillChar(hostA_full[0], N*N*SizeOf(rocblas_double_complex), 0);
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostA_full[j*N+i].x := i+j+2.5;
      hostA_full[j*N+i].y := (i-j)/2.0;
    end;

  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := hostA_full[j*N+i];
      Inc(k);
    end;

  for i := 0 to N-1 do begin hostX_orig[i].x := i+1.25; hostX_orig[i].y := -(i+1.25); end;

  for i := 0 to N-1 do begin expectedX[i].x := 0; expectedX[i].y := 0; end;
  for j := 0 to N-1 do
    for i := 0 to N-1 do
    begin
      term := hostA_full[j*N+i];
      ComplexMul(term, hostX_orig[j]);
      ComplexAdd(expectedX[i], term);
    end;

  hipMalloc(devAP, PACKED_SIZE*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX_orig[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ztpmv OK', Integer(rocblas_status_success),
    Integer(rocblas_ztpmv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devAP, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(expectedX[i].x, hostX_result[i].x, 1e-12, Format('ZTPMV X[%d].x', [i]));
    CheckEqualsRelative(expectedX[i].y, hostX_result[i].y, 1e-12, Format('ZTPMV X[%d].y', [i]));
  end;

  hipFree(devAP); hipFree(devX);
end;
procedure TTestRocBLAS_Level2.Test_STRSV;
const
  N = 5;
var
  devA, devX: hipDeviceptr_t;
  hostA: array[0..N*N-1] of Single;
  hostX_sol, hostB, hostX_result: array[0..N-1] of Single;
  i, j: Integer;
begin
  // Initialize upper triangular matrix A and known solution x_sol
  for j := 0 to N-1 do
    for i := 0 to j do
      hostA[j*N + i] := i + j + 1;
  for i := 0 to N-1 do hostA[i*N+i] := hostA[i*N+i] + 10; // Make non-singular
  for i := 0 to N-1 do hostX_sol[i] := i + 1.0;

  // Calculate b = A * x_sol on CPU
  FillChar(hostB, SizeOf(hostB), 0);
  for j := 0 to N-1 do
    for i := 0 to j do // Upper triangular part
      hostB[i] := hostB[i] + hostA[j*N + i] * hostX_sol[j];

  hipMalloc(devA, N*N*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));

  hipMemcpy(devA, @hostA[0], N*N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostB[0], N*SizeOf(Single), hipMemcpyHostToDevice); // devX starts with b

  AssertEquals('rocblas_strsv OK', Integer(rocblas_status_success),
    Integer(rocblas_strsv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devA, N, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(Single), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(hostX_sol[i], hostX_result[i], 1e-5, Format('STRSV X[%d]',[i]));

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_DTRSV;
const
  N = 5;
var
  devA, devX: hipDeviceptr_t;
  hostA: array[0..N*N-1] of Double;
  hostX_sol, hostB, hostX_result: array[0..N-1] of Double;
  i, j: Integer;
begin
  // Initialize upper triangular matrix A and known solution x_sol
  for j := 0 to N-1 do
    for i := 0 to j do
      hostA[j*N + i] := i + j + 1;
  for i := 0 to N-1 do hostA[i*N+i] := hostA[i*N+i] + 10; // Make non-singular
  for i := 0 to N-1 do hostX_sol[i] := i + 1.0;

  // Calculate b = A * x_sol on CPU
  FillChar(hostB, SizeOf(hostB), 0);
  for j := 0 to N-1 do
    for i := 0 to j do // Upper triangular part
      hostB[i] := hostB[i] + hostA[j*N + i] * hostX_sol[j];

  hipMalloc(devA, N*N*SizeOf(Double));
  hipMalloc(devX, N*SizeOf(Double));

  hipMemcpy(devA, @hostA[0], N*N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostB[0], N*SizeOf(Double), hipMemcpyHostToDevice); // devX starts with b

  AssertEquals('rocblas_dtrsv OK', Integer(rocblas_status_success),
    Integer(rocblas_dtrsv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devA, N, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(Double), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(hostX_sol[i], hostX_result[i], 1e-12, Format('DTRSV X[%d]',[i]));

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_CTRSV;
const
  N = 5;
var
  devA, devX: hipDeviceptr_t;
  hostA: array[0..N*N-1] of rocblas_float_complex;
  hostX_sol, hostB, hostX_result: array[0..N-1] of rocblas_float_complex;
  i, j: Integer;
  term: rocblas_float_complex;
begin
  // Initialize upper triangular matrix A and known solution x_sol
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostA[j*N + i].x := i + j + 1;
      hostA[j*N + i].y := (i - j)/2.0;
    end;
  for i := 0 to N-1 do hostA[i*N+i].x := hostA[i*N+i].x + 10; // Make non-singular
  for i := 0 to N-1 do begin hostX_sol[i].x := i+1; hostX_sol[i].y := -(i+1); end;

  // Calculate b = A * x_sol on CPU
  for i := 0 to N-1 do begin hostB[i].x := 0; hostB[i].y := 0; end;
  for j := 0 to N-1 do
    for i := 0 to j do // Upper triangular part
    begin
      term := hostA[j*N+i];
      ComplexMul(term, hostX_sol[j]);
      ComplexAdd(hostB[i], term);
    end;

  hipMalloc(devA, N*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostB[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ctrsv OK', Integer(rocblas_status_success),
    Integer(rocblas_ctrsv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devA, N, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(hostX_sol[i].x, hostX_result[i].x, 1e-5, Format('CTRSV X[%d].x',[i]));
    CheckEqualsRelative(hostX_sol[i].y, hostX_result[i].y, 1e-5, Format('CTRSV X[%d].y',[i]));
  end;

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_ZTRSV;
const
  N = 5;
var
  devA, devX: hipDeviceptr_t;
  hostA: array[0..N*N-1] of rocblas_double_complex;
  hostX_sol, hostB, hostX_result: array[0..N-1] of rocblas_double_complex;
  i, j: Integer;
  term: rocblas_double_complex;
begin
  // Initialize upper triangular matrix A and known solution x_sol
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostA[j*N + i].x := i + j + 1;
      hostA[j*N + i].y := (i - j)/2.0;
    end;
  for i := 0 to N-1 do hostA[i*N+i].x := hostA[i*N+i].x + 10; // Make non-singular
  for i := 0 to N-1 do begin hostX_sol[i].x := i+1; hostX_sol[i].y := -(i+1); end;

  // Calculate b = A * x_sol on CPU
  for i := 0 to N-1 do begin hostB[i].x := 0; hostB[i].y := 0; end;
  for j := 0 to N-1 do
    for i := 0 to j do // Upper triangular part
    begin
      term := hostA[j*N+i];
      ComplexMul(term, hostX_sol[j]);
      ComplexAdd(hostB[i], term);
    end;

  hipMalloc(devA, N*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostB[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ztrsv OK', Integer(rocblas_status_success),
    Integer(rocblas_ztrsv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devA, N, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(hostX_sol[i].x, hostX_result[i].x, 1e-12, Format('ZTRSV X[%d].x',[i]));
    CheckEqualsRelative(hostX_sol[i].y, hostX_result[i].y, 1e-12, Format('ZTRSV X[%d].y',[i]));
  end;

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_STPSV;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
var
  devAP, devX: hipDeviceptr_t;
  hostA_full: array of Single;
  hostAP: array[0..PACKED_SIZE-1] of Single;
  hostX_sol, hostB, hostX_result: array[0..N-1] of Single;
  i, j, k: Integer;
begin
  SetLength(hostA_full, N*N);

  // 1. Create a full upper triangular matrix
  for j := 0 to N-1 do
    for i := 0 to j do
      hostA_full[j*N+i] := i+j+2;
  for i := 0 to N-1 do hostA_full[i*N+i] := hostA_full[i*N+i] + 10;

  // 2. Convert to packed storage (upper)
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := hostA_full[j*N+i];
      Inc(k);
    end;

  // 3. Create problem b = A*x
  for i := 0 to N-1 do hostX_sol[i] := i + 1.0;
  FillChar(hostB, SizeOf(hostB), 0);
  for j := 0 to N-1 do
    for i := 0 to j do
      hostB[i] := hostB[i] + hostA_full[j*N+i] * hostX_sol[j];

  // 4. ROCm execution
  hipMalloc(devAP, PACKED_SIZE*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostB[0], N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_stpsv OK', Integer(rocblas_status_success),
    Integer(rocblas_stpsv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devAP, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(Single), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(hostX_sol[i], hostX_result[i], 1e-5, Format('STPSV X[%d]', [i]));

  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_DTPSV;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
var
  devAP, devX: hipDeviceptr_t;
  hostA_full: array of Double;
  hostAP: array[0..PACKED_SIZE-1] of Double;
  hostX_sol, hostB, hostX_result: array[0..N-1] of Double;
  i, j, k: Integer;
begin
  SetLength(hostA_full, N*N);

  for j := 0 to N-1 do
    for i := 0 to j do
      hostA_full[j*N+i] := i+j+2;
  for i := 0 to N-1 do hostA_full[i*N+i] := hostA_full[i*N+i] + 10;

  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := hostA_full[j*N+i];
      Inc(k);
    end;

  for i := 0 to N-1 do hostX_sol[i] := i + 1.0;
  FillChar(hostB, SizeOf(hostB), 0);
  for j := 0 to N-1 do
    for i := 0 to j do
      hostB[i] := hostB[i] + hostA_full[j*N+i] * hostX_sol[j];

  hipMalloc(devAP, PACKED_SIZE*SizeOf(Double));
  hipMalloc(devX, N*SizeOf(Double));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostB[0], N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dtpsv OK', Integer(rocblas_status_success),
    Integer(rocblas_dtpsv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devAP, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(Double), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(hostX_sol[i], hostX_result[i], 1e-12, Format('DTPSV X[%d]', [i]));

  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_CTPSV;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
var
  devAP, devX: hipDeviceptr_t;
  hostA_full: array of rocblas_float_complex;
  hostAP: array[0..PACKED_SIZE-1] of rocblas_float_complex;
  hostX_sol, hostB, hostX_result: array[0..N-1] of rocblas_float_complex;
  i, j, k: Integer;
  term: rocblas_float_complex;
begin
  SetLength(hostA_full, N*N);

  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostA_full[j*N+i].x := i+j+2;
      hostA_full[j*N+i].y := (i-j)/2.0;
    end;
  for i := 0 to N-1 do hostA_full[i*N+i].x := hostA_full[i*N+i].x + 10;

  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := hostA_full[j*N+i];
      Inc(k);
    end;

  for i := 0 to N-1 do begin hostX_sol[i].x := i+1; hostX_sol[i].y := -(i+1); end;
  for i := 0 to N-1 do begin hostB[i].x := 0; hostB[i].y := 0; end;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      term := hostA_full[j*N+i];
      ComplexMul(term, hostX_sol[j]);
      ComplexAdd(hostB[i], term);
    end;

  hipMalloc(devAP, PACKED_SIZE*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostB[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ctpsv OK', Integer(rocblas_status_success),
    Integer(rocblas_ctpsv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devAP, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(hostX_sol[i].x, hostX_result[i].x, 1e-5, Format('CTPSV X[%d].x', [i]));
    CheckEqualsRelative(hostX_sol[i].y, hostX_result[i].y, 1e-5, Format('CTPSV X[%d].y', [i]));
  end;

  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_ZTPSV;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
var
  devAP, devX: hipDeviceptr_t;
  hostA_full: array of rocblas_double_complex;
  hostAP: array[0..PACKED_SIZE-1] of rocblas_double_complex;
  hostX_sol, hostB, hostX_result: array[0..N-1] of rocblas_double_complex;
  i, j, k: Integer;
  term: rocblas_double_complex;
begin
  SetLength(hostA_full, N*N);

  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostA_full[j*N+i].x := i+j+2;
      hostA_full[j*N+i].y := (i-j)/2.0;
    end;
  for i := 0 to N-1 do hostA_full[i*N+i].x := hostA_full[i*N+i].x + 10;

  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := hostA_full[j*N+i];
      Inc(k);
    end;

  for i := 0 to N-1 do begin hostX_sol[i].x := i+1; hostX_sol[i].y := -(i+1); end;
  for i := 0 to N-1 do begin hostB[i].x := 0; hostB[i].y := 0; end;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      term := hostA_full[j*N+i];
      ComplexMul(term, hostX_sol[j]);
      ComplexAdd(hostB[i], term);
    end;

  hipMalloc(devAP, PACKED_SIZE*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostB[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ztpsv OK', Integer(rocblas_status_success),
    Integer(rocblas_ztpsv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, devAP, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(hostX_sol[i].x, hostX_result[i].x, 1e-12, Format('ZTPSV X[%d].x', [i]));
    CheckEqualsRelative(hostX_sol[i].y, hostX_result[i].y, 1e-12, Format('ZTPSV X[%d].y', [i]));
  end;

  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_STBSV;
const
  N = 6; K = 2; // N=Order, K=super-diagonals
  LDA = 3;     // LDA >= K+1
var
  devA, devX: hipDeviceptr_t;
  hostA_full: array of Single;
  hostA_band: array[0..LDA*N-1] of Single;
  hostX_sol, hostB, hostX_result: array[0..N-1] of Single;
  i, j: Integer;
begin
  SetLength(hostA_full, N*N);

  // 1. Create a full upper triangular banded matrix for reference
  FillChar(hostA_full[0], N*N*SizeOf(Single), 0);
  for j := 0 to N-1 do
    for i := Max(0, j-K) to j do
      hostA_full[j*N+i] := i+j+2;
  // Make diagonal strong to ensure it's non-singular
  for i := 0 to N-1 do hostA_full[i*N+i] := hostA_full[i*N+i] + N;

  // 2. Convert the full matrix to the required band storage format
  FillChar(hostA_band, SizeOf(hostA_band), 0);
  for j := 0 to N-1 do
    for i := max(0, j-K) to j do
      hostA_band[(K+i-j) + j*LDA] := hostA_full[j*N+i];

  // 3. Create known solution `x` and calculate right-hand side `b = A*x`
  for i := 0 to N-1 do hostX_sol[i] := i + 1.0;
  FillChar(hostB, SizeOf(hostB), 0);
  for j := 0 to N-1 do
    for i := 0 to N-1 do
      hostB[i] := hostB[i] + hostA_full[j*N+i] * hostX_sol[j];

  // 4. ROCm execution
  hipMalloc(devA, LDA*N*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));
  hipMemcpy(devA, @hostA_band[0], LDA*N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostB[0], N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_stbsv OK', Integer(rocblas_status_success),
    Integer(rocblas_stbsv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, K, devA, LDA, devX, 1)));

  // 5. Verify result
  hipMemcpy(@hostX_result[0], devX, N*SizeOf(Single), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(hostX_sol[i], hostX_result[i], 1e-5, Format('STBSV X[%d]', [i]));

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_DTBSV;
const
  N = 6; K = 2;
  LDA = 3;
var
  devA, devX: hipDeviceptr_t;
  hostA_full: array of Double;
  hostA_band: array[0..LDA*N-1] of Double;
  hostX_sol, hostB, hostX_result: array[0..N-1] of Double;
  i, j: Integer;
begin
  SetLength(hostA_full, N*N);

  FillChar(hostA_full[0], N*N*SizeOf(Double), 0);
  for j := 0 to N-1 do
    for i := Max(0, j-K) to j do
      hostA_full[j*N+i] := i+j+2;
  for i := 0 to N-1 do hostA_full[i*N+i] := hostA_full[i*N+i] + N;

  FillChar(hostA_band, SizeOf(hostA_band), 0);
  for j := 0 to N-1 do
    for i := max(0, j-K) to j do
      hostA_band[(K+i-j) + j*LDA] := hostA_full[j*N+i];

  for i := 0 to N-1 do hostX_sol[i] := i + 1.0;
  FillChar(hostB, SizeOf(hostB), 0);
  for j := 0 to N-1 do
    for i := 0 to N-1 do
      hostB[i] := hostB[i] + hostA_full[j*N+i] * hostX_sol[j];

  hipMalloc(devA, LDA*N*SizeOf(Double));
  hipMalloc(devX, N*SizeOf(Double));
  hipMemcpy(devA, @hostA_band[0], LDA*N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostB[0], N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dtbsv OK', Integer(rocblas_status_success),
    Integer(rocblas_dtbsv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, K, devA, LDA, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(Double), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
    CheckEqualsRelative(hostX_sol[i], hostX_result[i], 1e-12, Format('DTBSV X[%d]', [i]));

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_CTBSV;
const
  N = 6; K = 2;
  LDA = 3;
var
  devA, devX: hipDeviceptr_t;
  hostA_full: array of rocblas_float_complex;
  hostA_band: array[0..LDA*N-1] of rocblas_float_complex;
  hostX_sol, hostB, hostX_result: array[0..N-1] of rocblas_float_complex;
  i, j: Integer;
  term: rocblas_float_complex;
begin
  SetLength(hostA_full, N*N);

  FillChar(hostA_full[0], N*N*SizeOf(rocblas_float_complex), 0);
  for j := 0 to N-1 do
    for i := Max(0, j-K) to j do
    begin
      hostA_full[j*N+i].x := i+j+2;
      hostA_full[j*N+i].y := (i-j)/2.0;
    end;
  for i := 0 to N-1 do hostA_full[i*N+i].x := hostA_full[i*N+i].x + N;

  FillChar(hostA_band, SizeOf(hostA_band), 0);
  for j := 0 to N-1 do
    for i := max(0, j-K) to j do
      hostA_band[(K+i-j) + j*LDA] := hostA_full[j*N+i];

  for i := 0 to N-1 do begin hostX_sol[i].x := i+1; hostX_sol[i].y := -(i+1); end;
  for i := 0 to N-1 do begin hostB[i].x := 0; hostB[i].y := 0; end;
  for j := 0 to N-1 do
    for i := 0 to N-1 do
    begin
      term := hostA_full[j*N+i];
      ComplexMul(term, hostX_sol[j]);
      ComplexAdd(hostB[i], term);
    end;

  hipMalloc(devA, LDA*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devA, @hostA_band[0], LDA*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostB[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ctbsv OK', Integer(rocblas_status_success),
    Integer(rocblas_ctbsv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, K, devA, LDA, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(hostX_sol[i].x, hostX_result[i].x, 1e-5, Format('CTBSV X[%d].x', [i]));
    CheckEqualsRelative(hostX_sol[i].y, hostX_result[i].y, 1e-5, Format('CTBSV X[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_ZTBSV;
const
  N = 6; K = 2;
  LDA = 3;
var
  devA, devX: hipDeviceptr_t;
  hostA_full: array of rocblas_double_complex;
  hostA_band: array[0..LDA*N-1] of rocblas_double_complex;
  hostX_sol, hostB, hostX_result: array[0..N-1] of rocblas_double_complex;
  i, j: Integer;
  term: rocblas_double_complex;
begin
  SetLength(hostA_full, N*N);

  FillChar(hostA_full[0], N*N*SizeOf(rocblas_double_complex), 0);
  for j := 0 to N-1 do
    for i := Max(0, j-K) to j do
    begin
      hostA_full[j*N+i].x := i+j+2;
      hostA_full[j*N+i].y := (i-j)/2.0;
    end;
  for i := 0 to N-1 do hostA_full[i*N+i].x := hostA_full[i*N+i].x + N;

  FillChar(hostA_band, SizeOf(hostA_band), 0);
  for j := 0 to N-1 do
    for i := max(0, j-K) to j do
      hostA_band[(K+i-j) + j*LDA] := hostA_full[j*N+i];

  for i := 0 to N-1 do begin hostX_sol[i].x := i+1; hostX_sol[i].y := -(i+1); end;
  for i := 0 to N-1 do begin hostB[i].x := 0; hostB[i].y := 0; end;
  for j := 0 to N-1 do
    for i := 0 to N-1 do
    begin
      term := hostA_full[j*N+i];
      ComplexMul(term, hostX_sol[j]);
      ComplexAdd(hostB[i], term);
    end;

  hipMalloc(devA, LDA*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devA, @hostA_band[0], LDA*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostB[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ztbsv OK', Integer(rocblas_status_success),
    Integer(rocblas_ztbsv(handle, rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit, N, K, devA, LDA, devX, 1)));

  hipMemcpy(@hostX_result[0], devX, N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to N-1 do
  begin
    CheckEqualsRelative(hostX_sol[i].x, hostX_result[i].x, 1e-12, Format('ZTBSV X[%d].x', [i]));
    CheckEqualsRelative(hostX_sol[i].y, hostX_result[i].y, 1e-12, Format('ZTBSV X[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX);
end;
procedure TTestRocBLAS_Level2.Test_SGER;
const
  M = 4; N = 5;
  alpha: Single = 2.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array of Single;
  hostX: array[0..M-1] of Single;
  hostY: array[0..N-1] of Single;
  i, j: Integer;
begin
  SetLength(hostA, M*N);
  SetLength(hostA_orig, M*N);
  SetLength(expectedA, M*N);

  for j := 0 to N-1 do for i := 0 to M-1 do
    hostA[j*M + i] := (j+1)*10 + (i+1);
  hostA_orig := hostA; // Keep a copy
  for i := 0 to M-1 do hostX[i] := i + 1.0;
  for i := 0 to N-1 do hostY[i] := i + 2.0;

  // CPU reference: A := alpha*x*y**T + A
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to M-1 do
      expectedA[j*M + i] := expectedA[j*M + i] + alpha * hostX[i] * hostY[j];

  hipMalloc(devA, M*N*SizeOf(Single));
  hipMalloc(devX, M*SizeOf(Single));
  hipMalloc(devY, N*SizeOf(Single));
  hipMemcpy(devA, @hostA[0], M*N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], M*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_sger OK', Integer(rocblas_status_success),
    Integer(rocblas_sger(handle, M, N, @alpha, devX, 1, devY, 1, devA, M)));

  hipMemcpy(@hostA[0], devA, M*N*SizeOf(Single), hipMemcpyDeviceToHost);

  for i := 0 to M*N-1 do
    CheckEqualsRelative(expectedA[i], hostA[i], 1e-5, Format('SGER A[%d]',[i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_DGER;
const
  M = 4; N = 5;
  alpha: Double = 2.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array of Double;
  hostX: array[0..M-1] of Double;
  hostY: array[0..N-1] of Double;
  i, j: Integer;
begin
  SetLength(hostA, M*N);
  SetLength(hostA_orig, M*N);
  SetLength(expectedA, M*N);

  for j := 0 to N-1 do for i := 0 to M-1 do
    hostA[j*M + i] := (j+1)*10 + (i+1);
  hostA_orig := hostA; // Keep a copy
  for i := 0 to M-1 do hostX[i] := i + 1.0;
  for i := 0 to N-1 do hostY[i] := i + 2.0;

  // CPU reference: A := alpha*x*y**T + A
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to M-1 do
      expectedA[j*M + i] := expectedA[j*M + i] + alpha * hostX[i] * hostY[j];

  hipMalloc(devA, M*N*SizeOf(Double));
  hipMalloc(devX, M*SizeOf(Double));
  hipMalloc(devY, N*SizeOf(Double));
  hipMemcpy(devA, @hostA[0], M*N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], M*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dger OK', Integer(rocblas_status_success),
    Integer(rocblas_dger(handle, M, N, @alpha, devX, 1, devY, 1, devA, M)));

  hipMemcpy(@hostA[0], devA, M*N*SizeOf(Double), hipMemcpyDeviceToHost);

  for i := 0 to M*N-1 do
    CheckEqualsRelative(expectedA[i], hostA[i], 1e-12, Format('DGER A[%d]',[i]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_CGERU; // Unconjugated
const
  M = 4; N = 5;
  alpha: rocblas_float_complex = (x: 1.5; y: -0.5);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array of rocblas_float_complex;
  hostX: array[0..M-1] of rocblas_float_complex;
  hostY: array[0..N-1] of rocblas_float_complex;
  i, j: Integer;
  term: rocblas_float_complex;
begin
  SetLength(hostA, M*N);
  SetLength(hostA_orig, M*N);
  SetLength(expectedA, M*N);

  for j := 0 to N-1 do for i := 0 to M-1 do
  begin
    hostA[j*M + i].x := (j+1)*10 + (i+1);
    hostA[j*M + i].y := -((j+1)*5 + (i+1));
  end;
  hostA_orig := hostA;
  for i := 0 to M-1 do begin hostX[i].x := i + 1; hostX[i].y := i - 1; end;
  for i := 0 to N-1 do begin hostY[i].x := i + 2; hostY[i].y := i - 2; end;

  // CPU reference: A := alpha*x*y**T + A
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to M-1 do
    begin
      term := hostX[i];
      ComplexMul(term, hostY[j]); // x[i] * y[j]
      ComplexMul(term, alpha);     // alpha * x[i] * y[j]
      ComplexAdd(expectedA[j*M + i], term);
    end;

  hipMalloc(devA, M*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, M*SizeOf(rocblas_float_complex));
  hipMalloc(devY, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devA, @hostA[0], M*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], M*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_cgeru OK', Integer(rocblas_status_success),
    Integer(rocblas_cgeru(handle, M, N, @alpha, devX, 1, devY, 1, devA, M)));

  hipMemcpy(@hostA[0], devA, M*N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to M*N-1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA[i].x, 1e-5, Format('CGERU A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA[i].y, 1e-5, Format('CGERU A[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_ZGERU; // Unconjugated
const
  M = 4; N = 5;
  alpha: rocblas_double_complex = (x: 1.5; y: -0.5);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array of rocblas_double_complex;
  hostX: array[0..M-1] of rocblas_double_complex;
  hostY: array[0..N-1] of rocblas_double_complex;
  i, j: Integer;
  term: rocblas_double_complex;
begin
  SetLength(hostA, M*N);
  SetLength(hostA_orig, M*N);
  SetLength(expectedA, M*N);

  for j := 0 to N-1 do for i := 0 to M-1 do
  begin
    hostA[j*M + i].x := (j+1)*10 + (i+1);
    hostA[j*M + i].y := -((j+1)*5 + (i+1));
  end;
  hostA_orig := hostA;
  for i := 0 to M-1 do begin hostX[i].x := i + 1; hostX[i].y := i - 1; end;
  for i := 0 to N-1 do begin hostY[i].x := i + 2; hostY[i].y := i - 2; end;

  // CPU reference: A := alpha*x*y**T + A
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to M-1 do
    begin
      term := hostX[i];
      ComplexMul(term, hostY[j]); // x[i] * y[j]
      ComplexMul(term, alpha);     // alpha * x[i] * y[j]
      ComplexAdd(expectedA[j*M + i], term);
    end;

  hipMalloc(devA, M*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, M*SizeOf(rocblas_double_complex));
  hipMalloc(devY, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devA, @hostA[0], M*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], M*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zgeru OK', Integer(rocblas_status_success),
    Integer(rocblas_zgeru(handle, M, N, @alpha, devX, 1, devY, 1, devA, M)));

  hipMemcpy(@hostA[0], devA, M*N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to M*N-1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA[i].x, 1e-12, Format('ZGERU A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA[i].y, 1e-12, Format('ZGERU A[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_CGERC; // Conjugated
const
  M = 4; N = 5;
  alpha: rocblas_float_complex = (x: 1.5; y: -0.5);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array of rocblas_float_complex;
  hostX: array[0..M-1] of rocblas_float_complex;
  hostY: array[0..N-1] of rocblas_float_complex;
  i, j: Integer;
  term, y_conj: rocblas_float_complex;
begin
  SetLength(hostA, M*N);
  SetLength(hostA_orig, M*N);
  SetLength(expectedA, M*N);

  for j := 0 to N-1 do for i := 0 to M-1 do
  begin
    hostA[j*M + i].x := (j+1)*10 + (i+1);
    hostA[j*M + i].y := -((j+1)*5 + (i+1));
  end;
  hostA_orig := hostA;
  for i := 0 to M-1 do begin hostX[i].x := i + 1; hostX[i].y := i - 1; end;
  for i := 0 to N-1 do begin hostY[i].x := i + 2; hostY[i].y := i - 2; end;

  // CPU reference: A := alpha*x*conj(y)**T + A
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to M-1 do
    begin
      y_conj := hostY[j];
      y_conj.y := -y_conj.y; // Conjugate y

      term := hostX[i];
      ComplexMul(term, y_conj);  // x[i] * conj(y[j])
      ComplexMul(term, alpha);      // alpha * x[i] * conj(y[j])
      ComplexAdd(expectedA[j*M + i], term);
    end;

  hipMalloc(devA, M*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, M*SizeOf(rocblas_float_complex));
  hipMalloc(devY, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devA, @hostA[0], M*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], M*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_cgerc OK', Integer(rocblas_status_success),
    Integer(rocblas_cgerc(handle, M, N, @alpha, devX, 1, devY, 1, devA, M)));

  hipMemcpy(@hostA[0], devA, M*N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to M*N-1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA[i].x, 1e-5, Format('CGERC A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA[i].y, 1e-5, Format('CGERC A[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_ZGERC; // Conjugated
const
  M = 4; N = 5;
  alpha: rocblas_double_complex = (x: 1.5; y: -0.5);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array of rocblas_double_complex;
  hostX: array[0..M-1] of rocblas_double_complex;
  hostY: array[0..N-1] of rocblas_double_complex;
  i, j: Integer;
  term, y_conj: rocblas_double_complex;
begin
  SetLength(hostA, M*N);
  SetLength(hostA_orig, M*N);
  SetLength(expectedA, M*N);

  for j := 0 to N-1 do for i := 0 to M-1 do
  begin
    hostA[j*M + i].x := (j+1)*10 + (i+1);
    hostA[j*M + i].y := -((j+1)*5 + (i+1));
  end;
  hostA_orig := hostA;
  for i := 0 to M-1 do begin hostX[i].x := i + 1; hostX[i].y := i - 1; end;
  for i := 0 to N-1 do begin hostY[i].x := i + 2; hostY[i].y := i - 2; end;

  // CPU reference: A := alpha*x*conj(y)**T + A
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to M-1 do
    begin
      y_conj := hostY[j];
      y_conj.y := -y_conj.y; // Conjugate y

      term := hostX[i];
      ComplexMul(term, y_conj);  // x[i] * conj(y[j])
      ComplexMul(term, alpha);      // alpha * x[i] * conj(y[j])
      ComplexAdd(expectedA[j*M + i], term);
    end;

  hipMalloc(devA, M*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, M*SizeOf(rocblas_double_complex));
  hipMalloc(devY, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devA, @hostA[0], M*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], M*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zgerc OK', Integer(rocblas_status_success),
    Integer(rocblas_zgerc(handle, M, N, @alpha, devX, 1, devY, 1, devA, M)));

  hipMemcpy(@hostA[0], devA, M*N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to M*N-1 do
  begin
    CheckEqualsRelative(expectedA[i].x, hostA[i].x, 1e-12, Format('ZGERC A[%d].x', [i]));
    CheckEqualsRelative(expectedA[i].y, hostA[i].y, 1e-12, Format('ZGERC A[%d].y', [i]));
  end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_SSYR;
const
  N = 5;
  alpha: Single = 2.5;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array[0..N*N-1] of Single;
  hostX: array[0..N-1] of Single;
  i, j: Integer;
begin
  for j := 0 to N-1 do for i := 0 to N-1 do hostA[j*N+i] := (j+1)*10 + (i+1);
  hostA_orig := hostA;
  for i := 0 to N-1 do hostX[i] := i + 1.0;

  // CPU reference: A := alpha*x*x**T + A
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to j do // For uplo=upper, only update upper triangle
    begin
      expectedA[j*N + i] := expectedA[j*N + i] + alpha * hostX[i] * hostX[j];
      if i <> j then
        expectedA[i*N + j] := expectedA[j*N + i]; // Maintain symmetry
    end;

  hipMalloc(devA, N*N*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ssyr OK', Integer(rocblas_status_success),
    Integer(rocblas_ssyr(handle, rocblas_fill_upper, N, @alpha, devX, 1, devA, N)));

  hipMemcpy(@hostA[0], devA, N*N*SizeOf(Single), hipMemcpyDeviceToHost);

  // Check only the upper triangle, as the lower part is not guaranteed to be updated
  for j := 0 to N-1 do
    for i := 0 to j do
      CheckEqualsRelative(expectedA[j*N+i], hostA[j*N+i], 1e-5, Format('SSYR A[%d,%d]',[i,j]));

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_DSYR;
const
  N = 5;
  alpha: Double = 2.5;
var
  devA, devX: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array[0..N*N-1] of Double;
  hostX: array[0..N-1] of Double;
  i, j: Integer;
begin
  for j := 0 to N-1 do for i := 0 to N-1 do hostA[j*N+i] := (j+1)*10 + (i+1);
  hostA_orig := hostA;
  for i := 0 to N-1 do hostX[i] := i + 1.0;

  // CPU reference: A := alpha*x*x**T + A
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to j do // For uplo=upper, we only need to calculate the upper triangle
    begin
      expectedA[j*N + i] := expectedA[j*N + i] + alpha * hostX[i] * hostX[j];
    end;

  hipMalloc(devA, N*N*SizeOf(Double));
  hipMalloc(devX, N*SizeOf(Double));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dsyr OK', Integer(rocblas_status_success),
    Integer(rocblas_dsyr(handle, rocblas_fill_upper, N, @alpha, devX, 1, devA, N)));

  hipMemcpy(@hostA[0], devA, N*N*SizeOf(Double), hipMemcpyDeviceToHost);

  // Check only the upper triangle, as the lower part is not guaranteed to be updated
  for j := 0 to N-1 do
    for i := 0 to j do
      CheckEqualsRelative(expectedA[j*N+i], hostA[j*N+i], 1e-12, Format('DSYR A[%d,%d]',[i,j]));

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_CSYR;
const
  N = 5;
  alpha: rocblas_float_complex = (x: 1.5; y: -0.5);
var
  devA, devX: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array[0..N*N-1] of rocblas_float_complex;
  hostX: array[0..N-1] of rocblas_float_complex;
  i, j: Integer;
  term: rocblas_float_complex;
begin
  // Initialize matrix A and vector x
  for j := 0 to N-1 do for i := 0 to N-1 do
  begin
    hostA[j*N+i].x := i+j+1;
    hostA[j*N+i].y := (i-j)/2.0;
  end;
  hostA_orig := hostA;
  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+1); end;

  // CPU reference: A := alpha*x*x**T + A (note: not conjugated)
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to j do // For uplo=upper
    begin
      term := hostX[i];
      ComplexMul(term, hostX[j]); // x[i] * x[j]
      ComplexMul(term, alpha);     // alpha * x[i] * x[j]
      ComplexAdd(expectedA[j*N + i], term);
    end;

  hipMalloc(devA, N*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devA, @hostA_orig[0], N*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_csyr OK', Integer(rocblas_status_success),
    Integer(rocblas_csyr(handle, rocblas_fill_upper, N, @alpha, devX, 1, devA, N)));

  hipMemcpy(@hostA[0], devA, N*N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  // Check only the upper triangle
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(expectedA[j*N+i].x, hostA[j*N+i].x, 1e-5, Format('CSYR A[%d,%d].x', [i,j]));
      CheckEqualsRelative(expectedA[j*N+i].y, hostA[j*N+i].y, 1e-5, Format('CSYR A[%d,%d].y', [i,j]));
    end;

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_ZSYR;
const
  N = 5;
  alpha: rocblas_double_complex = (x: 1.5; y: -0.5);
var
  devA, devX: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array[0..N*N-1] of rocblas_double_complex;
  hostX: array[0..N-1] of rocblas_double_complex;
  i, j: Integer;
  term: rocblas_double_complex;
begin
  // Initialize matrix A and vector x
  for j := 0 to N-1 do for i := 0 to N-1 do
  begin
    hostA[j*N+i].x := i+j+1;
    hostA[j*N+i].y := (i-j)/2.0;
  end;
  hostA_orig := hostA;
  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+1); end;

  // CPU reference: A := alpha*x*x**T + A (note: not conjugated)
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to j do // For uplo=upper
    begin
      term := hostX[i];
      ComplexMul(term, hostX[j]); // x[i] * x[j]
      ComplexMul(term, alpha);     // alpha * x[i] * x[j]
      ComplexAdd(expectedA[j*N + i], term);
    end;

  hipMalloc(devA, N*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devA, @hostA_orig[0], N*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zsyr OK', Integer(rocblas_status_success),
    Integer(rocblas_zsyr(handle, rocblas_fill_upper, N, @alpha, devX, 1, devA, N)));

  hipMemcpy(@hostA[0], devA, N*N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  // Check only the upper triangle
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(expectedA[j*N+i].x, hostA[j*N+i].x, 1e-12, Format('ZSYR A[%d,%d].x', [i,j]));
      CheckEqualsRelative(expectedA[j*N+i].y, hostA[j*N+i].y, 1e-12, Format('ZSYR A[%d,%d].y', [i,j]));
    end;

  hipFree(devA); hipFree(devX);
end;
procedure TTestRocBLAS_Level2.Test_CHER;
const
  N = 5;
  alpha: Single = 2.5; // Note: alpha is REAL for HER
var
  devA, devX: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array[0..N*N-1] of rocblas_float_complex;
  hostX: array[0..N-1] of rocblas_float_complex;
  i, j: Integer;
  term: rocblas_float_complex;
begin
  for j := 0 to N-1 do for i := 0 to N-1 do
  begin
    hostA[j*N+i].x := (j+1)*10 + (i+1);
    hostA[j*N+i].y := -((j+1)*5 + (i+1));
  end;
  hostA_orig := hostA;
  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+1); end;

  // CPU reference: A := alpha*x*x**H + A
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to j do // For uplo=upper
    begin
      term := hostX[i];
      term.y := -term.y; // Conjugate x[j]
      ComplexMul(term, hostX[j]);
      term.x := term.x * alpha;
      term.y := term.y * alpha;
      ComplexAdd(expectedA[j*N + i], term);
    end;
  // Diagonal must be real after update
  for i := 0 to N-1 do expectedA[i*N+i].y := 0;

  hipMalloc(devA, N*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_cher OK', Integer(rocblas_status_success),
    Integer(rocblas_cher(handle, rocblas_fill_upper, N, @alpha, devX, 1, devA, N)));

  hipMemcpy(@hostA[0], devA, N*N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(expectedA[j*N+i].x, hostA[j*N+i].x, 1e-5, Format('CHER A[%d,%d].x', [i,j]));
      if i <> j then // Imaginary part of diagonal is undefined
         CheckEqualsRelative(expectedA[j*N+i].y, hostA[j*N+i].y, 1e-5, Format('CHER A[%d,%d].y', [i,j]));
    end;

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_ZHER;
const
  N = 5;
  alpha: Double = 2.5; // Note: alpha is REAL for HER
var
  devA, devX: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array[0..N*N-1] of rocblas_double_complex;
  hostX: array[0..N-1] of rocblas_double_complex;
  i, j: Integer;
  term, x_conj: rocblas_double_complex;
begin
  for j := 0 to N-1 do for i := 0 to N-1 do
  begin
    hostA[j*N+i].x := (j+1)*10 + (i+1);
    hostA[j*N+i].y := -((j+1)*5 + (i+1));
  end;
  hostA_orig := hostA;
  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+1); end;

  // CPU reference: A := alpha*x*x**H + A
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to j do // For uplo=upper
    begin
      x_conj := hostX[j];
      x_conj.y := -x_conj.y; // Conjugate x[j]

      term := hostX[i];
      ComplexMul(term, x_conj);
      term.x := term.x * alpha;
      term.y := term.y * alpha;
      ComplexAdd(expectedA[j*N + i], term);
    end;
  // Diagonal must be real after update
  for i := 0 to N-1 do expectedA[i*N+i].y := 0;

  hipMalloc(devA, N*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zher OK', Integer(rocblas_status_success),
    Integer(rocblas_zher(handle, rocblas_fill_upper, N, @alpha, devX, 1, devA, N)));

  hipMemcpy(@hostA[0], devA, N*N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(expectedA[j*N+i].x, hostA[j*N+i].x, 1e-12, Format('ZHER A[%d,%d].x', [i,j]));
      if i <> j then // Imaginary part of diagonal is undefined/untouched by rocBLAS
         CheckEqualsRelative(expectedA[j*N+i].y, hostA[j*N+i].y, 1e-12, Format('ZHER A[%d,%d].y', [i,j]));
    end;

  hipFree(devA); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_SSPR;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
  alpha: Single = 2.5;
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostAP_orig, expectedAP: array[0..PACKED_SIZE-1] of Single;
  hostX: array[0..N-1] of Single;
  i, j, k: Integer;
begin
  // Initialize packed symmetric matrix and vector
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := i + j + 1.5;
      inc(k);
    end;
  hostAP_orig := hostAP;
  for i := 0 to N-1 do hostX[i] := i + 1.0;

  // CPU Reference: A := alpha*x*x**T + A (in packed form)
  expectedAP := hostAP_orig;
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      expectedAP[k] := expectedAP[k] + alpha * hostX[i] * hostX[j];
      inc(k);
    end;

  hipMalloc(devAP, PACKED_SIZE*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_sspr OK', Integer(rocblas_status_success),
    Integer(rocblas_sspr(handle, rocblas_fill_upper, N, @alpha, devX, 1, devAP)));

  hipMemcpy(@hostAP[0], devAP, PACKED_SIZE*SizeOf(Single), hipMemcpyDeviceToHost);

  for i := 0 to PACKED_SIZE-1 do
    CheckEqualsRelative(expectedAP[i], hostAP[i], 1e-5, Format('SSPR AP[%d]', [i]));

  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_DSPR;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
  alpha: Double = 2.5;
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostAP_orig, expectedAP: array[0..PACKED_SIZE-1] of Double;
  hostX: array[0..N-1] of Double;
  i, j, k: Integer;
begin
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k] := i + j + 1.5;
      inc(k);
    end;
  hostAP_orig := hostAP;
  for i := 0 to N-1 do hostX[i] := i + 1.0;

  expectedAP := hostAP_orig;
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      expectedAP[k] := expectedAP[k] + alpha * hostX[i] * hostX[j];
      inc(k);
    end;

  hipMalloc(devAP, PACKED_SIZE*SizeOf(Double));
  hipMalloc(devX, N*SizeOf(Double));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dspr OK', Integer(rocblas_status_success),
    Integer(rocblas_dspr(handle, rocblas_fill_upper, N, @alpha, devX, 1, devAP)));

  hipMemcpy(@hostAP[0], devAP, PACKED_SIZE*SizeOf(Double), hipMemcpyDeviceToHost);

  for i := 0 to PACKED_SIZE-1 do
    CheckEqualsRelative(expectedAP[i], hostAP[i], 1e-12, Format('DSPR AP[%d]', [i]));

  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_CHPR;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
  alpha: Single = 2.5; // alpha is REAL for HPR
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostAP_orig, expectedAP: array[0..PACKED_SIZE-1] of rocblas_float_complex;
  hostX: array[0..N-1] of rocblas_float_complex;
  i, j, k: Integer;
  term, x_conj: rocblas_float_complex;
begin
  // Initialize packed Hermitian matrix and vector
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k].x := i+j+1.5;
      hostAP[k].y := (i-j)/2.0;
      if i=j then hostAP[k].y := 0; // Real diagonal
      inc(k);
    end;
  hostAP_orig := hostAP;
  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+1); end;

  // CPU reference: AP := alpha*x*x**H + AP
  expectedAP := hostAP_orig;
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      x_conj := hostX[j];
      x_conj.y := -x_conj.y; // Conjugate x[j]
      term := hostX[i];
      ComplexMul(term, x_conj);
      term.x := term.x * alpha;
      term.y := term.y * alpha;
      ComplexAdd(expectedAP[k], term);
      if i=j then expectedAP[k].y := 0; // Result diagonal is real
      inc(k);
    end;

  hipMalloc(devAP, PACKED_SIZE*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_chpr OK', Integer(rocblas_status_success),
    Integer(rocblas_chpr(handle, rocblas_fill_upper, N, @alpha, devX, 1, devAP)));

  hipMemcpy(@hostAP[0], devAP, PACKED_SIZE*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  for i := 0 to PACKED_SIZE-1 do
  begin
    CheckEqualsRelative(expectedAP[i].x, hostAP[i].x, 1e-5, Format('CHPR AP[%d].x', [i]));
    CheckEqualsRelative(expectedAP[i].y, hostAP[i].y, 1e-5, Format('CHPR AP[%d].y', [i]));
  end;

  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_ZHPR;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
  alpha: Double = 2.5; // alpha is REAL for HPR
var
  devAP, devX: hipDeviceptr_t;
  hostAP, hostAP_orig, expectedAP: array[0..PACKED_SIZE-1] of rocblas_double_complex;
  hostX: array[0..N-1] of rocblas_double_complex;
  i, j, k: Integer;
  term, x_conj: rocblas_double_complex;
begin
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      hostAP[k].x := i+j+1.5;
      hostAP[k].y := (i-j)/2.0;
      if i=j then hostAP[k].y := 0;
      inc(k);
    end;
  hostAP_orig := hostAP;
  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+1); end;

  expectedAP := hostAP_orig;
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      x_conj := hostX[j];
      x_conj.y := -x_conj.y;
      term := hostX[i];
      ComplexMul(term, x_conj);
      term.x := term.x * alpha;
      term.y := term.y * alpha;
      ComplexAdd(expectedAP[k], term);
      if i=j then expectedAP[k].y := 0;
      inc(k);
    end;

  hipMalloc(devAP, PACKED_SIZE*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zhpr OK', Integer(rocblas_status_success),
    Integer(rocblas_zhpr(handle, rocblas_fill_upper, N, @alpha, devX, 1, devAP)));

  hipMemcpy(@hostAP[0], devAP, PACKED_SIZE*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  for i := 0 to PACKED_SIZE-1 do
  begin
    CheckEqualsRelative(expectedAP[i].x, hostAP[i].x, 1e-12, Format('ZHPR AP[%d].x', [i]));
    CheckEqualsRelative(expectedAP[i].y, hostAP[i].y, 1e-12, Format('ZHPR AP[%d].y', [i]));
  end;

  hipFree(devAP); hipFree(devX);
end;

procedure TTestRocBLAS_Level2.Test_SSYR2;
const
  N = 5;
  alpha: Single = 1.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array[0..N*N-1] of Single;
  hostX, hostY: array[0..N-1] of Single;
  i, j: Integer;
begin
  // 1. Initialize matrices and vectors
  for j := 0 to N-1 do for i := 0 to j do
  begin
    hostA[j*N+i] := i+j+2;
    hostA[i*N+j] := i+j+2; //Symmetric
  end;
  hostA_orig := hostA;
  for i := 0 to N-1 do begin hostX[i] := i+1.0; hostY[i] := N-i+0.5; end;

  // 2. CPU Reference: A := alpha*x*y**T + alpha*y*x**T + A
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to j do // For uplo=upper
    begin
      expectedA[j*N+i] := expectedA[j*N+i] + alpha * hostX[i] * hostY[j] + alpha * hostY[i] * hostX[j];
    end;

  // 3. ROCm Execution
  hipMalloc(devA, N*N*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));
  hipMalloc(devY, N*SizeOf(Single));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_ssyr2 OK', Integer(rocblas_status_success),
    Integer(rocblas_ssyr2(handle, rocblas_fill_upper, N, @alpha, devX, 1, devY, 1, devA, N)));

  // 4. Verify results
  hipMemcpy(@hostA[0], devA, N*N*SizeOf(Single), hipMemcpyDeviceToHost);
  for j := 0 to N-1 do
    for i := 0 to j do
      CheckEqualsRelative(expectedA[j*N+i], hostA[j*N+i], 1e-5, Format('SSYR2 A[%d,%d]',[i,j]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_DSYR2;
const
  N = 5;
  alpha: Double = 1.5;
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array[0..N*N-1] of Double;
  hostX, hostY: array[0..N-1] of Double;
  i, j: Integer;
begin
  for j := 0 to N-1 do for i := 0 to j do
  begin
    hostA[j*N+i] := i+j+2;
    hostA[i*N+j] := i+j+2;
  end;
  hostA_orig := hostA;
  for i := 0 to N-1 do begin hostX[i] := i+1.0; hostY[i] := N-i+0.5; end;

  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      expectedA[j*N+i] := expectedA[j*N+i] + alpha * hostX[i] * hostY[j] + alpha * hostY[i] * hostX[j];
    end;

  hipMalloc(devA, N*N*SizeOf(Double));
  hipMalloc(devX, N*SizeOf(Double));
  hipMalloc(devY, N*SizeOf(Double));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dsyr2 OK', Integer(rocblas_status_success),
    Integer(rocblas_dsyr2(handle, rocblas_fill_upper, N, @alpha, devX, 1, devY, 1, devA, N)));

  hipMemcpy(@hostA[0], devA, N*N*SizeOf(Double), hipMemcpyDeviceToHost);
  for j := 0 to N-1 do
    for i := 0 to j do
      CheckEqualsRelative(expectedA[j*N+i], hostA[j*N+i], 1e-12, Format('DSYR2 A[%d,%d]',[i,j]));

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_CSYR2;
const
  N = 5;
  alpha: rocblas_float_complex = (x: 1.5; y: -0.5);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array[0..N*N-1] of rocblas_float_complex;
  hostX, hostY: array[0..N-1] of rocblas_float_complex;
  i, j: Integer;
  term1, term2: rocblas_float_complex;
begin
  for j := 0 to N-1 do for i := 0 to j do
  begin
    hostA[j*N+i].x := i+j+2;
    hostA[j*N+i].y := (i-j)/2.0;
    hostA[i*N+j] := hostA[j*N+i];
  end;
  hostA_orig := hostA;
  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+0.5); end;
  for i := 0 to N-1 do begin hostY[i].x := N-i; hostY[i].y := i-0.5; end;

  // CPU Ref: A := alpha*x*y**T + alpha*y*x**T + A
  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      term1 := hostX[i];
      ComplexMul(term1, hostY[j]); // x[i]*y[j]**T
      ComplexMul(term1, alpha);   // alpha*x[i]*y[j]**T

      term2 := hostY[i];
      ComplexMul(term2, hostX[j]); // y[i]*x[j]**T
      ComplexMul(term2, alpha);   // alpha*y[i]*x[j]**T

      ComplexAdd(expectedA[j*N+i], term1);
      ComplexAdd(expectedA[j*N+i], term2);
    end;

  hipMalloc(devA, N*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMalloc(devY, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_csyr2 OK', Integer(rocblas_status_success),
    Integer(rocblas_csyr2(handle, rocblas_fill_upper, N, @alpha, devX, 1, devY, 1, devA, N)));

  hipMemcpy(@hostA[0], devA, N*N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(expectedA[j*N+i].x, hostA[j*N+i].x, 1e-5, Format('CSYR2 A[%d,%d].x',[i,j]));
      CheckEqualsRelative(expectedA[j*N+i].y, hostA[j*N+i].y, 1e-5, Format('CSYR2 A[%d,%d].y',[i,j]));
    end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_ZSYR2;
const
  N = 5;
  alpha: rocblas_double_complex = (x: 1.5; y: -0.5);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array[0..N*N-1] of rocblas_double_complex;
  hostX, hostY: array[0..N-1] of rocblas_double_complex;
  i, j: Integer;
  term1, term2: rocblas_double_complex;
begin
  for j := 0 to N-1 do for i := 0 to j do
  begin
    hostA[j*N+i].x := i+j+2;
    hostA[j*N+i].y := (i-j)/2.0;
    hostA[i*N+j] := hostA[j*N+i];
  end;
  hostA_orig := hostA;
  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+0.5); end;
  for i := 0 to N-1 do begin hostY[i].x := N-i; hostY[i].y := i-0.5; end;

  expectedA := hostA_orig;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      term1 := hostX[i];
      ComplexMul(term1, hostY[j]);
      ComplexMul(term1, alpha);

      term2 := hostY[i];
      ComplexMul(term2, hostX[j]);
      ComplexMul(term2, alpha);

      ComplexAdd(expectedA[j*N+i], term1);
      ComplexAdd(expectedA[j*N+i], term2);
    end;

  hipMalloc(devA, N*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMalloc(devY, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zsyr2 OK', Integer(rocblas_status_success),
    Integer(rocblas_zsyr2(handle, rocblas_fill_upper, N, @alpha, devX, 1, devY, 1, devA, N)));

  hipMemcpy(@hostA[0], devA, N*N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(expectedA[j*N+i].x, hostA[j*N+i].x, 1e-12, Format('ZSYR2 A[%d,%d].x',[i,j]));
      CheckEqualsRelative(expectedA[j*N+i].y, hostA[j*N+i].y, 1e-12, Format('ZSYR2 A[%d,%d].y',[i,j]));
    end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_CHER2;
const
  N = 5;
  alpha: rocblas_float_complex = (x: 1.5; y: -0.5);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array[0..N*N-1] of rocblas_float_complex;
  hostX, hostY: array[0..N-1] of rocblas_float_complex;
  i, j: Integer;
  alpha_conj, term1, term2, x_conj, y_conj: rocblas_float_complex;
begin
  // 1. Initialize matrices and vectors
  for j := 0 to N-1 do for i := 0 to j do
  begin
    hostA[j*N+i].x := i+j+2;
    hostA[j*N+i].y := (i-j)/2.0;
    if i=j then hostA[j*N+i].y := 0; // Real diagonal
    hostA[i*N+j].x := hostA[j*N+i].x; // Hermitian
    hostA[i*N+j].y := -hostA[j*N+i].y;
  end;
  hostA_orig := hostA;
  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+0.5); end;
  for i := 0 to N-1 do begin hostY[i].x := N-i; hostY[i].y := i-0.5; end;

  // 2. CPU Reference: A := alpha*x*y**H + conj(alpha)*y*x**H + A
  expectedA := hostA_orig;
  alpha_conj := alpha;
  alpha_conj.y := -alpha_conj.y; // Conjugate alpha

  for j := 0 to N-1 do
    for i := 0 to j do // For uplo=upper
    begin
      // First term: alpha * x[i] * y[j]**H
      y_conj := hostY[j];
      y_conj.y := -y_conj.y; // Conjugate y
      term1 := hostX[i];
      ComplexMul(term1, y_conj);
      ComplexMul(term1, alpha);

      // Second term: conj(alpha) * y[i] * x[j]**H
      x_conj := hostX[j];
      x_conj.y := -x_conj.y; // Conjugate x
      term2 := hostY[i];
      ComplexMul(term2, x_conj);
      ComplexMul(term2, alpha_conj);

      ComplexAdd(expectedA[j*N+i], term1);
      ComplexAdd(expectedA[j*N+i], term2);
    end;
  // Ensure diagonal remains real
  for i := 0 to N-1 do expectedA[i*N+i].y := 0;

  // 3. ROCm Execution
  hipMalloc(devA, N*N*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMalloc(devY, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_cher2 OK', Integer(rocblas_status_success),
    Integer(rocblas_cher2(handle, rocblas_fill_upper, N, @alpha, devX, 1, devY, 1, devA, N)));

  // 4. Verify results
  hipMemcpy(@hostA[0], devA, N*N*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(expectedA[j*N+i].x, hostA[j*N+i].x, 1e-5, Format('CHER2 A[%d,%d].x',[i,j]));
      if i <> j then
        CheckEqualsRelative(expectedA[j*N+i].y, hostA[j*N+i].y, 1e-5, Format('CHER2 A[%d,%d].y',[i,j]));
    end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_ZHER2;
const
  N = 5;
  alpha: rocblas_double_complex = (x: 1.5; y: -0.5);
var
  devA, devX, devY: hipDeviceptr_t;
  hostA, hostA_orig, expectedA: array[0..N*N-1] of rocblas_double_complex;
  hostX, hostY: array[0..N-1] of rocblas_double_complex;
  i, j: Integer;
  alpha_conj, term1, term2, x_conj, y_conj: rocblas_double_complex;
begin
  for j := 0 to N-1 do for i := 0 to j do
  begin
    hostA[j*N+i].x := i+j+2;
    hostA[j*N+i].y := (i-j)/2.0;
    if i=j then hostA[j*N+i].y := 0;
    hostA[i*N+j].x := hostA[j*N+i].x;
    hostA[i*N+j].y := -hostA[j*N+i].y;
  end;
  hostA_orig := hostA;
  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+0.5); end;
  for i := 0 to N-1 do begin hostY[i].x := N-i; hostY[i].y := i-0.5; end;

  expectedA := hostA_orig;
  alpha_conj := alpha;
  alpha_conj.y := -alpha_conj.y;

  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      y_conj := hostY[j];
      y_conj.y := -y_conj.y;
      term1 := hostX[i];
      ComplexMul(term1, y_conj);
      ComplexMul(term1, alpha);

      x_conj := hostX[j];
      x_conj.y := -x_conj.y;
      term2 := hostY[i];
      ComplexMul(term2, x_conj);
      ComplexMul(term2, alpha_conj);

      ComplexAdd(expectedA[j*N+i], term1);
      ComplexAdd(expectedA[j*N+i], term2);
    end;
  for i := 0 to N-1 do expectedA[i*N+i].y := 0;

  hipMalloc(devA, N*N*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMalloc(devY, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devA, @hostA[0], N*N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zher2 OK', Integer(rocblas_status_success),
    Integer(rocblas_zher2(handle, rocblas_fill_upper, N, @alpha, devX, 1, devY, 1, devA, N)));

  hipMemcpy(@hostA[0], devA, N*N*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      CheckEqualsRelative(expectedA[j*N+i].x, hostA[j*N+i].x, 1e-12, Format('ZHER2 A[%d,%d].x',[i,j]));
      if i <> j then
        CheckEqualsRelative(expectedA[j*N+i].y, hostA[j*N+i].y, 1e-12, Format('ZHER2 A[%d,%d].y',[i,j]));
    end;

  hipFree(devA); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_SSPR2;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
  alpha: Single = 1.5;
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostAP_orig, expectedAP: array[0..PACKED_SIZE-1] of Single;
  hostX, hostY: array[0..N-1] of Single;
  i, j, k: Integer;
begin
  // 1. Initialize packed matrix and vectors
  k := 0;
  for j := 0 to N-1 do for i := 0 to j do
  begin
    hostAP[k] := i + j + 1.5;
    inc(k);
  end;
  hostAP_orig := hostAP;
  for i := 0 to N-1 do begin hostX[i] := i + 1.0; hostY[i] := N-i+0.5; end;

  // 2. CPU Reference: A := alpha*x*y**T + alpha*y*x**T + A
  expectedAP := hostAP_orig;
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      expectedAP[k] := expectedAP[k] + alpha * hostX[i] * hostY[j] + alpha * hostY[i] * hostX[j];
      inc(k);
    end;

  // 3. ROCm Execution
  hipMalloc(devAP, PACKED_SIZE*SizeOf(Single));
  hipMalloc(devX, N*SizeOf(Single));
  hipMalloc(devY, N*SizeOf(Single));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_sspr2 OK', Integer(rocblas_status_success),
    Integer(rocblas_sspr2(handle, rocblas_fill_upper, N, @alpha, devX, 1, devY, 1, devAP)));

  // 4. Verify results
  hipMemcpy(@hostAP[0], devAP, PACKED_SIZE*SizeOf(Single), hipMemcpyDeviceToHost);
  for i := 0 to PACKED_SIZE-1 do
    CheckEqualsRelative(expectedAP[i], hostAP[i], 1e-5, Format('SSPR2 AP[%d]', [i]));

  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_DSPR2;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
  alpha: Double = 1.5;
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostAP_orig, expectedAP: array[0..PACKED_SIZE-1] of Double;
  hostX, hostY: array[0..N-1] of Double;
  i, j, k: Integer;
begin
  k := 0;
  for j := 0 to N-1 do for i := 0 to j do
  begin
    hostAP[k] := i + j + 1.5;
    inc(k);
  end;
  hostAP_orig := hostAP;
  for i := 0 to N-1 do begin hostX[i] := i + 1.0; hostY[i] := N-i+0.5; end;

  expectedAP := hostAP_orig;
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      expectedAP[k] := expectedAP[k] + alpha * hostX[i] * hostY[j] + alpha * hostY[i] * hostX[j];
      inc(k);
    end;

  hipMalloc(devAP, PACKED_SIZE*SizeOf(Double));
  hipMalloc(devX, N*SizeOf(Double));
  hipMalloc(devY, N*SizeOf(Double));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_dspr2 OK', Integer(rocblas_status_success),
    Integer(rocblas_dspr2(handle, rocblas_fill_upper, N, @alpha, devX, 1, devY, 1, devAP)));

  hipMemcpy(@hostAP[0], devAP, PACKED_SIZE*SizeOf(Double), hipMemcpyDeviceToHost);
  for i := 0 to PACKED_SIZE-1 do
    CheckEqualsRelative(expectedAP[i], hostAP[i], 1e-12, Format('DSPR2 AP[%d]', [i]));

  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_CHPR2;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
  alpha: rocblas_float_complex = (x: 1.5; y: -0.5);
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostAP_orig, expectedAP: array[0..PACKED_SIZE-1] of rocblas_float_complex;
  hostX, hostY: array[0..N-1] of rocblas_float_complex;
  i, j, k: Integer;
  alpha_conj, term1, term2, x_conj, y_conj: rocblas_float_complex;
begin
  k := 0;
  for j := 0 to N-1 do for i := 0 to j do
  begin
    hostAP[k].x := i+j+2;
    hostAP[k].y := (i-j)/2.0;
    if i=j then hostAP[k].y := 0;
    inc(k);
  end;
  hostAP_orig := hostAP;
  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+0.5); end;
  for i := 0 to N-1 do begin hostY[i].x := N-i; hostY[i].y := i-0.5; end;

  // CPU Ref: AP := alpha*x*y**H + conj(alpha)*y*x**H + AP
  expectedAP := hostAP_orig;
  alpha_conj := alpha;
  alpha_conj.y := -alpha_conj.y;
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      y_conj := hostY[j];
      y_conj.y := -y_conj.y;
      term1 := hostX[i];
      ComplexMul(term1, y_conj);
      ComplexMul(term1, alpha);

      x_conj := hostX[j];
      x_conj.y := -x_conj.y;
      term2 := hostY[i];
      ComplexMul(term2, x_conj);
      ComplexMul(term2, alpha_conj);

      ComplexAdd(expectedAP[k], term1);
      ComplexAdd(expectedAP[k], term2);
      if i=j then expectedAP[k].y := 0;
      inc(k);
    end;

  hipMalloc(devAP, PACKED_SIZE*SizeOf(rocblas_float_complex));
  hipMalloc(devX, N*SizeOf(rocblas_float_complex));
  hipMalloc(devY, N*SizeOf(rocblas_float_complex));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_chpr2 OK', Integer(rocblas_status_success),
    Integer(rocblas_chpr2(handle, rocblas_fill_upper, N, @alpha, devX, 1, devY, 1, devAP)));

  hipMemcpy(@hostAP[0], devAP, PACKED_SIZE*SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  for i := 0 to PACKED_SIZE-1 do
  begin
    CheckEqualsRelative(expectedAP[i].x, hostAP[i].x, 1e-5, Format('CHPR2 AP[%d].x', [i]));
    CheckEqualsRelative(expectedAP[i].y, hostAP[i].y, 1e-5, Format('CHPR2 AP[%d].y', [i]));
  end;

  hipFree(devAP); hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level2.Test_ZHPR2;
const
  N = 6;
  PACKED_SIZE = (N * (N + 1)) div 2;
  alpha: rocblas_double_complex = (x: 1.5; y: -0.5);
var
  devAP, devX, devY: hipDeviceptr_t;
  hostAP, hostAP_orig, expectedAP: array[0..PACKED_SIZE-1] of rocblas_double_complex;
  hostX, hostY: array[0..N-1] of rocblas_double_complex;
  i, j, k: Integer;
  alpha_conj, term1, term2, x_conj, y_conj: rocblas_double_complex;
begin
  k := 0;
  for j := 0 to N-1 do for i := 0 to j do
  begin
    hostAP[k].x := i+j+2;
    hostAP[k].y := (i-j)/2.0;
    if i=j then hostAP[k].y := 0;
    inc(k);
  end;
  hostAP_orig := hostAP;
  for i := 0 to N-1 do begin hostX[i].x := i+1; hostX[i].y := -(i+0.5); end;
  for i := 0 to N-1 do begin hostY[i].x := N-i; hostY[i].y := i-0.5; end;

  expectedAP := hostAP_orig;
  alpha_conj := alpha;
  alpha_conj.y := -alpha_conj.y;
  k := 0;
  for j := 0 to N-1 do
    for i := 0 to j do
    begin
      y_conj := hostY[j];
      y_conj.y := -y_conj.y;
      term1 := hostX[i];
      ComplexMul(term1, y_conj);
      ComplexMul(term1, alpha);

      x_conj := hostX[j];
      x_conj.y := -x_conj.y;
      term2 := hostY[i];
      ComplexMul(term2, x_conj);
      ComplexMul(term2, alpha_conj);

      ComplexAdd(expectedAP[k], term1);
      ComplexAdd(expectedAP[k], term2);
      if i=j then expectedAP[k].y := 0;
      inc(k);
    end;

  hipMalloc(devAP, PACKED_SIZE*SizeOf(rocblas_double_complex));
  hipMalloc(devX, N*SizeOf(rocblas_double_complex));
  hipMalloc(devY, N*SizeOf(rocblas_double_complex));
  hipMemcpy(devAP, @hostAP[0], PACKED_SIZE*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devX, @hostX[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N*SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  AssertEquals('rocblas_zhpr2 OK', Integer(rocblas_status_success),
    Integer(rocblas_zhpr2(handle, rocblas_fill_upper, N, @alpha, devX, 1, devY, 1, devAP)));

  hipMemcpy(@hostAP[0], devAP, PACKED_SIZE*SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  for i := 0 to PACKED_SIZE-1 do
  begin
    CheckEqualsRelative(expectedAP[i].x, hostAP[i].x, 1e-12, Format('ZHPR2 AP[%d].x', [i]));
    CheckEqualsRelative(expectedAP[i].y, hostAP[i].y, 1e-12, Format('ZHPR2 AP[%d].y', [i]));
  end;

  hipFree(devAP); hipFree(devX); hipFree(devY);
end;


initialization
  RegisterTest(TTestRocBLAS_Level2);

end.
