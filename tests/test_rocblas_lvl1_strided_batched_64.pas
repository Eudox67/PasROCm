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

unit Test_rocblas_lvl1_strided_batched_64;

{$mode objfpc}{$H+}
{$linklib rocblas}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  { TTestRocBLAS_Level1_Strided_Batched_64 }
  // This unit tests the 64-bit Level 1 strided batched functions.
  // NOTE: According to the rocBLAS API, only the ROT, ROTG, ROTM, and ROTMG
  // function families have _strided_batched_64 variants.
  TTestRocBLAS_Level1_Strided_Batched_64 = class(TTestRocBLAS_Base)
  published
    procedure Test_SROT_Strided_Batched_64;
    procedure Test_DROT_Strided_Batched_64;
    procedure Test_CROT_Strided_Batched_64;
    procedure Test_CSROT_Strided_Batched_64;
    // Note: ZROT and ZDROT _strided_batched_64 do not exist in the API

    procedure Test_SROTG_Strided_Batched_64;
    procedure Test_DROTG_Strided_Batched_64;
    procedure Test_CROTG_Strided_Batched_64;
    procedure Test_ZROTG_Strided_Batched_64;

    procedure Test_SROTM_Strided_Batched_64;
    procedure Test_DROTM_Strided_Batched_64;

    procedure Test_SROTMG_Strided_Batched_64;
    procedure Test_DROTMG_Strided_Batched_64;
  end;

implementation

{ TTestRocBLAS_Level1_Strided_Batched_64 }

procedure TTestRocBLAS_Level1_Strided_Batched_64.Test_SROT_Strided_Batched_64;
const
  N: Int64 = 10;
  batch_count: Int64 = 3;
  incx: Int64 = 1;
  incy: Int64 = 1;
  stride = 16;
  total_size = 42; // stride * (batch_count - 1) + N;
type
  TSingleVector = array[0..41] of Single;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TSingleVector;
  d_x, d_y: hipDeviceptr_t;
  c, s: Single;
  b: Int64;
  i: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  c := 0.8; s := 0.6;
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      h_x[b*stride+i] := (b + 1) * 10.0 + i;
      h_y[b*stride+i] := (b + 1) * 20.0 - i;
      h_x_orig[b*stride+i] := h_x[b*stride+i];
      h_y_orig[b*stride+i] := h_y[b*stride+i];
      h_x_expected[b*stride+i] := c * h_x_orig[b*stride+i] + s * h_y_orig[b*stride+i];
      h_y_expected[b*stride+i] := -s * h_x_orig[b*stride+i] + c * h_y_orig[b*stride+i];
    end;

  // 2. Allocate and Copy
  hipMalloc(d_x, total_size * SizeOf(Single));
  hipMalloc(d_y, total_size * SizeOf(Single));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(Single), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_srot_strided_batched_64(handle, N, d_x, incx, stride, d_y, incy, stride, @c, @s, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_srot_strided_batched_64 is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_srot_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_x[0], d_x, total_size * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[0], d_y, total_size * SizeOf(Single), hipMemcpyDeviceToHost);
    // 4. Verify
    for b := 0 to batch_count - 1 do
      for i := 0 to N - 1 do
      begin
        CheckEquals(h_x_expected[b*stride+i], h_x[b*stride+i], 1e-5, 'SROT_Strided_64 x mismatch');
        CheckEquals(h_y_expected[b*stride+i], h_y[b*stride+i], 1e-5, 'SROT_Strided_64 y mismatch');
      end;
  end;

  // 5. Cleanup
  hipFree(d_x); hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched_64.Test_DROT_Strided_Batched_64;
const
  N: Int64 = 10;
  batch_count: Int64 = 3;
  incx: Int64 = 1;
  incy: Int64 = 1;
  stride = 16;
  total_size = 42; // stride * (batch_count - 1) + N;
type
  TDoubleVector = array[0..41] of Double;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TDoubleVector;
  d_x, d_y: hipDeviceptr_t;
  c, s: Double;
  b: Int64;
  i: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize
  c := Sqrt(2.0)/2.0; s := Sqrt(2.0)/2.0;
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      h_x[b*stride+i] := (b + 1.5) * 10.0 + i;
      h_y[b*stride+i] := (b + 1.5) * 20.0 - i;
      h_x_orig[b*stride+i] := h_x[b*stride+i];
      h_y_orig[b*stride+i] := h_y[b*stride+i];
      h_x_expected[b*stride+i] := c * h_x_orig[b*stride+i] + s * h_y_orig[b*stride+i];
      h_y_expected[b*stride+i] := -s * h_x_orig[b*stride+i] + c * h_y_orig[b*stride+i];
    end;

  // 2. Allocate and Copy
  hipMalloc(d_x, total_size * SizeOf(Double)); hipMalloc(d_y, total_size * SizeOf(Double));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);

  // 3. Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_drot_strided_batched_64(handle, N, d_x, incx, stride, d_y, incy, stride, @c, @s, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_drot_strided_batched_64 is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_drot_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_x[0], d_x, total_size * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[0], d_y, total_size * SizeOf(Double), hipMemcpyDeviceToHost);
    // 4. Verify
    for b := 0 to batch_count - 1 do
      for i := 0 to N - 1 do
      begin
        CheckEquals(h_x_expected[b*stride+i], h_x[b*stride+i], 1e-12, 'DROT_Strided_64 x mismatch');
        CheckEquals(h_y_expected[b*stride+i], h_y[b*stride+i], 1e-12, 'DROT_Strided_64 y mismatch');
      end;
  end;

  // 5. Cleanup
  hipFree(d_x); hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched_64.Test_CROT_Strided_Batched_64;
const
  N: Int64 = 10;
  batch_count: Int64 = 3;
  incx: Int64 = 1;
  incy: Int64 = 1;
  stride = 12;
    total_size = 34; // stride * (batch_count - 1) + N;
type
  TComplexVector = array[0..33] of rocblas_float_complex;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TComplexVector;
  d_x, d_y: hipDeviceptr_t;
  c: Single;
  s: rocblas_float_complex;
  b: Int64;
  i: Integer;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize
  c := 0.5;
  s.x := 0.8;
  s.y := Sqrt(1.0 - Sqr(c) - Sqr(s.x));
  for b := 0 to batch_count - 1 do
    for i := 0 to N-1 do
    begin
        h_x[b*stride+i].x := i+b; h_x[b*stride+i].y := i-b;
        h_y[b*stride+i].x := i+b+1; h_y[b*stride+i].y := i-b+1;
        h_x_orig[b*stride+i] := h_x[b*stride+i];
        h_y_orig[b*stride+i] := h_y[b*stride+i];
        // x' = c*x + s*y
        h_x_expected[b*stride+i].x := c*h_x_orig[b*stride+i].x + (s.x*h_y_orig[b*stride+i].x - s.y*h_y_orig[b*stride+i].y);
        h_x_expected[b*stride+i].y := c*h_x_orig[b*stride+i].y + (s.x*h_y_orig[b*stride+i].y + s.y*h_y_orig[b*stride+i].x);
        // y' = -conj(s)*x + c*y
        h_y_expected[b*stride+i].x := (-s.x*h_x_orig[b*stride+i].x - s.y*h_x_orig[b*stride+i].y) + c*h_y_orig[b*stride+i].x;
        h_y_expected[b*stride+i].y := (-s.x*h_x_orig[b*stride+i].y + s.y*h_x_orig[b*stride+i].x) + c*h_y_orig[b*stride+i].y;
    end;

  // 2. Allocate and Copy
  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_float_complex));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_crot_strided_batched_64(handle, N, d_x, incx, stride, d_y, incy, stride, @c, @s, batch_count);
  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_crot_strided_batched_64 is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_crot_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_x[0], d_x, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[0], d_y, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    for b := 0 to batch_count - 1 do
      for i := 0 to N-1 do
      begin
        CheckEquals(h_x_expected[b*stride+i].x, h_x[b*stride+i].x, 1e-5, 'CROT_Strided_64 x.real mismatch');
        CheckEquals(h_x_expected[b*stride+i].y, h_x[b*stride+i].y, 1e-5, 'CROT_Strided_64 x.imag mismatch');
        CheckEquals(h_y_expected[b*stride+i].x, h_y[b*stride+i].x, 1e-5, 'CROT_Strided_64 y.real mismatch');
        CheckEquals(h_y_expected[b*stride+i].y, h_y[b*stride+i].y, 1e-5, 'CROT_Strided_64 y.imag mismatch');
      end;
  end;

  // 4. Cleanup
  hipFree(d_x); hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched_64.Test_CSROT_Strided_Batched_64;
const
  N: Int64 = 10;
  batch_count: Int64 = 3;
  incx: Int64 = 1;
  incy: Int64 = 1;
  stride = 12;
    total_size = 34; // stride * (batch_count - 1) + N;
type
  TComplexVector = array[0..33] of rocblas_float_complex;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TComplexVector;
  d_x, d_y: hipDeviceptr_t;
  c, s: Single;
  b: Int64;
  i: Integer;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize
  c := 0.8; s := 0.6;
  for b := 0 to batch_count - 1 do
    for i := 0 to N-1 do
    begin
      h_x[b*stride+i].x := b + 1 + i*0.1; h_x[b*stride+i].y := b + 1 - i*0.1;
      h_y[b*stride+i].x := b + 2 + i*0.2; h_y[b*stride+i].y := b + 2 - i*0.2;
      h_x_orig[b*stride+i] := h_x[b*stride+i];
      h_y_orig[b*stride+i] := h_y[b*stride+i];
      h_x_expected[b*stride+i].x := c * h_x_orig[b*stride+i].x + s * h_y_orig[b*stride+i].x;
      h_x_expected[b*stride+i].y := c * h_x_orig[b*stride+i].y + s * h_y_orig[b*stride+i].y;
      h_y_expected[b*stride+i].x := -s * h_x_orig[b*stride+i].x + c * h_y_orig[b*stride+i].x;
      h_y_expected[b*stride+i].y := -s * h_x_orig[b*stride+i].y + c * h_y_orig[b*stride+i].y;
    end;

  // 2. Allocate and Copy
  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_float_complex));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_csrot_strided_batched_64(handle, N, d_x, incx, stride, d_y, incy, stride, @c, @s, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_csrot_strided_batched_64 is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_csrot_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_x[0], d_x, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[0], d_y, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    for b := 0 to batch_count - 1 do
      for i := 0 to N - 1 do
      begin
        CheckEquals(h_x_expected[b*stride+i].x, h_x[b*stride+i].x, 1e-5, 'CSROT_Strided_64 x.real mismatch');
        CheckEquals(h_x_expected[b*stride+i].y, h_x[b*stride+i].y, 1e-5, 'CSROT_Strided_64 x.imag mismatch');
        CheckEquals(h_y_expected[b*stride+i].x, h_y[b*stride+i].x, 1e-5, 'CSROT_Strided_64 y.real mismatch');
        CheckEquals(h_y_expected[b*stride+i].y, h_y[b*stride+i].y, 1e-5, 'CSROT_Strided_64 y.imag mismatch');
      end;
  end;

  // 4. Cleanup
  hipFree(d_x); hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched_64.Test_SROTG_Strided_Batched_64;
const
  batch_count: Int64 = 3;
  stride = 1;
type
  TSingleVector = array[0..2] of Single;
var
  h_a, h_b, h_c, h_s, h_a_orig, h_b_orig: TSingleVector;
  d_a, d_b, d_c, d_s: hipDeviceptr_t;
  b: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    h_a[b] := 3.0 + b; h_b[b] := 4.0 - b;
    h_a_orig[b] := h_a[b]; h_b_orig[b] := h_b[b];
  end;

  // 2. Allocate and Copy
  hipMalloc(d_a, batch_count * SizeOf(Single)); hipMalloc(d_b, batch_count * SizeOf(Single));
  hipMalloc(d_c, batch_count * SizeOf(Single)); hipMalloc(d_s, batch_count * SizeOf(Single));
  hipMemcpy(d_a, @h_a[0], batch_count * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_b, @h_b[0], batch_count * SizeOf(Single), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call rocBLAS
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_srotg_strided_batched_64(handle, d_a, stride, d_b, stride, d_c, stride, d_s, stride, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_srotg_strided_batched_64 is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_srotg_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    // 4. Copy results back
    hipMemcpy(@h_a[0], d_a, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_c[0], d_c, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_s[0], d_s, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

    // 5. Verify
    for b := 0 to batch_count - 1 do
    begin
      CheckEquals(1.0, Sqr(h_c[b]) + Sqr(h_s[b]), 1e-6, Format('SROTG_Strided_64 orthogonality failed batch %d', [b]));
      CheckEquals(h_a[b], h_c[b] * h_a_orig[b] + h_s[b] * h_b_orig[b], 1e-6, Format('SROTG_Strided_64 r value mismatch batch %d', [b]));
    end;
  end;

  // 6. Cleanup
  hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_s);
end;

procedure TTestRocBLAS_Level1_Strided_Batched_64.Test_DROTG_Strided_Batched_64;
const
  batch_count: Int64 = 3;
  stride = 1;
type
  TDoubleVector = array[0..2] of Double;
var
  h_a, h_b, h_c, h_s, h_a_orig, h_b_orig: TDoubleVector;
  d_a, d_b, d_c, d_s: hipDeviceptr_t;
  b: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    h_a[b] := 3.5 + b; h_b[b] := 4.5 - b;
    h_a_orig[b] := h_a[b]; h_b_orig[b] := h_b[b];
  end;

  // 2. Allocate and Copy
  hipMalloc(d_a, batch_count * SizeOf(Double)); hipMalloc(d_b, batch_count * SizeOf(Double));
  hipMalloc(d_c, batch_count * SizeOf(Double)); hipMalloc(d_s, batch_count * SizeOf(Double));
  hipMemcpy(d_a, @h_a[0], batch_count * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_b, @h_b[0], batch_count * SizeOf(Double), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_drotg_strided_batched_64(handle, d_a, stride, d_b, stride, d_c, stride, d_s, stride, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_drotg_strided_batched_64 is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_drotg_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_a[0], d_a, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_c[0], d_c, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_s[0], d_s, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

    // 4. Verify
    for b := 0 to batch_count - 1 do
    begin
      CheckEquals(1.0, Sqr(h_c[b]) + Sqr(h_s[b]), 1e-12, Format('DROTG_Strided_64 orthogonality failed batch %d', [b]));
      CheckEquals(h_a[b], h_c[b] * h_a_orig[b] + h_s[b] * h_b_orig[b], 1e-12, Format('DROTG_Strided_64 r value mismatch batch %d', [b]));
    end;
  end;

  // 5. Cleanup
  hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_s);
end;

procedure TTestRocBLAS_Level1_Strided_Batched_64.Test_CROTG_Strided_Batched_64;
const
  batch_count: Int64 = 3;
  stride = 1;
type
  TComplexVector = array[0..2] of rocblas_float_complex;
  TRealVector = array[0..2] of Single;
var
  h_a, h_b, h_s: TComplexVector;
  h_c: TRealVector;
  d_a, d_b, d_c, d_s: hipDeviceptr_t;
  b: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize
  for b := 0 to batch_count - 1 do
  begin
    h_a[b].x := 3.0 + b; h_a[b].y := 1.0 + b;
    h_b[b].x := 4.0 - b; h_b[b].y := 2.0 - b;
  end;

  // 2. Allocate and Copy
  hipMalloc(d_a, batch_count * SizeOf(rocblas_float_complex));
  hipMalloc(d_b, batch_count * SizeOf(rocblas_float_complex));
  hipMalloc(d_c, batch_count * SizeOf(Single));
  hipMalloc(d_s, batch_count * SizeOf(rocblas_float_complex));
  hipMemcpy(d_a, @h_a[0], batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_b, @h_b[0], batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_crotg_strided_batched_64(handle, d_a, stride, d_b, stride, d_c, stride, d_s, stride, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_crotg_strided_batched_64 is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_crotg_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_c[0], d_c, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_s[0], d_s, batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

    // 4. Verify
    for b := 0 to batch_count - 1 do
    begin
      CheckEquals(1.0, Sqr(h_c[b]) + Sqr(h_s[b].x) + Sqr(h_s[b].y), 1e-5, 'CROTG_Strided_64 orthogonality failed');
    end;
  end;

  // 5. Cleanup
  hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_s);
end;

procedure TTestRocBLAS_Level1_Strided_Batched_64.Test_ZROTG_Strided_Batched_64;
const
  batch_count: Int64 = 3;
  stride = 1;
type
  TComplexVector = array[0..2] of rocblas_double_complex;
  TRealVector = array[0..2] of Double;
var
  h_a, h_b, h_s: TComplexVector;
  h_c: TRealVector;
  d_a, d_b, d_c, d_s: hipDeviceptr_t;
  b: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize
  for b := 0 to batch_count - 1 do
  begin
    h_a[b].x := 3.5 + b; h_a[b].y := 1.5 + b;
    h_b[b].x := 4.5 - b; h_b[b].y := 2.5 - b;
  end;

  // 2. Allocate and Copy
  hipMalloc(d_a, batch_count * SizeOf(rocblas_double_complex));
  hipMalloc(d_b, batch_count * SizeOf(rocblas_double_complex));
  hipMalloc(d_c, batch_count * SizeOf(Double));
  hipMalloc(d_s, batch_count * SizeOf(rocblas_double_complex));
  hipMemcpy(d_a, @h_a[0], batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_b, @h_b[0], batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_zrotg_strided_batched_64(handle, d_a, stride, d_b, stride, d_c, stride, d_s, stride, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_zrotg_strided_batched_64 is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_zrotg_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_c[0], d_c, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_s[0], d_s, batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

    // 4. Verify
    for b := 0 to batch_count - 1 do
    begin
      CheckEquals(1.0, Sqr(h_c[b]) + Sqr(h_s[b].x) + Sqr(h_s[b].y), 1e-12, 'ZROTG_Strided_64 orthogonality failed');
    end;
  end;

  // 5. Cleanup
  hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_s);
end;

procedure TTestRocBLAS_Level1_Strided_Batched_64.Test_SROTM_Strided_Batched_64;
const
  N: Int64 = 10;
  batch_count: Int64 = 3;
  incx: Int64 = 1;
  incy: Int64 = 1;
  stride_x = 16;
  stride_y = 16;
  stride_param = 5;
  size_x = 42;
  size_param = 15;
type
  TSingleVector = array[0..41] of Single;
  TParamVector = array[0..14] of Single;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TSingleVector;
  h_param: TParamVector;
  d_x, d_y, d_param: hipDeviceptr_t;
  b: Int64;
  i: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    h_param[b*stride_param+0] := -1.0; // Flag for full matrix H
    h_param[b*stride_param+1] := 1.0 + b*0.1; // H11
    h_param[b*stride_param+2] := 2.0 + b*0.1; // H21
    h_param[b*stride_param+3] := 3.0 + b*0.1; // H12
    h_param[b*stride_param+4] := 4.0 + b*0.1; // H22

    for i := 0 to N - 1 do
    begin
      h_x[b*stride_x+i] := 1.0 + i; h_y[b*stride_y+i] := 2.0 - i;
      h_x_orig[b*stride_x+i] := h_x[b*stride_x+i];
      h_y_orig[b*stride_y+i] := h_y[b*stride_y+i];
      h_x_expected[b*stride_x+i] := h_param[b*stride_param+1] * h_x_orig[b*stride_x+i] + h_param[b*stride_param+3] * h_y_orig[b*stride_y+i];
      h_y_expected[b*stride_y+i] := h_param[b*stride_param+2] * h_x_orig[b*stride_x+i] + h_param[b*stride_param+4] * h_y_orig[b*stride_y+i];
    end;
  end;

  // 2. Allocate and Copy
  hipMalloc(d_x, size_x * SizeOf(Single)); hipMalloc(d_y, size_x * SizeOf(Single));
  hipMalloc(d_param, size_param * SizeOf(Single));
  hipMemcpy(d_x, @h_x[0], size_x * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], size_x * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_param, @h_param[0], size_param * SizeOf(Single), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_srotm_strided_batched_64(handle, N, d_x, incx, stride_x, d_y, incy, stride_y, d_param, stride_param, batch_count);
  begin
    AssertEquals('rocblas_srotm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_x[0], d_x, size_x * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[0], d_y, size_x * SizeOf(Single), hipMemcpyDeviceToHost);
    for b := 0 to batch_count - 1 do
      for i := 0 to N - 1 do
      begin
        CheckEquals(h_x_expected[b*stride_x+i], h_x[b*stride_x+i], 1e-5, 'SROTM_Strided_64 x mismatch');
        CheckEquals(h_y_expected[b*stride_y+i], h_y[b*stride_y+i], 1e-5, 'SROTM_Strided_64 y mismatch');
      end;
  end;

  // 4. Cleanup
  hipFree(d_x); hipFree(d_y); hipFree(d_param);
end;

procedure TTestRocBLAS_Level1_Strided_Batched_64.Test_DROTM_Strided_Batched_64;
const
  N: Int64 = 10;
  batch_count: Int64 = 3;
  incx: Int64 = 1;
  incy: Int64 = 1;
  stride_x = 16;
  stride_y = 16;
  stride_param = 5;
  size_x = 42;
  size_param = 15;
type
  TDoubleVector = array[0..41] of Double;
  TParamVector = array[0..14] of Double;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TDoubleVector;
  h_param: TParamVector;
  d_x, d_y, d_param: hipDeviceptr_t;
  b: Int64;
  i: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    h_param[b*stride_param+0] := -1.0;
    h_param[b*stride_param+1] := 1.5 + b*0.15; h_param[b*stride_param+2] := 2.5 + b*0.15;
    h_param[b*stride_param+3] := 3.5 + b*0.15; h_param[b*stride_param+4] := 4.5 + b*0.15;
    for i := 0 to N - 1 do
    begin
      h_x[b*stride_x+i] := 1.5 + i; h_y[b*stride_y+i] := 2.5 - i;
      h_x_orig[b*stride_x+i] := h_x[b*stride_x+i];
      h_y_orig[b*stride_y+i] := h_y[b*stride_y+i];
      h_x_expected[b*stride_x+i] := h_param[b*stride_param+1] * h_x_orig[b*stride_x+i] + h_param[b*stride_param+3] * h_y_orig[b*stride_y+i];
      h_y_expected[b*stride_y+i] := h_param[b*stride_param+2] * h_x_orig[b*stride_x+i] + h_param[b*stride_param+4] * h_y_orig[b*stride_y+i];
    end;
  end;

  // 2. Allocate and Copy
  hipMalloc(d_x, size_x * SizeOf(Double)); hipMalloc(d_y, size_x * SizeOf(Double));
  hipMalloc(d_param, size_param * SizeOf(Double));
  hipMemcpy(d_x, @h_x[0], size_x * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], size_x * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_param, @h_param[0], size_param * SizeOf(Double), hipMemcpyHostToDevice);

  // 3. Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_drotm_strided_batched_64(handle, N, d_x, incx, stride_x, d_y, incy, stride_y, d_param, stride_param, batch_count);
  begin
    AssertEquals('rocblas_drotm_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_x[0], d_x, size_x * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[0], d_y, size_x * SizeOf(Double), hipMemcpyDeviceToHost);
    for b := 0 to batch_count - 1 do
      for i := 0 to N - 1 do
      begin
        CheckEquals(h_x_expected[b*stride_x+i], h_x[b*stride_x+i], 1e-12, 'DROTM_Strided_64 x mismatch');
        CheckEquals(h_y_expected[b*stride_y+i], h_y[b*stride_y+i], 1e-12, 'DROTM_Strided_64 y mismatch');
      end;
  end;

  // 4. Cleanup
  hipFree(d_x); hipFree(d_y); hipFree(d_param);
end;

procedure TTestRocBLAS_Level1_Strided_Batched_64.Test_SROTMG_Strided_Batched_64;
const
  batch_count: Int64 = 3;
  stride = 1;
  param_stride = 5;
  total_scalars = 3;
  total_params = 15;
type
  TSingleVector = array[0..2] of Single;
  TParamVector = array[0..14] of Single;
var
  h_d1, h_d2, h_x1, h_y1: TSingleVector;
  h_param: TParamVector;
  d_d1, d_d2, d_x1, d_y1, d_param: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    h_d1[b * stride] := 4.0 + b; h_d2[b * stride] := 9.0 + b;
    h_x1[b * stride] := 2.0 + b; h_y1[b * stride] := 3.0 + b;
  end;

  // 2. Allocate and Copy
  hipMalloc(d_d1, total_scalars * SizeOf(Single)); hipMalloc(d_d2, total_scalars * SizeOf(Single));
  hipMalloc(d_x1, total_scalars * SizeOf(Single)); hipMalloc(d_y1, total_scalars * SizeOf(Single));
  hipMalloc(d_param, total_params * SizeOf(Single));
  hipMemcpy(d_d1, @h_d1[0], total_scalars * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_d2, @h_d2[0], total_scalars * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_x1, @h_x1[0], total_scalars * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_y1, @h_y1[0], total_scalars * SizeOf(Single), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_srotmg_strided_batched_64(handle, d_d1, stride, d_d2, stride, d_x1, stride, d_y1, stride, d_param, param_stride, batch_count);
  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_srotmg_strided_batched_64 is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_srotmg_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_d1[0], d_d1, total_scalars * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_d2[0], d_d2, total_scalars * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_x1[0], d_x1, total_scalars * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_param[0], d_param, total_params * SizeOf(Single), hipMemcpyDeviceToHost);

    // 4. Verify (Sanity check)
    for b := 0 to batch_count - 1 do
    begin
      AssertTrue(Format('SROTMG_Strided_64 d1 is not finite batch %d', [b]), not(IsInfinite(h_d1[b*stride]) or IsNan(h_d1[b*stride])));
      AssertTrue(Format('SROTMG_Strided_64 d2 is not finite batch %d', [b]), not(IsInfinite(h_d2[b*stride]) or IsNan(h_d2[b*stride])));
      AssertTrue(Format('SROTMG_Strided_64 x1 is not finite batch %d', [b]), not(IsInfinite(h_x1[b*stride]) or IsNan(h_x1[b*stride])));
      for i := 0 to 4 do
        AssertTrue(Format('SROTMG_Strided_64 param[%d] is not finite batch %d', [i, b]), not(IsInfinite(h_param[b*param_stride+i]) or IsNan(h_param[b*param_stride+i])));
    end;
  end;

  // 5. Cleanup
  hipFree(d_d1); hipFree(d_d2); hipFree(d_x1); hipFree(d_y1); hipFree(d_param);
end;

procedure TTestRocBLAS_Level1_Strided_Batched_64.Test_DROTMG_Strided_Batched_64;
const
  batch_count: Int64 = 3;
  stride = 1;
  param_stride = 5;
  total_scalars = 3;
  total_params = 15;
type
  TDoubleVector = array[0..2] of Double;
  TParamVector = array[0..14] of Double;
var
  h_d1, h_d2, h_x1, h_y1: TDoubleVector;
  h_param: TParamVector;
  d_d1, d_d2, d_x1, d_y1, d_param: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    h_d1[b * stride] := 4.5 + b; h_d2[b * stride] := 9.5 + b;
    h_x1[b * stride] := 2.5 + b; h_y1[b * stride] := 3.5 + b;
  end;

  // 2. Allocate and Copy
  hipMalloc(d_d1, total_scalars * SizeOf(Double)); hipMalloc(d_d2, total_scalars * SizeOf(Double));
  hipMalloc(d_x1, total_scalars * SizeOf(Double)); hipMalloc(d_y1, total_scalars * SizeOf(Double));
  hipMalloc(d_param, total_params * SizeOf(Double));
  hipMemcpy(d_d1, @h_d1[0], total_scalars * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_d2, @h_d2[0], total_scalars * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_x1, @h_x1[0], total_scalars * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_y1, @h_y1[0], total_scalars * SizeOf(Double), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_drotmg_strided_batched_64(handle, d_d1, stride, d_d2, stride, d_x1, stride, d_y1, stride, d_param, param_stride, batch_count);
  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_drotmg_strided_batched_64 is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_drotmg_strided_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_d1[0], d_d1, total_scalars * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_d2[0], d_d2, total_scalars * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_x1[0], d_x1, total_scalars * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_param[0], d_param, total_params * SizeOf(Double), hipMemcpyDeviceToHost);

    // 4. Verify
    for b := 0 to batch_count - 1 do
    begin
      AssertTrue(Format('DROTMG_Strided_64 d1 is not finite batch %d', [b]), not(IsInfinite(h_d1[b*stride]) or IsNan(h_d1[b*stride])));
      AssertTrue(Format('DROTMG_Strided_64 d2 is not finite batch %d', [b]), not(IsInfinite(h_d2[b*stride]) or IsNan(h_d2[b*stride])));
      AssertTrue(Format('DROTMG_Strided_64 x1 is not finite batch %d', [b]), not(IsInfinite(h_x1[b*stride]) or IsNan(h_x1[b*stride])));
      for i := 0 to 4 do
        AssertTrue(Format('DROTMG_Strided_64 param[%d] is not finite batch %d', [i, b]), not(IsInfinite(h_param[b*param_stride+i]) or IsNan(h_param[b*param_stride+i])));
    end;
  end;

  // 5. Cleanup
  hipFree(d_d1); hipFree(d_d2); hipFree(d_x1); hipFree(d_y1); hipFree(d_param);
end;

initialization
  RegisterTest(TTestRocBLAS_Level1_Strided_Batched_64);

end.
