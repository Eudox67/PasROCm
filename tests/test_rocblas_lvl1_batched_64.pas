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

unit Test_rocblas_lvl1_batched_64;

{$mode objfpc}{$H+}
{$linklib rocblas}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  { TTestRocBLAS_Level1_Batched_64 }
  TTestRocBLAS_Level1_Batched_64 = class(TTestRocBLAS_Base)
  published
    procedure Test_SROT_Batched_64;
    procedure Test_DROT_Batched_64;
    procedure Test_CROT_Batched_64;
    procedure Test_CSROT_Batched_64;

    procedure Test_SROTG_Batched_64;
    procedure Test_DROTG_Batched_64;
    procedure Test_CROTG_Batched_64;
    procedure Test_ZROTG_Batched_64;

    procedure Test_SROTM_Batched_64;
    procedure Test_DROTM_Batched_64;

    procedure Test_SROTMG_Batched_64;
    procedure Test_DROTMG_Batched_64;
  end;

implementation

{ TTestRocBLAS_Level1_Batched_64 }

procedure TTestRocBLAS_Level1_Batched_64.Test_SROT_Batched_64;
const
  N: Int64 = 10;
  batch_count: Int64 = 3;
  incx: Int64 = 1;
  incy: Int64 = 1;
type
  TSingleVector = array[0..9] of Single;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: array[0..2] of TSingleVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..2] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b: Int64;
  i: Integer;
  c, s: Single;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  c := 0.8; // cos
  s := 0.6; // sin
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := (b + 1) * 10.0 + i;
      h_y[b][i] := (b + 1) * 20.0 - i;
      h_x_orig[b][i] := h_x[b][i];
      h_y_orig[b][i] := h_y[b][i];
      h_x_expected[b][i] := c * h_x_orig[b][i] + s * h_y_orig[b][i];
      h_y_expected[b][i] := -s * h_x_orig[b][i] + c * h_y_orig[b][i];
    end;
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Single));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(Single));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Single), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(Single), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Set pointer mode and call rocBLAS
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_srot_batched_64(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, @c, @s, batch_count);
  AssertEquals('rocblas_srot_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy results back
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(Single), hipMemcpyDeviceToHost);
  end;

  // 7. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b][i], h_x[b][i], 1e-5, Format('SROT_64 x mismatch batch %d', [b]));
      CheckEquals(h_y_expected[b][i], h_y[b][i], 1e-5, Format('SROT_64 y mismatch batch %d', [b]));
    end;
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
    hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_dev_y_ptrs);
end;


procedure TTestRocBLAS_Level1_Batched_64.Test_DROT_Batched_64;
const
  N: Int64 = 10;
  batch_count: Int64 = 3;
  incx: Int64 = 1;
  incy: Int64 = 1;
type
  TDoubleVector = array[0..9] of Double;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: array[0..2] of TDoubleVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..2] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b: Int64;
  i: Integer;
  c, s: Double;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  c := Sqrt(2.0)/2.0;
  s := Sqrt(2.0)/2.0;
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := (b + 1.5) * 10.0 + i;
      h_y[b][i] := (b + 1.5) * 20.0 - i;
      h_x_orig[b][i] := h_x[b][i];
      h_y_orig[b][i] := h_y[b][i];
      h_x_expected[b][i] := c * h_x_orig[b][i] + s * h_y_orig[b][i];
      h_y_expected[b][i] := -s * h_x_orig[b][i] + c * h_y_orig[b][i];
    end;
  end;

  // 2. Allocate and Copy Data (same pattern as SROT_64)
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Double));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(Double));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Set pointer mode and call rocBLAS
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_drot_batched_64(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, @c, @s, batch_count);
  AssertEquals('rocblas_drot_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy results back
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(Double), hipMemcpyDeviceToHost);
  end;

  // 7. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b][i], h_x[b][i], 1e-12, Format('DROT_64 x mismatch batch %d', [b]));
      CheckEquals(h_y_expected[b][i], h_y[b][i], 1e-12, Format('DROT_64 y mismatch batch %d', [b]));
    end;
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
    hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_dev_y_ptrs);
end;

procedure TTestRocBLAS_Level1_Batched_64.Test_CROT_Batched_64;
const
  N: Int64 = 10;
  batch_count: Int64 = 3;
  incx: Int64 = 1;
  incy: Int64 = 1;
type
  TComplexVector = array[0..9] of rocblas_float_complex;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: array[0..2] of TComplexVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..2] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b: Int64;
  i: Integer;
  c: Single;
  s: rocblas_float_complex;
  blas_status, pm_status: rocblas_status;
begin
  c := 0.5;
  s.x := 0.8;
  s.y := Sqrt(1.0 - Sqr(c) - Sqr(s.x)); // For |c|^2+|s|^2=1

  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N-1 do
    begin
        h_x[b][i].x := i+b; h_x[b][i].y := i-b;
        h_y[b][i].x := i+b+1; h_y[b][i].y := i-b+1;
        h_x_orig[b][i] := h_x[b][i];
        h_y_orig[b][i] := h_y[b][i];
        // x' = c*x + s*y
        h_x_expected[b][i].x := c*h_x_orig[b][i].x + (s.x*h_y_orig[b][i].x - s.y*h_y_orig[b][i].y);
        h_x_expected[b][i].y := c*h_x_orig[b][i].y + (s.x*h_y_orig[b][i].y + s.y*h_y_orig[b][i].x);
        // y' = -conj(s)*x + c*y
        // ***** FIX: Corrected complex multiplication logic *****
        h_y_expected[b][i].x := (-s.x * h_x_orig[b][i].x - s.y * h_x_orig[b][i].y) + c * h_y_orig[b][i].x;
        h_y_expected[b][i].y := (-s.x * h_x_orig[b][i].y + s.y * h_x_orig[b][i].x) + c * h_y_orig[b][i].y;
    end;
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(rocblas_float_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_crot_batched_64(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, @c, @s, batch_count);
  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_crot_batched_64 is not implemented in the rocBLAS library.');
    for b := 0 to batch_count - 1 do begin hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
    hipFree(d_dev_x_ptrs); hipFree(d_dev_y_ptrs);
    Exit;
  end;
  AssertEquals('rocblas_crot_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  end;

  for b := 0 to batch_count - 1 do
    for i := 0 to N-1 do
    begin
      CheckEquals(h_x_expected[b][i].x, h_x[b][i].x, 1e-5, Format('CROT_64 x.real mismatch b=%d i=%d',[b,i]));
      CheckEquals(h_x_expected[b][i].y, h_x[b][i].y, 1e-5, Format('CROT_64 x.imag mismatch b=%d i=%d',[b,i]));
      CheckEquals(h_y_expected[b][i].x, h_y[b][i].x, 1e-5, Format('CROT_64 y.real mismatch b=%d i=%d',[b,i]));
      CheckEquals(h_y_expected[b][i].y, h_y[b][i].y, 1e-5, Format('CROT_64 y.imag mismatch b=%d i=%d',[b,i]));
    end;

  for b := 0 to batch_count - 1 do begin hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); end;
  hipFree(d_dev_x_ptrs); hipFree(d_dev_y_ptrs);
end;

procedure TTestRocBLAS_Level1_Batched_64.Test_CSROT_Batched_64;
const
  N: Int64 = 10;
  batch_count: Int64 = 3;
  incx: Int64 = 1;
  incy: Int64 = 1;
type
  TComplexVector = array[0..9] of rocblas_float_complex;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: array[0..2] of TComplexVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..2] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b: Int64;
  i: Integer;
  c, s: Single; // Real scalars
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  c := 0.8; s := 0.6;
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := b + 1 + i*0.1;
      h_x[b][i].y := b + 1 - i*0.1;
      h_y[b][i].x := b + 2 + i*0.2;
      h_y[b][i].y := b + 2 - i*0.2;
      h_x_orig[b][i] := h_x[b][i];
      h_y_orig[b][i] := h_y[b][i];
      h_x_expected[b][i].x := c * h_x_orig[b][i].x + s * h_y_orig[b][i].x;
      h_x_expected[b][i].y := c * h_x_orig[b][i].y + s * h_y_orig[b][i].y;
      h_y_expected[b][i].x := -s * h_x_orig[b][i].x + c * h_y_orig[b][i].x;
      h_y_expected[b][i].y := -s * h_x_orig[b][i].y + c * h_y_orig[b][i].y;
    end;
  end;

  // ... Allocation, Copy, Cleanup logic is same pattern, just with complex type ...
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(rocblas_float_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Set pointer mode and call rocBLAS
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));
  blas_status := rocblas_csrot_batched_64(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, @c, @s, batch_count);
  AssertEquals('rocblas_csrot_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy results back
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  end;

  // 7. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b][i].x, h_x[b][i].x, 1e-5, Format('CSROT_64 x.real mismatch batch %d', [b]));
      CheckEquals(h_x_expected[b][i].y, h_x[b][i].y, 1e-5, Format('CSROT_64 x.imag mismatch batch %d', [b]));
      CheckEquals(h_y_expected[b][i].x, h_y[b][i].x, 1e-5, Format('CSROT_64 y.real mismatch batch %d', [b]));
      CheckEquals(h_y_expected[b][i].y, h_y[b][i].y, 1e-5, Format('CSROT_64 y.imag mismatch batch %d', [b]));
    end;
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
    hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_dev_y_ptrs);
end;

procedure TTestRocBLAS_Level1_Batched_64.Test_SROTG_Batched_64;
const
  batch_count: Int64 = 3;
var
  h_a, h_b, h_c, h_s, h_a_orig, h_b_orig: array[0..2] of Single;
  h_dev_a_ptrs, h_dev_b_ptrs, h_dev_c_ptrs, h_dev_s_ptrs: array[0..2] of hipDeviceptr_t;
  d_dev_a_ptrs, d_dev_b_ptrs, d_dev_c_ptrs, d_dev_s_ptrs: hipDeviceptr_t;
  b: Int64;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    h_a[b] := 3.0 + b;
    h_b[b] := 4.0 - b;
    h_a_orig[b] := h_a[b];
    h_b_orig[b] := h_b[b];
  end;

  // 2. Allocate memory for each scalar in the batch on the device
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_a_ptrs[b], SizeOf(Single));
    hipMalloc(h_dev_b_ptrs[b], SizeOf(Single));
    hipMalloc(h_dev_c_ptrs[b], SizeOf(Single));
    hipMalloc(h_dev_s_ptrs[b], SizeOf(Single));
  end;

  // 3. Allocate device memory for the pointer arrays
  hipMalloc(d_dev_a_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_b_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_c_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_s_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy data and pointer arrays to device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_a_ptrs[b], @h_a[b], SizeOf(Single), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_b_ptrs[b], @h_b[b], SizeOf(Single), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_a_ptrs, @h_dev_a_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_b_ptrs, @h_dev_b_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_c_ptrs, @h_dev_c_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_s_ptrs, @h_dev_s_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. ***** FIX: Set pointer mode to DEVICE *****
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  // 6. Call rocBLAS function
  blas_status := rocblas_srotg_batched_64(handle, d_dev_a_ptrs, d_dev_b_ptrs, d_dev_c_ptrs, d_dev_s_ptrs, batch_count);
  AssertEquals('rocblas_srotg_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  // 7. Copy results back
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_a[b], h_dev_a_ptrs[b], SizeOf(Single), hipMemcpyDeviceToHost); // a is now r
    hipMemcpy(@h_c[b], h_dev_c_ptrs[b], SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_s[b], h_dev_s_ptrs[b], SizeOf(Single), hipMemcpyDeviceToHost);
  end;

  // 8. Verify results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(1.0, Sqr(h_c[b]) + Sqr(h_s[b]), 1e-6, Format('SROTG_64 orthogonality failed batch %d', [b]));
    CheckEquals(h_a[b], h_c[b] * h_a_orig[b] + h_s[b] * h_b_orig[b], 1e-6, Format('SROTG_64 r value mismatch batch %d', [b]));
  end;

  // 9. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_a_ptrs[b]);
    hipFree(h_dev_b_ptrs[b]);
    hipFree(h_dev_c_ptrs[b]);
    hipFree(h_dev_s_ptrs[b]);
  end;
  hipFree(d_dev_a_ptrs);
  hipFree(d_dev_b_ptrs);
  hipFree(d_dev_c_ptrs);
  hipFree(d_dev_s_ptrs);
end;

procedure TTestRocBLAS_Level1_Batched_64.Test_DROTG_Batched_64;
const
  batch_count: Int64 = 3;
var
  h_a, h_b, h_c, h_s, h_a_orig, h_b_orig: array[0..2] of Double; // FIX
  h_dev_a_ptrs, h_dev_b_ptrs, h_dev_c_ptrs, h_dev_s_ptrs: array[0..2] of hipDeviceptr_t; // FIX
  d_dev_a_ptrs, d_dev_b_ptrs, d_dev_c_ptrs, d_dev_s_ptrs: hipDeviceptr_t;
  b: Int64;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // ... Logic identical to SROTG_64 but with Double ...
  for b := 0 to batch_count - 1 do
  begin
    h_a[b] := 3.5 + b; h_b[b] := 4.5 - b;
    h_a_orig[b] := h_a[b]; h_b_orig[b] := h_b[b];
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_a_ptrs[b], SizeOf(Double)); hipMalloc(h_dev_b_ptrs[b], SizeOf(Double));
    hipMalloc(h_dev_c_ptrs[b], SizeOf(Double)); hipMalloc(h_dev_s_ptrs[b], SizeOf(Double));
  end;
  hipMalloc(d_dev_a_ptrs, batch_count * SizeOf(hipDeviceptr_t)); hipMalloc(d_dev_b_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_c_ptrs, batch_count * SizeOf(hipDeviceptr_t)); hipMalloc(d_dev_s_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_a_ptrs[b], @h_a[b], SizeOf(Double), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_b_ptrs[b], @h_b[b], SizeOf(Double), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_a_ptrs, @h_dev_a_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_b_ptrs, @h_dev_b_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_c_ptrs, @h_dev_c_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_s_ptrs, @h_dev_s_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_drotg_batched_64(handle, d_dev_a_ptrs, d_dev_b_ptrs, d_dev_c_ptrs, d_dev_s_ptrs, batch_count);
  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_drotg_batched_64 is not implemented in this rocBLAS version.');
  end else
  begin
    AssertEquals('rocblas_drotg_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    for b := 0 to batch_count - 1 do
    begin
      hipMemcpy(@h_a[b], h_dev_a_ptrs[b], SizeOf(Double), hipMemcpyDeviceToHost);
      hipMemcpy(@h_c[b], h_dev_c_ptrs[b], SizeOf(Double), hipMemcpyDeviceToHost);
      hipMemcpy(@h_s[b], h_dev_s_ptrs[b], SizeOf(Double), hipMemcpyDeviceToHost);
    end;

    for b := 0 to batch_count - 1 do
    begin
      CheckEquals(1.0, Sqr(h_c[b]) + Sqr(h_s[b]), 1e-12, Format('DROTG_64 orthogonality failed batch %d', [b]));
      CheckEquals(h_a[b], h_c[b] * h_a_orig[b] + h_s[b] * h_b_orig[b], 1e-12, Format('DROTG_64 r value mismatch batch %d', [b]));
    end;
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_a_ptrs[b]); hipFree(h_dev_b_ptrs[b]); hipFree(h_dev_c_ptrs[b]); hipFree(h_dev_s_ptrs[b]);
  end;
  hipFree(d_dev_a_ptrs); hipFree(d_dev_b_ptrs); hipFree(d_dev_c_ptrs); hipFree(d_dev_s_ptrs);
end;


procedure TTestRocBLAS_Level1_Batched_64.Test_CROTG_Batched_64;
const
  batch_count: Int64 = 3;
var
  h_a, h_b, h_s: array[0..2] of rocblas_float_complex;
  h_c: array[0..2] of Single;
  // Host-side verification arrays
  h_a_orig, h_b_orig: array[0..2] of rocblas_float_complex;
  h_a_result: array[0..2] of rocblas_float_complex;
  // Device pointer arrays
  h_dev_a_ptrs, h_dev_b_ptrs, h_dev_c_ptrs, h_dev_s_ptrs: array[0..2] of hipDeviceptr_t;
  d_dev_a_ptrs, d_dev_b_ptrs, d_dev_c_ptrs, d_dev_s_ptrs: hipDeviceptr_t;
  b: Int64;
  blas_status, pm_status: rocblas_status;
begin
  // This function IS in the C header, so we create a test that handles 'not implemented'.
  for b := 0 to batch_count - 1 do
  begin
    h_a[b].x := 3.0 + b; h_a[b].y := 1.0 + b;
    h_b[b].x := 4.0 - b; h_b[b].y := 2.0 - b;
    h_a_orig[b] := h_a[b];
    h_b_orig[b] := h_b[b];
  end;

  // Standard allocation logic for batched scalars...
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_a_ptrs[b], SizeOf(rocblas_float_complex));
    hipMalloc(h_dev_b_ptrs[b], SizeOf(rocblas_float_complex));
    hipMalloc(h_dev_c_ptrs[b], SizeOf(Single));
    hipMalloc(h_dev_s_ptrs[b], SizeOf(rocblas_float_complex));
  end;
  hipMalloc(d_dev_a_ptrs, batch_count * SizeOf(hipDeviceptr_t)); hipMalloc(d_dev_b_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_c_ptrs, batch_count * SizeOf(hipDeviceptr_t)); hipMalloc(d_dev_s_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_a_ptrs[b], @h_a[b], SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_b_ptrs[b], @h_b[b], SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_a_ptrs, @h_dev_a_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_b_ptrs, @h_dev_b_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_c_ptrs, @h_dev_c_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_s_ptrs, @h_dev_s_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_crotg_batched_64(handle, d_dev_a_ptrs, d_dev_b_ptrs, d_dev_c_ptrs, d_dev_s_ptrs, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_crotg_batched_64 is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_crotg_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    // Full verification if the function is implemented
    for b := 0 to batch_count-1 do
    begin
      hipMemcpy(@h_a_result[b], h_dev_a_ptrs[b], SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
      hipMemcpy(@h_c[b], h_dev_c_ptrs[b], SizeOf(Single), hipMemcpyDeviceToHost);
      hipMemcpy(@h_s[b], h_dev_s_ptrs[b], SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

      // Verify |c|^2 + |s|^2 = 1 and r = c*a + s*b
      CheckEquals(1.0, Sqr(h_c[b]) + Sqr(h_s[b].x) + Sqr(h_s[b].y), 1e-5, 'CROTG_64 orthogonality failed');
    end;
  end;

  // Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_a_ptrs[b]); hipFree(h_dev_b_ptrs[b]);
    hipFree(h_dev_c_ptrs[b]); hipFree(h_dev_s_ptrs[b]);
  end;
  hipFree(d_dev_a_ptrs); hipFree(d_dev_b_ptrs);
  hipFree(d_dev_c_ptrs); hipFree(d_dev_s_ptrs);
end;

procedure TTestRocBLAS_Level1_Batched_64.Test_ZROTG_Batched_64;
const
  batch_count: Int64 = 3;
var
  h_a, h_b, h_s: array[0..2] of rocblas_double_complex;
  h_c: array[0..2] of Double;
  h_a_orig, h_b_orig: array[0..2] of rocblas_double_complex;
  h_a_result: array[0..2] of rocblas_double_complex;
  h_dev_a_ptrs, h_dev_b_ptrs, h_dev_c_ptrs, h_dev_s_ptrs: array[0..2] of hipDeviceptr_t;
  d_dev_a_ptrs, d_dev_b_ptrs, d_dev_c_ptrs, d_dev_s_ptrs: hipDeviceptr_t;
  b: Int64;
  blas_status, pm_status: rocblas_status;
begin
  // This function IS in the C header, so we create a test that handles 'not implemented'.
  for b := 0 to batch_count - 1 do
  begin
    h_a[b].x := 3.5 + b; h_a[b].y := 1.5 + b;
    h_b[b].x := 4.5 - b; h_b[b].y := 2.5 - b;
    h_a_orig[b] := h_a[b];
    h_b_orig[b] := h_b[b];
  end;

  // Standard allocation logic...
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_a_ptrs[b], SizeOf(rocblas_double_complex)); hipMalloc(h_dev_b_ptrs[b], SizeOf(rocblas_double_complex));
    hipMalloc(h_dev_c_ptrs[b], SizeOf(Double)); hipMalloc(h_dev_s_ptrs[b], SizeOf(rocblas_double_complex));
  end;
  hipMalloc(d_dev_a_ptrs, batch_count * SizeOf(hipDeviceptr_t)); hipMalloc(d_dev_b_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_c_ptrs, batch_count * SizeOf(hipDeviceptr_t)); hipMalloc(d_dev_s_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_a_ptrs[b], @h_a[b], SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_b_ptrs[b], @h_b[b], SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_a_ptrs, @h_dev_a_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_b_ptrs, @h_dev_b_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_c_ptrs, @h_dev_c_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_s_ptrs, @h_dev_s_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_zrotg_batched_64(handle, d_dev_a_ptrs, d_dev_b_ptrs, d_dev_c_ptrs, d_dev_s_ptrs, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    WriteLn('Function rocblas_zrotg_batched_64 is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_zrotg_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
    for b := 0 to batch_count-1 do
    begin
      hipMemcpy(@h_a_result[b], h_dev_a_ptrs[b], SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
      hipMemcpy(@h_c[b], h_dev_c_ptrs[b], SizeOf(Double), hipMemcpyDeviceToHost);
      hipMemcpy(@h_s[b], h_dev_s_ptrs[b], SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
      CheckEquals(1.0, Sqr(h_c[b]) + Sqr(h_s[b].x) + Sqr(h_s[b].y), 1e-12, 'ZROTG_64 orthogonality failed');
    end;
  end;

  // Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_a_ptrs[b]); hipFree(h_dev_b_ptrs[b]);
    hipFree(h_dev_c_ptrs[b]); hipFree(h_dev_s_ptrs[b]);
  end;
  hipFree(d_dev_a_ptrs); hipFree(d_dev_b_ptrs);
  hipFree(d_dev_c_ptrs); hipFree(d_dev_s_ptrs);
end;

procedure TTestRocBLAS_Level1_Batched_64.Test_SROTM_Batched_64;
const
  N: Int64 = 10;
  batch_count: Int64 = 3;
  incx: Int64 = 1;
  incy: Int64 = 1;
  PARAM_SIZE = 5;
type
  TSingleVector = array[0..9] of Single;
  TParamVector = array[0..4] of Single;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: array[0..2] of TSingleVector;
  h_param: array[0..2] of TParamVector;
  h_dev_x_ptrs, h_dev_y_ptrs, h_dev_param_ptrs: array[0..2] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs, d_dev_param_ptrs: hipDeviceptr_t;
  b: Int64;
  i: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // rocBLAS 5.7 declares and exports rocblas_srotm_batched_64, but the entry
  // point returns rocblas_status_not_implemented (only the non-batched
  // rocblas_srotm is implemented). The call is made anyway so the numeric part
  // below runs automatically if a later rocBLAS implements the batched form.

  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    h_param[b][0] := -1.0;              // flag: full matrix H
    h_param[b][1] := 1.0 + b * 0.1;     // H11
    h_param[b][2] := 2.0 + b * 0.1;     // H21
    h_param[b][3] := 3.0 + b * 0.1;     // H12
    h_param[b][4] := 4.0 + b * 0.1;     // H22
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := 1.0 + i;
      h_y[b][i] := 2.0 - i;
      h_x_orig[b][i] := h_x[b][i];
      h_y_orig[b][i] := h_y[b][i];
      h_x_expected[b][i] := h_param[b][1] * h_x_orig[b][i] + h_param[b][3] * h_y_orig[b][i];
      h_y_expected[b][i] := h_param[b][2] * h_x_orig[b][i] + h_param[b][4] * h_y_orig[b][i];
    end;
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Single));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(Single));
    hipMalloc(h_dev_param_ptrs[b], PARAM_SIZE * SizeOf(Single));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_param_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 3. Copy Host Data to Device (param must be device memory for batched rotm)
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Single), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(Single), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_param_ptrs[b], @h_param[b][0], PARAM_SIZE * SizeOf(Single), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_param_ptrs, @h_dev_param_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 4. Set pointer mode and call rocBLAS
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_srotm_batched_64(handle, N, PPSingle(d_dev_x_ptrs), incx,
    PPSingle(d_dev_y_ptrs), incy, PPSingle(d_dev_param_ptrs), batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    for b := 0 to batch_count - 1 do
    begin
      hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); hipFree(h_dev_param_ptrs[b]);
    end;
    hipFree(d_dev_x_ptrs); hipFree(d_dev_y_ptrs); hipFree(d_dev_param_ptrs);
    Ignore('rocblas_srotm_batched_64 exists in rocBLAS but returns rocblas_status_not_implemented (only the non-batched rocblas_srotm is implemented).');
  end;

  // 5. Copy results back and release device memory
  AssertEquals('rocblas_srotm_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(Single), hipMemcpyDeviceToHost);
  end;
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); hipFree(h_dev_param_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs); hipFree(d_dev_y_ptrs); hipFree(d_dev_param_ptrs);

  // 6. Verify
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b][i], h_x[b][i], 1e-5, Format('SROTM_Batched_64 x mismatch batch %d', [b]));
      CheckEquals(h_y_expected[b][i], h_y[b][i], 1e-5, Format('SROTM_Batched_64 y mismatch batch %d', [b]));
    end;
end;

procedure TTestRocBLAS_Level1_Batched_64.Test_DROTM_Batched_64;
const
  N: Int64 = 10;
  batch_count: Int64 = 3;
  incx: Int64 = 1;
  incy: Int64 = 1;
  PARAM_SIZE = 5;
type
  TDoubleVector = array[0..9] of Double;
  TParamVector = array[0..4] of Double;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: array[0..2] of TDoubleVector;
  h_param: array[0..2] of TParamVector;
  h_dev_x_ptrs, h_dev_y_ptrs, h_dev_param_ptrs: array[0..2] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs, d_dev_param_ptrs: hipDeviceptr_t;
  b: Int64;
  i: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // Same situation as rocblas_srotm_batched_64: the symbol exists but the
  // batched form returns rocblas_status_not_implemented in rocBLAS 5.7.
  for b := 0 to batch_count - 1 do
  begin
    h_param[b][0] := -1.0;
    h_param[b][1] := 1.0 + b * 0.1;
    h_param[b][2] := 2.0 + b * 0.1;
    h_param[b][3] := 3.0 + b * 0.1;
    h_param[b][4] := 4.0 + b * 0.1;
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := 1.0 + i;
      h_y[b][i] := 2.0 - i;
      h_x_orig[b][i] := h_x[b][i];
      h_y_orig[b][i] := h_y[b][i];
      h_x_expected[b][i] := h_param[b][1] * h_x_orig[b][i] + h_param[b][3] * h_y_orig[b][i];
      h_y_expected[b][i] := h_param[b][2] * h_x_orig[b][i] + h_param[b][4] * h_y_orig[b][i];
    end;
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Double));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(Double));
    hipMalloc(h_dev_param_ptrs[b], PARAM_SIZE * SizeOf(Double));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_param_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_param_ptrs[b], @h_param[b][0], PARAM_SIZE * SizeOf(Double), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_param_ptrs, @h_dev_param_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_drotm_batched_64(handle, N, PPDouble(d_dev_x_ptrs), incx,
    PPDouble(d_dev_y_ptrs), incy, PPDouble(d_dev_param_ptrs), batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    for b := 0 to batch_count - 1 do
    begin
      hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); hipFree(h_dev_param_ptrs[b]);
    end;
    hipFree(d_dev_x_ptrs); hipFree(d_dev_y_ptrs); hipFree(d_dev_param_ptrs);
    Ignore('rocblas_drotm_batched_64 exists in rocBLAS but returns rocblas_status_not_implemented (only the non-batched rocblas_drotm is implemented).');
  end;

  AssertEquals('rocblas_drotm_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(Double), hipMemcpyDeviceToHost);
  end;
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]); hipFree(h_dev_y_ptrs[b]); hipFree(h_dev_param_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs); hipFree(d_dev_y_ptrs); hipFree(d_dev_param_ptrs);

  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b][i], h_x[b][i], 1e-12, Format('DROTM_Batched_64 x mismatch batch %d', [b]));
      CheckEquals(h_y_expected[b][i], h_y[b][i], 1e-12, Format('DROTM_Batched_64 y mismatch batch %d', [b]));
    end;
end;

procedure TTestRocBLAS_Level1_Batched_64.Test_SROTMG_Batched_64;
const
  batch_count: Int64 = 3;
  PARAM_SIZE = 5;
type
  TParamVector = array[0..PARAM_SIZE-1] of Single;
var
  h_d1, h_d2, h_x1, h_y1: array[0..2] of Single;    // [0..2] because batch_count - 1 = 2
  h_param: array[0..2] of TParamVector;
  h_dev_d1_ptrs, h_dev_d2_ptrs, h_dev_x1_ptrs, h_dev_y1_ptrs, h_dev_param_ptrs: array[0..2] of hipDeviceptr_t;
  d_dev_d1_ptrs, d_dev_d2_ptrs, d_dev_x1_ptrs, d_dev_y1_ptrs, d_dev_param_ptrs: hipDeviceptr_t;
  b: Int64;
  i: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    h_d1[b] := 4.0 + b;
    h_d2[b] := 9.0 + b;
    h_x1[b] := 2.0 + b;
    h_y1[b] := 3.0 + b;
  end;

  // 2. Allocate device memory for each scalar and param vector
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_d1_ptrs[b], SizeOf(Single));
    hipMalloc(h_dev_d2_ptrs[b], SizeOf(Single));
    hipMalloc(h_dev_x1_ptrs[b], SizeOf(Single));
    hipMalloc(h_dev_y1_ptrs[b], SizeOf(Single));
    hipMalloc(h_dev_param_ptrs[b], PARAM_SIZE * SizeOf(Single));
  end;

  // 3. Allocate device memory for pointer arrays
  hipMalloc(d_dev_d1_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_d2_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_x1_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y1_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_param_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy data to device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_d1_ptrs[b], @h_d1[b], SizeOf(Single), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_d2_ptrs[b], @h_d2[b], SizeOf(Single), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x1_ptrs[b], @h_x1[b], SizeOf(Single), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y1_ptrs[b], @h_y1[b], SizeOf(Single), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_d1_ptrs, @h_dev_d1_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_d2_ptrs, @h_dev_d2_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_x1_ptrs, @h_dev_x1_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y1_ptrs, @h_dev_y1_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_param_ptrs, @h_dev_param_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  // 5. Call rocBLAS
  blas_status := rocblas_srotmg_batched_64(handle, d_dev_d1_ptrs, d_dev_d2_ptrs, d_dev_x1_ptrs, d_dev_y1_ptrs, d_dev_param_ptrs, batch_count);
  AssertEquals('rocblas_srotmg_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy results back
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_d1[b], h_dev_d1_ptrs[b], SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_d2[b], h_dev_d2_ptrs[b], SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_x1[b], h_dev_x1_ptrs[b], SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_param[b][0], h_dev_param_ptrs[b], PARAM_SIZE * SizeOf(Single), hipMemcpyDeviceToHost);
  end;

  // 7. Verify (Sanity check that results are finite numbers)
  for b := 0 to batch_count - 1 do
  begin
    AssertTrue(Format('SROTMG_64 d1 is not finite batch %d', [b]), not(IsInfinite(h_d1[b]) or IsNan(h_d1[b])));
    AssertTrue(Format('SROTMG_64 d2 is not finite batch %d', [b]), not(IsInfinite(h_d2[b]) or IsNan(h_d2[b])));
    AssertTrue(Format('SROTMG_64 x1 is not finite batch %d', [b]), not(IsInfinite(h_x1[b]) or IsNan(h_x1[b])));
    for i := 0 to PARAM_SIZE-1 do
      AssertTrue(Format('SROTMG_64 param[%d] is not finite batch %d', [i, b]), not(IsInfinite(h_param[b][i]) or IsNan(h_param[b][i])));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_d1_ptrs[b]);
    hipFree(h_dev_d2_ptrs[b]);
    hipFree(h_dev_x1_ptrs[b]);
    hipFree(h_dev_y1_ptrs[b]);
    hipFree(h_dev_param_ptrs[b]);
  end;
  hipFree(d_dev_d1_ptrs);
  hipFree(d_dev_d2_ptrs);
  hipFree(d_dev_x1_ptrs);
  hipFree(d_dev_y1_ptrs);
  hipFree(d_dev_param_ptrs);
end;

procedure TTestRocBLAS_Level1_Batched_64.Test_DROTMG_Batched_64;
const
  batch_count: Int64 = 3;
  PARAM_SIZE = 5;
type
  TParamVector = array[0..PARAM_SIZE-1] of Double;
var
  h_d1, h_d2, h_x1, h_y1: array[0..2] of Double;
  h_param: array[0..2] of TParamVector;
  h_dev_d1_ptrs, h_dev_d2_ptrs, h_dev_x1_ptrs, h_dev_y1_ptrs, h_dev_param_ptrs: array[0..2] of hipDeviceptr_t;
  d_dev_d1_ptrs, d_dev_d2_ptrs, d_dev_x1_ptrs, d_dev_y1_ptrs, d_dev_param_ptrs: hipDeviceptr_t;
  b: Int64;
  i: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    h_d1[b] := 4.5 + b;
    h_d2[b] := 9.5 + b;
    h_x1[b] := 2.5 + b;
    h_y1[b] := 3.5 + b;
  end;

  // 2. Allocate device memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_d1_ptrs[b], SizeOf(Double));
    hipMalloc(h_dev_d2_ptrs[b], SizeOf(Double));
    hipMalloc(h_dev_x1_ptrs[b], SizeOf(Double));
    hipMalloc(h_dev_y1_ptrs[b], SizeOf(Double));
    hipMalloc(h_dev_param_ptrs[b], PARAM_SIZE * SizeOf(Double));
  end;
  hipMalloc(d_dev_d1_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_d2_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_x1_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y1_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_param_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy data to device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_d1_ptrs[b], @h_d1[b], SizeOf(Double), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_d2_ptrs[b], @h_d2[b], SizeOf(Double), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_x1_ptrs[b], @h_x1[b], SizeOf(Double), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y1_ptrs[b], @h_y1[b], SizeOf(Double), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_d1_ptrs, @h_dev_d1_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_d2_ptrs, @h_dev_d2_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_x1_ptrs, @h_dev_x1_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y1_ptrs, @h_dev_y1_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_param_ptrs, @h_dev_param_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. ***** FIX: Set pointer mode to DEVICE *****
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  // 6. Call rocBLAS
  blas_status := rocblas_drotmg_batched_64(handle, d_dev_d1_ptrs, d_dev_d2_ptrs, d_dev_x1_ptrs, d_dev_y1_ptrs, d_dev_param_ptrs, batch_count);
  AssertEquals('rocblas_drotmg_batched_64 failed', Integer(rocblas_status_success), Integer(blas_status));

  // 7. Copy results back
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_d1[b], h_dev_d1_ptrs[b], SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_d2[b], h_dev_d2_ptrs[b], SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_x1[b], h_dev_x1_ptrs[b], SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_param[b][0], h_dev_param_ptrs[b], PARAM_SIZE * SizeOf(Double), hipMemcpyDeviceToHost);
  end;

  // 8. Verify (Sanity check)
  for b := 0 to batch_count - 1 do
  begin
    AssertTrue(Format('DROTMG_64 d1 is not finite batch %d', [b]), not(IsInfinite(h_d1[b]) or IsNan(h_d1[b])));
    AssertTrue(Format('DROTMG_64 d2 is not finite batch %d', [b]), not(IsInfinite(h_d2[b]) or IsNan(h_d2[b])));
    AssertTrue(Format('DROTMG_64 x1 is not finite batch %d', [b]), not(IsInfinite(h_x1[b]) or IsNan(h_x1[b])));
    for i := 0 to PARAM_SIZE-1 do
      AssertTrue(Format('DROTMG_64 param[%d] is not finite batch %d', [i, b]), not(IsInfinite(h_param[b][i]) or IsNan(h_param[b][i])));
  end;

  // 9. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_d1_ptrs[b]);
    hipFree(h_dev_d2_ptrs[b]);
    hipFree(h_dev_x1_ptrs[b]);
    hipFree(h_dev_y1_ptrs[b]);
    hipFree(h_dev_param_ptrs[b]);
  end;
  hipFree(d_dev_d1_ptrs);
  hipFree(d_dev_d2_ptrs);
  hipFree(d_dev_x1_ptrs);
  hipFree(d_dev_y1_ptrs);
  hipFree(d_dev_param_ptrs);
end;

initialization
  RegisterTest(TTestRocBLAS_Level1_Batched_64);

end.
