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

unit Test_rocblas_lvl1_batched;

{$mode objfpc}{$H+}
{$linklib rocblas}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  TTestRocBLAS_Level1_Batched = class(TTestRocBLAS_Base)
  published
    procedure Test_SAXPY_Batched; // Single precision
    procedure Test_DAXPY_Batched; // Double precision
    procedure Test_CAXPY_Batched;
    procedure Test_ZAXPY_Batched;

    procedure Test_SCOPY_Batched;
    procedure Test_DCOPY_Batched;
    procedure Test_CCOPY_Batched;
    procedure Test_ZCOPY_Batched;

    procedure Test_SSCAL_Batched;
    procedure Test_DSCAL_Batched;
    procedure Test_CSSCAL_Batched; // Note: Wrapper uses rocblas_float_complex for alpha
    procedure Test_ZDSCAL_Batched; // Note: Wrapper uses rocblas_double_complex for alpha

    procedure Test_SDOT_Batched;
    procedure Test_DDOT_Batched;

    procedure Test_CDOTU_Batched;
    procedure Test_ZDOTU_Batched;
    procedure Test_CDOTC_Batched;
    procedure Test_ZDOTC_Batched;

    procedure Test_SNRM2_Batched;
    procedure Test_DNRM2_Batched;
    procedure Test_SCNRM2_Batched;
    procedure Test_DZNRM2_Batched;

    procedure Test_SASUM_Batched;
    procedure Test_DASUM_Batched;
    procedure Test_SCASUM_Batched;
    procedure Test_DZASUM_Batched;

    procedure Test_ISAMAX_Batched;
    procedure Test_IDAMAX_Batched;
    procedure Test_ICAMAX_Batched;
    procedure Test_IZAMAX_Batched;

    procedure Test_ISAMIN_Batched;
    procedure Test_IDAMIN_Batched;
    procedure Test_ICAMIN_Batched;
    procedure Test_IZAMIN_Batched;

    procedure Test_SSWAP_Batched;
    procedure Test_DSWAP_Batched;
    procedure Test_CSWAP_Batched;
    procedure Test_ZSWAP_Batched;

  end;

implementation

procedure TTestRocBLAS_Level1_Batched.Test_SAXPY_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  alpha: Single = 2.5;
type
  TSingleVector = array[0..N-1] of Single;
var
  h_x, h_y, h_y_expected: array[0..batch_count-1] of TSingleVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data and Expected Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := (b + 1) + (i + 1) / 10.0; // e.g. 1.1, 1.2 ... 2.1, 2.2 ...
      h_y[b][i] := (b + 1) * 10 + i;
      h_y_expected[b][i] := alpha * h_x[b][i] + h_y[b][i];
    end;
  end;

  // 2. Allocate Device Memory for each vector in the batch
  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Single));
    AssertEquals('hipMalloc for h_dev_x_ptrs['+IntToStr(b)+']', Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMalloc(h_dev_y_ptrs[b], N * SizeOf(Single));
    AssertEquals('hipMalloc for h_dev_y_ptrs['+IntToStr(b)+']', Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  // 3. Allocate Device Memory for the array of pointers
  hip_status := hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc for d_dev_x_ptrs', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  AssertEquals('hipMalloc for d_dev_y_ptrs', Integer(HIP_SUCCESS), Integer(hip_status));

  // 4. Copy Host Data to Device
  // a) Copy individual vectors
  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Single), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D for x batch '+IntToStr(b), Integer(HIP_SUCCESS), Integer(hip_status));
    hip_status := hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(Single), hipMemcpyHostToDevice);
    AssertEquals('hipMemcpy H2D for y batch '+IntToStr(b), Integer(HIP_SUCCESS), Integer(hip_status));
  end;
  // b) Copy the array of device pointers
  hip_status := hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D for x_ptrs array', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D for y_ptrs array', Integer(HIP_SUCCESS), Integer(hip_status));

  // 5. Call rocBLAS batched function
  blas_status := rocblas_saxpy_batched(handle, N, @alpha, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count);
  AssertEquals('rocblas_saxpy_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results from Device to Host
  for b := 0 to batch_count - 1 do
  begin
    hip_status := hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(Single), hipMemcpyDeviceToHost);
    AssertEquals('hipMemcpy D2H for y batch '+IntToStr(b), Integer(HIP_SUCCESS), Integer(hip_status));
  end;

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b][i], h_y[b][i], 1e-5,
        Format('SAXPY_Batched mismatch in batch %d, index %d', [b, i]));
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

procedure TTestRocBLAS_Level1_Batched.Test_DAXPY_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  alpha: Double = 3.14159;
type
  TDoubleVector = array[0..N-1] of Double;
var
  h_x, h_y, h_y_expected: array[0..batch_count-1] of TDoubleVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := (b + 1.5) + (i + 1.5) / 10.0;
      h_y[b][i] := (b + 1.5) * 100 + i;
      h_y_expected[b][i] := alpha * h_x[b][i] + h_y[b][i];
    end;
  end;

  // 2. Allocate Device Memory for each vector
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Double));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(Double));
  end;

  // 3. Allocate Device Memory for pointer array
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_daxpy_batched(handle, N, @alpha, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count);
  AssertEquals('rocblas_daxpy_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(Double), hipMemcpyDeviceToHost);
  end;

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b][i], h_y[b][i], 1e-12,
        Format('DAXPY_Batched mismatch in batch %d, index %d', [b, i]));
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

procedure TTestRocBLAS_Level1_Batched.Test_CAXPY_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  alpha: rocblas_float_complex;
  h_x, h_y, h_y_expected: array[0..batch_count-1] of TComplexVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  alpha.x := 1.5;
  alpha.y := -2.0;
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 1) + i;
      h_x[b][i].y := (b + 1) - i;
      h_y[b][i].x := b * 10;
      h_y[b][i].y := i * -5;
      // Expected: y = alpha * x + y
      h_y_expected[b][i].x := (alpha.x * h_x[b][i].x - alpha.y * h_x[b][i].y) + h_y[b][i].x;
      h_y_expected[b][i].y := (alpha.x * h_x[b][i].y + alpha.y * h_x[b][i].x) + h_y[b][i].y;
    end;
  end;

  // 2. Allocate Device Memory for each vector
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(rocblas_float_complex));
  end;

  // 3. Allocate Device Memory for pointer array
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_caxpy_batched(handle, N, @alpha, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count);
  AssertEquals('rocblas_caxpy_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  end;

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b][i].x, h_y[b][i].x, 1e-5,
        Format('CAXPY_Batched real mismatch in batch %d, index %d', [b, i]));
      CheckEquals(h_y_expected[b][i].y, h_y[b][i].y, 1e-5,
        Format('CAXPY_Batched imag mismatch in batch %d, index %d', [b, i]));
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

procedure TTestRocBLAS_Level1_Batched.Test_ZAXPY_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  alpha: rocblas_double_complex;
  h_x, h_y, h_y_expected: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  alpha.x := -0.5;
  alpha.y := 1.25;
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 0.5) + i;
      h_x[b][i].y := (b + 0.5) - i;
      h_y[b][i].x := b * 20.0;
      h_y[b][i].y := i * -15.0;
      // Expected: y = alpha * x + y
      h_y_expected[b][i].x := (alpha.x * h_x[b][i].x - alpha.y * h_x[b][i].y) + h_y[b][i].x;
      h_y_expected[b][i].y := (alpha.x * h_x[b][i].y + alpha.y * h_x[b][i].x) + h_y[b][i].y;
    end;
  end;

  // 2. Allocate Device Memory for each vector
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_double_complex));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(rocblas_double_complex));
  end;

  // 3. Allocate Device Memory for pointer array
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_zaxpy_batched(handle, N, @alpha, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count);
  AssertEquals('rocblas_zaxpy_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  end;

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b][i].x, h_y[b][i].x, 1e-12,
        Format('ZAXPY_Batched real mismatch in batch %d, index %d', [b, i]));
      CheckEquals(h_y_expected[b][i].y, h_y[b][i].y, 1e-12,
        Format('ZAXPY_Batched imag mismatch in batch %d, index %d', [b, i]));
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

procedure TTestRocBLAS_Level1_Batched.Test_SCOPY_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TSingleVector = array[0..N-1] of Single;
var
  h_x, h_y, h_y_expected: array[0..batch_count-1] of TSingleVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := (b + 1) * 10.0 + (i + 1) / 10.0; // Unique data for each batch
      h_y[b][i] := 0.0; // Initialize destination to zero
      h_y_expected[b][i] := h_x[b][i]; // Expected result is a direct copy
    end;
  end;

  // 2. Allocate Device Memory for each vector
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Single));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(Single));
  end;

  // 3. Allocate Device Memory for the array of pointers
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

  // 5. Call rocBLAS batched function
  blas_status := rocblas_scopy_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count);
  AssertEquals('rocblas_scopy_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results from Device to Host
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(Single), hipMemcpyDeviceToHost);
  end;

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b][i], h_y[b][i], 1e-6,
        Format('SCOPY_Batched mismatch in batch %d, index %d', [b, i]));
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

procedure TTestRocBLAS_Level1_Batched.Test_DCOPY_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TDoubleVector = array[0..N-1] of Double;
var
  h_x, h_y, h_y_expected: array[0..batch_count-1] of TDoubleVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := (b + 1) * 100.0 + (i + 1) / 100.0;
      h_y[b][i] := -1.0;
      h_y_expected[b][i] := h_x[b][i];
    end;
  end;

  // 2. Allocate Device Memory for each vector
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Double));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(Double));
  end;

  // 3. Allocate Device Memory for pointer array
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_dcopy_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count);
  AssertEquals('rocblas_dcopy_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(Double), hipMemcpyDeviceToHost);
  end;

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b][i], h_y[b][i], 1e-12,
        Format('DCOPY_Batched mismatch in batch %d, index %d', [b, i]));
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

procedure TTestRocBLAS_Level1_Batched.Test_CCOPY_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_x, h_y, h_y_expected: array[0..batch_count-1] of TComplexVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 1) + i;
      h_x[b][i].y := (b + 1) - i;
      h_y[b][i].x := 0.0;
      h_y[b][i].y := 0.0;
      h_y_expected[b][i] := h_x[b][i];
    end;
  end;

  // 2. Allocate Device Memory for each vector
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(rocblas_float_complex));
  end;

  // 3. Allocate Device Memory for pointer array
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_ccopy_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count);
  AssertEquals('rocblas_ccopy_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  end;

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b][i].x, h_y[b][i].x, 1e-6,
        Format('CCOPY_Batched real mismatch in batch %d, index %d', [b, i]));
      CheckEquals(h_y_expected[b][i].y, h_y[b][i].y, 1e-6,
        Format('CCOPY_Batched imag mismatch in batch %d, index %d', [b, i]));
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

procedure TTestRocBLAS_Level1_Batched.Test_ZCOPY_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_x, h_y, h_y_expected: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 1.5) + i;
      h_x[b][i].y := (b + 1.5) - i;
      h_y[b][i].x := -1.0;
      h_y[b][i].y := -1.0;
      h_y_expected[b][i] := h_x[b][i];
    end;
  end;

  // 2. Allocate Device Memory for each vector
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_double_complex));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(rocblas_double_complex));
  end;

  // 3. Allocate Device Memory for pointer array
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_zcopy_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count);
  AssertEquals('rocblas_zcopy_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  end;

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b][i].x, h_y[b][i].x, 1e-12,
        Format('ZCOPY_Batched real mismatch in batch %d, index %d', [b, i]));
      CheckEquals(h_y_expected[b][i].y, h_y[b][i].y, 1e-12,
        Format('ZCOPY_Batched imag mismatch in batch %d, index %d', [b, i]));
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

procedure TTestRocBLAS_Level1_Batched.Test_SSCAL_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  alpha: Single = 0.5;
type
  TSingleVector = array[0..N-1] of Single;
var
  h_x, h_x_expected: array[0..batch_count-1] of TSingleVector;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data and Expected Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := (b + 1) * 10.0 + i;
      h_x_expected[b][i] := alpha * h_x[b][i];
    end;
  end;

  // 2. Allocate Device Memory for each vector
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Single));
  end;

  // 3. Allocate Device Memory for the array of pointers
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Single), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS batched function
  blas_status := rocblas_sscal_batched(handle, N, @alpha, d_dev_x_ptrs, incx, batch_count);
  AssertEquals('rocblas_sscal_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results from Device to Host
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(Single), hipMemcpyDeviceToHost);
  end;

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b][i], h_x[b][i], 1e-6,
        Format('SSCAL_Batched mismatch in batch %d, index %d', [b, i]));
    end;
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
end;

procedure TTestRocBLAS_Level1_Batched.Test_DSCAL_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  alpha: Double = 1.2345;
type
  TDoubleVector = array[0..N-1] of Double;
var
  h_x, h_x_expected: array[0..batch_count-1] of TDoubleVector;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := (b + 1.5) * 20.0 + i;
      h_x_expected[b][i] := alpha * h_x[b][i];
    end;
  end;

  // 2. Allocate Device Memory for each vector
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Double));
  end;

  // 3. Allocate Device Memory for pointer array
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_dscal_batched(handle, N, @alpha, d_dev_x_ptrs, incx, batch_count);
  AssertEquals('rocblas_dscal_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(Double), hipMemcpyDeviceToHost);
  end;

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b][i], h_x[b][i], 1e-12,
        Format('DSCAL_Batched mismatch in batch %d, index %d', [b, i]));
    end;
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
end;

procedure TTestRocBLAS_Level1_Batched.Test_CSSCAL_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  alpha: Single = 2.5; // Real scalar for complex vector
type
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_x, h_x_expected: array[0..batch_count-1] of TComplexVector;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 1) + i;
      h_x[b][i].y := (b + 1) - i;
      h_x_expected[b][i].x := alpha * h_x[b][i].x;
      h_x_expected[b][i].y := alpha * h_x[b][i].y;
    end;
  end;

  // 2. Allocate Device Memory for each vector
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex));
  end;

  // 3. Allocate Device Memory for pointer array
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_csscal_batched(handle, N, @alpha, d_dev_x_ptrs, incx, batch_count);
  AssertEquals('rocblas_csscal_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  end;

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b][i].x, h_x[b][i].x, 1e-6,
        Format('CSSCAL_Batched real mismatch in batch %d, index %d', [b, i]));
      CheckEquals(h_x_expected[b][i].y, h_x[b][i].y, 1e-6,
        Format('CSSCAL_Batched imag mismatch in batch %d, index %d', [b, i]));
    end;
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
end;

procedure TTestRocBLAS_Level1_Batched.Test_ZDSCAL_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  alpha: Double = -1.5; // Real scalar for double-complex vector
type
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_x, h_x_expected: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 0.5) + i;
      h_x[b][i].y := (b + 0.5) - i;
      h_x_expected[b][i].x := alpha * h_x[b][i].x;
      h_x_expected[b][i].y := alpha * h_x[b][i].y;
    end;
  end;

  // 2. Allocate Device Memory for each vector
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_double_complex));
  end;

  // 3. Allocate Device Memory for pointer array
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_zdscal_batched(handle, N, @alpha, d_dev_x_ptrs, incx, batch_count);
  AssertEquals('rocblas_zdscal_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  end;

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b][i].x, h_x[b][i].x, 1e-12,
        Format('ZDSCAL_Batched real mismatch in batch %d, index %d', [b, i]));
      CheckEquals(h_x_expected[b][i].y, h_x[b][i].y, 1e-12,
        Format('ZDSCAL_Batched imag mismatch in batch %d, index %d', [b, i]));
    end;
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
end;

procedure TTestRocBLAS_Level1_Batched.Test_SDOT_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TSingleVector = array[0..N-1] of Single;
var
  h_x, h_y: array[0..batch_count-1] of TSingleVector;
  h_result, h_expected_result: array[0..batch_count-1] of Single;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs, d_result: hipDeviceptr_t;
  b, i: Integer;
  temp_sum: Double; // Use Double for intermediate summation to maintain precision
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data and Expected Results
  for b := 0 to batch_count - 1 do
  begin
    temp_sum := 0.0;
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := (b + 1) + i / 10.0;
      h_y[b][i] := (b + 1) * 2 - i / 10.0;
      temp_sum := temp_sum + h_x[b][i] * h_y[b][i];
    end;
    h_expected_result[b] := temp_sum;
    h_result[b] := 0.0;
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Single));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(Single));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_result, batch_count * SizeOf(Single));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Single), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(Single), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_result, @h_result[0], batch_count * SizeOf(Single), hipMemcpyHostToDevice);

  // 5. Call rocBLAS batched function
  blas_status := rocblas_sdot_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count, d_result);
  AssertEquals('rocblas_sdot_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results from Device to Host
  hipMemcpy(@h_result[0], d_result, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_result[b], h_result[b], 1e-4,
      Format('SDOT_Batched mismatch in batch %d', [b]));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
    hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_dev_y_ptrs);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1_Batched.Test_DDOT_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TDoubleVector = array[0..N-1] of Double;
var
  h_x, h_y: array[0..batch_count-1] of TDoubleVector;
  h_result, h_expected_result: array[0..batch_count-1] of Double;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs, d_result: hipDeviceptr_t;
  b, i: Integer;
  temp_sum: Extended; // Use Extended for max precision on intermediate host calc
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    temp_sum := 0.0;
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := (b + 1.5) + i / 20.0;
      h_y[b][i] := (b + 1.5) * 3 - i / 20.0;
      temp_sum := temp_sum + h_x[b][i] * h_y[b][i];
    end;
    h_expected_result[b] := temp_sum;
    h_result[b] := 0.0;
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Double));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(Double));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_result, batch_count * SizeOf(Double));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_result, @h_result[0], batch_count * SizeOf(Double), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_ddot_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count, d_result);
  AssertEquals('rocblas_ddot_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  hipMemcpy(@h_result[0], d_result, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_result[b], h_result[b], 1e-12,
      Format('DDOT_Batched mismatch in batch %d', [b]));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
    hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_dev_y_ptrs);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1_Batched.Test_CDOTU_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_x, h_y: array[0..batch_count-1] of TComplexVector;
  h_result, h_expected_result: array[0..batch_count-1] of rocblas_float_complex;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs, d_result: hipDeviceptr_t;
  b, i: Integer;
  sum_re, sum_im: Double;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    sum_re := 0.0; sum_im := 0.0;
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 1) + i * 0.1;
      h_x[b][i].y := (b + 1) - i * 0.1;
      h_y[b][i].x := i * 0.2 - b;
      h_y[b][i].y := i * 0.3 + b;
      // dotu = sum(x*y)
      sum_re := sum_re + (h_x[b][i].x * h_y[b][i].x - h_x[b][i].y * h_y[b][i].y);
      sum_im := sum_im + (h_x[b][i].x * h_y[b][i].y + h_x[b][i].y * h_y[b][i].x);
    end;
    h_expected_result[b].x := sum_re;
    h_expected_result[b].y := sum_im;
    h_result[b].x := 0.0;
    h_result[b].y := 0.0;
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(rocblas_float_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_result, batch_count * SizeOf(rocblas_float_complex));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_result, @h_result[0], batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_cdotu_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count, d_result);
  AssertEquals('rocblas_cdotu_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  hipMemcpy(@h_result[0], d_result, batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_result[b].x, h_result[b].x, 1e-4, Format('CDOTU_Batched real mismatch batch %d', [b]));
    CheckEquals(h_expected_result[b].y, h_result[b].y, 1e-4, Format('CDOTU_Batched imag mismatch batch %d', [b]));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
    hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_dev_y_ptrs);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1_Batched.Test_ZDOTU_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_x, h_y: array[0..batch_count-1] of TDoubleComplexVector;
  h_result, h_expected_result: array[0..batch_count-1] of rocblas_double_complex;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs, d_result: hipDeviceptr_t;
  b, i: Integer;
  sum_re, sum_im: Extended;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    sum_re := 0.0; sum_im := 0.0;
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 1.1) + i * 0.11;
      h_x[b][i].y := (b + 1.1) - i * 0.11;
      h_y[b][i].x := i * 0.22 - b;
      h_y[b][i].y := i * 0.33 + b;
      sum_re := sum_re + (h_x[b][i].x * h_y[b][i].x - h_x[b][i].y * h_y[b][i].y);
      sum_im := sum_im + (h_x[b][i].x * h_y[b][i].y + h_x[b][i].y * h_y[b][i].x);
    end;
    h_expected_result[b].x := sum_re;
    h_expected_result[b].y := sum_im;
    h_result[b].x := 0.0;
    h_result[b].y := 0.0;
  end;

  // 2. Allocate Device Memory
  // ... similar allocation logic as CDOTU ...
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_double_complex));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(rocblas_double_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_result, batch_count * SizeOf(rocblas_double_complex));

  // 4. Copy Host Data to Device
  // ... similar copy logic as CDOTU ...
   for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_result, @h_result[0], batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_zdotu_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count, d_result);
  AssertEquals('rocblas_zdotu_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  hipMemcpy(@h_result[0], d_result, batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_result[b].x, h_result[b].x, 1e-12, Format('ZDOTU_Batched real mismatch batch %d', [b]));
    CheckEquals(h_expected_result[b].y, h_result[b].y, 1e-12, Format('ZDOTU_Batched imag mismatch batch %d', [b]));
  end;

  // 8. Cleanup
  // ... similar cleanup logic as CDOTU ...
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
    hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_dev_y_ptrs);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1_Batched.Test_CDOTC_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_x, h_y: array[0..batch_count-1] of TComplexVector;
  h_result, h_expected_result: array[0..batch_count-1] of rocblas_float_complex;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs, d_result: hipDeviceptr_t;
  b, i: Integer;
  sum_re, sum_im: Double;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    sum_re := 0.0; sum_im := 0.0;
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 1) + i * 0.1;
      h_x[b][i].y := (b + 1) - i * 0.1;
      h_y[b][i].x := i * 0.2 - b;
      h_y[b][i].y := i * 0.3 + b;
      // dotc = sum(conj(x)*y)
      sum_re := sum_re + (h_x[b][i].x * h_y[b][i].x + h_x[b][i].y * h_y[b][i].y);
      sum_im := sum_im + (h_x[b][i].x * h_y[b][i].y - h_x[b][i].y * h_y[b][i].x);
    end;
    h_expected_result[b].x := sum_re;
    h_expected_result[b].y := sum_im;
    h_result[b].x := 0.0;
    h_result[b].y := 0.0;
  end;

  // ... Allocation, Copy, and Cleanup logic is identical to CDOTU ...
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(rocblas_float_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_result, batch_count * SizeOf(rocblas_float_complex));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_result, @h_result[0], batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_cdotc_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count, d_result);
  AssertEquals('rocblas_cdotc_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  hipMemcpy(@h_result[0], d_result, batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_result[b].x, h_result[b].x, 1e-4, Format('CDOTC_Batched real mismatch batch %d', [b]));
    CheckEquals(h_expected_result[b].y, h_result[b].y, 1e-4, Format('CDOTC_Batched imag mismatch batch %d', [b]));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
    hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_dev_y_ptrs);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1_Batched.Test_ZDOTC_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_x, h_y: array[0..batch_count-1] of TDoubleComplexVector;
  h_result, h_expected_result: array[0..batch_count-1] of rocblas_double_complex;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs, d_result: hipDeviceptr_t;
  b, i: Integer;
  sum_re, sum_im: Extended;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    sum_re := 0.0; sum_im := 0.0;
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 1.1) + i * 0.11;
      h_x[b][i].y := (b + 1.1) - i * 0.11;
      h_y[b][i].x := i * 0.22 - b;
      h_y[b][i].y := i * 0.33 + b;
      // dotc = sum(conj(x)*y)
      sum_re := sum_re + (h_x[b][i].x * h_y[b][i].x + h_x[b][i].y * h_y[b][i].y);
      sum_im := sum_im + (h_x[b][i].x * h_y[b][i].y - h_x[b][i].y * h_y[b][i].x);
    end;
    h_expected_result[b].x := sum_re;
    h_expected_result[b].y := sum_im;
    h_result[b].x := 0.0;
    h_result[b].y := 0.0;
  end;

  // ... Allocation, Copy, and Cleanup logic is identical to ZDOTU ...
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_double_complex));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(rocblas_double_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_result, batch_count * SizeOf(rocblas_double_complex));
   for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_result, @h_result[0], batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_zdotc_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count, d_result);
  AssertEquals('rocblas_zdotc_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  hipMemcpy(@h_result[0], d_result, batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_result[b].x, h_result[b].x, 1e-12, Format('ZDOTC_Batched real mismatch batch %d', [b]));
    CheckEquals(h_expected_result[b].y, h_result[b].y, 1e-12, Format('ZDOTC_Batched imag mismatch batch %d', [b]));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
    hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_dev_y_ptrs);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1_Batched.Test_SNRM2_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TSingleVector = array[0..N-1] of Single;
var
  h_x: array[0..batch_count-1] of TSingleVector;
  h_results, h_expected_results: array[0..batch_count-1] of Single;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data and Expected Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := Power(-1, i) * (b + 1 + i*0.2); // Alternating sign data
    end;
    // Calculate expected norm using helper function for accuracy
    h_expected_results[b] := CalculateRealNorm(N, PSingle(@h_x[b][0]), incx);
    h_results[b] := -1.0; // Initialize results to a known incorrect value
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Single));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Single)); // Results are an array of Singles

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Single), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS batched function
  blas_status := rocblas_snrm2_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_snrm2_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results from Device to Host
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-5,
      Format('SNRM2_Batched mismatch in batch %d', [b]));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_DNRM2_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TDoubleVector = array[0..N-1] of Double;
var
  h_x: array[0..batch_count-1] of TDoubleVector;
  h_results, h_expected_results: array[0..batch_count-1] of Double;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := Power(-1, i) * (b + 1.5 + i*0.25);
    end;
    h_expected_results[b] := CalculateRealNorm(N, PDouble(@h_x[b][0]), incx);
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Double));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Double));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_dnrm2_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_dnrm2_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-12,
      Format('DNRM2_Batched mismatch in batch %d', [b]));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_SCNRM2_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_x: array[0..batch_count-1] of TComplexVector;
  h_results, h_expected_results: array[0..batch_count-1] of Single; // Result is real
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 1) + i * 0.1;
      h_x[b][i].y := Power(-1, i) * ((b + 1) - i * 0.1);
    end;
    h_expected_results[b] := CalculateComplexNorm(N, Procfloat_complex(@h_x[b][0]), incx);
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Single)); // Result is array of Singles

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_scnrm2_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_scnrm2_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-5,
      Format('SCNRM2_Batched mismatch in batch %d', [b]));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_DZNRM2_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_x: array[0..batch_count-1] of TDoubleComplexVector;
  h_results, h_expected_results: array[0..batch_count-1] of Double; // Result is real
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 1.5) + i * 0.15;
      h_x[b][i].y := Power(-1, i) * ((b + 1.5) - i * 0.15);
    end;
    h_expected_results[b] := CalculateComplexNorm(N, Procdouble_complex(@h_x[b][0]), incx);
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_double_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Double)); // Result is array of Doubles

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_dznrm2_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_dznrm2_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-12,
      Format('DZNRM2_Batched mismatch in batch %d', [b]));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_SASUM_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TSingleVector = array[0..N-1] of Single;
var
  h_x: array[0..batch_count-1] of TSingleVector;
  h_results, h_expected_results: array[0..batch_count-1] of Single;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  temp_sum: Double;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data and Expected Results
  for b := 0 to batch_count - 1 do
  begin
    temp_sum := 0.0;
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := Power(-1, i) * (b + 1 + i*0.2);
      temp_sum := temp_sum + Abs(h_x[b][i]);
    end;
    h_expected_results[b] := temp_sum;
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Single));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Single));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Single), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS batched function
  blas_status := rocblas_sasum_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_sasum_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results from Device to Host
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-5,
      Format('SASUM_Batched mismatch in batch %d', [b]));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_DASUM_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TDoubleVector = array[0..N-1] of Double;
var
  h_x: array[0..batch_count-1] of TDoubleVector;
  h_results, h_expected_results: array[0..batch_count-1] of Double;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  temp_sum: Extended;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    temp_sum := 0.0;
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := Power(-1, i) * (b + 1.5 + i*0.25);
      temp_sum := temp_sum + Abs(h_x[b][i]);
    end;
    h_expected_results[b] := temp_sum;
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Double));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Double));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_dasum_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_dasum_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-12,
      Format('DASUM_Batched mismatch in batch %d', [b]));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_SCASUM_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_x: array[0..batch_count-1] of TComplexVector;
  h_results, h_expected_results: array[0..batch_count-1] of Single; // Result is real
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  temp_sum: Double;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    temp_sum := 0.0;
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := Power(-1, i) * ((b + 1) + i * 0.1);
      h_x[b][i].y := Power(-1, i+1) * ((b + 1) - i * 0.1);
      temp_sum := temp_sum + Abs(h_x[b][i].x) + Abs(h_x[b][i].y);
    end;
    h_expected_results[b] := temp_sum;
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Single));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_scasum_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_scasum_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-5,
      Format('SCASUM_Batched mismatch in batch %d', [b]));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_DZASUM_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_x: array[0..batch_count-1] of TDoubleComplexVector;
  h_results, h_expected_results: array[0..batch_count-1] of Double; // Result is real
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  temp_sum: Extended;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    temp_sum := 0.0;
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := Power(-1, i) * ((b + 1.5) + i * 0.15);
      h_x[b][i].y := Power(-1, i+1) * ((b + 1.5) - i * 0.15);
      temp_sum := temp_sum + Abs(h_x[b][i].x) + Abs(h_x[b][i].y);
    end;
    h_expected_results[b] := temp_sum;
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_double_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Double));

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_dzasum_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_dzasum_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-12,
      Format('DZASUM_Batched mismatch in batch %d', [b]));
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_ISAMAX_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TSingleVector = array[0..N-1] of Single;
var
  h_x: array[0..batch_count-1] of TSingleVector;
  h_results, h_expected_results: array[0..batch_count-1] of Int32;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  max_val: Single;
  max_idx: Int32;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data and Expected Results
  for b := 0 to batch_count - 1 do
  begin
    max_val := -1.0;
    max_idx := -1;
    for i := 0 to N - 1 do
    begin
      // Create data where max value is unique for each batch
      h_x[b][i] := Power(-1, i) * (b + 1 + i);
      if Abs(h_x[b][i]) > max_val then
      begin
        max_val := Abs(h_x[b][i]);
        max_idx := i + 1; // 1-based index
      end;
    end;
    h_expected_results[b] := max_idx;
    h_results[b] := -1;
  end;

  // 2. Allocate Device Memory
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Single));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Int32)); // Results are integers

  // 4. Copy Host Data to Device
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Single), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS batched function
  blas_status := rocblas_isamax_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_isamax_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results from Device to Host
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('ISAMAX_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_IDAMAX_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TDoubleVector = array[0..N-1] of Double;
var
  h_x: array[0..batch_count-1] of TDoubleVector;
  h_results, h_expected_results: array[0..batch_count-1] of Int32;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  max_val: Double;
  max_idx: Int32;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    max_val := -1.0;
    max_idx := -1;
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := Power(-1, i) * (b + 1.5 + i);
      if Abs(h_x[b][i]) > max_val then
      begin
        max_val := Abs(h_x[b][i]);
        max_idx := i + 1;
      end;
    end;
    h_expected_results[b] := max_idx;
    h_results[b] := -1;
  end;

  // ... (Allocation, Copy, Cleanup logic is identical to ISAMAX, using Double)
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Double));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_idamax_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_idamax_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  // 7. Verify
  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('IDAMAX_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_ICAMAX_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_x: array[0..batch_count-1] of TComplexVector;
  h_results, h_expected_results: array[0..batch_count-1] of Int32;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  max_val, current_val: Single;
  max_idx: Int32;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    max_val := -1.0;
    max_idx := -1;
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := Power(-1, i) * (b + 1 + i*0.1);
      h_x[b][i].y := Power(-1, i+1) * (b + 1 - i*0.1);
      current_val := Abs(h_x[b][i].x) + Abs(h_x[b][i].y);
      if current_val > max_val then
      begin
        max_val := current_val;
        max_idx := i + 1;
      end;
    end;
    h_expected_results[b] := max_idx;
    h_results[b] := -1;
  end;

  // ... (Allocation, Copy, Cleanup logic is identical, using rocblas_float_complex)
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_icamax_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_icamax_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  // 7. Verify
  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('ICAMAX_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_IZAMAX_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_x: array[0..batch_count-1] of TDoubleComplexVector;
  h_results, h_expected_results: array[0..batch_count-1] of Int32;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  max_val, current_val: Double;
  max_idx: Int32;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    max_val := -1.0;
    max_idx := -1;
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := Power(-1, i) * (b + 1.5 + i*0.15);
      h_x[b][i].y := Power(-1, i+1) * (b + 1.5 - i*0.15);
      current_val := Abs(h_x[b][i].x) + Abs(h_x[b][i].y);
      if current_val > max_val then
      begin
        max_val := current_val;
        max_idx := i + 1;
      end;
    end;
    h_expected_results[b] := max_idx;
    h_results[b] := -1;
  end;

  // ... (Allocation, Copy, Cleanup logic is identical, using rocblas_double_complex)
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_double_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_izamax_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_izamax_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  // 7. Verify
  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('IZAMAX_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_ISAMIN_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TSingleVector = array[0..N-1] of Single;
var
  h_x: array[0..batch_count-1] of TSingleVector;
  h_results, h_expected_results: array[0..batch_count-1] of Int32;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  min_val: Single;
  min_idx: Int32;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data and Expected Results
  for b := 0 to batch_count - 1 do
  begin
    min_val := 1E38; // A very large number
    min_idx := -1;
    for i := 0 to N - 1 do
    begin
      // Make the minimum value unique for each batch, e.g., at index b
      if i = b then
         h_x[b][i] := 0.1
      else
         h_x[b][i] := Power(-1, i) * (b + 1 + i);

      if Abs(h_x[b][i]) < min_val then
      begin
        min_val := Abs(h_x[b][i]);
        min_idx := i + 1; // 1-based index
      end;
    end;
    h_expected_results[b] := min_idx;
    h_results[b] := -1;
  end;

  // ... (Allocation, Copy, Cleanup logic is identical to ISAMAX)
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Single));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Single), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS batched function
  blas_status := rocblas_isamin_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_isamin_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results from Device to Host
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('ISAMIN_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_IDAMIN_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TDoubleVector = array[0..N-1] of Double;
var
  h_x: array[0..batch_count-1] of TDoubleVector;
  h_results, h_expected_results: array[0..batch_count-1] of Int32;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  min_val: Double;
  min_idx: Int32;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    min_val := 1E300;
    min_idx := -1;
    for i := 0 to N - 1 do
    begin
       if i = b+1 then
         h_x[b][i] := 0.01
       else
         h_x[b][i] := Power(-1, i) * (b + 1.5 + i);

      if Abs(h_x[b][i]) < min_val then
      begin
        min_val := Abs(h_x[b][i]);
        min_idx := i + 1;
      end;
    end;
    h_expected_results[b] := min_idx;
    h_results[b] := -1;
  end;

  // ... (Allocation, Copy, Cleanup logic is identical to IDAMAX)
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(Double));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(Double), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_idamin_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_idamin_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  // 7. Verify
  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('IDAMIN_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_ICAMIN_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_x: array[0..batch_count-1] of TComplexVector;
  h_results, h_expected_results: array[0..batch_count-1] of Int32;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  min_val, current_val: Single;
  min_idx: Int32;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    min_val := 1E38;
    min_idx := -1;
    for i := 0 to N - 1 do
    begin
      if i = b+2 then
      begin
        h_x[b][i].x := 0.01;
        h_x[b][i].y := -0.01;
      end
      else
      begin
        h_x[b][i].x := Power(-1, i) * (b + 1 + i*0.1);
        h_x[b][i].y := Power(-1, i+1) * (b + 1 - i*0.1);
      end;
      current_val := Abs(h_x[b][i].x) + Abs(h_x[b][i].y);
      if current_val < min_val then
      begin
        min_val := current_val;
        min_idx := i + 1;
      end;
    end;
    h_expected_results[b] := min_idx;
    h_results[b] := -1;
  end;

  // ... (Allocation, Copy, Cleanup logic is identical, using rocblas_float_complex)
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_icamin_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_icamin_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  // 7. Verify
  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('ICAMIN_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_IZAMIN_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
type
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_x: array[0..batch_count-1] of TDoubleComplexVector;
  h_results, h_expected_results: array[0..batch_count-1] of Int32;
  h_dev_x_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_results: hipDeviceptr_t;
  b, i: Integer;
  min_val, current_val: Double;
  min_idx: Int32;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    min_val := 1E300;
    min_idx := -1;
    for i := 0 to N - 1 do
    begin
      if i = b+3 then
      begin
        h_x[b][i].x := 0.001;
        h_x[b][i].y := 0.001;
      end
      else
      begin
        h_x[b][i].x := Power(-1, i) * (b + 1.5 + i*0.15);
        h_x[b][i].y := Power(-1, i+1) * (b + 1.5 - i*0.15);
      end;
      current_val := Abs(h_x[b][i].x) + Abs(h_x[b][i].y);
      if current_val < min_val then
      begin
        min_val := current_val;
        min_idx := i + 1;
      end;
    end;
    h_expected_results[b] := min_idx;
    h_results[b] := -1;
  end;

  // ... (Allocation, Copy, Cleanup logic is identical, using rocblas_double_complex)
  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_double_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  // 5. Call rocBLAS function
  blas_status := rocblas_izamin_batched(handle, N, d_dev_x_ptrs, incx, batch_count, d_results);
  AssertEquals('rocblas_izamin_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  // 7. Verify
  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('IZAMIN_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  // 8. Cleanup
  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Batched.Test_SSWAP_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TSingleVector = array[0..N-1] of Single;
var
  h_x, h_y, h_x_orig, h_y_orig: array[0..batch_count-1] of TSingleVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := (b + 1) * 10.0 + i;
      h_y[b][i] := (b + 1) * 100.0 + i;
    end;
    h_x_orig[b] := h_x[b];
    h_y_orig[b] := h_y[b];
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

  // 5. Call rocBLAS function
  blas_status := rocblas_sswap_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count);
  AssertEquals('rocblas_sswap_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 6. Copy Results D2H
  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(Single), hipMemcpyDeviceToHost);
  end;

  // 7. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_orig[b][i], h_x[b][i], 1e-6, Format('SSWAP_Batched x mismatch batch %d index %d', [b, i]));
      CheckEquals(h_x_orig[b][i], h_y[b][i], 1e-6, Format('SSWAP_Batched y mismatch batch %d index %d', [b, i]));
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

procedure TTestRocBLAS_Level1_Batched.Test_DSWAP_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TDoubleVector = array[0..N-1] of Double;
var
  h_x, h_y, h_x_orig, h_y_orig: array[0..batch_count-1] of TDoubleVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // ... Logic identical to SSWAP_Batched, but with Double type ...
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i] := (b + 1) * 10.0 + i;
      h_y[b][i] := (b + 1) * 100.0 + i;
    end;
    h_x_orig[b] := h_x[b];
    h_y_orig[b] := h_y[b];
  end;

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

  blas_status := rocblas_dswap_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count);
  AssertEquals('rocblas_dswap_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(Double), hipMemcpyDeviceToHost);
  end;

  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_orig[b][i], h_x[b][i], 1e-12, Format('DSWAP_Batched x mismatch batch %d index %d', [b, i]));
      CheckEquals(h_x_orig[b][i], h_y[b][i], 1e-12, Format('DSWAP_Batched y mismatch batch %d index %d', [b, i]));
    end;
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
    hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_dev_y_ptrs);
end;

procedure TTestRocBLAS_Level1_Batched.Test_CSWAP_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TComplexVector = array[0..N-1] of rocblas_float_complex;
var
  h_x, h_y, h_x_orig, h_y_orig: array[0..batch_count-1] of TComplexVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // ... Logic identical to SSWAP_Batched, but with rocblas_float_complex type ...
   for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 1) * 10.0 + i;
      h_x[b][i].y := (b + 1) * -1.0 - i;
      h_y[b][i].x := (b + 1) * 100.0 + i;
      h_y[b][i].y := (b + 1) * -10.0 - i;
    end;
    h_x_orig[b] := h_x[b];
    h_y_orig[b] := h_y[b];
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

  blas_status := rocblas_cswap_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count);
  AssertEquals('rocblas_cswap_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  end;

  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_orig[b][i].x, h_x[b][i].x, 1e-6, Format('CSWAP_Batched x.real mismatch batch %d index %d', [b, i]));
      CheckEquals(h_y_orig[b][i].y, h_x[b][i].y, 1e-6, Format('CSWAP_Batched x.imag mismatch batch %d index %d', [b, i]));
      CheckEquals(h_x_orig[b][i].x, h_y[b][i].x, 1e-6, Format('CSWAP_Batched y.real mismatch batch %d index %d', [b, i]));
      CheckEquals(h_x_orig[b][i].y, h_y[b][i].y, 1e-6, Format('CSWAP_Batched y.imag mismatch batch %d index %d', [b, i]));
    end;
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
    hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_dev_y_ptrs);
end;

procedure TTestRocBLAS_Level1_Batched.Test_ZSWAP_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
type
  TDoubleComplexVector = array[0..N-1] of rocblas_double_complex;
var
  h_x, h_y, h_x_orig, h_y_orig: array[0..batch_count-1] of TDoubleComplexVector;
  h_dev_x_ptrs, h_dev_y_ptrs: array[0..batch_count-1] of hipDeviceptr_t;
  d_dev_x_ptrs, d_dev_y_ptrs: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // ... Logic identical to SSWAP_Batched, but with rocblas_double_complex type ...
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b][i].x := (b + 1) * 10.0 + i;
      h_x[b][i].y := (b + 1) * -1.0 - i;
      h_y[b][i].x := (b + 1) * 100.0 + i;
      h_y[b][i].y := (b + 1) * -10.0 - i;
    end;
    h_x_orig[b] := h_x[b];
    h_y_orig[b] := h_y[b];
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipMalloc(h_dev_x_ptrs[b], N * SizeOf(rocblas_double_complex));
    hipMalloc(h_dev_y_ptrs[b], N * SizeOf(rocblas_double_complex));
  end;
  hipMalloc(d_dev_x_ptrs, batch_count * SizeOf(hipDeviceptr_t));
  hipMalloc(d_dev_y_ptrs, batch_count * SizeOf(hipDeviceptr_t));

  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(h_dev_x_ptrs[b], @h_x[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
    hipMemcpy(h_dev_y_ptrs[b], @h_y[b][0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  end;
  hipMemcpy(d_dev_x_ptrs, @h_dev_x_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);
  hipMemcpy(d_dev_y_ptrs, @h_dev_y_ptrs[0], batch_count * SizeOf(hipDeviceptr_t), hipMemcpyHostToDevice);

  blas_status := rocblas_zswap_batched(handle, N, d_dev_x_ptrs, incx, d_dev_y_ptrs, incy, batch_count);
  AssertEquals('rocblas_zswap_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  for b := 0 to batch_count - 1 do
  begin
    hipMemcpy(@h_x[b][0], h_dev_x_ptrs[b], N * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[b][0], h_dev_y_ptrs[b], N * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  end;

  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_orig[b][i].x, h_x[b][i].x, 1e-12, Format('ZSWAP_Batched x.real mismatch batch %d index %d', [b, i]));
      CheckEquals(h_y_orig[b][i].y, h_x[b][i].y, 1e-12, Format('ZSWAP_Batched x.imag mismatch batch %d index %d', [b, i]));
      CheckEquals(h_x_orig[b][i].x, h_y[b][i].x, 1e-12, Format('ZSWAP_Batched y.real mismatch batch %d index %d', [b, i]));
      CheckEquals(h_x_orig[b][i].y, h_y[b][i].y, 1e-12, Format('ZSWAP_Batched y.imag mismatch batch %d index %d', [b, i]));
    end;
  end;

  for b := 0 to batch_count - 1 do
  begin
    hipFree(h_dev_x_ptrs[b]);
    hipFree(h_dev_y_ptrs[b]);
  end;
  hipFree(d_dev_x_ptrs);
  hipFree(d_dev_y_ptrs);
end;


initialization
  RegisterTest(TTestRocBLAS_Level1_Batched);

end.

