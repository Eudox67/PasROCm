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

  Signature checks for the rocBLAS level-1 (vector, strided-batched)
  bindings: every call is made and its status is asserted, so a changed
  prototype or a missing symbol fails here first.
------------------------------------------------------------------------------}

unit Test_rocblas_lvl1_strided_batched;

{$mode objfpc}{$H+}
{$linklib rocblas}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  { TTestRocBLAS_Level1_Strided_Batched }
  TTestRocBLAS_Level1_Strided_Batched = class(TTestRocBLAS_Base)
  published
    procedure Test_SAXPY_Strided_Batched;
    procedure Test_DAXPY_Strided_Batched;
    procedure Test_CAXPY_Strided_Batched;
    procedure Test_ZAXPY_Strided_Batched;

    procedure Test_SCOPY_Strided_Batched;
    procedure Test_DCOPY_Strided_Batched;
    procedure Test_CCOPY_Strided_Batched;
    procedure Test_ZCOPY_Strided_Batched;

    procedure Test_SSCAL_Strided_Batched;
    procedure Test_DSCAL_Strided_Batched;
    procedure Test_CSCAL_Strided_Batched;
    procedure Test_ZSCAL_Strided_Batched;
    procedure Test_CSSCAL_Strided_Batched;
    procedure Test_ZDSCAL_Strided_Batched;

    procedure Test_SDOT_Strided_Batched;
    procedure Test_DDOT_Strided_Batched;
    procedure Test_CDOTU_Strided_Batched;
    procedure Test_ZDOTU_Strided_Batched;
    procedure Test_CDOTC_Strided_Batched;
    procedure Test_ZDOTC_Strided_Batched;

    procedure Test_SNRM2_Strided_Batched;
    procedure Test_DNRM2_Strided_Batched;
    procedure Test_SCNRM2_Strided_Batched;
    procedure Test_DZNRM2_Strided_Batched;

    procedure Test_SASUM_Strided_Batched;
    procedure Test_DASUM_Strided_Batched;
    procedure Test_SCASUM_Strided_Batched;
    procedure Test_DZASUM_Strided_Batched;

    procedure Test_ISAMAX_Strided_Batched;
    procedure Test_IDAMAX_Strided_Batched;
    procedure Test_ICAMAX_Strided_Batched;
    procedure Test_IZAMAX_Strided_Batched;

    procedure Test_ISAMIN_Strided_Batched;
    procedure Test_IDAMIN_Strided_Batched;
    procedure Test_ICAMIN_Strided_Batched;
    procedure Test_IZAMIN_Strided_Batched;

    procedure Test_SSWAP_Strided_Batched;
    procedure Test_DSWAP_Strided_Batched;
    procedure Test_CSWAP_Strided_Batched;
    procedure Test_ZSWAP_Strided_Batched;

    procedure Test_SROT_Strided_Batched;
    procedure Test_DROT_Strided_Batched;
    procedure Test_CROT_Strided_Batched;
    procedure Test_CSROT_Strided_Batched;
    procedure Test_ZROT_Strided_Batched;
    procedure Test_ZDROT_Strided_Batched;

    procedure Test_SROTG_Strided_Batched;
    procedure Test_DROTG_Strided_Batched;
    procedure Test_CROTG_Strided_Batched;
    procedure Test_ZROTG_Strided_Batched;

    procedure Test_SROTM_Strided_Batched;
    procedure Test_DROTM_Strided_Batched;

    procedure Test_SROTMG_Strided_Batched;
    procedure Test_DROTMG_Strided_Batched;
  end;

implementation

{ TTestRocBLAS_Level1_Strided_Batched }

procedure TTestRocBLAS_Level1_Strided_Batched.Test_SAXPY_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  alpha: Single = 2.5;
  // Make stride larger than N to test non-contiguous batches
  stride_x = 15;
  stride_y = 20;
  // Total size needed for the single contiguous memory block
  size_x = stride_x * (batch_count - 1) + N;
  size_y = stride_y * (batch_count - 1) + N;
type
  TSingleVector = array[0..size_x-1] of Single;
  TSingleVectorY = array[0..size_y-1] of Single;
var
  h_x, h_x_expected: TSingleVector;
  h_y, h_y_expected: TSingleVectorY;
  d_x, d_y: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data and Expected Results in a single large array
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      // Calculate the index based on stride
      h_x[b * stride_x + i * incx] := (b + 1) + (i + 1) / 10.0;
      h_y[b * stride_y + i * incy] := (b + 1) * 10 + i;
      h_y_expected[b * stride_y + i * incy] := alpha * h_x[b * stride_x + i * incx] + h_y[b * stride_y + i * incy];
    end;
  end;

  // 2. Allocate one large block of Device Memory for x and y
  hip_status := hipMalloc(d_x, size_x * SizeOf(Single));
  AssertEquals('hipMalloc for d_x failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_y, size_y * SizeOf(Single));
  AssertEquals('hipMalloc for d_y failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // 3. Copy the entire host block to the device block
  hip_status := hipMemcpy(d_x, @h_x[0], size_x * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D for x failed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(d_y, @h_y[0], size_y * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D for y failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // 4. Call rocBLAS strided batched function
  blas_status := rocblas_saxpy_strided_batched(handle, N, @alpha, d_x, incx, stride_x, d_y, incy, stride_y, batch_count);
  AssertEquals('rocblas_saxpy_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy the entire result block back from Device to Host
  hip_status := hipMemcpy(@h_y[0], d_y, size_y * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy D2H for y failed', Integer(HIP_SUCCESS), Integer(hip_status));

  // 6. Verify Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b * stride_y + i * incy], h_y[b * stride_y + i * incy], 1e-5,
        Format('SAXPY_Strided_Batched mismatch in batch %d, index %d', [b, i]));
    end;
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_DAXPY_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  alpha: Double = 3.14159;
  stride = 16; // Use same stride for simplicity
  total_size = stride * (batch_count - 1) + N;
type
  TDoubleVector = array[0..total_size-1] of Double;
var
  h_x, h_y, h_y_expected: TDoubleVector;
  d_x, d_y: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b * stride + i] := (b + 1.5) + (i + 1.5) / 10.0;
      h_y[b * stride + i] := (b + 1.5) * 100 + i;
      h_y_expected[b * stride + i] := alpha * h_x[b * stride + i] + h_y[b * stride + i];
    end;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(Double));
  hipMalloc(d_y, total_size * SizeOf(Double));

  // 3. Copy Host Data to Device
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);

  // 4. Call rocBLAS function
  blas_status := rocblas_daxpy_strided_batched(handle, N, @alpha, d_x, incx, stride, d_y, incy, stride, batch_count);
  AssertEquals('rocblas_daxpy_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy results back
  hipMemcpy(@h_y[0], d_y, total_size * SizeOf(Double), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b * stride + i], h_y[b * stride + i], 1e-12,
        Format('DAXPY_Strided_Batched mismatch in batch %d, index %d', [b, i]));
    end;
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_CAXPY_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count-1) + N;
type
  TComplexVector = array[0..total_size-1] of rocblas_float_complex;
var
  h_x, h_y, h_y_expected: TComplexVector;
  alpha: rocblas_float_complex;
  d_x, d_y: hipDeviceptr_t;
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
      h_x[b * stride + i].x := (b + 1) + i;
      h_x[b * stride + i].y := (b + 1) - i;
      h_y[b * stride + i].x := b * 10;
      h_y[b * stride + i].y := i * -5;
      h_y_expected[b * stride + i].x := (alpha.x * h_x[b * stride + i].x - alpha.y * h_x[b * stride + i].y) + h_y[b * stride + i].x;
      h_y_expected[b * stride + i].y := (alpha.x * h_x[b * stride + i].y + alpha.y * h_x[b * stride + i].x) + h_y[b * stride + i].y;
    end;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_float_complex));

  // 3. Copy Data
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_caxpy_strided_batched(handle, N, @alpha, d_x, incx, stride, d_y, incy, stride, batch_count);
  AssertEquals('rocblas_caxpy_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy back
  hipMemcpy(@h_y[0], d_y, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b*stride+i].x, h_y[b*stride+i].x, 1e-5, 'CAXPY_Strided real mismatch');
      CheckEquals(h_y_expected[b*stride+i].y, h_y[b*stride+i].y, 1e-5, 'CAXPY_Strided imag mismatch');
    end;
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ZAXPY_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count-1) + N;
type
  TComplexVector = array[0..total_size-1] of rocblas_double_complex;
var
  h_x, h_y, h_y_expected: TComplexVector;
  alpha: rocblas_double_complex;
  d_x, d_y: hipDeviceptr_t;
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
      h_x[b * stride + i].x := (b + 0.5) + i;
      h_x[b * stride + i].y := (b + 0.5) - i;
      h_y[b * stride + i].x := b * 20.0;
      h_y[b * stride + i].y := i * -15.0;
      h_y_expected[b * stride + i].x := (alpha.x * h_x[b * stride + i].x - alpha.y * h_x[b * stride + i].y) + h_y[b * stride + i].x;
      h_y_expected[b * stride + i].y := (alpha.x * h_x[b * stride + i].y + alpha.y * h_x[b * stride + i].x) + h_y[b * stride + i].y;
    end;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_double_complex));

  // 3. Copy Data
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_zaxpy_strided_batched(handle, N, @alpha, d_x, incx, stride, d_y, incy, stride, batch_count);
  AssertEquals('rocblas_zaxpy_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy back
  hipMemcpy(@h_y[0], d_y, total_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b*stride+i].x, h_y[b*stride+i].x, 1e-12, 'ZAXPY_Strided real mismatch');
      CheckEquals(h_y_expected[b*stride+i].y, h_y[b*stride+i].y, 1e-12, 'ZAXPY_Strided imag mismatch');
    end;
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_SCOPY_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride_x = 15;
  stride_y = 20;
  size_x = stride_x * (batch_count - 1) + N; // 40
  size_y = stride_y * (batch_count - 1) + N; // 50
type
  TSingleVectorX = array[0..39] of Single;
  TSingleVectorY = array[0..49] of Single;
var
  h_x: TSingleVectorX;
  h_y, h_y_expected: TSingleVectorY;
  d_x, d_y: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b * stride_x + i * incx] := (b + 1) * 10.0 + (i + 1) / 10.0;
      h_y[b * stride_y + i * incy] := 0.0; // Initialize destination
      h_y_expected[b * stride_y + i * incy] := h_x[b * stride_x + i * incx];
    end;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, size_x * SizeOf(Single));
  hipMalloc(d_y, size_y * SizeOf(Single));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], size_x * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], size_y * SizeOf(Single), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_scopy_strided_batched(handle, N, d_x, incx, stride_x, d_y, incy, stride_y, batch_count);
  AssertEquals('rocblas_scopy_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_y[0], d_y, size_y * SizeOf(Single), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b * stride_y + i * incy], h_y[b * stride_y + i * incy], 1e-6,
        Format('SCOPY_Strided_Batched mismatch in batch %d, index %d', [b, i]));
    end;
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_DCOPY_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TDoubleVector = array[0..41] of Double;
var
  h_x, h_y, h_y_expected: TDoubleVector;
  d_x, d_y: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b * stride + i] := (b + 1) * 100.0 + (i + 1) / 100.0;
      h_y[b * stride + i] := -1.0;
      h_y_expected[b * stride + i] := h_x[b * stride + i];
    end;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(Double));
  hipMalloc(d_y, total_size * SizeOf(Double));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_dcopy_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, batch_count);
  AssertEquals('rocblas_dcopy_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_y[0], d_y, total_size * SizeOf(Double), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b * stride + i], h_y[b * stride + i], 1e-12,
        Format('DCOPY_Strided_Batched mismatch in batch %d, index %d', [b, i]));
    end;
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_CCOPY_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count-1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_float_complex;
var
  h_x, h_y, h_y_expected: TComplexVector;
  d_x, d_y: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b * stride + i].x := (b + 1) + i;
      h_x[b * stride + i].y := (b + 1) - i;
      h_y[b * stride + i].x := 0.0;
      h_y[b * stride + i].y := 0.0;
      h_y_expected[b * stride + i] := h_x[b * stride + i];
    end;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_float_complex));

  // 3. Copy Data
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_ccopy_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, batch_count);
  AssertEquals('rocblas_ccopy_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy back
  hipMemcpy(@h_y[0], d_y, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b*stride+i].x, h_y[b*stride+i].x, 1e-6, 'CCOPY_Strided real mismatch');
      CheckEquals(h_y_expected[b*stride+i].y, h_y[b*stride+i].y, 1e-6, 'CCOPY_Strided imag mismatch');
    end;
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ZCOPY_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count-1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_double_complex;
var
  h_x, h_y, h_y_expected: TComplexVector;
  d_x, d_y: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b * stride + i].x := (b + 1.5) + i;
      h_x[b * stride + i].y := (b + 1.5) - i;
      h_y[b * stride + i].x := -1.0;
      h_y[b * stride + i].y := -1.0;
      h_y_expected[b * stride + i] := h_x[b * stride + i];
    end;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_double_complex));

  // 3. Copy Data
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_zcopy_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, batch_count);
  AssertEquals('rocblas_zcopy_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy back
  hipMemcpy(@h_y[0], d_y, total_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_expected[b*stride+i].x, h_y[b*stride+i].x, 1e-12, 'ZCOPY_Strided real mismatch');
      CheckEquals(h_y_expected[b*stride+i].y, h_y[b*stride+i].y, 1e-12, 'ZCOPY_Strided imag mismatch');
    end;
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_SSCAL_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  alpha: Single = 0.5;
  stride_x = 15;
  size_x = stride_x * (batch_count - 1) + N; // 40
type
  TSingleVector = array[0..39] of Single;
var
  h_x, h_x_expected: TSingleVector;
  d_x: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data and Expected Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b * stride_x + i * incx] := (b + 1) * 10.0 + i;
      h_x_expected[b * stride_x + i * incx] := alpha * h_x[b * stride_x + i * incx];
    end;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, size_x * SizeOf(Single));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], size_x * SizeOf(Single), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_sscal_strided_batched(handle, N, @alpha, d_x, incx, stride_x, batch_count);
  AssertEquals('rocblas_sscal_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_x[0], d_x, size_x * SizeOf(Single), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b * stride_x + i * incx], h_x[b * stride_x + i * incx], 1e-6,
        Format('SSCAL_Strided_Batched mismatch in batch %d, index %d', [b, i]));
    end;
  end;

  // 7. Cleanup
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_DSCAL_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  alpha: Double = 1.2345;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TDoubleVector = array[0..41] of Double;
var
  h_x, h_x_expected: TDoubleVector;
  d_x: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b * stride + i] := (b + 1.5) * 20.0 + i;
      h_x_expected[b * stride + i] := alpha * h_x[b * stride + i];
    end;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(Double));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_dscal_strided_batched(handle, N, @alpha, d_x, incx, stride, batch_count);
  AssertEquals('rocblas_dscal_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_x[0], d_x, total_size * SizeOf(Double), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b * stride + i], h_x[b * stride + i], 1e-12,
        Format('DSCAL_Strided_Batched mismatch in batch %d, index %d', [b, i]));
    end;
  end;

  // 7. Cleanup
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_CSCAL_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 12;
  total_size = stride * (batch_count-1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_float_complex;
var
  h_x, h_x_expected: TComplexVector;
  alpha: rocblas_float_complex;
  d_x: hipDeviceptr_t;
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
      h_x[b * stride + i].x := (b + 1) + i;
      h_x[b * stride + i].y := (b + 1) - i;
      // Expected: x = alpha * x
      h_x_expected[b*stride+i].x := alpha.x * h_x[b*stride+i].x - alpha.y * h_x[b*stride+i].y;
      h_x_expected[b*stride+i].y := alpha.x * h_x[b*stride+i].y + alpha.y * h_x[b*stride+i].x;
    end;
  end;

  // 2. Allocate and Copy
  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 3. Call rocBLAS
  blas_status := rocblas_cscal_strided_batched(handle, N, @alpha, d_x, incx, stride, batch_count);
  if blas_status = rocblas_status_not_implemented then
  begin
    Ignore('Function rocblas_cscal_strided_batched is not implemented in this rocBLAS version.');
  end
  else
  begin
      AssertEquals('rocblas_cscal_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));
      hipMemcpy(@h_x[0], d_x, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
      // 4. Verify
      for b := 0 to batch_count - 1 do
        for i := 0 to N - 1 do
        begin
          CheckEquals(h_x_expected[b*stride+i].x, h_x[b*stride+i].x, 1e-5, 'CSCAL_Strided real mismatch');
          CheckEquals(h_x_expected[b*stride+i].y, h_x[b*stride+i].y, 1e-5, 'CSCAL_Strided imag mismatch');
        end;
  end;

  // 5. Cleanup
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ZSCAL_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 12;
  total_size = stride * (batch_count-1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_double_complex;
var
  h_x, h_x_expected: TComplexVector;
  alpha: rocblas_double_complex;
  d_x: hipDeviceptr_t;
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
      h_x[b * stride + i].x := (b + 0.5) + i;
      h_x[b * stride + i].y := (b + 0.5) - i;
      h_x_expected[b*stride+i].x := alpha.x * h_x[b*stride+i].x - alpha.y * h_x[b*stride+i].y;
      h_x_expected[b*stride+i].y := alpha.x * h_x[b*stride+i].y + alpha.y * h_x[b*stride+i].x;
    end;
  end;

  // 2. Allocate and Copy
  hipMalloc(d_x, total_size * SizeOf(rocblas_double_complex));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  // 3. Call rocBLAS
  blas_status := rocblas_zscal_strided_batched(handle, N, @alpha, d_x, incx, stride, batch_count);
  if blas_status = rocblas_status_not_implemented then
  begin
    Ignore('Function rocblas_zscal_strided_batched is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_zscal_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_x[0], d_x, total_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    // 4. Verify
    for b := 0 to batch_count - 1 do
      for i := 0 to N - 1 do
      begin
        CheckEquals(h_x_expected[b*stride+i].x, h_x[b*stride+i].x, 1e-12, 'ZSCAL_Strided real mismatch');
        CheckEquals(h_x_expected[b*stride+i].y, h_x[b*stride+i].y, 1e-12, 'ZSCAL_Strided imag mismatch');
      end;
  end;

  // 5. Cleanup
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ZROT_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_double_complex;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TComplexVector;
  d_x, d_y, d_s: hipDeviceptr_t;
  d_c: hipDeviceptr_t; // c is PDouble
  c: Double;
  s: rocblas_double_complex;
  b, i: Integer;
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
  hipMalloc(d_x, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_c, SizeOf(Double));
  hipMalloc(d_s, SizeOf(rocblas_double_complex));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_c, @c, SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_s, @s, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_zrot_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, d_c, d_s, batch_count);
  if blas_status = rocblas_status_not_implemented then
  begin
    Ignore('Function rocblas_zrot_strided_batched is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_zrot_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_x[0], d_x, total_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[0], d_y, total_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    for b := 0 to batch_count - 1 do
      for i := 0 to N-1 do
      begin
        CheckEquals(h_x_expected[b*stride+i].x, h_x[b*stride+i].x, 1e-12, 'ZROT_Strided x.real mismatch');
        CheckEquals(h_x_expected[b*stride+i].y, h_x[b*stride+i].y, 1e-12, 'ZROT_Strided x.imag mismatch');
        CheckEquals(h_y_expected[b*stride+i].x, h_y[b*stride+i].x, 1e-12, 'ZROT_Strided y.real mismatch');
        CheckEquals(h_y_expected[b*stride+i].y, h_y[b*stride+i].y, 1e-12, 'ZROT_Strided y.imag mismatch');
      end;
  end;

  // 4. Cleanup
  hipFree(d_x); hipFree(d_y); hipFree(d_c); hipFree(d_s);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ZDROT_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_double_complex;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TComplexVector;
  d_x, d_y, d_c, d_s: hipDeviceptr_t;
  c, s: Double;
  b, i: Integer;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize
  c := Sqrt(2.0)/2.0;
  s := Sqrt(2.0)/2.0;
  for b := 0 to batch_count - 1 do
    for i := 0 to N-1 do
    begin
      h_x[b*stride+i].x := b + 1.5 + i*0.15; h_x[b*stride+i].y := b + 1.5 - i*0.15;
      h_y[b*stride+i].x := b + 2.5 + i*0.25; h_y[b*stride+i].y := b + 2.5 - i*0.25;
      h_x_orig[b*stride+i] := h_x[b*stride+i];
      h_y_orig[b*stride+i] := h_y[b*stride+i];
      h_x_expected[b*stride+i].x := c * h_x_orig[b*stride+i].x + s * h_y_orig[b*stride+i].x;
      h_x_expected[b*stride+i].y := c * h_x_orig[b*stride+i].y + s * h_y_orig[b*stride+i].y;
      h_y_expected[b*stride+i].x := -s * h_x_orig[b*stride+i].x + c * h_y_orig[b*stride+i].x;
      h_y_expected[b*stride+i].y := -s * h_x_orig[b*stride+i].y + c * h_y_orig[b*stride+i].y;
    end;

  // 2. Allocate and Copy
  hipMalloc(d_x, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_c, SizeOf(Double));
  hipMalloc(d_s, SizeOf(Double));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_c, @c, SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_s, @s, SizeOf(Double), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_zdrot_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, d_c, d_s, batch_count);
  if blas_status = rocblas_status_not_implemented then
  begin
    Ignore('Function rocblas_zdrot_strided_batched is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_zdrot_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_x[0], d_x, total_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[0], d_y, total_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
    for b := 0 to batch_count - 1 do
      for i := 0 to N - 1 do
      begin
        CheckEquals(h_x_expected[b*stride+i].x, h_x[b*stride+i].x, 1e-12, 'ZDROT_Strided x.real mismatch');
        CheckEquals(h_x_expected[b*stride+i].y, h_x[b*stride+i].y, 1e-12, 'ZDROT_Strided x.imag mismatch');
        CheckEquals(h_y_expected[b*stride+i].x, h_y[b*stride+i].x, 1e-12, 'ZDROT_Strided y.real mismatch');
        CheckEquals(h_y_expected[b*stride+i].y, h_y[b*stride+i].y, 1e-12, 'ZDROT_Strided y.imag mismatch');
      end;
  end;

  // 4. Cleanup
  hipFree(d_x); hipFree(d_y); hipFree(d_c); hipFree(d_s);
end;
procedure TTestRocBLAS_Level1_Strided_Batched.Test_CSSCAL_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  alpha: Single = 2.5;
  stride = 12;
  total_size = stride * (batch_count-1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_float_complex;
var
  h_x, h_x_expected: TComplexVector;
  d_x: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b * stride + i].x := (b + 1) + i;
      h_x[b * stride + i].y := (b + 1) - i;
      h_x_expected[b * stride + i].x := alpha * h_x[b * stride + i].x;
      h_x_expected[b * stride + i].y := alpha * h_x[b * stride + i].y;
    end;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_csscal_strided_batched(handle, N, @alpha, d_x, incx, stride, batch_count);
  AssertEquals('rocblas_csscal_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_x[0], d_x, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b*stride+i].x, h_x[b*stride+i].x, 1e-6, 'CSSCAL_Strided real mismatch');
      CheckEquals(h_x_expected[b*stride+i].y, h_x[b*stride+i].y, 1e-6, 'CSSCAL_Strided imag mismatch');
    end;
  end;

  // 7. Cleanup
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ZDSCAL_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  alpha: Double = -1.5;
  stride = 12;
  total_size = stride * (batch_count-1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_double_complex;
var
  h_x, h_x_expected: TComplexVector;
  d_x: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b * stride + i].x := (b + 0.5) + i;
      h_x[b * stride + i].y := (b + 0.5) - i;
      h_x_expected[b * stride + i].x := alpha * h_x[b * stride + i].x;
      h_x_expected[b * stride + i].y := alpha * h_x[b * stride + i].y;
    end;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(rocblas_double_complex));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_zdscal_strided_batched(handle, N, @alpha, d_x, incx, stride, batch_count);
  AssertEquals('rocblas_zdscal_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_x[0], d_x, total_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b*stride+i].x, h_x[b*stride+i].x, 1e-12, 'ZDSCAL_Strided real mismatch');
      CheckEquals(h_x_expected[b*stride+i].y, h_x[b*stride+i].y, 1e-12, 'ZDSCAL_Strided imag mismatch');
    end;
  end;

  // 7. Cleanup
  hipFree(d_x);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_SDOT_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TSingleVector = array[0..41] of Single;
  TResultVector = array[0..2] of Single;
var
  h_x, h_y: TSingleVector;
  h_result, h_expected_result: TResultVector;
  d_x, d_y, d_result: hipDeviceptr_t;
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
      h_x[b * stride + i] := (b + 1) + i / 10.0;
      h_y[b * stride + i] := (b + 1) * 2 - i / 10.0;
      temp_sum := temp_sum + h_x[b * stride + i] * h_y[b * stride + i];
    end;
    h_expected_result[b] := temp_sum;
    h_result[b] := 0.0;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(Single));
  hipMalloc(d_y, total_size * SizeOf(Single));
  hipMalloc(d_result, batch_count * SizeOf(Single));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(Single), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_sdot_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, batch_count, d_result);
  AssertEquals('rocblas_sdot_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_result[0], d_result, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_result[b], h_result[b], 1e-4, Format('SDOT_Strided_Batched mismatch batch %d', [b]));
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_DDOT_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TDoubleVector = array[0..41] of Double;
  TResultVector = array[0..2] of Double;
var
  h_x, h_y: TDoubleVector;
  h_result, h_expected_result: TResultVector;
  d_x, d_y, d_result: hipDeviceptr_t;
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
      h_x[b * stride + i] := (b + 1.5) + i / 20.0;
      h_y[b * stride + i] := (b + 1.5) * 3 - i / 20.0;
      temp_sum := temp_sum + h_x[b * stride + i] * h_y[b * stride + i];
    end;
    h_expected_result[b] := temp_sum;
    h_result[b] := 0.0;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(Double));
  hipMalloc(d_y, total_size * SizeOf(Double));
  hipMalloc(d_result, batch_count * SizeOf(Double));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_ddot_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, batch_count, d_result);
  AssertEquals('rocblas_ddot_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_result[0], d_result, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_result[b], h_result[b], 1e-12, Format('DDOT_Strided_Batched mismatch batch %d', [b]));
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_CDOTU_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_float_complex;
  TResultVector = array[0..2] of rocblas_float_complex;
var
  h_x, h_y: TComplexVector;
  h_result, h_expected_result: TResultVector;
  d_x, d_y, d_result: hipDeviceptr_t;
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
      h_x[b*stride+i].x := (b + 1) + i * 0.1; h_x[b*stride+i].y := (b + 1) - i * 0.1;
      h_y[b*stride+i].x := i * 0.2 - b;      h_y[b*stride+i].y := i * 0.3 + b;
      sum_re := sum_re + (h_x[b*stride+i].x * h_y[b*stride+i].x - h_x[b*stride+i].y * h_y[b*stride+i].y);
      sum_im := sum_im + (h_x[b*stride+i].x * h_y[b*stride+i].y + h_x[b*stride+i].y * h_y[b*stride+i].x);
    end;
    h_expected_result[b].x := sum_re;
    h_expected_result[b].y := sum_im;
    h_result[b].x := 0.0; h_result[b].y := 0.0;
  end;

  // 2. Allocate
  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_result, batch_count * SizeOf(rocblas_float_complex));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 4. Call
  blas_status := rocblas_cdotu_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, batch_count, d_result);
  AssertEquals('rocblas_cdotu_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_result[0], d_result, batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_result[b].x, h_result[b].x, 1e-4, 'CDOTU_Strided real mismatch');
    CheckEquals(h_expected_result[b].y, h_result[b].y, 1e-4, 'CDOTU_Strided imag mismatch');
  end;

  // 7. Cleanup
  hipFree(d_x); hipFree(d_y); hipFree(d_result);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ZDOTU_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_double_complex;
  TResultVector = array[0..2] of rocblas_double_complex;
var
  h_x, h_y: TComplexVector;
  h_result, h_expected_result: TResultVector;
  d_x, d_y, d_result: hipDeviceptr_t;
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
      h_x[b*stride+i].x := (b + 1.1) + i * 0.11; h_x[b*stride+i].y := (b + 1.1) - i * 0.11;
      h_y[b*stride+i].x := i * 0.22 - b;      h_y[b*stride+i].y := i * 0.33 + b;
      sum_re := sum_re + (h_x[b*stride+i].x * h_y[b*stride+i].x - h_x[b*stride+i].y * h_y[b*stride+i].y);
      sum_im := sum_im + (h_x[b*stride+i].x * h_y[b*stride+i].y + h_x[b*stride+i].y * h_y[b*stride+i].x);
    end;
    h_expected_result[b].x := sum_re;
    h_expected_result[b].y := sum_im;
    h_result[b].x := 0.0; h_result[b].y := 0.0;
  end;

  // 2. Allocate
  hipMalloc(d_x, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_result, batch_count * SizeOf(rocblas_double_complex));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  // 4. Call
  blas_status := rocblas_zdotu_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, batch_count, d_result);
  AssertEquals('rocblas_zdotu_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_result[0], d_result, batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_result[b].x, h_result[b].x, 1e-12, 'ZDOTU_Strided real mismatch');
    CheckEquals(h_expected_result[b].y, h_result[b].y, 1e-12, 'ZDOTU_Strided imag mismatch');
  end;

  // 7. Cleanup
  hipFree(d_x); hipFree(d_y); hipFree(d_result);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_CDOTC_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_float_complex;
  TResultVector = array[0..2] of rocblas_float_complex;
var
  h_x, h_y: TComplexVector;
  h_result, h_expected_result: TResultVector;
  d_x, d_y, d_result: hipDeviceptr_t;
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
      h_x[b*stride+i].x := (b + 1) + i * 0.1; h_x[b*stride+i].y := (b + 1) - i * 0.1;
      h_y[b*stride+i].x := i * 0.2 - b;      h_y[b*stride+i].y := i * 0.3 + b;
      // dotc = sum(conj(x)*y)
      sum_re := sum_re + (h_x[b*stride+i].x * h_y[b*stride+i].x + h_x[b*stride+i].y * h_y[b*stride+i].y);
      sum_im := sum_im + (h_x[b*stride+i].x * h_y[b*stride+i].y - h_x[b*stride+i].y * h_y[b*stride+i].x);
    end;
    h_expected_result[b].x := sum_re;
    h_expected_result[b].y := sum_im;
    h_result[b].x := 0.0; h_result[b].y := 0.0;
  end;

  // ... Allocation, Copy, Cleanup same as CDOTU ...
  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_result, batch_count * SizeOf(rocblas_float_complex));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 4. Call
  blas_status := rocblas_cdotc_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, batch_count, d_result);
  AssertEquals('rocblas_cdotc_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_result[0], d_result, batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_result[b].x, h_result[b].x, 1e-4, 'CDOTC_Strided real mismatch');
    CheckEquals(h_expected_result[b].y, h_result[b].y, 1e-4, 'CDOTC_Strided imag mismatch');
  end;

  // 7. Cleanup
  hipFree(d_x); hipFree(d_y); hipFree(d_result);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ZDOTC_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_double_complex;
  TResultVector = array[0..2] of rocblas_double_complex;
var
  h_x, h_y: TComplexVector;
  h_result, h_expected_result: TResultVector;
  d_x, d_y, d_result: hipDeviceptr_t;
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
      h_x[b*stride+i].x := (b + 1.1) + i * 0.11; h_x[b*stride+i].y := (b + 1.1) - i * 0.11;
      h_y[b*stride+i].x := i * 0.22 - b;      h_y[b*stride+i].y := i * 0.33 + b;
      // dotc = sum(conj(x)*y)
      sum_re := sum_re + (h_x[b*stride+i].x * h_y[b*stride+i].x + h_x[b*stride+i].y * h_y[b*stride+i].y);
      sum_im := sum_im + (h_x[b*stride+i].x * h_y[b*stride+i].y - h_x[b*stride+i].y * h_y[b*stride+i].x);
    end;
    h_expected_result[b].x := sum_re;
    h_expected_result[b].y := sum_im;
    h_result[b].x := 0.0; h_result[b].y := 0.0;
  end;

  // ... Allocation, Copy, Cleanup same as ZDOTU ...
  hipMalloc(d_x, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_result, batch_count * SizeOf(rocblas_double_complex));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  // 4. Call
  blas_status := rocblas_zdotc_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, batch_count, d_result);
  AssertEquals('rocblas_zdotc_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_result[0], d_result, batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_result[b].x, h_result[b].x, 1e-12, 'ZDOTC_Strided real mismatch');
    CheckEquals(h_expected_result[b].y, h_result[b].y, 1e-12, 'ZDOTC_Strided imag mismatch');
  end;

  // 7. Cleanup
  hipFree(d_x); hipFree(d_y); hipFree(d_result);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_SNRM2_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride_x = 15;
  size_x = stride_x * (batch_count - 1) + N; // 40
type
  TSingleVector = array[0..39] of Single;
  TResultVector = array[0..2] of Single;
var
  h_x: TSingleVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data and Expected Results
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b * stride_x + i * incx] := Power(-1, i) * (b + 1 + i*0.2);
    end;
    h_expected_results[b] := CalculateRealNorm(N, PSingle(@h_x[b * stride_x]), incx);
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, size_x * SizeOf(Single));
  hipMalloc(d_results, batch_count * SizeOf(Single));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], size_x * SizeOf(Single), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_snrm2_strided_batched(handle, N, d_x, incx, stride_x, batch_count, d_results);
  AssertEquals('rocblas_snrm2_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-5,
      Format('SNRM2_Strided_Batched mismatch in batch %d', [b]));
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_DNRM2_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TDoubleVector = array[0..41] of Double;
  TResultVector = array[0..2] of Double;
var
  h_x: TDoubleVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b * stride + i] := Power(-1, i) * (b + 1.5 + i*0.25);
    end;
    h_expected_results[b] := CalculateRealNorm(N, PDouble(@h_x[b * stride]), incx);
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(Double));
  hipMalloc(d_results, batch_count * SizeOf(Double));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_dnrm2_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_dnrm2_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-12,
      Format('DNRM2_Strided_Batched mismatch in batch %d', [b]));
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_SCNRM2_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_float_complex;
  TResultVector = array[0..2] of Single;
var
  h_x: TComplexVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b*stride+i].x := (b + 1) + i * 0.1;
      h_x[b*stride+i].y := Power(-1, i) * ((b + 1) - i * 0.1);
    end;
    h_expected_results[b] := CalculateComplexNorm(N, Procfloat_complex(@h_x[b*stride]), incx);
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_results, batch_count * SizeOf(Single));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_scnrm2_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_scnrm2_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-5,
      Format('SCNRM2_Strided_Batched mismatch in batch %d', [b]));
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_DZNRM2_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_double_complex;
  TResultVector = array[0..2] of Double;
var
  h_x: TComplexVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b*stride+i].x := (b + 1.5) + i * 0.15;
      h_x[b*stride+i].y := Power(-1, i) * ((b + 1.5) - i * 0.15);
    end;
    h_expected_results[b] := CalculateComplexNorm(N, Procdouble_complex(@h_x[b*stride]), incx);
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_results, batch_count * SizeOf(Double));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_dznrm2_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_dznrm2_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-12,
      Format('DZNRM2_Strided_Batched mismatch in batch %d', [b]));
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_SASUM_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride_x = 15;
  size_x = stride_x * (batch_count - 1) + N; // 40
type
  TSingleVector = array[0..39] of Single;
  TResultVector = array[0..2] of Single;
var
  h_x: TSingleVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
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
      h_x[b * stride_x + i * incx] := Power(-1, i) * (b + 1 + i*0.2);
      temp_sum := temp_sum + Abs(h_x[b * stride_x + i * incx]);
    end;
    h_expected_results[b] := temp_sum;
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, size_x * SizeOf(Single));
  hipMalloc(d_results, batch_count * SizeOf(Single));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], size_x * SizeOf(Single), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_sasum_strided_batched(handle, N, d_x, incx, stride_x, batch_count, d_results);
  AssertEquals('rocblas_sasum_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-5,
      Format('SASUM_Strided_Batched mismatch in batch %d', [b]));
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_DASUM_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TDoubleVector = array[0..41] of Double;
  TResultVector = array[0..2] of Double;
var
  h_x: TDoubleVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
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
      h_x[b * stride + i] := Power(-1, i) * (b + 1.5 + i*0.25);
      temp_sum := temp_sum + Abs(h_x[b * stride + i]);
    end;
    h_expected_results[b] := temp_sum;
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(Double));
  hipMalloc(d_results, batch_count * SizeOf(Double));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_dasum_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_dasum_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-12,
      Format('DASUM_Strided_Batched mismatch in batch %d', [b]));
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_SCASUM_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_float_complex;
  TResultVector = array[0..2] of Single;
var
  h_x: TComplexVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
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
      h_x[b*stride+i].x := Power(-1, i) * ((b + 1) + i * 0.1);
      h_x[b*stride+i].y := Power(-1, i+1) * ((b + 1) - i * 0.1);
      temp_sum := temp_sum + Abs(h_x[b*stride+i].x) + Abs(h_x[b*stride+i].y);
    end;
    h_expected_results[b] := temp_sum;
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_results, batch_count * SizeOf(Single));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_scasum_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_scasum_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-5,
      Format('SCASUM_Strided_Batched mismatch in batch %d', [b]));
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_DZASUM_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_double_complex;
  TResultVector = array[0..2] of Double;
var
  h_x: TComplexVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
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
      h_x[b*stride+i].x := Power(-1, i) * ((b + 1.5) + i * 0.15);
      h_x[b*stride+i].y := Power(-1, i+1) * ((b + 1.5) - i * 0.15);
      temp_sum := temp_sum + Abs(h_x[b*stride+i].x) + Abs(h_x[b*stride+i].y);
    end;
    h_expected_results[b] := temp_sum;
    h_results[b] := -1.0;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_results, batch_count * SizeOf(Double));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_dzasum_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_dzasum_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(h_expected_results[b], h_results[b], 1e-12,
      Format('DZASUM_Strided_Batched mismatch in batch %d', [b]));
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ISAMAX_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TSingleVector = array[0..41] of Single;
  TResultVector = array[0..2] of Int32;
var
  h_x: TSingleVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
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
      // Make the max value unique for each batch, e.g., at index b+i
      h_x[b * stride + i] := Power(-1, i) * (b + 1 + i);
      if Abs(h_x[b * stride + i]) > max_val then
      begin
        max_val := Abs(h_x[b * stride + i]);
        max_idx := i + 1; // 1-based index
      end;
    end;
    h_expected_results[b] := max_idx;
    h_results[b] := -1;
  end;

  // 2. Allocate Device Memory
  hipMalloc(d_x, total_size * SizeOf(Single));
  hipMalloc(d_results, batch_count * SizeOf(Int32));

  // 3. Copy H2D
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Single), hipMemcpyHostToDevice);

  // 4. Call rocBLAS
  blas_status := rocblas_isamax_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_isamax_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 5. Copy D2H
  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  // 6. Verify
  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('ISAMAX_Strided_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  // 7. Cleanup
  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_IDAMAX_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TDoubleVector = array[0..41] of Double;
  TResultVector = array[0..2] of Int32;
var
  h_x: TDoubleVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
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
      h_x[b * stride + i] := Power(-1, i) * (b + 1.5 + i);
      if Abs(h_x[b * stride + i]) > max_val then
      begin
        max_val := Abs(h_x[b * stride + i]);
        max_idx := i + 1;
      end;
    end;
    h_expected_results[b] := max_idx;
    h_results[b] := -1;
  end;

  // 2. Allocate, Copy, Call, Verify...
  hipMalloc(d_x, total_size * SizeOf(Double));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);

  blas_status := rocblas_idamax_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_idamax_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('IDAMAX_Strided_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ICAMAX_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_float_complex;
  TResultVector = array[0..2] of Int32;
var
  h_x: TComplexVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
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
      h_x[b*stride+i].x := Power(-1, i) * (b + 1 + i*0.1);
      h_x[b*stride+i].y := Power(-1, i+1) * (b + 1 - i*0.1);
      current_val := Abs(h_x[b*stride+i].x) + Abs(h_x[b*stride+i].y);
      if current_val > max_val then
      begin
        max_val := current_val;
        max_idx := i + 1;
      end;
    end;
    h_expected_results[b] := max_idx;
    h_results[b] := -1;
  end;

  // 2. Allocate, Copy, Call, Verify...
  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  blas_status := rocblas_icamax_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_icamax_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('ICAMAX_Strided_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_IZAMAX_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_double_complex;
  TResultVector = array[0..2] of Int32;
var
  h_x: TComplexVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
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
      h_x[b*stride+i].x := Power(-1, i) * (b + 1.5 + i*0.15);
      h_x[b*stride+i].y := Power(-1, i+1) * (b + 1.5 - i*0.15);
      current_val := Abs(h_x[b*stride+i].x) + Abs(h_x[b*stride+i].y);
      if current_val > max_val then
      begin
        max_val := current_val;
        max_idx := i + 1;
      end;
    end;
    h_expected_results[b] := max_idx;
    h_results[b] := -1;
  end;

  // 2. Allocate, Copy, Call, Verify...
  hipMalloc(d_x, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  blas_status := rocblas_izamax_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_izamax_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('IZAMAX_Strided_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ISAMIN_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TSingleVector = array[0..41] of Single;
  TResultVector = array[0..2] of Int32;
var
  h_x: TSingleVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
  b, i: Integer;
  min_val: Single;
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
      if i = b then
         h_x[b * stride + i] := 0.1
      else
         h_x[b * stride + i] := Power(-1, i) * (b + 1 + i);

      if Abs(h_x[b * stride + i]) < min_val then
      begin
        min_val := Abs(h_x[b * stride + i]);
        min_idx := i + 1;
      end;
    end;
    h_expected_results[b] := min_idx;
    h_results[b] := -1;
  end;

  // 2. Allocate, Copy, Call, Verify...
  hipMalloc(d_x, total_size * SizeOf(Single));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Single), hipMemcpyHostToDevice);

  blas_status := rocblas_isamin_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_isamin_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('ISAMIN_Strided_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_IDAMIN_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TDoubleVector = array[0..41] of Double;
  TResultVector = array[0..2] of Int32;
var
  h_x: TDoubleVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
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
         h_x[b * stride + i] := 0.01
       else
         h_x[b * stride + i] := Power(-1, i) * (b + 1.5 + i);

      if Abs(h_x[b * stride + i]) < min_val then
      begin
        min_val := Abs(h_x[b * stride + i]);
        min_idx := i + 1;
      end;
    end;
    h_expected_results[b] := min_idx;
    h_results[b] := -1;
  end;

  // 2. Allocate, Copy, Call, Verify...
  hipMalloc(d_x, total_size * SizeOf(Double));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);

  blas_status := rocblas_idamin_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_idamin_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('IDAMIN_Strided_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ICAMIN_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_float_complex;
  TResultVector = array[0..2] of Int32;
var
  h_x: TComplexVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
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
        h_x[b*stride+i].x := 0.01;
        h_x[b*stride+i].y := -0.01;
      end
      else
      begin
        h_x[b*stride+i].x := Power(-1, i) * (b + 1 + i*0.1);
        h_x[b*stride+i].y := Power(-1, i+1) * (b + 1 - i*0.1);
      end;
      current_val := Abs(h_x[b*stride+i].x) + Abs(h_x[b*stride+i].y);
      if current_val < min_val then
      begin
        min_val := current_val;
        min_idx := i + 1;
      end;
    end;
    h_expected_results[b] := min_idx;
    h_results[b] := -1;
  end;

  // 2. Allocate, Copy, Call, Verify...
  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  blas_status := rocblas_icamin_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_icamin_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('ICAMIN_Strided_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_IZAMIN_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_double_complex;
  TResultVector = array[0..2] of Int32;
var
  h_x: TComplexVector;
  h_results, h_expected_results: TResultVector;
  d_x, d_results: hipDeviceptr_t;
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
        h_x[b*stride+i].x := 0.001;
        h_x[b*stride+i].y := 0.001;
      end
      else
      begin
        h_x[b*stride+i].x := Power(-1, i) * (b + 1.5 + i*0.15);
        h_x[b*stride+i].y := Power(-1, i+1) * (b + 1.5 - i*0.15);
      end;
      current_val := Abs(h_x[b*stride+i].x) + Abs(h_x[b*stride+i].y);
      if current_val < min_val then
      begin
        min_val := current_val;
        min_idx := i + 1;
      end;
    end;
    h_expected_results[b] := min_idx;
    h_results[b] := -1;
  end;

  // 2. Allocate, Copy, Call, Verify...
  hipMalloc(d_x, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_results, batch_count * SizeOf(Int32));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  blas_status := rocblas_izamin_strided_batched(handle, N, d_x, incx, stride, batch_count, d_results);
  AssertEquals('rocblas_izamin_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@h_results[0], d_results, batch_count * SizeOf(Int32), hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    AssertEquals(Format('IZAMIN_Strided_Batched mismatch in batch %d', [b]),
      h_expected_results[b], h_results[b]);
  end;

  hipFree(d_x);
  hipFree(d_results);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_SSWAP_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TSingleVector = array[0..41] of Single;
var
  h_x, h_y, h_x_orig, h_y_orig: TSingleVector;
  d_x, d_y: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b * stride + i] := (b + 1) * 10.0 + i;
      h_y[b * stride + i] := (b + 1) * 100.0 + i;
    end;
  end;
  h_x_orig := h_x;
  h_y_orig := h_y;

  // 2. Allocate and Copy Data
  hipMalloc(d_x, total_size * SizeOf(Single));
  hipMalloc(d_y, total_size * SizeOf(Single));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(Single), hipMemcpyHostToDevice);

  // 3. Call rocBLAS
  blas_status := rocblas_sswap_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, batch_count);
  AssertEquals('rocblas_sswap_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 4. Copy results back
  hipMemcpy(@h_x[0], d_x, total_size * SizeOf(Single), hipMemcpyDeviceToHost);
  hipMemcpy(@h_y[0], d_y, total_size * SizeOf(Single), hipMemcpyDeviceToHost);

  // 5. Verify
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_orig[b*stride+i], h_x[b*stride+i], 1e-6, 'SSWAP_Strided x mismatch');
      CheckEquals(h_x_orig[b*stride+i], h_y[b*stride+i], 1e-6, 'SSWAP_Strided y mismatch');
    end;
  end;

  // 6. Cleanup
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_DSWAP_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TDoubleVector = array[0..41] of Double;
var
  h_x, h_y, h_x_orig, h_y_orig: TDoubleVector;
  d_x, d_y: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // ... Logic identical to SSWAP, but with Double ...
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b * stride + i] := (b + 1) * 10.0 + i;
      h_y[b * stride + i] := (b + 1) * 100.0 + i;
    end;
  end;
  h_x_orig := h_x;
  h_y_orig := h_y;

  hipMalloc(d_x, total_size * SizeOf(Double));
  hipMalloc(d_y, total_size * SizeOf(Double));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);

  blas_status := rocblas_dswap_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, batch_count);
  AssertEquals('rocblas_dswap_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@h_x[0], d_x, total_size * SizeOf(Double), hipMemcpyDeviceToHost);
  hipMemcpy(@h_y[0], d_y, total_size * SizeOf(Double), hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_orig[b*stride+i], h_x[b*stride+i], 1e-12, 'DSWAP_Strided x mismatch');
      CheckEquals(h_x_orig[b*stride+i], h_y[b*stride+i], 1e-12, 'DSWAP_Strided y mismatch');
    end;
  end;

  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_CSWAP_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_float_complex;
var
  h_x, h_y, h_x_orig, h_y_orig: TComplexVector;
  d_x, d_y: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // ... Logic identical to SSWAP, but with rocblas_float_complex ...
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b*stride+i].x := (b + 1) * 10.0 + i;
      h_x[b*stride+i].y := (b + 1) * -1.0 - i;
      h_y[b*stride+i].x := (b + 1) * 100.0 + i;
      h_y[b*stride+i].y := (b + 1) * -10.0 - i;
    end;
  end;
  h_x_orig := h_x;
  h_y_orig := h_y;

  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_float_complex));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  blas_status := rocblas_cswap_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, batch_count);
  AssertEquals('rocblas_cswap_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@h_x[0], d_x, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  hipMemcpy(@h_y[0], d_y, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_orig[b*stride+i].x, h_x[b*stride+i].x, 1e-6, 'CSWAP_Strided x.real mismatch');
      CheckEquals(h_y_orig[b*stride+i].y, h_x[b*stride+i].y, 1e-6, 'CSWAP_Strided x.imag mismatch');
      CheckEquals(h_x_orig[b*stride+i].x, h_y[b*stride+i].x, 1e-6, 'CSWAP_Strided y.real mismatch');
      CheckEquals(h_x_orig[b*stride+i].y, h_y[b*stride+i].y, 1e-6, 'CSWAP_Strided y.imag mismatch');
    end;
  end;

  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ZSWAP_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_double_complex;
var
  h_x, h_y, h_x_orig, h_y_orig: TComplexVector;
  d_x, d_y: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // ... Logic identical to SSWAP, but with rocblas_double_complex ...
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b*stride+i].x := (b + 1) * 10.0 + i;
      h_x[b*stride+i].y := (b + 1) * -1.0 - i;
      h_y[b*stride+i].x := (b + 1) * 100.0 + i;
      h_y[b*stride+i].y := (b + 1) * -10.0 - i;
    end;
  end;
  h_x_orig := h_x;
  h_y_orig := h_y;

  hipMalloc(d_x, total_size * SizeOf(rocblas_double_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_double_complex));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  blas_status := rocblas_zswap_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, batch_count);
  AssertEquals('rocblas_zswap_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  hipMemcpy(@h_x[0], d_x, total_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  hipMemcpy(@h_y[0], d_y, total_size * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_y_orig[b*stride+i].x, h_x[b*stride+i].x, 1e-12, 'ZSWAP_Strided x.real mismatch');
      CheckEquals(h_y_orig[b*stride+i].y, h_x[b*stride+i].y, 1e-12, 'ZSWAP_Strided x.imag mismatch');
      CheckEquals(h_x_orig[b*stride+i].x, h_y[b*stride+i].x, 1e-12, 'ZSWAP_Strided y.real mismatch');
      CheckEquals(h_x_orig[b*stride+i].y, h_y[b*stride+i].y, 1e-12, 'ZSWAP_Strided y.imag mismatch');
    end;
  end;

  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_SROT_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TSingleVector = array[0..41] of Single;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TSingleVector;
  d_x, d_y: hipDeviceptr_t;
  d_c, d_s: hipDeviceptr_t; // Device pointers for scalars
  c, s: Single;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  c := 0.8; s := 0.6;
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to N - 1 do
    begin
      h_x[b*stride+i] := (b + 1) * 10.0 + i;
      h_y[b*stride+i] := (b + 1) * 20.0 - i;
      h_x_orig[b*stride+i] := h_x[b*stride+i];
      h_y_orig[b*stride+i] := h_y[b*stride+i];
      h_x_expected[b*stride+i] := c * h_x_orig[b*stride+i] + s * h_y_orig[b*stride+i];
      h_y_expected[b*stride+i] := -s * h_x_orig[b*stride+i] + c * h_y_orig[b*stride+i];
    end;
  end;

  // 2. Allocate and Copy Data
  hipMalloc(d_x, total_size * SizeOf(Single));
  hipMalloc(d_y, total_size * SizeOf(Single));
  hipMalloc(d_c, SizeOf(Single)); // Allocate for single scalars on device
  hipMalloc(d_s, SizeOf(Single));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_c, @c, SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_s, @s, SizeOf(Single), hipMemcpyHostToDevice);

  // 3. Set pointer mode and call rocBLAS
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_srot_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, d_c, d_s, batch_count);
  AssertEquals('rocblas_srot_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 4. Copy results back
  hipMemcpy(@h_x[0], d_x, total_size * SizeOf(Single), hipMemcpyDeviceToHost);
  hipMemcpy(@h_y[0], d_y, total_size * SizeOf(Single), hipMemcpyDeviceToHost);

  // 5. Verify
  for b := 0 to batch_count - 1 do
    for i := 0 to N - 1 do
    begin
      CheckEquals(h_x_expected[b*stride+i], h_x[b*stride+i], 1e-5, 'SROT_Strided x mismatch');
      CheckEquals(h_y_expected[b*stride+i], h_y[b*stride+i], 1e-5, 'SROT_Strided y mismatch');
    end;

  // 6. Cleanup
  hipFree(d_x); hipFree(d_y); hipFree(d_c); hipFree(d_s);
end;

//procedure TTestRocBLAS_Level1_Strided_Batched.Test_DROT_Strided_Batched;
//begin
//  WriteLn('Function does not exist in the rocBLAS API');
//end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_DROT_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 16;
  total_size = stride * (batch_count - 1) + N; // 42
type
  TDoubleVector = array[0..41] of Double;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TDoubleVector;
  d_x, d_y, d_c, d_s: hipDeviceptr_t;
  c, s: Double;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  c := Sqrt(2.0)/2.0;
  s := Sqrt(2.0)/2.0;
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
  hipMalloc(d_x, total_size * SizeOf(Double));
  hipMalloc(d_y, total_size * SizeOf(Double));
  hipMalloc(d_c, SizeOf(Double));
  hipMalloc(d_s, SizeOf(Double));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_c, @c, SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_s, @s, SizeOf(Double), hipMemcpyHostToDevice);

  // 3. Set pointer mode and call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_drot_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, d_c, d_s, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    Ignore('Function rocblas_drot_strided_batched is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_drot_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));
    // 4. Copy results back
    hipMemcpy(@h_x[0], d_x, total_size * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[0], d_y, total_size * SizeOf(Double), hipMemcpyDeviceToHost);

    // 5. Verify
    for b := 0 to batch_count - 1 do
      for i := 0 to N - 1 do
      begin
        CheckEquals(h_x_expected[b*stride+i], h_x[b*stride+i], 1e-12, 'DROT_Strided x mismatch');
        CheckEquals(h_y_expected[b*stride+i], h_y[b*stride+i], 1e-12, 'DROT_Strided y mismatch');
      end;
  end;

  // 6. Cleanup
  hipFree(d_x); hipFree(d_y); hipFree(d_c); hipFree(d_s);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_CROT_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_float_complex;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TComplexVector;
  d_x, d_y, d_s: hipDeviceptr_t;
  d_c: hipDeviceptr_t; // c is PSingle
  c: Single;
  s: rocblas_float_complex;
  b, i: Integer;
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
  hipMalloc(d_c, SizeOf(Single));
  hipMalloc(d_s, SizeOf(rocblas_float_complex));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_c, @c, SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_s, @s, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_crot_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, d_c, d_s, batch_count);
  if blas_status = rocblas_status_not_implemented then
  begin
    Ignore('Function rocblas_crot_strided_batched is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_crot_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_x[0], d_x, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[0], d_y, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    for b := 0 to batch_count - 1 do
      for i := 0 to N-1 do
      begin
        CheckEquals(h_x_expected[b*stride+i].x, h_x[b*stride+i].x, 1e-5, 'CROT_Strided x.real mismatch');
        CheckEquals(h_x_expected[b*stride+i].y, h_x[b*stride+i].y, 1e-5, 'CROT_Strided x.imag mismatch');
        CheckEquals(h_y_expected[b*stride+i].x, h_y[b*stride+i].x, 1e-5, 'CROT_Strided y.real mismatch');
        CheckEquals(h_y_expected[b*stride+i].y, h_y[b*stride+i].y, 1e-5, 'CROT_Strided y.imag mismatch');
      end;
  end;

  // 4. Cleanup
  hipFree(d_x); hipFree(d_y); hipFree(d_c); hipFree(d_s);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_CSROT_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride = 12;
  total_size = stride * (batch_count - 1) + N; // 34
type
  TComplexVector = array[0..33] of rocblas_float_complex;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TComplexVector;
  d_x, d_y, d_c, d_s: hipDeviceptr_t;
  c, s: Single;
  b, i: Integer;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize
  c := 0.8; s := 0.6;
  for b := 0 to batch_count - 1 do
    for i := 0 to N-1 do
    begin
      h_x[b*stride+i].x := b + 1 + i*0.1; h_x[b*stride+i].y := b + 1 - i*0.1;
      h_y[b*stride+i].x := b + 2 + i*0.2; h_y[b*stride+i].y := b + 2 - i*0.2;
      h_x_orig[b*stride+i] := h_x[b*stride+i]; h_y_orig[b*stride+i] := h_y[b*stride+i];
      h_x_expected[b*stride+i].x := c * h_x_orig[b*stride+i].x + s * h_y_orig[b*stride+i].x;
      h_x_expected[b*stride+i].y := c * h_x_orig[b*stride+i].y + s * h_y_orig[b*stride+i].y;
      h_y_expected[b*stride+i].x := -s * h_x_orig[b*stride+i].x + c * h_y_orig[b*stride+i].x;
      h_y_expected[b*stride+i].y := -s * h_x_orig[b*stride+i].y + c * h_y_orig[b*stride+i].y;
    end;

  // 2. Allocate and Copy
  hipMalloc(d_x, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_y, total_size * SizeOf(rocblas_float_complex));
  hipMalloc(d_c, SizeOf(Single));
  hipMalloc(d_s, SizeOf(Single));
  hipMemcpy(d_x, @h_x[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], total_size * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_c, @c, SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_s, @s, SizeOf(Single), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_csrot_strided_batched(handle, N, d_x, incx, stride, d_y, incy, stride, d_c, d_s, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    Ignore('Function rocblas_csrot_strided_batched is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_csrot_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));
    hipMemcpy(@h_x[0], d_x, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[0], d_y, total_size * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
    for b := 0 to batch_count - 1 do
      for i := 0 to N - 1 do
      begin
        CheckEquals(h_x_expected[b*stride+i].x, h_x[b*stride+i].x, 1e-5, 'CSROT_Strided x.real mismatch');
        CheckEquals(h_x_expected[b*stride+i].y, h_x[b*stride+i].y, 1e-5, 'CSROT_Strided x.imag mismatch');
        CheckEquals(h_y_expected[b*stride+i].x, h_y[b*stride+i].x, 1e-5, 'CSROT_Strided y.real mismatch');
        CheckEquals(h_y_expected[b*stride+i].y, h_y[b*stride+i].y, 1e-5, 'CSROT_Strided y.imag mismatch');
      end;
  end;

  // 4. Cleanup
  hipFree(d_x); hipFree(d_y); hipFree(d_c); hipFree(d_s);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_SROTG_Strided_Batched;
const
  batch_count = 3;
  stride = 1; // Strided ROTG typically operates on contiguous arrays of scalars
  total_size = stride * (batch_count - 1) + 1; // = 3 for stride=1
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

  blas_status := rocblas_srotg_strided_batched(handle, d_a, stride, d_b, stride, d_c, stride, d_s, stride, batch_count);
  AssertEquals('rocblas_srotg_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));

  // 4. Copy results back
  hipMemcpy(@h_a[0], d_a, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);
  hipMemcpy(@h_c[0], d_c, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);
  hipMemcpy(@h_s[0], d_s, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);

  // 5. Verify
  for b := 0 to batch_count - 1 do
  begin
    CheckEquals(1.0, Sqr(h_c[b]) + Sqr(h_s[b]), 1e-6, Format('SROTG_Strided orthogonality failed batch %d', [b]));
    CheckEquals(h_a[b], h_c[b] * h_a_orig[b] + h_s[b] * h_b_orig[b], 1e-6, Format('SROTG_Strided r value mismatch batch %d', [b]));
  end;

  // 6. Cleanup
  hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_s);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_DROTG_Strided_Batched;
const
  batch_count = 3;
  stride = 1;
  total_size = stride * (batch_count - 1) + 1; // 3
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

  // 3. Set pointer mode and Call rocBLAS
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_drotg_strided_batched(handle, d_a, stride, d_b, stride, d_c, stride, d_s, stride, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    Ignore('Function rocblas_drotg_strided_batched is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_drotg_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));
    // 4. Copy results back
    hipMemcpy(@h_a[0], d_a, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_c[0], d_c, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_s[0], d_s, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);

    // 5. Verify
    for b := 0 to batch_count - 1 do
    begin
      CheckEquals(1.0, Sqr(h_c[b]) + Sqr(h_s[b]), 1e-12, Format('DROTG_Strided orthogonality failed batch %d', [b]));
      CheckEquals(h_a[b], h_c[b] * h_a_orig[b] + h_s[b] * h_b_orig[b], 1e-12, Format('DROTG_Strided r value mismatch batch %d', [b]));
    end;
  end;

  // 6. Cleanup
  hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_s);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_CROTG_Strided_Batched;
const
  batch_count = 3;
  stride = 1;
  total_size = stride * (batch_count - 1) + 1; // 3
type
  TComplexVector = array[0..2] of rocblas_float_complex;
  TRealVector = array[0..2] of Single;
var
  h_a, h_b, h_s, h_a_orig, h_b_orig: TComplexVector;
  h_c: TRealVector;
  d_a, d_b, d_c, d_s: hipDeviceptr_t;
  b: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    h_a[b].x := 3.0 + b; h_a[b].y := 1.0 + b;
    h_b[b].x := 4.0 - b; h_b[b].y := 2.0 - b;
    h_a_orig[b] := h_a[b];
    h_b_orig[b] := h_b[b];
  end;

  // 2. Allocate and Copy
  hipMalloc(d_a, batch_count * SizeOf(rocblas_float_complex));
  hipMalloc(d_b, batch_count * SizeOf(rocblas_float_complex));
  hipMalloc(d_c, batch_count * SizeOf(Single));
  hipMalloc(d_s, batch_count * SizeOf(rocblas_float_complex));
  hipMemcpy(d_a, @h_a[0], batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_b, @h_b[0], batch_count * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call rocBLAS
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_crotg_strided_batched(handle, d_a, stride, d_b, stride, d_c, stride, d_s, stride, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    Ignore('Function rocblas_crotg_strided_batched is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_crotg_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));
    // 4. Copy results back
    hipMemcpy(@h_c[0], d_c, batch_count * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_s[0], d_s, batch_count * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

    // 5. Verify
    for b := 0 to batch_count - 1 do
    begin
      CheckEquals(1.0, Sqr(h_c[b]) + Sqr(h_s[b].x) + Sqr(h_s[b].y), 1e-5, 'CROTG_Strided orthogonality failed');
    end;
  end;

  // 6. Cleanup
  hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_s);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_ZROTG_Strided_Batched;
const
  batch_count = 3;
  stride = 1;
  total_size = stride * (batch_count - 1) + 1; // 3
type
  TComplexVector = array[0..2] of rocblas_double_complex;
  TRealVector = array[0..2] of Double;
var
  h_a, h_b, h_s, h_a_orig, h_b_orig: TComplexVector;
  h_c: TRealVector;
  d_a, d_b, d_c, d_s: hipDeviceptr_t;
  b: Integer;
  hip_status: hipError_t;
  blas_status, pm_status: rocblas_status;
begin
  // 1. Initialize Host Data
  for b := 0 to batch_count - 1 do
  begin
    h_a[b].x := 3.5 + b; h_a[b].y := 1.5 + b;
    h_b[b].x := 4.5 - b; h_b[b].y := 2.5 - b;
    h_a_orig[b] := h_a[b];
    h_b_orig[b] := h_b[b];
  end;

  // 2. Allocate and Copy
  hipMalloc(d_a, batch_count * SizeOf(rocblas_double_complex));
  hipMalloc(d_b, batch_count * SizeOf(rocblas_double_complex));
  hipMalloc(d_c, batch_count * SizeOf(Double));
  hipMalloc(d_s, batch_count * SizeOf(rocblas_double_complex));
  hipMemcpy(d_a, @h_a[0], batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  hipMemcpy(d_b, @h_b[0], batch_count * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);

  // 3. Set pointer mode and Call rocBLAS
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_zrotg_strided_batched(handle, d_a, stride, d_b, stride, d_c, stride, d_s, stride, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    Ignore('Function rocblas_zrotg_strided_batched is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_zrotg_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));
    // 4. Copy results back
    hipMemcpy(@h_c[0], d_c, batch_count * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_s[0], d_s, batch_count * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);

    // 5. Verify
    for b := 0 to batch_count - 1 do
    begin
      CheckEquals(1.0, Sqr(h_c[b]) + Sqr(h_s[b].x) + Sqr(h_s[b].y), 1e-12, 'ZROTG_Strided orthogonality failed');
    end;
  end;

  // 6. Cleanup
  hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_s);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_SROTM_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride_x = 16;
  stride_y = 16;
  stride_param = 5;
  size_x = stride_x * (batch_count - 1) + N; // 42
  size_param = stride_param * (batch_count - 1) + 5; // 15
type
  TSingleVector = array[0..41] of Single;
  TParamVector = array[0..14] of Single;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TSingleVector;
  h_param: TParamVector;
  d_x, d_y, d_param: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. All setup and memory allocation is performed as usual
  for b := 0 to batch_count - 1 do
  begin
    h_param[b*stride_param+0] := -1.0; h_param[b*stride_param+1] := 1.0 + b*0.1;
    h_param[b*stride_param+2] := 2.0 + b*0.1; h_param[b*stride_param+3] := 3.0 + b*0.1;
    h_param[b*stride_param+4] := 4.0 + b*0.1;
    for i := 0 to N - 1 do
    begin
      h_x[b*stride_x+i] := 1.0 + i; h_y[b*stride_y+i] := 2.0 - i;
      h_x_orig[b*stride_x+i] := h_x[b*stride_x+i];
      h_y_orig[b*stride_y+i] := h_y[b*stride_y+i];
      h_x_expected[b*stride_x+i] := h_param[b*stride_param+1] * h_x_orig[b*stride_x+i] + h_param[b*stride_param+3] * h_y_orig[b*stride_y+i];
      h_y_expected[b*stride_y+i] := h_param[b*stride_param+2] * h_x_orig[b*stride_x+i] + h_param[b*stride_param+4] * h_y_orig[b*stride_y+i];
    end;
  end;

  hipMalloc(d_x, size_x * SizeOf(Single)); hipMalloc(d_y, size_x * SizeOf(Single));
  hipMalloc(d_param, size_param * SizeOf(Single));
  hipMemcpy(d_x, @h_x[0], size_x * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], size_x * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_param, @h_param[0], size_param * SizeOf(Single), hipMemcpyHostToDevice);

  // 2. Call the function
  // rocBLAS 5.7 only implements the strided rotm form when the handle is in
  // device pointer mode (the param vector must live in device memory).
  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_srotm_strided_batched(handle, N, d_x, incx, stride_x, d_y, incy, stride_y, d_param, stride_param, batch_count);

  // 3. Conditionally check the result
  begin
    AssertEquals('rocblas_srotm_strided_batched failed with an unexpected error', Integer(rocblas_status_success), Integer(blas_status));

    // Copy results back and verify
    hipMemcpy(@h_x[0], d_x, size_x * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[0], d_y, size_x * SizeOf(Single), hipMemcpyDeviceToHost);
    for b := 0 to batch_count - 1 do
      for i := 0 to N - 1 do
      begin
        CheckEquals(h_x_expected[b*stride_x+i], h_x[b*stride_x+i], 1e-5, 'SROTM_Strided x mismatch');
        CheckEquals(h_y_expected[b*stride_y+i], h_y[b*stride_y+i], 1e-5, 'SROTM_Strided y mismatch');
      end;
  end;

  // 4. Cleanup is always performed
  hipFree(d_x); hipFree(d_y); hipFree(d_param);
end;


procedure TTestRocBLAS_Level1_Strided_Batched.Test_DROTM_Strided_Batched;
const
  N = 10;
  batch_count = 3;
  incx = 1;
  incy = 1;
  stride_x = 16;
  stride_y = 16;
  stride_param = 5;
  size_x = stride_x * (batch_count - 1) + N; // 42
  size_param = stride_param * (batch_count - 1) + 5; // 15
type
  TDoubleVector = array[0..41] of Double;
  TParamVector = array[0..14] of Double;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: TDoubleVector;
  h_param: TParamVector;
  d_x, d_y, d_param: hipDeviceptr_t;
  b, i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // 1. All setup and memory allocation is performed as usual
  for b := 0 to batch_count - 1 do
  begin
    h_param[b*stride_param+0] := -1.0; // Flag
    h_param[b*stride_param+1] := 1.5 + b*0.15; // H11
    h_param[b*stride_param+2] := 2.5 + b*0.15; // H21
    h_param[b*stride_param+3] := 3.5 + b*0.15; // H12
    h_param[b*stride_param+4] := 4.5 + b*0.15; // H22

    for i := 0 to N - 1 do
    begin
      h_x[b*stride_x+i] := 1.5 + i; h_y[b*stride_y+i] := 2.5 - i;
      h_x_orig[b*stride_x+i] := h_x[b*stride_x+i];
      h_y_orig[b*stride_y+i] := h_y[b*stride_y+i];
      h_x_expected[b*stride_x+i] := h_param[b*stride_param+1] * h_x_orig[b*stride_x+i] + h_param[b*stride_param+3] * h_y_orig[b*stride_y+i];
      h_y_expected[b*stride_y+i] := h_param[b*stride_param+2] * h_x_orig[b*stride_x+i] + h_param[b*stride_param+4] * h_y_orig[b*stride_y+i];
    end;
  end;

  hipMalloc(d_x, size_x * SizeOf(Double));
  hipMalloc(d_y, size_x * SizeOf(Double));
  hipMalloc(d_param, size_param * SizeOf(Double));
  hipMemcpy(d_x, @h_x[0], size_x * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_y, @h_y[0], size_x * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(d_param, @h_param[0], size_param * SizeOf(Double), hipMemcpyHostToDevice);

  // 2. Call the function
  // rocBLAS 5.7 only implements the strided rotm form when the handle is in
  // device pointer mode (the param vector must live in device memory).
  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  blas_status := rocblas_drotm_strided_batched(handle, N, d_x, incx, stride_x, d_y, incy, stride_y, d_param, stride_param, batch_count);

  // 3. Conditionally check the result
  begin
    AssertEquals('rocblas_drotm_strided_batched failed with an unexpected error', Integer(rocblas_status_success), Integer(blas_status));

    // Copy results back and verify
    hipMemcpy(@h_x[0], d_x, size_x * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_y[0], d_y, size_x * SizeOf(Double), hipMemcpyDeviceToHost);
    for b := 0 to batch_count - 1 do
      for i := 0 to N - 1 do
      begin
        CheckEquals(h_x_expected[b*stride_x+i], h_x[b*stride_x+i], 1e-12, 'DROTM_Strided x mismatch');
        CheckEquals(h_y_expected[b*stride_y+i], h_y[b*stride_y+i], 1e-12, 'DROTM_Strided y mismatch');
      end;
  end;

  // 4. Cleanup is always performed
  hipFree(d_x); hipFree(d_y); hipFree(d_param);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_SROTMG_Strided_Batched;
const
  batch_count = 3;
  stride = 1; // Strided ROTMG typically operates on contiguous arrays of scalars
  PARAM_SIZE = 5;
  param_stride = 5;
  total_scalars = stride * (batch_count - 1) + 1; // = 3 for stride=1
  total_params = param_stride * (batch_count - 1) + 5; // 15
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
    h_d1[b * stride] := 4.0 + b;
    h_d2[b * stride] := 9.0 + b;
    h_x1[b * stride] := 2.0 + b;
    h_y1[b * stride] := 3.0 + b;
  end;

  // 2. Allocate device memory
  hipMalloc(d_d1, total_scalars * SizeOf(Single));
  hipMalloc(d_d2, total_scalars * SizeOf(Single));
  hipMalloc(d_x1, total_scalars * SizeOf(Single));
  hipMalloc(d_y1, total_scalars * SizeOf(Single));
  hipMalloc(d_param, total_params * SizeOf(Single));

  // 3. Copy data to device
  hipMemcpy(d_d1, @h_d1[0], total_scalars * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_d2, @h_d2[0], total_scalars * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_x1, @h_x1[0], total_scalars * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(d_y1, @h_y1[0], total_scalars * SizeOf(Single), hipMemcpyHostToDevice);

  // 4. Set pointer mode to DEVICE and Call rocBLAS
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode failed', Integer(rocblas_status_success), Integer(pm_status));

  blas_status := rocblas_srotmg_strided_batched(handle, d_d1, stride, d_d2, stride, d_x1, stride, d_y1, stride, d_param, param_stride, batch_count);
  if blas_status = rocblas_status_not_implemented then
  begin
    Ignore('Function rocblas_srotmg_strided_batched is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_srotmg_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));
    // 5. Copy results back if implemented
    hipMemcpy(@h_d1[0], d_d1, total_scalars * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_d2[0], d_d2, total_scalars * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_x1[0], d_x1, total_scalars * SizeOf(Single), hipMemcpyDeviceToHost);
    hipMemcpy(@h_param[0], d_param, total_params * SizeOf(Single), hipMemcpyDeviceToHost);

    // 6. Verify (Sanity check)
    for b := 0 to batch_count - 1 do
    begin
      AssertTrue(Format('SROTMG_Strided d1 is not finite batch %d', [b]), not(IsInfinite(h_d1[b*stride]) or IsNan(h_d1[b*stride])));
          AssertTrue(Format('SROTMG_Strided d2 is not finite batch %d', [b]), not(IsInfinite(h_d2[b*stride]) or IsNan(h_d2[b*stride])));
          AssertTrue(Format('SROTMG_Strided x1 is not finite batch %d', [b]), not(IsInfinite(h_x1[b*stride]) or IsNan(h_x1[b*stride])));
          for i := 0 to PARAM_SIZE - 1 do
            AssertTrue(Format('SROTMG_Strided param[%d] is not finite batch %d', [i, b]), not(IsInfinite(h_param[b*param_stride+i]) or IsNan(h_param[b*param_stride+i])));    end;
  end;

  // 7. Cleanup
  hipFree(d_d1); hipFree(d_d2); hipFree(d_x1); hipFree(d_y1); hipFree(d_param);
end;

procedure TTestRocBLAS_Level1_Strided_Batched.Test_DROTMG_Strided_Batched;
const
  batch_count = 3;
  stride = 1;
  PARAM_SIZE = 5;
  param_stride = 5;
  total_scalars = stride * (batch_count - 1) + 1; // 3
  total_params = param_stride * (batch_count - 1) + 5; // 15
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
    h_d1[b * stride] := 4.5 + b;
    h_d2[b * stride] := 9.5 + b;
    h_x1[b * stride] := 2.5 + b;
    h_y1[b * stride] := 3.5 + b;
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

  blas_status := rocblas_drotmg_strided_batched(handle, d_d1, stride, d_d2, stride, d_x1, stride, d_y1, stride, d_param, param_stride, batch_count);

  if blas_status = rocblas_status_not_implemented then
  begin
    Ignore('Function rocblas_drotmg_strided_batched is not implemented in this rocBLAS version.');
  end
  else
  begin
    AssertEquals('rocblas_drotmg_strided_batched failed', Integer(rocblas_status_success), Integer(blas_status));
    // 4. Copy results back
    hipMemcpy(@h_d1[0], d_d1, total_scalars * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_d2[0], d_d2, total_scalars * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_x1[0], d_x1, total_scalars * SizeOf(Double), hipMemcpyDeviceToHost);
    hipMemcpy(@h_param[0], d_param, total_params * SizeOf(Double), hipMemcpyDeviceToHost);

    // 5. Verify
    for b := 0 to batch_count - 1 do
    begin
      AssertTrue(Format('DROTMG_Strided d1 is not finite batch %d', [b]), not(IsInfinite(h_d1[b*stride]) or IsNan(h_d1[b*stride])));
      AssertTrue(Format('DROTMG_Strided d2 is not finite batch %d', [b]), not(IsInfinite(h_d2[b*stride]) or IsNan(h_d2[b*stride])));
      AssertTrue(Format('DROTMG_Strided x1 is not finite batch %d', [b]), not(IsInfinite(h_x1[b*stride]) or IsNan(h_x1[b*stride])));
      for i := 0 to PARAM_SIZE - 1 do
        AssertTrue(Format('DROTMG_Strided param[%d] is not finite batch %d', [i, b]), not(IsInfinite(h_param[b*param_stride+i]) or IsNan(h_param[b*param_stride+i])));    end;
  end;

  // 6. Cleanup
  hipFree(d_d1); hipFree(d_d2); hipFree(d_x1); hipFree(d_y1); hipFree(d_param);
end;

initialization
  RegisterTest(TTestRocBLAS_Level1_Strided_Batched);

end.
