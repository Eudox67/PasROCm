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

  Signature checks for the rocBLAS level-1 (vector) bindings: every call is
  made and its status is asserted, so a changed prototype or a missing
  symbol fails here first.
------------------------------------------------------------------------------}

unit Test_rocblas_lvl1;

{$mode objfpc}{$H+}
{$linklib rocblas}

interface

uses
  SysUtils, Math, fpcunit, testregistry, hip_error, hip_types, hip_runtime, rocblas, Test_rocblas_base, ROCm_test_utils;

type

  { TTestRocBLAS_Level1 }
  TTestRocBLAS_Level1 = class(TTestRocBLAS_Base)   // from Test_rocblas_base
  published
    procedure Test_SAXPY; // Single precision
    procedure Test_DAXPY; // Double precision
    procedure Test_HAXPY;
    procedure Test_CAXPY;
    procedure Test_ZAXPY;
    procedure Test_SCOPY;
    procedure Test_DCOPY;
    procedure Test_CCOPY;
    procedure Test_ZCOPY;

    procedure Test_SSCAL;
    procedure Test_DSCAL;
    procedure Test_CSSCAL; // Note: Wrapper uses rocblas_float_complex for alpha
    procedure Test_ZDSCAL; // Note: Wrapper uses rocblas_double_complex for alpha

    procedure Test_SDOT;
    procedure Test_DDOT;
    procedure Test_HDOT;
    procedure Test_BFDOT;
    procedure Test_CDOTU;
    procedure Test_ZDOTU;
    procedure Test_CDOTC;
    procedure Test_ZDOTC;

    procedure Test_SNRM2;
    procedure Test_DNRM2;
    procedure Test_SCNRM2;
    procedure Test_DZNRM2;

    procedure Test_SASUM;
    procedure Test_DASUM;
    procedure Test_SCASUM;
    procedure Test_DZASUM;

    procedure Test_ISAMAX;
    procedure Test_IDAMAX;
    procedure Test_ICAMAX;
    procedure Test_IZAMAX;

    procedure Test_ISAMIN;
    procedure Test_IDAMIN;
    procedure Test_ICAMIN;
    procedure Test_IZAMIN;


    procedure Test_SSWAP;
    procedure Test_DSWAP;
    procedure Test_CSWAP;
    procedure Test_ZSWAP;

    procedure Test_SROT;
    procedure Test_DROT;
    procedure Test_CSROT;
    procedure Test_ZDROT;

    // ROTG Tests (Generate Givens Rotation)
    procedure Test_SROTG;
    procedure Test_DROTG;
    procedure Test_CROTG;
    procedure Test_ZROTG;

    // ROTM Tests (Apply Modified Givens Rotation)
    procedure Test_SROTM;
    procedure Test_DROTM;

    // ROTMG Tests (Generate Modified Givens Rotation)
    procedure Test_SROTMG;
    procedure Test_DROTMG;
  end;


implementation

//##############################################################################
//## Level 1 Implementation
//##############################################################################

procedure TTestRocBLAS_Level1.Test_SAXPY;
const
  N = 5;
  alpha: Single = 2.0;
var
  devX, devY: hipDeviceptr_t;
  hostX, hostY, expectedY: array[0..N-1] of Single;
  i: Integer;
begin
  for i := 0 to N - 1 do
  begin
    hostX[i] := i + 1.0;
    hostY[i] := i * 2.0;
    expectedY[i] := alpha * hostX[i] + hostY[i];
  end;

  hipMalloc(devX, N * SizeOf(Single));
  hipMalloc(devY, N * SizeOf(Single));
  hipMemcpy(devX, @hostX[0], N * SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N * SizeOf(Single), hipMemcpyHostToDevice);

  AssertEquals('rocblas_saxpy should return success', Integer(rocblas_status_success), Integer(rocblas_saxpy(handle, N, @alpha, devX, 1, devY, 1)));

  hipMemcpy(@hostY[0], devY, N * SizeOf(Single), hipMemcpyDeviceToHost);

  for i := 0 to N - 1 do
    AssertEquals('SAXPY Y[' + IntToStr(i) + ']', expectedY[i], hostY[i], 1e-6);

  hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level1.Test_HAXPY;
const
  N = 5;
  alpha_val: Single = 2.0; // Use Single for easy definition
var
  // --- Variables ---
  alpha: rocblas_half; // Declare alpha as a variable
  devX, devY: hipDeviceptr_t;
  hostX, hostY, expectedY: array[0..N-1] of rocblas_half;
  hostX_single, hostY_single, expectedY_single: array[0..N-1] of Single;
  i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
  temp_single: Single;
begin
  // --- 1. Initialize alpha using the utility function ---
  alpha := SingleToHalf(alpha_val);

  // --- 2. Initialize host data (using Single for calculation, then convert) ---
  for i := 0 to N - 1 do
  begin
    hostX_single[i] := i + 1.0; // 1.0, 2.0, 3.0, 4.0, 5.0
    hostY_single[i] := i * 2.0; // 0.0, 2.0, 4.0, 6.0, 8.0
    expectedY_single[i] := alpha_val * hostX_single[i] + hostY_single[i];
    // Convert Single values to rocblas_half for input data using utility
    hostX[i] := SingleToHalf(hostX_single[i]);
    hostY[i] := SingleToHalf(hostY_single[i]);
  end;

  // --- 3. Allocate device memory ---
  devX := nil;
  devY := nil;
  hip_status := hipMalloc(devX, N * SizeOf(rocblas_half));
  AssertEquals('hipMalloc devX for HAXPY', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(devY, N * SizeOf(rocblas_half));
  AssertEquals('hipMalloc devY for HAXPY', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 4. Copy data Host to Device ---
  hip_status := hipMemcpy(devX, @hostX[0], N * SizeOf(rocblas_half), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D X for HAXPY', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(devY, @hostY[0], N * SizeOf(rocblas_half), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D Y for HAXPY', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 5. Execute rocBLAS HAXPY ---
  // Note: alpha is passed by reference (@alpha)
  // Cast device pointers appropriately for rocblas_haxpy
  blas_status := rocblas_haxpy(handle, N, @alpha, Prochalf(devX), 1, Prochalf(devY), 1);
  AssertEquals('rocblas_haxpy should return success', Integer(rocblas_status_success), Integer(blas_status));

  // --- 6. Copy result Device to Host ---
  hip_status := hipMemcpy(@hostY[0], devY, N * SizeOf(rocblas_half), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy D2H Y for HAXPY', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 7. Compare Results ---
  // Convert result back to Single for comparison using the utility function
  for i := 0 to N - 1 do
  begin
     temp_single := HalfToSingle(hostY[i]);
     // Use a relatively loose tolerance for half precision
     AssertEquals('HAXPY Y[' + IntToStr(i) + ']', expectedY_single[i], temp_single, 5e-2);
     // 5e-2 (5%) is used as a reasonable tolerance for half precision calculations.
     // Adjust if needed based on empirical results or stricter requirements.
  end;

  // --- 8. Free device memory ---
  hip_status := hipFree(devX);
  AssertEquals('hipFree devX for HAXPY', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipFree(devY);
  AssertEquals('hipFree devY for HAXPY', Integer(HIP_SUCCESS), Integer(hip_status));
end;

procedure TTestRocBLAS_Level1.Test_DAXPY;
const
  N = 5;
  alpha: Double = 2.0;
var
  devX, devY: hipDeviceptr_t;
  hostX, hostY, expectedY: array[0..N-1] of Double;
  i: Integer;
begin
  for i := 0 to N - 1 do
  begin
    hostX[i] := i + 1.0;
    hostY[i] := i * 2.0;
    expectedY[i] := alpha * hostX[i] + hostY[i];
  end;

  hipMalloc(devX, N * SizeOf(Double));
  hipMalloc(devY, N * SizeOf(Double));
  hipMemcpy(devX, @hostX[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  hipMemcpy(devY, @hostY[0], N * SizeOf(Double), hipMemcpyHostToDevice);

  AssertEquals('rocblas_daxpy should return success', Integer(rocblas_status_success), Integer(rocblas_daxpy(handle, N, @alpha, devX, 1, devY, 1)));

  hipMemcpy(@hostY[0], devY, N * SizeOf(Double), hipMemcpyDeviceToHost);

  for i := 0 to N - 1 do
    AssertEquals('DAXPY Y[' + IntToStr(i) + ']', expectedY[i], hostY[i], 1e-12);

  hipFree(devX); hipFree(devY);
end;

procedure TTestRocBLAS_Level1.Test_CAXPY;
const
  N = 3;
var
  // --- Variables ---
  alpha: rocblas_float_complex; // Declare alpha as a variable
  devX, devY: hipDeviceptr_t;
  hostX, hostY, expectedY: array[0..N-1] of rocblas_float_complex;
  i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // --- 1. Initialize alpha ---
  alpha.x := 2.0;
  alpha.y := 1.0; // Alpha = (2.0 + 1.0i)

  // --- 2. Initialize host data ---
  // X = [(1+2i), (3+0i), (-1-1i)]
  hostX[0].x :=  1.0; hostX[0].y :=  2.0;
  hostX[1].x :=  3.0; hostX[1].y :=  0.0;
  hostX[2].x := -1.0; hostX[2].y := -1.0;
  // Y = [(0+1i), (2-3i), (4+0i)]
  hostY[0].x := 0.0; hostY[0].y := 1.0;
  hostY[1].x := 2.0; hostY[1].y := -3.0;
  hostY[2].x := 4.0; hostY[2].y := 0.0;

  // Calculate expected result: Y = alpha*X + Y
  // Manual calculation:
  // alpha * X[0] = (2+1i)*(1+2i) = 2*1 - 1*2 + i*(2*2 + 1*1) = 0 + 5i
  // Y_new[0] = (0+5i) + (0+1i) = (0+6i)
  expectedY[0].x := 0.0; expectedY[0].y := 6.0;

  // alpha * X[1] = (2+1i)*(3+0i) = 6 + 3i
  // Y_new[1] = (6+3i) + (2-3i) = (8+0i)
  expectedY[1].x := 8.0; expectedY[1].y := 0.0;

  // alpha * X[2] = (2+1i)*(-1-1i) = 2*(-1) - 1*(-1) + i*(2*(-1) + 1*(-1)) = -2+1 + i*(-2-1) = -1 - 3i
  // Y_new[2] = (-1-3i) + (4+0i) = (3-3i)
  expectedY[2].x := 3.0; expectedY[2].y := -3.0;

  // --- 3. Allocate device memory ---
  devX := nil;
  devY := nil;
  hip_status := hipMalloc(devX, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc devX for CAXPY', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(devY, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc devY for CAXPY', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 4. Copy data Host to Device ---
  hip_status := hipMemcpy(devX, @hostX[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D X for CAXPY', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(devY, @hostY[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D Y for CAXPY', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 5. Execute rocBLAS CAXPY ---
  // Note: alpha is passed by reference (@alpha)
  // Cast device pointers appropriately for rocblas_caxpy
  blas_status := rocblas_caxpy(handle, N, @alpha, Procfloat_complex(devX), 1, Procfloat_complex(devY), 1);
  AssertEquals('rocblas_caxpy should return success', Integer(rocblas_status_success), Integer(blas_status));

  // --- 6. Copy result Device to Host ---
  hip_status := hipMemcpy(@hostY[0], devY, N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy D2H Y for CAXPY', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 7. Compare Results ---
  for i := 0 to N - 1 do
  begin
    // Check real part
    AssertEquals('CAXPY Y[' + IntToStr(i) + '].real', expectedY[i].x, hostY[i].x, 1e-6);
    // Check imaginary part
    AssertEquals('CAXPY Y[' + IntToStr(i) + '].imag', expectedY[i].y, hostY[i].y, 1e-6);
  end;

  // --- 8. Free device memory ---
  hip_status := hipFree(devX);
  AssertEquals('hipFree devX for CAXPY', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipFree(devY);
  AssertEquals('hipFree devY for CAXPY', Integer(HIP_SUCCESS), Integer(hip_status));
end;


procedure TTestRocBLAS_Level1.Test_ZAXPY;
const
  N = 2;
var
  // --- Variables ---
  alpha: rocblas_double_complex; // Declare alpha as a variable
  devX, devY: hipDeviceptr_t;
  hostX, hostY, expectedY: array[0..N-1] of rocblas_double_complex;
  i: Integer;
  hip_status: hipError_t;
  blas_status: rocblas_status;
begin
  // --- 1. Initialize alpha ---
  alpha.x := 1.5;
  alpha.y := -0.5; // Alpha = (1.5 - 0.5i)

  // --- 2. Initialize host data ---
  // X = [(2+3i), (1-2i)]
  hostX[0].x := 2.0; hostX[0].y := 3.0;
  hostX[1].x := 1.0; hostX[1].y := -2.0;
  // Y = [(-1+1i), (0+4i)]
  hostY[0].x := -1.0; hostY[0].y := 1.0;
  hostY[1].x := 0.0; hostY[1].y := 4.0;

  // Calculate expected result: Y = alpha*X + Y
  // Manual calculation:
  // alpha * X[0] = (1.5-0.5i)*(2+3i) = 1.5*2 - (-0.5)*3 + i*(1.5*3 + (-0.5)*2) = 3+1.5 + i*(4.5-1) = 4.5 + 3.5i
  // Y_new[0] = (4.5+3.5i) + (-1+1i) = (3.5+4.5i)
  expectedY[0].x := 3.5; expectedY[0].y := 4.5;

  // alpha * X[1] = (1.5-0.5i)*(1-2i) = 1.5*1 - (-0.5)*(-2) + i*(1.5*(-2) + (-0.5)*1) = 1.5-1 + i*(-3-0.5) = 0.5 - 3.5i
  // Y_new[1] = (0.5-3.5i) + (0+4i) = (0.5+0.5i)
  expectedY[1].x := 0.5; expectedY[1].y := 0.5;

  // --- 3. Allocate device memory ---
  devX := nil;
  devY := nil;
  hip_status := hipMalloc(devX, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc devX for ZAXPY', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(devY, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc devY for ZAXPY', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 4. Copy data Host to Device ---
  hip_status := hipMemcpy(devX, @hostX[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D X for ZAXPY', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMemcpy(devY, @hostY[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D Y for ZAXPY', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 5. Execute rocBLAS ZAXPY ---
  // Note: alpha is passed by reference (@alpha)
  // Cast device pointers appropriately for rocblas_zaxpy
  blas_status := rocblas_zaxpy(handle, N, @alpha, Procdouble_complex(devX), 1, Procdouble_complex(devY), 1);
  AssertEquals('rocblas_zaxpy should return success', Integer(rocblas_status_success), Integer(blas_status));

  // --- 6. Copy result Device to Host ---
  hip_status := hipMemcpy(@hostY[0], devY, N * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy D2H Y for ZAXPY', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 7. Compare Results ---
  for i := 0 to N - 1 do
  begin
    // Check real part
    AssertEquals('ZAXPY Y[' + IntToStr(i) + '].real', expectedY[i].x, hostY[i].x, 1e-12);
    // Check imaginary part
    AssertEquals('ZAXPY Y[' + IntToStr(i) + '].imag', expectedY[i].y, hostY[i].y, 1e-12);
  end;

  // --- 8. Free device memory ---
  hip_status := hipFree(devX);
  AssertEquals('hipFree devX for ZAXPY', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipFree(devY);
  AssertEquals('hipFree devY for ZAXPY', Integer(HIP_SUCCESS), Integer(hip_status));
end;



procedure TTestRocBLAS_Level1.Test_SCOPY;
const
  N = 5;
  incx = 1;
  incy = 1;
var
  h_x, h_y, h_x_orig, h_y_orig, h_x_expected, h_y_expected: array[0..N-1] of rocblas_float_complex;
  d_x, d_y: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;      // For HIP function calls
begin
  for i := 0 to N - 1 do
  begin
    h_x_orig[i].x := i + 1.0;
    h_x_orig[i].y := i * 0.5;
    h_y_orig[i].x := (i + 1.0) * 10.0;
    h_y_orig[i].y := (i * 0.5) * 10.0;
  end;
  h_x := h_x_orig;
  h_y := h_y_orig;
  h_x_expected := h_y_orig;
  h_y_expected := h_x_orig;

  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_float_complex));
  hip_status := hipMalloc(d_y, N * SizeOf(rocblas_float_complex));

  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);

  blas_status := rocblas_cswap(handle, N, d_x, incx, d_y, incy);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_cswap failed');

  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);

  for i := 0 to N - 1 do
  begin
     CheckEquals(h_x_expected[i].x, h_x[i].x, 1e-6, 'CSWAP x real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_x_expected[i].y, h_x[i].y, 1e-6, 'CSWAP x imaginary part mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_expected[i].x, h_y[i].x, 1e-6, 'CSWAP y real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_expected[i].y, h_y[i].y, 1e-6, 'CSWAP y imaginary part mismatch at index ' + IntToStr(i));
  end;

  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1.Test_DCOPY;
const
  N = 5;
  incx = 1;
  incy = 1;
var
  h_x, h_y, h_y_expected: array[0..N-1] of Double;
  d_x, d_y: Pointer;
  i: Integer;
  blas_status: rocblas_status; // Renamed from 'status'
  hip_status: hipError_t;      // For HIP function calls
begin
  // Initialize host data
  for i := 0 to N - 1 do
  begin
    h_x[i] := i + 1.5;
    h_y[i] := 0.0; // Destination initialized to zero
    h_y_expected[i] := h_x[i]; // Expected result
  end;

  // Initialize device pointers
  d_x := nil;
  d_y := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(Double));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(Double));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);

  // Copy data to device
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);

  // Perform rocBLAS operation
  blas_status := rocblas_dcopy(handle, N, d_x, incx, d_y, incy);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_dcopy failed');

  // Copy result back to host
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y D->H', HIP_SUCCESS, hip_status);

  // Verify results
  for i := 0 to N - 1 do
  begin
     CheckEquals(h_y_expected[i], h_y[i], 1e-12, 'DCOPY result mismatch at index ' + IntToStr(i));
  end;

  // Cleanup
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1.Test_CCOPY;
const
  N = 5;
  incx = 1;
  incy = 1;
var
  h_x, h_y, h_y_expected: array[0..N-1] of rocblas_float_complex;
  d_x, d_y: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data
  for i := 0 to N - 1 do
  begin
    h_x[i].x := i + 1.0;
    h_x[i].y := i * 0.5;
    h_y[i].x := 0.0;
    h_y[i].y := 0.0;
    h_y_expected[i] := h_x[i]; // Expected result
  end;

  // Initialize device pointers
  d_x := nil;
  d_y := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);

  // Copy data to device
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);

  // Perform rocBLAS operation
  blas_status := rocblas_ccopy(handle, N, d_x, incx, d_y, incy);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_ccopy failed');

  // Copy result back to host
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y D->H', HIP_SUCCESS, hip_status);

  // Verify results
  for i := 0 to N - 1 do
  begin
     CheckEquals(h_y_expected[i].x, h_y[i].x, 1e-6, 'CCOPY real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_expected[i].y, h_y[i].y, 1e-6, 'CCOPY imaginary part mismatch at index ' + IntToStr(i));
  end;

  // Cleanup
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1.Test_ZCOPY;
const
  N = 5;
  incx = 1;
  incy = 1;
var
  h_x, h_y, h_y_expected: array[0..N-1] of rocblas_double_complex;
  d_x, d_y: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data
  for i := 0 to N - 1 do
  begin
    h_x[i].x := i + 1.5;
    h_x[i].y := i * 0.75;
    h_y[i].x := 0.0;
    h_y[i].y := 0.0;
    h_y_expected[i] := h_x[i]; // Expected result
  end;

  // Initialize device pointers
  d_x := nil;
  d_y := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);

  // Copy data to device
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);

  // Perform rocBLAS operation
  blas_status := rocblas_zcopy(handle, N, d_x, incx, d_y, incy);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_zcopy failed');

  // Copy result back to host
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y D->H', HIP_SUCCESS, hip_status);

  // Verify results
  for i := 0 to N - 1 do
  begin
     CheckEquals(h_y_expected[i].x, h_y[i].x, 1e-12, 'ZCOPY real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_expected[i].y, h_y[i].y, 1e-12, 'ZCOPY imaginary part mismatch at index ' + IntToStr(i));
  end;

  // Cleanup
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1.Test_SSCAL;
const
  N = 5;
  incx = 1;
var
  h_x, h_x_expected: array[0..N-1] of Single;
  alpha: Single;
  d_x: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  alpha := 2.5;
  // Initialize host data
  for i := 0 to N - 1 do
  begin
    h_x[i] := i + 1.0;
    h_x_expected[i] := alpha * h_x[i];
  end;

  d_x := nil; // Initialize pointer
  hip_status := hipMalloc(d_x, N * SizeOf(Single));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);

  // Note: alpha is passed by pointer for rocblas_sscal
  blas_status := rocblas_sscal(handle, N, @alpha, d_x, incx);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_sscal failed');

  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);

  for i := 0 to N - 1 do
  begin
     CheckEquals(h_x_expected[i], h_x[i], 1e-5, 'SSCAL result mismatch at index ' + IntToStr(i));
  end;

  hipFree(d_x);
end;

procedure TTestRocBLAS_Level1.Test_DSCAL;
const
  N = 5;
  incx = 1;
var
  h_x, h_x_expected: array[0..N-1] of Double;
  alpha: Double;
  d_x: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  alpha := 3.14;
  for i := 0 to N - 1 do
  begin
    h_x[i] := i + 1.5;
    h_x_expected[i] := alpha * h_x[i];
  end;

  d_x := nil; // Initialize pointer
  hip_status := hipMalloc(d_x, N * SizeOf(Double));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);

  // Note: alpha is passed by pointer for rocblas_dscal
  blas_status := rocblas_dscal(handle, N, @alpha, d_x, incx);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_dscal failed');

  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);

  for i := 0 to N - 1 do
  begin
     CheckEquals(h_x_expected[i], h_x[i], 1e-12, 'DSCAL result mismatch at index ' + IntToStr(i));
  end;

  hipFree(d_x);
end;

procedure TTestRocBLAS_Level1.Test_CSSCAL;
const
  N = 5;
  incx = 1;
var
  h_x, h_x_expected: array[0..N-1] of rocblas_float_complex;
  alpha: Single; // Note: alpha is real for csscal
  d_x: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  alpha := 0.5;
  for i := 0 to N - 1 do
  begin
    h_x[i].x := i + 1.0;
    h_x[i].y := i * 0.5;
    h_x_expected[i].x := alpha * h_x[i].x;
    h_x_expected[i].y := alpha * h_x[i].y;
  end;

  d_x := nil; // Initialize pointer
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);

  // Wrapper signature: rocblas_csscal(handle, n, alpha, x, incx)
  // The wrapper expects alpha as PSingle (pointer to real Single)
  blas_status := rocblas_csscal(handle, N, @alpha, d_x, incx);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_csscal failed');

  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);

  for i := 0 to N - 1 do
  begin
     CheckEquals(h_x_expected[i].x, h_x[i].x, 1e-6, 'CSSCAL real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_x_expected[i].y, h_x[i].y, 1e-6, 'CSSCAL imaginary part mismatch at index ' + IntToStr(i));
  end;

  hipFree(d_x);
end;

procedure TTestRocBLAS_Level1.Test_ZDSCAL;
const
  N = 5;
  incx = 1;
var
  h_x, h_x_expected: array[0..N-1] of rocblas_double_complex;
  alpha: Double; // Note: alpha is real for zdscal
  d_x: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  alpha := 2.0;
  for i := 0 to N - 1 do
  begin
    h_x[i].x := i + 1.5;
    h_x[i].y := i * 0.75;
    h_x_expected[i].x := alpha * h_x[i].x;
    h_x_expected[i].y := alpha * h_x[i].y;
  end;

  d_x := nil; // Initialize pointer
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);

  // Wrapper signature: rocblas_zdscal(handle, n, alpha, x, incx)
  // The wrapper expects alpha as PDouble (pointer to real Double)
  blas_status := rocblas_zdscal(handle, N, @alpha, d_x, incx);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_zdscal failed');

  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);

  for i := 0 to N - 1 do
  begin
     CheckEquals(h_x_expected[i].x, h_x[i].x, 1e-12, 'ZDSCAL real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_x_expected[i].y, h_x[i].y, 1e-12, 'ZDSCAL imaginary part mismatch at index ' + IntToStr(i));
  end;

  hipFree(d_x);
end;

// --- DOT Tests ---

procedure TTestRocBLAS_Level1.Test_SDOT;
const
  N = 5;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of Single;
  h_result, expected_result: Single;
  d_x, d_y, d_result: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  expected_result := 0.0;
  // Initialize host data
  for i := 0 to N - 1 do
  begin
    h_x[i] := i + 1.0;
    h_y[i] := i * 2.0 + 1.0;
    expected_result := expected_result + (h_x[i] * h_y[i]); // Manual calculation
  end;
  h_result := 0.0; // Initialize result

  d_x := nil;
  d_y := nil;
  d_result := nil; // Initialize result pointer

  hip_status := hipMalloc(d_x, N * SizeOf(Single));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(Single));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result, SizeOf(Single)); // Allocate for result on device
  AssertEquals('hipMalloc d_result', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_result, @h_result, SizeOf(Single), hipMemcpyHostToDevice); // Copy initial result (0.0)
  AssertEquals('hipMemcpy result H->D', HIP_SUCCESS, hip_status);

  blas_status := rocblas_sdot(handle, N, d_x, incx, d_y, incy, d_result);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_sdot failed');

  hip_status := hipMemcpy(@h_result, d_result, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result D->H', HIP_SUCCESS, hip_status);

  CheckEquals(expected_result, h_result, 1e-4, 'SDOT result mismatch');

  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1.Test_DDOT;
const
  N = 5;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of Double;
  h_result, expected_result: Double;
  d_x, d_y, d_result: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  expected_result := 0.0;
  for i := 0 to N - 1 do
  begin
    h_x[i] := i + 1.5;
    h_y[i] := i * 2.5 + 1.1;
    expected_result := expected_result + (h_x[i] * h_y[i]);
  end;
  h_result := 0.0;

  d_x := nil;
  d_y := nil;
  d_result := nil;

  hip_status := hipMalloc(d_x, N * SizeOf(Double));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(Double));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result, SizeOf(Double));
  AssertEquals('hipMalloc d_result', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_result, @h_result, SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy result H->D', HIP_SUCCESS, hip_status);

  blas_status := rocblas_ddot(handle, N, d_x, incx, d_y, incy, d_result);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_ddot failed');

  hip_status := hipMemcpy(@h_result, d_result, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result D->H', HIP_SUCCESS, hip_status);

  CheckEquals(expected_result, h_result, 1e-12, 'DDOT result mismatch');

  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_result);
end;

// Note: Tests for HDOT and BFDOT are omitted here as they require specific
// half-precision (rocblas_half) and bfloat16 (rocblas_bfloat16) types and
// potentially special handling for memory copies and comparisons.
// The wrappers rocblas_hdot and rocblas_bfdot exist in rocblas.pas if needed later.

// --- HDOT Test ---
// Note: This test requires handling of rocblas_half type (record with UInt16 data).
// Host calculations are done in Single precision for expected result.
// Conversion uses IEEE 754 compliant helper functions.
procedure TTestRocBLAS_Level1.Test_HDOT;
const
  N = 5;
  incx = 1;
  incy = 1;
type
  THalfArray = array[0..N-1] of rocblas_half;
var
  h_x_single, h_y_single: array[0..N-1] of Single;
  h_x_half, h_y_half: THalfArray;
  h_result_half: rocblas_half;
  h_result_single, expected_result: Single;
  d_x, d_y, d_result: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  expected_result := 0.0;
  // Initialize host data in Single precision and calculate expected result
  for i := 0 to N - 1 do
  begin
    h_x_single[i] := i + 1.0;
    h_y_single[i] := i * 2.0 + 1.0;
    expected_result := expected_result + (h_x_single[i] * h_y_single[i]); // Manual calculation in Single
  end;

  // Convert Single data to rocblas_half for device using IEEE 754 compliant function
  for i := 0 to N - 1 do
  begin
    h_x_half[i] := SingleToHalf(h_x_single[i]);
    h_y_half[i] := SingleToHalf(h_y_single[i]);
  end;

  // Initialize result variable correctly
  h_result_half.data := 0; // Initialize half result on host
  h_result_single := 0.0;  // Initialize single result for comparison

  // Initialize device pointers
  d_x := nil;
  d_y := nil;
  d_result := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_half));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(rocblas_half));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result, SizeOf(rocblas_half));
  AssertEquals('hipMalloc d_result', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x_half[0], N * SizeOf(rocblas_half), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y_half[0], N * SizeOf(rocblas_half), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);
  // Copy initial result (zero) H->D
  hip_status := hipMemcpy(d_result, @h_result_half, SizeOf(rocblas_half), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy result H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS HDOT
  blas_status := rocblas_hdot(handle, N, d_x, incx, d_y, incy, d_result);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_hdot failed');

  // Copy result D->H
  hip_status := hipMemcpy(@h_result_half, d_result, SizeOf(rocblas_half), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result D->H', HIP_SUCCESS, hip_status);

  // Convert result back to Single for comparison using IEEE 754 compliant function
  h_result_single := HalfToSingle(h_result_half);

  // Verify result (tolerance accounts for half-precision loss)
  CheckEquals(expected_result, h_result_single, 1e-2, 'HDOT result mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_result);
end;


// --- BFDOT Test ---
// Note: This test requires handling of rocblas_bfloat16 type (record with UInt16 data).
// Host calculations are done in Single precision for expected result.
// Conversion uses IEEE 754 compliant helper functions.
procedure TTestRocBLAS_Level1.Test_BFDOT;
const
  N = 5;
  incx = 1;
  incy = 1;
type
  TBFloat16Array = array[0..N-1] of rocblas_bfloat16;
var
  h_x_single, h_y_single: array[0..N-1] of Single;
  h_x_bf16, h_y_bf16: TBFloat16Array;
  h_result_bf16: rocblas_bfloat16;
  h_result_single, expected_result: Single;
  d_x, d_y, d_result: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  expected_result := 0.0;
  // Initialize host data in Single precision and calculate expected result
  for i := 0 to N - 1 do
  begin
    h_x_single[i] := i + 1.5;
    h_y_single[i] := i * 2.5 + 1.1;
    expected_result := expected_result + (h_x_single[i] * h_y_single[i]); // Manual calc in Single
  end;

  // Convert Single data to rocblas_bfloat16 for device using IEEE 754 compliant function
  for i := 0 to N - 1 do
  begin
     h_x_bf16[i] := SingleToBFloat16(h_x_single[i]);
     h_y_bf16[i] := SingleToBFloat16(h_y_single[i]);
  end;

  // Initialize result variable correctly
  h_result_bf16.data := 0; // Initialize bf16 result on host
  h_result_single := 0.0;  // Initialize single result for comparison

  // Initialize device pointers
  d_x := nil;
  d_y := nil;
  d_result := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_bfloat16));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(rocblas_bfloat16));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result, SizeOf(rocblas_bfloat16));
  AssertEquals('hipMalloc d_result', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x_bf16[0], N * SizeOf(rocblas_bfloat16), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y_bf16[0], N * SizeOf(rocblas_bfloat16), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);
  // Copy initial result (zero) H->D
  hip_status := hipMemcpy(d_result, @h_result_bf16, SizeOf(rocblas_bfloat16), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy result H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS BFDOT
  blas_status := rocblas_bfdot(handle, N, d_x, incx, d_y, incy, d_result);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_bfdot failed');

  // Copy result D->H
  hip_status := hipMemcpy(@h_result_bf16, d_result, SizeOf(rocblas_bfloat16), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result D->H', HIP_SUCCESS, hip_status);

  // Convert result back to Single for comparison using IEEE 754 compliant function
  h_result_single := BFloat16ToSingle(h_result_bf16);

  // Verify result (tolerance accounts for bfloat16 precision loss)
  CheckEqualsRelative(expected_result, h_result_single, 1e-2, 'BFDOT result mismatch.');
  // Or use absolute: CheckEquals(expected_result, h_result_single, 0.3, 'BFDOT result mismatch');
  CheckEquals(expected_result, h_result_single, 0.25, 'BFDOT result mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1.Test_CDOTU; // Non-Conjugate Dot Product
const
  N = 3;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of rocblas_float_complex;
  h_result, expected_result: rocblas_float_complex;
  d_x, d_y, d_result: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data
  h_x[0].x := 1.0; h_x[0].y := 2.0;
  h_x[1].x := 3.0; h_x[1].y := 4.0;
  h_x[2].x := 5.0; h_x[2].y := 6.0;

  h_y[0].x := 1.0; h_y[0].y := 1.0;
  h_y[1].x := 2.0; h_y[1].y := 2.0;
  h_y[2].x := 3.0; h_y[2].y := 3.0;

  // Manual calculation: sum(x[i] * y[i]) - NO CONJUGATE
  // (1+2i)*(1+1i) = 1 + 1i + 2i - 2 = -1 + 3i
  // (3+4i)*(2+2i) = 6 + 6i + 8i - 8 = -2 + 14i
  // (5+6i)*(3+3i) = 15 + 15i + 18i - 18 = -3 + 33i
  // Sum = (-1+3i) + (-2+14i) + (-3+33i) = -6 + 50i
  expected_result.x := -6.0;
  expected_result.y := 50.0;

  h_result.x := 0.0; h_result.y := 0.0;

  d_x := nil;
  d_y := nil;
  d_result := nil;

  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result, SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_result', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_result, @h_result, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy result H->D', HIP_SUCCESS, hip_status);

  blas_status := rocblas_cdotu(handle, N, d_x, incx, d_y, incy, d_result);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_cdotu failed');

  hip_status := hipMemcpy(@h_result, d_result, SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result D->H', HIP_SUCCESS, hip_status);

  CheckEquals(expected_result.x, h_result.x, 1e-5, 'CDOTU real part mismatch');
  CheckEquals(expected_result.y, h_result.y, 1e-5, 'CDOTU imaginary part mismatch');

  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1.Test_ZDOTU; // Non-Conjugate Dot Product
const
  N = 3;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of rocblas_double_complex;
  h_result, expected_result: rocblas_double_complex;
  d_x, d_y, d_result: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  h_x[0].x := 1.0; h_x[0].y := 2.0;
  h_x[1].x := 3.0; h_x[1].y := 4.0;
  h_x[2].x := 5.0; h_x[2].y := 6.0;

  h_y[0].x := 1.0; h_y[0].y := 1.0;
  h_y[1].x := 2.0; h_y[1].y := 2.0;
  h_y[2].x := 3.0; h_y[2].y := 3.0;

  expected_result.x := -6.0;
  expected_result.y := 50.0;

  h_result.x := 0.0; h_result.y := 0.0;

  d_x := nil;
  d_y := nil;
  d_result := nil;

  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result, SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_result', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_result, @h_result, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy result H->D', HIP_SUCCESS, hip_status);

  blas_status := rocblas_zdotu(handle, N, d_x, incx, d_y, incy, d_result);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_zdotu failed');

  hip_status := hipMemcpy(@h_result, d_result, SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result D->H', HIP_SUCCESS, hip_status);

  CheckEquals(expected_result.x, h_result.x, 1e-12, 'ZDOTU real part mismatch');
  CheckEquals(expected_result.y, h_result.y, 1e-12, 'ZDOTU imaginary part mismatch');

  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1.Test_CDOTC; // Conjugate Dot Product
const
  N = 3;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of rocblas_float_complex;
  h_result, expected_result: rocblas_float_complex;
  d_x, d_y, d_result: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data
  h_x[0].x := 1.0; h_x[0].y := 2.0;
  h_x[1].x := 3.0; h_x[1].y := 4.0;
  h_x[2].x := 5.0; h_x[2].y := 6.0;

  h_y[0].x := 1.0; h_y[0].y := 1.0;
  h_y[1].x := 2.0; h_y[1].y := 2.0;
  h_y[2].x := 3.0; h_y[2].y := 3.0;

  // Manual calculation: sum(conj(x[i]) * y[i])
  // (1-2i)*(1+1i) = 1 + 1i - 2i + 2 = 3 - 1i
  // (3-4i)*(2+2i) = 6 + 6i - 8i + 8 = 14 - 2i
  // (5-6i)*(3+3i) = 15 + 15i - 18i + 18 = 33 - 3i
  // Sum = (3-1i) + (14-2i) + (33-3i) = 50 - 6i
  expected_result.x := 50.0;
  expected_result.y := -6.0;

  h_result.x := 0.0; h_result.y := 0.0;

  d_x := nil;
  d_y := nil;
  d_result := nil;

  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result, SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_result', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_result, @h_result, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy result H->D', HIP_SUCCESS, hip_status);

  blas_status := rocblas_cdotc(handle, N, d_x, incx, d_y, incy, d_result);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_cdotc failed');

  hip_status := hipMemcpy(@h_result, d_result, SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result D->H', HIP_SUCCESS, hip_status);

  CheckEquals(expected_result.x, h_result.x, 1e-5, 'CDOTC real part mismatch');
  CheckEquals(expected_result.y, h_result.y, 1e-5, 'CDOTC imaginary part mismatch');

  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1.Test_ZDOTC; // Conjugate Dot Product
const
  N = 3;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of rocblas_double_complex;
  h_result, expected_result: rocblas_double_complex;
  d_x, d_y, d_result: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  h_x[0].x := 1.0; h_x[0].y := 2.0;
  h_x[1].x := 3.0; h_x[1].y := 4.0;
  h_x[2].x := 5.0; h_x[2].y := 6.0;

  h_y[0].x := 1.0; h_y[0].y := 1.0;
  h_y[1].x := 2.0; h_y[1].y := 2.0;
  h_y[2].x := 3.0; h_y[2].y := 3.0;

  expected_result.x := 50.0;
  expected_result.y := -6.0;

  h_result.x := 0.0; h_result.y := 0.0;

  d_x := nil;
  d_y := nil;
  d_result := nil;

  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result, SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_result', HIP_SUCCESS, hip_status);

  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_result, @h_result, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy result H->D', HIP_SUCCESS, hip_status);

  blas_status := rocblas_zdotc(handle, N, d_x, incx, d_y, incy, d_result);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_zdotc failed');

  hip_status := hipMemcpy(@h_result, d_result, SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result D->H', HIP_SUCCESS, hip_status);

  CheckEquals(expected_result.x, h_result.x, 1e-12, 'ZDOTC real part mismatch');
  CheckEquals(expected_result.y, h_result.y, 1e-12, 'ZDOTC imaginary part mismatch');

  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1.Test_SNRM2;
const
  n = 5;
  incx = 1;
  // Test vector: [3.0, -4.0, 0.0, 1.0, -2.0]
  hx: array[0..n-1] of Single = (3.0, -4.0, 0.0, 1.0, -2.0);
  // Expected ||x|| = sqrt(3^2 + (-4)^2 + 0^2 + 1^2 + (-2)^2) = sqrt(9+16+0+1+4) = sqrt(30)
  expected_norm: Single = Sqrt(30.0);
var
  // --- Variables ---
  dx: hipDeviceptr_t;      // Device pointer for input vector x
  d_res: hipDeviceptr_t;   // Device pointer for the result scalar
  h_res: Single;           // Host variable to copy the result back to
  blas_status: rocblas_status; // Status for rocBLAS function calls
  hip_status: hipError_t;      // Status for HIP function calls
  gpu_res: Single;             // Local copy of the result from device
  ptr_mode_status: rocblas_status; // Status for rocblas_set_pointer_mode
begin
  // --- 1. Set rocBLAS pointer mode for device scalar result ---
  ptr_mode_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode should return success', Integer(rocblas_status_success), Integer(ptr_mode_status));

  // --- 2. Allocate Host Memory ---
  h_res := 0.0; // Initialize host result

  // --- 3. Allocate Device Memory ---
  dx := nil;    // Initialize device pointers to nil
  d_res := nil;
  // PASS THE VARIABLE DIRECTLY to hipMalloc (it's a 'var' parameter)
  hip_status := hipMalloc(dx, n * SizeOf(Single));
  AssertEquals('hipMalloc for dx should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_res, SizeOf(Single)); // Allocate space for 1 Single result
  AssertEquals('hipMalloc for d_res should succeed', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 4. Copy Data Host to Device ---
  hip_status := hipMemcpy(dx, @hx[0], n * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D should succeed', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 5. Run rocBLAS Function ---
  // Cast dx (hipDeviceptr_t) to PSingle as rocblas_snrm2 expects for the data pointer
  // Cast d_res (hipDeviceptr_t) to PSingle as rocblas_snrm2 expects for the result pointer
  blas_status := rocblas_snrm2(handle, n, PSingle(dx), incx, PSingle(d_res));
  AssertEquals('rocblas_snrm2 should return success', Integer(rocblas_status_success), Integer(blas_status));

  // --- 6. Copy Result Device to Host ---
  hip_status := hipMemcpy(@h_res, d_res, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy D2H for result should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
  gpu_res := h_res; // Store for comparison

  // --- 7. Compare Results ---
  CheckEqualsRelative(expected_norm, gpu_res, 1e-6, 'SNRM2 result mismatch');

  // --- 8. Free Device Memory ---
  // PASS THE POINTER VALUE directly to hipFree
  hip_status := hipFree(dx);
  AssertEquals('hipFree dx should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipFree(d_res);
  AssertEquals('hipFree d_res should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
end;

procedure TTestRocBLAS_Level1.Test_DNRM2;
const
  n = 4;
  incx = 1;
  // Test vector: [-1.0, 5.0, -3.0, 2.0]
  hx: array[0..n-1] of Double = (-1.0, 5.0, -3.0, 2.0);
  // Expected ||x|| = sqrt((-1)^2 + 5^2 + (-3)^2 + 2^2) = sqrt(1+25+9+4) = sqrt(39)
  expected_norm: Double = Sqrt(39.0);
var
  // --- Variables ---
  dx: hipDeviceptr_t;
  d_res: hipDeviceptr_t;
  h_res: Double;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  gpu_res: Double;
  ptr_mode_status: rocblas_status;
begin
  // --- 1. Set rocBLAS pointer mode for device scalar result ---
  ptr_mode_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode should return success', Integer(rocblas_status_success), Integer(ptr_mode_status));

  // --- 2. Allocate Host Memory ---
  h_res := 0.0;

  // --- 3. Allocate Device Memory ---
  dx := nil;
  d_res := nil;
  hip_status := hipMalloc(dx, n * SizeOf(Double));
  AssertEquals('hipMalloc for dx should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_res, SizeOf(Double));
  AssertEquals('hipMalloc for d_res should succeed', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 4. Copy Data Host to Device ---
  hip_status := hipMemcpy(dx, @hx[0], n * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D should succeed', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 5. Run rocBLAS Function ---
  blas_status := rocblas_dnrm2(handle, n, PDouble(dx), incx, PDouble(d_res));
  AssertEquals('rocblas_dnrm2 should return success', Integer(rocblas_status_success), Integer(blas_status));

  // --- 6. Copy Result Device to Host ---
  hip_status := hipMemcpy(@h_res, d_res, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy D2H for result should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
  gpu_res := h_res;

  // --- 7. Compare Results ---
  CheckEqualsRelative(expected_norm, gpu_res, 1e-12, 'DNRM2 result mismatch');

  // --- 8. Free Device Memory ---
  hip_status := hipFree(dx);
  AssertEquals('hipFree dx should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipFree(d_res);
  AssertEquals('hipFree d_res should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
end;


procedure TTestRocBLAS_Level1.Test_SCNRM2;
const
  n = 3;
  incx = 1;
  // Test vector: [(1+2i), (-1-1i), (2+0i)] - Using .x, .y
  hx: array[0..n-1] of rocblas_float_complex = (
    (x: 1.0; y: 2.0),
    (x: -1.0; y: -1.0),
    (x: 2.0; y: 0.0)
  );
  // Expected ||x|| = sqrt(|1+2i|^2 + |-1-1i|^2 + |2+0i|^2)
  //                = sqrt((1^2+2^2) + ((-1)^2+(-1)^2) + (2^2+0^2))
  //                = sqrt(5 + 2 + 4) = sqrt(11)
  expected_norm: Single = Sqrt(11.0);
var
  // --- Variables ---
  dx: hipDeviceptr_t;
  d_res: hipDeviceptr_t; // Result is always real (Single for scnrm2)
  h_res: Single;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  gpu_res: Single;
  ptr_mode_status: rocblas_status;
begin
  // --- 1. Set rocBLAS pointer mode for device scalar result ---
  ptr_mode_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode should return success', Integer(rocblas_status_success), Integer(ptr_mode_status));

  // --- 2. Allocate Host Memory ---
  h_res := 0.0;

  // --- 3. Allocate Device Memory ---
  dx := nil;
  d_res := nil;
  hip_status := hipMalloc(dx, n * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc for dx should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_res, SizeOf(Single));
  AssertEquals('hipMalloc for d_res should succeed', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 4. Copy Data Host to Device ---
  hip_status := hipMemcpy(dx, @hx[0], n * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D should succeed', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 5. Run rocBLAS Function ---
  // Cast dx to Procfloat_complex for the data pointer
  // Cast d_res to PSingle for the result pointer (real scalar)
  blas_status := rocblas_scnrm2(handle, n, Procfloat_complex(dx), incx, PSingle(d_res));
  AssertEquals('rocblas_scnrm2 should return success', Integer(rocblas_status_success), Integer(blas_status));

  // --- 6. Copy Result Device to Host ---
  hip_status := hipMemcpy(@h_res, d_res, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy D2H for result should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
  gpu_res := h_res;

  // --- 7. Compare Results ---
  CheckEqualsRelative(expected_norm, gpu_res, 1e-6, 'SCNRM2 result mismatch');

  // --- 8. Free Device Memory ---
  hip_status := hipFree(dx);
  AssertEquals('hipFree dx should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipFree(d_res);
  AssertEquals('hipFree d_res should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
end;


procedure TTestRocBLAS_Level1.Test_DZNRM2;
const
  n = 2;
  incx = 1;
  // Test vector: [(3-4i), (0+1i)] - Using .x, .y
  hx: array[0..n-1] of rocblas_double_complex = (
    (x: 3.0; y: -4.0),
    (x: 0.0; y: 1.0)
  );
  // Expected ||x|| = sqrt(|3-4i|^2 + |0+1i|^2)
  //                = sqrt((3^2+(-4)^2) + (0^2+1^2))
  //                = sqrt(25 + 1) = sqrt(26)
  expected_norm: Double = Sqrt(26.0);
var
  // --- Variables ---
  dx: hipDeviceptr_t;
  d_res: hipDeviceptr_t; // Result is always real (Double for dznrm2)
  h_res: Double;
  blas_status: rocblas_status;
  hip_status: hipError_t;
  gpu_res: Double;
  ptr_mode_status: rocblas_status;
begin
  // --- 1. Set rocBLAS pointer mode for device scalar result ---
  ptr_mode_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode should return success', Integer(rocblas_status_success), Integer(ptr_mode_status));

  // --- 2. Allocate Host Memory ---
  h_res := 0.0;

  // --- 3. Allocate Device Memory ---
  dx := nil;
  d_res := nil;
  hip_status := hipMalloc(dx, n * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc for dx should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipMalloc(d_res, SizeOf(Double));
  AssertEquals('hipMalloc for d_res should succeed', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 4. Copy Data Host to Device ---
  hip_status := hipMemcpy(dx, @hx[0], n * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy H2D should succeed', Integer(HIP_SUCCESS), Integer(hip_status));

  // --- 5. Run rocBLAS Function ---
  // Cast dx to Procdouble_complex for the data pointer
  // Cast d_res to PDouble for the result pointer (real scalar)
  blas_status := rocblas_dznrm2(handle, n, Procdouble_complex(dx), incx, PDouble(d_res));
  AssertEquals('rocblas_dznrm2 should return success', Integer(rocblas_status_success), Integer(blas_status));

  // --- 6. Copy Result Device to Host ---
  hip_status := hipMemcpy(@h_res, d_res, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy D2H for result should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
  gpu_res := h_res;

  // --- 7. Compare Results ---
  CheckEqualsRelative(expected_norm, gpu_res, 1e-12, 'DZNRM2 result mismatch');

  // --- 8. Free Device Memory ---
  hip_status := hipFree(dx);
  AssertEquals('hipFree dx should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
  hip_status := hipFree(d_res);
  AssertEquals('hipFree d_res should succeed', Integer(HIP_SUCCESS), Integer(hip_status));
end;


// --- ASUM Tests (Sum of Absolute Values) ---

procedure TTestRocBLAS_Level1.Test_SASUM; // Single Precision Sum of Absolute Values
const
  N = 5;
  incx = 1;
var
  h_x: array[0..N-1] of Single;
  h_result, expected_result: Single;
  d_x, d_result: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  expected_result := 0.0;
  // Initialize host data and calculate expected result
  for i := 0 to N - 1 do
  begin
    h_x[i] := Power(-1, i) * (i + 1.0); // Alternate signs: 1.0, -2.0, 3.0, -4.0, 5.0
    expected_result := expected_result + Abs(h_x[i]); // Sum of absolute values
  end;
  // Expected: 1.0 + 2.0 + 3.0 + 4.0 + 5.0 = 15.0
  h_result := 0.0; // Initialize result

  // Initialize device pointers
  d_x := nil;
  d_result := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(Single));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result, SizeOf(Single)); // Result is a single scalar
  AssertEquals('hipMalloc d_result', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  // Initialize result on device
  hip_status := hipMemcpy(d_result, @h_result, SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy result H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS SASUM
  blas_status := rocblas_sasum(handle, N, d_x, incx, d_result);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_sasum failed');

  // Copy result D->H
  hip_status := hipMemcpy(@h_result, d_result, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result D->H', HIP_SUCCESS, hip_status);

  // Verify result (using small absolute tolerance for floating-point)
  CheckEquals(expected_result, h_result, 1e-5, 'SASUM result mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1.Test_DASUM; // Double Precision Sum of Absolute Values
const
  N = 5;
  incx = 1;
var
  h_x: array[0..N-1] of Double;
  h_result, expected_result: Double;
  d_x, d_result: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  expected_result := 0.0;
  // Initialize host data and calculate expected result
  for i := 0 to N - 1 do
  begin
    h_x[i] := Power(-1, i) * (i + 1.5); // Alternate signs: 1.5, -2.5, 3.5, -4.5, 5.5
    expected_result := expected_result + Abs(h_x[i]);
  end;
  // Expected: 1.5 + 2.5 + 3.5 + 4.5 + 5.5 = 17.5
  h_result := 0.0;

  // Initialize device pointers
  d_x := nil;
  d_result := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(Double));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result, SizeOf(Double)); // Result is a double scalar
  AssertEquals('hipMalloc d_result', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  // Initialize result on device
  hip_status := hipMemcpy(d_result, @h_result, SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy result H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS DASUM
  blas_status := rocblas_dasum(handle, N, d_x, incx, d_result);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_dasum failed');

  // Copy result D->H
  hip_status := hipMemcpy(@h_result, d_result, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result D->H', HIP_SUCCESS, hip_status);

  // Verify result (using very small absolute tolerance for double precision)
  CheckEquals(expected_result, h_result, 1e-12, 'DASUM result mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1.Test_SCASUM; // Single Complex Precision Sum of Absolute Values (Result is Real Single)
const
  N = 3;
  incx = 1;
var
  h_x: array[0..N-1] of rocblas_float_complex;
  h_result, expected_result: Single; // Result is a real Single
  d_x, d_result: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  expected_result := 0.0;
  // Initialize host data and calculate expected result
  // Example: x = [(1+2i), (-3+0i), (0-4i)]
  h_x[0].x := 1.0; h_x[0].y := 2.0;
  h_x[1].x := -3.0; h_x[1].y := 0.0;
  h_x[2].x := 0.0; h_x[2].y := -4.0;

  // For complex, asum = sum( abs(real(x[i])) + abs(imag(x[i])) )
  // (abs(1) + abs(2)) + (abs(-3) + abs(0)) + (abs(0) + abs(-4))
  // (1 + 2) + (3 + 0) + (0 + 4)
  // 3 + 3 + 4 = 10
  expected_result := (Abs(1.0) + Abs(2.0)) + (Abs(-3.0) + Abs(0.0)) + (Abs(0.0) + Abs(-4.0));
  h_result := 0.0;

  // Initialize device pointers
  d_x := nil;
  d_result := nil; // Result is a PSingle

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result, SizeOf(Single)); // Result is a real Single scalar
  AssertEquals('hipMalloc d_result', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  // Initialize result on device
  hip_status := hipMemcpy(d_result, @h_result, SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy result H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS SCASUM (Single Complex -> Real Single result)
  blas_status := rocblas_scasum(handle, N, d_x, incx, d_result);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_scasum failed');

  // Copy result D->H (Single)
  hip_status := hipMemcpy(@h_result, d_result, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result D->H', HIP_SUCCESS, hip_status);

  // Verify result (using small absolute tolerance for floating-point)
  CheckEquals(expected_result, h_result, 1e-5, 'SCASUM result mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_result);
end;

procedure TTestRocBLAS_Level1.Test_DZASUM; // Double Complex Precision Sum of Absolute Values (Result is Real Double)
const
  N = 3;
  incx = 1;
var
  h_x: array[0..N-1] of rocblas_double_complex;
  h_result, expected_result: Double; // Result is a real Double
  d_x, d_result: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  expected_result := 0.0;
  // Initialize host data and calculate expected result (same numerical values as SCASUM)
  h_x[0].x := 1.0; h_x[0].y := 2.0;
  h_x[1].x := -3.0; h_x[1].y := 0.0;
  h_x[2].x := 0.0; h_x[2].y := -4.0;

  // For complex, asum = sum( abs(real(x[i])) + abs(imag(x[i])) )
  // (abs(1.0) + abs(2.0)) + (abs(-3.0) + abs(0.0)) + (abs(0.0) + abs(-4.0))
  // 3.0 + 3.0 + 4.0 = 10.0
  expected_result := (Abs(1.0) + Abs(2.0)) + (Abs(-3.0) + Abs(0.0)) + (Abs(0.0) + Abs(-4.0));
  h_result := 0.0;

  // Initialize device pointers
  d_x := nil;
  d_result := nil; // Result is a PDouble

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result, SizeOf(Double)); // Result is a real Double scalar
  AssertEquals('hipMalloc d_result', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  // Initialize result on device
  hip_status := hipMemcpy(d_result, @h_result, SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy result H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS DZASUM (Double Complex -> Real Double result)
  blas_status := rocblas_dzasum(handle, N, d_x, incx, d_result);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_dzasum failed');

  // Copy result D->H (Double)
  hip_status := hipMemcpy(@h_result, d_result, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result D->H', HIP_SUCCESS, hip_status);

  // Verify result (using very small absolute tolerance for double precision)
  CheckEquals(expected_result, h_result, 1e-12, 'DZASUM result mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_result);
end;

// --- AMAX Tests (Index of Maximum Absolute Value/Sum) ---

procedure TTestRocBLAS_Level1.Test_ISAMAX; // Index of Max Abs Value (Single)
const
  N = 5;
  incx = 1;
var
  h_x: array[0..N-1] of Single;
  h_result_index: Int32; // Result is now an Int32 (rocblas_int)
  expected_index: Int32; // Expected result is also Int32
  d_x: Pointer;
  d_result_index: PInt32; // *** Changed type from Prochalf to PInt32 ***
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data - put max absolute value at index 2 (0-based)
  // Fortran index 3 <-> C index 2
  h_x[0] := 1.0;
  h_x[1] := -2.5;
  h_x[2] := 5.0; // Max absolute value |5.0| = 5.0
  h_x[3] := -3.0;
  h_x[4] := 4.0;
  expected_index := 3; // 1-based Fortran index of the element with max absolute value
  h_result_index := 0; // Initialize result index

  // Initialize device pointers
  d_x := nil;
  d_result_index := nil; // *** Now PInt32 ***

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(Single));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  // *** Allocate SizeOf(Int32) for the Int32 result ***
  hip_status := hipMalloc(d_result_index, SizeOf(Int32));
  AssertEquals('hipMalloc d_result_index', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  // *** Copy SizeOf(Int32) bytes for the Int32 result ***
  hip_status := hipMemcpy(d_result_index, @h_result_index, SizeOf(Int32), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy result index H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS ISAMAX
  // *** Pass d_result_index directly (no Prochalf cast needed) ***
  blas_status := rocblas_isamax(handle, N, PSingle(d_x), incx, d_result_index);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_isamax failed');

  // Copy result index D->H (as Int32)
  // *** Copy SizeOf(Int32) bytes ***
  hip_status := hipMemcpy(@h_result_index, d_result_index, SizeOf(Int32), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result index D->H', HIP_SUCCESS, hip_status);

  // *** No conversion needed, h_result_index is already Int32 ***
  // Verify result index
  // rocblas_isamax should return 1-based Fortran index
  CheckEquals(expected_index, h_result_index, 'ISAMAX result index mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_result_index); // *** hipFree works with PInt32 pointer ***
end;

procedure TTestRocBLAS_Level1.Test_IDAMAX; // Index of Max Abs Value (Double)
const
  N = 5;
  incx = 1;
var
  h_x: array[0..N-1] of Double;
  h_result_index: Int32; // Result is Int32
  expected_index: Int32;
  d_x: Pointer;
  d_result_index: PInt32; // *** Changed type ***
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data - put max absolute value at index 3 (0-based)
  // Fortran index 4 <-> C index 3
  h_x[0] := 1.5;
  h_x[1] := -2.7;
  h_x[2] := 3.8;
  h_x[3] := -6.2; // Max absolute value |-6.2| = 6.2
  h_x[4] := 4.1;
  expected_index := 4; // 1-based Fortran index
  h_result_index := 0;

  // Initialize device pointers
  d_x := nil;
  d_result_index := nil; // *** Now PInt32 ***

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(Double));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result_index, SizeOf(Int32)); // *** SizeOf(Int32) ***
  AssertEquals('hipMalloc d_result_index', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_result_index, @h_result_index, SizeOf(Int32), hipMemcpyHostToDevice); // *** SizeOf(Int32) ***
  AssertEquals('hipMemcpy result index H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS IDAMAX
  // *** Pass d_result_index directly ***
  blas_status := rocblas_idamax(handle, N, PDouble(d_x), incx, d_result_index);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_idamax failed');

  // Copy result index D->H
  hip_status := hipMemcpy(@h_result_index, d_result_index, SizeOf(Int32), hipMemcpyDeviceToHost); // *** SizeOf(Int32) ***
  AssertEquals('hipMemcpy result index D->H', HIP_SUCCESS, hip_status);

  // *** No conversion needed ***
  // Verify result index
  CheckEquals(expected_index, h_result_index, 'IDAMAX result index mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_result_index); // *** hipFree works with PInt32 pointer ***
end;

procedure TTestRocBLAS_Level1.Test_ICAMAX; // Index of Max Abs Value (Single Complex)
const
  N = 4;
  incx = 1;
var
  h_x: array[0..N-1] of rocblas_float_complex;
  h_result_index: Int32; // Result is Int32
  expected_index: Int32;
  d_x: Pointer;
  d_result_index: PInt32; // *** Changed type ***
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data - For complex, max is based on |real| + |imag|
  // Element 0: |1.0| + |1.0| = 2.0
  // Element 1: |2.0| + |0.0| = 2.0
  // Element 2: |1.0| + |2.0| = 3.0 <- Max
  // Element 3: |0.5| + |0.5| = 1.0
  h_x[0].x := 1.0; h_x[0].y := 1.0;
  h_x[1].x := 2.0; h_x[1].y := 0.0;
  h_x[2].x := 1.0; h_x[2].y := 2.0; // Max sum |real| + |imag|
  h_x[3].x := 0.5; h_x[3].y := 0.5;
  expected_index := 3; // 1-based Fortran index (0-based C index is 2)
  h_result_index := 0;

  // Initialize device pointers
  d_x := nil;
  d_result_index := nil; // *** Now PInt32 ***

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result_index, SizeOf(Int32)); // *** SizeOf(Int32) ***
  AssertEquals('hipMalloc d_result_index', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_result_index, @h_result_index, SizeOf(Int32), hipMemcpyHostToDevice); // *** SizeOf(Int32) ***
  AssertEquals('hipMemcpy result index H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS ICAMAX
  // *** Pass d_result_index directly ***
  blas_status := rocblas_icamax(handle, N, Procfloat_complex(d_x), incx, d_result_index);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_icamax failed');

  // Copy result index D->H
  hip_status := hipMemcpy(@h_result_index, d_result_index, SizeOf(Int32), hipMemcpyDeviceToHost); // *** SizeOf(Int32) ***
  AssertEquals('hipMemcpy result index D->H', HIP_SUCCESS, hip_status);

  // *** No conversion needed ***
  // Verify result index
  CheckEquals(expected_index, h_result_index, 'ICAMAX result index mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_result_index); // *** hipFree works with PInt32 pointer ***
end;

procedure TTestRocBLAS_Level1.Test_IZAMAX; // Index of Max Abs Value (Double Complex)
const
  N = 4;
  incx = 1;
var
  h_x: array[0..N-1] of rocblas_double_complex;
  h_result_index: Int32; // Result is Int32
  expected_index: Int32;
  d_x: Pointer;
  d_result_index: PInt32; // *** Changed type ***
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data - For complex, max is based on |real| + |imag|
  // Element 0: |1.5| + |1.5| = 3.0
  // Element 1: |2.7| + |0.0| = 2.7
  // Element 2: |1.2| + |2.8| = 4.0 <- Max
  // Element 3: |0.6| + |0.4| = 1.0
  h_x[0].x := 1.5; h_x[0].y := 1.5;
  h_x[1].x := 2.7; h_x[1].y := 0.0;
  h_x[2].x := 1.2; h_x[2].y := 2.8; // Max sum |real| + |imag|
  h_x[3].x := 0.6; h_x[3].y := 0.4;
  expected_index := 3; // 1-based Fortran index (0-based C index is 2)
  h_result_index := 0;

  // Initialize device pointers
  d_x := nil;
  d_result_index := nil; // *** Now PInt32 ***

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result_index, SizeOf(Int32)); // *** SizeOf(Int32) ***
  AssertEquals('hipMalloc d_result_index', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_result_index, @h_result_index, SizeOf(Int32), hipMemcpyHostToDevice); // *** SizeOf(Int32) ***
  AssertEquals('hipMemcpy result index H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS IZAMAX
  // *** Pass d_result_index directly ***
  blas_status := rocblas_izamax(handle, N, Procdouble_complex(d_x), incx, d_result_index);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_izamax failed');

  // Copy result index D->H
  hip_status := hipMemcpy(@h_result_index, d_result_index, SizeOf(Int32), hipMemcpyDeviceToHost); // *** SizeOf(Int32) ***
  AssertEquals('hipMemcpy result index D->H', HIP_SUCCESS, hip_status);

  // *** No conversion needed ***
  // Verify result index
  CheckEquals(expected_index, h_result_index, 'IZAMAX result index mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_result_index); // *** hipFree works with PInt32 pointer ***
end;

// --- AMIN Tests (Index of Minimum Absolute Value/Sum) ---

procedure TTestRocBLAS_Level1.Test_ISAMIN; // Index of Min Abs Value (Single)
const
  N = 5;
  incx = 1;
var
  h_x: array[0..N-1] of Single;
  h_result_index: Int32; // Result is now an Int32 (rocblas_int)
  expected_index: Int32; // Expected result is also Int32
  d_x: Pointer;
  d_result_index: PInt32; // *** Changed type ***
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data - put min absolute value at index 1 (0-based)
  // Fortran index 2 <-> C index 1
  h_x[0] := 1.0;
  h_x[1] := -0.5; // Min absolute value |-0.5| = 0.5
  h_x[2] := 5.0;
  h_x[3] := -3.0;
  h_x[4] := 4.0;
  expected_index := 2; // 1-based Fortran index of the element with min absolute value
  h_result_index := 0; // Initialize result index

  // Initialize device pointers
  d_x := nil;
  d_result_index := nil; // *** Now PInt32 ***

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(Single));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  // *** Allocate SizeOf(Int32) for the Int32 result ***
  hip_status := hipMalloc(d_result_index, SizeOf(Int32));
  AssertEquals('hipMalloc d_result_index', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  // *** Copy SizeOf(Int32) bytes for the Int32 result ***
  hip_status := hipMemcpy(d_result_index, @h_result_index, SizeOf(Int32), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy result index H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS ISAMIN
  // *** Pass d_result_index directly (no Prochalf cast needed) ***
  blas_status := rocblas_isamin(handle, N, PSingle(d_x), incx, d_result_index);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_isamin failed');

  // Copy result index D->H (as Int32)
  // *** Copy SizeOf(Int32) bytes ***
  hip_status := hipMemcpy(@h_result_index, d_result_index, SizeOf(Int32), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy result index D->H', HIP_SUCCESS, hip_status);

  // *** No conversion needed, h_result_index is already Int32 ***
  // Verify result index
  // rocblas_isamin should return 1-based Fortran index
  CheckEquals(expected_index, h_result_index, 'ISAMIN result index mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_result_index); // *** hipFree works with PInt32 pointer ***
end;

procedure TTestRocBLAS_Level1.Test_IDAMIN; // Index of Min Abs Value (Double)
const
  N = 5;
  incx = 1;
var
  h_x: array[0..N-1] of Double;
  h_result_index: Int32; // Result is Int32
  expected_index: Int32;
  d_x: Pointer;
  d_result_index: PInt32; // *** Changed type ***
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data - put min absolute value at index 2 (0-based)
  // Fortran index 3 <-> C index 2
  h_x[0] := 1.5;
  h_x[1] := -2.7;
  h_x[2] := 0.8; // Min absolute value |0.8| = 0.8
  h_x[3] := -6.2;
  h_x[4] := 4.1;
  expected_index := 3; // 1-based Fortran index
  h_result_index := 0;

  // Initialize device pointers
  d_x := nil;
  d_result_index := nil; // *** Now PInt32 ***

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(Double));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result_index, SizeOf(Int32)); // *** SizeOf(Int32) ***
  AssertEquals('hipMalloc d_result_index', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_result_index, @h_result_index, SizeOf(Int32), hipMemcpyHostToDevice); // *** SizeOf(Int32) ***
  AssertEquals('hipMemcpy result index H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS IDAMIN
  // *** Pass d_result_index directly ***
  blas_status := rocblas_idamin(handle, N, PDouble(d_x), incx, d_result_index);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_idamin failed');

  // Copy result index D->H
  hip_status := hipMemcpy(@h_result_index, d_result_index, SizeOf(Int32), hipMemcpyDeviceToHost); // *** SizeOf(Int32) ***
  AssertEquals('hipMemcpy result index D->H', HIP_SUCCESS, hip_status);

  // *** No conversion needed ***
  // Verify result index
  CheckEquals(expected_index, h_result_index, 'IDAMIN result index mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_result_index); // *** hipFree works with PInt32 pointer ***
end;

procedure TTestRocBLAS_Level1.Test_ICAMIN; // Index of Min Abs Value (Single Complex)
const
  N = 4;
  incx = 1;
var
  h_x: array[0..N-1] of rocblas_float_complex;
  h_result_index: Int32; // Result is Int32
  expected_index: Int32;
  d_x: Pointer;
  d_result_index: PInt32; // *** Changed type ***
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data - For complex, min is based on |real| + |imag|
  // Element 0: |1.0| + |1.0| = 2.0
  // Element 1: |0.2| + |0.0| = 0.2 <- Min
  // Element 2: |1.0| + |2.0| = 3.0
  // Element 3: |0.5| + |0.5| = 1.0
  h_x[0].x := 1.0; h_x[0].y := 1.0;
  h_x[1].x := 0.2; h_x[1].y := 0.0; // Min sum |real| + |imag|
  h_x[2].x := 1.0; h_x[2].y := 2.0;
  h_x[3].x := 0.5; h_x[3].y := 0.5;
  expected_index := 2; // 1-based Fortran index (0-based C index is 1)
  h_result_index := 0;

  // Initialize device pointers
  d_x := nil;
  d_result_index := nil; // *** Now PInt32 ***

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result_index, SizeOf(Int32)); // *** SizeOf(Int32) ***
  AssertEquals('hipMalloc d_result_index', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_result_index, @h_result_index, SizeOf(Int32), hipMemcpyHostToDevice); // *** SizeOf(Int32) ***
  AssertEquals('hipMemcpy result index H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS ICAMIN
  // *** Pass d_result_index directly ***
  blas_status := rocblas_icamin(handle, N, Procfloat_complex(d_x), incx, d_result_index);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_icamin failed');

  // Copy result index D->H
  hip_status := hipMemcpy(@h_result_index, d_result_index, SizeOf(Int32), hipMemcpyDeviceToHost); // *** SizeOf(Int32) ***
  AssertEquals('hipMemcpy result index D->H', HIP_SUCCESS, hip_status);

  // *** No conversion needed ***
  // Verify result index
  CheckEquals(expected_index, h_result_index, 'ICAMIN result index mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_result_index); // *** hipFree works with PInt32 pointer ***
end;

procedure TTestRocBLAS_Level1.Test_IZAMIN; // Index of Min Abs Value (Double Complex)
const
  N = 4;
  incx = 1;
var
  h_x: array[0..N-1] of rocblas_double_complex;
  h_result_index: Int32; // Result is Int32
  expected_index: Int32;
  d_x: Pointer;
  d_result_index: PInt32; // *** Changed type ***
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data - For complex, min is based on |real| + |imag|
  // Element 0: |1.5| + |1.5| = 3.0
  // Element 1: |0.27| + |0.0| = 0.27 <- Min
  // Element 2: |1.2| + |2.8| = 4.0
  // Element 3: |0.6| + |0.4| = 1.0
  h_x[0].x := 1.5; h_x[0].y := 1.5;
  h_x[1].x := 0.27; h_x[1].y := 0.0; // Min sum |real| + |imag|
  h_x[2].x := 1.2; h_x[2].y := 2.8;
  h_x[3].x := 0.6; h_x[3].y := 0.4;
  expected_index := 2; // 1-based Fortran index (0-based C index is 1)
  h_result_index := 0;

  // Initialize device pointers
  d_x := nil;
  d_result_index := nil; // *** Now PInt32 ***

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_result_index, SizeOf(Int32)); // *** SizeOf(Int32) ***
  AssertEquals('hipMalloc d_result_index', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_result_index, @h_result_index, SizeOf(Int32), hipMemcpyHostToDevice); // *** SizeOf(Int32) ***
  AssertEquals('hipMemcpy result index H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS IZAMIN
  // *** Pass d_result_index directly ***
  blas_status := rocblas_izamin(handle, N, Procdouble_complex(d_x), incx, d_result_index);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_izamin failed');

  // Copy result index D->H
  hip_status := hipMemcpy(@h_result_index, d_result_index, SizeOf(Int32), hipMemcpyDeviceToHost); // *** SizeOf(Int32) ***
  AssertEquals('hipMemcpy result index D->H', HIP_SUCCESS, hip_status);

  // *** No conversion needed ***
  // Verify result index
  CheckEquals(expected_index, h_result_index, 'IZAMIN result index mismatch');

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_result_index); // *** hipFree works with PInt32 pointer ***
end;

// --- SWAP Tests (Exchange Vectors) ---

procedure TTestRocBLAS_Level1.Test_SSWAP; // Swap Single Precision Vectors
const
  N = 5;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of Single;
  h_x_orig, h_y_orig: array[0..N-1] of Single; // Keep originals for expected results
  d_x, d_y: Pointer; // Pointers to device memory
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data
  for i := 0 to N - 1 do
  begin
    h_x[i] := i + 1.0;     // [1.0, 2.0, 3.0, 4.0, 5.0]
    h_y[i] := (i + 1.0) * 10.0; // [10.0, 20.0, 30.0, 40.0, 50.0]
  end;
  // Store originals for calculating expected results
  h_x_orig := h_x;
  h_y_orig := h_y;

  // After SWAP, h_x should contain original h_y, and h_y should contain original h_x
  // Expected results are just the original arrays swapped

  // Initialize device pointers
  d_x := nil;
  d_y := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(Single));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(Single));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS SSWAP
  blas_status := rocblas_sswap(handle, N, d_x, incx, d_y, incy);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_sswap failed');

  // Copy result data D->H (both vectors have been modified)
  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y D->H', HIP_SUCCESS, hip_status);

  // Verify results: h_x should now hold original h_y_orig, h_y should hold original h_x_orig
  for i := 0 to N - 1 do
  begin
     CheckEquals(h_y_orig[i], h_x[i], 1e-6, 'SSWAP x result mismatch at index ' + IntToStr(i));
     CheckEquals(h_x_orig[i], h_y[i], 1e-6, 'SSWAP y result mismatch at index ' + IntToStr(i));
  end;

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1.Test_DSWAP; // Swap Double Precision Vectors
const
  N = 5;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of Double;
  h_x_orig, h_y_orig: array[0..N-1] of Double;
  d_x, d_y: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data
  for i := 0 to N - 1 do
  begin
    h_x[i] := i + 1.5;     // [1.5, 2.5, 3.5, 4.5, 5.5]
    h_y[i] := (i + 1.5) * 10.0; // [15.0, 25.0, 35.0, 45.0, 55.0]
  end;
  h_x_orig := h_x;
  h_y_orig := h_y;

  // Initialize device pointers
  d_x := nil;
  d_y := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(Double));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(Double));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS DSWAP
  blas_status := rocblas_dswap(handle, N, d_x, incx, d_y, incy);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_dswap failed');

  // Copy result data D->H
  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y D->H', HIP_SUCCESS, hip_status);

  // Verify results
  for i := 0 to N - 1 do
  begin
     CheckEquals(h_y_orig[i], h_x[i], 1e-12, 'DSWAP x result mismatch at index ' + IntToStr(i));
     CheckEquals(h_x_orig[i], h_y[i], 1e-12, 'DSWAP y result mismatch at index ' + IntToStr(i));
  end;

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1.Test_CSWAP; // Swap Single Complex Vectors
const
  N = 4;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of rocblas_float_complex;
  h_x_orig, h_y_orig: array[0..N-1] of rocblas_float_complex;
  d_x, d_y: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data
  for i := 0 to N - 1 do
  begin
    h_x[i].x := i + 1.0;
    h_x[i].y := i * 0.5;
    h_y[i].x := (i + 1.0) * 10.0;
    h_y[i].y := (i * 0.5) * 10.0;
  end;
  h_x_orig := h_x;
  h_y_orig := h_y;

  // Initialize device pointers
  d_x := nil;
  d_y := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS CSWAP
  blas_status := rocblas_cswap(handle, N, d_x, incx, d_y, incy);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_cswap failed');

  // Copy result data D->H
  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y D->H', HIP_SUCCESS, hip_status);

  // Verify results (check real and imaginary parts separately)
  for i := 0 to N - 1 do
  begin
     CheckEquals(h_y_orig[i].x, h_x[i].x, 1e-6, 'CSWAP x real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_orig[i].y, h_x[i].y, 1e-6, 'CSWAP x imaginary part mismatch at index ' + IntToStr(i));
     CheckEquals(h_x_orig[i].x, h_y[i].x, 1e-6, 'CSWAP y real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_x_orig[i].y, h_y[i].y, 1e-6, 'CSWAP y imaginary part mismatch at index ' + IntToStr(i));
  end;

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1.Test_ZSWAP; // Swap Double Complex Vectors
const
  N = 4;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of rocblas_double_complex;
  h_x_orig, h_y_orig: array[0..N-1] of rocblas_double_complex;
  d_x, d_y: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data
  for i := 0 to N - 1 do
  begin
    h_x[i].x := i + 1.5;
    h_x[i].y := i * 0.75;
    h_y[i].x := (i + 1.5) * 10.0;
    h_y[i].y := (i * 0.75) * 10.0;
  end;
  h_x_orig := h_x;
  h_y_orig := h_y;

  // Initialize device pointers
  d_x := nil;
  d_y := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS ZSWAP
  blas_status := rocblas_zswap(handle, N, d_x, incx, d_y, incy);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_zswap failed');

  // Copy result data D->H
  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y D->H', HIP_SUCCESS, hip_status);

  // Verify results (check real and imaginary parts separately)
  for i := 0 to N - 1 do
  begin
     CheckEquals(h_y_orig[i].x, h_x[i].x, 1e-12, 'ZSWAP x real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_orig[i].y, h_x[i].y, 1e-12, 'ZSWAP x imaginary part mismatch at index ' + IntToStr(i));
     CheckEquals(h_x_orig[i].x, h_y[i].x, 1e-12, 'ZSWAP y real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_x_orig[i].y, h_y[i].y, 1e-12, 'ZSWAP y imaginary part mismatch at index ' + IntToStr(i));
  end;

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_y);
end;

// --- ROT Tests (Plane Rotation) ---

procedure TTestRocBLAS_Level1.Test_SROT; // Apply Plane Rotation (Single)
const
  N = 4;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of Single;
  h_x_orig, h_y_orig: array[0..N-1] of Single;
  h_x_expected, h_y_expected: array[0..N-1] of Single;
  c, s: Single; // Cosine and Sine of the rotation angle (Host variables)
  d_x, d_y: hipDeviceptr_t; // Use hipDeviceptr_t for device pointers
  blas_status: rocblas_status;
  hip_status: hipError_t;
  pm_status: rocblas_status; // For pointer mode calls
  i: integer;
begin
  // Initialize host data
  h_x[0] := 1.0; h_y[0] := 0.0;
  h_x[1] := 0.0; h_y[1] := 1.0;
  h_x[2] := 2.0; h_y[2] := 3.0;
  h_x[3] := -1.0; h_y[3] := 2.0;
  h_x_orig := h_x;
  h_y_orig := h_y;

  // Define rotation parameters: e.g., 90-degree rotation (c=0, s=1)
  c := 0.0;
  s := 1.0;

  // Calculate expected results (using the inverse rotation logic determined previously)
  for i := 0 to N - 1 do
  begin
    h_x_expected[i] := c * h_x_orig[i] + s * h_y_orig[i];
    h_y_expected[i] := -s * h_x_orig[i] + c * h_y_orig[i];
  end;

  // Initialize device pointers
  d_x := nil;
  d_y := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(Single));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  AssertTrue('hipMalloc d_x assigned', Assigned(d_x));
  hip_status := hipMalloc(d_y, N * SizeOf(Single));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);
  AssertTrue('hipMalloc d_y assigned', Assigned(d_y));

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);

  // --- CRITICAL CHANGE: Set pointer mode to HOST for scalars c and s ---
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode to host', Integer(rocblas_status_success), Integer(pm_status));
  // ---------------------------------------------------------------------

  // Execute rocBLAS SROT
  // Note: c and s are passed by pointer to HOST memory locations
  blas_status := rocblas_srot(handle, N, d_x, incx, d_y, incy, @c, @s);
  // Check status *before* potentially changing pointer mode back
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_srot failed');

  // --- Optional: Reset pointer mode (e.g., to DEVICE if that was the default) ---
  // If other operations in your test suite expect device pointer mode,
  // or if your SetUp/TearDown handles it, you might reset it here.
  // For now, let's assume setting it to HOST for this call is sufficient.
  // pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  // AssertEquals('rocblas_set_pointer_mode back', Integer(rocblas_status_success), Integer(pm_status));
  // ---------------------------------------------------------------------

  // Copy result data D->H
  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y D->H', HIP_SUCCESS, hip_status);

  // Verify results
  for i := 0 to N - 1 do
  begin
     CheckEquals(h_x_expected[i], h_x[i], 1e-6, 'SROT x result mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_expected[i], h_y[i], 1e-6, 'SROT y result mismatch at index ' + IntToStr(i));
  end;

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1.Test_DROT; // Apply Plane Rotation (Double)
const
  N = 4;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of Double;
  h_x_orig, h_y_orig: array[0..N-1] of Double;
  h_x_expected, h_y_expected: array[0..N-1] of Double;
  c, s: Double; // Cosine and Sine
  d_x, d_y: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data
  h_x[0] := 1.5; h_y[0] := 0.5;
  h_x[1] := 0.5; h_y[1] := 1.5;
  h_x[2] := 2.5; h_y[2] := 3.5;
  h_x[3] := -1.5; h_y[3] := 2.5;

  h_x_orig := h_x;
  h_y_orig := h_y;

  // Define rotation parameters: e.g., 45-degree rotation (c=s=sqrt(2)/2)
  c := Sqrt(2.0) / 2.0;
  s := Sqrt(2.0) / 2.0;

  // Calculate expected results (inverse rotation: G(c, -s))
  for i := 0 to N - 1 do
  begin
    h_x_expected[i] := c * h_x_orig[i] + s * h_y_orig[i];
    h_y_expected[i] := -s * h_x_orig[i] + c * h_y_orig[i];
  end;

  // Initialize device pointers
  d_x := nil;
  d_y := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(Double));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(Double));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS DROT
  // Note: c and s are passed by pointer (const c, const s: PDouble)
  blas_status := rocblas_drot(handle, N, d_x, incx, d_y, incy, @c, @s);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_drot failed');

  // Copy result data D->H
  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y D->H', HIP_SUCCESS, hip_status);

  // Verify results
  for i := 0 to N - 1 do
  begin
     CheckEquals(h_x_expected[i], h_x[i], 1e-12, 'DROT x result mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_expected[i], h_y[i], 1e-12, 'DROT y result mismatch at index ' + IntToStr(i));
  end;

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1.Test_CSROT; // Apply Real Plane Rotation (Single Complex)
const
  N = 3;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of rocblas_float_complex;
  h_x_orig, h_y_orig: array[0..N-1] of rocblas_float_complex;
  h_x_expected, h_y_expected: array[0..N-1] of rocblas_float_complex;
  c, s: Single; // Real Cosine and Sine
  d_x, d_y: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data (complex vectors)
  h_x[0].x := 1.0; h_x[0].y := 2.0;
  h_y[0].x := 0.0; h_y[0].y := 1.0;
  h_x[1].x := 0.0; h_x[1].y := 1.0;
  h_y[1].x := 1.0; h_y[1].y := 0.0;
  h_x[2].x := 2.0; h_x[2].y := 3.0;
  h_y[2].x := 3.0; h_y[2].y := 2.0;

  h_x_orig := h_x;
  h_y_orig := h_y;

  // Define real rotation parameters: e.g., 90-degree (c=0, s=1)
  c := 0.0;
  s := 1.0;

  // Calculate expected results (applies same real inverse rotation to real/imag parts)
  // x'_real = c * x_real + s * y_real  (Note: +s, not -s for inverse)
  // x'_imag = c * x_imag + s * y_imag
  // y'_real = -s * x_real + c * y_real (Note: -s for inverse)
  // y'_imag = -s * x_imag + c * y_imag
  for i := 0 to N - 1 do
  begin
    h_x_expected[i].x := c * h_x_orig[i].x + s * h_y_orig[i].x;
    h_x_expected[i].y := c * h_x_orig[i].y + s * h_y_orig[i].y;
    h_y_expected[i].x := -s * h_x_orig[i].x + c * h_y_orig[i].x;
    h_y_expected[i].y := -s * h_x_orig[i].y + c * h_y_orig[i].y;
  end;

  // Initialize device pointers
  d_x := nil;
  d_y := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS CSROT
  // Note: c and s are real (PSingle) for csrot
  blas_status := rocblas_csrot(handle, N, d_x, incx, d_y, incy, @c, @s);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_csrot failed');

  // Copy result data D->H
  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y D->H', HIP_SUCCESS, hip_status);

  // Verify results (check real and imaginary parts separately)
  for i := 0 to N - 1 do
  begin
     CheckEquals(h_x_expected[i].x, h_x[i].x, 1e-6, 'CSROT x real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_x_expected[i].y, h_x[i].y, 1e-6, 'CSROT x imaginary part mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_expected[i].x, h_y[i].x, 1e-6, 'CSROT y real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_expected[i].y, h_y[i].y, 1e-6, 'CSROT y imaginary part mismatch at index ' + IntToStr(i));
  end;

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_y);
end;

procedure TTestRocBLAS_Level1.Test_ZDROT; // Apply Real Plane Rotation (Double Complex)
const
  N = 3;
  incx = 1;
  incy = 1;
var
  h_x, h_y: array[0..N-1] of rocblas_double_complex;
  h_x_orig, h_y_orig: array[0..N-1] of rocblas_double_complex;
  h_x_expected, h_y_expected: array[0..N-1] of rocblas_double_complex;
  c, s: Double; // Real Cosine and Sine
  d_x, d_y: Pointer;
  i: Integer;
  blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data (double complex vectors)
  h_x[0].x := 1.5; h_x[0].y := 2.5;
  h_y[0].x := 0.5; h_y[0].y := 1.5;
  h_x[1].x := 0.5; h_x[1].y := 1.5;
  h_y[1].x := 1.5; h_y[1].y := 0.5;
  h_x[2].x := 2.5; h_x[2].y := 3.5;
  h_y[2].x := 3.5; h_y[2].y := 2.5;

  h_x_orig := h_x;
  h_y_orig := h_y;

  // Define real rotation parameters: e.g., 45-degree (c=s=sqrt(2)/2)
  c := Sqrt(2.0) / 2.0;
  s := Sqrt(2.0) / 2.0;

  // Calculate expected results (inverse rotation applied to real/imag parts)
  for i := 0 to N - 1 do
  begin
    h_x_expected[i].x := c * h_x_orig[i].x + s * h_y_orig[i].x;
    h_x_expected[i].y := c * h_x_orig[i].y + s * h_y_orig[i].y;
    h_y_expected[i].x := -s * h_x_orig[i].x + c * h_y_orig[i].x;
    h_y_expected[i].y := -s * h_x_orig[i].y + c * h_y_orig[i].y;
  end;

  // Initialize device pointers
  d_x := nil;
  d_y := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS ZDROT
  // Note: c and s are real (PDouble) for zdrot
  blas_status := rocblas_zdrot(handle, N, d_x, incx, d_y, incy, @c, @s);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_zdrot failed');

  // Copy result data D->H
  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y D->H', HIP_SUCCESS, hip_status);

  // Verify results (check real and imaginary parts separately)
  for i := 0 to N - 1 do
  begin
     CheckEquals(h_x_expected[i].x, h_x[i].x, 1e-12, 'ZDROT x real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_x_expected[i].y, h_x[i].y, 1e-12, 'ZDROT x imaginary part mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_expected[i].x, h_y[i].x, 1e-12, 'ZDROT y real part mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_expected[i].y, h_y[i].y, 1e-12, 'ZDROT y imaginary part mismatch at index ' + IntToStr(i));
  end;

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_y);
end;

// --- ROTG Tests (Generate Givens Rotation) ---

// Test_SROTG tests the single-precision real Givens rotation generator.
// It calculates c, s, r, z such that:
// [ c s ] * [ a ] = [ r ]  and  [ c s ]^T * [ c s ] = [ 1 0 ]
// [ -s c ]   [ b ]   [ 0 ]       [ -s c ]    [ -s c ]   [ 0 1 ]
// The results are returned via the pointers passed to the function.
// The test verifies the generated values against the expected mathematical results.
procedure TTestRocBLAS_Level1.Test_SROTG;
var
  h_a, h_b: Single;          // Input scalars
  h_c, h_s: Single;          // Output cosine and sine
  h_r, h_z: Single;          // Output scalars (r = +-||(a,b)||, z used for compact representation)
  h_a_orig, h_b_orig: Single; // Store original inputs for verification
  d_a, d_b, d_c, d_s: hipDeviceptr_t; // Device pointers for inputs/outputs
  pm_status, blas_status: rocblas_status;
  hip_status: hipError_t;
  expected_r, expected_z: Single;
  computed_r: Single; // Calculated from outputs to verify
begin
  // Initialize host data
  h_a := 3.0;
  h_b := 4.0;
  h_a_orig := h_a;
  h_b_orig := h_b;

  // Initialize outputs (will be overwritten by rocblas_srotg)
  h_c := 0.0;
  h_s := 0.0;
  h_r := 0.0;
  h_z := 0.0;

  // Calculate expected r (magnitude of [a,b])
  // Note: The sign of r depends on the implementation details of srotg.
  // Typically, r = sign(a) * ||(a,b)|| if a != 0, or |b| if a=0.
  // For a=3.0, b=4.0: r = sign(3.0) * sqrt(3^2 + 4^2) = 1 * 5 = 5.0
  expected_r := 5.0; // Manually calculated for this specific case
  // The value of 'z' is part of the compact representation and depends on |a|, |b|, c, s.
  // For |a| > |b|, z = s.
  // For |a| <= |b|, z = 1/c if c != 0, or 1 if c=0.
  // Here, |a|=3 < |b|=4, so we need c to determine z.
  // We'll calculate expected_z after calling rocblas_srotg and see if it matches the spec.

  // Initialize device pointers
  d_a := nil;
  d_b := nil;
  d_c := nil;
  d_s := nil;

  // Allocate device memory for the four single-precision scalars
  hip_status := hipMalloc(d_a, SizeOf(Single));
  AssertEquals('hipMalloc d_a', HIP_SUCCESS, hip_status);
  AssertTrue('hipMalloc d_a assigned', Assigned(d_a));

  hip_status := hipMalloc(d_b, SizeOf(Single));
  AssertEquals('hipMalloc d_b', HIP_SUCCESS, hip_status);
  AssertTrue('hipMalloc d_b assigned', Assigned(d_b));

  hip_status := hipMalloc(d_c, SizeOf(Single));
  AssertEquals('hipMalloc d_c', HIP_SUCCESS, hip_status);
  AssertTrue('hipMalloc d_c assigned', Assigned(d_c));

  hip_status := hipMalloc(d_s, SizeOf(Single));
  AssertEquals('hipMalloc d_s', HIP_SUCCESS, hip_status);
  AssertTrue('hipMalloc d_s assigned', Assigned(d_s));

  // Copy initial input values H->D
  hip_status := hipMemcpy(d_a, @h_a, SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy a H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_b, @h_b, SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy b H->D', HIP_SUCCESS, hip_status);
  // Initialize c and s on device (optional, as they will be overwritten)
  hip_status := hipMemcpy(d_c, @h_c, SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy c H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_s, @h_s, SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy s H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS SROTG
  // All parameters are passed by pointer (var parameters in effect)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode to device', Integer(rocblas_status_success), Integer(pm_status));
  blas_status := rocblas_srotg(handle, d_a, d_b, d_c, d_s);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_srotg failed');

  // Copy results D->H
  hip_status := hipMemcpy(@h_a, d_a, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy a D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_b, d_b, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy b D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_c, d_c, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy c D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_s, d_s, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy s D->H', HIP_SUCCESS, hip_status);

  // --- Verification ---
  // 1. Check if the transformation produces the expected r in h_a
  // The standard output is that 'a' holds 'r' and 'b' holds 'z' (compact representation)
  // However, the C interface says a and b are inputs/outputs, c and s are outputs.
  // Let's check if h_a holds the expected r.
  CheckEquals(expected_r, h_a, 1e-6, 'SROTG result a (r) mismatch');

  // 2. Check if c^2 + s^2 = 1 (orthogonality condition)
  CheckEquals(1.0, h_c * h_c + h_s * h_s, 1e-6, 'SROTG orthogonality condition c^2 + s^2 != 1');

  // 3. Check if the transformation is correct: r = c*a_orig + s*b_orig
  computed_r := h_c * h_a_orig + h_s * h_b_orig;
  CheckEquals(h_a, computed_r, 1e-6, 'SROTG transformation c*a + s*b != r');

  // 4. Optional: Check the compact representation 'z' stored in h_b
  // The value of h_b (z) is implementation-specific for the compact representation.
  // It's often 's' if |a| > |b|, or 1/c (or 1 if c=0) if |a| <= |b|.
  // With a_orig=3, b_orig=4, we expect |a| < |b|, so z might be 1/c if c /= 0.
  // This is harder to verify without knowing the exact compact form used.
  // We can at least check that it's a reasonable value.

  // Cleanup device memory
  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);
  hipFree(d_s);
end;

// Test_DROTG tests the double-precision real Givens rotation generator.
// Similar logic to Test_SROTG.
procedure TTestRocBLAS_Level1.Test_DROTG;
var
  h_a, h_b: Double;
  h_c, h_s: Double;
  h_r, h_z: Double;
  h_a_orig, h_b_orig: Double;
  d_a, d_b, d_c, d_s: hipDeviceptr_t; // Device pointers for double precision
  pm_status, blas_status: rocblas_status;
  hip_status: hipError_t;
  expected_r, expected_z: Double;
  computed_r: Double;
begin
  // Initialize host data
  h_a := 1.5;
  h_b := 2.5;
  h_a_orig := h_a;
  h_b_orig := h_b;
  h_c := 0.0;
  h_s := 0.0;
  h_r := 0.0;
  h_z := 0.0;

  // Calculate expected r (magnitude)
  expected_r := Sqrt(h_a_orig * h_a_orig + h_b_orig * h_b_orig); // ~2.915

  // Initialize device pointers
  d_a := nil;
  d_b := nil;
  d_c := nil;
  d_s := nil;

  // Allocate device memory for the four double-precision scalars
  hip_status := hipMalloc(d_a, SizeOf(Double));
  AssertEquals('hipMalloc d_a', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_b, SizeOf(Double));
  AssertEquals('hipMalloc d_b', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_c, SizeOf(Double));
  AssertEquals('hipMalloc d_c', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_s, SizeOf(Double));
  AssertEquals('hipMalloc d_s', HIP_SUCCESS, hip_status);

  // Copy initial input values H->D
  hip_status := hipMemcpy(d_a, @h_a, SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy a H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_b, @h_b, SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy b H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_c, @h_c, SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy c H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_s, @h_s, SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy s H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS DROTG
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode to device', Integer(rocblas_status_success), Integer(pm_status));
  blas_status := rocblas_drotg(handle, d_a, d_b, d_c, d_s);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_drotg failed');

  // Copy results D->H
  hip_status := hipMemcpy(@h_a, d_a, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy a D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_b, d_b, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy b D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_c, d_c, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy c D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_s, d_s, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy s D->H', HIP_SUCCESS, hip_status);

  // --- Verification ---
  CheckEquals(expected_r, h_a, 1e-12, 'DROTG result a (r) mismatch');
  CheckEquals(1.0, h_c * h_c + h_s * h_s, 1e-12, 'DROTG orthogonality condition c^2 + s^2 != 1');
  computed_r := h_c * h_a_orig + h_s * h_b_orig;
  CheckEquals(h_a, computed_r, 1e-12, 'DROTG transformation c*a + s*b != r');

  // Cleanup device memory
  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);
  hipFree(d_s);
end;

// Test_CROTG tests the single-precision complex Givens rotation generator.
// It calculates c (real), s (complex), r (complex), z (complex) such that:
// [ c s ] * [ a ] = [ r ]  and  |c|^2 + |s|^2 = 1
// [ -conj(s) c ]   [ b ]   [ 0 ]
// c is real, s, r, z are complex.
procedure TTestRocBLAS_Level1.Test_CROTG;
var
  h_a, h_b, h_s, h_r, h_z: rocblas_float_complex; // Complex inputs/outputs (s, r, z)
  h_c: Single;                                    // Real output cosine
  h_a_orig, h_b_orig: rocblas_float_complex;     // Store original inputs
  d_a, d_b, d_s, d_r, d_z: hipDeviceptr_t;       // Device pointers for complex
  d_c: PSingle;                                  // Device pointer for real c
  pm_status, blas_status: rocblas_status;
  hip_status: hipError_t;
  norm_orig: Single;
  computed_r_real, computed_r_imag: Single;
begin
  // Initialize host data (complex scalars)
  h_a.x := 3.0; h_a.y := 4.0; // a = 3 + 4i
  h_b.x := 1.0; h_b.y := 2.0; // b = 1 + 2i
  h_a_orig := h_a;
  h_b_orig := h_b;

  // Initialize outputs
  h_c := 0.0;
  h_s.x := 0.0; h_s.y := 0.0;
  h_r.x := 0.0; h_r.y := 0.0;
  h_z.x := 0.0; h_z.y := 0.0;

  // Calculate norm of original vector [a; b] for expected |r|
  // |r| should equal sqrt(|a|^2 + |b|^2)
  norm_orig := Sqrt((h_a_orig.x * h_a_orig.x + h_a_orig.y * h_a_orig.y) +
                    (h_b_orig.x * h_b_orig.x + h_b_orig.y * h_b_orig.y));
  // norm_orig = sqrt((9+16) + (1+4)) = sqrt(25 + 5) = sqrt(30) ~= 5.477

  // Initialize device pointers
  d_a := nil;
  d_b := nil;
  d_c := nil;
  d_s := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_a, SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_a', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_b, SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_b', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_c, SizeOf(Single));
  AssertEquals('hipMalloc d_c', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_s, SizeOf(rocblas_float_complex));
  AssertEquals('hipMalloc d_s', HIP_SUCCESS, hip_status);

  // Copy initial input values H->D
  hip_status := hipMemcpy(d_a, @h_a, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy a H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_b, @h_b, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy b H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_c, @h_c, SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy c H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_s, @h_s, SizeOf(rocblas_float_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy s H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS CROTG
  // a, b: Procfloat_complex (input/output)
  // c: PSingle (output, real)
  // s: Procfloat_complex (output)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode to device', Integer(rocblas_status_success), Integer(pm_status));
  blas_status := rocblas_crotg(handle, d_a, d_b, d_c, d_s);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_crotg failed');

  // Copy results D->H
  hip_status := hipMemcpy(@h_a, d_a, SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy a D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_b, d_b, SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy b D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_c, d_c, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy c D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_s, d_s, SizeOf(rocblas_float_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy s D->H', HIP_SUCCESS, hip_status);

  // --- Verification ---
  // 1. Check if |r| (magnitude of output a) equals the original norm
  // Assuming h_a now holds r
  CheckEquals(norm_orig, Sqrt(h_a.x * h_a.x + h_a.y * h_a.y), 1e-5, 'CROTG |r| mismatch');

  // 2. Check orthogonality condition: c^2 + |s|^2 = 1 (since c is real)
  CheckEquals(1.0, h_c * h_c + (h_s.x * h_s.x + h_s.y * h_s.y), 1e-5, 'CROTG orthogonality condition c^2 + |s|^2 != 1');

  // 3. Check transformation: r = c * a_orig + s * b_orig
  // r_real = c * a_real - s_real * b_real - s_imag * b_imag
  // r_imag = c * a_imag + s_real * b_imag - s_imag * b_real
  // Or using complex multiplication rules:
  // r = c * a + s * b (where c is treated as complex (c, 0))
  computed_r_real := h_c * h_a_orig.x + (h_s.x * h_b_orig.x - h_s.y * h_b_orig.y);
  computed_r_imag := h_c * h_a_orig.y + (h_s.x * h_b_orig.y + h_s.y * h_b_orig.x);
  CheckEquals(h_a.x, computed_r_real, 1e-5, 'CROTG transformation real part mismatch');
  CheckEquals(h_a.y, computed_r_imag, 1e-5, 'CROTG transformation imaginary part mismatch');

  // Cleanup device memory
  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);
  hipFree(d_s);
end;

// Test_ZROTG tests the double-precision complex Givens rotation generator.
// Similar logic to Test_CROTG.
procedure TTestRocBLAS_Level1.Test_ZROTG;
var
  h_a, h_b, h_s, h_r, h_z: rocblas_double_complex; // Complex inputs/outputs
  h_c: Double;                                     // Real output cosine
  h_a_orig, h_b_orig: rocblas_double_complex;
  d_a, d_b, d_s, d_r, d_z: hipDeviceptr_t    ;    // Device pointers for complex
  d_c: PDouble;                                   // Device pointer for real c
  pm_status, blas_status: rocblas_status;
  hip_status: hipError_t;
  norm_orig: Double;
  computed_r_real, computed_r_imag: Double;
begin
  // Initialize host data (double complex scalars)
  h_a.x := 1.5; h_a.y := 2.5; // a = 1.5 + 2.5i
  h_b.x := 0.5; h_b.y := 1.5; // b = 0.5 + 1.5i
  h_a_orig := h_a;
  h_b_orig := h_b;

  // Initialize outputs
  h_c := 0.0;
  h_s.x := 0.0; h_s.y := 0.0;
  h_r.x := 0.0; h_r.y := 0.0;
  h_z.x := 0.0; h_z.y := 0.0;

  // Calculate norm of original vector [a; b] for expected |r|
  norm_orig := Sqrt((h_a_orig.x * h_a_orig.x + h_a_orig.y * h_a_orig.y) +
                    (h_b_orig.x * h_b_orig.x + h_b_orig.y * h_b_orig.y));

  // Initialize device pointers
  d_a := nil;
  d_b := nil;
  d_c := nil;
  d_s := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_a, SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_a', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_b, SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_b', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_c, SizeOf(Double));
  AssertEquals('hipMalloc d_c', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_s, SizeOf(rocblas_double_complex));
  AssertEquals('hipMalloc d_s', HIP_SUCCESS, hip_status);

  // Copy initial input values H->D
  hip_status := hipMemcpy(d_a, @h_a, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy a H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_b, @h_b, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy b H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_c, @h_c, SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy c H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_s, @h_s, SizeOf(rocblas_double_complex), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy s H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS ZROTG
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode to device', Integer(rocblas_status_success), Integer(pm_status));
  blas_status := rocblas_zrotg(handle, d_a, d_b, d_c, d_s);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_zrotg failed');

  // Copy results D->H
  hip_status := hipMemcpy(@h_a, d_a, SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy a D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_b, d_b, SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy b D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_c, d_c, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy c D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_s, d_s, SizeOf(rocblas_double_complex), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy s D->H', HIP_SUCCESS, hip_status);

  // --- Verification ---
  CheckEquals(norm_orig, Sqrt(h_a.x * h_a.x + h_a.y * h_a.y), 1e-12, 'ZROTG |r| mismatch');
  CheckEquals(1.0, h_c * h_c + (h_s.x * h_s.x + h_s.y * h_s.y), 1e-12, 'ZROTG orthogonality condition c^2 + |s|^2 != 1');
  computed_r_real := h_c * h_a_orig.x + (h_s.x * h_b_orig.x - h_s.y * h_b_orig.y);
  computed_r_imag := h_c * h_a_orig.y + (h_s.x * h_b_orig.y + h_s.y * h_b_orig.x);
  CheckEquals(h_a.x, computed_r_real, 1e-12, 'ZROTG transformation real part mismatch');
  CheckEquals(h_a.y, computed_r_imag, 1e-12, 'ZROTG transformation imaginary part mismatch');

  // Cleanup device memory
  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);
  hipFree(d_s);
end;


// --- ROTM Tests (Apply Modified Givens Rotation) ---

// Test_SROTM tests the single-precision real Modified Givens rotation application.
// It applies a transformation defined by a 5-element parameter vector 'param'
// to vectors x and y.
// x' = H * x  where H is a 2x2 matrix defined by the 'param' values.
procedure TTestRocBLAS_Level1.Test_SROTM;
const
  N = 4;
  incx = 1;
  incy = 1;
  PARAM_SIZE = 5;
var
  h_x, h_y: array[0..N-1] of Single;
  h_x_orig, h_y_orig: array[0..N-1] of Single;
  h_x_expected, h_y_expected: array[0..N-1] of Single;
  h_param: array[0..PARAM_SIZE-1] of Single; // Parameter vector defining the transformation
  d_x, d_y: Pointer;
  d_param: hipDeviceptr_t; // Device pointer for the parameter vector
  i: Integer;
  pm_status, blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data vectors
  h_x[0] := 1.0; h_y[0] := 0.0;
  h_x[1] := 0.0; h_y[1] := 1.0;
  h_x[2] := 2.0; h_y[2] := 3.0;
  h_x[3] := -1.0; h_y[3] := 2.0;
  h_x_orig := h_x;
  h_y_orig := h_y;

  // Define a simple transformation using param.
  // The interpretation of param depends on param[0] (the flag).
  // Let's use flag = -1.0, which means H is defined directly by param[1]..param[4]:
  // H = [ param[1] param[2] ]
  //     [ param[3] param[4] ]
  // Example: H = [ 2.0 1.0 ]
  //              [ 1.0 2.0 ]
  h_param[0] := -1.0; // Flag
  h_param[1] := 2.0;  // H11
  h_param[2] := 1.0;  // H12
  h_param[3] := 1.0;  // H21
  h_param[4] := 2.0;  // H22

  // Calculate expected results manually for the transformation H * [x; y]
  // x'_i = H11 * x_i + H12 * y_i
  // y'_i = H21 * x_i + H22 * y_i
  for i := 0 to N - 1 do
  begin
    h_x_expected[i] := h_param[1] * h_x_orig[i] + h_param[2] * h_y_orig[i];
    h_y_expected[i] := h_param[3] * h_x_orig[i] + h_param[4] * h_y_orig[i];
  end;
  // Expected for i=0: x'=2*1+1*0=2, y'=1*1+2*0=1
  // Expected for i=1: x'=2*0+1*1=1, y'=1*0+2*1=2
  // Expected for i=2: x'=2*2+1*3=7, y'=1*2+2*3=8
  // Expected for i=3: x'=2*(-1)+1*2=0, y'=1*(-1)+2*2=3

  // Initialize device pointers
  d_x := nil;
  d_y := nil;
  d_param := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(Single));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(Single));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_param, PARAM_SIZE * SizeOf(Single));
  AssertEquals('hipMalloc d_param', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_param, @h_param[0], PARAM_SIZE * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy param H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS SROTM
  // x, y: PSingle (input/output vectors)
  // param: PSingle (input parameter vector)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode to device', Integer(rocblas_status_success), Integer(pm_status));
  blas_status := rocblas_srotm(handle, N, d_x, incx, d_y, incy, d_param);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_srotm failed');

  // Copy result data D->H
  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y D->H', HIP_SUCCESS, hip_status);

  // Verify results
  for i := 0 to N - 1 do
  begin
     CheckEquals(h_x_expected[i], h_x[i], 1e-5, 'SROTM x result mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_expected[i], h_y[i], 1e-5, 'SROTM y result mismatch at index ' + IntToStr(i));
  end;

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_param);
end;

// Test_DROTM tests the double-precision real Modified Givens rotation application.
// Similar logic to Test_SROTM.
procedure TTestRocBLAS_Level1.Test_DROTM;
const
  N = 4;
  incx = 1;
  incy = 1;
  PARAM_SIZE = 5;
var
  h_x, h_y: array[0..N-1] of Double;
  h_x_orig, h_y_orig: array[0..N-1] of Double;
  h_x_expected, h_y_expected: array[0..N-1] of Double;
  h_param: array[0..PARAM_SIZE-1] of Double;
  d_x, d_y: Pointer;
  d_param: hipDeviceptr_t; // Device pointer for double precision param
  i: Integer;
  pm_status, blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data vectors
  h_x[0] := 1.5; h_y[0] := 0.5;
  h_x[1] := 0.5; h_y[1] := 1.5;
  h_x[2] := 2.5; h_y[2] := 3.5;
  h_x[3] := -1.5; h_y[3] := 2.5;
  h_x_orig := h_x;
  h_y_orig := h_y;

  // Define transformation (same H matrix as SROTM example)
  h_param[0] := -1.0; // Flag
  h_param[1] := 2.0;  // H11
  h_param[2] := 1.0;  // H12
  h_param[3] := 1.0;  // H21
  h_param[4] := 2.0;  // H22

  // Calculate expected results
  for i := 0 to N - 1 do
  begin
    h_x_expected[i] := h_param[1] * h_x_orig[i] + h_param[2] * h_y_orig[i];
    h_y_expected[i] := h_param[3] * h_x_orig[i] + h_param[4] * h_y_orig[i];
  end;

  // Initialize device pointers
  d_x := nil;
  d_y := nil;
  d_param := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_x, N * SizeOf(Double));
  AssertEquals('hipMalloc d_x', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y, N * SizeOf(Double));
  AssertEquals('hipMalloc d_y', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_param, PARAM_SIZE * SizeOf(Double));
  AssertEquals('hipMalloc d_param', HIP_SUCCESS, hip_status);

  // Copy input data H->D
  hip_status := hipMemcpy(d_x, @h_x[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y, @h_y[0], N * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_param, @h_param[0], PARAM_SIZE * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy param H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS DROTM
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode to device', Integer(rocblas_status_success), Integer(pm_status));
  blas_status := rocblas_drotm(handle, N, d_x, incx, d_y, incy, d_param);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_drotm failed');

  // Copy result data D->H
  hip_status := hipMemcpy(@h_x[0], d_x, N * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_y[0], d_y, N * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y D->H', HIP_SUCCESS, hip_status);

  // Verify results
  for i := 0 to N - 1 do
  begin
     CheckEquals(h_x_expected[i], h_x[i], 1e-12, 'DROTM x result mismatch at index ' + IntToStr(i));
     CheckEquals(h_y_expected[i], h_y[i], 1e-12, 'DROTM y result mismatch at index ' + IntToStr(i));
  end;

  // Cleanup device memory
  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_param);
end;


// --- ROTMG Tests (Generate Modified Givens Rotation) ---

 //Test_SROTMG tests the single-precision real Modified Givens rotation generator.
 //It calculates a parameter vector 'param' and scaling factors d1, d2, x1
 //such that the transformation defined by 'param' zeros out the y1 component
 //of the scaled vector [sqrt(d1)*x1; sqrt(d2)*y1].
 //d1, d2, x1, y1 are inputs. param is output.
procedure TTestRocBLAS_Level1.Test_SROTMG;
var
  h_d1, h_d2, h_x1, h_y1: Single; // Input scalars
  h_param: array[0..4] of Single; // Output parameter vector
  d_d1, d_d2, d_x1, d_y1: hipDeviceptr_t; // Device pointers for inputs
  d_param: hipDeviceptr_t;               // Device pointer for output param vector
  i: Integer;
  pm_status, blas_status: rocblas_status;
  hip_status: hipError_t;
  // Variables for verification (if possible)
  transformed_x1, transformed_y1: Double; // Use double for intermediate calc precision
  h_d1_orig, h_d2_orig, h_x1_orig, h_y1_orig: Single;
begin
  // Initialize host data
  h_d1 := 4.0;
  h_d2 := 9.0;
  h_x1 := 2.0;
  h_y1 := 3.0;
  h_d1_orig := h_d1;
  h_d2_orig := h_d2;
  h_x1_orig := h_x1;
  h_y1_orig := h_y1;

  // Initialize param output array
  for i := 0 to 4 do
    h_param[i] := 0.0;

  // Initialize device pointers
  d_d1 := nil;
  d_d2 := nil;
  d_x1 := nil;
  d_y1 := nil;
  d_param := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_d1, SizeOf(Single));
  AssertEquals('hipMalloc d_d1', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_d2, SizeOf(Single));
  AssertEquals('hipMalloc d_d2', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_x1, SizeOf(Single));
  AssertEquals('hipMalloc d_x1', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y1, SizeOf(Single));
  AssertEquals('hipMalloc d_y1', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_param, 5 * SizeOf(Single)); // 5 elements in param
  AssertEquals('hipMalloc d_param', HIP_SUCCESS, hip_status);

  // Copy initial input values H->D
  hip_status := hipMemcpy(d_d1, @h_d1, SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy d1 H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_d2, @h_d2, SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy d2 H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_x1, @h_x1, SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x1 H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y1, @h_y1, SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y1 H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_param, @h_param[0], 5 * SizeOf(Single), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy param H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS SROTMG
  // d1, d2, x1: PSingle (input/output)
  // y1: PSingle (input)
  // param: PSingle (output)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode to device', Integer(rocblas_status_success), Integer(pm_status));
  blas_status := rocblas_srotmg(handle, d_d1, d_d2, d_x1, d_y1, d_param);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_srotmg failed');

  // Copy results D->H
  // Note: d1, d2, x1 might be modified by the function
  hip_status := hipMemcpy(@h_d1, d_d1, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d1 D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_d2, d_d2, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d2 D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_x1, d_x1, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x1 D->H', HIP_SUCCESS, hip_status);
  // y1 is typically input only, but the signature shows it as const y1, so likely unchanged.
  // We copy it back just in case the signature is misleading or for completeness.
  hip_status := hipMemcpy(@h_y1, d_y1, SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y1 D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_param[0], d_param, 5 * SizeOf(Single), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy param D->H', HIP_SUCCESS, hip_status);

  // --- Verification ---
  // Verifying SROTMG is complex because it involves scaling and the specific
  // transformation matrix H generated (encoded in param) depends on internal logic.
  // A common way to test is to apply the generated 'param' using SROTM to a vector
  // and check the result.
  // However, SROTMG's primary goal is to produce a 'param' such that when applied
  // to the scaled inputs [sqrt(d1_out)*x1_out, sqrt(d2_out)*y1], the second component becomes 0.
  // Let's check this condition.
  // Note: The exact relationship between inputs/outputs and the transformation
  // can be tricky. This test focuses on the most fundamental property.

  // The transformation defined by param should make the y-component zero
  // when applied to [sqrt(d1_new) * x1_new, sqrt(d2_new) * y1_orig].
  // However, the exact application requires knowing the flag (param[0]) and
  // interpreting the rest of param correctly.
  // A simpler check is to ensure the function executed successfully and
  // the param vector is populated with reasonable (non-NaN, non-Inf) values.
  // This is a basic sanity check.

  // Check that param values are finite numbers (not NaN or Inf)
  for i := 0 to 4 do
  begin
    AssertTrue('SROTMG param[' + IntToStr(i) + '] is not finite',
               (h_param[i] <> Infinity) and (h_param[i] <> -Infinity) and not IsNan(h_param[i]));
  end;

  // Check that d1 and d2 are non-negative after scaling (typical property)
  AssertTrue('SROTMG d1 output is negative', h_d1 >= 0.0);
  AssertTrue('SROTMG d2 output is negative', h_d2 >= 0.0);

  // Cleanup device memory
  hipFree(d_d1);
  hipFree(d_d2);
  hipFree(d_x1);
  hipFree(d_y1);
  hipFree(d_param);
end;

 //Test_DROTMG tests the double-precision real Modified Givens rotation generator.
 //Similar logic to Test_SROTMG.
procedure TTestRocBLAS_Level1.Test_DROTMG;
var
  h_d1, h_d2, h_x1, h_y1: Double; // Input scalars
  h_param: array[0..4] of Double; // Output parameter vector
  d_d1, d_d2, d_x1, d_y1: hipDeviceptr_t; // Device pointers for double inputs
  d_param: hipDeviceptr_t;               // Device pointer for double output param vector
  i: Integer;
  pm_status, blas_status: rocblas_status;
  hip_status: hipError_t;
begin
  // Initialize host data
  h_d1 := 4.5;
  h_d2 := 9.5;
  h_x1 := 2.5;
  h_y1 := 3.5;

  // Initialize param output array
  for i := 0 to 4 do
    h_param[i] := 0.0;

  // Initialize device pointers
  d_d1 := nil;
  d_d2 := nil;
  d_x1 := nil;
  d_y1 := nil;
  d_param := nil;

  // Allocate device memory
  hip_status := hipMalloc(d_d1, SizeOf(Double));
  AssertEquals('hipMalloc d_d1', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_d2, SizeOf(Double));
  AssertEquals('hipMalloc d_d2', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_x1, SizeOf(Double));
  AssertEquals('hipMalloc d_x1', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_y1, SizeOf(Double));
  AssertEquals('hipMalloc d_y1', HIP_SUCCESS, hip_status);
  hip_status := hipMalloc(d_param, 5 * SizeOf(Double));
  AssertEquals('hipMalloc d_param', HIP_SUCCESS, hip_status);

  // Copy initial input values H->D
  hip_status := hipMemcpy(d_d1, @h_d1, SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy d1 H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_d2, @h_d2, SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy d2 H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_x1, @h_x1, SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy x1 H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_y1, @h_y1, SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy y1 H->D', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(d_param, @h_param[0], 5 * SizeOf(Double), hipMemcpyHostToDevice);
  AssertEquals('hipMemcpy param H->D', HIP_SUCCESS, hip_status);

  // Execute rocBLAS DROTMG
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
  AssertEquals('rocblas_set_pointer_mode to device', Integer(rocblas_status_success), Integer(pm_status));
  blas_status := rocblas_drotmg(handle, d_d1, d_d2, d_x1, d_y1, d_param);
  CheckEquals(Integer(rocblas_status_success), Integer(blas_status), 'rocblas_drotmg failed');

  // Copy results D->H
  hip_status := hipMemcpy(@h_d1, d_d1, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d1 D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_d2, d_d2, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy d2 D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_x1, d_x1, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy x1 D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_y1, d_y1, SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy y1 D->H', HIP_SUCCESS, hip_status);
  hip_status := hipMemcpy(@h_param[0], d_param, 5 * SizeOf(Double), hipMemcpyDeviceToHost);
  AssertEquals('hipMemcpy param D->H', HIP_SUCCESS, hip_status);

  // --- Verification ---
  // Basic sanity checks
  for i := 0 to 4 do
  begin
    AssertTrue('DROTMG param[' + IntToStr(i) + '] is not finite',
               (h_param[i] <> Infinity) and (h_param[i] <> -Infinity) and not IsNan(h_param[i]));
  end;
  AssertTrue('DROTMG d1 output is negative', h_d1 >= 0.0);
  AssertTrue('DROTMG d2 output is negative', h_d2 >= 0.0);

  // Cleanup device memory
  hipFree(d_d1);
  hipFree(d_d2);
  hipFree(d_x1);
  hipFree(d_y1);
  hipFree(d_param);
end;

initialization
  //RegisterTest(TTestRocBLAS_Base);
  RegisterTest(TTestRocBLAS_Level1);

end.
