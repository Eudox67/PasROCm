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

unit Test_rocblas_lvl3_batched;

{$mode objfpc}{$H+}
{$linklib rocblas}

interface

uses
  SysUtils, Math, fpcunit, testregistry, rocblas, hip_error, hip_types, hip_runtime, Test_rocblas_base, ROCm_test_utils;

type

  TTestRocBLAS_Level3_Batched = class(TTestRocBLAS_Base)
  published
    procedure Test_SGEMM_Batched;
    procedure Test_DGEMM_Batched;
    procedure Test_CGEMM_Batched;
    procedure Test_ZGEMM_Batched;
    procedure Test_SSYMM_Batched;
    procedure Test_DSYMM_Batched;
    procedure Test_CSYMM_Batched;
    procedure Test_ZSYMM_Batched;
    procedure Test_STRMM_Batched;
    procedure Test_DTRMM_Batched;
    procedure Test_CTRMM_Batched;
    procedure Test_ZTRMM_Batched;
    procedure Test_STRSM_Batched;
    procedure Test_DTRSM_Batched;
    procedure Test_CTRSM_Batched;
    procedure Test_ZTRSM_Batched;
    procedure Test_SSYRK_Batched;
    procedure Test_DSYRK_Batched;
    procedure Test_CSYRK_Batched;
    procedure Test_ZSYRK_Batched;
    procedure Test_CHERK_Batched;
    procedure Test_ZHERK_Batched;
    procedure Test_SSyr2k_Batched;
    //procedure Test_DSyr2k_Batched;
    procedure Test_CSyr2k_Batched;
    procedure Test_ZSyr2k_Batched;
    procedure Test_CHER2K_Batched;
    procedure Test_ZHER2K_Batched;
    // Add 64-bit and strided-batched tests if needed, following the same pattern
  end;


implementation

// NOTE: device pointers (devA_array, devB_array, devC_array) must reside in device memory, not host memory
procedure TTestRocBLAS_Level3_Batched.Test_SGEMM_Batched;
const
  M = 4; N = 3; K = 5; batch_count = 1;
  alpha: Single = 2.0; beta: Single = 1.5;
var
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC, expectedC: array of Single;
  devA_array, devB_array, devC_array: array of PSingle;
  devA_array_device, devB_array_device, devC_array_device: PPointer;  // Device-allocated arrays
  i, j, l, b: Integer;
begin
  SetLength(hostA, M*K*batch_count);
  SetLength(hostB, K*N*batch_count);
  SetLength(hostC, M*N*batch_count);
  SetLength(expectedC, M*N*batch_count);
  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M*K-1 do hostA[b*M*K + i] := i + 1 + b;
    for i := 0 to K*N-1 do hostB[b*K*N + i] := i + 1 + b;
    for i := 0 to M*N-1 do hostC[b*M*N + i] := i + 1 + b;

    for i := 0 to M - 1 do for j := 0 to N - 1 do
    begin
      expectedC[b*M*N + j*M+i] := beta * hostC[b*M*N + j*M+i];
      for l := 0 to K - 1 do
        expectedC[b*M*N + j*M+i] := expectedC[b*M*N + j*M+i] + alpha * hostA[b*M*K + l*M+i] * hostB[b*K*N + j*K+l];
    end;
  end;

  // Allocate device memory for matrices
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, M*K*SizeOf(Single)*batch_count));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, K*N*SizeOf(Single)*batch_count));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, M*N*SizeOf(Single)*batch_count));
  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], M*K*SizeOf(Single)*batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], K*N*SizeOf(Single)*batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC', HIP_SUCCESS, hipMemcpy(devC, @hostC[0], M*N*SizeOf(Single)*batch_count, hipMemcpyHostToDevice));

  // Initialize device pointer arrays
  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PSingle(PtrUInt(devA) + b*M*K*SizeOf(Single));
    devB_array[b] := PSingle(PtrUInt(devB) + b*K*N*SizeOf(Single));
    devC_array[b] := PSingle(PtrUInt(devC) + b*M*N*SizeOf(Single));
  end;

  // Allocate and copy device pointer arrays
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMemcpy devA_array_device', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set pointer mode (alpha, beta are host pointers)
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host)));

  // Call rocBLAS
  AssertEquals('rocblas_sgemm_batched should return success', Integer(rocblas_status_success), Integer(rocblas_sgemm_batched(
    handle, rocblas_operation_none, rocblas_operation_none, M, N, K, @alpha, PPSingle(devA_array_device), M,
    PPSingle(devB_array_device), K, @beta, PPSingle(devC_array_device), M, batch_count
  )));

  // Synchronize to catch potential async errors
  AssertEquals('hipDeviceSynchronize post-rocblas_sgemm_batched', HIP_SUCCESS, hipDeviceSynchronize());

  // Copy result back to host and validate
  AssertEquals('hipMemcpy devC result', HIP_SUCCESS, hipMemcpy(@hostC[0], devC, M*N*SizeOf(Single)*batch_count, hipMemcpyDeviceToHost));
  for i := 0 to M*N*batch_count-1 do
    // Use CheckEqualsRelative for better floating-point comparison if available in base class/utils
    AssertEquals('SGEMM_Batched C[' + IntToStr(i) + ']', expectedC[i], hostC[i], 1e-4);

  // Cleanup
  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
end;


// NOTE: device pointers (devA_array, devB_array, devC_array) must reside in device memory, not host memory
procedure TTestRocBLAS_Level3_Batched.Test_DGEMM_Batched;
const
  M = 4; N = 3; K = 5; batch_count = 1;
  alpha: Double = 2.0; beta: Double = 1.5;
var
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC, expectedC: array of Double;
  devA_array, devB_array, devC_array: array of PDouble;
  devA_array_device, devB_array_device, devC_array_device: PPointer;  // Device-allocated arrays
  i, j, l, b: Integer;
begin
  // Initialize host arrays
  SetLength(hostA, M*K*batch_count);
  SetLength(hostB, K*N*batch_count);
  SetLength(hostC, M*N*batch_count);
  SetLength(expectedC, M*N*batch_count);
  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);

  // Populate host data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M*K-1 do hostA[b*M*K + i] := i + 1 + b;
    for i := 0 to K*N-1 do hostB[b*K*N + i] := i + 1 + b;
    for i := 0 to M*N-1 do hostC[b*M*N + i] := i + 1 + b;

    for i := 0 to M - 1 do for j := 0 to N - 1 do
    begin
      expectedC[b*M*N + j*M+i] := beta * hostC[b*M*N + j*M+i];
      for l := 0 to K - 1 do
        expectedC[b*M*N + j*M+i] := expectedC[b*M*N + j*M+i] + alpha * hostA[b*M*K + l*M+i] * hostB[b*K*N + j*K+l];
    end;
  end;

  // Allocate device memory for matrices
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, M*K*SizeOf(Double)*batch_count));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, K*N*SizeOf(Double)*batch_count));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, M*N*SizeOf(Double)*batch_count));
  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], M*K*SizeOf(Double)*batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], K*N*SizeOf(Double)*batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC', HIP_SUCCESS, hipMemcpy(devC, @hostC[0], M*N*SizeOf(Double)*batch_count, hipMemcpyHostToDevice));

  // Initialize device pointer arrays
  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PDouble(PtrUInt(devA) + b*M*K*SizeOf(Double));
    devB_array[b] := PDouble(PtrUInt(devB) + b*K*N*SizeOf(Double));
    devC_array[b] := PDouble(PtrUInt(devC) + b*M*N*SizeOf(Double));
  end;

  // Allocate and copy device pointer arrays
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMemcpy devA_array_device', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set pointer mode (alpha, beta are host pointers)
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host)));

  // Call rocBLAS
  AssertEquals('rocblas_dgemm_batched should return success', Integer(rocblas_status_success), Integer(rocblas_dgemm_batched(
    handle, rocblas_operation_none, rocblas_operation_none, M, N, K, @alpha, PPDouble(devA_array_device), M,
    PPDouble(devB_array_device), K, @beta, PPDouble(devC_array_device), M, batch_count
  )));

  // Synchronize to catch potential async errors
  AssertEquals('hipDeviceSynchronize post-rocblas_dgemm_batched', HIP_SUCCESS, hipDeviceSynchronize());

  // Copy result back to host and validate
  AssertEquals('hipMemcpy devC result', HIP_SUCCESS, hipMemcpy(@hostC[0], devC, M*N*SizeOf(Double)*batch_count, hipMemcpyDeviceToHost));
  for i := 0 to M*N*batch_count-1 do
    // Use CheckEqualsRelative for better floating-point comparison if available in base class/utils
    AssertEquals('DGEMM_Batched C[' + IntToStr(i) + ']', expectedC[i], hostC[i], 1e-12);

  // Cleanup
  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
end;

// --- CGEMM_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_CGEMM_Batched;
const
  M = 4; N = 3; K = 5; batch_count = 2; // Use batch_count > 1 for better testing
var
  alpha, beta: rocblas_float_complex;
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC, expectedC: array of rocblas_float_complex;
  devA_array, devB_array, devC_array: array of Procfloat_complex; // Array of device pointers
  devA_array_device, devB_array_device, devC_array_device: PPointer;  // Device-allocated array of pointers
  i, j, l, b: Integer;
  t1_r, t1_i, res_r, res_i: Double;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  alpha.x := 2.0; alpha.y := -1.0;
  beta.x := 1.5; beta.y := 0.5;

  SetLength(hostA, M * K * batch_count);
  SetLength(hostB, K * N * batch_count);
  SetLength(hostC, M * N * batch_count);
  SetLength(expectedC, M * N * batch_count);
  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M * K - 1 do
    begin
      hostA[b * M * K + i].x := (i + 1 + b); // Real part
      hostA[b * M * K + i].y := (i + 1 + b) * 0.1; // Imaginary part
    end;
    for i := 0 to K * N - 1 do
    begin
      hostB[b * K * N + i].x := (i + 1 + b) * 0.5;
      hostB[b * K * N + i].y := -(i + 1 + b) * 0.2;
    end;
    for i := 0 to M * N - 1 do
    begin
      hostC[b * M * N + i].x := (i + 1 + b) * 0.2;
      hostC[b * M * N + i].y := (i + 1 + b) * 0.3;
    end;

    // Calculate expected result for this batch item: C = alpha * A * B + beta * C
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
      begin
        // C[i,j] = beta * C[i,j] + sum_{l=0}^{K-1} (alpha * A[i,l] * B[l,j])
        expectedC[b * M * N + j * M + i].x := beta.x * hostC[b * M * N + j * M + i].x - beta.y * hostC[b * M * N + j * M + i].y;
        expectedC[b * M * N + j * M + i].y := beta.x * hostC[b * M * N + j * M + i].y + beta.y * hostC[b * M * N + j * M + i].x;

        for l := 0 to K - 1 do
        begin
          // alpha * A[i,l]
          // (a_r + a_i*i) * (b_r + b_i*i) = (a_r*b_r - a_i*b_i) + (a_r*b_i + a_i*b_r)*i
          // alpha * A[i,l] * B[l,j]
          // Let t1 = alpha * A[i,l]
          // result += t1 * B[l,j]
          // (t1_r + t1_i*i) * (b_r + b_i*i) = (t1_r*b_r - t1_i*b_i) + (t1_r*b_i + t1_i*b_r)*i
          t1_r := alpha.x * hostA[b * M * K + l * M + i].x - alpha.y * hostA[b * M * K + l * M + i].y;
          t1_i := alpha.x * hostA[b * M * K + l * M + i].y + alpha.y * hostA[b * M * K + l * M + i].x;

          res_r := t1_r * hostB[b * K * N + j * K + l].x - t1_i * hostB[b * K * N + j * K + l].y;
          res_i := t1_r * hostB[b * K * N + j * K + l].y + t1_i * hostB[b * K * N + j * K + l].x;

          expectedC[b * M * N + j * M + i].x := expectedC[b * M * N + j * M + i].x + res_r;
          expectedC[b * M * N + j * M + i].y := expectedC[b * M * N + j * M + i].y + res_i;
        end;
      end;
  end;

  // Allocate device memory for matrices
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, M * K * SizeOf(rocblas_float_complex) * batch_count));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, K * N * SizeOf(rocblas_float_complex) * batch_count));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, M * N * SizeOf(rocblas_float_complex) * batch_count));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], M * K * SizeOf(rocblas_float_complex) * batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], K * N * SizeOf(rocblas_float_complex) * batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC', HIP_SUCCESS, hipMemcpy(devC, @hostC[0], M * N * SizeOf(rocblas_float_complex) * batch_count, hipMemcpyHostToDevice));

  // Initialize device pointer arrays (pointers to offsets within the large allocated buffers)
  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procfloat_complex(PtrUInt(devA) + b * M * K * SizeOf(rocblas_float_complex));
    devB_array[b] := Procfloat_complex(PtrUInt(devB) + b * K * N * SizeOf(rocblas_float_complex));
    devC_array[b] := Procfloat_complex(PtrUInt(devC) + b * M * N * SizeOf(rocblas_float_complex));
  end;

  // Allocate and copy device pointer arrays (arrays of pointers residing on the device)
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count * SizeOf(Pointer)));

  AssertEquals('hipMemcpy devA_array_device', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set pointer mode (alpha, beta are host pointers)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // Call rocBLAS CGEMM_Batched
  blas_status := rocblas_cgemm_batched(
    handle,
    rocblas_operation_none, // transA
    rocblas_operation_none, // transB
    M, N, K,
    @alpha, // alpha (host pointer)
    PPProcfloat_complex(devA_array_device), // A array (device pointer to array of device pointers)
    M, // lda
    PPProcfloat_complex(devB_array_device), // B array (device pointer to array of device pointers)
    K, // ldb
    @beta, // beta (host pointer)
    PPProcfloat_complex(devC_array_device), // C array (device pointer to array of device pointers)
    M, // ldc
    batch_count
  );
  AssertEquals('rocblas_cgemm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // Synchronize to catch potential async errors
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_cgemm_batched', HIP_SUCCESS, sync_status);

  // Copy result back to host
  AssertEquals('hipMemcpy devC result', HIP_SUCCESS, hipMemcpy(@hostC[0], devC, M * N * SizeOf(rocblas_float_complex) * batch_count, hipMemcpyDeviceToHost));

  // Validate results
  for i := 0 to M * N * batch_count - 1 do
  begin
    // Use a relative or absolute tolerance check. Assuming CheckEqualsRelative exists or similar logic is in base/utils.
    // For simplicity here, using AssertEquals with a small absolute tolerance.
    // A more robust check would be in CheckEqualsRelative if available.
    AssertEquals('CGEMM_Batched C[' + IntToStr(i) + '].x', expectedC[i].x, hostC[i].x, 1e-3);
    AssertEquals('CGEMM_Batched C[' + IntToStr(i) + '].y', expectedC[i].y, hostC[i].y, 1e-3);
  end;


  // Cleanup
  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
end;


// --- ZGEMM_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_ZGEMM_Batched;
const
  M = 3; N = 4; K = 2; batch_count = 3; // Different sizes for variety
var
  alpha, beta: rocblas_double_complex;
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC, expectedC: array of rocblas_double_complex;
  devA_array, devB_array, devC_array: array of Procdouble_complex;
  devA_array_device, devB_array_device, devC_array_device: PPointer;
  i, j, l, b: Integer;
  t1_r, t1_i, res_r, res_i: Extended;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  alpha.x := 1.5; alpha.y := 0.8;
  beta.x := -0.5; beta.y := 1.2;

  SetLength(hostA, M * K * batch_count);
  SetLength(hostB, K * N * batch_count);
  SetLength(hostC, M * N * batch_count);
  SetLength(expectedC, M * N * batch_count);
  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    for i := 0 to M * K - 1 do
    begin
      hostA[b * M * K + i].x := (i + 1 + b) * 1.1;
      hostA[b * M * K + i].y := (i + 1 + b) * -0.2;
    end;
    for i := 0 to K * N - 1 do
    begin
      hostB[b * K * N + i].x := (i + 1 + b) * 0.8;
      hostB[b * K * N + i].y := (i + 1 + b) * 0.3;
    end;
    for i := 0 to M * N - 1 do
    begin
      hostC[b * M * N + i].x := (i + 1 + b) * -0.1;
      hostC[b * M * N + i].y := (i + 1 + b) * 0.4;
    end;

    // Calculate expected result for this batch item
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
      begin
        expectedC[b * M * N + j * M + i].x := beta.x * hostC[b * M * N + j * M + i].x - beta.y * hostC[b * M * N + j * M + i].y;
        expectedC[b * M * N + j * M + i].y := beta.x * hostC[b * M * N + j * M + i].y + beta.y * hostC[b * M * N + j * M + i].x;

        for l := 0 to K - 1 do
        begin
          t1_r := alpha.x * hostA[b * M * K + l * M + i].x - alpha.y * hostA[b * M * K + l * M + i].y;
          t1_i := alpha.x * hostA[b * M * K + l * M + i].y + alpha.y * hostA[b * M * K + l * M + i].x;

          res_r := t1_r * hostB[b * K * N + j * K + l].x - t1_i * hostB[b * K * N + j * K + l].y;
          res_i := t1_r * hostB[b * K * N + j * K + l].y + t1_i * hostB[b * K * N + j * K + l].x;

          expectedC[b * M * N + j * M + i].x := expectedC[b * M * N + j * M + i].x + res_r;
          expectedC[b * M * N + j * M + i].y := expectedC[b * M * N + j * M + i].y + res_i;
        end;
      end;
  end;

  // Allocate device memory for matrices
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, M * K * SizeOf(rocblas_double_complex) * batch_count));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, K * N * SizeOf(rocblas_double_complex) * batch_count));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, M * N * SizeOf(rocblas_double_complex) * batch_count));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], M * K * SizeOf(rocblas_double_complex) * batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], K * N * SizeOf(rocblas_double_complex) * batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC', HIP_SUCCESS, hipMemcpy(devC, @hostC[0], M * N * SizeOf(rocblas_double_complex) * batch_count, hipMemcpyHostToDevice));

  // Initialize device pointer arrays
  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + b * M * K * SizeOf(rocblas_double_complex));
    devB_array[b] := Procdouble_complex(PtrUInt(devB) + b * K * N * SizeOf(rocblas_double_complex));
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + b * M * N * SizeOf(rocblas_double_complex));
  end;

  // Allocate and copy device pointer arrays
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count * SizeOf(Pointer)));

  AssertEquals('hipMemcpy devA_array_device', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set pointer mode (alpha, beta are host pointers)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // Call rocBLAS ZGEMM_Batched
  blas_status := rocblas_zgemm_batched(
    handle,
    rocblas_operation_none, // transA
    rocblas_operation_none, // transB
    M, N, K,
    @alpha, // alpha (host pointer)
    PPProcdouble_complex(devA_array_device), // A array
    M, // lda
    PPProcdouble_complex(devB_array_device), // B array
    K, // ldb
    @beta, // beta (host pointer)
    PPProcdouble_complex(devC_array_device), // C array
    M, // ldc
    batch_count
  );
  AssertEquals('rocblas_zgemm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // Synchronize to catch potential async errors
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_zgemm_batched', HIP_SUCCESS, sync_status);

  // Copy result back to host
  AssertEquals('hipMemcpy devC result', HIP_SUCCESS, hipMemcpy(@hostC[0], devC, M * N * SizeOf(rocblas_double_complex) * batch_count, hipMemcpyDeviceToHost));

  // Validate results
  for i := 0 to M * N * batch_count - 1 do
  begin
    AssertEquals('ZGEMM_Batched C[' + IntToStr(i) + '].x', expectedC[i].x, hostC[i].x, 1e-12);
    AssertEquals('ZGEMM_Batched C[' + IntToStr(i) + '].y', expectedC[i].y, hostC[i].y, 1e-12);
  end;


  // Cleanup
  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
end;


// --- SSYMM_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_SSYMM_Batched;
const
  M = 3; N = 3; batch_count = 2;
var
  side: rocblas_side;
  uplo: rocblas_fill;
  alpha, beta: Single;
  dim_A: Integer;
  lda, ldb, ldc: Integer;
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC, expectedC: array of Single;
  devA_array, devB_array, devC_array: array of PSingle;
  devA_array_device, devB_array_device, devC_array_device: PPointer;
  i, j, k, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  side := rocblas_side_left;
  uplo := rocblas_fill_upper;
  alpha := 1.5;
  beta := 2.0;

  if side = rocblas_side_left then
    dim_A := M
  else
    dim_A := N;

  lda := dim_A;
  ldb := M;
  ldc := M;

  SetLength(hostA, lda * dim_A * batch_count);
  SetLength(hostB, ldb * N * batch_count);
  SetLength(hostC, ldc * N * batch_count);
  SetLength(expectedC, ldc * N * batch_count);
  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Zero out matrices first
    for i := 0 to lda * dim_A - 1 do
      hostA[b * lda * dim_A + i] := 0.0;
    for i := 0 to ldb * N - 1 do
      hostB[b * ldb * N + i] := 0.0;
    for i := 0 to ldc * N - 1 do
      hostC[b * ldc * N + i] := 0.0;

    // Fill symmetric matrix A (upper triangle)
    if uplo = rocblas_fill_upper then
    begin
      for i := 0 to dim_A - 1 do
        for j := i to dim_A - 1 do
          hostA[b * lda * dim_A + j * lda + i] := (i + j + 1 + b) * 0.5;
      // Copy to lower triangle
      for i := 0 to dim_A - 1 do
        for j := 0 to i - 1 do
          hostA[b * lda * dim_A + j * lda + i] := hostA[b * lda * dim_A + i * lda + j];
    end;

    // Fill matrix B
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
        hostB[b * ldb * N + j * ldb + i] := (i + j + 1 + b) * 0.2;

    // Fill matrix C
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
        hostC[b * ldc * N + j * ldc + i] := (i + j + 1 + b) * 0.1;

    // Calculate expected result for this batch item: C = alpha * A * B + beta * C (side=left)
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
      begin
        expectedC[b * ldc * N + j * ldc + i] := beta * hostC[b * ldc * N + j * ldc + i];
        for k := 0 to dim_A - 1 do // dim_A = M for side_left
          expectedC[b * ldc * N + j * ldc + i] := expectedC[b * ldc * N + j * ldc + i] + alpha * hostA[b * lda * dim_A + k * lda + i] * hostB[b * ldb * N + j * ldb + k];
      end;
  end;

  // Allocate device memory for matrices
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, lda * dim_A * SizeOf(Single) * batch_count));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, ldb * N * SizeOf(Single) * batch_count));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, ldc * N * SizeOf(Single) * batch_count));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], lda * dim_A * SizeOf(Single) * batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], ldb * N * SizeOf(Single) * batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC', HIP_SUCCESS, hipMemcpy(devC, @hostC[0], ldc * N * SizeOf(Single) * batch_count, hipMemcpyHostToDevice));

  // Initialize device pointer arrays
  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PSingle(PtrUInt(devA) + b * lda * dim_A * SizeOf(Single));
    devB_array[b] := PSingle(PtrUInt(devB) + b * ldb * N * SizeOf(Single));
    devC_array[b] := PSingle(PtrUInt(devC) + b * ldc * N * SizeOf(Single));
  end;

  // Allocate and copy device pointer arrays
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count * SizeOf(Pointer)));

  AssertEquals('hipMemcpy devA_array_device', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set pointer mode (alpha, beta are host pointers)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // Call rocBLAS SSYMM_Batched
  blas_status := rocblas_ssymm_batched(
    handle,
    side,
    uplo,
    M, N,
    @alpha, // alpha (host pointer)
    PPSingle(devA_array_device), // A array
    lda, // lda
    PPSingle(devB_array_device), // B array
    ldb, // ldb
    @beta, // beta (host pointer)
    PPSingle(devC_array_device), // C array (output)
    ldc, // ldc
    batch_count
  );
  AssertEquals('rocblas_ssymm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // Synchronize to catch potential async errors
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ssymm_batched', HIP_SUCCESS, sync_status);

  // Copy result back to host
  AssertEquals('hipMemcpy devC result', HIP_SUCCESS, hipMemcpy(@hostC[0], devC, ldc * N * SizeOf(Single) * batch_count, hipMemcpyDeviceToHost));

  // Validate results (basic check or using CheckEqualsRelative if available)
  for i := 0 to ldc * N * batch_count - 1 do
  begin
    // Assuming CheckEqualsRelative is available in the base class or utils
    // CheckEqualsRelative('SSYMM_Batched C[' + IntToStr(i) + ']', expectedC[i], hostC[i], 1e-3);
    // For now, using AssertEquals with tolerance
    AssertEquals('SSYMM_Batched C[' + IntToStr(i) + ']', expectedC[i], hostC[i], 1e-3);
  end;

  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
end;


// --- DSYMM_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_DSYMM_Batched;
const
  M = 2; N = 4; batch_count = 3; // Different dimensions
var
  side: rocblas_side;
  uplo: rocblas_fill;
  alpha, beta: Double;
  dim_A: Integer;
  lda, ldb, ldc: Integer;
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC, expectedC: array of Double;
  devA_array, devB_array, devC_array: array of PDouble;
  devA_array_device, devB_array_device, devC_array_device: PPointer;
  i, j, k, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  side := rocblas_side_right; // Different side
  uplo := rocblas_fill_lower; // Different uplo
  alpha := 0.8;
  beta := 1.2;

  if side = rocblas_side_left then
    dim_A := M
  else
    dim_A := N; // For side_right, dim_A = N

  lda := dim_A;
  ldb := M;
  ldc := M;

  SetLength(hostA, lda * dim_A * batch_count);
  SetLength(hostB, ldb * N * batch_count);
  SetLength(hostC, ldc * N * batch_count);
  SetLength(expectedC, ldc * N * batch_count);
  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Zero out matrices first
    for i := 0 to lda * dim_A - 1 do
      hostA[b * lda * dim_A + i] := 0.0;
    for i := 0 to ldb * N - 1 do
      hostB[b * ldb * N + i] := 0.0;
    for i := 0 to ldc * N - 1 do
      hostC[b * ldc * N + i] := 0.0;

    // Fill symmetric matrix A (lower triangle for this test)
    if uplo = rocblas_fill_lower then
    begin
      for i := 0 to dim_A - 1 do
        for j := 0 to i do
          hostA[b * lda * dim_A + j * lda + i] := (i + j + 1 + b) * 0.7;
      // Copy to upper triangle
      for i := 0 to dim_A - 1 do
        for j := i + 1 to dim_A - 1 do
          hostA[b * lda * dim_A + j * lda + i] := hostA[b * lda * dim_A + i * lda + j];
    end;

    // Fill matrix B
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
        hostB[b * ldb * N + j * ldb + i] := (i + j + 1 + b) * 0.3;

    // Fill matrix C
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
        hostC[b * ldc * N + j * ldc + i] := (i + j + 1 + b) * -0.1;

    // Calculate expected result for this batch item: C = alpha * B * A + beta * C (side=right)
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
      begin
        expectedC[b * ldc * N + j * ldc + i] := beta * hostC[b * ldc * N + j * ldc + i];
        for k := 0 to dim_A - 1 do // dim_A = N for side_right
          expectedC[b * ldc * N + j * ldc + i] := expectedC[b * ldc * N + j * ldc + i] + alpha * hostB[b * ldb * N + k * ldb + i] * hostA[b * lda * dim_A + j * lda + k];
      end;
  end;

  // Allocate device memory for matrices
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, lda * dim_A * SizeOf(Double) * batch_count));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, ldb * N * SizeOf(Double) * batch_count));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, ldc * N * SizeOf(Double) * batch_count));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], lda * dim_A * SizeOf(Double) * batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], ldb * N * SizeOf(Double) * batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC', HIP_SUCCESS, hipMemcpy(devC, @hostC[0], ldc * N * SizeOf(Double) * batch_count, hipMemcpyHostToDevice));

  // Initialize device pointer arrays
  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := PDouble(PtrUInt(devA) + b * lda * dim_A * SizeOf(Double));
    devB_array[b] := PDouble(PtrUInt(devB) + b * ldb * N * SizeOf(Double));
    devC_array[b] := PDouble(PtrUInt(devC) + b * ldc * N * SizeOf(Double));
  end;

  // Allocate and copy device pointer arrays
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count * SizeOf(Pointer)));

  AssertEquals('hipMemcpy devA_array_device', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set pointer mode (alpha, beta are host pointers)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // Call rocBLAS DSYMM_Batched
  blas_status := rocblas_dsymm_batched(
    handle,
    side,
    uplo,
    M, N,
    @alpha, // alpha (host pointer)
    PPDouble(devA_array_device), // A array
    lda, // lda
    PPDouble(devB_array_device), // B array
    ldb, // ldb
    @beta, // beta (host pointer)
    PPDouble(devC_array_device), // C array (output)
    ldc, // ldc
    batch_count
  );
  AssertEquals('rocblas_dsymm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // Synchronize to catch potential async errors
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_dsymm_batched', HIP_SUCCESS, sync_status);

  // Copy result back to host
  AssertEquals('hipMemcpy devC result', HIP_SUCCESS, hipMemcpy(@hostC[0], devC, ldc * N * SizeOf(Double) * batch_count, hipMemcpyDeviceToHost));

  // Validate results
  for i := 0 to ldc * N * batch_count - 1 do
  begin
    // AssertEquals('DSYMM_Batched C[' + IntToStr(i) + ']', expectedC[i], hostC[i], 1e-12);
    // Using CheckEqualsRelative-like tolerance check
    AssertEquals('DSYMM_Batched C[' + IntToStr(i) + ']', expectedC[i], hostC[i], 1e-10);
  end;


  // Cleanup
  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
end;


// --- CSYMM_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_CSYMM_Batched;
const
  M = 3; N = 2; batch_count = 1; // Smaller batch for complex
var
  side: rocblas_side;
  uplo: rocblas_fill;
  alpha, beta: rocblas_float_complex;
  dim_A: Integer;
  lda, ldb, ldc: Integer;
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC, expectedC: array of rocblas_float_complex;
  devA_array, devB_array, devC_array: array of Procfloat_complex;
  devA_array_device, devB_array_device, devC_array_device: PPointer;
  i, j, k, b: Integer;
  temp_r, temp_i, prod_r, prod_i: Double;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  side := rocblas_side_left;
  uplo := rocblas_fill_upper;
  alpha.x := 1.0; alpha.y := 0.5;
  beta.x := 0.8; beta.y := -0.2;

  if side = rocblas_side_left then
    dim_A := M
  else
    dim_A := N;

  lda := dim_A;
  ldb := M;
  ldc := M;

  SetLength(hostA, lda * dim_A * batch_count);
  SetLength(hostB, ldb * N * batch_count);
  SetLength(hostC, ldc * N * batch_count);
  SetLength(expectedC, ldc * N * batch_count);
  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Zero out matrices first
    for i := 0 to lda * dim_A - 1 do
    begin
      hostA[b * lda * dim_A + i].x := 0.0;
      hostA[b * lda * dim_A + i].y := 0.0;
    end;
    for i := 0 to ldb * N - 1 do
    begin
      hostB[b * ldb * N + i].x := 0.0;
      hostB[b * ldb * N + i].y := 0.0;
    end;
    for i := 0 to ldc * N - 1 do
    begin
      hostC[b * ldc * N + i].x := 0.0;
      hostC[b * ldc * N + i].y := 0.0;
    end;

    // Fill symmetric matrix A (upper triangle)
    if uplo = rocblas_fill_upper then
    begin
      for i := 0 to dim_A - 1 do
        for j := i to dim_A - 1 do
        begin
          hostA[b * lda * dim_A + j * lda + i].x := (i + j + 1 + b) * 0.4;
          hostA[b * lda * dim_A + j * lda + i].y := (i - j + b) * 0.1;
        end;
      // Copy to lower triangle (symmetric, not hermitian)
      for i := 0 to dim_A - 1 do
        for j := 0 to i - 1 do
        begin
          hostA[b * lda * dim_A + j * lda + i].x := hostA[b * lda * dim_A + i * lda + j].x;
          hostA[b * lda * dim_A + j * lda + i].y := hostA[b * lda * dim_A + i * lda + j].y;
        end;
    end;

    // Fill matrix B
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
      begin
        hostB[b * ldb * N + j * ldb + i].x := (i + j + 1 + b) * 0.2;
        hostB[b * ldb * N + j * ldb + i].y := (i - j + b) * -0.1;
      end;

    // Fill matrix C
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
      begin
        hostC[b * ldc * N + j * ldc + i].x := (i + j + 1 + b) * 0.05;
        hostC[b * ldc * N + j * ldc + i].y := (i - j + b) * 0.02;
      end;

    // Calculate expected result for this batch item: C = alpha * A * B + beta * C (side=left)
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
      begin
        // expectedC = beta * hostC
        expectedC[b * ldc * N + j * ldc + i].x := beta.x * hostC[b * ldc * N + j * ldc + i].x - beta.y * hostC[b * ldc * N + j * ldc + i].y;
        expectedC[b * ldc * N + j * ldc + i].y := beta.x * hostC[b * ldc * N + j * ldc + i].y + beta.y * hostC[b * ldc * N + j * ldc + i].x;

        for k := 0 to dim_A - 1 do // dim_A = M for side_left
        begin
          // temp = alpha * hostA * hostB
          // (a_r + a_i*i) * (b_r + b_i*i) = (a_r*b_r - a_i*b_i) + (a_r*b_i + a_i*b_r)*i
          temp_r := alpha.x * hostA[b * lda * dim_A + k * lda + i].x - alpha.y * hostA[b * lda * dim_A + k * lda + i].y;
          temp_i := alpha.x * hostA[b * lda * dim_A + k * lda + i].y + alpha.y * hostA[b * lda * dim_A + k * lda + i].x;

          prod_r := temp_r * hostB[b * ldb * N + j * ldb + k].x - temp_i * hostB[b * ldb * N + j * ldb + k].y;
          prod_i := temp_r * hostB[b * ldb * N + j * ldb + k].y + temp_i * hostB[b * ldb * N + j * ldb + k].x;

          expectedC[b * ldc * N + j * ldc + i].x := expectedC[b * ldc * N + j * ldc + i].x + prod_r;
          expectedC[b * ldc * N + j * ldc + i].y := expectedC[b * ldc * N + j * ldc + i].y + prod_i;
        end;
      end;
  end;

  // Allocate device memory for matrices
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, lda * dim_A * SizeOf(rocblas_float_complex) * batch_count));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, ldb * N * SizeOf(rocblas_float_complex) * batch_count));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, ldc * N * SizeOf(rocblas_float_complex) * batch_count));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], lda * dim_A * SizeOf(rocblas_float_complex) * batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], ldb * N * SizeOf(rocblas_float_complex) * batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC', HIP_SUCCESS, hipMemcpy(devC, @hostC[0], ldc * N * SizeOf(rocblas_float_complex) * batch_count, hipMemcpyHostToDevice));

  // Initialize device pointer arrays
  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procfloat_complex(PtrUInt(devA) + b * lda * dim_A * SizeOf(rocblas_float_complex));
    devB_array[b] := Procfloat_complex(PtrUInt(devB) + b * ldb * N * SizeOf(rocblas_float_complex));
    devC_array[b] := Procfloat_complex(PtrUInt(devC) + b * ldc * N * SizeOf(rocblas_float_complex));
  end;

  // Allocate and copy device pointer arrays
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count * SizeOf(Pointer)));

  AssertEquals('hipMemcpy devA_array_device', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set pointer mode (alpha, beta are host pointers)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // Call rocBLAS CSYMM_Batched
  blas_status := rocblas_csymm_batched(
    handle,
    side,
    uplo,
    M, N,
    @alpha, // alpha (host pointer)
    PPProcfloat_complex(devA_array_device), // A array
    lda, // lda
    PPProcfloat_complex(devB_array_device), // B array
    ldb, // ldb
    @beta, // beta (host pointer)
    PPProcfloat_complex(devC_array_device), // C array (output)
    ldc, // ldc
    batch_count
  );
  AssertEquals('rocblas_csymm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // Synchronize to catch potential async errors
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_csymm_batched', HIP_SUCCESS, sync_status);

  // Copy result back to host
  AssertEquals('hipMemcpy devC result', HIP_SUCCESS, hipMemcpy(@hostC[0], devC, ldc * N * SizeOf(rocblas_float_complex) * batch_count, hipMemcpyDeviceToHost));

  // Validate results
  for i := 0 to ldc * N * batch_count - 1 do
  begin
    AssertEquals('CSYMM_Batched C[' + IntToStr(i) + '].x', expectedC[i].x, hostC[i].x, 1e-3);
    AssertEquals('CSYMM_Batched C[' + IntToStr(i) + '].y', expectedC[i].y, hostC[i].y, 1e-3);
  end;


  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
end;


procedure TTestRocBLAS_Level3_Batched.Test_ZSYMM_Batched;
const
  M = 2; N = 3; batch_count = 2; // Different dimensions
var
  side: rocblas_side;
  uplo: rocblas_fill;
  alpha, beta: rocblas_double_complex;
  dim_A: Integer;
  lda, ldb, ldc: Integer;
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC, expectedC: array of rocblas_double_complex;
  devA_array, devB_array, devC_array: array of Procdouble_complex;
  devA_array_device, devB_array_device, devC_array_device: PPointer;
  i, j, k, b: Integer;
  temp_r, temp_i, prod_r, prod_i: Extended;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  side := rocblas_side_right; // Different side
  uplo := rocblas_fill_lower; // Different uplo
  alpha.x := 0.5; alpha.y := -0.3;
  beta.x := 1.1; beta.y := 0.4;

  if side = rocblas_side_left then
    dim_A := M
  else
    dim_A := N; // For side_right, dim_A = N

  lda := dim_A;
  ldb := M;
  ldc := M;

  SetLength(hostA, lda * dim_A * batch_count);
  SetLength(hostB, ldb * N * batch_count);
  SetLength(hostC, ldc * N * batch_count);
  SetLength(expectedC, ldc * N * batch_count);
  SetLength(devA_array, batch_count);
  SetLength(devB_array, batch_count);
  SetLength(devC_array, batch_count);

  // Initialize host data
  for b := 0 to batch_count - 1 do
  begin
    // Zero out matrices first
    for i := 0 to lda * dim_A - 1 do
    begin
      hostA[b * lda * dim_A + i].x := 0.0;
      hostA[b * lda * dim_A + i].y := 0.0;
    end;
    for i := 0 to ldb * N - 1 do
    begin
      hostB[b * ldb * N + i].x := 0.0;
      hostB[b * ldb * N + i].y := 0.0;
    end;
    for i := 0 to ldc * N - 1 do
    begin
      hostC[b * ldc * N + i].x := 0.0;
      hostC[b * ldc * N + i].y := 0.0;
    end;

    // Fill symmetric matrix A (lower triangle for this test)
    if uplo = rocblas_fill_lower then
    begin
      for i := 0 to dim_A - 1 do
        for j := 0 to i do
        begin
          hostA[b * lda * dim_A + j * lda + i].x := (i + j + 1 + b) * 0.6;
          hostA[b * lda * dim_A + j * lda + i].y := (i - j + b) * -0.15;
        end;
      // Copy to upper triangle (symmetric, not hermitian)
      for i := 0 to dim_A - 1 do
        for j := i + 1 to dim_A - 1 do
        begin
          hostA[b * lda * dim_A + j * lda + i].x := hostA[b * lda * dim_A + i * lda + j].x;
          hostA[b * lda * dim_A + j * lda + i].y := hostA[b * lda * dim_A + i * lda + j].y;
        end;
    end;

    // Fill matrix B
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
      begin
        hostB[b * ldb * N + j * ldb + i].x := (i + j + 1 + b) * 0.25;
        hostB[b * ldb * N + j * ldb + i].y := (i - j + b) * 0.08;
      end;

    // Fill matrix C
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
      begin
        hostC[b * ldc * N + j * ldc + i].x := (i + j + 1 + b) * -0.07;
        hostC[b * ldc * N + j * ldc + i].y := (i - j + b) * 0.03;
      end;

    // Calculate expected result for this batch item: C = alpha * B * A + beta * C (side=right)
    for i := 0 to M - 1 do
      for j := 0 to N - 1 do
      begin
        // expectedC = beta * hostC
        expectedC[b * ldc * N + j * ldc + i].x := beta.x * hostC[b * ldc * N + j * ldc + i].x - beta.y * hostC[b * ldc * N + j * ldc + i].y;
        expectedC[b * ldc * N + j * ldc + i].y := beta.x * hostC[b * ldc * N + j * ldc + i].y + beta.y * hostC[b * ldc * N + j * ldc + i].x;

        for k := 0 to dim_A - 1 do // dim_A = N for side_right
        begin
          // temp = alpha * hostB * hostA
          temp_r := alpha.x * hostB[b * ldb * N + k * ldb + i].x - alpha.y * hostB[b * ldb * N + k * ldb + i].y;
          temp_i := alpha.x * hostB[b * ldb * N + k * ldb + i].y + alpha.y * hostB[b * ldb * N + k * ldb + i].x;

          prod_r := temp_r * hostA[b * lda * dim_A + j * lda + k].x - temp_i * hostA[b * lda * dim_A + j * lda + k].y;
          prod_i := temp_r * hostA[b * lda * dim_A + j * lda + k].y + temp_i * hostA[b * lda * dim_A + j * lda + k].x;

          expectedC[b * ldc * N + j * ldc + i].x := expectedC[b * ldc * N + j * ldc + i].x + prod_r;
          expectedC[b * ldc * N + j * ldc + i].y := expectedC[b * ldc * N + j * ldc + i].y + prod_i;
        end;
      end;
  end;

  // Allocate device memory for matrices
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, lda * dim_A * SizeOf(rocblas_double_complex) * batch_count));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, ldb * N * SizeOf(rocblas_double_complex) * batch_count));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, ldc * N * SizeOf(rocblas_double_complex) * batch_count));

  AssertEquals('hipMemcpy devA', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], lda * dim_A * SizeOf(rocblas_double_complex) * batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], ldb * N * SizeOf(rocblas_double_complex) * batch_count, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC', HIP_SUCCESS, hipMemcpy(devC, @hostC[0], ldc * N * SizeOf(rocblas_double_complex) * batch_count, hipMemcpyHostToDevice));

  // Initialize device pointer arrays
  for b := 0 to batch_count - 1 do
  begin
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + b * lda * dim_A * SizeOf(rocblas_double_complex));
    devB_array[b] := Procdouble_complex(PtrUInt(devB) + b * ldb * N * SizeOf(rocblas_double_complex));
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + b * ldc * N * SizeOf(rocblas_double_complex));
  end;

  // Allocate and copy device pointer arrays
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count * SizeOf(Pointer)));

  AssertEquals('hipMemcpy devA_array_device', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set pointer mode (alpha, beta are host pointers)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // Call rocBLAS ZSYMM_Batched
  blas_status := rocblas_zsymm_batched(
    handle,
    side,
    uplo,
    M, N,
    @alpha, // alpha (host pointer)
    PPProcdouble_complex(devA_array_device), // A array
    lda, // lda
    PPProcdouble_complex(devB_array_device), // B array
    ldb, // ldb
    @beta, // beta (host pointer)
    PPProcdouble_complex(devC_array_device), // C array (output)
    ldc, // ldc
    batch_count
  );
  AssertEquals('rocblas_zsymm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // Synchronize to catch potential async errors
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_zsymm_batched', HIP_SUCCESS, sync_status);

  // Copy result back to host
  AssertEquals('hipMemcpy devC result', HIP_SUCCESS, hipMemcpy(@hostC[0], devC, ldc * N * SizeOf(rocblas_double_complex) * batch_count, hipMemcpyDeviceToHost));

  // Validate results
  for i := 0 to ldc * N * batch_count - 1 do
  begin
    AssertEquals('ZSYMM_Batched C[' + IntToStr(i) + '].x', expectedC[i].x, hostC[i].x, 1e-12);
    AssertEquals('ZSYMM_Batched C[' + IntToStr(i) + '].y', expectedC[i].y, hostC[i].y, 1e-12);
  end;


  // Cleanup
  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
end;

// --- STRMM_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_STRMM_Batched;
const
  M_val = 2;
  N_val = 3;
  batch_count_val = 2;
var
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  alpha: Single;
  dim_A: Integer;
  lda_val, ldb_val, ldc_val: Integer; // Include ldc_val
  devA, devB: hipDeviceptr_t;
  hostA, hostB_initial, hostB_computed: array of Single;
  devA_array, devB_array: array of PSingle;
  devA_array_device, devB_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  side := rocblas_side_left;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;
  alpha := 1.5;

  if side = rocblas_side_left then
    dim_A := M_val
  else
    dim_A := N_val;

  lda_val := dim_A;
  ldb_val := M_val;
  ldc_val := ldb_val; // For in-place operation, ldc must equal ldb


  // Validate sizes before allocation (basic checks)
  AssertTrue('M must be > 0', M_val > 0);
  AssertTrue('N must be > 0', N_val > 0);
  AssertTrue('lda must be >= max(1, dim_A)', lda_val >= Max(1, dim_A));
  AssertTrue('ldb must be >= max(1, M)', ldb_val >= Max(1, M_val));
  AssertTrue('ldc must be >= max(1, M)', ldc_val >= Max(1, M_val)); // Added ldc check
  AssertTrue('batch_count must be > 0', batch_count_val > 0);

  SetLength(hostA, lda_val * dim_A * batch_count_val);
  SetLength(hostB_initial, ldb_val * N_val * batch_count_val);
  SetLength(hostB_computed, ldb_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devB_array, batch_count_val);

  // Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out matrices first
    for i := 0 to lda_val * dim_A - 1 do
      hostA[b * lda_val * dim_A + i] := 0.0;
    for i := 0 to ldb_val * N_val - 1 do
      hostB_initial[b * ldb_val * N_val + i] := 0.0;

    // Fill triangular matrix A (upper triangle)
    if uplo = rocblas_fill_upper then
    begin
      for i := 0 to dim_A - 1 do
        for j := i to dim_A - 1 do
          hostA[b * lda_val * dim_A + j * lda_val + i] := (i + j + 1 + b) * 0.5;
      // Lower triangle remains zero
    end;

    // Fill matrix B initial
    for i := 0 to M_val - 1 do
      for j := 0 to N_val - 1 do
        hostB_initial[b * ldb_val * N_val + j * ldb_val + i] := (i + j + 1 + b) * 0.2;

    // Copy initial B to computed B (for input and result comparison)
    Move(hostB_initial[b * ldb_val * N_val], hostB_computed[b * ldb_val * N_val], ldb_val * N_val * SizeOf(Single));
  end;

  // Allocate device memory for matrices
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, lda_val * dim_A * SizeOf(Single) * batch_count_val));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, ldb_val * N_val * SizeOf(Single) * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devB should not be nil', devB <> nil);

  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], lda_val * dim_A * SizeOf(Single) * batch_count_val, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB_computed[0], ldb_val * N_val * SizeOf(Single) * batch_count_val, hipMemcpyHostToDevice));

  // Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    devA_array[b] := PSingle(PtrUInt(devA) + (b * lda_val * dim_A * SizeOf(Single)));
    devB_array[b] := PSingle(PtrUInt(devB) + (b * ldb_val * N_val * SizeOf(Single)));
    // Note: devB_array will be used for both B (input) and C (output) pointers
  end;

  // Allocate and copy device pointer arrays
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devB_array_device should not be nil', devB_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device H2D', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set pointer mode (alpha is host pointer)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // Call rocBLAS STRMM_Batched

  blas_status := rocblas_strmm_batched(
    handle,
    side,
    uplo,
    transA,
    diag,
    M_val, N_val,
    @alpha, // alpha (host pointer)
    PPSingle(devA_array_device), // A array (array of input matrix pointers)
    lda_val, // lda
    PPSingle(devB_array_device), // B array (array of input matrix pointers)
    ldb_val, // ldb
    PPSingle(devB_array_device), // C array (array of output matrix pointers) - SAME as B for in-place
    ldc_val, // ldc - SAME as ldb for in-place operation
    batch_count_val // batch_count (14th parameter)
  );
  AssertEquals('rocblas_strmm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // Synchronize to catch potential async errors
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_strmm_batched', HIP_SUCCESS, sync_status);

  // Copy result back to host (result overwrites hostB_computed)
  AssertEquals('hipMemcpy devB D2H', HIP_SUCCESS, hipMemcpy(@hostB_computed[0], devB, ldb_val * N_val * SizeOf(Single) * batch_count_val, hipMemcpyDeviceToHost));

  // Validate results (basic difference check per batch)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('STRMM_Batched result B should differ from initial B for batch ' + IntToStr(b),
      not CompareMem(@hostB_initial[b * ldb_val * N_val], @hostB_computed[b * ldb_val * N_val], ldb_val * N_val * SizeOf(Single)));
  end;


  // Cleanup
  hipFree(devA);
  hipFree(devB);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
end;



// --- DTRMM_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_DTRMM_Batched;
const
  M_val = 3;
  N_val = 2;
  batch_count_val = 3; // Different dimensions
var
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  alpha: Double;
  dim_A: Integer;
  lda_val, ldb_val, ldc_val: Integer; // Include ldc_val
  devA, devB: hipDeviceptr_t;
  hostA, hostB_initial, hostB_computed: array of Double;
  devA_array, devB_array: array of PDouble;
  devA_array_device, devB_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  side := rocblas_side_right; // Different side
  uplo := rocblas_fill_lower; // Different uplo
  transA := rocblas_operation_transpose; // Different transA
  diag := rocblas_diagonal_unit; // Different diag
  alpha := 0.8;

  if side = rocblas_side_left then
    dim_A := M_val
  else
    dim_A := N_val; // For side_right, dim_A = N

  lda_val := dim_A;
  ldb_val := M_val;
  ldc_val := ldb_val; // For in-place operation, ldc must equal ldb

  // Validate sizes before allocation (basic checks)
  AssertTrue('M must be > 0', M_val > 0);
  AssertTrue('N must be > 0', N_val > 0);
  AssertTrue('lda must be >= max(1, dim_A)', lda_val >= Max(1, dim_A));
  AssertTrue('ldb must be >= max(1, M)', ldb_val >= Max(1, M_val));
  AssertTrue('ldc must be >= max(1, M)', ldc_val >= Max(1, M_val));
  AssertTrue('batch_count must be > 0', batch_count_val > 0);

  SetLength(hostA, lda_val * dim_A * batch_count_val);
  SetLength(hostB_initial, ldb_val * N_val * batch_count_val);
  SetLength(hostB_computed, ldb_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devB_array, batch_count_val);

  // Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out matrices first
    for i := 0 to lda_val * dim_A - 1 do
      hostA[b * lda_val * dim_A + i] := 0.0;
    for i := 0 to ldb_val * N_val - 1 do
      hostB_initial[b * ldb_val * N_val + i] := 0.0;

    // Fill triangular matrix A (lower triangle, unit diagonal for this test)
    if uplo = rocblas_fill_lower then
    begin
      for i := 0 to dim_A - 1 do
        for j := 0 to i do
        begin
          if (diag = rocblas_diagonal_unit) and (i = j) then
            hostA[b * lda_val * dim_A + j * lda_val + i] := 1.0 // Explicitly set unit diagonal
          else
            hostA[b * lda_val * dim_A + j * lda_val + i] := (i + j + 1 + b) * 0.7;
        end;
      // Upper triangle remains zero
    end;

    // Fill matrix B initial
    for i := 0 to M_val - 1 do
      for j := 0 to N_val - 1 do
        hostB_initial[b * ldb_val * N_val + j * ldb_val + i] := (i + j + 1 + b) * 0.3;

    // Copy initial B to computed B (for input and result comparison)
    Move(hostB_initial[b * ldb_val * N_val], hostB_computed[b * ldb_val * N_val], ldb_val * N_val * SizeOf(Double));
  end;

  // Allocate device memory for matrices
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, lda_val * dim_A * SizeOf(Double) * batch_count_val));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, ldb_val * N_val * SizeOf(Double) * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devB should not be nil', devB <> nil);

  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], lda_val * dim_A * SizeOf(Double) * batch_count_val, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB_computed[0], ldb_val * N_val * SizeOf(Double) * batch_count_val, hipMemcpyHostToDevice));

  // Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    devA_array[b] := PDouble(PtrUInt(devA) + (b * lda_val * dim_A * SizeOf(Double)));
    devB_array[b] := PDouble(PtrUInt(devB) + (b * ldb_val * N_val * SizeOf(Double)));
    // Note: devB_array will be used for both B (input) and C (output) pointers
  end;

  // Allocate and copy device pointer arrays
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devB_array_device should not be nil', devB_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device H2D', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set pointer mode (alpha is host pointer)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // Call rocBLAS DTRMM_Batched
  blas_status := rocblas_dtrmm_batched(
    handle,
    side,
    uplo,
    transA,
    diag,
    M_val, N_val,
    @alpha, // alpha (host pointer)
    PPDouble(devA_array_device), // A array (array of input matrix pointers)
    lda_val, // lda
    PPDouble(devB_array_device), // B array (array of input matrix pointers)
    ldb_val, // ldb
    PPDouble(devB_array_device), // C array (array of output matrix pointers) - SAME as B for in-place
    ldc_val, // ldc - SAME as ldb for in-place operation
    batch_count_val // batch_count (14th parameter)
  );
  AssertEquals('rocblas_dtrmm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // Synchronize to catch potential async errors
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_dtrmm_batched', HIP_SUCCESS, sync_status);

  // Copy result back to host (result overwrites hostB_computed)
  AssertEquals('hipMemcpy devB D2H', HIP_SUCCESS, hipMemcpy(@hostB_computed[0], devB, ldb_val * N_val * SizeOf(Double) * batch_count_val, hipMemcpyDeviceToHost));

  // Validate results (basic difference check per batch)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('DTRMM_Batched result B should differ from initial B for batch ' + IntToStr(b),
      not CompareMem(@hostB_initial[b * ldb_val * N_val], @hostB_computed[b * ldb_val * N_val], ldb_val * N_val * SizeOf(Double)));
  end;


  // Cleanup
  hipFree(devA);
  hipFree(devB);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
end;


// --- CTRMM_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_CTRMM_Batched;
const
  M_val = 2;
  N_val = 2;
  batch_count_val = 1; // Smaller batch for complex
var
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  alpha: rocblas_float_complex;
  dim_A: Integer;
  lda_val, ldb_val, ldc_val: Integer; // Include ldc_val
  devA, devB: hipDeviceptr_t;
  hostA, hostB_initial, hostB_computed: array of rocblas_float_complex;
  devA_array, devB_array: array of Procfloat_complex;
  devA_array_device, devB_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  side := rocblas_side_left;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;
  alpha.x := 1.0;
  alpha.y := 0.5;

  if side = rocblas_side_left then
    dim_A := M_val
  else
    dim_A := N_val;

  lda_val := dim_A;
  ldb_val := M_val;
  ldc_val := ldb_val; // For in-place operation, ldc must equal ldb

  // Validate sizes before allocation (basic checks)
  AssertTrue('M must be > 0', M_val > 0);
  AssertTrue('N must be > 0', N_val > 0);
  AssertTrue('lda must be >= max(1, dim_A)', lda_val >= Max(1, dim_A));
  AssertTrue('ldb must be >= max(1, M)', ldb_val >= Max(1, M_val));
  AssertTrue('ldc must be >= max(1, M)', ldc_val >= Max(1, M_val));
  AssertTrue('batch_count must be > 0', batch_count_val > 0);

  SetLength(hostA, lda_val * dim_A * batch_count_val);
  SetLength(hostB_initial, ldb_val * N_val * batch_count_val);
  SetLength(hostB_computed, ldb_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devB_array, batch_count_val);

  // Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out matrices first
    for i := 0 to lda_val * dim_A - 1 do
    begin
      hostA[b * lda_val * dim_A + i].x := 0.0;
      hostA[b * lda_val * dim_A + i].y := 0.0;
    end;
    for i := 0 to ldb_val * N_val - 1 do
    begin
      hostB_initial[b * ldb_val * N_val + i].x := 0.0;
      hostB_initial[b * ldb_val * N_val + i].y := 0.0;
    end;

    // Fill triangular matrix A (upper triangle)
    if uplo = rocblas_fill_upper then
    begin
      for i := 0 to dim_A - 1 do
        for j := i to dim_A - 1 do
        begin
          hostA[b * lda_val * dim_A + j * lda_val + i].x := (i + j + 1 + b) * 0.4;
          hostA[b * lda_val * dim_A + j * lda_val + i].y := (i - j + b) * 0.1;
        end;
      // Lower triangle remains zero
    end;

    // Fill matrix B initial
    for i := 0 to M_val - 1 do
      for j := 0 to N_val - 1 do
      begin
        hostB_initial[b * ldb_val * N_val + j * ldb_val + i].x := (i + j + 1 + b) * 0.2;
        hostB_initial[b * ldb_val * N_val + j * ldb_val + i].y := (i - j + b) * -0.1;
      end;

    // Copy initial B to computed B (for input and result comparison)
    Move(hostB_initial[b * ldb_val * N_val], hostB_computed[b * ldb_val * N_val], ldb_val * N_val * SizeOf(rocblas_float_complex));
  end;

  // Allocate device memory for matrices
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, lda_val * dim_A * SizeOf(rocblas_float_complex) * batch_count_val));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, ldb_val * N_val * SizeOf(rocblas_float_complex) * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devB should not be nil', devB <> nil);

  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], lda_val * dim_A * SizeOf(rocblas_float_complex) * batch_count_val, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB_computed[0], ldb_val * N_val * SizeOf(rocblas_float_complex) * batch_count_val, hipMemcpyHostToDevice));

  // Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    devA_array[b] := Procfloat_complex(PtrUInt(devA) + (b * lda_val * dim_A * SizeOf(rocblas_float_complex)));
    devB_array[b] := Procfloat_complex(PtrUInt(devB) + (b * ldb_val * N_val * SizeOf(rocblas_float_complex)));
    // Note: devB_array will be used for both B (input) and C (output) pointers
  end;

  // Allocate and copy device pointer arrays
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devB_array_device should not be nil', devB_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device H2D', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set pointer mode (alpha is host pointer)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // Call rocBLAS CTRMM_Batched
  blas_status := rocblas_ctrmm_batched(
    handle,
    side,
    uplo,
    transA,
    diag,
    M_val, N_val,
    @alpha, // alpha (host pointer)
    PPProcfloat_complex(devA_array_device), // A array (array of input matrix pointers)
    lda_val, // lda
    PPProcfloat_complex(devB_array_device), // B array (array of input matrix pointers)
    ldb_val, // ldb
    PPProcfloat_complex(devB_array_device), // C array (array of output matrix pointers) - SAME as B for in-place
    ldc_val, // ldc - SAME as ldb for in-place operation
    batch_count_val // batch_count (14th parameter)
  );
  AssertEquals('rocblas_ctrmm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // Synchronize to catch potential async errors
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ctrmm_batched', HIP_SUCCESS, sync_status);

  // Copy result back to host (result overwrites hostB_computed)
  AssertEquals('hipMemcpy devB D2H', HIP_SUCCESS, hipMemcpy(@hostB_computed[0], devB, ldb_val * N_val * SizeOf(rocblas_float_complex) * batch_count_val, hipMemcpyDeviceToHost));

  // Validate results (basic difference check per batch)
  for b := 0 to batch_count_val - 1 do
  begin
     AssertTrue('CTRMM_Batched result B should differ from initial B for batch ' + IntToStr(b),
      not CompareMem(@hostB_initial[b * ldb_val * N_val], @hostB_computed[b * ldb_val * N_val], ldb_val * N_val * SizeOf(rocblas_float_complex)));
  end;


  // Cleanup
  hipFree(devA);
  hipFree(devB);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
end;


// --- ZTRMM_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_ZTRMM_Batched;
const
  M_val = 3;
  N_val = 3;
  batch_count_val = 2; // Different dimensions
var
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  alpha: rocblas_double_complex;
  dim_A: Integer;
  lda_val, ldb_val, ldc_val: Integer; // Include ldc_val
  devA, devB: hipDeviceptr_t;
  hostA, hostB_initial, hostB_computed: array of rocblas_double_complex;
  devA_array, devB_array: array of Procdouble_complex;
  devA_array_device, devB_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  side := rocblas_side_right; // Different side
  uplo := rocblas_fill_lower; // Different uplo
  transA := rocblas_operation_conjugate_transpose; // Different transA
  diag := rocblas_diagonal_unit; // Different diag
  alpha.x := 0.5;
  alpha.y := -0.3;

  if side = rocblas_side_left then
    dim_A := M_val
  else
    dim_A := N_val; // For side_right, dim_A = N

  lda_val := dim_A;
  ldb_val := M_val;
  ldc_val := ldb_val; // For in-place operation, ldc must equal ldb

  // Validate sizes before allocation (basic checks)
  AssertTrue('M must be > 0', M_val > 0);
  AssertTrue('N must be > 0', N_val > 0);
  AssertTrue('lda must be >= max(1, dim_A)', lda_val >= Max(1, dim_A));
  AssertTrue('ldb must be >= max(1, M)', ldb_val >= Max(1, M_val));
  AssertTrue('ldc must be >= max(1, M)', ldc_val >= Max(1, M_val));
  AssertTrue('batch_count must be > 0', batch_count_val > 0);

  SetLength(hostA, lda_val * dim_A * batch_count_val);
  SetLength(hostB_initial, ldb_val * N_val * batch_count_val);
  SetLength(hostB_computed, ldb_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devB_array, batch_count_val);

  // Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out matrices first
    for i := 0 to lda_val * dim_A - 1 do
    begin
      hostA[b * lda_val * dim_A + i].x := 0.0;
      hostA[b * lda_val * dim_A + i].y := 0.0;
    end;
    for i := 0 to ldb_val * N_val - 1 do
    begin
      hostB_initial[b * ldb_val * N_val + i].x := 0.0;
      hostB_initial[b * ldb_val * N_val + i].y := 0.0;
    end;

    // Fill triangular matrix A (lower triangle, unit diagonal for this test)
    if uplo = rocblas_fill_lower then
    begin
      for i := 0 to dim_A - 1 do
        for j := 0 to i do
        begin
          if (diag = rocblas_diagonal_unit) and (i = j) then
            hostA[b * lda_val * dim_A + j * lda_val + i].x := 1.0 // Explicitly set unit diagonal
          else
          begin
            hostA[b * lda_val * dim_A + j * lda_val + i].x := (i + j + 1 + b) * 0.6;
            hostA[b * lda_val * dim_A + j * lda_val + i].y := (i - j + b) * -0.15;
          end;
        end;
      // Upper triangle remains zero
    end;

    // Fill matrix B initial
    for i := 0 to M_val - 1 do
      for j := 0 to N_val - 1 do
      begin
        hostB_initial[b * ldb_val * N_val + j * ldb_val + i].x := (i + j + 1 + b) * 0.25;
        hostB_initial[b * ldb_val * N_val + j * ldb_val + i].y := (i - j + b) * 0.08;
      end;

    // Copy initial B to computed B (for input and result comparison)
    Move(hostB_initial[b * ldb_val * N_val], hostB_computed[b * ldb_val * N_val], ldb_val * N_val * SizeOf(rocblas_double_complex));
  end;

  // Allocate device memory for matrices
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, lda_val * dim_A * SizeOf(rocblas_double_complex) * batch_count_val));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, ldb_val * N_val * SizeOf(rocblas_double_complex) * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devB should not be nil', devB <> nil);

  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], lda_val * dim_A * SizeOf(rocblas_double_complex) * batch_count_val, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB_computed[0], ldb_val * N_val * SizeOf(rocblas_double_complex) * batch_count_val, hipMemcpyHostToDevice));

  // Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + (b * lda_val * dim_A * SizeOf(rocblas_double_complex)));
    devB_array[b] := Procdouble_complex(PtrUInt(devB) + (b * ldb_val * N_val * SizeOf(rocblas_double_complex)));
    // Note: devB_array will be used for both B (input) and C (output) pointers
  end;

  // Allocate and copy device pointer arrays
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devB_array_device should not be nil', devB_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device H2D', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set pointer mode (alpha is host pointer)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // Call rocBLAS ZTRMM_Batched
  blas_status := rocblas_ztrmm_batched(
    handle,
    side,
    uplo,
    transA,
    diag,
    M_val, N_val,
    @alpha, // alpha (host pointer)
    PPProcdouble_complex(devA_array_device), // A array (array of input matrix pointers)
    lda_val, // lda
    PPProcdouble_complex(devB_array_device), // B array (array of input matrix pointers)
    ldb_val, // ldb
    PPProcdouble_complex(devB_array_device), // C array (array of output matrix pointers) - SAME as B for in-place
    ldc_val, // ldc - SAME as ldb for in-place operation
    batch_count_val // batch_count (14th parameter)
  );
  AssertEquals('rocblas_ztrmm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // Synchronize to catch potential async errors
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ztrmm_batched', HIP_SUCCESS, sync_status);

  // Copy result back to host (result overwrites hostB_computed)
  AssertEquals('hipMemcpy devB D2H', HIP_SUCCESS, hipMemcpy(@hostB_computed[0], devB, ldb_val * N_val * SizeOf(rocblas_double_complex) * batch_count_val, hipMemcpyDeviceToHost));

  // Validate results (basic difference check per batch)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('ZTRMM_Batched result B should differ from initial B for batch ' + IntToStr(b),
      not CompareMem(@hostB_initial[b * ldb_val * N_val], @hostB_computed[b * ldb_val * N_val], ldb_val * N_val * SizeOf(rocblas_double_complex)));
  end;

  // Cleanup
  hipFree(devA);
  hipFree(devB);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
end;


// --- STRSM_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_STRSM_Batched;
const
  M_val = 2;
  N_val = 3;
  batch_count_val = 2;
var
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  alpha: Single;
  dim_A: Integer;
  lda_val, ldb_val: Integer;
  devA, devB: hipDeviceptr_t;
  hostA, hostB_initial, hostB_computed: array of Single; // Real Single Precision
  devA_array, devB_array: array of PSingle;
  devA_array_device, devB_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  // 1. Initialize matrix dimensions and parameters
  side := rocblas_side_left;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;
  alpha := 2.0;

  if side = rocblas_side_left then
    dim_A := M_val
  else
    dim_A := N_val;

  lda_val := dim_A;
  ldb_val := M_val;

  // 2. Set up test data
  SetLength(hostA, lda_val * dim_A * batch_count_val);
  SetLength(hostB_initial, ldb_val * N_val * batch_count_val);
  SetLength(hostB_computed, ldb_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devB_array, batch_count_val);

  // 3. Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out matrices first
    for i := 0 to lda_val * dim_A - 1 do
      hostA[b * lda_val * dim_A + i] := 0.0;
    for i := 0 to ldb_val * N_val - 1 do
      hostB_initial[b * ldb_val * N_val + i] := 0.0;

    // Fill triangular matrix A (Upper triangle, non-singular)
    // A = [ 2.0  1.0 ]
    //     [ 0.0  3.0 ]
    if uplo = rocblas_fill_upper then
    begin
      for i := 0 to dim_A - 1 do
        for j := i to dim_A - 1 do
          hostA[b * lda_val * dim_A + j * lda_val + i] := (i + j + 1 + b) * 0.5 + 1.0; // Ensure diagonal dominance/non-zero
      // Lower triangle remains zero
    end;

    // Fill matrix B initial (RHS)
    for i := 0 to M_val - 1 do
      for j := 0 to N_val - 1 do
        hostB_initial[b * ldb_val * N_val + j * ldb_val + i] := (i + j + 1 + b) * 0.2;

    // Copy initial B matrix for computation (RHS -> Solution)
    Move(hostB_initial[b * ldb_val * N_val], hostB_computed[b * ldb_val * N_val], SizeOf(Single) * ldb_val * N_val);
  end;

  // 4. Allocate device memory
  devA := nil;
  devB := nil;
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(Single) * lda_val * dim_A * batch_count_val));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, SizeOf(Single) * ldb_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devB should not be nil', devB <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(Single) * lda_val * dim_A * batch_count_val, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB_computed[0], SizeOf(Single) * ldb_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  // 6. Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    devA_array[b] := PSingle(PtrUInt(devA) + b * lda_val * dim_A * SizeOf(Single));
    devB_array[b] := PSingle(PtrUInt(devB) + b * ldb_val * N_val * SizeOf(Single));
  end;

  // 7. Allocate and copy device pointer arrays
  devA_array_device := nil;
  devB_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devB_array_device should not be nil', devB_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device H2D', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // 8. Set rocBLAS pointer mode (alpha is real scalar on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // 9. Call rocblas_strsm_batched (12 parameters - CORRECT for TRSM_Batched - in-place solve)
  blas_status := rocblas_strsm_batched(
    handle,
    side,
    uplo,
    transA,
    diag,
    M_val, N_val,
    @alpha,   // alpha (real, host pointer)
    PPSingle(devA_array_device), // A array (array of device pointers to triangular matrices)
    lda_val, // lda
    PPSingle(devB_array_device), // B array (array of device pointers to RHS matrices, overwritten by solution X)
    ldb_val, // ldb
    batch_count_val // batch_count
  );
  AssertEquals('rocblas_strsm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // 10. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_strsm_batched', HIP_SUCCESS, sync_status);

  // 11. Copy result D2H (Solution X overwrites hostB_computed)
  AssertEquals('hipMemcpy devB D2H', HIP_SUCCESS, hipMemcpy(@hostB_computed[0], devB, SizeOf(Single) * ldb_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  // 12. Basic check (check if the solution matrix differs from the initial RHS)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('Solution matrix B should differ from initial RHS matrix B for batch ' + IntToStr(b),
      not CompareMem(@hostB_initial[b * ldb_val * N_val], @hostB_computed[b * ldb_val * N_val], SizeOf(Single) * ldb_val * N_val));
  end;

  // 13. Free device memory
  hipFree(devA);
  hipFree(devB);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
end;


// --- DTRSM_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_DTRSM_Batched;
const
  M_val = 3;
  N_val = 2;
  batch_count_val = 3; // Different dimensions
var
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  alpha: Double; // Real Double Precision Scalar
  dim_A: Integer;
  lda_val, ldb_val: Integer;
  devA, devB: hipDeviceptr_t;
  hostA, hostB_initial, hostB_computed: array of Double; // Real Double Precision
  devA_array, devB_array: array of PDouble;
  devA_array_device, devB_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  // 1. Initialize matrix dimensions and parameters
  side := rocblas_side_right; // Different side
  uplo := rocblas_fill_lower; // Different uplo
  transA := rocblas_operation_transpose; // Different transA
  diag := rocblas_diagonal_unit; // Different diag
  alpha := 0.5;

  if side = rocblas_side_left then
    dim_A := M_val
  else
    dim_A := N_val; // For side=right, dim_A = N

  lda_val := dim_A;
  ldb_val := M_val;

  // 2. Set up test data
  SetLength(hostA, lda_val * dim_A * batch_count_val);
  SetLength(hostB_initial, ldb_val * N_val * batch_count_val);
  SetLength(hostB_computed, ldb_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devB_array, batch_count_val);

  // 3. Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out matrices first
    for i := 0 to lda_val * dim_A - 1 do
      hostA[b * lda_val * dim_A + i] := 0.0;
    for i := 0 to ldb_val * N_val - 1 do
      hostB_initial[b * ldb_val * N_val + i] := 0.0;

    // Fill triangular matrix A (Lower triangle, unit diagonal)
    // A = [ 1.0  0.0 ] (Upper part ignored)
    //     [ 1.5  1.0 ] (Lower triangle, diag implicitly 1)
    if uplo = rocblas_fill_lower then
    begin
      for i := 0 to dim_A - 1 do
        for j := 0 to i do
        begin
          if (diag = rocblas_diagonal_unit) and (i = j) then
            hostA[b * lda_val * dim_A + j * lda_val + i] := 1.0 // Explicit unit diag for clarity, though rocBLAS ignores it
          else
            hostA[b * lda_val * dim_A + j * lda_val + i] := (i + j + 1 + b) * 0.7 + 0.5;
        end;
      // Upper triangle remains zero
    end;

    // Fill matrix B initial (RHS)
    for i := 0 to M_val - 1 do
      for j := 0 to N_val - 1 do
        hostB_initial[b * ldb_val * N_val + j * ldb_val + i] := (i + j + 1 + b) * 0.3;

    // Copy initial B matrix for computation (RHS -> Solution)
    Move(hostB_initial[b * ldb_val * N_val], hostB_computed[b * ldb_val * N_val], SizeOf(Double) * ldb_val * N_val);
  end;

  // 4. Allocate device memory
  devA := nil;
  devB := nil;
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(Double) * lda_val * dim_A * batch_count_val));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, SizeOf(Double) * ldb_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devB should not be nil', devB <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(Double) * lda_val * dim_A * batch_count_val, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB_computed[0], SizeOf(Double) * ldb_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  // 6. Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    devA_array[b] := PDouble(PtrUInt(devA) + b * lda_val * dim_A * SizeOf(Double));
    devB_array[b] := PDouble(PtrUInt(devB) + b * ldb_val * N_val * SizeOf(Double));
  end;

  // 7. Allocate and copy device pointer arrays
  devA_array_device := nil;
  devB_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devB_array_device should not be nil', devB_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device H2D', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // 8. Set rocBLAS pointer mode (alpha is real scalar on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // 9. Call rocblas_dtrsm_batched (12 parameters - CORRECT for TRSM_Batched - in-place solve)
  blas_status := rocblas_dtrsm_batched(
    handle,
    side,
    uplo,
    transA,
    diag,
    M_val, N_val,
    @alpha,   // alpha (real, host pointer)
    PPDouble(devA_array_device), // A array (array of device pointers to triangular matrices)
    lda_val, // lda
    PPDouble(devB_array_device), // B array (array of device pointers to RHS matrices, overwritten by solution X)
    ldb_val, // ldb
    batch_count_val // batch_count
  );
  AssertEquals('rocblas_dtrsm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // 10. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_dtrsm_batched', HIP_SUCCESS, sync_status);

  // 11. Copy result D2H (Solution X overwrites hostB_computed)
  AssertEquals('hipMemcpy devB D2H', HIP_SUCCESS, hipMemcpy(@hostB_computed[0], devB, SizeOf(Double) * ldb_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  // 12. Basic check (check if the solution matrix differs from the initial RHS)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('Solution matrix B should differ from initial RHS matrix B for batch ' + IntToStr(b),
      not CompareMem(@hostB_initial[b * ldb_val * N_val], @hostB_computed[b * ldb_val * N_val], SizeOf(Double) * ldb_val * N_val));
  end;

  // 13. Free device memory
  hipFree(devA);
  hipFree(devB);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
end;


// --- CTRSM_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_CTRSM_Batched;
const
  M_val = 2;
  N_val = 2;
  batch_count_val = 1; // Smaller batch for complex
var
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  alpha: rocblas_float_complex;
  dim_A: Integer;
  lda_val, ldb_val: Integer;
  devA, devB: hipDeviceptr_t;
  hostA, hostB_initial, hostB_computed: array of rocblas_float_complex;
  devA_array, devB_array: array of Procfloat_complex;
  devA_array_device, devB_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  // 1. Initialize matrix dimensions and parameters
  side := rocblas_side_left;
  uplo := rocblas_fill_upper;
  transA := rocblas_operation_none;
  diag := rocblas_diagonal_non_unit;
  alpha.x := 2.0;
  alpha.y := 0.0;

  if side = rocblas_side_left then
    dim_A := M_val
  else
    dim_A := N_val;

  lda_val := dim_A;
  ldb_val := M_val;

  // 2. Set up test data
  SetLength(hostA, lda_val * dim_A * batch_count_val);
  SetLength(hostB_initial, ldb_val * N_val * batch_count_val);
  SetLength(hostB_computed, ldb_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devB_array, batch_count_val);

  // 3. Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out matrices first
    for i := 0 to lda_val * dim_A - 1 do
    begin
      hostA[b * lda_val * dim_A + i].x := 0.0;
      hostA[b * lda_val * dim_A + i].y := 0.0;
    end;
    for i := 0 to ldb_val * N_val - 1 do
    begin
      hostB_initial[b * ldb_val * N_val + i].x := 0.0;
      hostB_initial[b * ldb_val * N_val + i].y := 0.0;
    end;

    // Fill triangular matrix A (Upper triangle, non-singular)
    // A = [ 2.0+0.0i  1.0+1.0i ]
    //     [ 0.0+0.0i  3.0+0.0i ]
    if uplo = rocblas_fill_upper then
    begin
      for i := 0 to dim_A - 1 do
        for j := i to dim_A - 1 do
        begin
          hostA[b * lda_val * dim_A + j * lda_val + i].x := (i + j + 1 + b) * 0.4 + 1.0;
          hostA[b * lda_val * dim_A + j * lda_val + i].y := (i - j + b) * 0.1;
        end;
      // Lower triangle remains zero
    end;

    // Fill matrix B initial (RHS)
    for i := 0 to M_val - 1 do
      for j := 0 to N_val - 1 do
      begin
        hostB_initial[b * ldb_val * N_val + j * ldb_val + i].x := (i + j + 1 + b) * 0.2;
        hostB_initial[b * ldb_val * N_val + j * ldb_val + i].y := (i - j + b) * 0.05;
      end;

    // Copy initial B matrix for computation (RHS -> Solution)
    Move(hostB_initial[b * ldb_val * N_val], hostB_computed[b * ldb_val * N_val], SizeOf(rocblas_float_complex) * ldb_val * N_val);
  end;

  // 4. Allocate device memory
  devA := nil;
  devB := nil;
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(rocblas_float_complex) * lda_val * dim_A * batch_count_val));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, SizeOf(rocblas_float_complex) * ldb_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devB should not be nil', devB <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(rocblas_float_complex) * lda_val * dim_A * batch_count_val, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB_computed[0], SizeOf(rocblas_float_complex) * ldb_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  // 6. Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    devA_array[b] := Procfloat_complex(PtrUInt(devA) + b * lda_val * dim_A * SizeOf(rocblas_float_complex));
    devB_array[b] := Procfloat_complex(PtrUInt(devB) + b * ldb_val * N_val * SizeOf(rocblas_float_complex));
  end;

  // 7. Allocate and copy device pointer arrays
  devA_array_device := nil;
  devB_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devB_array_device should not be nil', devB_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device H2D', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // 8. Set rocBLAS pointer mode (alpha is complex scalar on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // 9. Call rocblas_ctrsm_batched (12 parameters - CORRECT for TRSM_Batched - in-place solve)
  blas_status := rocblas_ctrsm_batched(
    handle,
    side,
    uplo,
    transA,
    diag,
    M_val, N_val,
    @alpha,   // alpha (complex, host pointer)
    PPProcfloat_complex(devA_array_device), // A array (array of device pointers to triangular matrices)
    lda_val, // lda
    PPProcfloat_complex(devB_array_device), // B array (array of device pointers to RHS matrices, overwritten by solution X)
    ldb_val, // ldb
    batch_count_val // batch_count
  );
  AssertEquals('rocblas_ctrsm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // 10. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ctrsm_batched', HIP_SUCCESS, sync_status);

  // 11. Copy result D2H (Solution X overwrites hostB_computed)
  AssertEquals('hipMemcpy devB D2H', HIP_SUCCESS, hipMemcpy(@hostB_computed[0], devB, SizeOf(rocblas_float_complex) * ldb_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  // 12. Basic check (check if the solution matrix differs from the initial RHS)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('Solution matrix B should differ from initial RHS matrix B for batch ' + IntToStr(b),
      not CompareMem(@hostB_initial[b * ldb_val * N_val], @hostB_computed[b * ldb_val * N_val], SizeOf(rocblas_float_complex) * ldb_val * N_val));
  end;

  // 13. Free device memory
  hipFree(devA);
  hipFree(devB);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
end;


// --- ZTRSM_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_ZTRSM_Batched;
const
  M_val = 3;
  N_val = 3;
  batch_count_val = 2; // Different dimensions
var
  side: rocblas_side;
  uplo: rocblas_fill;
  transA: rocblas_operation;
  diag: rocblas_diagonal;
  alpha: rocblas_double_complex;
  dim_A: Integer;
  lda_val, ldb_val: Integer;
  devA, devB: hipDeviceptr_t;
  hostA, hostB_initial, hostB_computed: array of rocblas_double_complex;
  devA_array, devB_array: array of Procdouble_complex;
  devA_array_device, devB_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  // 1. Initialize matrix dimensions and parameters
  side := rocblas_side_right; // Different side
  uplo := rocblas_fill_lower; // Different uplo
  transA := rocblas_operation_conjugate_transpose; // Different transA
  diag := rocblas_diagonal_unit; // Different diag
  alpha.x := 0.5;
  alpha.y := 0.0;

  if side = rocblas_side_left then
    dim_A := M_val
  else
    dim_A := N_val; // For side=right, dim_A = N

  lda_val := dim_A;
  ldb_val := M_val;

  // 2. Set up test data
  SetLength(hostA, lda_val * dim_A * batch_count_val);
  SetLength(hostB_initial, ldb_val * N_val * batch_count_val);
  SetLength(hostB_computed, ldb_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devB_array, batch_count_val);

  // 3. Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out matrices first
    for i := 0 to lda_val * dim_A - 1 do
    begin
      hostA[b * lda_val * dim_A + i].x := 0.0;
      hostA[b * lda_val * dim_A + i].y := 0.0;
    end;
    for i := 0 to ldb_val * N_val - 1 do
    begin
      hostB_initial[b * ldb_val * N_val + i].x := 0.0;
      hostB_initial[b * ldb_val * N_val + i].y := 0.0;
    end;

    // Fill triangular matrix A (Lower triangle, unit diagonal)
    // A = [ 1.0+0.0i  0.0+0.0i ] (Upper part ignored)
    //     [ 1.0+0.5i  1.0+0.0i ] (Lower triangle, diag implicitly 1)
    if uplo = rocblas_fill_lower then
    begin
      for i := 0 to dim_A - 1 do
        for j := 0 to i do
        begin
          if (diag = rocblas_diagonal_unit) and (i = j) then
          begin
            hostA[b * lda_val * dim_A + j * lda_val + i].x := 1.0; // Explicit unit diag for clarity
            hostA[b * lda_val * dim_A + j * lda_val + i].y := 0.0;
          end
          else
          begin
            hostA[b * lda_val * dim_A + j * lda_val + i].x := (i + j + 1 + b) * 0.6 + 0.2;
            hostA[b * lda_val * dim_A + j * lda_val + i].y := (i - j + b) * -0.15;
          end;
        end;
      // Upper triangle remains zero
    end;

    // Fill matrix B initial (RHS)
    for i := 0 to M_val - 1 do
      for j := 0 to N_val - 1 do
      begin
        hostB_initial[b * ldb_val * N_val + j * ldb_val + i].x := (i + j + 1 + b) * 0.25;
        hostB_initial[b * ldb_val * N_val + j * ldb_val + i].y := -(i - j + b) * 0.08;
      end;

    // Copy initial B matrix for computation (RHS -> Solution)
    Move(hostB_initial[b * ldb_val * N_val], hostB_computed[b * ldb_val * N_val], SizeOf(rocblas_double_complex) * ldb_val * N_val);
  end;

  // 4. Allocate device memory
  devA := nil;
  devB := nil;
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(rocblas_double_complex) * lda_val * dim_A * batch_count_val));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, SizeOf(rocblas_double_complex) * ldb_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devB should not be nil', devB <> nil);

  // 5. Copy data H2D
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(rocblas_double_complex) * lda_val * dim_A * batch_count_val, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB_computed[0], SizeOf(rocblas_double_complex) * ldb_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  // 6. Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + b * lda_val * dim_A * SizeOf(rocblas_double_complex));
    devB_array[b] := Procdouble_complex(PtrUInt(devB) + b * ldb_val * N_val * SizeOf(rocblas_double_complex));
  end;

  // 7. Allocate and copy device pointer arrays
  devA_array_device := nil;
  devB_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devB_array_device should not be nil', devB_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device H2D', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // 8. Set rocBLAS pointer mode (alpha is complex scalar on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // 9. Call rocblas_ztrsm_batched (12 parameters - CORRECT for TRSM_Batched - in-place solve)
  blas_status := rocblas_ztrsm_batched(
    handle,
    side,
    uplo,
    transA,
    diag,
    M_val, N_val,
    @alpha,   // alpha (complex, host pointer)
    PPProcdouble_complex(devA_array_device), // A array (array of device pointers to triangular matrices)
    lda_val, // lda
    PPProcdouble_complex(devB_array_device), // B array (array of device pointers to RHS matrices, overwritten by solution X)
    ldb_val, // ldb
    batch_count_val // batch_count
  );
  AssertEquals('rocblas_ztrsm_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // 10. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ztrsm_batched', HIP_SUCCESS, sync_status);

  // 11. Copy result D2H (Solution X overwrites hostB_computed)
  AssertEquals('hipMemcpy devB D2H', HIP_SUCCESS, hipMemcpy(@hostB_computed[0], devB, SizeOf(rocblas_double_complex) * ldb_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  // 12. Basic check (check if the solution matrix differs from the initial RHS)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('Solution matrix B should differ from initial RHS matrix B for batch ' + IntToStr(b),
      not CompareMem(@hostB_initial[b * ldb_val * N_val], @hostB_computed[b * ldb_val * N_val], SizeOf(rocblas_double_complex) * ldb_val * N_val));
  end;

  hipFree(devA);
  hipFree(devB);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
end;


// --- SSYRK_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_SSYRK_Batched;
const
  N_val = 3;
  K_val = 2;
  batch_count_val = 2;
var
  uplo: rocblas_fill;
  transA: rocblas_operation;
  alpha, beta: Single; // Real Single Precision Scalars
  lda_val, ldc_val: Integer;
  devA, devC: hipDeviceptr_t;
  hostA, hostC_initial, hostC_computed: array of Single; // Real Single Precision
  devA_array, devC_array: array of PSingle;
  devA_array_device, devC_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  // 1. Initialize matrix dimensions and parameters
  uplo := rocblas_fill_upper; // Update upper triangle of C
  transA := rocblas_operation_none; // Operation: C = alpha*A*A^T + beta*C
  // Calculate leading dimensions
  if transA = rocblas_operation_none then
    lda_val := N_val // A is N x K
  else
    lda_val := K_val; // A is K x N
  ldc_val := N_val; // C is N x N

  alpha := 1.0;
  beta := 1.0;

  // 2. Set up test data sizes
  if transA = rocblas_operation_none then
  begin
    SetLength(hostA, lda_val * K_val * batch_count_val); // A is N x K
  end
  else
  begin
    SetLength(hostA, lda_val * N_val * batch_count_val); // A is K x N
  end;
  SetLength(hostC_initial, ldc_val * N_val * batch_count_val); // C is N x N
  SetLength(hostC_computed, ldc_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devC_array, batch_count_val);

  // 3. Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out A first
    if transA = rocblas_operation_none then
    begin
      for i := 0 to N_val * K_val - 1 do
        hostA[b * N_val * K_val + i] := 0.0;
    end
    else
    begin
      for i := 0 to K_val * N_val - 1 do
        hostA[b * K_val * N_val + i] := 0.0;
    end;

    // Zero out C
    for i := 0 to N_val * N_val - 1 do
    begin
      hostC_initial[b * N_val * N_val + i] := 0.0;
      hostC_computed[b * N_val * N_val + i] := 0.0;
    end;

    // Fill matrix A with test data
    // Example: A (3x2) = [ 1.0  2.0 ]
    //                   [ 3.0  4.0 ]
    //                   [ 5.0  6.0 ]
    if transA = rocblas_operation_none then
    begin
      for i := 0 to N_val - 1 do
        for j := 0 to K_val - 1 do
          hostA[b * N_val * K_val + j * N_val + i] := (i * K_val + j + 1 + b) * 0.5; // Column-major for A matrix
    end;

    // Fill initial symmetric matrix C
    // Example: C (3x3) = [ 2.0  1.0  3.0 ] (Upper triangle)
    //                    [ 1.0  3.0  2.0 ]
    //                    [ 3.0  2.0  1.0 ] (Lower triangle, copy of upper)
    for i := 0 to N_val - 1 do
      for j := 0 to N_val - 1 do
        hostC_initial[b * N_val * N_val + j * N_val + i] := (i + j + 1 + b) * 0.2; // Column-major for C matrix

    // Copy initial C matrix for computation
    Move(hostC_initial[b * N_val * N_val], hostC_computed[b * N_val * N_val], SizeOf(Single) * N_val * N_val);
  end;

  // 4. Allocate device memory
  devA := nil;
  devC := nil;
  if transA = rocblas_operation_none then
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(Single) * N_val * K_val * batch_count_val))
  else
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(Single) * K_val * N_val * batch_count_val));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, SizeOf(Single) * ldc_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devC should not be nil', devC <> nil);

  // 5. Copy data H2D
  if transA = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(Single) * N_val * K_val * batch_count_val, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(Single) * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy devC H2D', HIP_SUCCESS, hipMemcpy(devC, @hostC_computed[0], SizeOf(Single) * ldc_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  // 6. Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    if transA = rocblas_operation_none then
    begin
      devA_array[b] := PSingle(PtrUInt(devA) + b * N_val * K_val * SizeOf(Single));
    end
    else
    begin
      devA_array[b] := PSingle(PtrUInt(devA) + b * K_val * N_val * SizeOf(Single));
    end;
    devC_array[b] := PSingle(PtrUInt(devC) + b * ldc_val * N_val * SizeOf(Single));
  end;

  // 7. Allocate and copy device pointer arrays
  devA_array_device := nil;
  devC_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devC_array_device should not be nil', devC_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device H2D', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // 8. Set rocBLAS pointer mode (alpha, beta are real scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // 9. Call rocblas_ssyrk_batched (12 parameters)
  if transA = rocblas_operation_none then
  begin
    blas_status := rocblas_ssyrk_batched(
      handle,
      uplo,
      transA,
      N_val, K_val,
      @alpha,   // alpha (real, host pointer)
      PPSingle(devA_array_device), // A array (array of device pointers)
      lda_val, // lda
      @beta,    // beta (real, host pointer)
      PPSingle(devC_array_device), // C array (array of device pointers, symmetric matrix, output)
      ldc_val, // ldc
      batch_count_val // batch_count
    );
  end
  else // transA = rocblas_operation_transpose
  begin
    blas_status := rocblas_ssyrk_batched(
      handle,
      uplo,
      transA,
      N_val, K_val,
      @alpha,   // alpha (real, host pointer)
      PPSingle(devA_array_device), // A array (array of device pointers)
      lda_val, // lda
      @beta,    // beta (real, host pointer)
      PPSingle(devC_array_device), // C array (array of device pointers, symmetric matrix, output)
      ldc_val, // ldc
      batch_count_val // batch_count
    );
  end;
  AssertEquals('rocblas_ssyrk_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // 10. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ssyrk_batched', HIP_SUCCESS, sync_status);

  // 11. Copy result D2H
  AssertEquals('hipMemcpy devC D2H', HIP_SUCCESS, hipMemcpy(@hostC_computed[0], devC, SizeOf(Single) * ldc_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  // 12. Basic check (check if the matrix differs)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('Result matrix C should differ from initial matrix C for batch ' + IntToStr(b),
      not CompareMem(@hostC_initial[b * ldc_val * N_val], @hostC_computed[b * ldc_val * N_val], SizeOf(Single) * ldc_val * N_val));
  end;

  // 13. Free device memory
  hipFree(devA);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devC_array_device);
end;


// --- DSYRK_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_DSYRK_Batched;
const
  N_val = 3;
  K_val = 2;
  batch_count_val = 3; // Different dimensions
var
  uplo: rocblas_fill;
  transA: rocblas_operation;
  alpha, beta: Double; // Real Double Precision Scalars
  lda_val, ldc_val: Integer;
  devA, devC: hipDeviceptr_t;
  hostA, hostC_initial, hostC_computed: array of Double; // Real Double Precision
  devA_array, devC_array: array of PDouble;
  devA_array_device, devC_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  // 1. Initialize matrix dimensions and parameters
  uplo := rocblas_fill_lower; // Update lower triangle of C
  transA := rocblas_operation_transpose; // Operation: C = alpha*A^T*A + beta*C
  // Calculate leading dimensions
  if transA = rocblas_operation_none then
    lda_val := N_val
  else
    lda_val := K_val;
  ldc_val := N_val; // C is N x N

  alpha := 0.5;
  beta := 2.0;

  // 2. Set up test data sizes
  if transA = rocblas_operation_none then
  begin
    SetLength(hostA, lda_val * K_val * batch_count_val); // A is N x K
  end
  else
  begin
    SetLength(hostA, lda_val * N_val * batch_count_val); // A is K x N
  end;
  SetLength(hostC_initial, ldc_val * N_val * batch_count_val); // C is N x N
  SetLength(hostC_computed, ldc_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devC_array, batch_count_val);

  // 3. Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out A first
    if transA = rocblas_operation_none then
    begin
      for i := 0 to N_val * K_val - 1 do
        hostA[b * N_val * K_val + i] := 0.0;
    end
    else
    begin
      for i := 0 to K_val * N_val - 1 do
        hostA[b * K_val * N_val + i] := 0.0;
    end;

    // Zero out C
    for i := 0 to N_val * N_val - 1 do
    begin
      hostC_initial[b * N_val * N_val + i] := 0.0;
      hostC_computed[b * N_val * N_val + i] := 0.0;
    end;

    // Fill matrix A with test data
    // Example: A (2x3) = [ 1.0  3.0  5.0 ]
    //                   [ 2.0  4.0  6.0 ]
    if transA = rocblas_operation_transpose then
    begin
      for i := 0 to K_val - 1 do
        for j := 0 to N_val - 1 do
          hostA[b * K_val * N_val + j * K_val + i] := (i * N_val + j + 1 + b) * 0.7; // Column-major for A matrix
    end;

    // Fill initial symmetric matrix C
    for i := 0 to N_val - 1 do
      for j := 0 to N_val - 1 do
        hostC_initial[b * N_val * N_val + j * N_val + i] := (i + j + 1 + b) * 0.3; // Column-major for C matrix

    // Copy initial C matrix for computation
    Move(hostC_initial[b * N_val * N_val], hostC_computed[b * N_val * N_val], SizeOf(Double) * N_val * N_val);
  end;

  // 4. Allocate device memory
  devA := nil;
  devC := nil;
  if transA = rocblas_operation_none then
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(Double) * N_val * K_val * batch_count_val))
  else
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(Double) * K_val * N_val * batch_count_val));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, SizeOf(Double) * ldc_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devC should not be nil', devC <> nil);

  // 5. Copy data H2D
  if transA = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(Double) * N_val * K_val * batch_count_val, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(Double) * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy devC H2D', HIP_SUCCESS, hipMemcpy(devC, @hostC_computed[0], SizeOf(Double) * ldc_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  // 6. Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    if transA = rocblas_operation_none then
    begin
      devA_array[b] := PDouble(PtrUInt(devA) + b * N_val * K_val * SizeOf(Double));
    end
    else
    begin
      devA_array[b] := PDouble(PtrUInt(devA) + b * K_val * N_val * SizeOf(Double));
    end;
    devC_array[b] := PDouble(PtrUInt(devC) + b * ldc_val * N_val * SizeOf(Double));
  end;

  // 7. Allocate and copy device pointer arrays
  devA_array_device := nil;
  devC_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devC_array_device should not be nil', devC_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device H2D', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // 8. Set rocBLAS pointer mode (alpha, beta are real scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // 9. Call rocblas_dsyrk_batched (12 parameters)
  if transA = rocblas_operation_none then
  begin
    blas_status := rocblas_dsyrk_batched(
      handle,
      uplo,
      transA,
      N_val, K_val,
      @alpha,   // alpha (real, host pointer)
      PPDouble(devA_array_device), // A array (array of device pointers)
      lda_val, // lda
      @beta,    // beta (real, host pointer)
      PPDouble(devC_array_device), // C array (array of device pointers, symmetric matrix, output)
      ldc_val, // ldc
      batch_count_val // batch_count
    );
  end
  else // transA = rocblas_operation_transpose
  begin
    blas_status := rocblas_dsyrk_batched(
      handle,
      uplo,
      transA,
      N_val, K_val,
      @alpha,   // alpha (real, host pointer)
      PPDouble(devA_array_device), // A array (array of device pointers)
      lda_val, // lda
      @beta,    // beta (real, host pointer)
      PPDouble(devC_array_device), // C array (array of device pointers, symmetric matrix, output)
      ldc_val, // ldc
      batch_count_val // batch_count
    );
  end;
  AssertEquals('rocblas_dsyrk_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // 10. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_dsyrk_batched', HIP_SUCCESS, sync_status);

  // 11. Copy result D2H
  AssertEquals('hipMemcpy devC D2H', HIP_SUCCESS, hipMemcpy(@hostC_computed[0], devC, SizeOf(Double) * ldc_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  // 12. Basic check (check if the matrix differs)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('Result matrix C should differ from initial matrix C for batch ' + IntToStr(b),
      not CompareMem(@hostC_initial[b * ldc_val * N_val], @hostC_computed[b * ldc_val * N_val], SizeOf(Double) * ldc_val * N_val));
  end;

  // 13. Free device memory
  hipFree(devA);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devC_array_device);
end;


// --- CSYRK_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_CSYRK_Batched;
const
  N_val = 3;
  K_val = 2;
  batch_count_val = 1; // Smaller batch for complex
var
  uplo: rocblas_fill;
  transA: rocblas_operation;
  alpha, beta: rocblas_float_complex; // Complex Scalars
  lda_val, ldc_val: Integer;
  devA, devC: hipDeviceptr_t;
  hostA, hostC_initial, hostC_computed: array of rocblas_float_complex;
  devA_array, devC_array: array of Procfloat_complex;
  devA_array_device, devC_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  // 1. Initialize matrix dimensions and parameters
  uplo := rocblas_fill_upper; // Update upper triangle of C
  transA := rocblas_operation_none; // Operation: C = alpha*A*A^T + beta*C
  // Calculate leading dimensions
  if transA = rocblas_operation_none then
    lda_val := N_val // A is N x K
  else
    lda_val := K_val; // A is K x N
  ldc_val := N_val; // C is N x N

  alpha.x := 1.0;
  alpha.y := 0.0;
  beta.x := 1.0;
  beta.y := 0.0;

  // 2. Set up test data sizes
  if transA = rocblas_operation_none then
  begin
    SetLength(hostA, lda_val * K_val * batch_count_val); // A is N x K
  end
  else
  begin
    SetLength(hostA, lda_val * N_val * batch_count_val); // A is K x N
  end;
  SetLength(hostC_initial, ldc_val * N_val * batch_count_val); // C is N x N
  SetLength(hostC_computed, ldc_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devC_array, batch_count_val);

  // 3. Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out A first
    if transA = rocblas_operation_none then
    begin
      for i := 0 to N_val * K_val - 1 do
      begin
        hostA[b * N_val * K_val + i].x := 0.0;
        hostA[b * N_val * K_val + i].y := 0.0;
      end;
    end
    else
    begin
      for i := 0 to K_val * N_val - 1 do
      begin
        hostA[b * K_val * N_val + i].x := 0.0;
        hostA[b * K_val * N_val + i].y := 0.0;
      end;
    end;

    // Zero out C
    for i := 0 to N_val * N_val - 1 do
    begin
      hostC_initial[b * N_val * N_val + i].x := 0.0;
      hostC_initial[b * N_val * N_val + i].y := 0.0;
      hostC_computed[b * N_val * N_val + i].x := 0.0;
      hostC_computed[b * N_val * N_val + i].y := 0.0;
    end;

    // Fill matrix A with test data
    // Example: A (3x2) = [ 1+1i  2+0i ]
    //                   [ 3+0i  4+1i ]
    //                   [ 5+0i  6+0i ]
    if transA = rocblas_operation_none then
    begin
      for i := 0 to N_val - 1 do
        for j := 0 to K_val - 1 do
        begin
          hostA[b * N_val * K_val + j * N_val + i].x := (i * K_val + j + 1 + b) * 0.4; // Real part
          hostA[b * N_val * K_val + j * N_val + i].y := (i - j + b) * 0.1; // Imaginary part
        end;
    end;

    // Fill initial symmetric matrix C
    // Example: C (3x3) = [ 2+0i  1+1i  3+0i ] (Upper triangle)
    //                    [ 1-1i  3+0i  2+1i ] (Lower triangle would be conj-transpose for Hermitian,
    //                                          but for SYRK it's just symmetric, so copy upper)
    //                    [ 3+0i  2-1i  1+0i ]
    // For SYMMETRIC (not Hermitian), C[i][j] = C[j][i] (no conjugate).
    for i := 0 to N_val - 1 do
      for j := 0 to N_val - 1 do
      begin
        hostC_initial[b * N_val * N_val + j * N_val + i].x := (i + j + 1 + b) * 0.2;
        hostC_initial[b * N_val * N_val + j * N_val + i].y := (i - j + b) * 0.05;
      end;

    // Copy initial C matrix for computation
    Move(hostC_initial[b * N_val * N_val], hostC_computed[b * N_val * N_val], SizeOf(rocblas_float_complex) * N_val * N_val);
  end;

  // 4. Allocate device memory
  devA := nil;
  devC := nil;
  if transA = rocblas_operation_none then
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(rocblas_float_complex) * N_val * K_val * batch_count_val))
  else
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(rocblas_float_complex) * K_val * N_val * batch_count_val));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, SizeOf(rocblas_float_complex) * ldc_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devC should not be nil', devC <> nil);

  // 5. Copy data H2D
  if transA = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(rocblas_float_complex) * N_val * K_val * batch_count_val, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(rocblas_float_complex) * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy devC H2D', HIP_SUCCESS, hipMemcpy(devC, @hostC_computed[0], SizeOf(rocblas_float_complex) * ldc_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  // 6. Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    if transA = rocblas_operation_none then
    begin
      devA_array[b] := Procfloat_complex(PtrUInt(devA) + b * N_val * K_val * SizeOf(rocblas_float_complex));
    end
    else
    begin
      devA_array[b] := Procfloat_complex(PtrUInt(devA) + b * K_val * N_val * SizeOf(rocblas_float_complex));
    end;
    devC_array[b] := Procfloat_complex(PtrUInt(devC) + b * ldc_val * N_val * SizeOf(rocblas_float_complex));
  end;

  // 7. Allocate and copy device pointer arrays
  devA_array_device := nil;
  devC_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devC_array_device should not be nil', devC_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device H2D', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // 8. Set rocBLAS pointer mode (alpha, beta are complex scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // 9. Call rocblas_csyrk_batched (12 parameters)
  if transA = rocblas_operation_none then
  begin
    blas_status := rocblas_csyrk_batched(
      handle,
      uplo,
      transA,
      N_val, K_val,
      @alpha,   // alpha (complex, host pointer)
      PPProcfloat_complex(devA_array_device), // A array (array of device pointers)
      lda_val, // lda
      @beta,    // beta (complex, host pointer)
      PPProcfloat_complex(devC_array_device), // C array (array of device pointers, symmetric matrix, output)
      ldc_val, // ldc
      batch_count_val // batch_count
    );
  end
  else // transA = rocblas_operation_transpose
  begin
    blas_status := rocblas_csyrk_batched(
      handle,
      uplo,
      transA,
      N_val, K_val,
      @alpha,   // alpha (complex, host pointer)
      PPProcfloat_complex(devA_array_device), // A array (array of device pointers)
      lda_val, // lda
      @beta,    // beta (complex, host pointer)
      PPProcfloat_complex(devC_array_device), // C array (array of device pointers, symmetric matrix, output)
      ldc_val, // ldc
      batch_count_val // batch_count
    );
  end;
  AssertEquals('rocblas_csyrk_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // 10. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_csyrk_batched', HIP_SUCCESS, sync_status);

  // 11. Copy result D2H
  AssertEquals('hipMemcpy devC D2H', HIP_SUCCESS, hipMemcpy(@hostC_computed[0], devC, SizeOf(rocblas_float_complex) * ldc_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  // 12. Basic check (check if the matrix differs)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('Result matrix C should differ from initial matrix C for batch ' + IntToStr(b),
      not CompareMem(@hostC_initial[b * ldc_val * N_val], @hostC_computed[b * ldc_val * N_val], SizeOf(rocblas_float_complex) * ldc_val * N_val));
  end;

  // 13. Free device memory
  hipFree(devA);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devC_array_device);
end;


// --- ZSYRK_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_ZSYRK_Batched;
const
  N_val = 3;
  K_val = 2;
  batch_count_val = 2; // Different dimensions
var
  uplo: rocblas_fill;
  transA: rocblas_operation;
  alpha, beta: rocblas_double_complex; // Complex Scalars
  lda_val, ldc_val: Integer;
  devA, devC: hipDeviceptr_t;
  hostA, hostC_initial, hostC_computed: array of rocblas_double_complex;
  devA_array, devC_array: array of Procdouble_complex;
  devA_array_device, devC_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  // 1. Initialize matrix dimensions and parameters
  uplo := rocblas_fill_lower; // Update lower triangle of C
  transA := rocblas_operation_transpose; // Operation: C = alpha*A^T*A + beta*C
  // Calculate leading dimensions
  if transA = rocblas_operation_none then
    lda_val := N_val
  else
    lda_val := K_val;
  ldc_val := N_val; // C is N x N

  alpha.x := 0.5;
  alpha.y := 0.1;
  beta.x := 2.0;
  beta.y := -0.3;

  // 2. Set up test data sizes
  if transA = rocblas_operation_none then
  begin
    SetLength(hostA, lda_val * K_val * batch_count_val); // A is N x K
  end
  else
  begin
    SetLength(hostA, lda_val * N_val * batch_count_val); // A is K x N
  end;
  SetLength(hostC_initial, ldc_val * N_val * batch_count_val); // C is N x N
  SetLength(hostC_computed, ldc_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devC_array, batch_count_val);

  // 3. Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out A first
    if transA = rocblas_operation_none then
    begin
      for i := 0 to N_val * K_val - 1 do
      begin
        hostA[b * N_val * K_val + i].x := 0.0;
        hostA[b * N_val * K_val + i].y := 0.0;
      end;
    end
    else
    begin
      for i := 0 to K_val * N_val - 1 do
      begin
        hostA[b * K_val * N_val + i].x := 0.0;
        hostA[b * K_val * N_val + i].y := 0.0;
      end;
    end;

    // Zero out C
    for i := 0 to N_val * N_val - 1 do
    begin
      hostC_initial[b * N_val * N_val + i].x := 0.0;
      hostC_initial[b * N_val * N_val + i].y := 0.0;
      hostC_computed[b * N_val * N_val + i].x := 0.0;
      hostC_computed[b * N_val * N_val + i].y := 0.0;
    end;

    // Fill matrix A with test data
    // Example: A (2x3) = [ 1+1i  3+0i  5+0i ]
    //                   [ 2+0i  4+1i  6+0i ]
    if transA = rocblas_operation_transpose then
    begin
      for i := 0 to K_val - 1 do
        for j := 0 to N_val - 1 do
        begin
          hostA[b * K_val * N_val + j * K_val + i].x := (i * N_val + j + 1 + b) * 0.6; // Real part
          hostA[b * K_val * N_val + j * K_val + i].y := (i - j + b) * -0.15; // Imaginary part
        end;
    end;

    // Fill initial symmetric matrix C
    for i := 0 to N_val - 1 do
      for j := 0 to N_val - 1 do
      begin
        hostC_initial[b * N_val * N_val + j * N_val + i].x := (i + j + 1 + b) * 0.25;
        hostC_initial[b * N_val * N_val + j * N_val + i].y := (i - j + b) * 0.08;
      end;

    // Copy initial C matrix for computation
    Move(hostC_initial[b * N_val * N_val], hostC_computed[b * N_val * N_val], SizeOf(rocblas_double_complex) * N_val * N_val);
  end;

  // 4. Allocate device memory
  devA := nil;
  devC := nil;
  if transA = rocblas_operation_none then
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(rocblas_double_complex) * N_val * K_val * batch_count_val))
  else
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(rocblas_double_complex) * K_val * N_val * batch_count_val));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, SizeOf(rocblas_double_complex) * ldc_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devC should not be nil', devC <> nil);

  // 5. Copy data H2D
  if transA = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(rocblas_double_complex) * N_val * K_val * batch_count_val, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(rocblas_double_complex) * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy devC H2D', HIP_SUCCESS, hipMemcpy(devC, @hostC_computed[0], SizeOf(rocblas_double_complex) * ldc_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  // 6. Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    if transA = rocblas_operation_none then
    begin
      devA_array[b] := Procdouble_complex(PtrUInt(devA) + b * N_val * K_val * SizeOf(rocblas_double_complex));
    end
    else
    begin
      devA_array[b] := Procdouble_complex(PtrUInt(devA) + b * K_val * N_val * SizeOf(rocblas_double_complex));
    end;
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + b * ldc_val * N_val * SizeOf(rocblas_double_complex));
  end;

  // 7. Allocate and copy device pointer arrays
  devA_array_device := nil;
  devC_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devC_array_device should not be nil', devC_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device H2D', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // 8. Set rocBLAS pointer mode (alpha, beta are complex scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // 9. Call rocblas_zsyrk_batched (12 parameters)
  if transA = rocblas_operation_none then
  begin
    blas_status := rocblas_zsyrk_batched(
      handle,
      uplo,
      transA,
      N_val, K_val,
      @alpha,   // alpha (complex, host pointer)
      PPProcdouble_complex(devA_array_device), // A array (array of device pointers)
      lda_val, // lda
      @beta,    // beta (complex, host pointer)
      PPProcdouble_complex(devC_array_device), // C array (array of device pointers, symmetric matrix, output)
      ldc_val, // ldc
      batch_count_val // batch_count
    );
  end
  else // transA = rocblas_operation_transpose
  begin
    blas_status := rocblas_zsyrk_batched(
      handle,
      uplo,
      transA,
      N_val, K_val,
      @alpha,   // alpha (complex, host pointer)
      PPProcdouble_complex(devA_array_device), // A array (array of device pointers)
      lda_val, // lda
      @beta,    // beta (complex, host pointer)
      PPProcdouble_complex(devC_array_device), // C array (array of device pointers, symmetric matrix, output)
      ldc_val, // ldc
      batch_count_val // batch_count
    );
  end;
  AssertEquals('rocblas_zsyrk_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // 10. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_zsyrk_batched', HIP_SUCCESS, sync_status);

  // 11. Copy result D2H
  AssertEquals('hipMemcpy devC D2H', HIP_SUCCESS, hipMemcpy(@hostC_computed[0], devC, SizeOf(rocblas_double_complex) * ldc_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  // 12. Basic check (check if the matrix differs)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('Result matrix C should differ from initial matrix C for batch ' + IntToStr(b),
      not CompareMem(@hostC_initial[b * ldc_val * N_val], @hostC_computed[b * ldc_val * N_val], SizeOf(rocblas_double_complex) * ldc_val * N_val));
  end;

  // 13. Free device memory
  hipFree(devA);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devC_array_device);
end;


// --- CHERK_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_CHERK_Batched;
const
  N_val = 3;
  K_val = 2;
  batch_count_val = 2;
var
  uplo: rocblas_fill;
  transA: rocblas_operation;
  alpha: Single; // alpha is real for HERK
  beta: Single;  // beta is real for HERK
  lda_val, ldc_val: Integer;
  devA, devC: hipDeviceptr_t;
  hostA, hostC_initial, hostC_computed: array of rocblas_float_complex;
  devA_array, devC_array: array of Procfloat_complex;
  devA_array_device, devC_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  uplo := rocblas_fill_upper; // Update upper triangle of C
  transA := rocblas_operation_none; // Operation: C = alpha * A * A^H + beta * C
  // Calculate leading dimensions
  if transA = rocblas_operation_none then
    lda_val := N_val // A is N x K
  else
    lda_val := K_val; // A is K x N
  ldc_val := N_val; // C is N x N (Hermitian)

  alpha := 1.0; // Real scalar
  beta := 1.0;  // Real scalar

  // 2. Set up test data sizes
  if transA = rocblas_operation_none then
    SetLength(hostA, lda_val * K_val * batch_count_val) // A is N x K
  else
    SetLength(hostA, lda_val * N_val * batch_count_val); // A is K x N
  SetLength(hostC_initial, ldc_val * N_val * batch_count_val); // C is N x N
  SetLength(hostC_computed, ldc_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devC_array, batch_count_val);

  // 3. Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out A first
    for i := 0 to High(hostA) do
    begin
      hostA[i].x := 0.0;
      hostA[i].y := 0.0;
    end;
    // Zero out C
    for i := 0 to High(hostC_initial) do
    begin
      hostC_initial[i].x := 0.0;
      hostC_initial[i].y := 0.0;
    end;

    // Fill matrix A with test data
    // Example: A (3x2) = [ 1+1i  2+0i ]
    //                   [ 3+0i  4+1i ]
    //                   [ 5+0i  6+0i ]
    if transA = rocblas_operation_none then
    begin
      for i := 0 to N_val - 1 do
        for j := 0 to K_val - 1 do
        begin
          hostA[b * lda_val * K_val + j * lda_val + i].x := (i + j + 1 + b) * 0.5;
          hostA[b * lda_val * K_val + j * lda_val + i].y := (i - j + b) * 0.1;
        end;
    end;

    // Fill initial Hermitian matrix C
    // Example: C (3x3) = [ 2+0i  1+1i  3+0i ] (Upper triangle)
    //                    [ 1-1i  3+0i  2+1i ] (Lower triangle = conj-transpose of upper)
    //                    [ 3+0i  2-1i  1+0i ] (Diagonal is real)
    hostC_initial[b * ldc_val * N_val + 0 + 0 * ldc_val].x := 2.0; hostC_initial[b * ldc_val * N_val + 0 + 0 * ldc_val].y := 0.0; // C[0][0]
    hostC_initial[b * ldc_val * N_val + 1 + 1 * ldc_val].x := 3.0; hostC_initial[b * ldc_val * N_val + 1 + 1 * ldc_val].y := 0.0; // C[1][1]
    hostC_initial[b * ldc_val * N_val + 2 + 2 * ldc_val].x := 1.0; hostC_initial[b * ldc_val * N_val + 2 + 2 * ldc_val].y := 0.0; // C[2][2]

    if uplo = rocblas_fill_upper then
    begin
      hostC_initial[b * ldc_val * N_val + 0 + 1 * ldc_val].x := 1.0; hostC_initial[b * ldc_val * N_val + 0 + 1 * ldc_val].y := 1.0; // C[0][1]
      hostC_initial[b * ldc_val * N_val + 0 + 2 * ldc_val].x := 3.0; hostC_initial[b * ldc_val * N_val + 0 + 2 * ldc_val].y := 0.0; // C[0][2]
      hostC_initial[b * ldc_val * N_val + 1 + 2 * ldc_val].x := 2.0; hostC_initial[b * ldc_val * N_val + 1 + 2 * ldc_val].y := 1.0; // C[1][2]
      // Fill lower triangle based on upper (Hermitian property)
      hostC_initial[b * ldc_val * N_val + 1 + 0 * ldc_val].x := 1.0; hostC_initial[b * ldc_val * N_val + 1 + 0 * ldc_val].y := -1.0; // C[1][0] = conj(C[0][1])
      hostC_initial[b * ldc_val * N_val + 2 + 0 * ldc_val].x := 3.0; hostC_initial[b * ldc_val * N_val + 2 + 0 * ldc_val].y := 0.0;  // C[2][0] = conj(C[0][2])
      hostC_initial[b * ldc_val * N_val + 2 + 1 * ldc_val].x := 2.0; hostC_initial[b * ldc_val * N_val + 2 + 1 * ldc_val].y := -1.0; // C[2][1] = conj(C[1][2])
    end
    else // rocblas_fill_lower
    begin
      hostC_initial[b * ldc_val * N_val + 1 + 0 * ldc_val].x := 1.0; hostC_initial[b * ldc_val * N_val + 1 + 0 * ldc_val].y := -1.0; // C[1][0]
      hostC_initial[b * ldc_val * N_val + 2 + 0 * ldc_val].x := 3.0; hostC_initial[b * ldc_val * N_val + 2 + 0 * ldc_val].y := 0.0;  // C[2][0]
      hostC_initial[b * ldc_val * N_val + 2 + 1 * ldc_val].x := 2.0; hostC_initial[b * ldc_val * N_val + 2 + 1 * ldc_val].y := -1.0; // C[2][1]
      // Fill upper triangle based on lower (Hermitian property)
      hostC_initial[b * ldc_val * N_val + 0 + 1 * ldc_val].x := 1.0; hostC_initial[b * ldc_val * N_val + 0 + 1 * ldc_val].y := 1.0; // C[0][1] = conj(C[1][0])
      hostC_initial[b * ldc_val * N_val + 0 + 2 * ldc_val].x := 3.0; hostC_initial[b * ldc_val * N_val + 0 + 2 * ldc_val].y := 0.0;  // C[0][2] = conj(C[2][0])
      hostC_initial[b * ldc_val * N_val + 1 + 2 * ldc_val].x := 2.0; hostC_initial[b * ldc_val * N_val + 1 + 2 * ldc_val].y := 1.0;  // C[1][2] = conj(C[2][1])
    end;

    // Copy initial C matrix for computation
    Move(hostC_initial[b * ldc_val * N_val], hostC_computed[b * ldc_val * N_val], SizeOf(rocblas_float_complex) * ldc_val * N_val);
  end;

  // 4. Allocate device memory
  devA := nil;
  devC := nil;
  if transA = rocblas_operation_none then
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(rocblas_float_complex) * N_val * K_val * batch_count_val))
  else
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(rocblas_float_complex) * K_val * N_val * batch_count_val));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, SizeOf(rocblas_float_complex) * ldc_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devC should not be nil', devC <> nil);

  // 5. Copy data H2D
  if transA = rocblas_operation_none then
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(rocblas_float_complex) * N_val * K_val * batch_count_val, hipMemcpyHostToDevice))
  else
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(rocblas_float_complex) * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC H2D', HIP_SUCCESS, hipMemcpy(devC, @hostC_computed[0], SizeOf(rocblas_float_complex) * ldc_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  // 6. Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    if transA = rocblas_operation_none then
    begin
      devA_array[b] := Procfloat_complex(PtrUInt(devA) + b * N_val * K_val * SizeOf(rocblas_float_complex));
    end
    else
    begin
      devA_array[b] := Procfloat_complex(PtrUInt(devA) + b * K_val * N_val * SizeOf(rocblas_float_complex));
    end;
    devC_array[b] := Procfloat_complex(PtrUInt(devC) + b * ldc_val * N_val * SizeOf(rocblas_float_complex));
  end;

  // 7. Allocate and copy device pointer arrays
  devA_array_device := nil;
  devC_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devC_array_device should not be nil', devC_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device H2D', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // 8. Set rocBLAS pointer mode (alpha, beta are real scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // 9. Call rocblas_cherk_batched (12 parameters)
  if transA = rocblas_operation_none then
  begin
    blas_status := rocblas_cherk_batched(
      handle,
      uplo,
      transA,
      N_val, K_val,
      @alpha,   // alpha (real, host pointer)
      PPProcfloat_complex(devA_array_device), // A array (array of device pointers)
      lda_val, // lda
      @beta,    // beta (real, host pointer)
      PPProcfloat_complex(devC_array_device), // C array (array of device pointers, Hermitian matrix, output)
      ldc_val, // ldc
      batch_count_val // batch_count
    );
  end
  else // transA = rocblas_operation_transpose/conjugate_transpose
  begin
    blas_status := rocblas_cherk_batched(
      handle,
      uplo,
      transA,
      N_val, K_val,
      @alpha,   // alpha (real, host pointer)
      PPProcfloat_complex(devA_array_device), // A array (array of device pointers)
      lda_val, // lda
      @beta,    // beta (real, host pointer)
      PPProcfloat_complex(devC_array_device), // C array (array of device pointers, Hermitian matrix, output)
      ldc_val, // ldc
      batch_count_val // batch_count
    );
  end;
  AssertEquals('rocblas_cherk_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // 10. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_cherk_batched', HIP_SUCCESS, sync_status);

  // 11. Copy result D2H
  AssertEquals('hipMemcpy devC D2H', HIP_SUCCESS, hipMemcpy(@hostC_computed[0], devC, SizeOf(rocblas_float_complex) * ldc_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  // 12. Basic check (check if the matrix differs)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('Result matrix C should differ from initial matrix C for batch ' + IntToStr(b),
      not CompareMem(@hostC_initial[b * ldc_val * N_val], @hostC_computed[b * ldc_val * N_val], SizeOf(rocblas_float_complex) * ldc_val * N_val));
  end;

  // 13. Free device memory
  hipFree(devA);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devC_array_device);

end;


// --- ZHERK_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_ZHERK_Batched;
const
  N_val = 3;
  K_val = 2;
  batch_count_val = 3; // Different dimensions
var
  uplo: rocblas_fill;
  transA: rocblas_operation;
  alpha: Double; // alpha is real for HERK
  beta: Double;  // beta is real for HERK
  lda_val, ldc_val: Integer;
  devA, devC: hipDeviceptr_t;
  hostA, hostC_initial, hostC_computed: array of rocblas_double_complex;
  devA_array, devC_array: array of Procdouble_complex;
  devA_array_device, devC_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin

  uplo := rocblas_fill_lower; // Update lower triangle of C
  transA := rocblas_operation_conjugate_transpose; // Operation: C = alpha * A^H * A + beta * C
  // Calculate leading dimensions
  if transA = rocblas_operation_none then
    lda_val := N_val // A is N x K
  else
    lda_val := K_val; // A is K x N
  ldc_val := N_val; // C is N x N (Hermitian)

  alpha := 0.5; // Real scalar
  beta := 2.0;  // Real scalar

  // 2. Set up test data sizes
  if transA = rocblas_operation_none then
    SetLength(hostA, lda_val * K_val * batch_count_val) // A is N x K
  else
    SetLength(hostA, lda_val * N_val * batch_count_val); // A is K x N
  SetLength(hostC_initial, ldc_val * N_val * batch_count_val); // C is N x N
  SetLength(hostC_computed, ldc_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devC_array, batch_count_val);

  // 3. Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out A first
    for i := 0 to High(hostA) do
    begin
      hostA[i].x := 0.0;
      hostA[i].y := 0.0;
    end;
    // Zero out C
    for i := 0 to High(hostC_initial) do
    begin
      hostC_initial[i].x := 0.0;
      hostC_initial[i].y := 0.0;
    end;

    // Fill matrix A with test data
    // Example: A (2x3) = [ 1+1i  3+0i  5+0i ]
    //                   [ 2+0i  4+1i  6+0i ]
    if transA = rocblas_operation_conjugate_transpose then
    begin
      for i := 0 to K_val - 1 do
        for j := 0 to N_val - 1 do
        begin
          hostA[b * lda_val * N_val + j * lda_val + i].x := (i * N_val + j + 1 + b) * 0.6;
          hostA[b * lda_val * N_val + j * lda_val + i].y := (i - j + b) * -0.15;
        end;
    end;

    // Fill initial Hermitian matrix C
    // Example: C (3x3) = [ 2+0i  1+1i  3+0i ]
    //                    [ 1-1i  3+0i  2+1i ]
    //                    [ 3+0i  2-1i  1+0i ]
    hostC_initial[b * ldc_val * N_val + 0 + 0 * ldc_val].x := 2.0; hostC_initial[b * ldc_val * N_val + 0 + 0 * ldc_val].y := 0.0; // C[0][0]
    hostC_initial[b * ldc_val * N_val + 1 + 1 * ldc_val].x := 3.0; hostC_initial[b * ldc_val * N_val + 1 + 1 * ldc_val].y := 0.0; // C[1][1]
    hostC_initial[b * ldc_val * N_val + 2 + 2 * ldc_val].x := 1.0; hostC_initial[b * ldc_val * N_val + 2 + 2 * ldc_val].y := 0.0; // C[2][2]

    if uplo = rocblas_fill_upper then
    begin
      hostC_initial[b * ldc_val * N_val + 0 + 1 * ldc_val].x := 1.0; hostC_initial[b * ldc_val * N_val + 0 + 1 * ldc_val].y := 1.0; // C[0][1]
      hostC_initial[b * ldc_val * N_val + 0 + 2 * ldc_val].x := 3.0; hostC_initial[b * ldc_val * N_val + 0 + 2 * ldc_val].y := 0.0; // C[0][2]
      hostC_initial[b * ldc_val * N_val + 1 + 2 * ldc_val].x := 2.0; hostC_initial[b * ldc_val * N_val + 1 + 2 * ldc_val].y := 1.0; // C[1][2]
      // Fill lower triangle based on upper (Hermitian property)
      hostC_initial[b * ldc_val * N_val + 1 + 0 * ldc_val].x := 1.0; hostC_initial[b * ldc_val * N_val + 1 + 0 * ldc_val].y := -1.0; // C[1][0] = conj(C[0][1])
      hostC_initial[b * ldc_val * N_val + 2 + 0 * ldc_val].x := 3.0; hostC_initial[b * ldc_val * N_val + 2 + 0 * ldc_val].y := 0.0;  // C[2][0] = conj(C[0][2])
      hostC_initial[b * ldc_val * N_val + 2 + 1 * ldc_val].x := 2.0; hostC_initial[b * ldc_val * N_val + 2 + 1 * ldc_val].y := -1.0; // C[2][1] = conj(C[1][2])
    end
    else // rocblas_fill_lower
    begin
      hostC_initial[b * ldc_val * N_val + 1 + 0 * ldc_val].x := 1.0; hostC_initial[b * ldc_val * N_val + 1 + 0 * ldc_val].y := -1.0; // C[1][0]
      hostC_initial[b * ldc_val * N_val + 2 + 0 * ldc_val].x := 3.0; hostC_initial[b * ldc_val * N_val + 2 + 0 * ldc_val].y := 0.0;  // C[2][0]
      hostC_initial[b * ldc_val * N_val + 2 + 1 * ldc_val].x := 2.0; hostC_initial[b * ldc_val * N_val + 2 + 1 * ldc_val].y := -1.0; // C[2][1]
      // Fill upper triangle based on lower (Hermitian property)
      hostC_initial[b * ldc_val * N_val + 0 + 1 * ldc_val].x := 1.0; hostC_initial[b * ldc_val * N_val + 0 + 1 * ldc_val].y := 1.0; // C[0][1] = conj(C[1][0])
      hostC_initial[b * ldc_val * N_val + 0 + 2 * ldc_val].x := 3.0; hostC_initial[b * ldc_val * N_val + 0 + 2 * ldc_val].y := 0.0;  // C[0][2] = conj(C[2][0])
      hostC_initial[b * ldc_val * N_val + 1 + 2 * ldc_val].x := 2.0; hostC_initial[b * ldc_val * N_val + 1 + 2 * ldc_val].y := 1.0;  // C[1][2] = conj(C[2][1])
    end;

    // Copy initial C matrix for computation
    Move(hostC_initial[b * ldc_val * N_val], hostC_computed[b * ldc_val * N_val], SizeOf(rocblas_double_complex) * ldc_val * N_val);
  end;

  // 4. Allocate device memory
  devA := nil;
  devC := nil;
  if transA = rocblas_operation_none then
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(rocblas_double_complex) * N_val * K_val * batch_count_val))
  else
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(rocblas_double_complex) * K_val * N_val * batch_count_val));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, SizeOf(rocblas_double_complex) * ldc_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devC should not be nil', devC <> nil);

  // 5. Copy data H2D
  if transA = rocblas_operation_none then
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(rocblas_double_complex) * N_val * K_val * batch_count_val, hipMemcpyHostToDevice))
  else
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(rocblas_double_complex) * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC H2D', HIP_SUCCESS, hipMemcpy(devC, @hostC_computed[0], SizeOf(rocblas_double_complex) * ldc_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  // 6. Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    if transA = rocblas_operation_none then
    begin
      devA_array[b] := Procdouble_complex(PtrUInt(devA) + b * N_val * K_val * SizeOf(rocblas_double_complex));
    end
    else
    begin
      devA_array[b] := Procdouble_complex(PtrUInt(devA) + b * K_val * N_val * SizeOf(rocblas_double_complex));
    end;
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + b * ldc_val * N_val * SizeOf(rocblas_double_complex));
  end;

  // 7. Allocate and copy device pointer arrays
  devA_array_device := nil;
  devC_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devC_array_device should not be nil', devC_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device H2D', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // 8. Set rocBLAS pointer mode (alpha, beta are real scalars on host)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // 9. Call rocblas_zherk_batched (12 parameters)
  if transA = rocblas_operation_none then
  begin
    blas_status := rocblas_zherk_batched(
      handle,
      uplo,
      transA,
      N_val, K_val,
      @alpha,   // alpha (real, host pointer)
      PPProcdouble_complex(devA_array_device), // A array (array of device pointers)
      lda_val, // lda
      @beta,    // beta (real, host pointer)
      PPProcdouble_complex(devC_array_device), // C array (array of device pointers, Hermitian matrix, output)
      ldc_val, // ldc
      batch_count_val // batch_count
    );
  end
  else // transA = rocblas_operation_transpose/conjugate_transpose
  begin
    blas_status := rocblas_zherk_batched(
      handle,
      uplo,
      transA,
      N_val, K_val,
      @alpha,   // alpha (real, host pointer)
      PPProcdouble_complex(devA_array_device), // A array (array of device pointers)
      lda_val, // lda
      @beta,    // beta (real, host pointer)
      PPProcdouble_complex(devC_array_device), // C array (array of device pointers, Hermitian matrix, output)
      ldc_val, // ldc
      batch_count_val // batch_count
    );
  end;
  AssertEquals('rocblas_zherk_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // 10. Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_zherk_batched', HIP_SUCCESS, sync_status);

  // 11. Copy result D2H
  AssertEquals('hipMemcpy devC D2H', HIP_SUCCESS, hipMemcpy(@hostC_computed[0], devC, SizeOf(rocblas_double_complex) * ldc_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  // 12. Basic check (check if the matrix differs)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('Result matrix C should differ from initial matrix C for batch ' + IntToStr(b),
      not CompareMem(@hostC_initial[b * ldc_val * N_val], @hostC_computed[b * ldc_val * N_val], SizeOf(rocblas_double_complex) * ldc_val * N_val));
  end;

  // 13. Free device memory
  hipFree(devA);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devC_array_device);


end;


// --- SSYR2K_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_SSyr2k_Batched;
const
  N_val = 3;
  K_val = 2;
  batch_count_val = 2;
var
  uplo: rocblas_fill;
  trans: rocblas_operation;
  alpha, beta: Single;
  lda_val, ldb_val, ldc_val: Integer;
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC_initial, hostC_computed: array of Single;
  devA_array, devB_array, devC_array: array of PSingle;
  devA_array_device, devB_array_device, devC_array_device: PPointer;
  i, j, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  uplo := rocblas_fill_upper;
  trans := rocblas_operation_none;
  alpha := 1.0;
  beta := 1.0;

  if trans = rocblas_operation_none then
  begin
    lda_val := N_val;
    ldb_val := N_val;
  end
  else
  begin
    lda_val := K_val;
    ldb_val := K_val;
  end;
  ldc_val := N_val;

  SetLength(hostA, lda_val * K_val * batch_count_val);
  SetLength(hostB, ldb_val * K_val * batch_count_val);
  SetLength(hostC_initial, ldc_val * N_val * batch_count_val);
  SetLength(hostC_computed, ldc_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devB_array, batch_count_val);
  SetLength(devC_array, batch_count_val);

  for b := 0 to batch_count_val - 1 do
  begin
    for i := 0 to High(hostA) do
      hostA[i] := 0.0;
    for i := 0 to High(hostB) do
      hostB[i] := 0.0;
    for i := 0 to High(hostC_initial) do
      hostC_initial[i] := 0.0;

    if trans = rocblas_operation_none then
    begin
      for i := 0 to N_val - 1 do
        for j := 0 to K_val - 1 do
        begin
          hostA[b * lda_val * K_val + j * lda_val + i] := (i * K_val + j + 1 + b) * 0.5;
          hostB[b * ldb_val * K_val + j * ldb_val + i] := (i * K_val + j + 1 + b) * 0.3;
        end;
    end;

    for i := 0 to N_val - 1 do
      for j := 0 to N_val - 1 do
        hostC_initial[b * ldc_val * N_val + j * ldc_val + i] := (i + j + 1 + b) * 0.2;

    Move(hostC_initial[b * ldc_val * N_val], hostC_computed[b * ldc_val * N_val], SizeOf(Single) * ldc_val * N_val);
  end;

  devA := nil;
  devB := nil;
  devC := nil;
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(Single) * N_val * K_val * batch_count_val));
    AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, SizeOf(Single) * N_val * K_val * batch_count_val));
  end
  else
  begin
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(Single) * K_val * N_val * batch_count_val));
    AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, SizeOf(Single) * K_val * N_val * batch_count_val));
  end;
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, SizeOf(Single) * ldc_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devB should not be nil', devB <> nil);
  AssertTrue('Device pointer devC should not be nil', devC <> nil);

  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(Single) * N_val * K_val * batch_count_val, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], SizeOf(Single) * N_val * K_val * batch_count_val, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(Single) * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], SizeOf(Single) * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy devC H2D', HIP_SUCCESS, hipMemcpy(devC, @hostC_computed[0], SizeOf(Single) * ldc_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  for b := 0 to batch_count_val - 1 do
  begin
    if trans = rocblas_operation_none then
    begin
      devA_array[b] := PSingle(PtrUInt(devA) + b * N_val * K_val * SizeOf(Single));
      devB_array[b] := PSingle(PtrUInt(devB) + b * N_val * K_val * SizeOf(Single));
    end
    else
    begin
      devA_array[b] := PSingle(PtrUInt(devA) + b * K_val * N_val * SizeOf(Single));
      devB_array[b] := PSingle(PtrUInt(devB) + b * K_val * N_val * SizeOf(Single));
    end;
    devC_array[b] := PSingle(PtrUInt(devC) + b * ldc_val * N_val * SizeOf(Single));
  end;

  devA_array_device := nil;
  devB_array_device := nil;
  devC_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devB_array_device should not be nil', devB_array_device <> nil);
  AssertTrue('Device pointer devC_array_device should not be nil', devC_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device H2D', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device H2D', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  if trans = rocblas_operation_none then
  begin
    blas_status := rocblas_ssyr2k_batched(
      handle,
      uplo,
      trans,
      N_val, K_val,
      @alpha,
      PPSingle(devA_array_device), lda_val,
      PPSingle(devB_array_device), ldb_val,
      @beta,
      PPSingle(devC_array_device), ldc_val,
      batch_count_val
    );
  end
  else
  begin
    blas_status := rocblas_ssyr2k_batched(
      handle,
      uplo,
      trans,
      N_val, K_val,
      @alpha,
      PPSingle(devA_array_device), lda_val,
      PPSingle(devB_array_device), ldb_val,
      @beta,
      PPSingle(devC_array_device), ldc_val,
      batch_count_val
    );
  end;
  AssertEquals('rocblas_ssyr2k_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_ssyr2k_batched', HIP_SUCCESS, sync_status);

  AssertEquals('hipMemcpy devC D2H', HIP_SUCCESS, hipMemcpy(@hostC_computed[0], devC, SizeOf(Single) * ldc_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('Result matrix C should differ from initial matrix C for batch ' + IntToStr(b),
      not CompareMem(@hostC_initial[b * ldc_val * N_val], @hostC_computed[b * ldc_val * N_val], SizeOf(Single) * ldc_val * N_val));
  end;

  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
end;

// --- CSYR2K_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_CSYR2K_Batched;
const
  N_val = 3;
  K_val = 2;
  batch_count_val = 2;
var
  uplo: rocblas_fill;
  trans: rocblas_operation;
  alpha, beta: rocblas_float_complex;
  lda_val, ldb_val, ldc_val: Integer;
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC_initial, hostC_computed: array of rocblas_float_complex;
  devA_array, devB_array, devC_array: array of Procfloat_complex; // Array of device pointers
  devA_array_device, devB_array_device, devC_array_device: PPointer;  // Device-allocated arrays
  i, j, l, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  uplo := rocblas_fill_upper;
  trans := rocblas_operation_none;
  alpha.x := 1.0;
  alpha.y := 0.5;
  beta.x := 0.8;
  beta.y := -0.2;

  if trans = rocblas_operation_none then
  begin
    lda_val := N_val;
    ldb_val := N_val;
  end
  else
  begin
    lda_val := K_val;
    ldb_val := K_val;
  end;
  ldc_val := N_val;

  SetLength(hostA, lda_val * K_val * batch_count_val);
  SetLength(hostB, ldb_val * K_val * batch_count_val);
  SetLength(hostC_initial, ldc_val * N_val * batch_count_val);
  SetLength(hostC_computed, ldc_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devB_array, batch_count_val);
  SetLength(devC_array, batch_count_val);

  // Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    // Zero out matrices first
    for i := 0 to High(hostA) do
    begin
      hostA[i].x := 0.0;
      hostA[i].y := 0.0;
    end;
    for i := 0 to High(hostB) do
    begin
      hostB[i].x := 0.0;
      hostB[i].y := 0.0;
    end;
    for i := 0 to High(hostC_initial) do
    begin
      hostC_initial[i].x := 0.0;
      hostC_initial[i].y := 0.0;
    end;

    // Fill matrix A with test data
    if trans = rocblas_operation_none then
    begin
      for i := 0 to N_val - 1 do
        for j := 0 to K_val - 1 do
        begin
          hostA[b * lda_val * K_val + j * lda_val + i].x := (i * K_val + j + 1 + b) * 0.5;
          hostA[b * lda_val * K_val + j * lda_val + i].y := (i - j + b) * 0.1;
        end;
    end;

    // Fill matrix B with test data
    if trans = rocblas_operation_none then
    begin
      for i := 0 to N_val - 1 do
        for j := 0 to K_val - 1 do
        begin
          hostB[b * ldb_val * K_val + j * ldb_val + i].x := (i * K_val + j + 1 + b) * 0.3;
          hostB[b * ldb_val * K_val + j * ldb_val + i].y := (i + j + b) * -0.05;
        end;
    end;

    // Fill initial symmetric matrix C
    for i := 0 to N_val - 1 do
      for j := 0 to N_val - 1 do
      begin
        hostC_initial[b * ldc_val * N_val + j * ldc_val + i].x := (i + j + 1 + b) * 0.2;
        hostC_initial[b * ldc_val * N_val + j * ldc_val + i].y := (i - j + b) * 0.05;
      end;

    // Copy initial C matrix for computation
    Move(hostC_initial[b * ldc_val * N_val], hostC_computed[b * ldc_val * N_val], SizeOf(rocblas_float_complex) * ldc_val * N_val);
  end;

  // Allocate device memory for matrices
  devA := nil;
  devB := nil;
  devC := nil;
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(rocblas_float_complex) * N_val * K_val * batch_count_val));
    AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, SizeOf(rocblas_float_complex) * N_val * K_val * batch_count_val));
  end
  else
  begin
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(rocblas_float_complex) * K_val * N_val * batch_count_val));
    AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, SizeOf(rocblas_float_complex) * K_val * N_val * batch_count_val));
  end;
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, SizeOf(rocblas_float_complex) * ldc_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devB should not be nil', devB <> nil);
  AssertTrue('Device pointer devC should not be nil', devC <> nil);

  // Copy data H2D
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(rocblas_float_complex) * N_val * K_val * batch_count_val, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], SizeOf(rocblas_float_complex) * N_val * K_val * batch_count_val, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(rocblas_float_complex) * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], SizeOf(rocblas_float_complex) * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy devC H2D', HIP_SUCCESS, hipMemcpy(devC, @hostC_computed[0], SizeOf(rocblas_float_complex) * ldc_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  // Initialize device pointer arrays
  for b := 0 to batch_count_val - 1 do
  begin
    if trans = rocblas_operation_none then
    begin
      devA_array[b] := Procfloat_complex(PtrUInt(devA) + b * N_val * K_val * SizeOf(rocblas_float_complex));
      devB_array[b] := Procfloat_complex(PtrUInt(devB) + b * N_val * K_val * SizeOf(rocblas_float_complex));
    end
    else
    begin
      devA_array[b] := Procfloat_complex(PtrUInt(devA) + b * K_val * N_val * SizeOf(rocblas_float_complex));
      devB_array[b] := Procfloat_complex(PtrUInt(devB) + b * K_val * N_val * SizeOf(rocblas_float_complex));
    end;
    devC_array[b] := Procfloat_complex(PtrUInt(devC) + b * ldc_val * N_val * SizeOf(rocblas_float_complex));
  end;

  // Allocate and copy device pointer arrays
  devA_array_device := nil;
  devB_array_device := nil;
  devC_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devB_array_device should not be nil', devB_array_device <> nil);
  AssertTrue('Device pointer devC_array_device should not be nil', devC_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device H2D', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device H2D', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set pointer mode
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // Call rocblas_csyr2k_batched
  if trans = rocblas_operation_none then
  begin
    blas_status := rocblas_csyr2k_batched(
      handle,
      uplo,
      trans,
      N_val, K_val,
      @alpha,
      PPProcfloat_complex(devA_array_device), lda_val,
      PPProcfloat_complex(devB_array_device), ldb_val,
      @beta,
      PPProcfloat_complex(devC_array_device), ldc_val,
      batch_count_val
    );
  end
  else
  begin
    blas_status := rocblas_csyr2k_batched(
      handle,
      uplo,
      trans,
      N_val, K_val,
      @alpha,
      PPProcfloat_complex(devA_array_device), lda_val,
      PPProcfloat_complex(devB_array_device), ldb_val,
      @beta,
      PPProcfloat_complex(devC_array_device), ldc_val,
      batch_count_val
    );
  end;
  AssertEquals('rocblas_csyr2k_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_csyr2k_batched', HIP_SUCCESS, sync_status);

  // Copy result D2H
  AssertEquals('hipMemcpy devC D2H', HIP_SUCCESS, hipMemcpy(@hostC_computed[0], devC, SizeOf(rocblas_float_complex) * ldc_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  // Basic check (verify result is not the initial matrix)
  for b := 0 to batch_count_val - 1 do
  begin
    AssertTrue('Result matrix C should differ from initial matrix C for batch ' + IntToStr(b),
      not CompareMem(@hostC_initial[b * ldc_val * N_val], @hostC_computed[b * ldc_val * N_val], SizeOf(rocblas_float_complex) * ldc_val * N_val));
  end;

  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
end;


// --- ZSYR2K_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_ZSYR2K_Batched;
const
  N_val = 4;
  K_val = 3;
  batch_count_val = 3; // Test with multiple batches
var
  // --- Parameters ---
  uplo: rocblas_fill;
  trans: rocblas_operation;
  alpha, beta: rocblas_double_complex;

  // --- Matrix Dimensions and Leading Dimensions ---
  dim_A_rows, dim_A_cols: Integer;
  dim_B_rows, dim_B_cols: Integer;
  lda_val, ldb_val, ldc_val: Integer;

  // --- Memory Sizes ---
  size_A_per_batch, size_B_per_batch, size_C_per_batch: Integer;
  total_size_A, total_size_B, total_size_C: Integer;

  // --- Host Data ---
  hostA, hostB, hostC_initial, hostC_computed: array of rocblas_double_complex; // Double complex data

  // --- Device Data Buffers ---
  devA, devB, devC: hipDeviceptr_t; // Large contiguous buffers for all batches

  // --- Device Pointer Arrays (Host-side arrays of device pointers) ---
  devA_array, devB_array, devC_array: array of Procdouble_complex;

  // --- Device-Allocated Arrays of Pointers (Residing on the device) ---
  devA_array_device, devB_array_device, devC_array_device: PPointer;

  // --- Loop Variables ---
  i, j, l, b: Integer;
  batch_offset_A, batch_offset_B, batch_offset_C: Integer;
  dest_index,linear_index: Integer;

  // --- Return Statuses ---
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
begin
  // --- 1. Initialize Parameters ---
  uplo := rocblas_fill_upper; // Test upper triangle
  trans := rocblas_operation_none; // Test no transpose operation
  alpha.x := 1.0; alpha.y := 0.5; // Complex alpha
  beta.x := 0.8; beta.y := -0.2;  // Complex beta

  // --- 2. Calculate Matrix Dimensions and Leading Dimensions ---
  if trans = rocblas_operation_none then
  begin
    dim_A_rows := N_val;
    dim_A_cols := K_val;
    dim_B_rows := N_val;
    dim_B_cols := K_val;
  end
  else // trans = rocblas_operation_transpose (or conjugate_transpose for real SYR2K)
  begin
    dim_A_rows := K_val;
    dim_A_cols := N_val;
    dim_B_rows := K_val;
    dim_B_cols := N_val;
  end;
  lda_val := dim_A_rows;
  ldb_val := dim_B_rows;
  ldc_val := N_val; // C is always N x N

  // --- 3. Calculate Memory Sizes ---
  size_A_per_batch := lda_val * dim_A_cols;
  size_B_per_batch := ldb_val * dim_B_cols;
  size_C_per_batch := ldc_val * N_val;
  total_size_A := size_A_per_batch * batch_count_val;
  total_size_B := size_B_per_batch * batch_count_val;
  total_size_C := size_C_per_batch * batch_count_val;

  // --- 4. Allocate Host Arrays ---
  SetLength(hostA, total_size_A);
  SetLength(hostB, total_size_B);
  SetLength(hostC_initial, total_size_C);
  SetLength(hostC_computed, total_size_C); // Separate array for result
  SetLength(devA_array, batch_count_val);
  SetLength(devB_array, batch_count_val);
  SetLength(devC_array, batch_count_val);

  // --- 5. Initialize Host Data to Zero ---
  for i := 0 to High(hostA) do
  begin
    hostA[i].x := 0.0;
    hostA[i].y := 0.0;
  end;
  for i := 0 to High(hostB) do
  begin
    hostB[i].x := 0.0;
    hostB[i].y := 0.0;
  end;
  for i := 0 to High(hostC_initial) do
  begin
    hostC_initial[i].x := 0.0;
    hostC_initial[i].y := 0.0;
  end;
  for i := 0 to High(hostC_computed) do
  begin
    hostC_computed[i].x := 0.0;
    hostC_computed[i].y := 0.0;
  end;

  // --- 6. Fill Host Data with Test Values ---
  for b := 0 to batch_count_val - 1 do
  begin
    // --- Fill Matrix A ---
    batch_offset_A := b * size_A_per_batch;
    for i := 0 to dim_A_rows - 1 do
      for j := 0 to dim_A_cols - 1 do
      begin
        // Column-major indexing: element (i,j) is at index (batch_offset + j * lda + i)
        linear_index := batch_offset_A + j * lda_val + i;
        // Implicit bounds check: loop bounds ensure linear_index is within [0, total_size_A)
        hostA[linear_index].x := (i * dim_A_cols + j + 1 + b) * 0.6;
        hostA[linear_index].y := (i - j + b) * -0.15;
      end;

    // --- Fill Matrix B ---
    batch_offset_B := b * size_B_per_batch;
    for i := 0 to dim_B_rows - 1 do
      for j := 0 to dim_B_cols - 1 do
      begin
        // Column-major indexing: element (i,j) is at index (batch_offset + j * ldb + i)
        linear_index := batch_offset_B + j * ldb_val + i;
        // Implicit bounds check: loop bounds ensure linear_index is within [0, total_size_B)
        hostB[linear_index].x := (i * dim_B_cols + j + 1 + b) * 0.4;
        hostB[linear_index].y := (i + j + b) * 0.08;
      end;

    // --- Fill Initial Matrix C (Symmetric) ---
    batch_offset_C := b * size_C_per_batch;
    for i := 0 to N_val - 1 do
      for j := 0 to N_val - 1 do
      begin
        // Column-major indexing: element (i,j) is at index (batch_offset + j * ldc + i)
        linear_index := batch_offset_C + j * ldc_val + i;
        // Implicit bounds check: loop bounds ensure linear_index is within [0, total_size_C)
        hostC_initial[linear_index].x := (i + j + 1 + b) * -0.07;
        hostC_initial[linear_index].y := (i - j + b) * 0.03;
      end;

    // --- Enforce Symmetry on hostC_initial based on uplo ---
    if uplo = rocblas_fill_upper then
    begin
      // Copy upper triangle to lower triangle (symmetric, no conjugate)
      for i := 1 to N_val - 1 do // Iterate over lower triangle rows (skip diagonal)
        for j := 0 to i - 1 do // Iterate over lower triangle columns
        begin
          // C[i][j] = C[j][i]
          linear_index := batch_offset_C + j * ldc_val + i; // Source (upper triangle)
          dest_index := batch_offset_C + i * ldc_val + j; // Destination (lower triangle)
          // Implicit bounds check: loop bounds ensure indices are within [0, total_size_C)
          hostC_initial[dest_index].x := hostC_initial[linear_index].x;
          hostC_initial[dest_index].y := hostC_initial[linear_index].y;
        end;
    end
    else // rocblas_fill_lower
    begin
      // Copy lower triangle to upper triangle (symmetric, no conjugate)
      for i := 0 to N_val - 2 do // Iterate over upper triangle rows
        for j := i + 1 to N_val - 1 do // Iterate over upper triangle columns (skip diagonal)
        begin
          // C[i][j] = C[j][i]
          linear_index := batch_offset_C + j * ldc_val + i; // Source (lower triangle)
          dest_index := batch_offset_C + i * ldc_val + j; // Destination (upper triangle)
          // Implicit bounds check: loop bounds ensure indices are within [0, total_size_C)
          hostC_initial[dest_index].x := hostC_initial[linear_index].x;
          hostC_initial[dest_index].y := hostC_initial[linear_index].y;
        end;
    end;

    // --- Copy Initial C to Computed C for this batch ---
    Move(hostC_initial[batch_offset_C], hostC_computed[batch_offset_C], SizeOf(rocblas_double_complex) * size_C_per_batch);
  end;

  // --- 7. Allocate Device Memory ---
  devA := nil;
  devB := nil;
  devC := nil;
  AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, SizeOf(rocblas_double_complex) * total_size_A));
  AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, SizeOf(rocblas_double_complex) * total_size_B));
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, SizeOf(rocblas_double_complex) * total_size_C));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devB should not be nil', devB <> nil);
  AssertTrue('Device pointer devC should not be nil', devC <> nil);

  // --- 8. Copy Data Host to Device ---
  AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], SizeOf(rocblas_double_complex) * total_size_A, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], SizeOf(rocblas_double_complex) * total_size_B, hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC H2D', HIP_SUCCESS, hipMemcpy(devC, @hostC_computed[0], SizeOf(rocblas_double_complex) * total_size_C, hipMemcpyHostToDevice));

  // --- 9. Set Up Device Pointer Arrays (Host-Side) ---
  for b := 0 to batch_count_val - 1 do
  begin
    // Calculate the device pointer to the start of the b-th batch's data within the large buffer
    devA_array[b] := Procdouble_complex(PtrUInt(devA) + b * SizeOf(rocblas_double_complex) * size_A_per_batch);
    devB_array[b] := Procdouble_complex(PtrUInt(devB) + b * SizeOf(rocblas_double_complex) * size_B_per_batch);
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + b * SizeOf(rocblas_double_complex) * size_C_per_batch);
  end;

  // --- 10. Allocate and Copy Device Pointer Arrays (Device-Side) ---
  devA_array_device := nil;
  devB_array_device := nil;
  devC_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devB_array_device should not be nil', devB_array_device <> nil);
  AssertTrue('Device pointer devC_array_device should not be nil', devC_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device H2D', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device H2D', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // --- 11. Set rocBLAS Pointer Mode ---
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // --- 12. Call rocblas_zsyr2k_batched ---
  blas_status := rocblas_zsyr2k_batched(
    handle,
    uplo,
    trans,
    N_val, K_val,
    @alpha, // alpha (host pointer to complex scalar)
    PPProcdouble_complex(devA_array_device), // A array (device pointer to array of device pointers)
    lda_val, // lda
    PPProcdouble_complex(devB_array_device), // B array (device pointer to array of device pointers)
    ldb_val, // ldb
    @beta, // beta (host pointer to complex scalar)
    PPProcdouble_complex(devC_array_device), // C array (device pointer to array of device pointers, output)
    ldc_val, // ldc
    batch_count_val // batch_count
  );
  AssertEquals('rocblas_zsyr2k_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // --- 13. Synchronize ---
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_zsyr2k_batched', HIP_SUCCESS, sync_status);

  // --- 14. Copy Result Back to Host ---
  AssertEquals('hipMemcpy devC D2H', HIP_SUCCESS, hipMemcpy(@hostC_computed[0], devC, SizeOf(rocblas_double_complex) * total_size_C, hipMemcpyDeviceToHost));

  // --- 15. Validate Results ---
  for b := 0 to batch_count_val - 1 do
  begin
    batch_offset_C := b * size_C_per_batch;
    AssertTrue('Result matrix C should differ from initial matrix C for batch ' + IntToStr(b),
      not CompareMem(@hostC_initial[batch_offset_C], @hostC_computed[batch_offset_C], SizeOf(rocblas_double_complex) * size_C_per_batch));
  end;


  // --- 16. Free Device Memory ---
  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
end;



// --- CHER2K_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_CHER2K_Batched;
const
  N_val = 3;
  K_val = 2;
  batch_count_val = 2;
var
  uplo: rocblas_fill;
  trans: rocblas_operation;
  alpha: rocblas_float_complex; // alpha is complex for HER2K
  beta: Single;                 // beta is real for HER2K
  lda_val, ldb_val, ldc_val: Integer;
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC_initial, hostC_computed: array of rocblas_float_complex;
  devA_array, devB_array, devC_array: array of Procdouble_complex;
  devA_array_device, devB_array_device, devC_array_device: PPointer;
  elem_size: Integer;
  i, j, l, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  batch_offset_A, batch_offset_B, batch_offset_C: Integer;
  dest_index, linear_index: Integer;
begin
  uplo := rocblas_fill_upper;
  trans := rocblas_operation_none;
  alpha.x := 1.0;
  alpha.y := 0.5;
  beta := 1.5; // Real beta

  elem_size := SizeOf(rocblas_float_complex);

  if trans = rocblas_operation_none then
  begin
    lda_val := N_val;
    ldb_val := N_val;
  end
  else
  begin
    lda_val := K_val;
    ldb_val := K_val;
  end;
  ldc_val := N_val;

  SetLength(hostA, lda_val * K_val * batch_count_val);
  SetLength(hostB, ldb_val * K_val * batch_count_val);
  SetLength(hostC_initial, ldc_val * N_val * batch_count_val);
  SetLength(hostC_computed, ldc_val * N_val * batch_count_val);
  SetLength(devA_array, batch_count_val);
  SetLength(devB_array, batch_count_val);
  SetLength(devC_array, batch_count_val);

  // Initialize host data
  for b := 0 to batch_count_val - 1 do
  begin
    for i := 0 to High(hostA) do
    begin
      hostA[i].x := 0.0;
      hostA[i].y := 0.0;
    end;
    for i := 0 to High(hostB) do
    begin
      hostB[i].x := 0.0;
      hostB[i].y := 0.0;
    end;
    for i := 0 to High(hostC_initial) do
    begin
      hostC_initial[i].x := 0.0;
      hostC_initial[i].y := 0.0;
    end;

    // Fill matrix A with test data
    if trans = rocblas_operation_none then
    begin
      batch_offset_A := b * lda_val * K_val;
      for i := 0 to N_val - 1 do
        for j := 0 to K_val - 1 do
        begin
          linear_index := batch_offset_A + j * lda_val + i;
          AssertTrue('Index A out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostA)));
          hostA[linear_index].x := (i * K_val + j + 1 + b) * 0.5;
          hostA[linear_index].y := (i - j + b) * 0.1;
        end;
    end
    else
    begin
      batch_offset_A := b * lda_val * N_val;
      for i := 0 to K_val - 1 do
        for j := 0 to N_val - 1 do
        begin
          linear_index := batch_offset_A + j * lda_val + i;
          AssertTrue('Index A out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostA)));
          hostA[linear_index].x := (i * N_val + j + 1 + b) * 0.6;
          hostA[linear_index].y := (i - j + b) * -0.15;
        end;
    end;

    // Fill matrix B with test data
    if trans = rocblas_operation_none then
    begin
      batch_offset_B := b * ldb_val * K_val;
      for i := 0 to N_val - 1 do
        for j := 0 to K_val - 1 do
        begin
          linear_index := batch_offset_B + j * ldb_val + i;
          AssertTrue('Index B out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostB)));
          hostB[linear_index].x := (i * K_val + j + 1 + b) * 0.3;
          hostB[linear_index].y := (i + j + b) * -0.05;
        end;
    end
    else
    begin
      batch_offset_B := b * ldb_val * N_val;
      for i := 0 to K_val - 1 do
        for j := 0 to N_val - 1 do
        begin
          linear_index := batch_offset_B + j * ldb_val + i;
          AssertTrue('Index B out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostB)));
          hostB[linear_index].x := (i * N_val + j + 1 + b) * 0.4;
          hostB[linear_index].y := (i + j + b) * 0.08;
        end;
    end;

    // Fill initial Hermitian matrix C
    batch_offset_C := b * ldc_val * N_val;
    for i := 0 to N_val - 1 do
      for j := 0 to N_val - 1 do
      begin
        linear_index := batch_offset_C + j * ldc_val + i;
        AssertTrue('Index C out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostC_initial)));
        hostC_initial[linear_index].x := (i + j + 1 + b) * 0.2;
        hostC_initial[linear_index].y := (i - j + b) * 0.05;
      end;

    // Enforce Hermitian property on hostC_initial based on uplo
    if uplo = rocblas_fill_upper then
    begin
      // Copy upper triangle to lower triangle (Hermitian: C[i,j] = conj(C[j,i]))
      for i := 1 to N_val - 1 do
        for j := 0 to i - 1 do
        begin
          linear_index := batch_offset_C + j * ldc_val + i; // Source (upper triangle)
          dest_index := batch_offset_C + i * ldc_val + j; // Destination (lower triangle)
          AssertTrue('Source index C (upper->lower) out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostC_initial)));
          AssertTrue('Dest index C (upper->lower) out of bounds for batch ' + IntToStr(b), (dest_index >= 0) and (dest_index < Length(hostC_initial)));
          hostC_initial[dest_index].x := hostC_initial[linear_index].x;  // Real part stays the same
          hostC_initial[dest_index].y := -hostC_initial[linear_index].y; // Imaginary part is negated (conjugate)
        end;
    end
    else
    begin
      // Copy lower triangle to upper triangle (Hermitian: C[i,j] = conj(C[j,i]))
      for i := 0 to N_val - 2 do
        for j := i + 1 to N_val - 1 do
        begin
          linear_index := batch_offset_C + j * ldc_val + i; // Source (lower triangle)
          dest_index := batch_offset_C + i * ldc_val + j; // Destination (upper triangle)
          AssertTrue('Source index C (lower->upper) out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostC_initial)));
          AssertTrue('Dest index C (lower->upper) out of bounds for batch ' + IntToStr(b), (dest_index >= 0) and (dest_index < Length(hostC_initial)));
          hostC_initial[dest_index].x := hostC_initial[linear_index].x;  // Real part stays the same
          hostC_initial[dest_index].y := -hostC_initial[linear_index].y; // Imaginary part is negated (conjugate)
        end;
    end;

    // Copy initial C matrix for computation
    Move(hostC_initial[batch_offset_C], hostC_computed[batch_offset_C], elem_size * ldc_val * N_val);
  end;

  // Allocate device memory
  devA := nil;
  devB := nil;
  devC := nil;
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, elem_size * N_val * K_val * batch_count_val));
    AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, elem_size * N_val * K_val * batch_count_val));
  end
  else
  begin
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, elem_size * K_val * N_val * batch_count_val));
    AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, elem_size * K_val * N_val * batch_count_val));
  end;
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, elem_size * ldc_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devB should not be nil', devB <> nil);
  AssertTrue('Device pointer devC should not be nil', devC <> nil);

  // Copy data H2D
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], elem_size * N_val * K_val * batch_count_val, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], elem_size * N_val * K_val * batch_count_val, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], elem_size * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], elem_size * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy devC H2D', HIP_SUCCESS, hipMemcpy(devC, @hostC_computed[0], elem_size * ldc_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  // Initialize device pointer arrays (host-side arrays pointing into large buffers)
  for b := 0 to batch_count_val - 1 do
  begin
    if trans = rocblas_operation_none then
    begin
      devA_array[b] := Procdouble_complex(PtrUInt(devA) + b * elem_size * N_val * K_val);
      devB_array[b] := Procdouble_complex(PtrUInt(devB) + b * elem_size * N_val * K_val);
    end
    else
    begin
      devA_array[b] := Procdouble_complex(PtrUInt(devA) + b * elem_size * K_val * N_val);
      devB_array[b] := Procdouble_complex(PtrUInt(devB) + b * elem_size * K_val * N_val);
    end;
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + b * elem_size * ldc_val * N_val);
  end;

  // Allocate and copy device pointer arrays (device-side arrays of pointers)
  devA_array_device := nil;
  devB_array_device := nil;
  devC_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devB_array_device should not be nil', devB_array_device <> nil);
  AssertTrue('Device pointer devC_array_device should not be nil', devC_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device H2D', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device H2D', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // Set rocBLAS pointer mode (alpha, beta are host pointers)
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // Call rocblas_cher2k_batched
  blas_status := rocblas_cher2k_batched(
    handle,
    uplo,
    trans,
    N_val, K_val,
    @alpha,   // alpha (complex, host pointer)
    PPProcfloat_complex(devA_array_device), // A array
    lda_val,
    PPProcfloat_complex(devB_array_device), // B array
    ldb_val,
    @beta,    // beta (real, host pointer)
    PPProcfloat_complex(devC_array_device), // C array (output)
    ldc_val,
    batch_count_val
  );
  AssertEquals('rocblas_cher2k_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // Synchronize
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_cher2k_batched', HIP_SUCCESS, sync_status);

  // Copy result back to host
  AssertEquals('hipMemcpy devC D2H', HIP_SUCCESS, hipMemcpy(@hostC_computed[0], devC, elem_size * ldc_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  // Basic validation: check if result differs from initial matrix
  for b := 0 to batch_count_val - 1 do
  begin
    batch_offset_C := b * ldc_val * N_val;
    AssertTrue('Result matrix C should differ from initial matrix C for batch ' + IntToStr(b),
      not CompareMem(@hostC_initial[batch_offset_C], @hostC_computed[batch_offset_C], elem_size * ldc_val * N_val));
  end;

  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
end;


// --- ZHER2K_Batched ---
procedure TTestRocBLAS_Level3_Batched.Test_ZHER2K_Batched;
const
  N_val = 4;
  K_val = 3;
  batch_count_val = 1; // Smaller batch for double complex
var
  uplo: rocblas_fill;
  trans: rocblas_operation;
  alpha: rocblas_double_complex; // alpha is complex for HER2K
  beta: Double;                  // beta is real for HER2K
  lda_val, ldb_val, ldc_val: Integer;
  size_A_per_batch, size_B_per_batch, size_C_per_batch: Integer;
  total_size_A, total_size_B, total_size_C: Integer;
  devA, devB, devC: hipDeviceptr_t;
  hostA, hostB, hostC_initial, hostC_computed: array of rocblas_double_complex;
  devA_array, devB_array, devC_array: array of Procdouble_complex;
  devA_array_device, devB_array_device, devC_array_device: PPointer;
  elem_size: Integer;
  i, j, l, b: Integer;
  blas_status: rocblas_status;
  pm_status: rocblas_status;
  sync_status: Integer; // hipError_t
  batch_offset_A, batch_offset_B, batch_offset_C: Integer;
  dest_index, linear_index: Integer;
begin
  uplo := rocblas_fill_lower; // Different uplo
  trans := rocblas_operation_conjugate_transpose; // Different trans (common for HER2K)
  alpha.x := 0.8;
  alpha.y := -0.3;
  beta := 2.1; // Real beta

  elem_size := SizeOf(rocblas_double_complex);

  // --- Calculate Leading Dimensions ---
  if trans = rocblas_operation_none then
  begin
    lda_val := N_val; // A is N x K
    ldb_val := N_val; // B is N x K
  end
  else
  begin
    lda_val := K_val; // A is K x N
    ldb_val := K_val; // B is K x N
  end;
  ldc_val := N_val; // C is N x N

  // --- Calculate Memory Sizes (FIXED) ---
  // Calculate size per batch based on actual matrix dimensions and leading dimension
  if trans = rocblas_operation_none then
  begin
    // A is N x K, lda = N -> size per batch = N * K (but leading dimension is N)
    size_A_per_batch := lda_val * K_val; // = N_val * K_val
    // B is N x K, ldb = N -> size per batch = N * K (but leading dimension is N)
    size_B_per_batch := ldb_val * K_val; // = N_val * K_val
  end
  else
  begin
    // A is K x N, lda = K -> size per batch = K * N (leading dimension is K)
    size_A_per_batch := lda_val * N_val; // = K_val * N_val
    // B is K x N, ldb = K -> size per batch = K * N (leading dimension is K)
    size_B_per_batch := ldb_val * N_val; // = K_val * N_val
  end;
  // C is always N x N, ldc = N -> size per batch = N * N (leading dimension is N)
  size_C_per_batch := ldc_val * N_val; // = N_val * N_val

  total_size_A := size_A_per_batch * batch_count_val;
  total_size_B := size_B_per_batch * batch_count_val;
  total_size_C := size_C_per_batch * batch_count_val;

  // --- Allocate Host Arrays (FIXED) ---
  SetLength(hostA, total_size_A);
  SetLength(hostB, total_size_B);
  SetLength(hostC_initial, total_size_C);
  SetLength(hostC_computed, total_size_C); // Separate array for result
  SetLength(devA_array, batch_count_val);
  SetLength(devB_array, batch_count_val);
  SetLength(devC_array, batch_count_val);

  // --- Initialize Host Data ---
  for b := 0 to batch_count_val - 1 do
  begin
    for i := 0 to High(hostA) do
    begin
      hostA[i].x := 0.0;
      hostA[i].y := 0.0;
    end;
    for i := 0 to High(hostB) do
    begin
      hostB[i].x := 0.0;
      hostB[i].y := 0.0;
    end;
    for i := 0 to High(hostC_initial) do
    begin
      hostC_initial[i].x := 0.0;
      hostC_initial[i].y := 0.0;
    end;

    // --- Fill Matrix A with test data ---
    if trans = rocblas_operation_none then
    begin
      // A is N x K
      batch_offset_A := b * lda_val * K_val;
      for i := 0 to N_val - 1 do
        for j := 0 to K_val - 1 do
        begin
          linear_index := batch_offset_A + j * lda_val + i; // Column-major: col * ld + row
          AssertTrue('Index A (trans=none) out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostA)));
          hostA[linear_index].x := (i * K_val + j + 1 + b) * 0.8;
          hostA[linear_index].y := (i - j + b) * -0.18;
        end;
    end
    else
    begin
      // A is K x N
      batch_offset_A := b * lda_val * N_val;
      for i := 0 to K_val - 1 do
        for j := 0 to N_val - 1 do
        begin
          linear_index := batch_offset_A + j * lda_val + i; // Column-major: col * ld + row
          AssertTrue('Index A (trans!=none) out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostA)));
          hostA[linear_index].x := (i * N_val + j + 1 + b) * 0.8;
          hostA[linear_index].y := (i - j + b) * -0.18;
        end;
    end;

    // --- Fill Matrix B with test data ---
    if trans = rocblas_operation_none then
    begin
      // B is N x K
      batch_offset_B := b * ldb_val * K_val;
      for i := 0 to N_val - 1 do
        for j := 0 to K_val - 1 do
        begin
          linear_index := batch_offset_B + j * ldb_val + i; // Column-major: col * ld + row
          AssertTrue('Index B (trans=none) out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostB)));
          hostB[linear_index].x := (i * K_val + j + 1 + b) * 0.4;
          hostB[linear_index].y := (i + j + b) * -0.09;
        end;
    end
    else
    begin
      // B is K x N
      batch_offset_B := b * ldb_val * N_val;
      for i := 0 to K_val - 1 do
        for j := 0 to N_val - 1 do
        begin
          linear_index := batch_offset_B + j * ldb_val + i; // Column-major: col * ld + row
          AssertTrue('Index B (trans!=none) out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostB)));
          hostB[linear_index].x := (i * N_val + j + 1 + b) * 0.4;
          hostB[linear_index].y := (i + j + b) * -0.09;
        end;
    end;

    // --- Fill Initial Hermitian Matrix C ---
    batch_offset_C := b * ldc_val * N_val;
    for i := 0 to N_val - 1 do
      for j := 0 to N_val - 1 do
      begin
        linear_index := batch_offset_C + j * ldc_val + i; // Column-major: col * ld + row
        AssertTrue('Index C out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostC_initial)));
        hostC_initial[linear_index].x := (i + j + 1 + b) * -0.07;
        hostC_initial[linear_index].y := (i - j + b) * 0.03;
      end;

    // --- Enforce Hermitian Property on hostC_initial based on uplo ---
    if uplo = rocblas_fill_upper then
    begin
      // Copy upper triangle to lower triangle (Hermitian: C[i,j] = conj(C[j,i]))
      for i := 1 to N_val - 1 do // Iterate over lower triangle rows (skip diagonal)
        for j := 0 to i - 1 do // Iterate over lower triangle columns
        begin
          linear_index := batch_offset_C + j * ldc_val + i; // Source (upper triangle)
          dest_index := batch_offset_C + i * ldc_val + j; // Destination (lower triangle)
          AssertTrue('Source index C (upper->lower) out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostC_initial)));
          AssertTrue('Dest index C (upper->lower) out of bounds for batch ' + IntToStr(b), (dest_index >= 0) and (dest_index < Length(hostC_initial)));
          hostC_initial[dest_index].x := hostC_initial[linear_index].x;   // Real part stays the same
          hostC_initial[dest_index].y := -hostC_initial[linear_index].y;  // Imaginary part is negated (conjugate)
        end;
    end
    else
    begin
      // Copy lower triangle to upper triangle (Hermitian: C[i,j] = conj(C[j,i]))
      for i := 0 to N_val - 2 do // Iterate over upper triangle rows
        for j := i + 1 to N_val - 1 do // Iterate over upper triangle columns (skip diagonal)
        begin
          linear_index := batch_offset_C + j * ldc_val + i; // Source (lower triangle)
          dest_index := batch_offset_C + i * ldc_val + j; // Destination (upper triangle)
          AssertTrue('Source index C (lower->upper) out of bounds for batch ' + IntToStr(b), (linear_index >= 0) and (linear_index < Length(hostC_initial)));
          AssertTrue('Dest index C (lower->upper) out of bounds for batch ' + IntToStr(b), (dest_index >= 0) and (dest_index < Length(hostC_initial)));
          hostC_initial[dest_index].x := hostC_initial[linear_index].x;   // Real part stays the same
          hostC_initial[dest_index].y := -hostC_initial[linear_index].y;  // Imaginary part is negated (conjugate)
        end;
    end;

    // --- Copy Initial C matrix for computation ---
    Move(hostC_initial[batch_offset_C], hostC_computed[batch_offset_C], elem_size * ldc_val * N_val);
  end;

  // --- Allocate Device Memory ---
  devA := nil;
  devB := nil;
  devC := nil;
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, elem_size * N_val * K_val * batch_count_val));
    AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, elem_size * N_val * K_val * batch_count_val));
  end
  else
  begin
    AssertEquals('hipMalloc devA', HIP_SUCCESS, hipMalloc(devA, elem_size * K_val * N_val * batch_count_val));
    AssertEquals('hipMalloc devB', HIP_SUCCESS, hipMalloc(devB, elem_size * K_val * N_val * batch_count_val));
  end;
  AssertEquals('hipMalloc devC', HIP_SUCCESS, hipMalloc(devC, elem_size * ldc_val * N_val * batch_count_val));
  AssertTrue('Device pointer devA should not be nil', devA <> nil);
  AssertTrue('Device pointer devB should not be nil', devB <> nil);
  AssertTrue('Device pointer devC should not be nil', devC <> nil);

  // --- Copy Data H2D ---
  if trans = rocblas_operation_none then
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], elem_size * N_val * K_val * batch_count_val, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], elem_size * N_val * K_val * batch_count_val, hipMemcpyHostToDevice));
  end
  else
  begin
    AssertEquals('hipMemcpy devA H2D', HIP_SUCCESS, hipMemcpy(devA, @hostA[0], elem_size * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
    AssertEquals('hipMemcpy devB H2D', HIP_SUCCESS, hipMemcpy(devB, @hostB[0], elem_size * K_val * N_val * batch_count_val, hipMemcpyHostToDevice));
  end;
  AssertEquals('hipMemcpy devC H2D', HIP_SUCCESS, hipMemcpy(devC, @hostC_computed[0], elem_size * ldc_val * N_val * batch_count_val, hipMemcpyHostToDevice));

  // --- Initialize Device Pointer Arrays (Host-Side) ---
  for b := 0 to batch_count_val - 1 do
  begin
    if trans = rocblas_operation_none then
    begin
      devA_array[b] := Procdouble_complex(PtrUInt(devA) + b * elem_size * N_val * K_val);
      devB_array[b] := Procdouble_complex(PtrUInt(devB) + b * elem_size * N_val * K_val);
    end
    else
    begin
      devA_array[b] := Procdouble_complex(PtrUInt(devA) + b * elem_size * K_val * N_val);
      devB_array[b] := Procdouble_complex(PtrUInt(devB) + b * elem_size * K_val * N_val);
    end;
    devC_array[b] := Procdouble_complex(PtrUInt(devC) + b * elem_size * ldc_val * N_val);
  end;

  // --- Allocate and Copy Device Pointer Arrays (Device-Side) ---
  devA_array_device := nil;
  devB_array_device := nil;
  devC_array_device := nil;
  AssertEquals('hipMalloc devA_array_device', HIP_SUCCESS, hipMalloc(devA_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devB_array_device', HIP_SUCCESS, hipMalloc(devB_array_device, batch_count_val * SizeOf(Pointer)));
  AssertEquals('hipMalloc devC_array_device', HIP_SUCCESS, hipMalloc(devC_array_device, batch_count_val * SizeOf(Pointer)));
  AssertTrue('Device pointer devA_array_device should not be nil', devA_array_device <> nil);
  AssertTrue('Device pointer devB_array_device should not be nil', devB_array_device <> nil);
  AssertTrue('Device pointer devC_array_device should not be nil', devC_array_device <> nil);

  AssertEquals('hipMemcpy devA_array_device H2D', HIP_SUCCESS, hipMemcpy(devA_array_device, @devA_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devB_array_device H2D', HIP_SUCCESS, hipMemcpy(devB_array_device, @devB_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));
  AssertEquals('hipMemcpy devC_array_device H2D', HIP_SUCCESS, hipMemcpy(devC_array_device, @devC_array[0], batch_count_val * SizeOf(Pointer), hipMemcpyHostToDevice));

  // --- Set rocBLAS Pointer Mode ---
  pm_status := rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  AssertEquals('rocblas_set_pointer_mode', Integer(rocblas_status_success), Integer(pm_status));

  // --- Call rocblas_zher2k_batched ---
  if trans = rocblas_operation_none then
  begin
    blas_status := rocblas_zher2k_batched(
      handle,
      uplo,
      trans,
      N_val, K_val,
      @alpha,   // alpha (complex, host pointer)
      PPProcdouble_complex(devA_array_device), // A array
      lda_val, // lda
      PPProcdouble_complex(devB_array_device), // B array
      ldb_val, // ldb
      @beta,    // beta (real, host pointer)
      PPProcdouble_complex(devC_array_device), // C array (output)
      ldc_val, // ldc
      batch_count_val
    );
  end
  else
  begin
    blas_status := rocblas_zher2k_batched(
      handle,
      uplo,
      trans,
      N_val, K_val,
      @alpha,   // alpha (complex, host pointer)
      PPProcdouble_complex(devA_array_device), // A array
      lda_val, // lda
      PPProcdouble_complex(devB_array_device), // B array
      ldb_val, // ldb
      @beta,    // beta (real, host pointer)
      PPProcdouble_complex(devC_array_device), // C array (output)
      ldc_val, // ldc
      batch_count_val
    );
  end;
  AssertEquals('rocblas_zher2k_batched should return success', Integer(rocblas_status_success), Integer(blas_status));

  // --- Synchronize ---
  sync_status := hipDeviceSynchronize();
  AssertEquals('hipDeviceSynchronize post-rocblas_zher2k_batched', HIP_SUCCESS, sync_status);

  // --- Copy Result Back to Host ---
  AssertEquals('hipMemcpy devC D2H', HIP_SUCCESS, hipMemcpy(@hostC_computed[0], devC, elem_size * ldc_val * N_val * batch_count_val, hipMemcpyDeviceToHost));

  // --- Validate Results ---
  for b := 0 to batch_count_val - 1 do
  begin
    batch_offset_C := b * ldc_val * N_val;
    AssertTrue('Result matrix C should differ from initial matrix C for batch ' + IntToStr(b),
      not CompareMem(@hostC_initial[batch_offset_C], @hostC_computed[batch_offset_C], elem_size * ldc_val * N_val));
  end;

  hipFree(devA);
  hipFree(devB);
  hipFree(devC);
  hipFree(devA_array_device);
  hipFree(devB_array_device);
  hipFree(devC_array_device);
end;

initialization
  RegisterTest(TTestRocBLAS_Level3_Batched);

end.
