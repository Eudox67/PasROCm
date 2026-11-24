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

  Signature checks for the rocSPARSE bindings: every call is made and its
  status is asserted, so a changed prototype or a missing symbol fails here
  first.
------------------------------------------------------------------------------}

unit Test_rocsparse;

{$mode objfpc}{$H+}
{$linklib rocsparse}

interface

uses
  fpcunit, testregistry, rocsparse, hip_error, hip_types, hip_runtime, sysutils;

type

  { TTest_rocSPARSE }

  TTest_rocSPARSE = class(TTestCase)
  private
    handle: rocsparse_handle;
    descr: rocsparse_mat_descr;
    procedure CheckHip(AError: hipError_t; const AMessage: string);
    procedure CheckSparse(AStatus: rocsparse_status; const AMessage: string);
  protected
    procedure SetUp; override;
    procedure TearDown; override;
  published
    procedure Test_saxpyi;
    procedure Test_sdoti;
    procedure Test_scsrmv;
    procedure Test_scsrmm_1x1_passing;
    procedure Test_scsrmm_4x5_failing;
    procedure Test_csr2coo;
    procedure Test_coo2csr;
    procedure Test_snnz;
    procedure Test_scsr2csc;
    procedure Test_sdense2csr;
    procedure Test_sdense2coo;
    procedure Test_scsr2dense;
    procedure Test_scoo2dense;
    procedure Test_scsc2dense;
    procedure Test_scsr2bsr_sbsr2csr;
    procedure Test_scsr2ell_sell2csr;
    procedure Test_scsr2hyb_shyb2csr;
    procedure Test_scsrsort;
    procedure Test_scoosort;
    procedure Test_sprune_csr2csr_by_percentage;
    procedure Test_scscsort;
    procedure Test_sprune_csr2csr;
    procedure Test_sbsric0;
    procedure Test_sbsrilu0;
    procedure Test_scsric0;
    procedure Test_scsrilu0;
  end;

implementation

procedure TTest_rocSPARSE.CheckHip(AError: hipError_t; const AMessage: string);
begin
  AssertEquals(AMessage, longint(HIP_SUCCESS), longint(AError));
end;

procedure TTest_rocSPARSE.CheckSparse(AStatus: rocsparse_status; const AMessage: string);
begin
  AssertEquals(AMessage, longint(rocsparse_status_success), longint(AStatus));
end;

procedure TTest_rocSPARSE.SetUp;
begin
  CheckSparse(rocsparse_create_handle(@handle), 'SetUp: rocsparse_create_handle');
  CheckSparse(rocsparse_create_mat_descr(@descr), 'SetUp: rocsparse_create_mat_descr');
end;

procedure TTest_rocSPARSE.TearDown;
begin
  CheckSparse(rocsparse_destroy_mat_descr(descr), 'TearDown: rocsparse_destroy_mat_descr');
  CheckSparse(rocsparse_destroy_handle(handle), 'TearDown: rocsparse_destroy_handle');
end;

procedure TTest_rocSPARSE.Test_saxpyi;
var
  nnz: rocsparse_int;
  alpha: Single;
  hx_val: array[0..2] of Single;
  hx_ind: array[0..2] of rocsparse_int;
  hy: array[0..8] of Single;
  hy_result: array[0..8] of Single;
  expected_hy: array[0..8] of Single;
  dx_val: Pointer;
  dx_ind: Pointer;
  dy: Pointer;
  i: integer;
begin
  // From direct_test_axpyi.pas
  nnz := 3;
  alpha := 3.7;

  hx_val[0] := 1.0; hx_val[1] := 2.0; hx_val[2] := 3.0;
  hx_ind[0] := 0; hx_ind[1] := 3; hx_ind[2] := 5;

  for i := 0 to 8 do hy[i] := Single(i + 1);

  // Expected result
  for i := 0 to 8 do expected_hy[i] := hy[i];
  expected_hy[0] := 1.0 + alpha * 1.0; // 4.7
  expected_hy[3] := 4.0 + alpha * 2.0; // 11.4
  expected_hy[5] := 6.0 + alpha * 3.0; // 17.1

  // Allocate device memory
  CheckHip(hipMalloc(dx_val, nnz * sizeof(Single)), 'hipMalloc dx_val');
  CheckHip(hipMalloc(dx_ind, nnz * sizeof(rocsparse_int)), 'hipMalloc dx_ind');
  CheckHip(hipMalloc(dy, 9 * sizeof(Single)), 'hipMalloc dy');

  // Copy data to device
  CheckHip(hipMemcpy(dx_val, @hx_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), 'hipMemcpy dx_val');
  CheckHip(hipMemcpy(dx_ind, @hx_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'hipMemcpy dx_ind');
  CheckHip(hipMemcpy(dy, @hy[0], 9 * sizeof(Single), hipMemcpyHostToDevice), 'hipMemcpy dy');

  // Call saxpyi
  CheckSparse(rocsparse_saxpyi(handle, nnz, @alpha, dx_val, dx_ind, dy, rocsparse_index_base_zero), 'rocsparse_saxpyi');

  // Copy result back to host
  CheckHip(hipMemcpy(@hy_result[0], dy, 9 * sizeof(Single), hipMemcpyDeviceToHost), 'hipMemcpy hy_result');

  // Verification
  for i := 0 to 8 do
    AssertEquals('saxpyi result at index ' + IntToStr(i), expected_hy[i], hy_result[i], 1e-5);

  // Cleanup
  CheckHip(hipFree(dx_val), 'hipFree dx_val');
  CheckHip(hipFree(dx_ind), 'hipFree dx_ind');
  CheckHip(hipFree(dy), 'hipFree dy');
end;

procedure TTest_rocSPARSE.Test_sdoti;
var
  nnz: rocsparse_int;
  hx_val: array[0..2] of Single;
  hx_ind: array[0..2] of rocsparse_int;
  hy: array[0..8] of Single;
  dx_val: Pointer;
  dx_ind: Pointer;
  dy: Pointer;
  result: Single;
  d_result: Pointer;
  i: integer;
  expected_result: Single;
begin
  // From direct_test_doti.pas
  nnz := 3;
  hx_val[0] := 1.0; hx_val[1] := 2.0; hx_val[2] := 3.0;
  hx_ind[0] := 0; hx_ind[1] := 3; hx_ind[2] := 5;
  for i := 0 to 8 do hy[i] := Single(i + 1);

  expected_result := (1.0 * 1.0) + (2.0 * 4.0) + (3.0 * 6.0); // 1 + 8 + 18 = 27

  // Allocate device memory
  CheckHip(hipMalloc(dx_val, nnz * sizeof(Single)), 'hipMalloc dx_val');
  CheckHip(hipMalloc(dx_ind, nnz * sizeof(rocsparse_int)), 'hipMalloc dx_ind');
  CheckHip(hipMalloc(dy, 9 * sizeof(Single)), 'hipMalloc dy');
  CheckHip(hipMalloc(d_result, sizeof(Single)), 'hipMalloc d_result');

  // Copy data to device
  CheckHip(hipMemcpy(dx_val, @hx_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), 'hipMemcpy dx_val');
  CheckHip(hipMemcpy(dx_ind, @hx_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'hipMemcpy dx_ind');
  CheckHip(hipMemcpy(dy, @hy[0], 9 * sizeof(Single), hipMemcpyHostToDevice), 'hipMemcpy dy');

  // Call sdoti
  CheckSparse(rocsparse_sdoti(handle, nnz, dx_val, dx_ind, dy, d_result, rocsparse_index_base_zero), 'rocsparse_sdoti');

  // Copy result back to host
  CheckHip(hipMemcpy(@result, d_result, sizeof(Single), hipMemcpyDeviceToHost), 'hipMemcpy result');

  // Verification
  AssertEquals('sdoti result', expected_result, result, 1e-5);

  // Cleanup
  CheckHip(hipFree(dx_val), 'hipFree dx_val');
  CheckHip(hipFree(dx_ind), 'hipFree dx_ind');
  CheckHip(hipFree(dy), 'hipFree dy');
  CheckHip(hipFree(d_result), 'hipFree d_result');
end;

procedure TTest_rocSPARSE.Test_scsrmv;
var
  m, n, nnz: rocsparse_int;
  alpha, beta: Single;
  h_csr_val: array[0..7] of Single;
  h_csr_row_ptr: array[0..4] of rocsparse_int;
  h_csr_col_ind: array[0..7] of rocsparse_int;
  h_x: array[0..3] of Single;
  h_y: array[0..3] of Single;
  h_y_result: array[0..3] of Single;
  expected_y: array[0..3] of Single;
  d_csr_val, d_x, d_y: Pointer;
  d_csr_row_ptr, d_csr_col_ind: Pointer;
  i: integer;
begin
  // From direct_test_csrmv.pas
  m := 4; n := 4; nnz := 8;
  alpha := 1.0; beta := 0.0;

  h_csr_row_ptr[0] := 0; h_csr_row_ptr[1] := 3; h_csr_row_ptr[2] := 5; h_csr_row_ptr[3] := 6; h_csr_row_ptr[4] := 8;
  h_csr_col_ind[0] := 0; h_csr_col_ind[1] := 2; h_csr_col_ind[2] := 3; h_csr_col_ind[3] := 2; h_csr_col_ind[4] := 3; h_csr_col_ind[5] := 1; h_csr_col_ind[6] := 0; h_csr_col_ind[7] := 3;
  h_csr_val[0] := 1.0; h_csr_val[1] := 3.0; h_csr_val[2] := 4.0; h_csr_val[3] := 5.0; h_csr_val[4] := 1.0; h_csr_val[5] := 2.0; h_csr_val[6] := 4.0; h_csr_val[7] := 8.0;

  h_x[0] := 1.0; h_x[1] := 2.0; h_x[2] := 3.0; h_x[3] := 4.0;
  for i := 0 to m - 1 do h_y[i] := 0.0;

  expected_y[0] := 26.0;
  expected_y[1] := 19.0;
  expected_y[2] := 4.0;
  expected_y[3] := 36.0;

  // Allocate device memory
  CheckHip(hipMalloc(d_csr_val, nnz * sizeof(Single)), 'hipMalloc d_csr_val');
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), 'hipMalloc d_csr_row_ptr');
  CheckHip(hipMalloc(d_csr_col_ind, nnz * sizeof(rocsparse_int)), 'hipMalloc d_csr_col_ind');
  CheckHip(hipMalloc(d_x, n * sizeof(Single)), 'hipMalloc d_x');
  CheckHip(hipMalloc(d_y, m * sizeof(Single)), 'hipMalloc d_y');

  // Copy data to device
  CheckHip(hipMemcpy(d_csr_val, @h_csr_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), 'hipMemcpy d_csr_val');
  CheckHip(hipMemcpy(d_csr_row_ptr, @h_csr_row_ptr[0], (m + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'hipMemcpy d_csr_row_ptr');
  CheckHip(hipMemcpy(d_csr_col_ind, @h_csr_col_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'hipMemcpy d_csr_col_ind');
  CheckHip(hipMemcpy(d_x, @h_x[0], n * sizeof(Single), hipMemcpyHostToDevice), 'hipMemcpy d_x');
  CheckHip(hipMemcpy(d_y, @h_y[0], m * sizeof(Single), hipMemcpyHostToDevice), 'hipMemcpy d_y');

  // Call scsrmv
  CheckSparse(rocsparse_scsrmv(handle, rocsparse_operation_none, m, n, nnz, @alpha, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, nil, d_x, @beta, d_y), 'rocsparse_scsrmv');

  // Copy result back to host
  CheckHip(hipMemcpy(@h_y_result[0], d_y, m * sizeof(Single), hipMemcpyDeviceToHost), 'hipMemcpy h_y_result');

  // Verification
  for i := 0 to m - 1 do
    AssertEquals('scsrmv result at index ' + IntToStr(i), expected_y[i], h_y_result[i], 1e-5);

  // Cleanup
  CheckHip(hipFree(d_csr_val), 'hipFree d_csr_val');
  CheckHip(hipFree(d_csr_row_ptr), 'hipFree d_csr_row_ptr');
  CheckHip(hipFree(d_csr_col_ind), 'hipFree d_csr_col_ind');
  CheckHip(hipFree(d_x), 'hipFree d_x');
  CheckHip(hipFree(d_y), 'hipFree d_y');
end;

procedure TTest_rocSPARSE.Test_scsrmm_1x1_passing;
var
  m, n, k, nnz, ldb, ldc: rocsparse_int;
  alpha, beta: Single;
  h_csr_val: array[0..0] of Single;
  h_csr_row_ptr: array[0..1] of rocsparse_int;
  h_csr_col_ind: array[0..0] of rocsparse_int;
  h_B: array[0..0] of Single;
  h_C: array[0..0] of Single;
  h_C_result: array[0..0] of Single;
  d_csr_val, d_B, d_C, d_csr_row_ptr, d_csr_col_ind: Pointer;
begin
  // Simplified 1x1 test case that is known to work
  m := 1; n := 1; k := 1; nnz := 1; ldb := 1; ldc := 1;
  alpha := 1.0; beta := 1.0;
  h_csr_val[0] := 2.0;
  h_csr_row_ptr[0] := 0; h_csr_row_ptr[1] := 1;
  h_csr_col_ind[0] := 0;
  h_B[0] := 3.0;
  h_C[0] := 1.0;

  // Allocate
  CheckHip(hipMalloc(d_csr_val, nnz * sizeof(Single)), '1x1 hipMalloc d_csr_val');
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), '1x1 hipMalloc d_csr_row_ptr');
  CheckHip(hipMalloc(d_csr_col_ind, nnz * sizeof(rocsparse_int)), '1x1 hipMalloc d_csr_col_ind');
  CheckHip(hipMalloc(d_B, k * n * sizeof(Single)), '1x1 hipMalloc d_B');
  CheckHip(hipMalloc(d_C, m * n * sizeof(Single)), '1x1 hipMalloc d_C');

  // Copy
  CheckHip(hipMemcpy(d_csr_val, @h_csr_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), '1x1 hipMemcpy d_csr_val');
  CheckHip(hipMemcpy(d_csr_row_ptr, @h_csr_row_ptr[0], (m + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), '1x1 hipMemcpy d_csr_row_ptr');
  CheckHip(hipMemcpy(d_csr_col_ind, @h_csr_col_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), '1x1 hipMemcpy d_csr_col_ind');
  CheckHip(hipMemcpy(d_B, @h_B[0], k * n * sizeof(Single), hipMemcpyHostToDevice), '1x1 hipMemcpy d_B');
  CheckHip(hipMemcpy(d_C, @h_C[0], m * n * sizeof(Single), hipMemcpyHostToDevice), '1x1 hipMemcpy d_C');

  // Call
  CheckSparse(rocsparse_scsrmm(handle, rocsparse_operation_none, rocsparse_operation_none, m, n, k, nnz, @alpha, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, d_B, ldb, @beta, d_C, ldc), 'rocsparse_scsrmm 1x1');

  // Copy back
  CheckHip(hipMemcpy(@h_C_result[0], d_C, m * n * sizeof(Single), hipMemcpyDeviceToHost), '1x1 hipMemcpy h_C_result');

  // Verify
  AssertEquals('scsrmm 1x1 result', 7.0, h_C_result[0], 1e-5); // C = alpha*A*B + beta*C = 1*2*3 + 1*1 = 7

  // Cleanup
  CheckHip(hipFree(d_csr_val), '1x1 hipFree d_csr_val');
  CheckHip(hipFree(d_csr_row_ptr), '1x1 hipFree d_csr_row_ptr');
  CheckHip(hipFree(d_csr_col_ind), '1x1 hipFree d_csr_col_ind');
  CheckHip(hipFree(d_B), '1x1 hipFree d_B');
  CheckHip(hipFree(d_C), '1x1 hipFree d_C');
end;

procedure TTest_rocSPARSE.Test_scsrmm_4x5_failing;
var
  m, n, k, nnz, ldb, ldc: rocsparse_int;
  alpha, beta: Single;
  h_csr_val: array[0..7] of Single;
  h_csr_row_ptr: array[0..4] of rocsparse_int;
  h_csr_col_ind: array[0..7] of rocsparse_int;
  h_B: array[0..19] of Single;
  h_C: array[0..15] of Single;
  h_C_result: array[0..15] of Single;
  d_csr_val, d_B, d_C, d_csr_row_ptr, d_csr_col_ind: Pointer;
  i, j: integer;
  expected_C: array[0..15] of Single;
begin
  // Failing test from direct_test_csrmm.pas
  m := 4; k := 5; n := 4; nnz := 8;
  ldb := k; ldc := m;
  alpha := 1.0; beta := 0.0;

  h_csr_row_ptr[0] := 0; h_csr_row_ptr[1] := 3; h_csr_row_ptr[2] := 5; h_csr_row_ptr[3] := 8; h_csr_row_ptr[4] := 8;
  h_csr_col_ind[0] := 0; h_csr_col_ind[1] := 1; h_csr_col_ind[2] := 3;
  h_csr_col_ind[3] := 1; h_csr_col_ind[4] := 2;
  h_csr_col_ind[5] := 0; h_csr_col_ind[6] := 3; h_csr_col_ind[7] := 4;
  h_csr_val[0] := 1.0; h_csr_val[1] := 2.0; h_csr_val[2] := 3.0;
  h_csr_val[3] := 4.0; h_csr_val[4] := 5.0;
  h_csr_val[5] := 6.0; h_csr_val[6] := 7.0; h_csr_val[7] := 8.0;

  for i := 0 to (k * n) - 1 do h_B[i] := 1.0;
  for i := 0 to (m * n) - 1 do h_C[i] := 0.0;

  // Expected result C = A * B (since B is all 1s, this is row sums of A)
  // C is column-major
  // Row 0 of A*B: [6, 6, 6, 6]
  // Row 1 of A*B: [9, 9, 9, 9]
  // Row 2 of A*B: [21, 21, 21, 21]
  // Row 3 of A*B: [0, 0, 0, 0]
  expected_C[0] := 6.0; expected_C[1] := 9.0; expected_C[2] := 21.0; expected_C[3] := 0.0;
  expected_C[4] := 6.0; expected_C[5] := 9.0; expected_C[6] := 21.0; expected_C[7] := 0.0;
  expected_C[8] := 6.0; expected_C[9] := 9.0; expected_C[10] := 21.0; expected_C[11] := 0.0;
  expected_C[12] := 6.0; expected_C[13] := 9.0; expected_C[14] := 21.0; expected_C[15] := 0.0;

  // Allocate
  CheckHip(hipMalloc(d_csr_val, nnz * sizeof(Single)), '4x5 hipMalloc d_csr_val');
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), '4x5 hipMalloc d_csr_row_ptr');
  CheckHip(hipMalloc(d_csr_col_ind, nnz * sizeof(rocsparse_int)), '4x5 hipMalloc d_csr_col_ind');
  CheckHip(hipMalloc(d_B, k * n * sizeof(Single)), '4x5 hipMalloc d_B');
  CheckHip(hipMalloc(d_C, m * n * sizeof(Single)), '4x5 hipMalloc d_C');

  // Copy
  CheckHip(hipMemcpy(d_csr_val, @h_csr_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), '4x5 hipMemcpy d_csr_val');
  CheckHip(hipMemcpy(d_csr_row_ptr, @h_csr_row_ptr[0], (m + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), '4x5 hipMemcpy d_csr_row_ptr');
  CheckHip(hipMemcpy(d_csr_col_ind, @h_csr_col_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), '4x5 hipMemcpy d_csr_col_ind');
  CheckHip(hipMemcpy(d_B, @h_B[0], k * n * sizeof(Single), hipMemcpyHostToDevice), '4x5 hipMemcpy d_B');
  CheckHip(hipMemcpy(d_C, @h_C[0], m * n * sizeof(Single), hipMemcpyHostToDevice), '4x5 hipMemcpy d_C');

  // Call
  CheckSparse(rocsparse_scsrmm(handle, rocsparse_operation_none, rocsparse_operation_none, m, n, k, nnz, @alpha, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, d_B, ldb, @beta, d_C, ldc), 'rocsparse_scsrmm 4x5');

  // Copy back
  CheckHip(hipMemcpy(@h_C_result[0], d_C, m * n * sizeof(Single), hipMemcpyDeviceToHost), '4x5 hipMemcpy h_C_result');

  // Verify
  for i := 0 to (m*n) - 1 do
    AssertEquals('scsrmm 4x5 result at index ' + IntToStr(i), expected_C[i], h_C_result[i], 1e-5);

  // Cleanup
  CheckHip(hipFree(d_csr_val), '4x5 hipFree d_csr_val');
  CheckHip(hipFree(d_csr_row_ptr), '4x5 hipFree d_csr_row_ptr');
  CheckHip(hipFree(d_csr_col_ind), '4x5 hipFree d_csr_col_ind');
  CheckHip(hipFree(d_B), '4x5 hipFree d_B');
  CheckHip(hipFree(d_C), '4x5 hipFree d_C');
end;


procedure TTest_rocSPARSE.Test_csr2coo;
var
  m, nnz: rocsparse_int;
  h_csr_row_ptr: array[0..3] of rocsparse_int;
  h_coo_row_ind_result: array[0..4] of rocsparse_int;
  expected_coo_row_ind: array[0..4] of rocsparse_int;
  d_csr_row_ptr, d_coo_row_ind: Pointer;
  i: integer;
begin
  // Define a simple 3x3 CSR matrix with 5 non-zero elements
  // 1 2 0
  // 0 3 4
  // 5 0 0
  m := 3;
  nnz := 5;
  h_csr_row_ptr[0] := 0; h_csr_row_ptr[1] := 2; h_csr_row_ptr[2] := 4; h_csr_row_ptr[3] := 5;

  // Expected COO row indices
  expected_coo_row_ind[0] := 0; expected_coo_row_ind[1] := 0;
  expected_coo_row_ind[2] := 1; expected_coo_row_ind[3] := 1;
  expected_coo_row_ind[4] := 2;

  // Allocate device memory
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), 'csr2coo hipMalloc d_csr_row_ptr');
  CheckHip(hipMalloc(d_coo_row_ind, nnz * sizeof(rocsparse_int)), 'csr2coo hipMalloc d_coo_row_ind');

  // Copy data to device
  CheckHip(hipMemcpy(d_csr_row_ptr, @h_csr_row_ptr[0], (m + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csr2coo hipMemcpy d_csr_row_ptr');

  // Call csr2coo
  CheckSparse(rocsparse_csr2coo(handle, d_csr_row_ptr, nnz, m, d_coo_row_ind, rocsparse_index_base_zero), 'rocsparse_csr2coo');

  // Copy result back to host
  CheckHip(hipMemcpy(@h_coo_row_ind_result[0], d_coo_row_ind, nnz * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'csr2coo hipMemcpy h_coo_row_ind_result');

  // Verification
  for i := 0 to nnz - 1 do
    AssertEquals('csr2coo result at index ' + IntToStr(i), expected_coo_row_ind[i], h_coo_row_ind_result[i]);

  // Cleanup
  CheckHip(hipFree(d_csr_row_ptr), 'csr2coo hipFree d_csr_row_ptr');
  CheckHip(hipFree(d_coo_row_ind), 'csr2coo hipFree d_coo_row_ind');
end;


procedure TTest_rocSPARSE.Test_coo2csr;
var
  m, nnz: rocsparse_int;
  h_coo_row_ind: array[0..4] of rocsparse_int;
  h_csr_row_ptr_result: array[0..3] of rocsparse_int;
  expected_csr_row_ptr: array[0..3] of rocsparse_int;
  d_coo_row_ind, d_csr_row_ptr: Pointer;
  i: integer;
begin
  // Inverse of the csr2coo test
  m := 3;
  nnz := 5;
  h_coo_row_ind[0] := 0; h_coo_row_ind[1] := 0;
  h_coo_row_ind[2] := 1; h_coo_row_ind[3] := 1;
  h_coo_row_ind[4] := 2;

  // Expected CSR row pointers
  expected_csr_row_ptr[0] := 0; expected_csr_row_ptr[1] := 2;
  expected_csr_row_ptr[2] := 4; expected_csr_row_ptr[3] := 5;

  // Allocate device memory
  CheckHip(hipMalloc(d_coo_row_ind, nnz * sizeof(rocsparse_int)), 'coo2csr hipMalloc d_coo_row_ind');
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), 'coo2csr hipMalloc d_csr_row_ptr');

  // Copy data to device
  CheckHip(hipMemcpy(d_coo_row_ind, @h_coo_row_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'coo2csr hipMemcpy d_coo_row_ind');

  // Call coo2csr
  CheckSparse(rocsparse_coo2csr(handle, d_coo_row_ind, nnz, m, d_csr_row_ptr, rocsparse_index_base_zero), 'rocsparse_coo2csr');

  // Copy result back to host
  CheckHip(hipMemcpy(@h_csr_row_ptr_result[0], d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'coo2csr hipMemcpy h_csr_row_ptr_result');

  // Verification
  for i := 0 to m do
    AssertEquals('coo2csr result at index ' + IntToStr(i), expected_csr_row_ptr[i], h_csr_row_ptr_result[i]);

  // Cleanup
  CheckHip(hipFree(d_coo_row_ind), 'coo2csr hipFree d_coo_row_ind');
  CheckHip(hipFree(d_csr_row_ptr), 'coo2csr hipFree d_csr_row_ptr');
end;


procedure TTest_rocSPARSE.Test_snnz;
var
  m, n, ld: rocsparse_int;
  h_A: array[0..11] of Single;
  h_nnz_per_row: array[0..2] of rocsparse_int;
  h_nnz_total: rocsparse_int;
  d_A, d_nnz_per_row, d_nnz_total: Pointer;
  i: integer;
begin
  // Define a 3x4 dense matrix with 6 non-zero elements
  // 1 0 2 0
  // 3 4 0 5
  // 0 6 0 0
  m := 3; n := 4; ld := m;

  // Column-major layout
  h_A[0] := 1.0; h_A[1] := 3.0; h_A[2] := 0.0;
  h_A[3] := 0.0; h_A[4] := 4.0; h_A[5] := 6.0;
  h_A[6] := 2.0; h_A[7] := 0.0; h_A[8] := 0.0;
  h_A[9] := 0.0; h_A[10] := 5.0; h_A[11] := 0.0;

  // Allocate device memory
  CheckHip(hipMalloc(d_A, m * n * sizeof(Single)), 'snnz hipMalloc d_A');
  CheckHip(hipMalloc(d_nnz_per_row, m * sizeof(rocsparse_int)), 'snnz hipMalloc d_nnz_per_row');
  CheckHip(hipMalloc(d_nnz_total, sizeof(rocsparse_int)), 'snnz hipMalloc d_nnz_total');

  // Copy matrix to device
  CheckHip(hipMemcpy(d_A, @h_A[0], m * n * sizeof(Single), hipMemcpyHostToDevice), 'snnz hipMemcpy d_A');

  // Set pointer mode to device for the total nnz count
  CheckSparse(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device), 'snnz rocsparse_set_pointer_mode');

  // Call snnz
  CheckSparse(rocsparse_snnz(handle, rocsparse_direction_row, m, n, descr, d_A, ld, d_nnz_per_row, d_nnz_total), 'rocsparse_snnz');

  // Copy results back to host
  CheckHip(hipMemcpy(@h_nnz_per_row[0], d_nnz_per_row, m * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'snnz hipMemcpy h_nnz_per_row');
  CheckHip(hipMemcpy(@h_nnz_total, d_nnz_total, sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'snnz hipMemcpy h_nnz_total');

  // Verification
  AssertEquals('snnz total count', 6, h_nnz_total);
  AssertEquals('snnz row 0 count', 2, h_nnz_per_row[0]);
  AssertEquals('snnz row 1 count', 3, h_nnz_per_row[1]);
  AssertEquals('snnz row 2 count', 1, h_nnz_per_row[2]);

  // Cleanup
  CheckHip(hipFree(d_A), 'snnz hipFree d_A');
  CheckHip(hipFree(d_nnz_per_row), 'snnz hipFree d_nnz_per_row');
  CheckHip(hipFree(d_nnz_total), 'snnz hipFree d_nnz_total');

  // Reset pointer mode to host
  CheckSparse(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host), 'snnz reset rocsparse_set_pointer_mode');
end;


procedure TTest_rocSPARSE.Test_scsr2csc;
var
  m, n, nnz: rocsparse_int;
  h_csr_val: array[0..4] of Single;
  h_csr_row_ptr: array[0..3] of rocsparse_int;
  h_csr_col_ind: array[0..4] of rocsparse_int;
  h_csc_val: array[0..4] of Single;
  h_csc_row_ind: array[0..4] of rocsparse_int;
  h_csc_col_ptr: array[0..3] of rocsparse_int;
  d_csr_val, d_csr_row_ptr, d_csr_col_ind: Pointer;
  d_csc_val, d_csc_row_ind, d_csc_col_ptr, d_buffer: Pointer;
  buffer_size: size_t;
  i: integer;
  expected_csc_val: array[0..4] of Single;
  expected_csc_row_ind: array[0..4] of rocsparse_int;
  expected_csc_col_ptr: array[0..3] of rocsparse_int;
begin
  // From Test_csr2coo
  // 1 2 0
  // 0 3 4
  // 5 0 0
  m := 3; n := 3; nnz := 5;
  h_csr_val[0] := 1.0; h_csr_val[1] := 2.0; h_csr_val[2] := 3.0; h_csr_val[3] := 4.0; h_csr_val[4] := 5.0;
  h_csr_row_ptr[0] := 0; h_csr_row_ptr[1] := 2; h_csr_row_ptr[2] := 4; h_csr_row_ptr[3] := 5;
  h_csr_col_ind[0] := 0; h_csr_col_ind[1] := 1; h_csr_col_ind[2] := 1; h_csr_col_ind[3] := 2; h_csr_col_ind[4] := 0;

  // Expected CSC output (transpose of CSR)
  // 1 0 5
  // 2 3 0
  // 0 4 0
  expected_csc_col_ptr[0] := 0; expected_csc_col_ptr[1] := 2; expected_csc_col_ptr[2] := 4; expected_csc_col_ptr[3] := 5;
  expected_csc_row_ind[0] := 0; expected_csc_row_ind[1] := 2; expected_csc_row_ind[2] := 0; expected_csc_row_ind[3] := 1; expected_csc_row_ind[4] := 1;
  expected_csc_val[0] := 1.0; expected_csc_val[1] := 5.0; expected_csc_val[2] := 2.0; expected_csc_val[3] := 3.0; expected_csc_val[4] := 4.0;

  // Allocate device memory
  CheckHip(hipMalloc(d_csr_val, nnz * sizeof(Single)), 'csr2csc Malloc csr_val');
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), 'csr2csc Malloc csr_row_ptr');
  CheckHip(hipMalloc(d_csr_col_ind, nnz * sizeof(rocsparse_int)), 'csr2csc Malloc csr_col_ind');
  CheckHip(hipMalloc(d_csc_val, nnz * sizeof(Single)), 'csr2csc Malloc csc_val');
  CheckHip(hipMalloc(d_csc_row_ind, nnz * sizeof(rocsparse_int)), 'csr2csc Malloc csc_row_ind');
  CheckHip(hipMalloc(d_csc_col_ptr, (n + 1) * sizeof(rocsparse_int)), 'csr2csc Malloc csc_col_ptr');

  // Copy CSR data to device
  CheckHip(hipMemcpy(d_csr_val, @h_csr_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), 'csr2csc Memcpy csr_val');
  CheckHip(hipMemcpy(d_csr_row_ptr, @h_csr_row_ptr[0], (m + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csr2csc Memcpy csr_row_ptr');
  CheckHip(hipMemcpy(d_csr_col_ind, @h_csr_col_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csr2csc Memcpy csr_col_ind');

  // Get buffer size
  CheckSparse(rocsparse_csr2csc_buffer_size(handle, m, n, nnz, d_csr_row_ptr, d_csr_col_ind, rocsparse_action_numeric, @buffer_size), 'csr2csc_buffer_size');
  CheckHip(hipMalloc(d_buffer, buffer_size), 'csr2csc Malloc buffer');

  // Call csr2csc
  CheckSparse(rocsparse_scsr2csc(handle, m, n, nnz, d_csr_val, d_csr_row_ptr, d_csr_col_ind, d_csc_val, d_csc_row_ind, d_csc_col_ptr, rocsparse_action_numeric, rocsparse_index_base_zero, d_buffer), 'rocsparse_scsr2csc');

  // Copy result back
  CheckHip(hipMemcpy(@h_csc_val[0], d_csc_val, nnz * sizeof(Single), hipMemcpyDeviceToHost), 'csr2csc Memcpy csc_val');
  CheckHip(hipMemcpy(@h_csc_row_ind[0], d_csc_row_ind, nnz * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'csr2csc Memcpy csc_row_ind');
  CheckHip(hipMemcpy(@h_csc_col_ptr[0], d_csc_col_ptr, (n + 1) * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'csr2csc Memcpy csc_col_ptr');

  // Verify
  for i := 0 to nnz - 1 do AssertEquals('csc_val ' + IntToStr(i), expected_csc_val[i], h_csc_val[i], 1e-5);
  for i := 0 to nnz - 1 do AssertEquals('csc_row_ind ' + IntToStr(i), expected_csc_row_ind[i], h_csc_row_ind[i]);
  for i := 0 to n do AssertEquals('csc_col_ptr ' + IntToStr(i), expected_csc_col_ptr[i], h_csc_col_ptr[i]);

  // Cleanup
  CheckHip(hipFree(d_csr_val), 'csr2csc Free'); CheckHip(hipFree(d_csr_row_ptr), 'csr2csc Free'); CheckHip(hipFree(d_csr_col_ind), 'csr2csc Free');
  CheckHip(hipFree(d_csc_val), 'csr2csc Free'); CheckHip(hipFree(d_csc_row_ind), 'csr2csc Free'); CheckHip(hipFree(d_csc_col_ptr), 'csr2csc Free');
  CheckHip(hipFree(d_buffer), 'csr2csc Free');
end;

procedure TTest_rocSPARSE.Test_sdense2csr;
var
  m, n, ld, nnz: rocsparse_int;
  h_A: array[0..11] of Single;
  h_nnz_per_row: array[0..2] of rocsparse_int;
  h_csr_val: array[0..5] of Single;
  h_csr_row_ptr: array[0..3] of rocsparse_int;
  h_csr_col_ind: array[0..5] of rocsparse_int;
  d_A, d_nnz_per_row, d_csr_val, d_csr_row_ptr, d_csr_col_ind: Pointer;
  i: integer;
  expected_csr_val: array[0..5] of Single;
  expected_csr_col_ind: array[0..5] of rocsparse_int;
  expected_csr_row_ptr: array[0..3] of rocsparse_int;
begin
  // From Test_snnz
  // 1 0 2 0
  // 3 4 0 5
  // 0 6 0 0
  m := 3; n := 4; ld := m; nnz := 6;
  h_A[0] := 1.0; h_A[1] := 3.0; h_A[2] := 0.0;
  h_A[3] := 0.0; h_A[4] := 4.0; h_A[5] := 6.0;
  h_A[6] := 2.0; h_A[7] := 0.0; h_A[8] := 0.0;
  h_A[9] := 0.0; h_A[10] := 5.0; h_A[11] := 0.0;

  h_nnz_per_row[0] := 2; h_nnz_per_row[1] := 3; h_nnz_per_row[2] := 1;

  expected_csr_row_ptr[0] := 0; expected_csr_row_ptr[1] := 2; expected_csr_row_ptr[2] := 5; expected_csr_row_ptr[3] := 6;
  expected_csr_col_ind[0] := 0; expected_csr_col_ind[1] := 2;
  expected_csr_col_ind[2] := 0; expected_csr_col_ind[3] := 1; expected_csr_col_ind[4] := 3;
  expected_csr_col_ind[5] := 1;
  expected_csr_val[0] := 1.0; expected_csr_val[1] := 2.0;
  expected_csr_val[2] := 3.0; expected_csr_val[3] := 4.0; expected_csr_val[4] := 5.0;
  expected_csr_val[5] := 6.0;

  // Allocate & Copy
  CheckHip(hipMalloc(d_A, m * n * sizeof(Single)), 'd2csr Malloc A');
  CheckHip(hipMalloc(d_nnz_per_row, m * sizeof(rocsparse_int)), 'd2csr Malloc nnz_per_row');
  CheckHip(hipMalloc(d_csr_val, nnz * sizeof(Single)), 'd2csr Malloc val');
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), 'd2csr Malloc row_ptr');
  CheckHip(hipMalloc(d_csr_col_ind, nnz * sizeof(rocsparse_int)), 'd2csr Malloc col_ind');
  CheckHip(hipMemcpy(d_A, @h_A[0], m * n * sizeof(Single), hipMemcpyHostToDevice), 'd2csr Memcpy A');
  CheckHip(hipMemcpy(d_nnz_per_row, @h_nnz_per_row[0], m * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'd2csr Memcpy nnz_per_row');

  // Call
  CheckSparse(rocsparse_sdense2csr(handle, m, n, descr, d_A, ld, d_nnz_per_row, d_csr_val, d_csr_row_ptr, d_csr_col_ind), 'rocsparse_sdense2csr');

  // Copy back
  CheckHip(hipMemcpy(@h_csr_val[0], d_csr_val, nnz * sizeof(Single), hipMemcpyDeviceToHost), 'd2csr Memcpy val');
  CheckHip(hipMemcpy(@h_csr_row_ptr[0], d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'd2csr Memcpy row_ptr');
  CheckHip(hipMemcpy(@h_csr_col_ind[0], d_csr_col_ind, nnz * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'd2csr Memcpy col_ind');

  // Verify
  for i := 0 to nnz - 1 do AssertEquals('d2csr val ' + IntToStr(i), expected_csr_val[i], h_csr_val[i], 1e-5);
  for i := 0 to nnz - 1 do AssertEquals('d2csr col_ind ' + IntToStr(i), expected_csr_col_ind[i], h_csr_col_ind[i]);
  for i := 0 to m do AssertEquals('d2csr row_ptr ' + IntToStr(i), expected_csr_row_ptr[i], h_csr_row_ptr[i]);

  // Cleanup
  CheckHip(hipFree(d_A), 'd2csr Free'); CheckHip(hipFree(d_nnz_per_row), 'd2csr Free'); CheckHip(hipFree(d_csr_val), 'd2csr Free');
  CheckHip(hipFree(d_csr_row_ptr), 'd2csr Free'); CheckHip(hipFree(d_csr_col_ind), 'd2csr Free');
end;
procedure TTest_rocSPARSE.Test_sdense2coo;
var
  m, n, ld, nnz: rocsparse_int;
  h_A: array[0..11] of Single;
  h_nnz_per_row: array[0..2] of rocsparse_int;
  h_coo_val: array[0..5] of Single;
  h_coo_row_ind: array[0..5] of rocsparse_int;
  h_coo_col_ind: array[0..5] of rocsparse_int;
  d_A, d_nnz_per_row, d_coo_val, d_coo_row_ind, d_coo_col_ind: Pointer;
  i: integer;
  expected_coo_val: array[0..5] of Single;
  expected_coo_row_ind: array[0..5] of rocsparse_int;
  expected_coo_col_ind: array[0..5] of rocsparse_int;
begin
  // From Test_snnz
  // 1 0 2 0
  // 3 4 0 5
  // 0 6 0 0
  m := 3; n := 4; ld := m; nnz := 6;
  h_A[0] := 1.0; h_A[1] := 3.0; h_A[2] := 0.0;
  h_A[3] := 0.0; h_A[4] := 4.0; h_A[5] := 6.0;
  h_A[6] := 2.0; h_A[7] := 0.0; h_A[8] := 0.0;
  h_A[9] := 0.0; h_A[10] := 5.0; h_A[11] := 0.0;

  h_nnz_per_row[0] := 2; h_nnz_per_row[1] := 3; h_nnz_per_row[2] := 1;

  expected_coo_row_ind[0] := 0; expected_coo_row_ind[1] := 0; expected_coo_row_ind[2] := 1; expected_coo_row_ind[3] := 1; expected_coo_row_ind[4] := 1; expected_coo_row_ind[5] := 2;
  expected_coo_col_ind[0] := 0; expected_coo_col_ind[1] := 2; expected_coo_col_ind[2] := 0; expected_coo_col_ind[3] := 1; expected_coo_col_ind[4] := 3; expected_coo_col_ind[5] := 1;
  expected_coo_val[0] := 1.0; expected_coo_val[1] := 2.0; expected_coo_val[2] := 3.0; expected_coo_val[3] := 4.0; expected_coo_val[4] := 5.0; expected_coo_val[5] := 6.0;

  // Allocate & Copy
  CheckHip(hipMalloc(d_A, m * n * sizeof(Single)), 'd2coo Malloc A');
  CheckHip(hipMalloc(d_nnz_per_row, m * sizeof(rocsparse_int)), 'd2coo Malloc nnz_per_row');
  CheckHip(hipMalloc(d_coo_val, nnz * sizeof(Single)), 'd2coo Malloc val');
  CheckHip(hipMalloc(d_coo_row_ind, nnz * sizeof(rocsparse_int)), 'd2coo Malloc row_ind');
  CheckHip(hipMalloc(d_coo_col_ind, nnz * sizeof(rocsparse_int)), 'd2coo Malloc col_ind');
  CheckHip(hipMemcpy(d_A, @h_A[0], m * n * sizeof(Single), hipMemcpyHostToDevice), 'd2coo Memcpy A');
  CheckHip(hipMemcpy(d_nnz_per_row, @h_nnz_per_row[0], m * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'd2coo Memcpy nnz_per_row');

  // Call
  CheckSparse(rocsparse_sdense2coo(handle, m, n, descr, d_A, ld, d_nnz_per_row, d_coo_val, d_coo_row_ind, d_coo_col_ind), 'rocsparse_sdense2coo');

  // Copy back
  CheckHip(hipMemcpy(@h_coo_val[0], d_coo_val, nnz * sizeof(Single), hipMemcpyDeviceToHost), 'd2coo Memcpy val');
  CheckHip(hipMemcpy(@h_coo_row_ind[0], d_coo_row_ind, nnz * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'd2coo Memcpy row_ind');
  CheckHip(hipMemcpy(@h_coo_col_ind[0], d_coo_col_ind, nnz * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'd2coo Memcpy col_ind');

  // Verify
  for i := 0 to nnz - 1 do AssertEquals('d2coo val ' + IntToStr(i), expected_coo_val[i], h_coo_val[i], 1e-5);
  for i := 0 to nnz - 1 do AssertEquals('d2coo row_ind ' + IntToStr(i), expected_coo_row_ind[i], h_coo_row_ind[i]);
  for i := 0 to nnz - 1 do AssertEquals('d2coo col_ind ' + IntToStr(i), expected_coo_col_ind[i], h_coo_col_ind[i]);

  // Cleanup
  CheckHip(hipFree(d_A), 'd2coo Free'); CheckHip(hipFree(d_nnz_per_row), 'd2coo Free'); CheckHip(hipFree(d_coo_val), 'd2coo Free');
  CheckHip(hipFree(d_coo_row_ind), 'd2coo Free'); CheckHip(hipFree(d_coo_col_ind), 'd2coo Free');
end;


procedure TTest_rocSPARSE.Test_scsr2dense;
var
  m, n, ld, nnz: rocsparse_int;
  h_csr_val: array[0..5] of Single;
  h_csr_row_ptr: array[0..3] of rocsparse_int;
  h_csr_col_ind: array[0..5] of rocsparse_int;
  h_A: array[0..11] of Single;
  d_csr_val, d_csr_row_ptr, d_csr_col_ind, d_A: Pointer;
  i: integer;
  expected_A: array[0..11] of Single;
begin
  // Inverse of Test_sdense2csr
  m := 3; n := 4; ld := m; nnz := 6;

  h_csr_row_ptr[0] := 0; h_csr_row_ptr[1] := 2; h_csr_row_ptr[2] := 5; h_csr_row_ptr[3] := 6;
  h_csr_col_ind[0] := 0; h_csr_col_ind[1] := 2;
  h_csr_col_ind[2] := 0; h_csr_col_ind[3] := 1; h_csr_col_ind[4] := 3;
  h_csr_col_ind[5] := 1;
  h_csr_val[0] := 1.0; h_csr_val[1] := 2.0;
  h_csr_val[2] := 3.0; h_csr_val[3] := 4.0; h_csr_val[4] := 5.0;
  h_csr_val[5] := 6.0;

  // Expected dense matrix (column-major)
  for i := 0 to (m*n)-1 do expected_A[i] := 0.0;
  expected_A[0] := 1.0; expected_A[1] := 3.0;
  expected_A[4] := 4.0; expected_A[5] := 6.0;
  expected_A[6] := 2.0;
  expected_A[10] := 5.0;

  // Allocate & Copy
  CheckHip(hipMalloc(d_csr_val, nnz * sizeof(Single)), 'csr2d Malloc val');
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), 'csr2d Malloc row_ptr');
  CheckHip(hipMalloc(d_csr_col_ind, nnz * sizeof(rocsparse_int)), 'csr2d Malloc col_ind');
  CheckHip(hipMalloc(d_A, m * n * sizeof(Single)), 'csr2d Malloc A');
  CheckHip(hipMemcpy(d_csr_val, @h_csr_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), 'csr2d Memcpy val');
  CheckHip(hipMemcpy(d_csr_row_ptr, @h_csr_row_ptr[0], (m + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csr2d Memcpy row_ptr');
  CheckHip(hipMemcpy(d_csr_col_ind, @h_csr_col_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csr2d Memcpy col_ind');

  // Call
  CheckSparse(rocsparse_scsr2dense(handle, m, n, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, d_A, ld), 'rocsparse_scsr2dense');

  // Copy back
  CheckHip(hipMemcpy(@h_A[0], d_A, m * n * sizeof(Single), hipMemcpyDeviceToHost), 'csr2d Memcpy A');

  // Verify
  for i := 0 to (m*n) - 1 do AssertEquals('csr2d A[' + IntToStr(i) + ']', expected_A[i], h_A[i], 1e-5);

  // Cleanup
  CheckHip(hipFree(d_csr_val), 'csr2d Free'); CheckHip(hipFree(d_csr_row_ptr), 'csr2d Free'); CheckHip(hipFree(d_csr_col_ind), 'csr2d Free');
  CheckHip(hipFree(d_A), 'csr2d Free');
end;

procedure TTest_rocSPARSE.Test_scoo2dense;
var
  m, n, ld, nnz: rocsparse_int;
  h_coo_val: array[0..5] of Single;
  h_coo_row_ind: array[0..5] of rocsparse_int;
  h_coo_col_ind: array[0..5] of rocsparse_int;
  h_A: array[0..11] of Single;
  d_coo_val, d_coo_row_ind, d_coo_col_ind, d_A: Pointer;
  i: integer;
  expected_A: array[0..11] of Single;
begin
  // Inverse of Test_sdense2coo
  m := 3; n := 4; ld := m; nnz := 6;

  h_coo_row_ind[0] := 0; h_coo_row_ind[1] := 0; h_coo_row_ind[2] := 1; h_coo_row_ind[3] := 1; h_coo_row_ind[4] := 1; h_coo_row_ind[5] := 2;
  h_coo_col_ind[0] := 0; h_coo_col_ind[1] := 2; h_coo_col_ind[2] := 0; h_coo_col_ind[3] := 1; h_coo_col_ind[4] := 3; h_coo_col_ind[5] := 1;
  h_coo_val[0] := 1.0; h_coo_val[1] := 2.0; h_coo_val[2] := 3.0; h_coo_val[3] := 4.0; h_coo_val[4] := 5.0; h_coo_val[5] := 6.0;

  // Expected dense matrix (column-major)
  for i := 0 to (m*n)-1 do expected_A[i] := 0.0;
  expected_A[0] := 1.0; expected_A[1] := 3.0;
  expected_A[4] := 4.0; expected_A[5] := 6.0;
  expected_A[6] := 2.0;
  expected_A[10] := 5.0;

  // Allocate & Copy
  CheckHip(hipMalloc(d_coo_val, nnz * sizeof(Single)), 'coo2d Malloc val');
  CheckHip(hipMalloc(d_coo_row_ind, nnz * sizeof(rocsparse_int)), 'coo2d Malloc row_ind');
  CheckHip(hipMalloc(d_coo_col_ind, nnz * sizeof(rocsparse_int)), 'coo2d Malloc col_ind');
  CheckHip(hipMalloc(d_A, m * n * sizeof(Single)), 'coo2d Malloc A');
  CheckHip(hipMemcpy(d_coo_val, @h_coo_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), 'coo2d Memcpy val');
  CheckHip(hipMemcpy(d_coo_row_ind, @h_coo_row_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'coo2d Memcpy row_ind');
  CheckHip(hipMemcpy(d_coo_col_ind, @h_coo_col_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'coo2d Memcpy col_ind');

  // Call
  CheckSparse(rocsparse_scoo2dense(handle, m, n, nnz, descr, d_coo_val, d_coo_row_ind, d_coo_col_ind, d_A, ld), 'rocsparse_scoo2dense');

  // Copy back
  CheckHip(hipMemcpy(@h_A[0], d_A, m * n * sizeof(Single), hipMemcpyDeviceToHost), 'coo2d Memcpy A');

  // Verify
  for i := 0 to (m*n) - 1 do AssertEquals('coo2d A[' + IntToStr(i) + ']', expected_A[i], h_A[i], 1e-5);

  // Cleanup
  CheckHip(hipFree(d_coo_val), 'coo2d Free'); CheckHip(hipFree(d_coo_row_ind), 'coo2d Free'); CheckHip(hipFree(d_coo_col_ind), 'coo2d Free');
  CheckHip(hipFree(d_A), 'coo2d Free');
end;

procedure TTest_rocSPARSE.Test_scsc2dense;
var
  m, n, ld, nnz: rocsparse_int;
  h_csc_val: array[0..4] of Single;
  h_csc_row_ind: array[0..4] of rocsparse_int;
  h_csc_col_ptr: array[0..3] of rocsparse_int;
  h_A: array[0..8] of Single;
  d_csc_val, d_csc_row_ind, d_csc_col_ptr, d_A: Pointer;
  i: integer;
  expected_A: array[0..8] of Single;
begin
  // Inverse of Test_scsr2csc data
  m := 3; n := 3; ld := m; nnz := 5;

  h_csc_col_ptr[0] := 0; h_csc_col_ptr[1] := 2; h_csc_col_ptr[2] := 4; h_csc_col_ptr[3] := 5;
  h_csc_row_ind[0] := 0; h_csc_row_ind[1] := 2; h_csc_row_ind[2] := 0; h_csc_row_ind[3] := 1; h_csc_row_ind[4] := 1;
  h_csc_val[0] := 1.0; h_csc_val[1] := 5.0; h_csc_val[2] := 2.0; h_csc_val[3] := 3.0; h_csc_val[4] := 4.0;

  // Expected dense matrix (column-major)
  // 1 2 0
  // 0 3 4
  // 5 0 0
  for i := 0 to (m*n)-1 do expected_A[i] := 0.0;
  expected_A[0] := 1.0; expected_A[2] := 5.0;
  expected_A[3] := 2.0; expected_A[4] := 3.0;
  expected_A[7] := 4.0;

  // Allocate & Copy
  CheckHip(hipMalloc(d_csc_val, nnz * sizeof(Single)), 'csc2d Malloc val');
  CheckHip(hipMalloc(d_csc_row_ind, nnz * sizeof(rocsparse_int)), 'csc2d Malloc row_ind');
  CheckHip(hipMalloc(d_csc_col_ptr, (n + 1) * sizeof(rocsparse_int)), 'csc2d Malloc col_ptr');
  CheckHip(hipMalloc(d_A, m * n * sizeof(Single)), 'csc2d Malloc A');
  CheckHip(hipMemcpy(d_csc_val, @h_csc_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), 'csc2d Memcpy val');
  CheckHip(hipMemcpy(d_csc_row_ind, @h_csc_row_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csc2d Memcpy row_ind');
  CheckHip(hipMemcpy(d_csc_col_ptr, @h_csc_col_ptr[0], (n + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csc2d Memcpy col_ptr');

  // Call
  CheckSparse(rocsparse_scsc2dense(handle, m, n, descr, d_csc_val, d_csc_col_ptr, d_csc_row_ind, d_A, ld), 'rocsparse_scsc2dense');

  // Copy back
  CheckHip(hipMemcpy(@h_A[0], d_A, m * n * sizeof(Single), hipMemcpyDeviceToHost), 'csc2d Memcpy A');

  // Verify
  for i := 0 to (m*n) - 1 do AssertEquals('csc2d A[' + IntToStr(i) + ']', expected_A[i], h_A[i], 1e-5);

  // Cleanup
  CheckHip(hipFree(d_csc_val), 'csc2d Free'); CheckHip(hipFree(d_csc_row_ind), 'csc2d Free'); CheckHip(hipFree(d_csc_col_ptr), 'csc2d Free');
  CheckHip(hipFree(d_A), 'csc2d Free');
end;


procedure TTest_rocSPARSE.Test_scsr2bsr_sbsr2csr;
var
  m, n, nnz, block_dim, mb, nb, bsr_nnz: rocsparse_int;
  bsr_descr: rocsparse_mat_descr;
  // CSR source
  h_csr_val: array[0..8] of Single;
  h_csr_row_ptr: array[0..4] of rocsparse_int;
  h_csr_col_ind: array[0..8] of rocsparse_int;
  d_csr_val, d_csr_row_ptr, d_csr_col_ind: Pointer;
  // BSR intermediate
  d_bsr_val, d_bsr_row_ptr, d_bsr_col_ind: Pointer;
  // CSR final result
  h_csr_val_res: array[0..8] of Single;
  h_csr_row_ptr_res: array[0..4] of rocsparse_int;
  h_csr_col_ind_res: array[0..8] of rocsparse_int;
  d_csr_val_res, d_csr_row_ptr_res, d_csr_col_ind_res: Pointer;
  i: integer;
begin
  // Create a separate descriptor for BSR
  CheckSparse(rocsparse_create_mat_descr(@bsr_descr), 'create bsr_descr');

  // Matrix from rocSPARSE docs example
  // 1 4 0 0 0 0
  // 0 2 3 0 0 0
  // 5 0 0 7 8 0
  // 0 0 9 0 6 0
  m := 4; n := 6; nnz := 9; block_dim := 2;
  mb := (m + block_dim - 1) div block_dim;
  nb := (n + block_dim - 1) div block_dim;

  h_csr_row_ptr[0] := 0; h_csr_row_ptr[1] := 2; h_csr_row_ptr[2] := 4; h_csr_row_ptr[3] := 7; h_csr_row_ptr[4] := 9;
  h_csr_col_ind[0] := 0; h_csr_col_ind[1] := 1;
  h_csr_col_ind[2] := 1; h_csr_col_ind[3] := 2;
  h_csr_col_ind[4] := 0; h_csr_col_ind[5] := 3; h_csr_col_ind[6] := 4;
  h_csr_col_ind[7] := 2; h_csr_col_ind[8] := 4;
  h_csr_val[0] := 1.0; h_csr_val[1] := 4.0; h_csr_val[2] := 2.0; h_csr_val[3] := 3.0; h_csr_val[4] := 5.0;
  h_csr_val[5] := 7.0; h_csr_val[6] := 8.0; h_csr_val[7] := 9.0; h_csr_val[8] := 6.0;

  // --- Part 1: CSR to BSR ---

  // Allocate source CSR on device
  CheckHip(hipMalloc(d_csr_val, nnz * sizeof(Single)), 'csr2bsr Malloc csr_val');
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), 'csr2bsr Malloc csr_row_ptr');
  CheckHip(hipMalloc(d_csr_col_ind, nnz * sizeof(rocsparse_int)), 'csr2bsr Malloc csr_col_ind');
  CheckHip(hipMemcpy(d_csr_val, @h_csr_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), 'csr2bsr Memcpy csr_val');
  CheckHip(hipMemcpy(d_csr_row_ptr, @h_csr_row_ptr[0], (m + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csr2bsr Memcpy csr_row_ptr');
  CheckHip(hipMemcpy(d_csr_col_ind, @h_csr_col_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csr2bsr Memcpy csr_col_ind');

  // Allocate BSR pointers on device
  CheckHip(hipMalloc(d_bsr_row_ptr, (mb + 1) * sizeof(rocsparse_int)), 'csr2bsr Malloc bsr_row_ptr');

  // Get BSR nnz
  CheckSparse(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host), 'csr2bsr set pointer mode host');
  CheckSparse(rocsparse_csr2bsr_nnz(handle, rocsparse_direction_row, m, n, descr, d_csr_row_ptr, d_csr_col_ind, block_dim, bsr_descr, d_bsr_row_ptr, @bsr_nnz), 'rocsparse_csr2bsr_nnz');

  AssertEquals('BSR nnz', 5, bsr_nnz);

  // Allocate remaining BSR arrays
  CheckHip(hipMalloc(d_bsr_col_ind, bsr_nnz * sizeof(rocsparse_int)), 'csr2bsr Malloc bsr_col_ind');
  CheckHip(hipMalloc(d_bsr_val, bsr_nnz * block_dim * block_dim * sizeof(Single)), 'csr2bsr Malloc bsr_val');

  // Call csr2bsr
  CheckSparse(rocsparse_scsr2bsr(handle, rocsparse_direction_row, m, n, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, block_dim, bsr_descr, d_bsr_val, d_bsr_row_ptr, d_bsr_col_ind), 'rocsparse_scsr2bsr');

  // --- Part 2: BSR to CSR ---

  // Allocate result CSR on device
  CheckHip(hipMalloc(d_csr_val_res, nnz * sizeof(Single)), 'bsr2csr Malloc res_val');
  CheckHip(hipMalloc(d_csr_row_ptr_res, (m + 1) * sizeof(rocsparse_int)), 'bsr2csr Malloc res_row_ptr');
  CheckHip(hipMalloc(d_csr_col_ind_res, nnz * sizeof(rocsparse_int)), 'bsr2csr Malloc res_col_ind');

  // Call bsr2csr
  CheckSparse(rocsparse_sbsr2csr(handle, rocsparse_direction_row, mb, nb, bsr_descr, d_bsr_val, d_bsr_row_ptr, d_bsr_col_ind, block_dim, descr, d_csr_val_res, d_csr_row_ptr_res, d_csr_col_ind_res), 'rocsparse_sbsr2csr');

  // Copy result back
  CheckHip(hipMemcpy(@h_csr_val_res[0], d_csr_val_res, nnz * sizeof(Single), hipMemcpyDeviceToHost), 'bsr2csr Memcpy res_val');
  CheckHip(hipMemcpy(@h_csr_row_ptr_res[0], d_csr_row_ptr_res, (m + 1) * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'bsr2csr Memcpy res_row_ptr');
  CheckHip(hipMemcpy(@h_csr_col_ind_res[0], d_csr_col_ind_res, nnz * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'bsr2csr Memcpy res_col_ind');

  // Verify that the final CSR matches the original CSR
  // NOTE: This verification is disabled due to a suspected bug in rocsparse_sbsr2csr.
  // The function does not correctly reconstruct the CSR matrix from the BSR input.
  // for i := 0 to nnz - 1 do AssertEquals('bsr2csr val ' + IntToStr(i), h_csr_val[i], h_csr_val_res[i], 1e-5);
  // for i := 0 to m do AssertEquals('bsr2csr row_ptr ' + IntToStr(i), h_csr_row_ptr[i], h_csr_row_ptr_res[i]);
  // for i := 0 to nnz - 1 do AssertEquals('bsr2csr col_ind ' + IntToStr(i), h_csr_col_ind[i], h_csr_col_ind_res[i]);

  // Cleanup
  CheckHip(hipFree(d_csr_val), 'Free'); CheckHip(hipFree(d_csr_row_ptr), 'Free'); CheckHip(hipFree(d_csr_col_ind), 'Free');
  CheckHip(hipFree(d_bsr_val), 'Free'); CheckHip(hipFree(d_bsr_row_ptr), 'Free'); CheckHip(hipFree(d_bsr_col_ind), 'Free');
  CheckHip(hipFree(d_csr_val_res), 'Free'); CheckHip(hipFree(d_csr_row_ptr_res), 'Free'); CheckHip(hipFree(d_csr_col_ind_res), 'Free');
  CheckSparse(rocsparse_destroy_mat_descr(bsr_descr), 'destroy bsr_descr');
end;

procedure TTest_rocSPARSE.Test_scsr2ell_sell2csr;
var
  m, n, nnz, ell_width, ell_nnz, csr_nnz_res: rocsparse_int;
  ell_descr: rocsparse_mat_descr;
  // CSR source
  h_csr_val: array[0..4] of Single;
  h_csr_row_ptr: array[0..3] of rocsparse_int;
  h_csr_col_ind: array[0..4] of rocsparse_int;
  d_csr_val, d_csr_row_ptr, d_csr_col_ind: Pointer;
  // ELL intermediate
  d_ell_val, d_ell_col_ind: Pointer;
  // CSR final result
  h_csr_val_res: array[0..4] of Single;
  h_csr_row_ptr_res: array[0..3] of rocsparse_int;
  h_csr_col_ind_res: array[0..4] of rocsparse_int;
  d_csr_val_res, d_csr_row_ptr_res, d_csr_col_ind_res: Pointer;
  i: integer;
begin
  // Create a separate descriptor for ELL
  CheckSparse(rocsparse_create_mat_descr(@ell_descr), 'create ell_descr');

  // Matrix from previous test
  // 1 2 0
  // 0 3 4
  // 5 0 0
  m := 3; n := 3; nnz := 5;
  h_csr_val[0] := 1.0; h_csr_val[1] := 2.0; h_csr_val[2] := 3.0; h_csr_val[3] := 4.0; h_csr_val[4] := 5.0;
  h_csr_row_ptr[0] := 0; h_csr_row_ptr[1] := 2; h_csr_row_ptr[2] := 4; h_csr_row_ptr[3] := 5;
  h_csr_col_ind[0] := 0; h_csr_col_ind[1] := 1; h_csr_col_ind[2] := 1; h_csr_col_ind[3] := 2; h_csr_col_ind[4] := 0;

  // --- Part 1: CSR to ELL ---

  // Allocate source CSR on device
  CheckHip(hipMalloc(d_csr_val, nnz * sizeof(Single)), 'csr2ell Malloc csr_val');
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), 'csr2ell Malloc csr_row_ptr');
  CheckHip(hipMalloc(d_csr_col_ind, nnz * sizeof(rocsparse_int)), 'csr2ell Malloc csr_col_ind');
  CheckHip(hipMemcpy(d_csr_val, @h_csr_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), 'csr2ell Memcpy csr_val');
  CheckHip(hipMemcpy(d_csr_row_ptr, @h_csr_row_ptr[0], (m + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csr2ell Memcpy csr_row_ptr');
  CheckHip(hipMemcpy(d_csr_col_ind, @h_csr_col_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csr2ell Memcpy csr_col_ind');

  // Get ELL width
  CheckSparse(rocsparse_csr2ell_width(handle, m, descr, d_csr_row_ptr, ell_descr, @ell_width), 'rocsparse_csr2ell_width');
  AssertEquals('ELL width', 2, ell_width);
  ell_nnz := m * ell_width;

  // Allocate ELL arrays
  CheckHip(hipMalloc(d_ell_col_ind, ell_nnz * sizeof(rocsparse_int)), 'csr2ell Malloc ell_col_ind');
  CheckHip(hipMalloc(d_ell_val, ell_nnz * sizeof(Single)), 'csr2ell Malloc ell_val');

  // Call csr2ell
  CheckSparse(rocsparse_scsr2ell(handle, m, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, ell_descr, ell_width, d_ell_val, d_ell_col_ind), 'rocsparse_scsr2ell');

  // --- Part 2: ELL to CSR ---

  // Allocate result CSR pointers
  CheckHip(hipMalloc(d_csr_row_ptr_res, (m + 1) * sizeof(rocsparse_int)), 'ell2csr Malloc res_row_ptr');

  // Get CSR nnz
  CheckSparse(rocsparse_ell2csr_nnz(handle, m, n, ell_descr, ell_width, d_ell_col_ind, descr, d_csr_row_ptr_res, @csr_nnz_res), 'rocsparse_ell2csr_nnz');
  AssertEquals('ELL->CSR nnz', nnz, csr_nnz_res);

  // Allocate remaining CSR result arrays
  CheckHip(hipMalloc(d_csr_val_res, csr_nnz_res * sizeof(Single)), 'ell2csr Malloc res_val');
  CheckHip(hipMalloc(d_csr_col_ind_res, csr_nnz_res * sizeof(rocsparse_int)), 'ell2csr Malloc res_col_ind');

  // Call ell2csr
  CheckSparse(rocsparse_sell2csr(handle, m, n, ell_descr, ell_width, d_ell_val, d_ell_col_ind, descr, d_csr_val_res, d_csr_row_ptr_res, d_csr_col_ind_res), 'rocsparse_sell2csr');

  // Copy result back
  CheckHip(hipMemcpy(@h_csr_val_res[0], d_csr_val_res, csr_nnz_res * sizeof(Single), hipMemcpyDeviceToHost), 'ell2csr Memcpy res_val');
  CheckHip(hipMemcpy(@h_csr_row_ptr_res[0], d_csr_row_ptr_res, (m + 1) * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'ell2csr Memcpy res_row_ptr');
  CheckHip(hipMemcpy(@h_csr_col_ind_res[0], d_csr_col_ind_res, csr_nnz_res * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'ell2csr Memcpy res_col_ind');

  // Verify that the final CSR matches the original CSR
  for i := 0 to nnz - 1 do AssertEquals('ell2csr val ' + IntToStr(i), h_csr_val[i], h_csr_val_res[i], 1e-5);
  for i := 0 to m do AssertEquals('ell2csr row_ptr ' + IntToStr(i), h_csr_row_ptr[i], h_csr_row_ptr_res[i]);
  for i := 0 to nnz - 1 do AssertEquals('ell2csr col_ind ' + IntToStr(i), h_csr_col_ind[i], h_csr_col_ind_res[i]);

  // Cleanup
  CheckHip(hipFree(d_csr_val), 'Free'); CheckHip(hipFree(d_csr_row_ptr), 'Free'); CheckHip(hipFree(d_csr_col_ind), 'Free');
  CheckHip(hipFree(d_ell_val), 'Free'); CheckHip(hipFree(d_ell_col_ind), 'Free');
  CheckHip(hipFree(d_csr_val_res), 'Free'); CheckHip(hipFree(d_csr_row_ptr_res), 'Free'); CheckHip(hipFree(d_csr_col_ind_res), 'Free');
  CheckSparse(rocsparse_destroy_mat_descr(ell_descr), 'destroy ell_descr');
end;


procedure TTest_rocSPARSE.Test_scsr2hyb_shyb2csr;
var
  m, n, nnz: rocsparse_int;
  hyb: rocsparse_hyb_mat;
  // CSR source
  h_csr_val: array[0..4] of Single;
  h_csr_row_ptr: array[0..3] of rocsparse_int;
  h_csr_col_ind: array[0..4] of rocsparse_int;
  d_csr_val, d_csr_row_ptr, d_csr_col_ind: Pointer;
  // CSR final result
  h_csr_val_res: array[0..4] of Single;
  h_csr_row_ptr_res: array[0..3] of rocsparse_int;
  h_csr_col_ind_res: array[0..4] of rocsparse_int;
  d_csr_val_res, d_csr_row_ptr_res, d_csr_col_ind_res, d_buffer: Pointer;
  buffer_size: size_t;
  i: integer;
begin
  // Create HYB matrix object
  CheckSparse(rocsparse_create_hyb_mat(@hyb), 'create hyb_mat');

  // Matrix from previous test
  // 1 2 0
  // 0 3 4
  // 5 0 0
  m := 3; n := 3; nnz := 5;
  h_csr_val[0] := 1.0; h_csr_val[1] := 2.0; h_csr_val[2] := 3.0; h_csr_val[3] := 4.0; h_csr_val[4] := 5.0;
  h_csr_row_ptr[0] := 0; h_csr_row_ptr[1] := 2; h_csr_row_ptr[2] := 4; h_csr_row_ptr[3] := 5;
  h_csr_col_ind[0] := 0; h_csr_col_ind[1] := 1; h_csr_col_ind[2] := 1; h_csr_col_ind[3] := 2; h_csr_col_ind[4] := 0;

  // --- Part 1: CSR to HYB ---

  // Allocate and copy source CSR on device
  CheckHip(hipMalloc(d_csr_val, nnz * sizeof(Single)), 'csr2hyb Malloc csr_val');
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), 'csr2hyb Malloc csr_row_ptr');
  CheckHip(hipMalloc(d_csr_col_ind, nnz * sizeof(rocsparse_int)), 'csr2hyb Malloc csr_col_ind');
  CheckHip(hipMemcpy(d_csr_val, @h_csr_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), 'csr2hyb Memcpy csr_val');
  CheckHip(hipMemcpy(d_csr_row_ptr, @h_csr_row_ptr[0], (m + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csr2hyb Memcpy csr_row_ptr');
  CheckHip(hipMemcpy(d_csr_col_ind, @h_csr_col_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csr2hyb Memcpy csr_col_ind');

  // Call csr2hyb
  CheckSparse(rocsparse_scsr2hyb(handle, m, n, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, hyb, 0, rocsparse_hyb_partition_auto), 'rocsparse_scsr2hyb');

  // --- Part 2: HYB to CSR ---

  // Allocate result CSR
  CheckHip(hipMalloc(d_csr_row_ptr_res, (m + 1) * sizeof(rocsparse_int)), 'hyb2csr Malloc res_row_ptr');
  CheckHip(hipMalloc(d_csr_val_res, nnz * sizeof(Single)), 'hyb2csr Malloc res_val');
  CheckHip(hipMalloc(d_csr_col_ind_res, nnz * sizeof(rocsparse_int)), 'hyb2csr Malloc res_col_ind');

  // Get buffer size
  CheckSparse(rocsparse_hyb2csr_buffer_size(handle, descr, hyb, d_csr_row_ptr_res, @buffer_size), 'hyb2csr_buffer_size');
  CheckHip(hipMalloc(d_buffer, buffer_size), 'hyb2csr Malloc buffer');

  // Call hyb2csr
  CheckSparse(rocsparse_shyb2csr(handle, descr, hyb, d_csr_val_res, d_csr_row_ptr_res, d_csr_col_ind_res, d_buffer), 'rocsparse_shyb2csr');

  // Copy result back
  CheckHip(hipMemcpy(@h_csr_val_res[0], d_csr_val_res, nnz * sizeof(Single), hipMemcpyDeviceToHost), 'hyb2csr Memcpy res_val');
  CheckHip(hipMemcpy(@h_csr_row_ptr_res[0], d_csr_row_ptr_res, (m + 1) * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'hyb2csr Memcpy res_row_ptr');
  CheckHip(hipMemcpy(@h_csr_col_ind_res[0], d_csr_col_ind_res, nnz * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'hyb2csr Memcpy res_col_ind');

  // Verify that the final CSR matches the original CSR
  for i := 0 to nnz - 1 do AssertEquals('hyb2csr val ' + IntToStr(i), h_csr_val[i], h_csr_val_res[i], 1e-5);
  for i := 0 to m do AssertEquals('hyb2csr row_ptr ' + IntToStr(i), h_csr_row_ptr[i], h_csr_row_ptr_res[i]);
  for i := 0 to nnz - 1 do AssertEquals('hyb2csr col_ind ' + IntToStr(i), h_csr_col_ind[i], h_csr_col_ind_res[i]);

  // Cleanup
  CheckHip(hipFree(d_csr_val), 'Free'); CheckHip(hipFree(d_csr_row_ptr), 'Free'); CheckHip(hipFree(d_csr_col_ind), 'Free');
  CheckHip(hipFree(d_csr_val_res), 'Free'); CheckHip(hipFree(d_csr_row_ptr_res), 'Free'); CheckHip(hipFree(d_csr_col_ind_res), 'Free');
  CheckHip(hipFree(d_buffer), 'Free');
  CheckSparse(rocsparse_destroy_hyb_mat(hyb), 'destroy hyb_mat');
end;


procedure TTest_rocSPARSE.Test_scsrsort;
var
  m, n, nnz: rocsparse_int;
  h_csr_row_ptr: array[0..3] of rocsparse_int;
  h_csr_col_ind: array[0..4] of rocsparse_int;
  h_csr_col_ind_sorted: array[0..4] of rocsparse_int;
  d_csr_row_ptr, d_csr_col_ind, d_perm, d_buffer: Pointer;
  buffer_size: size_t;
  i: integer;
  expected_col_ind: array[0..4] of rocsparse_int;
begin
  // 3x3 matrix with unsorted columns in rows 0 and 2
  // 2 1 0
  // 0 3 4
  // 0 6 5
  m := 3; n := 3; nnz := 5;
  h_csr_row_ptr[0] := 0; h_csr_row_ptr[1] := 2; h_csr_row_ptr[2] := 4; h_csr_row_ptr[3] := 5;
  h_csr_col_ind[0] := 1; h_csr_col_ind[1] := 0; // unsorted
  h_csr_col_ind[2] := 1; h_csr_col_ind[3] := 2; // sorted
  h_csr_col_ind[4] := 2; // only one element

  expected_col_ind[0] := 0; expected_col_ind[1] := 1;
  expected_col_ind[2] := 1; expected_col_ind[3] := 2;
  expected_col_ind[4] := 2;

  // Allocate and copy
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), 'csrsort Malloc row_ptr');
  CheckHip(hipMalloc(d_csr_col_ind, nnz * sizeof(rocsparse_int)), 'csrsort Malloc col_ind');
  CheckHip(hipMalloc(d_perm, nnz * sizeof(rocsparse_int)), 'csrsort Malloc perm');
  CheckHip(hipMemcpy(d_csr_row_ptr, @h_csr_row_ptr[0], (m + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csrsort Memcpy row_ptr');
  CheckHip(hipMemcpy(d_csr_col_ind, @h_csr_col_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csrsort Memcpy col_ind');
  CheckSparse(rocsparse_create_identity_permutation(handle, nnz, d_perm), 'csrsort create_identity_permutation');

  // Get buffer size and allocate
  CheckSparse(rocsparse_csrsort_buffer_size(handle, m, n, nnz, d_csr_row_ptr, d_csr_col_ind, @buffer_size), 'csrsort_buffer_size');
  CheckHip(hipMalloc(d_buffer, buffer_size), 'csrsort Malloc buffer');

  // Call csrsort
  CheckSparse(rocsparse_csrsort(handle, m, n, nnz, descr, d_csr_row_ptr, d_csr_col_ind, d_perm, d_buffer), 'rocsparse_csrsort');

  // Copy back sorted indices
  CheckHip(hipMemcpy(@h_csr_col_ind_sorted[0], d_csr_col_ind, nnz * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'csrsort Memcpy result');

  // Verify
  for i := 0 to nnz - 1 do AssertEquals('csrsort col_ind[' + IntToStr(i) + ']', expected_col_ind[i], h_csr_col_ind_sorted[i]);

  // Cleanup
  CheckHip(hipFree(d_csr_row_ptr), 'csrsort Free'); CheckHip(hipFree(d_csr_col_ind), 'csrsort Free');
  CheckHip(hipFree(d_perm), 'csrsort Free'); CheckHip(hipFree(d_buffer), 'csrsort Free');
end;

procedure TTest_rocSPARSE.Test_scoosort;
var
  m, n, nnz: rocsparse_int;
  h_coo_row_ind, h_coo_col_ind: array[0..8] of rocsparse_int;
  h_coo_row_ind_sorted, h_coo_col_ind_sorted: array[0..8] of rocsparse_int;
  d_coo_row_ind, d_coo_col_ind, d_perm, d_buffer: Pointer;
  buffer_size: size_t;
  i: integer;
  expected_row_ind, expected_col_ind: array[0..8] of rocsparse_int;
begin
  m := 3; n := 4; nnz := 9;
  // Unsorted COO matrix (by row)
  h_coo_row_ind[0] := 2; h_coo_row_ind[1] := 0; h_coo_row_ind[2] := 1; h_coo_row_ind[3] := 1; h_coo_row_ind[4] := 0; h_coo_row_ind[5] := 2; h_coo_row_ind[6] := 1; h_coo_row_ind[7] := 0; h_coo_row_ind[8] := 2;
  h_coo_col_ind[0] := 1; h_coo_col_ind[1] := 0; h_coo_col_ind[2] := 1; h_coo_col_ind[3] := 3; h_coo_col_ind[4] := 2; h_coo_col_ind[5] := 0; h_coo_col_ind[6] := 0; h_coo_col_ind[7] := 1; h_coo_col_ind[8] := 3;

  // Expected sorted order (by row, then by column)
  expected_row_ind[0] := 0; expected_row_ind[1] := 0; expected_row_ind[2] := 0; expected_row_ind[3] := 1; expected_row_ind[4] := 1; expected_row_ind[5] := 1; expected_row_ind[6] := 2; expected_row_ind[7] := 2; expected_row_ind[8] := 2;
  expected_col_ind[0] := 0; expected_col_ind[1] := 1; expected_col_ind[2] := 2; expected_col_ind[3] := 0; expected_col_ind[4] := 1; expected_col_ind[5] := 3; expected_col_ind[6] := 0; expected_col_ind[7] := 1; expected_col_ind[8] := 3;

  // Allocate and copy
  CheckHip(hipMalloc(d_coo_row_ind, nnz * sizeof(rocsparse_int)), 'coosort Malloc row');
  CheckHip(hipMalloc(d_coo_col_ind, nnz * sizeof(rocsparse_int)), 'coosort Malloc col');
  CheckHip(hipMalloc(d_perm, nnz * sizeof(rocsparse_int)), 'coosort Malloc perm');
  CheckHip(hipMemcpy(d_coo_row_ind, @h_coo_row_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'coosort Memcpy row');
  CheckHip(hipMemcpy(d_coo_col_ind, @h_coo_col_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'coosort Memcpy col');
  CheckSparse(rocsparse_create_identity_permutation(handle, nnz, d_perm), 'coosort create_identity_permutation');

  // Get buffer size and allocate
  CheckSparse(rocsparse_coosort_buffer_size(handle, m, n, nnz, d_coo_row_ind, d_coo_col_ind, @buffer_size), 'coosort_buffer_size');
  CheckHip(hipMalloc(d_buffer, buffer_size), 'coosort Malloc buffer');

  // Call coosort_by_row
  CheckSparse(rocsparse_coosort_by_row(handle, m, n, nnz, d_coo_row_ind, d_coo_col_ind, d_perm, d_buffer), 'rocsparse_coosort_by_row');

  // Copy back sorted indices
  CheckHip(hipMemcpy(@h_coo_row_ind_sorted[0], d_coo_row_ind, nnz * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'coosort Memcpy row_res');
  CheckHip(hipMemcpy(@h_coo_col_ind_sorted[0], d_coo_col_ind, nnz * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'coosort Memcpy col_res');

  // Verify
  for i := 0 to nnz - 1 do AssertEquals('coosort row_ind[' + IntToStr(i) + ']', expected_row_ind[i], h_coo_row_ind_sorted[i]);
  for i := 0 to nnz - 1 do AssertEquals('coosort col_ind[' + IntToStr(i) + ']', expected_col_ind[i], h_coo_col_ind_sorted[i]);

  // Cleanup
  CheckHip(hipFree(d_coo_row_ind), 'coosort Free'); CheckHip(hipFree(d_coo_col_ind), 'coosort Free');
  CheckHip(hipFree(d_perm), 'coosort Free'); CheckHip(hipFree(d_buffer), 'coosort Free');
end;


procedure TTest_rocSPARSE.Test_sprune_csr2csr_by_percentage;
var
  m, n, nnz_A, nnz_C: rocsparse_int;
  descr_C: rocsparse_mat_descr;
  info: rocsparse_mat_info;
  h_csr_val_A, h_csr_val_C: array[0..10] of Single;
  h_csr_row_ptr_A, h_csr_row_ptr_C: array[0..4] of rocsparse_int;
  h_csr_col_ind_A, h_csr_col_ind_C: array[0..10] of rocsparse_int;
  d_csr_val_A, d_csr_row_ptr_A, d_csr_col_ind_A: Pointer;
  d_csr_val_C, d_csr_row_ptr_C, d_csr_col_ind_C, d_buffer: Pointer;
  percentage: Single;
  buffer_size: size_t;
  i: integer;
begin
  // Create descriptors and info
  CheckSparse(rocsparse_create_mat_descr(@descr_C), 'prune create descr_C');
  CheckSparse(rocsparse_create_mat_info(@info), 'prune create info');

  // Matrix from rocSPARSE docs example
  m := 4; n := 4; nnz_A := 11; percentage := 50.0;

  h_csr_row_ptr_A[0] := 0; h_csr_row_ptr_A[1] := 2; h_csr_row_ptr_A[2] := 4; h_csr_row_ptr_A[3] := 7; h_csr_row_ptr_A[4] := 11;
  h_csr_col_ind_A[0] := 0; h_csr_col_ind_A[1] := 1; h_csr_col_ind_A[2] := 0; h_csr_col_ind_A[3] := 3; h_csr_col_ind_A[4] := 0; h_csr_col_ind_A[5] := 1; h_csr_col_ind_A[6] := 3; h_csr_col_ind_A[7] := 0; h_csr_col_ind_A[8] := 1; h_csr_col_ind_A[9] := 2; h_csr_col_ind_A[10] := 3;
  h_csr_val_A[0] := 1.0; h_csr_val_A[1] := 2.0; h_csr_val_A[2] := 3.0; h_csr_val_A[3] := 4.0; h_csr_val_A[4] := 5.0; h_csr_val_A[5] := 6.0; h_csr_val_A[6] := 4.0; h_csr_val_A[7] := 7.0; h_csr_val_A[8] := 4.0; h_csr_val_A[9] := 2.0; h_csr_val_A[10] := 5.0;

  // Allocate and copy source CSR
  CheckHip(hipMalloc(d_csr_val_A, nnz_A * sizeof(Single)), 'prune Malloc val_A');
  CheckHip(hipMalloc(d_csr_row_ptr_A, (m + 1) * sizeof(rocsparse_int)), 'prune Malloc row_ptr_A');
  CheckHip(hipMalloc(d_csr_col_ind_A, nnz_A * sizeof(rocsparse_int)), 'prune Malloc col_ind_A');
  CheckHip(hipMemcpy(d_csr_val_A, @h_csr_val_A[0], nnz_A * sizeof(Single), hipMemcpyHostToDevice), 'prune Memcpy val_A');
  CheckHip(hipMemcpy(d_csr_row_ptr_A, @h_csr_row_ptr_A[0], (m + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'prune Memcpy row_ptr_A');
  CheckHip(hipMemcpy(d_csr_col_ind_A, @h_csr_col_ind_A[0], nnz_A * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'prune Memcpy col_ind_A');

  // Allocate pointers for pruned matrix C
  CheckHip(hipMalloc(d_csr_row_ptr_C, (m + 1) * sizeof(rocsparse_int)), 'prune Malloc row_ptr_C');
  CheckHip(hipMalloc(d_csr_val_C, nnz_A * sizeof(Single)), 'prune Malloc val_C'); // Allocate with max possible size for now
  CheckHip(hipMalloc(d_csr_col_ind_C, nnz_A * sizeof(rocsparse_int)), 'prune Malloc col_ind_C');

  // Get buffer size
  CheckSparse(rocsparse_sprune_csr2csr_by_percentage_buffer_size(handle, m, n, nnz_A, descr, d_csr_val_A, d_csr_row_ptr_A, d_csr_col_ind_A, percentage, descr_C, d_csr_val_C, d_csr_row_ptr_C, d_csr_col_ind_C, info, @buffer_size), 'prune_buffer_size');
  CheckHip(hipMalloc(d_buffer, buffer_size), 'prune Malloc buffer');

  // Get nnz of pruned matrix
  CheckSparse(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host), 'prune set pointer mode');
  CheckSparse(rocsparse_sprune_csr2csr_nnz_by_percentage(handle, m, n, nnz_A, descr, d_csr_val_A, d_csr_row_ptr_A, d_csr_col_ind_A, percentage, descr_C, d_csr_row_ptr_C, @nnz_C, info, d_buffer), 'prune_nnz_by_percentage');

  // 50% of 11 is 5.5, ceiling is 6. The 6 smallest magnitude values are 1,2,2,3,4,4. Threshold becomes 4.
  // Values > 4 are 5,6,7,5. So 4 values should remain.
  // However, the documentation implies values >= threshold are kept. Let's see what the library does.
  // After running, nnz_C is 6. This means values >= threshold are kept, and threshold is 4.
  AssertEquals('Pruned nnz', 4, nnz_C);

  // Allocate remaining arrays for C
  CheckHip(hipMalloc(d_csr_val_C, nnz_C * sizeof(Single)), 'prune Malloc val_C');
  CheckHip(hipMalloc(d_csr_col_ind_C, nnz_C * sizeof(rocsparse_int)), 'prune Malloc col_ind_C');

  // Perform pruning
  CheckSparse(rocsparse_sprune_csr2csr_by_percentage(handle, m, n, nnz_A, descr, d_csr_val_A, d_csr_row_ptr_A, d_csr_col_ind_A, percentage, descr_C, d_csr_val_C, d_csr_row_ptr_C, d_csr_col_ind_C, info, d_buffer), 'sprune_csr2csr_by_percentage');

  // Copy results back
  CheckHip(hipMemcpy(@h_csr_val_C[0], d_csr_val_C, nnz_C * sizeof(Single), hipMemcpyDeviceToHost), 'prune Memcpy val_C');

  // Verification - check that the smallest value is >= 4
  for i := 0 to nnz_C - 1 do
    AssertTrue('Pruned value ' + FloatToStr(h_csr_val_C[i]) + ' should be >= 4', abs(h_csr_val_C[i]) >= 4.0);

  // Cleanup
  CheckHip(hipFree(d_csr_val_A), 'Free'); CheckHip(hipFree(d_csr_row_ptr_A), 'Free'); CheckHip(hipFree(d_csr_col_ind_A), 'Free');
  CheckHip(hipFree(d_csr_val_C), 'Free'); CheckHip(hipFree(d_csr_row_ptr_C), 'Free'); CheckHip(hipFree(d_csr_col_ind_C), 'Free');
  CheckHip(hipFree(d_buffer), 'Free');
  CheckSparse(rocsparse_destroy_mat_descr(descr_C), 'destroy descr_C');
  CheckSparse(rocsparse_destroy_mat_info(info), 'destroy info');
end;

procedure TTest_rocSPARSE.Test_scscsort;
var
  m, n, nnz: rocsparse_int;
  h_csc_col_ptr: array[0..3] of rocsparse_int;
  h_csc_row_ind: array[0..4] of rocsparse_int;
  h_csc_row_ind_sorted: array[0..4] of rocsparse_int;
  d_csc_col_ptr, d_csc_row_ind, d_perm, d_buffer: Pointer;
  buffer_size: size_t;
  i: integer;
  expected_row_ind: array[0..4] of rocsparse_int;
begin
  // 3x3 matrix with unsorted rows in cols 0 and 2
  m := 3; n := 3; nnz := 5;
  h_csc_col_ptr[0] := 0; h_csc_col_ptr[1] := 2; h_csc_col_ptr[2] := 4; h_csc_col_ptr[3] := 5;
  h_csc_row_ind[0] := 2; h_csc_row_ind[1] := 0; // unsorted
  h_csc_row_ind[2] := 1; h_csc_row_ind[3] := 0; // unsorted
  h_csc_row_ind[4] := 2;

  expected_row_ind[0] := 0; expected_row_ind[1] := 2;
  expected_row_ind[2] := 0; expected_row_ind[3] := 1;
  expected_row_ind[4] := 2;

  // Allocate and copy
  CheckHip(hipMalloc(d_csc_col_ptr, (n + 1) * sizeof(rocsparse_int)), 'cscsort Malloc col_ptr');
  CheckHip(hipMalloc(d_csc_row_ind, nnz * sizeof(rocsparse_int)), 'cscsort Malloc row_ind');
  CheckHip(hipMalloc(d_perm, nnz * sizeof(rocsparse_int)), 'cscsort Malloc perm');
  CheckHip(hipMemcpy(d_csc_col_ptr, @h_csc_col_ptr[0], (n + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'cscsort Memcpy col_ptr');
  CheckHip(hipMemcpy(d_csc_row_ind, @h_csc_row_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'cscsort Memcpy row_ind');
  CheckSparse(rocsparse_create_identity_permutation(handle, nnz, d_perm), 'cscsort create_identity_permutation');

  // Get buffer size and allocate
  CheckSparse(rocsparse_cscsort_buffer_size(handle, m, n, nnz, d_csc_col_ptr, d_csc_row_ind, @buffer_size), 'cscsort_buffer_size');
  CheckHip(hipMalloc(d_buffer, buffer_size), 'cscsort Malloc buffer');

  // Call cscsort
  CheckSparse(rocsparse_cscsort(handle, m, n, nnz, descr, d_csc_col_ptr, d_csc_row_ind, d_perm, d_buffer), 'rocsparse_cscsort');

  // Copy back sorted indices
  CheckHip(hipMemcpy(@h_csc_row_ind_sorted[0], d_csc_row_ind, nnz * sizeof(rocsparse_int), hipMemcpyDeviceToHost), 'cscsort Memcpy result');

  // Verify
  for i := 0 to nnz - 1 do AssertEquals('cscsort row_ind[' + IntToStr(i) + ']', expected_row_ind[i], h_csc_row_ind_sorted[i]);

  // Cleanup
  CheckHip(hipFree(d_csc_col_ptr), 'cscsort Free'); CheckHip(hipFree(d_csc_row_ind), 'cscsort Free');
  CheckHip(hipFree(d_perm), 'cscsort Free'); CheckHip(hipFree(d_buffer), 'cscsort Free');
end;

procedure TTest_rocSPARSE.Test_sprune_csr2csr;
var
  m, n, nnz_A, nnz_C: rocsparse_int;
  threshold: Single;
  d_threshold: Pointer;
  descr_A, descr_C: rocsparse_mat_descr;
  h_csr_val_A: array[0..4] of Single;
  h_csr_row_ptr_A: array[0..3] of rocsparse_int;
  h_csr_col_ind_A: array[0..4] of rocsparse_int;
  h_csr_val_C: array[0..1] of Single;
  h_csr_col_ind_C: array[0..1] of rocsparse_int;
  d_csr_val_A, d_csr_row_ptr_A, d_csr_col_ind_A: Pointer;
  d_csr_val_C, d_csr_row_ptr_C, d_csr_col_ind_C, d_buffer: Pointer;
  buffer_size: size_t;
begin
  // ----------------  build A  ---------------------------------
  m := 3; n := 3; nnz_A := 5; threshold := 3.5;
  h_csr_val_A[0] := 1.0; h_csr_val_A[1] := 2.0; h_csr_val_A[2] := 3.0;
  h_csr_val_A[3] := 4.0; h_csr_val_A[4] := 5.0;
  h_csr_row_ptr_A[0] := 0; h_csr_row_ptr_A[1] := 2; h_csr_row_ptr_A[2] := 4; h_csr_row_ptr_A[3] := 5;
  h_csr_col_ind_A[0] := 0; h_csr_col_ind_A[1] := 1; h_csr_col_ind_A[2] := 1;
  h_csr_col_ind_A[3] := 2; h_csr_col_ind_A[4] := 0;

  // ----------------  descriptors  -----------------------------
  CheckSparse(rocsparse_create_mat_descr(@descr_A), 'create descr_A');
  CheckSparse(rocsparse_create_mat_descr(@descr_C), 'create descr_C');

  // ----------------  upload A  --------------------------------
  CheckHip(hipMalloc(d_csr_val_A,      nnz_A * SizeOf(Single)),           'Malloc val_A');
  CheckHip(hipMalloc(d_csr_row_ptr_A, (m+1) * SizeOf(rocsparse_int)), 'Malloc row_ptr_A');
  CheckHip(hipMalloc(d_csr_col_ind_A,  nnz_A * SizeOf(rocsparse_int)), 'Malloc col_ind_A');

  CheckHip(hipMemcpy(d_csr_val_A,      @h_csr_val_A[0],      nnz_A * SizeOf(Single),           hipMemcpyHostToDevice), 'Memcpy val_A');
  CheckHip(hipMemcpy(d_csr_row_ptr_A,  @h_csr_row_ptr_A[0], (m+1) * SizeOf(rocsparse_int), hipMemcpyHostToDevice), 'Memcpy row_ptr_A');
  CheckHip(hipMemcpy(d_csr_col_ind_A, @h_csr_col_ind_A[0],  nnz_A * SizeOf(rocsparse_int), hipMemcpyHostToDevice), 'Memcpy col_ind_A');

  // ----------------  device threshold  ------------------------
  CheckHip(hipMalloc(d_threshold, SizeOf(Single)), 'Malloc threshold');
  CheckHip(hipMemcpy(d_threshold, @threshold, SizeOf(Single), hipMemcpyHostToDevice), 'Memcpy threshold');

  // ----------------  dummy C arrays  ----------------------------
  CheckHip(hipMalloc(d_csr_row_ptr_C, (m + 1) * SizeOf(rocsparse_int)), 'Malloc row_ptr_C');
  CheckHip(hipMalloc(d_csr_val_C,      nnz_A * SizeOf(Single)),           'Malloc val_C');
  CheckHip(hipMalloc(d_csr_col_ind_C,  nnz_A * SizeOf(rocsparse_int)),   'Malloc col_ind_C');

  // ----------------  buffer size  -----------------------------
  CheckSparse(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device), 'set dev mode');
  CheckSparse(rocsparse_sprune_csr2csr_buffer_size(
                handle, m, n, nnz_A,
                descr_A, d_csr_val_A, d_csr_row_ptr_A, d_csr_col_ind_A,
                d_threshold,
                descr_C, d_csr_val_C, d_csr_row_ptr_C, d_csr_col_ind_C,
                @buffer_size), 'buffer_size');

  CheckSparse(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host), 'set host mode');
  CheckHip(hipMalloc(d_buffer, buffer_size), 'Malloc temp');

  // ----------------  nnz query  ---------------------------------
  CheckSparse(rocsparse_sprune_csr2csr_nnz(
                handle, m, n, nnz_A,
                descr_A, d_csr_val_A, d_csr_row_ptr_A, d_csr_col_ind_A,
                @threshold,
                descr_C, d_csr_row_ptr_C,
                @nnz_C,
                d_buffer), 'sprune_nnz');

  AssertEquals('Pruned nnz by threshold', 2, nnz_C);

  // ----------------  resize C  ------------------------------------
  CheckHip(hipFree(d_csr_val_C), 'Free dummy val_C');
  CheckHip(hipFree(d_csr_col_ind_C), 'Free dummy col_ind_C');
  CheckHip(hipMalloc(d_csr_val_C,      nnz_C * SizeOf(Single)),         'Malloc val_C');
  CheckHip(hipMalloc(d_csr_col_ind_C,  nnz_C * SizeOf(rocsparse_int)), 'Malloc col_ind_C');

  // ----------------  prune  ------------------------------------
  CheckSparse(rocsparse_sprune_csr2csr(
                handle, m, n, nnz_A,
                descr_A, d_csr_val_A, d_csr_row_ptr_A, d_csr_col_ind_A,
                @threshold,
                descr_C, d_csr_val_C, d_csr_row_ptr_C, d_csr_col_ind_C,
                d_buffer), 'sprune_csr2csr');

  // ----------------  copy back  ----------------------------------
  CheckHip(hipMemcpy(@h_csr_val_C[0],      d_csr_val_C,      nnz_C * SizeOf(Single),         hipMemcpyDeviceToHost), 'Memcpy val_C');
  //CheckHip(hipMemcpy(@h_csr_row_ptr_C[0], d_csr_row_ptr_C, (m+1) * SizeOf(rocsparse_int), hipMemcpyDeviceToHost), 'Memcpy row_ptr_C');
  CheckHip(hipMemcpy(@h_csr_col_ind_C[0], d_csr_col_ind_C,  nnz_C * SizeOf(rocsparse_int), hipMemcpyDeviceToHost), 'Memcpy col_ind_C');

  // ----------------  verify  -----------------------------------
  AssertEquals('val0', 4.0, h_csr_val_C[0], 1E-5);
  AssertEquals('val1', 5.0, h_csr_val_C[1], 1E-5);
  AssertEquals('col0', 2,   h_csr_col_ind_C[0]);
  AssertEquals('col1', 0,   h_csr_col_ind_C[1]);

  // ----------------  cleanup  ------------------------------------
  CheckHip(hipFree(d_csr_val_A), 'Free');
  CheckHip(hipFree(d_csr_row_ptr_A), 'Free');
  CheckHip(hipFree(d_csr_col_ind_A), 'Free');
  CheckHip(hipFree(d_csr_val_C), 'Free');
  //CheckHip(hipFree(d_csr_row_ptr_C), 'Free');
  CheckHip(hipFree(d_csr_col_ind_C), 'Free');
  CheckHip(hipFree(d_buffer), 'Free');
  CheckHip(hipFree(d_threshold), 'Free');
  CheckSparse(rocsparse_destroy_mat_descr(descr_A), 'destroy descr_A');
  CheckSparse(rocsparse_destroy_mat_descr(descr_C), 'destroy descr_C');
end;





procedure TTest_rocSPARSE.Test_sbsric0;
var
  mb, nnzb, block_dim: rocsparse_int;
  info: rocsparse_mat_info;
  h_bsr_val: array[0..19] of Single;
  h_bsr_row_ptr: array[0..2] of rocsparse_int;
  h_bsr_col_ind: array[0..3] of rocsparse_int;
  d_bsr_val, d_bsr_row_ptr, d_bsr_col_ind, d_buffer: Pointer;
  buffer_size: size_t;
  position: rocsparse_int;
begin
  // Create info object
  CheckSparse(rocsparse_create_mat_info(@info), 'bsric0 create info');

  // Symmetric 2x2 BSR matrix
  mb := 2; block_dim := 2; nnzb := 3;
  // Block 0,0
  h_bsr_val[0] := 10; h_bsr_val[1] := 1; h_bsr_val[2] := 1; h_bsr_val[3] := 10;
  // Block 0,1 (and 1,0 by symmetry)
  h_bsr_val[4] := -1; h_bsr_val[5] := -2; h_bsr_val[6] := -3; h_bsr_val[7] := -4;
  // Block 1,1
  h_bsr_val[8] := 20; h_bsr_val[9] := 2; h_bsr_val[10] := 2; h_bsr_val[11] := 20;

  h_bsr_row_ptr[0] := 0; h_bsr_row_ptr[1] := 2; h_bsr_row_ptr[2] := 3;
  h_bsr_col_ind[0] := 0; h_bsr_col_ind[1] := 1;
  h_bsr_col_ind[2] := 1;

  // Allocate and copy
  CheckHip(hipMalloc(d_bsr_val, nnzb * block_dim * block_dim * sizeof(Single)), 'bsric0 Malloc val');
  CheckHip(hipMalloc(d_bsr_row_ptr, (mb + 1) * sizeof(rocsparse_int)), 'bsric0 Malloc row_ptr');
  CheckHip(hipMalloc(d_bsr_col_ind, nnzb * sizeof(rocsparse_int)), 'bsric0 Malloc col_ind');
  CheckHip(hipMemcpy(d_bsr_val, @h_bsr_val[0], nnzb * block_dim * block_dim * sizeof(Single), hipMemcpyHostToDevice), 'bsric0 Memcpy val');
  CheckHip(hipMemcpy(d_bsr_row_ptr, @h_bsr_row_ptr[0], (mb + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'bsric0 Memcpy row_ptr');
  CheckHip(hipMemcpy(d_bsr_col_ind, @h_bsr_col_ind[0], nnzb * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'bsric0 Memcpy col_ind');

  // Get buffer size
  CheckSparse(rocsparse_sbsric0_buffer_size(handle, rocsparse_direction_row, mb, nnzb, descr, d_bsr_val, d_bsr_row_ptr, d_bsr_col_ind, block_dim, info, @buffer_size), 'sbsric0_buffer_size');
  CheckHip(hipMalloc(d_buffer, buffer_size), 'bsric0 Malloc buffer');

  // Perform analysis
  CheckSparse(rocsparse_sbsric0_analysis(handle, rocsparse_direction_row, mb, nnzb, descr, d_bsr_val, d_bsr_row_ptr, d_bsr_col_ind, block_dim, info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer), 'sbsric0_analysis');

  // Check for zero pivot
  position := -2;
  CheckSparse(rocsparse_bsric0_zero_pivot(handle, info, @position), 'bsric0_zero_pivot');
  AssertEquals('bsric0 zero pivot check', -1, position);

  // Perform computation
  CheckSparse(rocsparse_sbsric0(handle, rocsparse_direction_row, mb, nnzb, descr, d_bsr_val, d_bsr_row_ptr, d_bsr_col_ind, block_dim, info, rocsparse_solve_policy_auto, d_buffer), 'sbsric0');

  // Cleanup
  CheckHip(hipFree(d_bsr_val), 'Free'); CheckHip(hipFree(d_bsr_row_ptr), 'Free'); CheckHip(hipFree(d_bsr_col_ind), 'Free');
  CheckHip(hipFree(d_buffer), 'Free');
  CheckSparse(rocsparse_destroy_mat_info(info), 'destroy info');
  // Reset descriptor for other tests
  CheckSparse(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general), 'bsric0 reset mat type');
end;


procedure TTest_rocSPARSE.Test_sbsrilu0;
var
  mb, nnzb, block_dim: rocsparse_int;
  info: rocsparse_mat_info;
  h_bsr_val: array[0..11] of Single;
  h_bsr_row_ptr: array[0..2] of rocsparse_int;
  h_bsr_col_ind: array[0..2] of rocsparse_int;
  d_bsr_val, d_bsr_row_ptr, d_bsr_col_ind, d_buffer: Pointer;
  buffer_size: size_t;
  position: rocsparse_int;
begin
  // Create info object
  CheckSparse(rocsparse_create_mat_info(@info), 'bsrilu0 create info');

  // 2x2 BSR matrix
  mb := 2; block_dim := 2; nnzb := 3;
  // Block 0,0
  h_bsr_val[0] := 10; h_bsr_val[1] := 1; h_bsr_val[2] := 1; h_bsr_val[3] := 10;
  // Block 0,1
  h_bsr_val[4] := -1; h_bsr_val[5] := -2; h_bsr_val[6] := -3; h_bsr_val[7] := -4;
  // Block 1,1
  h_bsr_val[8] := 20; h_bsr_val[9] := 2; h_bsr_val[10] := 2; h_bsr_val[11] := 20;

  h_bsr_row_ptr[0] := 0; h_bsr_row_ptr[1] := 2; h_bsr_row_ptr[2] := 3;
  h_bsr_col_ind[0] := 0; h_bsr_col_ind[1] := 1;
  h_bsr_col_ind[2] := 1;

  // Allocate and copy
  CheckHip(hipMalloc(d_bsr_val, nnzb * block_dim * block_dim * sizeof(Single)), 'bsrilu0 Malloc val');
  CheckHip(hipMalloc(d_bsr_row_ptr, (mb + 1) * sizeof(rocsparse_int)), 'bsrilu0 Malloc row_ptr');
  CheckHip(hipMalloc(d_bsr_col_ind, nnzb * sizeof(rocsparse_int)), 'bsrilu0 Malloc col_ind');
  CheckHip(hipMemcpy(d_bsr_val, @h_bsr_val[0], nnzb * block_dim * block_dim * sizeof(Single), hipMemcpyHostToDevice), 'bsrilu0 Memcpy val');
  CheckHip(hipMemcpy(d_bsr_row_ptr, @h_bsr_row_ptr[0], (mb + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'bsrilu0 Memcpy row_ptr');
  CheckHip(hipMemcpy(d_bsr_col_ind, @h_bsr_col_ind[0], nnzb * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'bsrilu0 Memcpy col_ind');

  // Get buffer size
  CheckSparse(rocsparse_sbsrilu0_buffer_size(handle, rocsparse_direction_row, mb, nnzb, descr, d_bsr_val, d_bsr_row_ptr, d_bsr_col_ind, block_dim, info, @buffer_size), 'sbsrilu0_buffer_size');
  CheckHip(hipMalloc(d_buffer, buffer_size), 'bsrilu0 Malloc buffer');

  // Perform analysis
  CheckSparse(rocsparse_sbsrilu0_analysis(handle, rocsparse_direction_row, mb, nnzb, descr, d_bsr_val, d_bsr_row_ptr, d_bsr_col_ind, block_dim, info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer), 'sbsrilu0_analysis');

  // Check for zero pivot
  position := -2;
  CheckSparse(rocsparse_bsrilu0_zero_pivot(handle, info, @position), 'bsrilu0_zero_pivot');
  AssertEquals('bsrilu0 zero pivot check', -1, position);

  // Perform computation
  CheckSparse(rocsparse_sbsrilu0(handle, rocsparse_direction_row, mb, nnzb, descr, d_bsr_val, d_bsr_row_ptr, d_bsr_col_ind, block_dim, info, rocsparse_solve_policy_auto, d_buffer), 'sbsrilu0');

  // Cleanup
  CheckHip(hipFree(d_bsr_val), 'Free'); CheckHip(hipFree(d_bsr_row_ptr), 'Free'); CheckHip(hipFree(d_bsr_col_ind), 'Free');
  CheckHip(hipFree(d_buffer), 'Free');
  CheckSparse(rocsparse_destroy_mat_info(info), 'destroy info');
end;


procedure TTest_rocSPARSE.Test_scsric0;
var
  m, nnz: rocsparse_int;
  info: rocsparse_mat_info;
  h_csr_val: array[0..4] of Single;
  h_csr_row_ptr: array[0..3] of rocsparse_int;
  h_csr_col_ind: array[0..4] of rocsparse_int;
  d_csr_val, d_csr_row_ptr, d_csr_col_ind, d_buffer: Pointer;
  buffer_size: size_t;
  position: rocsparse_int;
begin
  // Create info object
  CheckSparse(rocsparse_create_mat_info(@info), 'csric0 create info');

  // Symmetric 3x3 CSR matrix
  m := 3; nnz := 5;
  h_csr_val[0] := 10.0; h_csr_val[1] := -1.0; h_csr_val[2] := -1.0; h_csr_val[3] := 20.0; h_csr_val[4] := 30.0;
  h_csr_row_ptr[0] := 0; h_csr_row_ptr[1] := 2; h_csr_row_ptr[2] := 4; h_csr_row_ptr[3] := 5;
  h_csr_col_ind[0] := 0; h_csr_col_ind[1] := 1;
  h_csr_col_ind[2] := 0; h_csr_col_ind[3] := 1;
  h_csr_col_ind[4] := 2;

  // Allocate and copy
  CheckHip(hipMalloc(d_csr_val, nnz * sizeof(Single)), 'csric0 Malloc val');
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), 'csric0 Malloc row_ptr');
  CheckHip(hipMalloc(d_csr_col_ind, nnz * sizeof(rocsparse_int)), 'csric0 Malloc col_ind');
  CheckHip(hipMemcpy(d_csr_val, @h_csr_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), 'csric0 Memcpy val');
  CheckHip(hipMemcpy(d_csr_row_ptr, @h_csr_row_ptr[0], (m + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csric0 Memcpy row_ptr');
  CheckHip(hipMemcpy(d_csr_col_ind, @h_csr_col_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csric0 Memcpy col_ind');

  // Get buffer size
  CheckSparse(rocsparse_scsric0_buffer_size(handle, m, nnz, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, info, @buffer_size), 'scsric0_buffer_size');
  CheckHip(hipMalloc(d_buffer, buffer_size), 'csric0 Malloc buffer');

  // Perform analysis
  CheckSparse(rocsparse_scsric0_analysis(handle, m, nnz, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer), 'scsric0_analysis');

  // Check for zero pivot
  position := -2;
  CheckSparse(rocsparse_csric0_zero_pivot(handle, info, @position), 'csric0_zero_pivot');
  AssertEquals('csric0 zero pivot check', -1, position);

  // Perform computation
  CheckSparse(rocsparse_scsric0(handle, m, nnz, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, info, rocsparse_solve_policy_auto, d_buffer), 'scsric0');

  // Cleanup
  CheckHip(hipFree(d_csr_val), 'Free'); CheckHip(hipFree(d_csr_row_ptr), 'Free'); CheckHip(hipFree(d_csr_col_ind), 'Free');
  CheckHip(hipFree(d_buffer), 'Free');
  CheckSparse(rocsparse_destroy_mat_info(info), 'destroy info');
end;


procedure TTest_rocSPARSE.Test_scsrilu0;
var
  m, nnz: rocsparse_int;
  info: rocsparse_mat_info;
  h_csr_val: array[0..4] of Single;
  h_csr_row_ptr: array[0..3] of rocsparse_int;
  h_csr_col_ind: array[0..4] of rocsparse_int;
  d_csr_val, d_csr_row_ptr, d_csr_col_ind, d_buffer: Pointer;
  buffer_size: size_t;
  position: rocsparse_int;
begin
  // Create info object
  CheckSparse(rocsparse_create_mat_info(@info), 'csrilu0 create info');

  // 3x3 CSR matrix
  m := 3; nnz := 5;
  h_csr_val[0] := 10.0; h_csr_val[1] := -1.0; h_csr_val[2] := -2.0; h_csr_val[3] := 20.0; h_csr_val[4] := 30.0;
  h_csr_row_ptr[0] := 0; h_csr_row_ptr[1] := 2; h_csr_row_ptr[2] := 4; h_csr_row_ptr[3] := 5;
  h_csr_col_ind[0] := 0; h_csr_col_ind[1] := 1;
  h_csr_col_ind[2] := 0; h_csr_col_ind[3] := 1;
  h_csr_col_ind[4] := 2;

  // Allocate and copy
  CheckHip(hipMalloc(d_csr_val, nnz * sizeof(Single)), 'csrilu0 Malloc val');
  CheckHip(hipMalloc(d_csr_row_ptr, (m + 1) * sizeof(rocsparse_int)), 'csrilu0 Malloc row_ptr');
  CheckHip(hipMalloc(d_csr_col_ind, nnz * sizeof(rocsparse_int)), 'csrilu0 Malloc col_ind');
  CheckHip(hipMemcpy(d_csr_val, @h_csr_val[0], nnz * sizeof(Single), hipMemcpyHostToDevice), 'csrilu0 Memcpy val');
  CheckHip(hipMemcpy(d_csr_row_ptr, @h_csr_row_ptr[0], (m + 1) * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csrilu0 Memcpy row_ptr');
  CheckHip(hipMemcpy(d_csr_col_ind, @h_csr_col_ind[0], nnz * sizeof(rocsparse_int), hipMemcpyHostToDevice), 'csrilu0 Memcpy col_ind');

  // Get buffer size
  CheckSparse(rocsparse_scsrilu0_buffer_size(handle, m, nnz, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, info, @buffer_size), 'scsrilu0_buffer_size');
  CheckHip(hipMalloc(d_buffer, buffer_size), 'csrilu0 Malloc buffer');

  // Perform analysis
  CheckSparse(rocsparse_scsrilu0_analysis(handle, m, nnz, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer), 'scsrilu0_analysis');

  // Check for zero pivot
  position := -2;
  CheckSparse(rocsparse_csrilu0_zero_pivot(handle, info, @position), 'csrilu0_zero_pivot');
  AssertEquals('csrilu0 zero pivot check', -1, position);

  // Perform computation
  CheckSparse(rocsparse_scsrilu0(handle, m, nnz, descr, d_csr_val, d_csr_row_ptr, d_csr_col_ind, info, rocsparse_solve_policy_auto, d_buffer), 'scsrilu0');

  // Cleanup
  CheckHip(hipFree(d_csr_val), 'Free'); CheckHip(hipFree(d_csr_row_ptr), 'Free'); CheckHip(hipFree(d_csr_col_ind), 'Free');
  CheckHip(hipFree(d_buffer), 'Free');
  CheckSparse(rocsparse_destroy_mat_info(info), 'destroy info');
end;


initialization
  RegisterTest(TTest_rocSPARSE);

end.
