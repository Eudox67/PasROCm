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

unit Test_rocfft;

{$mode objfpc}{$H+}
//{$linklib rocfft}
//{$DEFINE ROCM_6_3_2}

interface

uses
  fpcunit, testregistry, hip_runtime, hip_types, rocfft;

type
  TTest_rocFFT = class(TTestCase)
  published
    procedure TestSetupAndCleanup;
    procedure TestPlanCreateAndDestroy;
    procedure TestPlanDescriptionCreateAndDestroy;
    procedure TestPlanDescriptionSetScaleFactor;
    procedure TestPlanDescriptionSetDataLayout;
    procedure TestPlanDescriptionSetComm;
    procedure TestExecutionInfoCreateAndDestroy;
    procedure TestExecutionInfoSetWorkBuffer;
    procedure TestExecutionInfoSetStream;
    procedure TestExecutionInfoSetLoadCallback;
    procedure TestExecutionInfoSetStoreCallback;
    procedure TestPlanGetWorkBufferSize;
    procedure TestFieldAndBrickHelpers;
    procedure TestCacheManagement;
    procedure TestGetVersionString;
    procedure TestFFT_SinglePrecision_ComplexForward;
    procedure TestFFT_DoublePrecision_ComplexForward;
    procedure TestFFT_SinglePrecision_ComplexInverse;
    procedure TestFFT_DoublePrecision_ComplexInverse;
    procedure TestFFT_SinglePrecision_RealForward;
    procedure TestFFT_DoublePrecision_RealForward;
    procedure TestFFT_SinglePrecision_RealInverse;
    procedure TestFFT_DoublePrecision_RealInverse;
    // Add more test methods as needed
  end;

implementation

procedure TTest_rocFFT.TestSetupAndCleanup;
var
  st_setup: rocfft_status;
begin
  st_setup := rocfft_setup;
  AssertEquals('rocfft_setup should return success', rocfft_status_success, st_setup);
  st_setup := rocfft_cleanup;
  AssertEquals('rocfft_cleanup should return success', rocfft_status_success, st_setup);
end;

procedure TTest_rocFFT.TestPlanCreateAndDestroy;
var
  plan: rocfft_plan;
  lengths: array[0..0] of UIntPtr;
  st_plan: rocfft_status;
begin
  lengths[0] := 8; // 1D FFT of length 8
  st_plan := rocfft_setup;
  AssertEquals('rocfft_setup should return success', rocfft_status_success, st_plan);
  st_plan := rocfft_plan_create(plan, rocfft_placement_inplace, rocfft_transform_type_complex_forward, rocfft_precision_single, 1, @lengths[0], 1, nil);
  AssertEquals('rocfft_plan_create should return success', rocfft_status_success, st_plan);
  st_plan := rocfft_plan_destroy(plan);
  AssertEquals('rocfft_plan_destroy should return success', rocfft_status_success, st_plan);
  st_plan := rocfft_cleanup;
  AssertEquals('rocfft_cleanup should return success', rocfft_status_success, st_plan);
end;

procedure TTest_rocFFT.TestPlanDescriptionCreateAndDestroy;
var
  desc: rocfft_plan_description;
  st_desc: rocfft_status;
begin
  st_desc := rocfft_plan_description_create(desc);
  AssertEquals('rocfft_plan_description_create should return success', rocfft_status_success, st_desc);
  st_desc := rocfft_plan_description_destroy(desc);
  AssertEquals('rocfft_plan_description_destroy should return success', rocfft_status_success, st_desc);
end;

procedure TTest_rocFFT.TestPlanDescriptionSetScaleFactor;
var
  desc: rocfft_plan_description;
  st_scale: rocfft_status;
begin
  st_scale := rocfft_plan_description_create(desc);
  AssertEquals('create desc', rocfft_status_success, st_scale);
  st_scale := rocfft_plan_description_set_scale_factor(desc, 2.0);
  AssertEquals('set scale factor', rocfft_status_success, st_scale);
  st_scale := rocfft_plan_description_destroy(desc);
  AssertEquals('destroy desc', rocfft_status_success, st_scale);
end;

procedure TTest_rocFFT.TestPlanDescriptionSetDataLayout;
var
  desc: rocfft_plan_description;
  st_layout: rocfft_status;
  in_offsets, out_offsets, in_strides, out_strides: array[0..0] of UIntPtr;
begin
  st_layout := rocfft_plan_description_create(desc);
  AssertEquals('create desc', rocfft_status_success, st_layout);
  in_offsets[0] := 0; out_offsets[0] := 0;
  in_strides[0] := 1; out_strides[0] := 1;
  st_layout := rocfft_plan_description_set_data_layout(desc,
    rocfft_array_type_complex_interleaved, rocfft_array_type_complex_interleaved,
    @in_offsets[0], @out_offsets[0],
    1, @in_strides[0], 1,
    1, @out_strides[0], 1);
  AssertEquals('set data layout', rocfft_status_success, st_layout);
  st_layout := rocfft_plan_description_destroy(desc);
  AssertEquals('destroy desc', rocfft_status_success, st_layout);
end;

procedure TTest_rocFFT.TestPlanDescriptionSetComm;
var
  desc: rocfft_plan_description;
  st_comm: rocfft_status;
begin
  st_comm := rocfft_plan_description_create(desc);
  AssertEquals('create desc', rocfft_status_success, st_comm);
  st_comm := rocfft_plan_description_set_comm(desc, rocfft_comm_none, nil);
  AssertEquals('set comm', rocfft_status_success, st_comm);
  st_comm := rocfft_plan_description_destroy(desc);
  AssertEquals('destroy desc', rocfft_status_success, st_comm);
end;

procedure TTest_rocFFT.TestExecutionInfoCreateAndDestroy;
var
  info: rocfft_execution_info;
  st_info: rocfft_status;
begin
  st_info := rocfft_execution_info_create(info);
  AssertEquals('create info', rocfft_status_success, st_info);
  st_info := rocfft_execution_info_destroy(info);
  AssertEquals('destroy info', rocfft_status_success, st_info);
end;

procedure TTest_rocFFT.TestExecutionInfoSetWorkBuffer;
var
  info: rocfft_execution_info;
  st_work: rocfft_status;
  buf: Pointer;
begin
  st_work := rocfft_execution_info_create(info);
  AssertEquals('create info', rocfft_status_success, st_work);
  buf := nil;
  // rocFFT accepts (NULL, 0) as "no work buffer supplied"...
  st_work := rocfft_execution_info_set_work_buffer(info, buf, 0);
  AssertEquals('set work buffer (NULL+0 is accepted)', rocfft_status_success, st_work);
  // ...and rejects a NULL buffer when a non-zero size is requested.
  st_work := rocfft_execution_info_set_work_buffer(info, buf, 64);
  AssertEquals('set work buffer (NULL+64 is invalid)', rocfft_status_invalid_work_buffer, st_work);
  st_work := rocfft_execution_info_destroy(info);
  AssertEquals('destroy info', rocfft_status_success, st_work);
end;

procedure TTest_rocFFT.TestExecutionInfoSetStream;
var
  info: rocfft_execution_info;
  st_stream: rocfft_status;
  stream: hipStream_t;
begin
  st_stream := rocfft_execution_info_create(info);
  AssertEquals('create info', rocfft_status_success, st_stream);
  stream := nil;
  st_stream := rocfft_execution_info_set_stream(info, stream);
  AssertEquals('set stream', rocfft_status_success, st_stream);
  st_stream := rocfft_execution_info_destroy(info);
  AssertEquals('destroy info', rocfft_status_success, st_stream);
end;

procedure TTest_rocFFT.TestExecutionInfoSetLoadCallback;
var
  info: rocfft_execution_info;
  st_load: rocfft_status;
  cb_functions, cb_data: Pointer;
begin
  st_load := rocfft_execution_info_create(info);
  AssertEquals('create info', rocfft_status_success, st_load);
  cb_functions := nil; cb_data := nil;
  st_load := rocfft_execution_info_set_load_callback(info, @cb_functions, @cb_data, 0);
  AssertEquals('set load callback', rocfft_status_success, st_load);
  st_load := rocfft_execution_info_destroy(info);
  AssertEquals('destroy info', rocfft_status_success, st_load);
end;

procedure TTest_rocFFT.TestExecutionInfoSetStoreCallback;
var
  info: rocfft_execution_info;
  st_store: rocfft_status;
  cb_functions, cb_data: Pointer;
begin
  st_store := rocfft_execution_info_create(info);
  AssertEquals('create info', rocfft_status_success, st_store);
  cb_functions := nil; cb_data := nil;
  st_store := rocfft_execution_info_set_store_callback(info, @cb_functions, @cb_data, 0);
  AssertEquals('set store callback', rocfft_status_success, st_store);
  st_store := rocfft_execution_info_destroy(info);
  AssertEquals('destroy info', rocfft_status_success, st_store);
end;

procedure TTest_rocFFT.TestPlanGetWorkBufferSize;
var
  plan: rocfft_plan;
  lengths: array[0..0] of UIntPtr;
  st_buf: rocfft_status;
  size: UIntPtr;
begin
  lengths[0] := 8;
  st_buf := rocfft_setup;
  AssertEquals('setup', rocfft_status_success, st_buf);
  st_buf := rocfft_plan_create(plan, rocfft_placement_inplace, rocfft_transform_type_complex_forward, rocfft_precision_single, 1, @lengths[0], 1, nil);
  AssertEquals('plan create', rocfft_status_success, st_buf);
  size := 0;
  st_buf := rocfft_plan_get_work_buffer_size(plan, size);
  AssertEquals('get work buffer size', rocfft_status_success, st_buf);
  st_buf := rocfft_plan_destroy(plan);
  AssertEquals('plan destroy', rocfft_status_success, st_buf);
  st_buf := rocfft_cleanup;
  AssertEquals('cleanup', rocfft_status_success, st_buf);
end;

procedure TTest_rocFFT.TestFieldAndBrickHelpers;
var
  field: rocfft_field;
  brick: rocfft_brick;
  st_field: rocfft_status;
  lower, upper, stride: array[0..1] of UIntPtr;
begin
  st_field := rocfft_field_create(field);
  AssertEquals('field create', rocfft_status_success, st_field);
  lower[0] := 0; lower[1] := 0;
  upper[0] := 7; upper[1] := 7;
  stride[0] := 1; stride[1] := 1;
  // dim_with_batch must be the FFT dimension count plus one batch dimension,
  // so a value below 2 is rejected by rocFFT.
  st_field := rocfft_brick_create(brick, @lower[0], @upper[0], @stride[0], 1, 0);
  AssertEquals('brick create rejects dim_with_batch < 2', rocfft_status_invalid_dimensions, st_field);
  st_field := rocfft_brick_create(brick, @lower[0], @upper[0], @stride[0], 2, 0);
  AssertEquals('brick create', rocfft_status_success, st_field);
  st_field := rocfft_field_add_brick(field, brick);
  AssertEquals('field add brick', rocfft_status_success, st_field);
  st_field := rocfft_brick_destroy(brick);
  AssertEquals('brick destroy', rocfft_status_success, st_field);
  st_field := rocfft_field_destroy(field);
  AssertEquals('field destroy', rocfft_status_success, st_field);
end;

procedure TTest_rocFFT.TestCacheManagement;
var
  buffer: Pointer;
  buffer_len: UIntPtr;
  st_cache: rocfft_status;
begin
  buffer := nil; buffer_len := 0;
  st_cache := rocfft_cache_serialize(buffer, buffer_len);
  AssertTrue('cache serialize returns success or failure', (st_cache = rocfft_status_success) or (st_cache = rocfft_status_failure));
  if buffer <> nil then
  begin
    st_cache := rocfft_cache_buffer_free(buffer);
    AssertEquals('cache buffer free', rocfft_status_success, st_cache);
  end;
  // cache_deserialize is not tested as it requires a valid buffer
end;

procedure TTest_rocFFT.TestGetVersionString;
var
  buf: array[0..127] of Char;
  st_ver: rocfft_status;
begin
  st_ver := rocfft_get_version_string(@buf[0], 128);
  AssertEquals('get version string', rocfft_status_success, st_ver);
end;

// --- Calculation/Precision Tests ---

procedure TTest_rocFFT.TestFFT_SinglePrecision_ComplexForward;
var
  plan: rocfft_plan;
  info: rocfft_execution_info;
  lengths: array[0..0] of UIntPtr;
  in_host, out_host: array[0..15] of Single;
  in_dev: Pointer;
  st: rocfft_status;
  i: Integer;
begin
  lengths[0] := 8;
  // Input: impulse (1 at index 0, 0 elsewhere)
  for i := 0 to 15 do begin
    if i = 0 then in_host[i] := 1.0 else in_host[i] := 0.0;
    out_host[i] := 0.0;
  end;
  st := hipMalloc(in_dev, 16 * SizeOf(Single)); AssertEquals('hipMalloc', HIP_SUCCESS, st);
  st := hipMemcpy(in_dev, @in_host[0], 16 * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy', HIP_SUCCESS, st);
  st := rocfft_setup; AssertEquals('rocfft_setup', rocfft_status_success, st);
  st := rocfft_plan_create(plan, rocfft_placement_inplace, rocfft_transform_type_complex_forward, rocfft_precision_single, 1, @lengths[0], 1, nil); AssertEquals('plan_create', rocfft_status_success, st);
  st := rocfft_execution_info_create(info); AssertEquals('info_create', rocfft_status_success, st);
  st := rocfft_execute(plan, @in_dev, @in_dev, info); AssertEquals('execute', rocfft_status_success, st);
  st := hipMemcpy(@out_host[0], in_dev, 16 * SizeOf(Single), hipMemcpyDeviceToHost); AssertEquals('hipMemcpy out', HIP_SUCCESS, st);
  st := rocfft_plan_destroy(plan); AssertEquals('plan_destroy', rocfft_status_success, st);
  st := rocfft_execution_info_destroy(info); AssertEquals('info_destroy', rocfft_status_success, st);
  st := hipFree(in_dev); AssertEquals('hipFree', HIP_SUCCESS, st);
  st := rocfft_cleanup; AssertEquals('rocfft_cleanup', rocfft_status_success, st);
  // FFT of impulse should be all 1s (real), 0s (imag)
  for i := 0 to 7 do
    AssertEquals('FFT impulse real', 1.0, out_host[2*i], 1e-5);
  for i := 0 to 7 do
    AssertEquals('FFT impulse imag', 0.0, out_host[2*i+1], 1e-5);
end;

procedure TTest_rocFFT.TestFFT_DoublePrecision_ComplexForward;
var
  plan: rocfft_plan;
  info: rocfft_execution_info;
  lengths: array[0..0] of UIntPtr;
  in_host, out_host: array[0..15] of Double;
  in_dev: Pointer;
  st: rocfft_status;
  i: Integer;
begin
  lengths[0] := 8;
  for i := 0 to 15 do begin
    if i = 0 then in_host[i] := 1.0 else in_host[i] := 0.0;
    out_host[i] := 0.0;
  end;
  st := hipMalloc(in_dev, 16 * SizeOf(Double)); AssertEquals('hipMalloc', HIP_SUCCESS, st);
  st := hipMemcpy(in_dev, @in_host[0], 16 * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy', HIP_SUCCESS, st);
  st := rocfft_setup; AssertEquals('rocfft_setup', rocfft_status_success, st);
  st := rocfft_plan_create(plan, rocfft_placement_inplace, rocfft_transform_type_complex_forward, rocfft_precision_double, 1, @lengths[0], 1, nil); AssertEquals('plan_create', rocfft_status_success, st);
  st := rocfft_execution_info_create(info); AssertEquals('info_create', rocfft_status_success, st);
  st := rocfft_execute(plan, @in_dev, @in_dev, info); AssertEquals('execute', rocfft_status_success, st);
  st := hipMemcpy(@out_host[0], in_dev, 16 * SizeOf(Double), hipMemcpyDeviceToHost); AssertEquals('hipMemcpy out', HIP_SUCCESS, st);
  st := rocfft_plan_destroy(plan); AssertEquals('plan_destroy', rocfft_status_success, st);
  st := rocfft_execution_info_destroy(info); AssertEquals('info_destroy', rocfft_status_success, st);
  st := hipFree(in_dev); AssertEquals('hipFree', HIP_SUCCESS, st);
  st := rocfft_cleanup; AssertEquals('rocfft_cleanup', rocfft_status_success, st);
  for i := 0 to 7 do
    AssertEquals('FFT impulse real', 1.0, out_host[2*i], 1e-12);
  for i := 0 to 7 do
    AssertEquals('FFT impulse imag', 0.0, out_host[2*i+1], 1e-12);
end;

procedure TTest_rocFFT.TestFFT_SinglePrecision_ComplexInverse;
var
  plan: rocfft_plan;
  info: rocfft_execution_info;
  lengths: array[0..0] of UIntPtr;
  in_host, out_host: array[0..15] of Single;
  in_dev: Pointer;
  st: rocfft_status;
  i: Integer;
begin
  lengths[0] := 8;
  // Input: all 1s (real), 0s (imag)
  for i := 0 to 15 do begin
    if i mod 2 = 0 then in_host[i] := 1.0 else in_host[i] := 0.0;
    out_host[i] := 0.0;
  end;
  st := hipMalloc(in_dev, 16 * SizeOf(Single)); AssertEquals('hipMalloc', HIP_SUCCESS, st);
  st := hipMemcpy(in_dev, @in_host[0], 16 * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy', HIP_SUCCESS, st);
  st := rocfft_setup; AssertEquals('rocfft_setup', rocfft_status_success, st);
  st := rocfft_plan_create(plan, rocfft_placement_inplace, rocfft_transform_type_complex_inverse, rocfft_precision_single, 1, @lengths[0], 1, nil); AssertEquals('plan_create', rocfft_status_success, st);
  st := rocfft_execution_info_create(info); AssertEquals('info_create', rocfft_status_success, st);
  st := rocfft_execute(plan, @in_dev, @in_dev, info); AssertEquals('execute', rocfft_status_success, st);
  st := hipMemcpy(@out_host[0], in_dev, 16 * SizeOf(Single), hipMemcpyDeviceToHost); AssertEquals('hipMemcpy out', HIP_SUCCESS, st);
  st := rocfft_plan_destroy(plan); AssertEquals('plan_destroy', rocfft_status_success, st);
  st := rocfft_execution_info_destroy(info); AssertEquals('info_destroy', rocfft_status_success, st);
  st := hipFree(in_dev); AssertEquals('hipFree', HIP_SUCCESS, st);
  st := rocfft_cleanup; AssertEquals('rocfft_cleanup', rocfft_status_success, st);
  // Inverse FFT of all-ones (real) should be impulse (real=8 at index 0, 0 elsewhere)
  AssertEquals('IFFT all-ones real[0]', 8.0, out_host[0], 1e-4);
  AssertEquals('IFFT all-ones imag[0]', 0.0, out_host[1], 1e-4);
  for i := 1 to 7 do begin
    AssertEquals('IFFT all-ones real', 0.0, out_host[2*i], 1e-4);
    AssertEquals('IFFT all-ones imag', 0.0, out_host[2*i+1], 1e-4);
  end;
end;

procedure TTest_rocFFT.TestFFT_DoublePrecision_ComplexInverse;
var
  plan: rocfft_plan;
  info: rocfft_execution_info;
  lengths: array[0..0] of UIntPtr;
  in_host, out_host: array[0..15] of Double;
  in_dev: Pointer;
  st: rocfft_status;
  i: Integer;
begin
  lengths[0] := 8;
  for i := 0 to 15 do begin
    if i mod 2 = 0 then in_host[i] := 1.0 else in_host[i] := 0.0;
    out_host[i] := 0.0;
  end;
  st := hipMalloc(in_dev, 16 * SizeOf(Double)); AssertEquals('hipMalloc', HIP_SUCCESS, st);
  st := hipMemcpy(in_dev, @in_host[0], 16 * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy', HIP_SUCCESS, st);
  st := rocfft_setup; AssertEquals('rocfft_setup', rocfft_status_success, st);
  st := rocfft_plan_create(plan, rocfft_placement_inplace, rocfft_transform_type_complex_inverse, rocfft_precision_double, 1, @lengths[0], 1, nil); AssertEquals('plan_create', rocfft_status_success, st);
  st := rocfft_execution_info_create(info); AssertEquals('info_create', rocfft_status_success, st);
  st := rocfft_execute(plan, @in_dev, @in_dev, info); AssertEquals('execute', rocfft_status_success, st);
  st := hipMemcpy(@out_host[0], in_dev, 16 * SizeOf(Double), hipMemcpyDeviceToHost); AssertEquals('hipMemcpy out', HIP_SUCCESS, st);
  st := rocfft_plan_destroy(plan); AssertEquals('plan_destroy', rocfft_status_success, st);
  st := rocfft_execution_info_destroy(info); AssertEquals('info_destroy', rocfft_status_success, st);
  st := hipFree(in_dev); AssertEquals('hipFree', HIP_SUCCESS, st);
  st := rocfft_cleanup; AssertEquals('rocfft_cleanup', rocfft_status_success, st);
  AssertEquals('IFFT all-ones real[0]', 8.0, out_host[0], 1e-10);
  AssertEquals('IFFT all-ones imag[0]', 0.0, out_host[1], 1e-10);
  for i := 1 to 7 do begin
    AssertEquals('IFFT all-ones real', 0.0, out_host[2*i], 1e-10);
    AssertEquals('IFFT all-ones imag', 0.0, out_host[2*i+1], 1e-10);
  end;
end;

procedure TTest_rocFFT.TestFFT_SinglePrecision_RealForward;
var
  plan: rocfft_plan;
  info: rocfft_execution_info;
  lengths: array[0..0] of UIntPtr;
  in_host: array[0..7] of Single;
  out_host: array[0..9] of Single; // 5 complex = 10 floats
  in_dev, out_dev: Pointer;
  in_ptrs, out_ptrs: array[0..0] of Pointer;
  st: rocfft_status;
  i: Integer;
  in_check: array[0..7] of Single;
begin
  lengths[0] := 8;
  // Input: impulse (1.0 at index 0, rest 0.0)
  for i := 0 to 7 do in_host[i] := 0.0;
  in_host[0] := 1.0;

  for i := 0 to 9 do out_host[i] := 0.0;

  st := hipMalloc(in_dev, 8 * SizeOf(Single)); AssertEquals('hipMalloc in', HIP_SUCCESS, st);
  st := hipMalloc(out_dev, 10 * SizeOf(Single)); AssertEquals('hipMalloc out', HIP_SUCCESS, st);
  st := hipMemcpy(in_dev, @in_host[0], 8 * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy in', HIP_SUCCESS, st);
  st := hipMemset(out_dev, 0, 10 * SizeOf(Single)); AssertEquals('hipMemset out', HIP_SUCCESS, st);

  st := rocfft_setup; AssertEquals('rocfft_setup', rocfft_status_success, st);
  st := rocfft_plan_create(plan, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_single, 1, @lengths[0], 1, nil); AssertEquals('plan_create', rocfft_status_success, st);
  st := rocfft_execution_info_create(info); AssertEquals('info_create', rocfft_status_success, st);

  in_ptrs[0] := in_dev;
  out_ptrs[0] := out_dev;
  st := rocfft_execute(plan, @in_ptrs, @out_ptrs, info); AssertEquals('execute', rocfft_status_success, st);
  st := hipMemcpy(@out_host[0], out_dev, 10 * SizeOf(Single), hipMemcpyDeviceToHost); AssertEquals('hipMemcpy out', HIP_SUCCESS, st);

  st := rocfft_plan_destroy(plan); AssertEquals('plan_destroy', rocfft_status_success, st);
  st := rocfft_execution_info_destroy(info); AssertEquals('info_destroy', rocfft_status_success, st);
  st := hipFree(in_dev); AssertEquals('hipFree in', HIP_SUCCESS, st);
  st := hipFree(out_dev); AssertEquals('hipFree out', HIP_SUCCESS, st);
  st := rocfft_cleanup; AssertEquals('rocfft_cleanup', rocfft_status_success, st);

  // FFT of an impulse is a constant 1.0 in the frequency domain (unscaled)
  // DC component (real)
  AssertEquals('R2C FFT impulse DC real', 1.0, out_host[0], 1e-5);
  // DC component (imag)
  AssertEquals('R2C FFT impulse DC imag', 0.0, out_host[1], 1e-5);

  // Other frequencies
  for i := 1 to 3 do begin
    AssertEquals('R2C FFT impulse real part', 1.0, out_host[2*i], 1e-5);
    AssertEquals('R2C FFT impulse imag part', 0.0, out_host[2*i+1], 1e-5);
  end;

  // Nyquist frequency (real)
  AssertEquals('R2C FFT impulse Nyquist real', 1.0, out_host[8], 1e-5);
  // Nyquist frequency (imag)
  AssertEquals('R2C FFT impulse Nyquist imag', 0.0, out_host[9], 1e-5);
end;

procedure TTest_rocFFT.TestFFT_DoublePrecision_RealForward;
var
  plan: rocfft_plan;
  info: rocfft_execution_info;
  lengths: array[0..0] of UIntPtr;
  in_host: array[0..7] of Double;
  out_host: array[0..9] of Double;
  in_dev, out_dev: Pointer;
  in_ptrs, out_ptrs: array[0..0] of Pointer;
  st: rocfft_status;
  i: Integer;
  in_check: array[0..7] of Double;
begin
  lengths[0] := 8;
  // Input: impulse (1.0 at index 0, rest 0.0)
  for i := 0 to 7 do in_host[i] := 0.0;
  in_host[0] := 1.0;

  for i := 0 to 9 do out_host[i] := 0.0;

  st := hipMalloc(in_dev, 8 * SizeOf(Double)); AssertEquals('hipMalloc in', HIP_SUCCESS, st);
  st := hipMalloc(out_dev, 10 * SizeOf(Double)); AssertEquals('hipMalloc out', HIP_SUCCESS, st);
  st := hipMemcpy(in_dev, @in_host[0], 8 * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy in', HIP_SUCCESS, st);
  st := hipMemset(out_dev, 0, 10 * SizeOf(Double)); AssertEquals('hipMemset out', HIP_SUCCESS, st);

  st := rocfft_setup; AssertEquals('rocfft_setup', rocfft_status_success, st);
  st := rocfft_plan_create(plan, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_double, 1, @lengths[0], 1, nil); AssertEquals('plan_create', rocfft_status_success, st);
  st := rocfft_execution_info_create(info); AssertEquals('info_create', rocfft_status_success, st);

  in_ptrs[0] := in_dev;
  out_ptrs[0] := out_dev;
  st := rocfft_execute(plan, @in_ptrs, @out_ptrs, info); AssertEquals('execute', rocfft_status_success, st);
  st := hipMemcpy(@out_host[0], out_dev, 10 * SizeOf(Double), hipMemcpyDeviceToHost); AssertEquals('hipMemcpy out', HIP_SUCCESS, st);

  st := rocfft_plan_destroy(plan); AssertEquals('plan_destroy', rocfft_status_success, st);
  st := rocfft_execution_info_destroy(info); AssertEquals('info_destroy', rocfft_status_success, st);
  st := hipFree(in_dev); AssertEquals('hipFree in', HIP_SUCCESS, st);
  st := hipFree(out_dev); AssertEquals('hipFree out', HIP_SUCCESS, st);
  st := rocfft_cleanup; AssertEquals('rocfft_cleanup', rocfft_status_success, st);

  // FFT of an impulse is a constant 1.0 in the frequency domain (unscaled)
  // DC component (real)
  AssertEquals('R2C FFT impulse DC real', 1.0, out_host[0], 1e-12);
  // DC component (imag)
  AssertEquals('R2C FFT impulse DC imag', 0.0, out_host[1], 1e-12);

  // Other frequencies
  for i := 1 to 3 do begin
    AssertEquals('R2C FFT impulse real part', 1.0, out_host[2*i], 1e-12);
    AssertEquals('R2C FFT impulse imag part', 0.0, out_host[2*i+1], 1e-12);
  end;

  // Nyquist frequency (real)
  AssertEquals('R2C FFT impulse Nyquist real', 1.0, out_host[8], 1e-12);
  // Nyquist frequency (imag)
  AssertEquals('R2C FFT impulse Nyquist imag', 0.0, out_host[9], 1e-12);
end;

procedure TTest_rocFFT.TestFFT_SinglePrecision_RealInverse;
var
  plan: rocfft_plan;
  info: rocfft_execution_info;
  lengths: array[0..0] of UIntPtr;
  in_host: array[0..9] of Single;
  out_host: array[0..7] of Single;
  in_dev, out_dev: Pointer;
  st: rocfft_status;
  i: Integer;
begin
  lengths[0] := 8;
  // Input: DC bin = 8, all others 0 (impulse in time domain)
  for i := 0 to 9 do in_host[i] := 0.0;
  in_host[0] := 8.0;
  for i := 0 to 7 do out_host[i] := 0.0;
  st := hipMalloc(in_dev, 10 * SizeOf(Single)); AssertEquals('hipMalloc in', HIP_SUCCESS, st);
  st := hipMalloc(out_dev, 8 * SizeOf(Single)); AssertEquals('hipMalloc out', HIP_SUCCESS, st);
  st := hipMemcpy(in_dev, @in_host[0], 10 * SizeOf(Single), hipMemcpyHostToDevice); AssertEquals('hipMemcpy in', HIP_SUCCESS, st);
  st := hipMemset(out_dev, 0, 8 * SizeOf(Single)); AssertEquals('hipMemset out', HIP_SUCCESS, st);
  st := rocfft_setup; AssertEquals('rocfft_setup', rocfft_status_success, st);
  st := rocfft_plan_create(plan, rocfft_placement_notinplace, rocfft_transform_type_real_inverse, rocfft_precision_single, 1, @lengths[0], 1, nil); AssertEquals('plan_create', rocfft_status_success, st);
  st := rocfft_execution_info_create(info); AssertEquals('info_create', rocfft_status_success, st);
  st := rocfft_execute(plan, @in_dev, @out_dev, info); AssertEquals('execute', rocfft_status_success, st);
  st := hipMemcpy(@out_host[0], out_dev, 8 * SizeOf(Single), hipMemcpyDeviceToHost); AssertEquals('hipMemcpy out', HIP_SUCCESS, st);
  st := rocfft_plan_destroy(plan); AssertEquals('plan_destroy', rocfft_status_success, st);
  st := rocfft_execution_info_destroy(info); AssertEquals('info_destroy', rocfft_status_success, st);
  st := hipFree(in_dev); AssertEquals('hipFree in', HIP_SUCCESS, st);
  st := hipFree(out_dev); AssertEquals('hipFree out', HIP_SUCCESS, st);
  st := rocfft_cleanup; AssertEquals('rocfft_cleanup', rocfft_status_success, st);
  // Output should be all 8s (un-normalized)
  for i := 0 to 7 do
    AssertEquals('C2R IFFT DC', 8.0, out_host[i], 1e-5);
end;

procedure TTest_rocFFT.TestFFT_DoublePrecision_RealInverse;
var
  plan: rocfft_plan;
  info: rocfft_execution_info;
  lengths: array[0..0] of UIntPtr;
  in_host: array[0..9] of Double;
  out_host: array[0..7] of Double;
  in_dev, out_dev: Pointer;
  st: rocfft_status;
  i: Integer;
begin
  lengths[0] := 8;
  for i := 0 to 9 do in_host[i] := 0.0;
  in_host[0] := 8.0;
  for i := 0 to 7 do out_host[i] := 0.0;
  st := hipMalloc(in_dev, 10 * SizeOf(Double)); AssertEquals('hipMalloc in', HIP_SUCCESS, st);
  st := hipMalloc(out_dev, 8 * SizeOf(Double)); AssertEquals('hipMalloc out', HIP_SUCCESS, st);
  st := hipMemcpy(in_dev, @in_host[0], 10 * SizeOf(Double), hipMemcpyHostToDevice); AssertEquals('hipMemcpy in', HIP_SUCCESS, st);
  st := hipMemset(out_dev, 0, 8 * SizeOf(Double)); AssertEquals('hipMemset out', HIP_SUCCESS, st);
  st := rocfft_setup; AssertEquals('rocfft_setup', rocfft_status_success, st);
  st := rocfft_plan_create(plan, rocfft_placement_notinplace, rocfft_transform_type_real_inverse, rocfft_precision_double, 1, @lengths[0], 1, nil); AssertEquals('plan_create', rocfft_status_success, st);
  st := rocfft_execution_info_create(info); AssertEquals('info_create', rocfft_status_success, st);
  st := rocfft_execute(plan, @in_dev, @out_dev, info); AssertEquals('execute', rocfft_status_success, st);
  st := hipMemcpy(@out_host[0], out_dev, 8 * SizeOf(Double), hipMemcpyDeviceToHost); AssertEquals('hipMemcpy out', HIP_SUCCESS, st);
  st := rocfft_plan_destroy(plan); AssertEquals('plan_destroy', rocfft_status_success, st);
  st := rocfft_execution_info_destroy(info); AssertEquals('info_destroy', rocfft_status_success, st);
  st := hipFree(in_dev); AssertEquals('hipFree in', HIP_SUCCESS, st);
  st := hipFree(out_dev); AssertEquals('hipFree out', HIP_SUCCESS, st);
  st := rocfft_cleanup; AssertEquals('rocfft_cleanup', rocfft_status_success, st);
  // Output should be all 8s (un-normalized)
  for i := 0 to 7 do
    AssertEquals('C2R IFFT DC', 8.0, out_host[i], 1e-12);
end;

initialization
  RegisterTest(TTest_rocFFT);

end. 
