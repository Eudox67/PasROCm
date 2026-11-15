{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_histogram_range_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_histogram_range.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_histogram_range
------------------------------------------------------------------------------}

program test_histogram_range;

{$mode objfpc}{$H+}

uses
  ctypes, hip_error, hip_types, hip_runtime, rocprim_bindings, sysutils;

const
  NUM_ITEMS = 1024;
  NUM_BINS = 5;
  NUM_LEVELS = NUM_BINS + 1;
var
  h_in: array[0..NUM_ITEMS-1] of cint;
  h_out, h_cpu_out: array[0..NUM_BINS-1] of cint;
  h_levels: array[0..NUM_LEVELS-1] of cint = (0, 10, 20, 30, 40, 50);
  d_in, d_out, d_levels, d_temp: Pointer;
  i: integer;
  stream: hipStream_t;
  status: hipError_t;
  test_passed: boolean;
  temp_size: size_t;

begin
  // Initialize host data
  writeln('Initializing host data...');
  for i := 0 to NUM_ITEMS - 1 do
  begin
    h_in[i] := i mod 50;
  end;

  // Calculate CPU histogram
  for i := 0 to NUM_BINS - 1 do
  begin
    h_cpu_out[i] := 0;
  end;

  for i := 0 to NUM_ITEMS - 1 do
  begin
    if (h_in[i] >= 0) and (h_in[i] < 10) then
      h_cpu_out[0] := h_cpu_out[0] + 1
    else if (h_in[i] >= 10) and (h_in[i] < 20) then
      h_cpu_out[1] := h_cpu_out[1] + 1
    else if (h_in[i] >= 20) and (h_in[i] < 30) then
      h_cpu_out[2] := h_cpu_out[2] + 1
    else if (h_in[i] >= 30) and (h_in[i] < 40) then
      h_cpu_out[3] := h_cpu_out[3] + 1
    else if (h_in[i] >= 40) and (h_in[i] < 50) then
      h_cpu_out[4] := h_cpu_out[4] + 1;
  end;

  // Allocate device memory
  writeln('Allocating device memory...');
  status := hipMalloc(d_in, NUM_ITEMS * sizeof(cint));
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMalloc failed: ', status);
    halt(1);
  end;
  status := hipMalloc(d_out, NUM_BINS * sizeof(cint));
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMalloc failed: ', status);
    hipFree(d_in);
    halt(1);
  end;
  status := hipMalloc(d_levels, NUM_LEVELS * sizeof(cint));
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMalloc failed: ', status);
    hipFree(d_in);
    hipFree(d_out);
    halt(1);
  end;

  // Copy data from host to device
  writeln('Copying data from host to device...');
  status := hipMemcpy(d_in, @h_in[0], NUM_ITEMS * sizeof(cint), hipMemcpyHostToDevice);
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMemcpy failed: ', status);
    hipFree(d_in);
    hipFree(d_out);
    hipFree(d_levels);
    halt(1);
  end;
  status := hipMemcpy(d_levels, @h_levels[0], NUM_LEVELS * sizeof(cint), hipMemcpyHostToDevice);
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMemcpy failed: ', status);
    hipFree(d_in);
    hipFree(d_out);
    hipFree(d_levels);
    halt(1);
  end;

  // Get temporary storage size
  status := rocprim_histogram_range_int(nil, @temp_size, d_in, NUM_ITEMS, d_out, NUM_LEVELS, d_levels, nil, false);
  if status <> HIP_SUCCESS then
  begin
    writeln('rocprim_histogram_range_int (size) failed: ', status);
    halt(1);
  end;

  status := hipMalloc(d_temp, temp_size);
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMalloc for temp storage failed: ', status);
    halt(1);
  end;


  // Create stream
  status := hipStreamCreate(stream);
  if status <> HIP_SUCCESS then
  begin
    writeln('hipStreamCreate failed: ', status);
    hipFree(d_in);
    hipFree(d_out);
    hipFree(d_levels);
    hipFree(d_temp);
    halt(1);
  end;

  // Execute histogram
  writeln('Executing histogram...');
  status := rocprim_histogram_range_int(d_temp, @temp_size, d_in, NUM_ITEMS, d_out, NUM_LEVELS, d_levels, stream, false);
  if status <> HIP_SUCCESS then
  begin
    writeln('rocprim_histogram_range_int failed: ', status);
    hipStreamDestroy(stream);
    hipFree(d_in);
    hipFree(d_out);
    hipFree(d_levels);
    hipFree(d_temp);
    halt(1);
  end;

  status := hipStreamSynchronize(stream);
  if status <> HIP_SUCCESS then
  begin
    writeln('hipStreamSynchronize failed: ', status);
    hipStreamDestroy(stream);
    hipFree(d_in);
    hipFree(d_out);
    hipFree(d_levels);
    hipFree(d_temp);
    halt(1);
  end;

  // Copy results from device to host
  writeln('Copying results from device to host...');
  status := hipMemcpy(@h_out[0], d_out, NUM_BINS * sizeof(cint), hipMemcpyDeviceToHost);
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMemcpy failed: ', status);
    hipStreamDestroy(stream);
    hipFree(d_in);
    hipFree(d_out);
    hipFree(d_levels);
    hipFree(d_temp);
    halt(1);
  end;

  // Verify results
  writeln('Verifying results...');
  test_passed := True;
  for i := 0 to NUM_BINS - 1 do
  begin
    if h_out[i] <> h_cpu_out[i] then
    begin
      writeln('Mismatch at index ', i, ': expected ', h_cpu_out[i], ', got ', h_out[i]);
      test_passed := False;
    end;
  end;

  if test_passed then
    writeln('Test PASSED')
  else
    writeln('Test FAILED');

  // Clean up
  writeln('Cleaning up...');
  hipStreamDestroy(stream);
  hipFree(d_in);
  hipFree(d_out);
  hipFree(d_levels);
  hipFree(d_temp);
end.
