program test_warp_scan;

{$mode objfpc}{$H+}

uses
  ctypes, hip_runtime, rocprim_bindings;

const
  NUM_ITEMS = 1024;
  BLOCK_SIZE = 256;
  WARP_SIZE = 32;
var
  h_in: array[0..NUM_ITEMS-1] of cint;
  h_out, h_cpu_out: array[0..NUM_ITEMS-1] of cint;
  d_in, d_out: Pointer;
  i: integer;
  stream: hipStream_t;
  status: hipError_t;
  test_passed: boolean;

begin
  // Initialize host data
  writeln('Initializing host data...');
  for i := 0 to NUM_ITEMS - 1 do
  begin
    h_in[i] := 1;
  end;

  // Calculate CPU scan
  for i := 0 to NUM_ITEMS - 1 do
  begin
    if (i mod WARP_SIZE) = 0 then
      h_cpu_out[i] := h_in[i]
    else
      h_cpu_out[i] := h_cpu_out[i-1] + h_in[i];
  end;

  // Allocate device memory
  writeln('Allocating device memory...');
  status := hipMalloc(d_in, NUM_ITEMS * sizeof(cint));
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMalloc failed: ', status);
    halt(1);
  end;
  status := hipMalloc(d_out, NUM_ITEMS * sizeof(cint));
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMalloc failed: ', status);
    hipFree(d_in);
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
    halt(1);
  end;

  // Create stream
  status := hipStreamCreate(stream);
  if status <> HIP_SUCCESS then
  begin
    writeln('hipStreamCreate failed: ', status);
    hipFree(d_in);
    hipFree(d_out);
    halt(1);
  end;

  // Execute warp scan
  writeln('Executing warp scan...');
  status := rocprim_warp_scan_int32(d_out, d_in, NUM_ITEMS, stream);
  if status <> HIP_SUCCESS then
  begin
    writeln('rocprim_warp_scan_int failed: ', status);
    hipStreamDestroy(stream);
    hipFree(d_in);
    hipFree(d_out);
    halt(1);
  end;

  status := hipStreamSynchronize(stream);
  if status <> HIP_SUCCESS then
  begin
    writeln('hipStreamSynchronize failed: ', status);
    hipStreamDestroy(stream);
    hipFree(d_in);
    hipFree(d_out);
    halt(1);
  end;

  // Copy results from device to host
  writeln('Copying results from device to host...');
  status := hipMemcpy(@h_out[0], d_out, NUM_ITEMS * sizeof(cint), hipMemcpyDeviceToHost);
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMemcpy failed: ', status);
    hipStreamDestroy(stream);
    hipFree(d_in);
    hipFree(d_out);
    halt(1);
  end;

  // Verify results
  writeln('Verifying results...');
  test_passed := True;
  for i := 0 to NUM_ITEMS - 1 do
  begin
    if h_out[i] <> h_cpu_out[i] then
    begin
      writeln('Mismatch at index ', i, ': expected ', h_cpu_out[i], ', got ', h_out[i]);
      test_passed := False;
      break;
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
end.
