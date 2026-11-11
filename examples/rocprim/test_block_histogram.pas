program test_block_histogram;

{$mode objfpc}{$H+}

uses
  ctypes, hip_runtime, rocprim_bindings, sysutils;

const
  NUM_ITEMS = 1024;
  BLOCK_SIZE = 256;
  BINS = 256;
var
  h_in: array[0..NUM_ITEMS-1] of cint;
  h_out, h_cpu_out: array[0..((NUM_ITEMS + BLOCK_SIZE - 1) div BLOCK_SIZE) * BINS - 1] of cint;
  d_in, d_out: Pointer;
  i, j, block_index: integer;
  stream: hipStream_t;
  status: hipError_t;
  test_passed: boolean;
  num_blocks: integer;

begin
  num_blocks := (NUM_ITEMS + BLOCK_SIZE - 1) div BLOCK_SIZE;

  // Initialize host data
  writeln('Initializing host data...');
  for i := 0 to NUM_ITEMS - 1 do
  begin
    h_in[i] := i mod BINS;
  end;

  // Calculate CPU histogram
  for i := 0 to num_blocks * BINS - 1 do
  begin
    h_cpu_out[i] := 0;
  end;

  for block_index := 0 to num_blocks - 1 do
  begin
    for i := 0 to BLOCK_SIZE - 1 do
    begin
      j := block_index * BLOCK_SIZE + i;
      if j < NUM_ITEMS then
      begin
        h_cpu_out[block_index * BINS + h_in[j]] := h_cpu_out[block_index * BINS + h_in[j]] + 1;
      end;
    end;
  end;

  // Allocate device memory
  writeln('Allocating device memory...');
  status := hipMalloc(d_in, NUM_ITEMS * sizeof(cint));
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMalloc failed: ', status);
    halt(1);
  end;
  status := hipMalloc(d_out, num_blocks * BINS * sizeof(cint));
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

  // Execute block histogram
  writeln('Executing block histogram...');
  status := rocprim_block_histogram_int(d_out, d_in, NUM_ITEMS, BLOCK_SIZE, BINS, stream);
  if status <> HIP_SUCCESS then
  begin
    writeln('rocprim_block_histogram_int failed: ', status);
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
  status := hipMemcpy(@h_out[0], d_out, num_blocks * BINS * sizeof(cint), hipMemcpyDeviceToHost);
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
  for i := 0 to num_blocks * BINS - 1 do
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
