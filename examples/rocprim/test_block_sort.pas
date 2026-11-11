program test_block_sort;

{$mode objfpc}{$H+}

uses
  ctypes, hip_runtime, rocprim_bindings, sysutils;

const
  NUM_ITEMS = 1024;
  BLOCK_SIZE = 256;
var
  h_in: array[0..NUM_ITEMS-1] of cint;
  h_out, h_cpu_out: array[0..NUM_ITEMS-1] of cint;
  d_in, d_out: Pointer;
  i, j, start_index: integer;
  stream: hipStream_t;
  status: hipError_t;
  test_passed: boolean;

procedure SortBlock(var a: array of cint; from, to_ : integer);
var
  i, j, temp: cint;
begin
  for i := from to to_ - 1 do
  begin
    for j := i + 1 to to_ do
    begin
      if a[i] > a[j] then
      begin
        temp := a[i];
        a[i] := a[j];
        a[j] := temp;
      end;
    end;
  end;
end;

begin
  // Initialize host data
  writeln('Initializing host data...');
  for i := 0 to NUM_ITEMS - 1 do
  begin
    h_in[i] := NUM_ITEMS - i;
  end;
  h_cpu_out := h_in;

  // Calculate CPU sort
  for i := 0 to (NUM_ITEMS div BLOCK_SIZE) - 1 do
  begin
    start_index := i * BLOCK_SIZE;
    SortBlock(h_cpu_out, start_index, start_index + BLOCK_SIZE - 1);
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

  // Execute block sort
  writeln('Executing block sort...');
  status := rocprim_block_sort_int(d_out, d_in, NUM_ITEMS, BLOCK_SIZE, stream);
  if status <> HIP_SUCCESS then
  begin
    writeln('rocprim_block_sort_int failed: ', status);
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
