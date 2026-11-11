program test_merge_sort_pairs;

{$mode objfpc}{$H+}

uses
  ctypes, hip_runtime, rocprim_bindings, sysutils;

const
  NUM_ITEMS = 1024;
type
  TKeyPair = record
    key: cint;
    value: cint;
  end;
var
  h_keys_in, h_keys_out, h_cpu_keys_out: array[0..NUM_ITEMS-1] of cint;
  h_values_in, h_values_out, h_cpu_values_out: array[0..NUM_ITEMS-1] of cint;
  d_keys_in, d_keys_out, d_values_in, d_values_out, d_temp: Pointer;
  i: integer;
  stream: hipStream_t;
  status: hipError_t;
  test_passed: boolean;
  temp_size: size_t;
  pairs: array of TKeyPair;

procedure SortPairs(var a: array of TKeyPair);
var
  i, j: integer;
  temp: TKeyPair;
begin
  for i := Low(a) to High(a) - 1 do
  begin
    for j := i + 1 to High(a) do
    begin
      if a[i].key > a[j].key then
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
  SetLength(pairs, NUM_ITEMS);
  for i := 0 to NUM_ITEMS - 1 do
  begin
    h_keys_in[i] := NUM_ITEMS - i;
    h_values_in[i] := i;
    pairs[i].key := h_keys_in[i];
    pairs[i].value := h_values_in[i];
  end;

  // Calculate CPU sort
  SortPairs(pairs);
  for i := 0 to NUM_ITEMS - 1 do
  begin
    h_cpu_keys_out[i] := pairs[i].key;
    h_cpu_values_out[i] := pairs[i].value;
  end;

  // Allocate device memory
  writeln('Allocating device memory...');
  status := hipMalloc(d_keys_in, NUM_ITEMS * sizeof(cint));
  if status <> HIP_SUCCESS then halt(1);
  status := hipMalloc(d_keys_out, NUM_ITEMS * sizeof(cint));
  if status <> HIP_SUCCESS then halt(1);
  status := hipMalloc(d_values_in, NUM_ITEMS * sizeof(cint));
  if status <> HIP_SUCCESS then halt(1);
  status := hipMalloc(d_values_out, NUM_ITEMS * sizeof(cint));
  if status <> HIP_SUCCESS then halt(1);

  // Copy data from host to device
  writeln('Copying data from host to device...');
  status := hipMemcpy(d_keys_in, @h_keys_in[0], NUM_ITEMS * sizeof(cint), hipMemcpyHostToDevice);
  if status <> HIP_SUCCESS then halt(1);
  status := hipMemcpy(d_values_in, @h_values_in[0], NUM_ITEMS * sizeof(cint), hipMemcpyHostToDevice);
  if status <> HIP_SUCCESS then halt(1);

  // Get temporary storage size
  status := rocprim_merge_sort_pairs_int(nil, @temp_size, d_keys_in, d_keys_out, d_values_in, d_values_out, NUM_ITEMS, nil, false);
  if status <> HIP_SUCCESS then
  begin
    writeln('rocprim_merge_sort_pairs_int (size) failed: ', status);
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
  if status <> HIP_SUCCESS then halt(1);

  // Execute sort
  writeln('Executing sort...');
  status := rocprim_merge_sort_pairs_int(d_temp, @temp_size, d_keys_in, d_keys_out, d_values_in, d_values_out, NUM_ITEMS, stream, false);
  if status <> HIP_SUCCESS then
  begin
    writeln('rocprim_merge_sort_pairs_int failed: ', status);
    halt(1);
  end;

  status := hipStreamSynchronize(stream);
  if status <> HIP_SUCCESS then halt(1);

  // Copy results from device to host
  writeln('Copying results from device to host...');
  status := hipMemcpy(@h_keys_out[0], d_keys_out, NUM_ITEMS * sizeof(cint), hipMemcpyDeviceToHost);
  if status <> HIP_SUCCESS then halt(1);
  status := hipMemcpy(@h_values_out[0], d_values_out, NUM_ITEMS * sizeof(cint), hipMemcpyDeviceToHost);
  if status <> HIP_SUCCESS then halt(1);

  // Verify results
  writeln('Verifying results...');
  test_passed := True;
  for i := 0 to NUM_ITEMS - 1 do
  begin
    if (h_keys_out[i] <> h_cpu_keys_out[i]) or (h_values_out[i] <> h_cpu_values_out[i]) then
    begin
      writeln('Mismatch at index ', i);
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
  hipFree(d_keys_in);
  hipFree(d_keys_out);
  hipFree(d_values_in);
  hipFree(d_values_out);
  hipFree(d_temp);
end.
