program test_segmented_radix_sort_pairs;

{$mode objfpc}{$H+}

uses
  ctypes,
  hip_bindings,
  rocprim_bindings,
  SysUtils;

procedure CheckHIP(err: hipError_t; const msg: string);
begin
  if err <> hipSuccess then
  begin
    WriteLn('HIP Error: ', err, ' at ', msg);
    Halt(1);
  end;
end;

procedure CPUSortPairs(var keys, values: array of cint; start, count: integer);
var i, j, temp_key, temp_val: integer;
begin
  for i := start to start + count - 2 do
  begin
    for j := i + 1 to start + count - 1 do
    begin
        if keys[i] > keys[j] then
        begin
            temp_key := keys[i];
            keys[i] := keys[j];
            keys[j] := temp_key;

            temp_val := values[i];
            values[i] := values[j];
            values[j] := temp_val;
        end;
    end;
  end;
end;

var
  h_keys_input: array[0..7] of cint = (3, 1, 4, 1, 5, 9, 2, 6);
  h_values_input: array[0..7] of cint = (10, 20, 30, 40, 50, 60, 70, 80);
  h_offsets: array[0..3] of cint = (0, 2, 5, 8);
  h_keys_output: array[0..7] of cint;
  h_values_output: array[0..7] of cint;
  cpu_keys_output: array[0..7] of cint;
  cpu_values_output: array[0..7] of cint;

  d_keys_input: Pointer = nil;
  d_values_input: Pointer = nil;
  d_offsets: Pointer = nil;
  d_keys_output: Pointer = nil;
  d_values_output: Pointer = nil;
  d_temp_storage: Pointer = nil;

  size, segments: cuint;
  temp_storage_size: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::segmented_radix_sort_pairs test...');

  size := Length(h_keys_input);
  segments := Length(h_offsets) - 1;
  WriteLn(' - Input size: ', size);
  WriteLn(' - Number of segments: ', segments);

  Write(' - Input keys: ');
  for i := 0 to size - 1 do Write(h_keys_input[i], ' ');
  WriteLn;
  Write(' - Input values: ');
  for i := 0 to size - 1 do Write(h_values_input[i], ' ');
  WriteLn;

  // CPU reference
  for i := 0 to size - 1 do
  begin
    cpu_keys_output[i] := h_keys_input[i];
    cpu_values_output[i] := h_values_input[i];
  end;
  for i := 0 to segments - 1 do
  begin
    CPUSortPairs(cpu_keys_output, cpu_values_output, h_offsets[i], h_offsets[i+1] - h_offsets[i]);
  end;

  Write(' - Expected keys output: ');
  for i := 0 to size - 1 do Write(cpu_keys_output[i], ' ');
  WriteLn;
  Write(' - Expected values output: ');
  for i := 0 to size - 1 do Write(cpu_values_output[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Create');

  CheckHIP(hipMalloc(@d_keys_input, size * sizeof(cint)), 'Alloc d_keys_input');
  CheckHIP(hipMalloc(@d_values_input, size * sizeof(cint)), 'Alloc d_values_input');
  CheckHIP(hipMalloc(@d_offsets, Length(h_offsets) * sizeof(cint)), 'Alloc d_offsets');
  CheckHIP(hipMalloc(@d_keys_output, size * sizeof(cint)), 'Alloc d_keys_output');
  CheckHIP(hipMalloc(@d_values_output, size * sizeof(cint)), 'Alloc d_values_output');

  CheckHIP(hipMemcpy(d_keys_input, @h_keys_input[0], size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD keys_input');
  CheckHIP(hipMemcpy(d_values_input, @h_values_input[0], size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD values_input');
  CheckHIP(hipMemcpy(d_offsets, @h_offsets[0], Length(h_offsets) * sizeof(cint), hipMemcpyHostToDevice), 'HtoD offsets');

  temp_storage_size := 0;
  CheckHIP(
    rocprim_segmented_radix_sort_pairs_int(nil, @temp_storage_size, d_keys_input, d_keys_output, d_values_input, d_values_output, size, segments, d_offsets, Pointer(NativeUInt(d_offsets) + sizeof(cint)), 0, 8 * sizeof(cint), stream, True),
    'Get temp storage size'
  );
  WriteLn(' - Temp storage size: ', temp_storage_size, ' bytes');

  if temp_storage_size > 0 then
    CheckHIP(hipMalloc(@d_temp_storage, temp_storage_size), 'Alloc d_temp_storage');

  WriteLn(' - Executing on GPU...');
  CheckHIP(
    rocprim_segmented_radix_sort_pairs_int(d_temp_storage, @temp_storage_size, d_keys_input, d_keys_output, d_values_input, d_values_output, size, segments, d_offsets, Pointer(NativeUInt(d_offsets) + sizeof(cint)), 0, 8 * sizeof(cint), stream, True),
    'GPU segmented_radix_sort_pairs'
  );

  CheckHIP(hipMemcpy(@h_keys_output[0], d_keys_output, size * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH keys_output');
  CheckHIP(hipMemcpy(@h_values_output[0], d_values_output, size * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH values_output');

  Write(' - GPU keys output: ');
  for i := 0 to size - 1 do Write(h_keys_output[i], ' ');
  WriteLn;
  Write(' - GPU values output: ');
  for i := 0 to size - 1 do Write(h_values_output[i], ' ');
  WriteLn;

  match := True;
  for i := 0 to size - 1 do
  begin
    if (h_keys_output[i] <> cpu_keys_output[i]) or (h_values_output[i] <> cpu_values_output[i]) then
    begin
      match := False;
      break;
    end;
  end;

  if match then
    WriteLn('SUCCESS: GPU result matches CPU result!')
  else
    WriteLn('FAILURE: GPU result does not match CPU result.');

  if assigned(d_temp_storage) then hipFree(d_temp_storage);
  hipFree(d_keys_input);
  hipFree(d_values_input);
  hipFree(d_offsets);
  hipFree(d_keys_output);
  hipFree(d_values_output);
  hipStreamDestroy(stream);

  WriteLn('Test complete.');
end.
