program test_segmented_radix_sort;

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

procedure CPUSort(var a: array of cint; start, count: integer);
var i, j, key: integer;
begin
  for i := start + 1 to start + count - 1 do
  begin
    key := a[i];
    j := i - 1;
    while (j >= start) and (a[j] > key) do
    begin
      a[j+1] := a[j];
      j := j - 1;
    end;
    a[j+1] := key;
  end;
end;

var
  h_input: array[0..7] of cint = (3, 1, 4, 1, 5, 9, 2, 6);
  h_offsets: array[0..3] of cint = (0, 2, 5, 8);
  h_output: array[0..7] of cint;
  cpu_output: array[0..7] of cint;

  d_input: Pointer = nil;
  d_offsets: Pointer = nil;
  d_output: Pointer = nil;
  d_temp_storage: Pointer = nil;

  size, segments: cuint;
  temp_storage_size: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::segmented_radix_sort_keys test...');

  size := Length(h_input);
  segments := Length(h_offsets) - 1;
  WriteLn(' - Input size: ', size);
  WriteLn(' - Number of segments: ', segments);

  Write(' - Input: ');
  for i := 0 to size - 1 do Write(h_input[i], ' ');
  WriteLn;

  // CPU reference
  for i := 0 to size - 1 do cpu_output[i] := h_input[i];
  for i := 0 to segments - 1 do
  begin
    CPUSort(cpu_output, h_offsets[i], h_offsets[i+1] - h_offsets[i]);
  end;

  Write(' - Expected output: ');
  for i := 0 to size - 1 do Write(cpu_output[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Create');

  CheckHIP(hipMalloc(@d_input, size * sizeof(cint)), 'Alloc d_input');
  CheckHIP(hipMalloc(@d_offsets, Length(h_offsets) * sizeof(cint)), 'Alloc d_offsets');
  CheckHIP(hipMalloc(@d_output, size * sizeof(cint)), 'Alloc d_output');

  CheckHIP(hipMemcpy(d_input, @h_input[0], size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD input');
  CheckHIP(hipMemcpy(d_offsets, @h_offsets[0], Length(h_offsets) * sizeof(cint), hipMemcpyHostToDevice), 'HtoD offsets');

  temp_storage_size := 0;
  CheckHIP(
    rocprim_segmented_radix_sort_keys_int(nil, @temp_storage_size, d_input, d_output, size, segments, d_offsets, Pointer(NativeUInt(d_offsets) + sizeof(cint)), 0, 8 * sizeof(cint), stream, True),
    'Get temp storage size'
  );
  WriteLn(' - Temp storage size: ', temp_storage_size, ' bytes');

  if temp_storage_size > 0 then
    CheckHIP(hipMalloc(@d_temp_storage, temp_storage_size), 'Alloc d_temp_storage');

  WriteLn(' - Executing on GPU...');
  CheckHIP(
    rocprim_segmented_radix_sort_keys_int(d_temp_storage, @temp_storage_size, d_input, d_output, size, segments, d_offsets, Pointer(NativeUInt(d_offsets) + sizeof(cint)), 0, 8 * sizeof(cint), stream, True),
    'GPU segmented_radix_sort_keys'
  );

  CheckHIP(hipMemcpy(@h_output[0], d_output, size * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH output');

  Write(' - GPU output: ');
  for i := 0 to size - 1 do Write(h_output[i], ' ');
  WriteLn;

  match := True;
  for i := 0 to size - 1 do
  begin
    if h_output[i] <> cpu_output[i] then
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
  hipFree(d_input);
  hipFree(d_offsets);
  hipFree(d_output);
  hipStreamDestroy(stream);

  WriteLn('Test complete.');
end.
