program test_segmented_reduce;

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

var
  h_input: array[0..7] of cint = (1, 2, 3, 4, 5, 6, 7, 8);
  h_offsets: array[0..3] of cint = (0, 2, 5, 8);
  h_output: array[0..2] of cint;
  cpu_output: array[0..2] of cint;

  d_input: Pointer = nil;
  d_offsets: Pointer = nil;
  d_output: Pointer = nil;
  d_temp_storage: Pointer = nil;

  segments: cuint;
  temp_storage_size: csize_t;
  i, j, sum: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::segmented_reduce test...');

  segments := Length(h_offsets) - 1;
  WriteLn(' - Number of segments: ', segments);

  Write(' - Input: ');
  for i := 0 to High(h_input) do Write(h_input[i], ' ');
  WriteLn;

  Write(' - Offsets: ');
  for i := 0 to High(h_offsets) do Write(h_offsets[i], ' ');
  WriteLn;

  // CPU reference
  for i := 0 to segments - 1 do
  begin
    sum := 0;
    for j := h_offsets[i] to h_offsets[i+1] - 1 do
    begin
      sum := sum + h_input[j];
    end;
    cpu_output[i] := sum;
  end;

  Write(' - Expected output: ');
  for i := 0 to segments - 1 do Write(cpu_output[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Create');

  CheckHIP(hipMalloc(@d_input, Length(h_input) * sizeof(cint)), 'Alloc d_input');
  CheckHIP(hipMalloc(@d_offsets, Length(h_offsets) * sizeof(cint)), 'Alloc d_offsets');
  CheckHIP(hipMalloc(@d_output, segments * sizeof(cint)), 'Alloc d_output');

  CheckHIP(hipMemcpy(d_input, @h_input[0], Length(h_input) * sizeof(cint), hipMemcpyHostToDevice), 'HtoD input');
  CheckHIP(hipMemcpy(d_offsets, @h_offsets[0], Length(h_offsets) * sizeof(cint), hipMemcpyHostToDevice), 'HtoD offsets');

  temp_storage_size := 0;
  CheckHIP(
    rocprim_segmented_reduce_int(nil, @temp_storage_size, d_input, d_output, segments, d_offsets, Pointer(NativeUInt(d_offsets) + sizeof(cint)), 0, stream, True),
    'Get temp storage size'
  );
  WriteLn(' - Temp storage size: ', temp_storage_size, ' bytes');

  if temp_storage_size > 0 then
    CheckHIP(hipMalloc(@d_temp_storage, temp_storage_size), 'Alloc d_temp_storage')
  else
    d_temp_storage := nil;

  WriteLn(' - Executing on GPU...');
  CheckHIP(
    rocprim_segmented_reduce_int(d_temp_storage, @temp_storage_size, d_input, d_output, segments, d_offsets, Pointer(NativeUInt(d_offsets) + sizeof(cint)), 0, stream, True),
    'GPU segmented_reduce'
  );

  CheckHIP(hipMemcpy(@h_output[0], d_output, segments * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH output');

  Write(' - GPU output: ');
  for i := 0 to segments - 1 do Write(h_output[i], ' ');
  WriteLn;

  match := True;
  for i := 0 to segments - 1 do
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
