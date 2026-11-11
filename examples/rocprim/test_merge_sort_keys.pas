program test_merge_sort_keys;

{$mode objfpc}{$H+}

uses
  ctypes,
  hip_bindings,
  rocprim_bindings,
  SysUtils,
  Math;

procedure CheckHIP(err: hipError_t; const msg: string);
begin
  if err <> hipSuccess then
  begin
    WriteLn('HIP Error: ', err, ' at ', msg);
    Halt(1);
  end;
end;

var
  h_keys_input: array[0..7] of cint = (6, 3, 5, 4, 1, 8, 2, 7);
  h_keys_output: array[0..7] of cint;
  cpu_keys_output: array[0..7] of cint;

  d_keys_input: Pointer = nil;
  d_keys_output: Pointer = nil;
  d_temp_storage: Pointer = nil;

  size, temp_storage_size: csize_t;
  i, j: cint;
  temp: cint;

  stream: hipStream_t;

  match: Boolean;

begin
  WriteLn('Starting rocprim::merge_sort (keys only) test...');

  size := Length(h_keys_input);

  WriteLn(' - Input size: ', size);
  
  Write(' - Input keys: ');
  for i := 0 to size - 1 do Write(h_keys_input[i], ' ');
  WriteLn;

  // CPU reference (simple sort)
  for i := 0 to size - 1 do cpu_keys_output[i] := h_keys_input[i];
  // Bubble sort for simplicity
  for i := 0 to size - 2 do
  begin
    for j := i + 1 to size - 1 do
    begin
      if cpu_keys_output[i] > cpu_keys_output[j] then
      begin
        temp := cpu_keys_output[i];
        cpu_keys_output[i] := cpu_keys_output[j];
        cpu_keys_output[j] := temp;
      end;
    end;
  end;

  Write(' - Expected (CPU) keys: ');
  for i := 0 to size - 1 do Write(cpu_keys_output[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate( @stream), 'Stream Create');

  CheckHIP(hipMalloc( @d_keys_input, size * sizeof(cint)), 'Alloc d_keys_input');
  CheckHIP(hipMalloc( @d_keys_output, size * sizeof(cint)), 'Alloc d_keys_output');

  CheckHIP(hipMemcpy(d_keys_input, @h_keys_input[0], size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD keys_input');

  temp_storage_size := 0;
  CheckHIP(
    rocprim_merge_sort_keys_int(nil, @temp_storage_size, Pointer(d_keys_input), Pointer(d_keys_output), size, stream, True),
    'Get temp storage size'
  );
  WriteLn(' - Temp storage size: ', temp_storage_size, ' bytes');

  if temp_storage_size > 0 then
    CheckHIP(hipMalloc( @d_temp_storage, temp_storage_size), 'Alloc d_temp_storage')
  else
    d_temp_storage := nil;

  WriteLn(' - Executing on GPU...');
  CheckHIP(
    rocprim_merge_sort_keys_int(Pointer(d_temp_storage), @temp_storage_size, Pointer(d_keys_input), Pointer(d_keys_output), size, stream, True),
    'GPU merge_sort_keys'
  );

  CheckHIP(hipMemcpy( @h_keys_output[0], d_keys_output, size * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH keys_output');

  Write(' - GPU output keys: ');
  for i := 0 to size - 1 do Write(h_keys_output[i], ' ');
  WriteLn;

  match := True;
  for i := 0 to size - 1 do
  begin
    if h_keys_output[i] <> cpu_keys_output[i] then
    begin
      WriteLn('  Mismatch at index ', i, ': GPU=', h_keys_output[i], ', Expected=', cpu_keys_output[i]);
      match := False;
    end;
  end;

  if match then
    WriteLn('SUCCESS: GPU result matches CPU result!')
  else
    WriteLn('FAILURE: GPU result does not match CPU result.');

  if assigned(d_temp_storage) then hipFree(d_temp_storage);
  hipFree(d_keys_input);
  hipFree(d_keys_output);
  hipStreamDestroy(stream);

  WriteLn('Test complete.');
end.
