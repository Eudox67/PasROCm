program test_radix_sort_keys_desc;

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

procedure CPUSortDesc(var a: array of cint);
var i, j, key: integer;
begin
  for i := 1 to High(a) do
  begin
    key := a[i];
    j := i - 1;
    while (j >= 0) and (a[j] < key) do
    begin
      a[j+1] := a[j];
      j := j - 1;
    end;
    a[j+1] := key;
  end;
end;

var
  h_input: array[0..7] of cint = (3, 1, 4, 1, 5, 9, 2, 6);
  h_output: array[0..7] of cint;
  cpu_output: array[0..7] of cint;

  d_input: Pointer = nil;
  d_output: Pointer = nil;
  d_temp_storage: Pointer = nil;

  size: cuint;
  temp_storage_size: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::radix_sort_keys_desc test...');

  size := Length(h_input);
  WriteLn(' - Input size: ', size);

  Write(' - Input: ');
  for i := 0 to size - 1 do Write(h_input[i], ' ');
  WriteLn;

  // CPU reference
  for i := 0 to size - 1 do cpu_output[i] := h_input[i];
  CPUSortDesc(cpu_output);

  Write(' - Expected output: ');
  for i := 0 to size - 1 do Write(cpu_output[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Create');

  CheckHIP(hipMalloc(@d_input, size * sizeof(cint)), 'Alloc d_input');
  CheckHIP(hipMalloc(@d_output, size * sizeof(cint)), 'Alloc d_output');

  CheckHIP(hipMemcpy(d_input, @h_input[0], size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD input');

  temp_storage_size := 0;
  CheckHIP(
    rocprim_radix_sort_keys_desc_int(nil, @temp_storage_size, d_input, d_output, size, 0, 8 * sizeof(cint), stream, True),
    'Get temp storage size'
  );
  WriteLn(' - Temp storage size: ', temp_storage_size, ' bytes');

  if temp_storage_size > 0 then
    CheckHIP(hipMalloc(@d_temp_storage, temp_storage_size), 'Alloc d_temp_storage');

  WriteLn(' - Executing on GPU...');
  CheckHIP(
    rocprim_radix_sort_keys_desc_int(d_temp_storage, @temp_storage_size, d_input, d_output, size, 0, 8 * sizeof(cint), stream, True),
    'GPU radix_sort_keys_desc'
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
  hipFree(d_output);
  hipStreamDestroy(stream);

  WriteLn('Test complete.');
end.
