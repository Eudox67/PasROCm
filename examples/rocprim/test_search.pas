program test_search;

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
  h_haystack: array[0..9] of cint = (0, 1, 2, 3, 4, 5, 4, 3, 2, 1);
  h_needle: array[0..2] of cint = (4, 3, 2);
  h_output: cint;
  cpu_output: cint;

  d_haystack: Pointer = nil;
  d_needle: Pointer = nil;
  d_output: Pointer = nil;
  d_temp_storage: Pointer = nil;

  haystack_size, needle_size, temp_storage_size: csize_t;
  i, j: Integer;
  stream: hipStream_t;
  found: Boolean;

begin
  WriteLn('Starting rocprim::search test...');

  haystack_size := Length(h_haystack);
  needle_size := Length(h_needle);

  WriteLn(' - Haystack size: ', haystack_size);
  WriteLn(' - Needle size: ', needle_size);

  Write(' - Haystack: ');
  for i := 0 to haystack_size - 1 do Write(h_haystack[i], ' ');
  WriteLn;

  Write(' - Needle: ');
  for i := 0 to needle_size - 1 do Write(h_needle[i], ' ');
  WriteLn;

  // CPU reference
  cpu_output := haystack_size; // Default to not found
  if needle_size <= haystack_size then
  begin
    for i := 0 to haystack_size - needle_size do
    begin
      found := True;
      for j := 0 to needle_size - 1 do
      begin
        if h_haystack[i + j] <> h_needle[j] then
        begin
          found := False;
          break;
        end;
      end;
      if found then
      begin
        cpu_output := i;
        break;
      end;
    end;
  end;

  WriteLn(' - Expected output (index): ', cpu_output);

  CheckHIP(hipStreamCreate(@stream), 'Stream Create');

  CheckHIP(hipMalloc(@d_haystack, haystack_size * sizeof(cint)), 'Alloc d_haystack');
  CheckHIP(hipMalloc(@d_needle, needle_size * sizeof(cint)), 'Alloc d_needle');
  CheckHIP(hipMalloc(@d_output, sizeof(cint)), 'Alloc d_output');

  CheckHIP(hipMemcpy(d_haystack, @h_haystack[0], haystack_size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD haystack');
  CheckHIP(hipMemcpy(d_needle, @h_needle[0], needle_size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD needle');

  temp_storage_size := 0;
  CheckHIP(
    rocprim_search_int(nil, @temp_storage_size, d_haystack, d_needle, d_output, haystack_size, needle_size, stream, True),
    'Get temp storage size'
  );
  WriteLn(' - Temp storage size: ', temp_storage_size, ' bytes');

  if temp_storage_size > 0 then
    CheckHIP(hipMalloc(@d_temp_storage, temp_storage_size), 'Alloc d_temp_storage')
  else
    d_temp_storage := nil;

  WriteLn(' - Executing on GPU...');
  CheckHIP(
    rocprim_search_int(d_temp_storage, @temp_storage_size, d_haystack, d_needle, d_output, haystack_size, needle_size, stream, True),
    'GPU search'
  );

  CheckHIP(hipMemcpy(@h_output, d_output, sizeof(cint), hipMemcpyDeviceToHost), 'DtoH output');

  WriteLn(' - GPU output (index): ', h_output);

  if h_output = cpu_output then
    WriteLn('SUCCESS: GPU result matches CPU result!')
  else
    WriteLn('FAILURE: GPU result does not match CPU result.');

  if assigned(d_temp_storage) then hipFree(d_temp_storage);
  hipFree(d_haystack);
  hipFree(d_needle);
  hipFree(d_output);
  hipStreamDestroy(stream);

  WriteLn('Test complete.');
end.
