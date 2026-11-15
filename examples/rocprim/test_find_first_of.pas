{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_find_first_of_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_find_first_of.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_find_first_of
------------------------------------------------------------------------------}

program test_find_first_of;

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
  h_input: array[0..7] of cint = (6, 3, 5, 4, 1, 8, 2, 7);
  h_keys: array[0..1] of cint = (10, 5);
  h_output: array[0..0] of csize_t;
  cpu_output: array[0..0] of csize_t;

  d_input: Pointer = nil;
  d_keys: Pointer = nil;
  d_output: Pointer = nil;
  d_temp_storage: Pointer = nil;

  input_size, keys_size, temp_storage_size: csize_t;
  i, j: cint;

  stream: hipStream_t;

  match: Boolean;

begin
  WriteLn('Starting rocprim::find_first_of test...');

  input_size := Length(h_input);
  keys_size := Length(h_keys);

  WriteLn(' - Input size: ', input_size);
  WriteLn(' - Keys size: ', keys_size);
  
  Write(' - Input: ');
  for i := 0 to input_size - 1 do Write(h_input[i], ' ');
  WriteLn;
  Write(' - Keys: ');
  for i := 0 to keys_size - 1 do Write(h_keys[i], ' ');
  WriteLn;

  cpu_output[0] := input_size; // Default to not found
  for i := 0 to input_size - 1 do
  begin
    for j := 0 to keys_size - 1 do
    begin
      if h_input[i] = h_keys[j] then
      begin
        cpu_output[0] := i;
        break;
      end;
    end;
    if cpu_output[0] <> input_size then Break;
  end;

  WriteLn(' - Expected (CPU): ', cpu_output[0]);

  CheckHIP(hipStreamCreate( @stream), 'Stream Create');

  CheckHIP(hipMalloc( @d_input, input_size * sizeof(cint)), 'Alloc d_input');
  CheckHIP(hipMalloc( @d_keys, keys_size * sizeof(cint)), 'Alloc d_keys');
  CheckHIP(hipMalloc( @d_output, 1 * sizeof(csize_t)), 'Alloc d_output');

  CheckHIP(hipMemcpy(d_input, @h_input[0], input_size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD input');
  CheckHIP(hipMemcpy(d_keys, @h_keys[0], keys_size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD keys');

  temp_storage_size := 0;
  CheckHIP(
    rocprim_find_first_of_int(nil, @temp_storage_size, Pointer(d_input), Pointer(d_keys), Pointer(d_output), input_size, keys_size, stream, True),
    'Get temp storage size'
  );
  WriteLn(' - Temp storage size: ', temp_storage_size, ' bytes');

  if temp_storage_size > 0 then
    CheckHIP(hipMalloc( @d_temp_storage, temp_storage_size), 'Alloc d_temp_storage')
  else
    d_temp_storage := nil;

  WriteLn(' - Executing on GPU...');
  CheckHIP(
    rocprim_find_first_of_int(Pointer(d_temp_storage), @temp_storage_size, Pointer(d_input), Pointer(d_keys), Pointer(d_output), input_size, keys_size, stream, True),
    'GPU find_first_of'
  );

  CheckHIP(hipMemcpy( @h_output[0], d_output, 1 * sizeof(csize_t), hipMemcpyDeviceToHost), 'DtoH output');

  WriteLn(' - GPU output: ', h_output[0]);

  if h_output[0] = cpu_output[0] then
    WriteLn('SUCCESS: GPU result matches CPU result!')
  else
    WriteLn('FAILURE: GPU result does not match CPU result.');

  if assigned(d_temp_storage) then hipFree(d_temp_storage);
  hipFree(d_input);
  hipFree(d_keys);
  hipFree(d_output);
  hipStreamDestroy(stream);

  WriteLn('Test complete.');
end.
