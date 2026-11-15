{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_partial_sort_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_partial_sort.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_partial_sort
------------------------------------------------------------------------------}

program test_partial_sort;

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
  h_keys: array[0..7] of cint = (6, 3, 5, 4, 1, 8, 2, 7);
  h_keys_output: array[0..7] of cint;
  cpu_keys_output: array[0..7] of cint;

  d_keys: Pointer = nil;
  d_temp_storage: Pointer = nil;

  middle, size, temp_storage_size: csize_t;
  i, j, temp: cint;

  stream: hipStream_t;

  match: Boolean;

begin
  WriteLn('Starting rocprim::partial_sort test...');

  size := Length(h_keys);
  middle := 4; // Sort the first 5 elements

  WriteLn(' - Input size: ', size);
  WriteLn(' - Middle index: ', middle);
  
  Write(' - Input keys: ');
  for i := 0 to size - 1 do Write(h_keys[i], ' ');
  WriteLn;

  // CPU reference: sort the prefix, the rest is undefined
  for i := 0 to size - 1 do cpu_keys_output[i] := h_keys[i];
  // Bubble sort for simplicity on the whole array to find the correct prefix
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

  Write(' - Expected (CPU) sorted prefix: ');
  for i := 0 to middle do Write(cpu_keys_output[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate( @stream), 'Stream Create');

  CheckHIP(hipMalloc( @d_keys, size * sizeof(cint)), 'Alloc d_keys');

  CheckHIP(hipMemcpy(d_keys, @h_keys[0], size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD keys');

  temp_storage_size := 0;
  CheckHIP(
    rocprim_partial_sort_int(nil, @temp_storage_size, Pointer(d_keys), middle, size, stream, True),
    'Get temp storage size'
  );
  WriteLn(' - Temp storage size: ', temp_storage_size, ' bytes');

  if temp_storage_size > 0 then
    CheckHIP(hipMalloc( @d_temp_storage, temp_storage_size), 'Alloc d_temp_storage')
  else
    d_temp_storage := nil;

  WriteLn(' - Executing on GPU...');
  CheckHIP(
    rocprim_partial_sort_int(Pointer(d_temp_storage), @temp_storage_size, Pointer(d_keys), middle, size, stream, True),
    'GPU partial_sort'
  );

  CheckHIP(hipMemcpy( @h_keys_output[0], d_keys, size * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH keys_output');

  Write(' - GPU output keys: ');
  for i := 0 to size - 1 do Write(h_keys_output[i], ' ');
  WriteLn;

  // Verification: Check if the prefix is sorted correctly.
  // The order of elements after `middle` is not guaranteed.
  match := True;
  for i := 0 to middle do
  begin
    if h_keys_output[i] <> cpu_keys_output[i] then
    begin
      WriteLn('  Mismatch at index ', i, ': GPU=', h_keys_output[i], ', Expected=', cpu_keys_output[i]);
      match := False;
    end;
  end;

  if match then
    WriteLn('SUCCESS: GPU sorted prefix matches CPU result!')
  else
    WriteLn('FAILURE: GPU sorted prefix does not match CPU result.');

  if assigned(d_temp_storage) then hipFree(d_temp_storage);
  hipFree(d_keys);
  hipStreamDestroy(stream);

  WriteLn('Test complete.');
end.
