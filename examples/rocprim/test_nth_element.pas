{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_nth_element_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_nth_element.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_nth_element
------------------------------------------------------------------------------}

program test_nth_element;

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

  nth_element_idx: csize_t;
  size: csize_t;
  temp_storage_size: csize_t;
  i, j: cint;
  temp: cint;

  stream: hipStream_t;

  match: Boolean;

begin
  WriteLn('Starting rocprim::nth_element test...');

  size := Length(h_keys_input);
  nth_element_idx := 4; // Find the 4th smallest element (0-indexed)

  WriteLn(' - Input size: ', size);
  WriteLn(' - Nth element index: ', nth_element_idx);
  
  Write(' - Input keys: ');
  for i := 0 to size - 1 do Write(h_keys_input[i], ' ');
  WriteLn;

  // CPU reference (sort and pick nth element)
  for i := 0 to size - 1 do cpu_keys_output[i] := h_keys_input[i];
  // Simple sort for reference
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
  // The element at nth_element_idx should be the correct one
  // The elements before it should be <= it, and after it should be >= it.
  // For simplicity, we'll just check the value at nth_element_idx
  // and that the output is a permutation of the input.

  WriteLn(' - Expected (CPU) nth element value: ', cpu_keys_output[nth_element_idx]);

  CheckHIP(hipStreamCreate( @stream), 'Stream Create');

  CheckHIP(hipMalloc( @d_keys_input, size * sizeof(cint)), 'Alloc d_keys_input');
  CheckHIP(hipMalloc( @d_keys_output, size * sizeof(cint)), 'Alloc d_keys_output');

  CheckHIP(hipMemcpy(d_keys_input, @h_keys_input[0], size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD keys_input');

  temp_storage_size := 0;
  CheckHIP(
    rocprim_nth_element_int(nil, @temp_storage_size, Pointer(d_keys_input), Pointer(d_keys_output), nth_element_idx, size, stream, True),
    'Get temp storage size'
  );
  WriteLn(' - Temp storage size: ', temp_storage_size, ' bytes');

  if temp_storage_size > 0 then
    CheckHIP(hipMalloc( @d_temp_storage, temp_storage_size), 'Alloc d_temp_storage')
  else
    d_temp_storage := nil;

  WriteLn(' - Executing on GPU...');
  CheckHIP(
    rocprim_nth_element_int(Pointer(d_temp_storage), @temp_storage_size, Pointer(d_keys_input), Pointer(d_keys_output), nth_element_idx, size, stream, True),
    'GPU nth_element'
  );

  CheckHIP(hipMemcpy( @h_keys_output[0], d_keys_output, size * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH keys_output');

  Write(' - GPU output keys: ');
  for i := 0 to size - 1 do Write(h_keys_output[i], ' ');
  WriteLn;

  // Verification: Check if the nth element is correct and other elements are partitioned correctly
  match := True;
  if h_keys_output[nth_element_idx] <> cpu_keys_output[nth_element_idx] then
  begin
    match := False;
    WriteLn('  Mismatch at nth element: GPU=', h_keys_output[nth_element_idx], ', Expected=', cpu_keys_output[nth_element_idx]);
  end;

  // Check partitioning: elements before nth should be <= nth, elements after nth should be >= nth
  if match then
  begin
    for i := 0 to nth_element_idx - 1 do
    begin
      if h_keys_output[i] > h_keys_output[nth_element_idx] then
      begin
        match := False;
        WriteLn('  Partition mismatch before nth element at index ', i, ': ', h_keys_output[i], ' > ', h_keys_output[nth_element_idx]);
        Break;
      end;
    end;
  end;

  if match then
  begin
    for i := nth_element_idx + 1 to size - 1 do
    begin
      if h_keys_output[i] < h_keys_output[nth_element_idx] then
      begin
        match := False;
        WriteLn('  Partition mismatch after nth element at index ', i, ': ', h_keys_output[i], ' < ', h_keys_output[nth_element_idx]);
        Break;
      end;
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
