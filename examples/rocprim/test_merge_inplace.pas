{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_merge_inplace_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_merge_inplace.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_merge_inplace
------------------------------------------------------------------------------}

program test_merge_inplace;

{$mode objfpc}{$H+}

uses
  ctypes,
  hip_bindings,      // For hipMalloc, hipMemcpy, etc.
  rocprim_bindings;  // For the rocprim shim functions

// Helper procedure to check HIP API call results
procedure CheckHIP(err: hipError_t; const msg: string);
begin
  if err <> hipSuccess then
  begin
    WriteLn('HIP Error: ', err, ' at ', msg);
    Halt(1);
  end;
end;

// Helper procedure for CPU merge (for verification)
procedure cpu_merge_inplace(var data: array of cint; left_size, right_size: csize_t);
var
  i, j, k: csize_t;
  temp: array of cint;
begin
  SetLength(temp, left_size + right_size);
  i := 0; // index for left part
  j := left_size; // index for right part
  k := 0; // index for temp array

  while (i < left_size) and (j < left_size + right_size) do
  begin
    if data[i] < data[j] then
    begin
      temp[k] := data[i];
      Inc(i);
    end
    else
    begin
      temp[k] := data[j];
      Inc(j);
    end;
    Inc(k);
  end;

  while i < left_size do
  begin
    temp[k] := data[i];
    Inc(i);
    Inc(k);
  end;

  while j < left_size + right_size do
  begin
    temp[k] := data[j];
    Inc(j);
    Inc(k);
  end;

  for i := 0 to left_size + right_size - 1 do
    data[i] := temp[i];
end;

var
  // Host data
  h_data: array[0..7] of cint = (1, 3, 5, 7, 0, 2, 4, 6);
  h_output: array[0..7] of cint;
  cpu_output: array[0..7] of cint;

  // Device pointers
  d_data: Pointer = nil;
  d_temp_storage: Pointer = nil;

  // Other variables
  left_size: csize_t;
  right_size: csize_t;
  total_size: csize_t;
  temp_storage_size: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::merge_inplace test...');

  // --- 1. Setup --- //
  left_size := 4;
  right_size := 4;
  total_size := left_size + right_size;
  WriteLn(' - Left size: ', left_size);
  WriteLn(' - Right size: ', right_size);

  Write(' - Input: ');
  for i := 0 to High(h_data) do Write(h_data[i], ' ');
  WriteLn;

  // Prepare CPU expected output
  for i := 0 to High(h_data) do cpu_output[i] := h_data[i];
  cpu_merge_inplace(cpu_output, left_size, right_size);

  Write(' - Expected output: ');
  for i := 0 to High(cpu_output) do Write(cpu_output[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_data, total_size * sizeof(cint)), 'Allocating d_data');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_data, @h_data[0], total_size * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_data to d_data');

  // --- 4. Get temporary storage size and allocate --- //
  temp_storage_size := 0;
  CheckHIP(
    rocprim_merge_inplace_int(nil, @temp_storage_size, d_data, left_size, right_size, stream),
    'Getting temporary storage size'
  );
  CheckHIP(hipMalloc(@d_temp_storage, temp_storage_size), 'Allocating d_temp_storage');

  // --- 5. Perform merge_inplace on the GPU --- //
  WriteLn(' - Performing merge_inplace on GPU...');
  CheckHIP(
    rocprim_merge_inplace_int(d_temp_storage, @temp_storage_size, d_data, left_size, right_size, stream),
    'Performing merge_inplace'
  );

  // --- 6. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_output[0], d_data, total_size * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_data to h_output');

  // --- 7. Verification and Cleanup --- //
  Write(' - GPU output: ');
  for i := 0 to total_size - 1 do Write(h_output[i], ' ');
  WriteLn;

  match := true;
  for i := 0 to total_size - 1 do
  begin
    if h_output[i] <> cpu_output[i] then
    begin
      match := false;
      break;
    end;
  end;

  if match then
    WriteLn('SUCCESS: GPU result matches CPU result.')
  else
    WriteLn('FAILURE: GPU result does not match CPU result.');

  // Free device memory
  CheckHIP(hipFree(d_data), 'Freeing d_data');
  CheckHIP(hipFree(d_temp_storage), 'Freeing d_temp_storage');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
