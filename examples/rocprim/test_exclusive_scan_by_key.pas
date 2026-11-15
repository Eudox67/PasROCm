{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_exclusive_scan_by_key_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_exclusive_scan_by_key.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_exclusive_scan_by_key
------------------------------------------------------------------------------}

program test_exclusive_scan_by_key;

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

var
  // Host data
  h_keys_in: array[0..9] of cint = (1, 1, 1, 2, 10, 10, 10, 88, 88, 88);
  h_values_in: array[0..9] of cint = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  h_values_out: array[0..9] of cint;
  cpu_values_out: array[0..9] of cint = (0, 1, 3, 0, 0, 5, 11, 0, 8, 17);
  initial_value: cint = 0;

  // Device pointers
  d_keys_in: Pointer = nil;
  d_values_in: Pointer = nil;
  d_values_out: Pointer = nil;

  // Other variables
  num_items: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;
  current_key: cint;
  current_aggregate: cint;

begin
  WriteLn('Starting rocprim::exclusive_scan_by_key test...');

  // --- 1. Setup --- //
  num_items := Length(h_keys_in);
  WriteLn(' - Number of items to scan: ', num_items);
  WriteLn(' - Initial value: ', initial_value);

  // Calculate expected result on the CPU
  current_key := h_keys_in[0];
  current_aggregate := initial_value;
  for i := 0 to High(h_keys_in) do
  begin
    if h_keys_in[i] <> current_key then
    begin
      current_key := h_keys_in[i];
      current_aggregate := initial_value;
    end;
    cpu_values_out[i] := current_aggregate;
    current_aggregate := current_aggregate + h_values_in[i];
  end;

  Write(' - Input keys: ');
  for i := 0 to High(h_keys_in) do Write(h_keys_in[i], ' ');
  WriteLn;
  Write(' - Input values: ');
  for i := 0 to High(h_values_in) do Write(h_values_in[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_keys_in, num_items * sizeof(cint)), 'Allocating d_keys_in');
  CheckHIP(hipMalloc(@d_values_in, num_items * sizeof(cint)), 'Allocating d_values_in');
  CheckHIP(hipMalloc(@d_values_out, num_items * sizeof(cint)), 'Allocating d_values_out');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_keys_in, @h_keys_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_keys_in to d_keys_in');
  CheckHIP(hipMemcpy(d_values_in, @h_values_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_values_in to d_values_in');

  // --- 4. Perform exclusive_scan_by_key on the GPU --- //
  WriteLn(' - Performing exclusive_scan_by_key on GPU...');
  CheckHIP(
    rocprim_exclusive_scan_by_key_int(d_keys_in, d_values_in, d_values_out, num_items, initial_value, stream),
    'Performing exclusive_scan_by_key'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_values_out[0], d_values_out, num_items * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_values_out to h_values_out');

  // --- 6. Verification and Cleanup --- //
  Write(' - GPU values out: ');
  for i := 0 to num_items - 1 do Write(h_values_out[i], ' ');
  WriteLn;

  match := true;
  for i := 0 to num_items - 1 do
  begin
    if h_values_out[i] <> cpu_values_out[i] then
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
  CheckHIP(hipFree(d_keys_in), 'Freeing d_keys_in');
  CheckHIP(hipFree(d_values_in), 'Freeing d_values_in');
  CheckHIP(hipFree(d_values_out), 'Freeing d_values_out');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
