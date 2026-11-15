{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_fill_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_fill.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_fill
------------------------------------------------------------------------------}

program test_fill;

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
  h_output: array[0..9] of cint;
  cpu_output: array[0..9] of cint;

  // Device pointers
  d_output: Pointer = nil;

  // Other variables
  num_items: csize_t;
  fill_value: cint;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::fill test...');

  // --- 1. Setup --- //
  num_items := 10;
  fill_value := 42;
  WriteLn(' - Number of items: ', num_items);
  WriteLn(' - Fill value: ', fill_value);

  // Prepare CPU expected output
  for i := 0 to num_items - 1 do
    cpu_output[i] := fill_value;

  Write(' - Expected output: ');
  for i := 0 to num_items - 1 do Write(cpu_output[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_output, num_items * sizeof(cint)), 'Allocating d_output');

  // --- 3. Perform fill on the GPU --- //
  WriteLn(' - Performing fill on GPU...');
  CheckHIP(
    rocprim_fill_int(d_output, fill_value, num_items, stream),
    'Performing fill'
  );

  // --- 4. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_output[0], d_output, num_items * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_output to h_output');

  // --- 5. Verification and Cleanup --- //
  Write(' - GPU output: ');
  for i := 0 to num_items - 1 do Write(h_output[i], ' ');
  WriteLn;

  match := true;
  for i := 0 to num_items - 1 do
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
  CheckHIP(hipFree(d_output), 'Freeing d_output');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.