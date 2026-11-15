{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_inclusive_scan_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_scan.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_scan
------------------------------------------------------------------------------}

program test_scan;

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
  h_in: array[0..9] of cint = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  h_out: array[0..9] of cint;
  cpu_result: array[0..9] of cint;

  // Device pointers
  d_in: Pointer = nil;
  d_out: Pointer = nil;

  // Other variables
  num_items: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::inclusive_scan test...');

  // --- 1. Setup --- //
  num_items := Length(h_in);
  WriteLn(' - Number of items to scan: ', num_items);

  // Calculate expected result on the CPU
  cpu_result[0] := h_in[0];
  for i := 1 to High(h_in) do
  begin
    cpu_result[i] := cpu_result[i-1] + h_in[i];
  end;
  WriteLn(' - CPU result: [1, 3, 6, ...]');

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_in, num_items * sizeof(cint)), 'Allocating d_in');
  CheckHIP(hipMalloc(@d_out, num_items * sizeof(cint)), 'Allocating d_out');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_in, @h_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_in to d_in');

  // --- 4. Perform scan on the GPU --- //
  // The shim now handles temporary storage internally.
  WriteLn(' - Performing inclusive scan on GPU...');
  CheckHIP(
    rocprim_inclusive_scan_int(d_out, d_in, num_items, stream),
    'Performing inclusive scan'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_out[0], d_out, num_items * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_out to h_out');

  // --- 6. Verification and Cleanup --- //
  WriteLn(' - GPU result: [1, 3, 6, ...]');

  match := true;
  for i := 0 to High(h_in) do
  begin
    if h_out[i] <> cpu_result[i] then
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
  CheckHIP(hipFree(d_in), 'Freeing d_in');
  CheckHIP(hipFree(d_out), 'Freeing d_out');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
