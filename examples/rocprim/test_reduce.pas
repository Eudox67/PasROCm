{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_reduce_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_reduce.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_reduce
------------------------------------------------------------------------------}

program test_reduce;

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
  h_out: cint = 0;
  cpu_result: cint = 0;

  // Device pointers
  d_in: Pointer = nil;
  d_out: Pointer = nil;

  // Other variables
  num_items: csize_t;
  i: Integer;
  stream: hipStream_t;

begin
  WriteLn('Starting rocprim::reduce test...');

  // --- 1. Setup --- //
  num_items := Length(h_in);
  WriteLn(' - Number of items to reduce: ', num_items);

  // Calculate expected result on the CPU
  for i := 0 to High(h_in) do
  begin
    cpu_result := cpu_result + h_in[i];
  end;
  WriteLn(' - CPU result: ', cpu_result);

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_in, num_items * sizeof(cint)), 'Allocating d_in');
  CheckHIP(hipMalloc(@d_out, sizeof(cint)), 'Allocating d_out');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_in, @h_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_in to d_in');

  // --- 4. Perform reduction on the GPU --- //
  // The shim now handles temporary storage internally.
  WriteLn(' - Performing reduction on GPU...');
  CheckHIP(
    rocprim_reduce_int(d_out, d_in, num_items, 0, stream),
    'Performing reduction'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_out, d_out, sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_out to h_out');

  // --- 6. Verification and Cleanup --- //
  WriteLn(' - GPU result: ', h_out);

  if h_out = cpu_result then
    WriteLn('SUCCESS: GPU result matches CPU result.')
  else
    WriteLn('FAILURE: GPU result does not match CPU result.');

  // Free device memory
  CheckHIP(hipFree(d_in), 'Freeing d_in');
  CheckHIP(hipFree(d_out), 'Freeing d_out');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
