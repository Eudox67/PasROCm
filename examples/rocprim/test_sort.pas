{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_radix_sort_keys_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_sort.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_sort
------------------------------------------------------------------------------}

program test_sort;

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
  h_in: array[0..9] of cint = (5, 8, 3, 1, 9, 4, 2, 7, 6, 0);
  h_out: array[0..9] of cint;
  cpu_result: array[0..9] of cint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

  // Device pointers
  d_keys: Pointer = nil;

  // Other variables
  num_items: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::radix_sort_keys test...');

  // --- 1. Setup --- //
  num_items := Length(h_in);
  WriteLn(' - Number of items to sort: ', num_items);

  // Display original and expected data
  Write(' - Input: ');
  for i := 0 to High(h_in) do Write(h_in[i], ' ');
  WriteLn;
  Write(' - Expected: ');
  for i := 0 to High(cpu_result) do Write(cpu_result[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_keys, num_items * sizeof(cint)), 'Allocating d_keys');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_keys, @h_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_in to d_keys');

  // --- 4. Perform sort on the GPU --- //
  // The shim now handles temporary storage internally.
  WriteLn(' - Performing radix sort on GPU...');
  CheckHIP(
    rocprim_radix_sort_keys_int(d_keys, num_items, stream),
    'Performing radix sort'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_out[0], d_keys, num_items * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_keys to h_out');

  // --- 6. Verification and Cleanup --- //
  Write(' - GPU result: ');
  for i := 0 to High(h_out) do Write(h_out[i], ' ');
  WriteLn;

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
  CheckHIP(hipFree(d_keys), 'Freeing d_keys');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
