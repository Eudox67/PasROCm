{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_histogram_even_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_histogram_even.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_histogram_even
------------------------------------------------------------------------------}

program test_histogram_even;

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
  h_samples_in: array[0..9] of cint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  h_histogram_out: array[0..4] of cint;
  cpu_histogram_out: array[0..4] of cint = (2, 2, 2, 2, 2);
  levels: cuint = 6;
  lower_level: cint = 0;
  upper_level: cint = 10;

  // Device pointers
  d_samples_in: Pointer = nil;
  d_histogram_out: Pointer = nil;

  // Other variables
  num_items: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::histogram_even test...');

  // --- 1. Setup --- //
  num_items := Length(h_samples_in);
  WriteLn(' - Number of samples: ', num_items);
  WriteLn(' - Levels: ', levels);
  WriteLn(' - Lower level: ', lower_level);
  WriteLn(' - Upper level: ', upper_level);

  // Calculate expected result on the CPU
  // For this simple case, each bin should have 2 elements
  // Bins: [0,2), [2,4), [4,6), [6,8), [8,10)
  // Samples: 0,1,2,3,4,5,6,7,8,9
  // Bin 0: 0,1
  // Bin 1: 2,3
  // Bin 2: 4,5
  // Bin 3: 6,7
  // Bin 4: 8,9
  // So, cpu_histogram_out should be (2,2,2,2,2)

  Write(' - Input samples: ');
  for i := 0 to High(h_samples_in) do Write(h_samples_in[i], ' ');
  WriteLn;
  Write(' - Expected histogram: ');
  for i := 0 to High(cpu_histogram_out) do Write(cpu_histogram_out[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_samples_in, num_items * sizeof(cint)), 'Allocating d_samples_in');
  CheckHIP(hipMalloc(@d_histogram_out, levels * sizeof(cint)), 'Allocating d_histogram_out');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_samples_in, @h_samples_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_samples_in to d_samples_in');

  // --- 4. Perform histogram_even on the GPU --- //
  WriteLn(' - Performing histogram_even on GPU...');
  CheckHIP(
    rocprim_histogram_even_int(d_samples_in, num_items, d_histogram_out, levels, lower_level, upper_level, stream),
    'Performing histogram_even'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_histogram_out[0], d_histogram_out, levels * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_histogram_out to h_histogram_out');

  // --- 6. Verification and Cleanup --- //
  Write(' - GPU histogram: ');
  for i := 0 to levels - 1 do Write(h_histogram_out[i], ' ');
  WriteLn;

  match := true;
  for i := 0 to levels - 1 do
  begin
    if h_histogram_out[i] <> cpu_histogram_out[i] then
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
  CheckHIP(hipFree(d_samples_in), 'Freeing d_samples_in');
  CheckHIP(hipFree(d_histogram_out), 'Freeing d_histogram_out');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
