{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_partition_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_partition.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_partition
------------------------------------------------------------------------------}

program test_partition;

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
  h_selected_count: csize_t = 0;
  cpu_result: array[0..9] of cint;
  cpu_selected_count: csize_t = 0;
  pivot: cint = 5;

  // Device pointers
  d_in: Pointer = nil;
  d_out: Pointer = nil;
  d_selected_count: Pointer = nil;

  // Other variables
  num_items: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;
  next_selected: Integer;
  next_rejected: Integer;
  j: Integer;
  temp: cint;

begin
  WriteLn('Starting rocprim::partition test...');

  // --- 1. Setup --- //
  num_items := Length(h_in);
  WriteLn(' - Number of items to partition: ', num_items);
  WriteLn(' - Pivot: ', pivot);

  // Calculate expected result on the CPU
  next_selected := 0;
  next_rejected := High(h_in);
  for i := 0 to High(h_in) do
  begin
    if h_in[i] < pivot then
    begin
      cpu_result[next_selected] := h_in[i];
      Inc(next_selected);
    end
    else
    begin
      cpu_result[next_rejected] := h_in[i];
      Dec(next_rejected);
    end;
  end;
  cpu_selected_count := next_selected;

  Write(' - Input: ');
  for i := 0 to High(h_in) do Write(h_in[i], ' ');
  WriteLn;
  Write(' - Expected selected: ');
  for i := 0 to cpu_selected_count - 1 do Write(cpu_result[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_in, num_items * sizeof(cint)), 'Allocating d_in');
  CheckHIP(hipMalloc(@d_out, num_items * sizeof(cint)), 'Allocating d_out');
  CheckHIP(hipMalloc(@d_selected_count, sizeof(csize_t)), 'Allocating d_selected_count');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_in, @h_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_in to d_in');

  // --- 4. Perform partition on the GPU --- //
  WriteLn(' - Performing partition on GPU...');
  CheckHIP(
    rocprim_partition_int(d_in, d_out, d_selected_count, num_items, pivot, stream),
    'Performing partition'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_out[0], d_out, num_items * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_out to h_out');
  CheckHIP(hipMemcpy(@h_selected_count, d_selected_count, sizeof(csize_t), hipMemcpyDeviceToHost), 'Copying d_selected_count to h_selected_count');

  // --- 6. Verification and Cleanup --- //
  WriteLn(' - GPU selected count: ', h_selected_count);
  Write(' - GPU result: ');
  for i := 0 to num_items - 1 do Write(h_out[i], ' ');
  WriteLn;

  match := (h_selected_count = cpu_selected_count);
  if(match) then
  begin
    // Sort both sub-arrays for comparison, as order is not guaranteed for rejected items
    // and only guaranteed for selected items if the algorithm is stable.
    // For simplicity, we sort both.
    // A more robust test would check for stability if required.
    // FPC's built-in sort is not available, so we do a simple bubble sort for the test.
    for i := 0 to cpu_selected_count - 2 do
    begin
      for j := i + 1 to cpu_selected_count - 1 do
      begin
        if (h_out[i] > h_out[j]) then
        begin
          temp := h_out[i];
          h_out[i] := h_out[j];
          h_out[j] := temp;
        end;
        if (cpu_result[i] > cpu_result[j]) then
        begin
          temp := cpu_result[i];
          cpu_result[i] := cpu_result[j];
          cpu_result[j] := temp;
        end;
      end;
    end;

    for i := 0 to cpu_selected_count - 1 do
    begin
      if h_out[i] <> cpu_result[i] then
      begin
        match := false;
        break;
      end;
    end;
  end;

  if match then
    WriteLn('SUCCESS: GPU result matches CPU result.')
  else
    WriteLn('FAILURE: GPU result does not match CPU result.');

  // Free device memory
  CheckHIP(hipFree(d_in), 'Freeing d_in');
  CheckHIP(hipFree(d_out), 'Freeing d_out');
  CheckHIP(hipFree(d_selected_count), 'Freeing d_selected_count');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
