program test_select;

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

begin
  WriteLn('Starting rocprim::select test...');

  // --- 1. Setup --- //
  num_items := Length(h_in);
  WriteLn(' - Number of items to select from: ', num_items);
  WriteLn(' - Pivot: ', pivot);

  // Calculate expected result on the CPU
  cpu_selected_count := 0;
  for i := 0 to High(h_in) do
  begin
    if h_in[i] < pivot then
    begin
      cpu_result[cpu_selected_count] := h_in[i];
      Inc(cpu_selected_count);
    end;
  end;

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

  // --- 4. Perform select on the GPU --- //
  WriteLn(' - Performing select on GPU...');
  CheckHIP(
    rocprim_select_int(d_in, d_out, d_selected_count, num_items, pivot, stream),
    'Performing select'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_out[0], d_out, num_items * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_out to h_out');
  CheckHIP(hipMemcpy(@h_selected_count, d_selected_count, sizeof(csize_t), hipMemcpyDeviceToHost), 'Copying d_selected_count to h_selected_count');

  // --- 6. Verification and Cleanup --- //
  WriteLn(' - GPU selected count: ', h_selected_count);
  Write(' - GPU result: ');
  for i := 0 to h_selected_count - 1 do Write(h_out[i], ' ');
  WriteLn;

  match := (h_selected_count = cpu_selected_count);
  if(match) then
  begin
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
