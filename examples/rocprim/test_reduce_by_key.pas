program test_reduce_by_key;

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
  h_unique_keys_out: array[0..3] of cint;
  h_aggregates_out: array[0..3] of cint;
  h_unique_count_out: csize_t = 0;
  cpu_unique_keys_out: array[0..3] of cint = (1, 2, 10, 88);
  cpu_aggregates_out: array[0..3] of cint = (6, 4, 18, 27);
  cpu_unique_count_out: csize_t = 4;

  // Device pointers
  d_keys_in: Pointer = nil;
  d_values_in: Pointer = nil;
  d_unique_keys_out: Pointer = nil;
  d_aggregates_out: Pointer = nil;
  d_unique_count_out: Pointer = nil;

  // Other variables
  num_items: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::reduce_by_key test...');

  // --- 1. Setup --- //
  num_items := Length(h_keys_in);
  WriteLn(' - Number of items to reduce: ', num_items);

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
  CheckHIP(hipMalloc(@d_unique_keys_out, cpu_unique_count_out * sizeof(cint)), 'Allocating d_unique_keys_out');
  CheckHIP(hipMalloc(@d_aggregates_out, cpu_unique_count_out * sizeof(cint)), 'Allocating d_aggregates_out');
  CheckHIP(hipMalloc(@d_unique_count_out, sizeof(csize_t)), 'Allocating d_unique_count_out');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_keys_in, @h_keys_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_keys_in to d_keys_in');
  CheckHIP(hipMemcpy(d_values_in, @h_values_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_values_in to d_values_in');

  // --- 4. Perform reduce_by_key on the GPU --- //
  WriteLn(' - Performing reduce_by_key on GPU...');
  CheckHIP(
    rocprim_reduce_by_key_int(d_keys_in, d_values_in, num_items, d_unique_keys_out, d_aggregates_out, d_unique_count_out, stream),
    'Performing reduce_by_key'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_unique_keys_out[0], d_unique_keys_out, cpu_unique_count_out * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_unique_keys_out to h_unique_keys_out');
  CheckHIP(hipMemcpy(@h_aggregates_out[0], d_aggregates_out, cpu_unique_count_out * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_aggregates_out to h_aggregates_out');
  CheckHIP(hipMemcpy(@h_unique_count_out, d_unique_count_out, sizeof(csize_t), hipMemcpyDeviceToHost), 'Copying d_unique_count_out to h_unique_count_out');

  // --- 6. Verification and Cleanup --- //
  WriteLn(' - GPU unique count: ', h_unique_count_out);
  Write(' - GPU unique keys: ');
  for i := 0 to h_unique_count_out - 1 do Write(h_unique_keys_out[i], ' ');
  WriteLn;
  Write(' - GPU aggregates: ');
  for i := 0 to h_unique_count_out - 1 do Write(h_aggregates_out[i], ' ');
  WriteLn;

  match := (h_unique_count_out = cpu_unique_count_out);
  if(match) then
  begin
    for i := 0 to cpu_unique_count_out - 1 do
    begin
      if (h_unique_keys_out[i] <> cpu_unique_keys_out[i]) or (h_aggregates_out[i] <> cpu_aggregates_out[i]) then
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
  CheckHIP(hipFree(d_keys_in), 'Freeing d_keys_in');
  CheckHIP(hipFree(d_values_in), 'Freeing d_values_in');
  CheckHIP(hipFree(d_unique_keys_out), 'Freeing d_unique_keys_out');
  CheckHIP(hipFree(d_aggregates_out), 'Freeing d_aggregates_out');
  CheckHIP(hipFree(d_unique_count_out), 'Freeing d_unique_count_out');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
