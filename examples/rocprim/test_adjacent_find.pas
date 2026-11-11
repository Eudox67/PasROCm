program test_adjacent_find;

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
  h_input: array[0..9] of cint = (1, 2, 3, 3, 4, 5, 6, 7, 8, 9);
  h_output: csize_t;
  cpu_output: csize_t;

  // Device pointers
  d_input: Pointer = nil;
  d_output: Pointer = nil;
  d_temp_storage: Pointer = nil;

  // Other variables
  num_items: csize_t;
  temp_storage_size: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::adjacent_find test...');

  // --- 1. Setup --- //
  num_items := Length(h_input);
  WriteLn(' - Number of items: ', num_items);

  Write(' - Input: ');
  for i := 0 to High(h_input) do Write(h_input[i], ' ');
  WriteLn;

  // Prepare CPU expected output (first occurrence of adjacent equal elements)
  cpu_output := num_items; // Default to not found
  for i := 0 to num_items - 2 do
  begin
    if h_input[i] = h_input[i+1] then
    begin
      cpu_output := i;
      break;
    end;
  end;

  WriteLn(' - Expected output (index of first adjacent equal pair): ', cpu_output);

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_input, num_items * sizeof(cint)), 'Allocating d_input');
  CheckHIP(hipMalloc(@d_output, sizeof(csize_t)), 'Allocating d_output');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_input, @h_input[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_input to d_input');

  // --- 4. Get temporary storage size and allocate --- //
  temp_storage_size := 0;
  CheckHIP(
    rocprim_adjacent_find_int(nil, @temp_storage_size, d_input, d_output, num_items, stream),
    'Getting temporary storage size'
  );
  CheckHIP(hipMalloc(@d_temp_storage, temp_storage_size), 'Allocating d_temp_storage');

  // --- 5. Perform adjacent_find on the GPU --- //
  WriteLn(' - Performing adjacent_find on GPU...');
  CheckHIP(
    rocprim_adjacent_find_int(d_temp_storage, @temp_storage_size, d_input, d_output, num_items, stream),
    'Performing adjacent_find'
  );

  // --- 6. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_output, d_output, sizeof(csize_t), hipMemcpyDeviceToHost), 'Copying d_output to h_output');

  // --- 7. Verification and Cleanup --- //
  WriteLn(' - GPU output (index of first adjacent equal pair): ', h_output);

  match := (h_output = cpu_output);

  if match then
    WriteLn('SUCCESS: GPU result matches CPU result.')
  else
    WriteLn('FAILURE: GPU result does not match CPU result.');

  // Free device memory
  CheckHIP(hipFree(d_input), 'Freeing d_input');
  CheckHIP(hipFree(d_output), 'Freeing d_output');
  CheckHIP(hipFree(d_temp_storage), 'Freeing d_temp_storage');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
