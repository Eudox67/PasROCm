program test_adjacent_difference;

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
  h_input: array[0..9] of cint = (8, 7, 6, 5, 4, 3, 2, 1, 0, -1);
  h_output: array[0..9] of cint;
  cpu_output: array[0..9] of cint;

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
  WriteLn('Starting rocprim::adjacent_difference test...');

  // --- 1. Setup --- //
  num_items := Length(h_input);
  WriteLn(' - Number of items: ', num_items);

  Write(' - Input: ');
  for i := 0 to High(h_input) do Write(h_input[i], ' ');
  WriteLn;

  // Prepare CPU expected output (using minus operator: input[i] - input[i-1])
  cpu_output[0] := h_input[0];
  for i := 1 to num_items - 1 do
    cpu_output[i] := h_input[i] - h_input[i-1];

  Write(' - Expected output: ');
  for i := 0 to num_items - 1 do Write(cpu_output[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_input, num_items * sizeof(cint)), 'Allocating d_input');
  CheckHIP(hipMalloc(@d_output, num_items * sizeof(cint)), 'Allocating d_output');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_input, @h_input[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_input to d_input');

  // --- 4. Get temporary storage size and allocate --- //
  temp_storage_size := 0;
  CheckHIP(
    rocprim_adjacent_difference_int(nil, @temp_storage_size, d_input, d_output, num_items, stream),
    'Getting temporary storage size'
  );
  CheckHIP(hipMalloc(@d_temp_storage, temp_storage_size), 'Allocating d_temp_storage');

  // --- 5. Perform adjacent_difference on the GPU --- //
  WriteLn(' - Performing adjacent_difference on GPU...');
  CheckHIP(
    rocprim_adjacent_difference_int(d_temp_storage, @temp_storage_size, d_input, d_output, num_items, stream),
    'Performing adjacent_difference'
  );

  // --- 6. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_output[0], d_output, num_items * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_output to h_output');

  // --- 7. Verification and Cleanup --- //
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
  CheckHIP(hipFree(d_input), 'Freeing d_input');
  CheckHIP(hipFree(d_output), 'Freeing d_output');
  CheckHIP(hipFree(d_temp_storage), 'Freeing d_temp_storage');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.