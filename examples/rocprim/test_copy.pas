program test_copy;

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
  h_input: array[0..9] of cint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  h_output: array[0..9] of cint;
  cpu_output: array[0..9] of cint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

  // Device pointers
  d_input: Pointer = nil;
  d_output: Pointer = nil;

  // Other variables
  num_items: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::copy test...');

  // --- 1. Setup --- //
  num_items := Length(h_input);
  WriteLn(' - Number of items: ', num_items);

  Write(' - Input: ');
  for i := 0 to High(h_input) do Write(h_input[i], ' ');
  WriteLn;
  Write(' - Expected output: ');
  for i := 0 to High(cpu_output) do Write(cpu_output[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_input, num_items * sizeof(cint)), 'Allocating d_input');
  CheckHIP(hipMalloc(@d_output, num_items * sizeof(cint)), 'Allocating d_output');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_input, @h_input[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_input to d_input');

  // --- 4. Perform copy on the GPU --- //
  WriteLn(' - Performing copy on GPU...');
  CheckHIP(
    rocprim_copy_int(d_input, d_output, num_items, stream),
    'Performing copy'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_output[0], d_output, num_items * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_output to h_output');

  // --- 6. Verification and Cleanup --- //
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
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
