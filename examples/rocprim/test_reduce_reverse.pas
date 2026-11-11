program test_reduce_reverse;

{$mode objfpc}{$H+}

uses
  ctypes,
  hip_bindings,
  rocprim_bindings;

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
  h_input: array[0..9] of cint;
  h_output: cint;
  d_input: Pointer = nil;
  d_output: Pointer = nil;
  num_items: csize_t;
  i: Integer;
  stream: hipStream_t;
  expected_sum: cint;

begin
  WriteLn('Starting rocprim::reduce with reverse_iterator test...');

  // --- 1. Setup --- //
  num_items := Length(h_input);
  expected_sum := 0;
  for i := 0 to High(h_input) do
  begin
    h_input[i] := i + 1;
    expected_sum := expected_sum + h_input[i];
  end;

  WriteLn(' - Number of items: ', num_items);
  Write(' - Input: ');
  for i := 0 to High(h_input) do Write(h_input[i], ' ');
  WriteLn;
  WriteLn(' - Expected output: ', expected_sum);

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_input, num_items * sizeof(cint)), 'Allocating d_input');
  CheckHIP(hipMalloc(@d_output, sizeof(cint)), 'Allocating d_output');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_input, @h_input[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_input to d_input');

  // --- 4. Perform reduce on the GPU --- //
  WriteLn(' - Performing reduce on GPU with reverse_iterator...');
  CheckHIP(
    rocprim_reduce_reverse_int(PCint(d_output), PCint(d_input), num_items, 0, stream),
    'Performing reduce with reverse_iterator'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_output, d_output, sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_output to h_output');

  // --- 6. Verification and Cleanup --- //
  WriteLn(' - GPU output: ', h_output);

  if h_output = expected_sum then
    WriteLn('SUCCESS: GPU result matches expected result.')
  else
    WriteLn('FAILURE: GPU result does not match expected result.');

  // Free device memory
  CheckHIP(hipFree(d_input), 'Freeing d_input');
  CheckHIP(hipFree(d_output), 'Freeing d_output');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');
  WriteLn('Test complete.');
end.
