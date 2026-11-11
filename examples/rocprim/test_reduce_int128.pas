program test_reduce_int128;

{$mode objfpc}{$H+}

uses
  ctypes,
  lgint128,
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
  h_input: array[0..9] of TInt128;
  h_output: TInt128;
  d_input: Pointer = nil;
  d_output: Pointer = nil;
  num_items: csize_t;
  i: Integer;
  stream: hipStream_t;
  expected_sum: TInt128;
  zero: TInt128;

begin
  WriteLn('Starting rocprim::reduce test for TInt128...');

  // --- 1. Setup --- //
  num_items := Length(h_input);
  expected_sum.SetZero();
  zero.SetZero();

  for i := 0 to High(h_input) do
  begin
    h_input[i] := TInt128.Encode(i + 1);
    expected_sum := expected_sum + h_input[i];
  end;

  WriteLn(' - Number of items: ', num_items);
  Write(' - Input: ');
  for i := 0 to High(h_input) do Write(h_input[i].ToString, ' ');
  WriteLn;
  WriteLn(' - Expected output: ', expected_sum.ToString);

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_input, num_items * sizeof(TInt128)), 'Allocating d_input');
  CheckHIP(hipMalloc(@d_output, sizeof(TInt128)), 'Allocating d_output');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_input, @h_input[0], num_items * sizeof(TInt128), hipMemcpyHostToDevice), 'Copying h_input to d_input');

  // --- 4. Perform reduce on the GPU --- //
  WriteLn(' - Performing reduce on GPU...');
  CheckHIP(
    rocprim_reduce_int128(PInt128(d_output), PInt128(d_input), num_items, zero, stream),
    'Performing reduce'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_output, d_output, sizeof(TInt128), hipMemcpyDeviceToHost), 'Copying d_output to h_output');

  // --- 6. Verification and Cleanup --- //
  WriteLn(' - GPU output: ', h_output.ToString);

  if TInt128.Compare(h_output, expected_sum) = 0 then
    WriteLn('SUCCESS: GPU result matches expected result.')
  else
    WriteLn('FAILURE: GPU result does not match expected result.');

  // Free device memory
  CheckHIP(hipFree(d_input), 'Freeing d_input');
  CheckHIP(hipFree(d_output), 'Freeing d_output');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');
  WriteLn('Test complete.');
end.
