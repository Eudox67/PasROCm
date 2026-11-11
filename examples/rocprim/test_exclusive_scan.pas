program test_exclusive_scan;

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
  h_in: array[0..9] of cint = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  h_out: array[0..9] of cint;
  cpu_result: array[0..9] of cint;
  initial_value: cint = 0;

  // Device pointers
  d_in: Pointer = nil;
  d_out: Pointer = nil;

  // Other variables
  num_items: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::exclusive_scan test...');

  // --- 1. Setup --- //
  num_items := Length(h_in);
  WriteLn(' - Number of items to scan: ', num_items);
  WriteLn(' - Initial value: ', initial_value);

  // Calculate expected result on the CPU
  cpu_result[0] := initial_value;
  for i := 1 to High(h_in) do
  begin
    cpu_result[i] := cpu_result[i-1] + h_in[i-1];
  end;
  WriteLn(' - CPU result: [0, 1, 3, ...]');

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_in, num_items * sizeof(cint)), 'Allocating d_in');
  CheckHIP(hipMalloc(@d_out, num_items * sizeof(cint)), 'Allocating d_out');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_in, @h_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_in to d_in');

  // --- 4. Perform scan on the GPU --- //
  // The shim now handles temporary storage internally.
  WriteLn(' - Performing exclusive scan on GPU...');
  CheckHIP(
    rocprim_exclusive_scan_int(d_out, d_in, num_items, initial_value, stream),
    'Performing exclusive scan'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_out[0], d_out, num_items * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_out to h_out');

  // --- 6. Verification and Cleanup --- //
  WriteLn(' - GPU result: [0, 1, 3, ...]');

  // Print arrays for debugging
  Write('CPU: ');
  for i := 0 to High(cpu_result) do Write(cpu_result[i], ' ');
  WriteLn;
  Write('GPU: ');
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
  CheckHIP(hipFree(d_in), 'Freeing d_in');
  CheckHIP(hipFree(d_out), 'Freeing d_out');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
