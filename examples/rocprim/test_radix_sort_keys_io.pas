program test_radix_sort_keys_io;

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
  h_keys_in: array[0..9] of cint = (5, 8, 3, 1, 9, 4, 2, 7, 6, 0);
  h_keys_out: array[0..9] of cint;
  cpu_keys_out: array[0..9] of cint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  begin_bit: cuint = 0;
  end_bit: cuint = 32; // 8 * sizeof(int)

  // Device pointers
  d_keys_in: Pointer = nil;
  d_keys_out: Pointer = nil;

  // Other variables
  num_items: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::radix_sort_keys_io test...');

  // --- 1. Setup --- //
  num_items := Length(h_keys_in);
  WriteLn(' - Number of items to sort: ', num_items);

  Write(' - Input keys: ');
  for i := 0 to High(h_keys_in) do Write(h_keys_in[i], ' ');
  WriteLn;
  Write(' - Expected output: ');
  for i := 0 to High(cpu_keys_out) do Write(cpu_keys_out[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_keys_in, num_items * sizeof(cint)), 'Allocating d_keys_in');
  CheckHIP(hipMalloc(@d_keys_out, num_items * sizeof(cint)), 'Allocating d_keys_out');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_keys_in, @h_keys_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_keys_in to d_keys_in');

  // --- 4. Perform radix_sort_keys_io on the GPU --- //
  WriteLn(' - Performing radix_sort_keys_io on GPU...');
  CheckHIP(
    rocprim_radix_sort_keys_int_io(d_keys_in, d_keys_out, num_items, begin_bit, end_bit, stream),
    'Performing radix_sort_keys_io'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_keys_out[0], d_keys_out, num_items * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_keys_out to h_keys_out');

  // --- 6. Verification and Cleanup --- //
  Write(' - GPU output: ');
  for i := 0 to num_items - 1 do Write(h_keys_out[i], ' ');
  WriteLn;

  match := true;
  for i := 0 to num_items - 1 do
  begin
    if h_keys_out[i] <> cpu_keys_out[i] then
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
  CheckHIP(hipFree(d_keys_in), 'Freeing d_keys_in');
  CheckHIP(hipFree(d_keys_out), 'Freeing d_keys_out');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
