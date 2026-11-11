program test_merge;

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
  h_input1: array[0..4] of cint = (0, 2, 4, 6, 8);
  h_input2: array[0..4] of cint = (1, 3, 5, 7, 9);
  h_output: array[0..9] of cint;
  cpu_output: array[0..9] of cint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

  // Device pointers
  d_input1: Pointer = nil;
  d_input2: Pointer = nil;
  d_output: Pointer = nil;

  // Other variables
  input1_size: csize_t;
  input2_size: csize_t;
  output_size: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::merge test...');

  // --- 1. Setup --- //
  input1_size := Length(h_input1);
  input2_size := Length(h_input2);
  output_size := input1_size + input2_size;
  WriteLn(' - Input1 size: ', input1_size);
  WriteLn(' - Input2 size: ', input2_size);

  Write(' - Input1: ');
  for i := 0 to High(h_input1) do Write(h_input1[i], ' ');
  WriteLn;
  Write(' - Input2: ');
  for i := 0 to High(h_input2) do Write(h_input2[i], ' ');
  WriteLn;
  Write(' - Expected output: ');
  for i := 0 to High(cpu_output) do Write(cpu_output[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_input1, input1_size * sizeof(cint)), 'Allocating d_input1');
  CheckHIP(hipMalloc(@d_input2, input2_size * sizeof(cint)), 'Allocating d_input2');
  CheckHIP(hipMalloc(@d_output, output_size * sizeof(cint)), 'Allocating d_output');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_input1, @h_input1[0], input1_size * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_input1 to d_input1');
  CheckHIP(hipMemcpy(d_input2, @h_input2[0], input2_size * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_input2 to d_input2');

  // --- 4. Perform merge on the GPU --- //
  WriteLn(' - Performing merge on GPU...');
  CheckHIP(
    rocprim_merge_int(d_input1, d_input2, d_output, input1_size, input2_size, stream),
    'Performing merge'
  );

  // --- 5. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_output[0], d_output, output_size * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_output to h_output');

  // --- 6. Verification and Cleanup --- //
  Write(' - GPU output: ');
  for i := 0 to output_size - 1 do Write(h_output[i], ' ');
  WriteLn;

  match := true;
  for i := 0 to output_size - 1 do
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
  CheckHIP(hipFree(d_input1), 'Freeing d_input1');
  CheckHIP(hipFree(d_input2), 'Freeing d_input2');
  CheckHIP(hipFree(d_output), 'Freeing d_output');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
