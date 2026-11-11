program test_simple_kernel;

{$mode objfpc}{$H+}

uses
  ctypes,
  hip_bindings;

// Helper procedure to check HIP API call results
procedure CheckHIP(err: hipError_t; const msg: string);
begin
  if err <> hipSuccess then
  begin
    WriteLn('HIP Error: ', err, ' at ', msg);
    Halt(1);
  end;
end;

// --- Simple Kernel --- //
const
  librocprimshim = 'rocprim_shim';

function simple_kernel_test(output: PCint; value: cint; stream: hipStream_t): hipError_t; cdecl; external librocprimshim;

var
  // Host data
  h_output: array[0..0] of cint;
  cpu_output: array[0..0] of cint;

  // Device pointers
  d_output: Pointer = nil;

  // Other variables
  num_items: csize_t;
  test_value: cint;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting simple kernel test...');

  // --- 1. Setup --- //
  num_items := 1;
  test_value := 42;
  cpu_output[0] := test_value;

  WriteLn(' - Expected output: ', cpu_output[0]);

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_output, num_items * sizeof(cint)), 'Allocating d_output');

  // --- 3. Perform kernel launch on the GPU --- //
  WriteLn(' - Performing kernel launch on GPU...');
  CheckHIP(
    simple_kernel_test(d_output, test_value, stream),
    'Performing simple kernel test'
  );

  // --- 4. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_output[0], d_output, num_items * sizeof(cint), hipMemcpyDeviceToHost), 'Copying d_output to h_output');

  // --- 5. Verification and Cleanup --- //
  WriteLn(' - GPU output: ', h_output[0]);

  match := (h_output[0] = cpu_output[0]);

  if match then
    WriteLn('SUCCESS: GPU result matches CPU result.')
  else
    WriteLn('FAILURE: GPU result does not match CPU result.');

  // Free device memory
  CheckHIP(hipFree(d_output), 'Freeing d_output');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
