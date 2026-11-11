program test_search_n;

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

procedure RunTest(h_input: array of cint; value: cint; count: csize_t; op: rocprim_binary_op; expected_index: csize_t; const test_name: string);
var
  h_output: csize_t;

  // Device pointers
  d_input: Pointer = nil;
  d_output: Pointer = nil;

  // Other variables
  num_items: csize_t;
  temp_storage_size: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('--------------------------------------------------');
  WriteLn('Running Test: ', test_name);

  // --- 1. Setup --- //
  num_items := Length(h_input);
  WriteLn(' - Number of items: ', num_items);
  WriteLn(' - Searching for ', count, ' consecutive instances of value ', value);

  Write(' - Input: ');
  for i := 0 to High(h_input) do Write(h_input[i], ' ');
  WriteLn;

  WriteLn(' - Expected output (index): ', expected_index);

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // --- 2. Allocate device memory --- //
  CheckHIP(hipMalloc(@d_input, num_items * sizeof(cint)), 'Allocating d_input');
  CheckHIP(hipMalloc(@d_output, sizeof(csize_t)), 'Allocating d_output');

  // --- 3. Copy data from Host to Device --- //
  CheckHIP(hipMemcpy(d_input, @h_input[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_input to d_input');

  // --- 4. Get temporary storage size and allocate --- //
  CheckHIP(
    rocprim_search_n_int(nil, @temp_storage_size, d_input, d_output, num_items, count, value, op, stream, true),
    'Querying search_n size'
  );

  // --- 5. Perform search_n on the GPU --- //
  WriteLn(' - Performing search_n on GPU...');
  CheckHIP(
    rocprim_search_n_int(d_input, @temp_storage_size, d_input, d_output, num_items, count, value, op, stream, true),
    'Performing search_n'
  );

  // --- 6. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_output, d_output, sizeof(csize_t), hipMemcpyDeviceToHost), 'Copying d_output to h_output');

  // --- 7. Verification and Cleanup --- //
  WriteLn(' - GPU output (index): ', h_output);

  match := (h_output = expected_index);

  if match then
    WriteLn('SUCCESS: GPU result matches expected result.')
  else
    WriteLn('FAILURE: GPU result does not match expected result.');

  // Free device memory
  CheckHIP(hipFree(d_input), 'Freeing d_input');
  CheckHIP(hipFree(d_output), 'Freeing d_output');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');
  WriteLn('--------------------------------------------------');
end;

var
  // Test case 1: Match found (equal_to)
  h_input_match: array[0..9] of cint = (0, 1, 3, 3, 3, 3, 3, 4, 5, 6);
  value_match: cint = 3;
  count_match: csize_t = 5;
  expected_index_match: csize_t = 2;

  // Test case 2: No match (equal_to)
  h_input_no_match: array[0..9] of cint = (1, 1, 1, 1, 0, 1, 1, 1, 0, 1);
  value_no_match: cint = 1;
  count_no_match: csize_t = 5;
  expected_index_no_match: csize_t = 10; // Expect haystack_size

  // Test case 3: Match found (greater_than)
  h_input_greater: array[0..9] of cint = (0, 1, 5, 5, 5, 5, 5, 4, 5, 6);
  value_greater: cint = 4;
  count_greater: csize_t = 5;
  expected_index_greater: csize_t = 2;

begin
  WriteLn('Starting rocprim::search_n test...');

  RunTest(h_input_match, value_match, count_match, rocprim_binary_op_equal_to, expected_index_match, 'Match Found (equal_to)');
  RunTest(h_input_no_match, value_no_match, count_no_match, rocprim_binary_op_equal_to, expected_index_no_match, 'No Match Found (equal_to)');
  RunTest(h_input_greater, value_greater, count_greater, rocprim_binary_op_greater_than, expected_index_greater, 'Match Found (greater_than)');

end.
