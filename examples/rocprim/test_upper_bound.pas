program test_upper_bound;

{$mode objfpc}{$H+}

uses
  SysUtils,
  rocprim_thread;  // Your unit with upper_bound

const
  NUM_ITEMS = 100;

var
  h_in: array[0..99] of LongInt;        // Sorted input: 1 to 100
  i: LongInt;
  test_vals: array[0..5] of LongInt;    // Test values (edge cases)
  expected_idx: array[0..5] of LongInt; // Expected returns
  actual_idx: LongInt;
  match_count: LongInt = 0;
  all_good: Boolean;                    // Moved to var (for property check)
  j: LongInt;                           // Moved to var (for nested loops)

begin
  WriteLn('--- Testing rocprim_thread.pas: upper_bound (first > val) ---');

  // Step 1: Initialize sorted input array (1 to 100)
  WriteLn('Initializing sorted array (1 to 100)...');
  for i := 0 to 99 do
    h_in[i] := i + 1;  // h_in = [1, 2, ..., 100]

  // Step 2: Define test cases and expected indices
  // Cases: Below all (0), in-array end (50), after value (51), at end (100), beyond (101), at start (1)
  test_vals[0] := 0;    expected_idx[0] := 0;   // First >0 is index 0 (value 1)
  test_vals[1] := 50;   expected_idx[1] := 50;  // First >50 is index 50 (value 51)
  test_vals[2] := 51;   expected_idx[2] := 51;  // First >51 is index 51 (value 52)
  test_vals[3] := 100;  expected_idx[3] := 100; // First >100 is end 100
  test_vals[4] := 101;  expected_idx[4] := 100; // Beyond end → 100
  test_vals[5] := 1;    expected_idx[5] := 1;  // First >1 is index 1 (value 2)

  // Step 3: Run upper_bound for each case
  WriteLn('Running tests...');
  for i := 0 to 5 do
  begin
    actual_idx := upper_bound(@h_in[0], NUM_ITEMS, test_vals[i]);

    WriteLn(Format('val = %d: returned index = %d (expected %d)', [test_vals[i], actual_idx, expected_idx[i]]));
    
    // Verify property: All before <= val; all at/after > val
    all_good := True;
    for j := 0 to Pred(actual_idx) do
      if h_in[j] > test_vals[i] then
      begin
        WriteLn('  Property FAIL: index ', j, ' = ', h_in[j], ' > ', test_vals[i], ' (should be <=)');
        all_good := False;
        break;
      end;
    for j := actual_idx to 99 do
      if h_in[j] <= test_vals[i] then
      begin
        WriteLn('  Property FAIL: index ', j, ' = ', h_in[j], ' <= ', test_vals[i], ' (should be >)');
        all_good := False;
        break;
      end;
    
    if actual_idx = expected_idx[i] then
      WriteLn('  ✓ PASSED')
    else
    begin
      WriteLn('  ✗ FAILED');
      all_good := False;
    end;
    
    if all_good then
      Inc(match_count);
  end;

  // Final result
  if match_count = 6 then
    WriteLn('Test PASSED: All ', match_count, ' of 6 cases match!')
  else
    WriteLn('Test FAILED: Only ', match_count, ' of 6 cases matched');

  WriteLn('--- upper_bound test complete ---');
end.
