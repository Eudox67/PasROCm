program test_thread_scan_exc;

{$mode objfpc}{$H+}

uses
  SysUtils,
  rocprim_types,
  rocprim_thread;  // Your unit with thread_scan_exclusive and TSum

const
  NUM_ITEMS = 100;

var
  h_in: array[0..99] of LongInt;        // Input: 1 to 100
  h_out: array[0..99] of LongInt;       // Output from scan
  expected: array[0..99] of LongInt;    // Expected exclusive prefixes (manual calc)
  i, j: LongInt;
  op_sum: TBinaryOpInt;                 // Procedural var for + operator
  match_count: LongInt;
  
// SumOp: Matches TBinaryOpInt exactly (LongInt)
function SumOp(a, b: LongInt): LongInt;
begin
  Result := a + b;
end;


begin
  WriteLn('--- Testing rocprim_thread.pas: thread_scan_exclusive ---');

  // Step 1: Initialize input (1 to 100)
  WriteLn('Initializing input data (1 to 100)...');
  for i := 0 to 99 do
    h_in[i] := i + 1;  // h_in = [1, 2, ..., 100]

  // Step 2: Manually compute expected exclusive scan (prefix sum up to i-1)
  // expected[0] = 0 (prefix=0), expected[1] = 1, expected[2] = 1+2=3, ..., expected[99] = sum(1..99)=4950
  WriteLn('Computing expected exclusive scan...');
  expected[0] := 0;  // Starts with prefix=0
  for i := 1 to 99 do
  begin
    expected[i] := expected[i - 1] + h_in[i - 1];  // Cumulative sum up to i-1
  end;
  // Verify last: sum(1..99) = 99*100/2 = 4950
  WriteLn('Expected exclusive[0]=0, ... exclusive[99]=', expected[99]);  // 4950

  // Step 3: Assign sum operator
  op_sum := @SumOp;  // Pointer to (binary +)

  // Step 4: Execute thread_scan_exclusive (prefix=0)
  WriteLn('Executing thread_scan_exclusive (prefix=0)...');
  thread_scan_exclusive(PInteger(@h_in[0]), PInteger(@h_out[0]), NUM_ITEMS, op_sum, 0);  // Prefix=0

  // Step 5: Verify results (compare h_out to expected)
  WriteLn('Verifying results...');
  WriteLn('Sample checks:');
  WriteLn('  h_out[0] = ', h_out[0], ' (expected 0)');
  WriteLn('  h_out[1] = ', h_out[1], ' (expected 1)');
  WriteLn('  h_out[2] = ', h_out[2], ' (expected 3)');
  WriteLn('  h_out[3] = ', h_out[3], ' (expected 6)');

  // Full verification
  match_count := 0;
  for i := 0 to 99 do
  begin
    if h_out[i] = expected[i] then
      Inc(match_count)
    else
    begin
      WriteLn('  MISMATCH at index ', i, ': h_out=', h_out[i], ' expected=', expected[i]);
      break;
    end;
  end;

  if match_count = NUM_ITEMS then
    WriteLn('Test PASSED: All ', NUM_ITEMS, ' values match!')
  else
    WriteLn('Test FAILED: Only ', match_count, '/', NUM_ITEMS, ' matched');

  WriteLn('--- Exclusive scan test complete ---');
end.
