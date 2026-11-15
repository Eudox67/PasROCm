{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  A supporting program for the rocPRIM examples; it uses the HIP
  runtime directly rather than a rocPRIM primitive.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_thread_scan_inc.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_thread_scan_inc
------------------------------------------------------------------------------}

program test_thread_scan_inc;

{$mode objfpc}{$H+}

uses
  SysUtils,
  rocprim_types,
  rocprim_thread;  // Your unit with thread_scan_inclusive and TSum

const
  NUM_ITEMS = 100;

var
  h_in: array[0..99] of LongInt;        // Input: 1 to 100
  h_out: array[0..99] of LongInt;       // Output from scan
  expected: array[0..99] of LongInt;    // Expected inclusive prefixes
  i, j: LongInt;
  op_sum: TBinaryOpInt;                 // Procedural var for + operator
  match_count: LongInt;
  
// SumOp: Matches TBinaryOpInt exactly (LongInt)
function SumOp(a, b: LongInt): LongInt;
begin
  Result := a + b;
end;


begin
  WriteLn('--- Testing rocprim_thread.pas: thread_scan_inclusive ---');

  // Step 1: Initialize input (1 to 100)
  WriteLn('Initializing input data (1 to 100)...');
  for i := 0 to 99 do
    h_in[i] := i + 1;  // h_in = [1, 2, ..., 100]

  // Step 2: Manually compute expected inclusive scan (sum up to i, including input[i])
  // expected[0] = 1, expected[1] = 1+2=3, expected[2] = 1+2+3=6, ..., expected[99] = sum(1..100)=5050
  WriteLn('Computing expected inclusive scan...');
  expected[0] := h_in[0];  // Start with first element
  for i := 1 to 99 do
  begin
    expected[i] := expected[i - 1] + h_in[i];  // Cumulative sum including i
  end;
  // Verify last: sum(1..100) = 100*101/2 = 5050
  WriteLn('Expected inclusive[0]=1, ... inclusive[99]=', expected[99]);  // 5050

  // Step 3: Assign sum operator
  op_sum := @SumOp;  // Pointer to (binary +)

  // Step 4: Execute thread_scan_inclusive (no prefix—starts with input[0])
  WriteLn('Executing thread_scan_inclusive...');
  thread_scan_inclusive(PInteger(@h_in[0]), PInteger(@h_out[0]), NUM_ITEMS, op_sum);

  // Step 5: Verify results
  WriteLn('Verifying results...');
  WriteLn('Sample checks:');
  WriteLn('  h_out[0] = ', h_out[0], ' (expected 1)' );
  WriteLn('  h_out[1] = ', h_out[1], ' (expected 3)' );
  WriteLn('  h_out[2] = ', h_out[2], ' (expected 6)' );
  WriteLn('  h_out[3] = ', h_out[3], ' (expected 10)');

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

  WriteLn('--- Inclusive scan test complete ---');
end.
