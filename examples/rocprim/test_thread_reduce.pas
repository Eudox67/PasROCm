{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  A supporting program for the rocPRIM examples; it uses the HIP
  runtime directly rather than a rocPRIM primitive.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_thread_reduce.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_thread_reduce
------------------------------------------------------------------------------}

program test_thread_reduce;

{$mode objfpc}{$H+}  // ObjFPC for rocPRIM compatibility

uses
  SysUtils,  // For PInteger = ^LongInt, writeln
  rocprim_types,  // For TBinaryOpInt
  rocprim_thread; // Your unit

const
  NUM_ITEMS: LongInt = 100;

var
  h_in: array[0..99] of LongInt;  // Use LongInt
  i: LongInt;
  cpu_sum: LongInt;
  thread_reduce_sum: LongInt;
  fSumOp: TBinaryOpInt;

// SumOp: Matches TBinaryOpInt exactly (LongInt)
function SumOp(a, b: LongInt): LongInt;
begin
  Result := a + b;
end;

begin
  WriteLn('--- Testing rocprim_thread.pas: thread_reduce ---');

  // Initialize data (1 to 100, sum = 5050)
  WriteLn('Initializing data...');
  cpu_sum := 0;
  for i := 0 to Pred(NUM_ITEMS) do
  begin
    h_in[i] := i + 1;
    cpu_sum := cpu_sum + h_in[i];  // Shorthand
  end;
  WriteLn('Expected CPU sum: ', cpu_sum);

  fSumOp := @SumOp;
  // Execute thread_reduce (pass fSumOp directly—no @)
  WriteLn('Executing thread_reduce...');
  thread_reduce_sum := thread_reduce(PInteger(@h_in[0]), NUM_ITEMS, fSumOp);  

  // Verify
  WriteLn('thread_reduce result: ', thread_reduce_sum);
  if cpu_sum = thread_reduce_sum then
    WriteLn('Test PASSED: Results match!')
  else
    WriteLn('Test FAILED: CPU=', cpu_sum, ' vs. thread_reduce=', thread_reduce_sum);

  WriteLn('--- Test complete ---');
  ReadLn;  // Pause if interactive
end.
