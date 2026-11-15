{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  A supporting program for the rocPRIM examples; it uses the HIP
  runtime directly rather than a rocPRIM primitive.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_numeric_limits.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_numeric_limits
------------------------------------------------------------------------------}

program test_numeric_limits;

{$mode objfpc}{$H+}

uses
  ctypes,
  math, // For IsNan
  rocprim_bindings;

begin
  WriteLn('Testing rocprim::numeric_limits...');

  // Test int limits
  WriteLn('Integer Limits:');
  WriteLn('  min: ', rocprim_numeric_limits_int_min);
  WriteLn('  max: ', rocprim_numeric_limits_int_max);

  // Test float limits
  WriteLn('Float Limits:');
  WriteLn('  epsilon: ', rocprim_numeric_limits_float_epsilon);
  WriteLn('  infinity: ', rocprim_numeric_limits_float_infinity);

  // Test for NaN
  if IsNan(rocprim_numeric_limits_float_nan) then
    WriteLn('  NaN test: SUCCESS')
  else
    WriteLn('  NaN test: FAILURE');

  WriteLn('Test complete.');
end.
