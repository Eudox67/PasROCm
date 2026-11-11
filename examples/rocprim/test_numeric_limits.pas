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
