{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_reduce_int128: a device-wide reduction over 128-bit
  integers. The host side uses the operators on int128_t in rocprim_types, so
  the reference sum is ordinary Pascal arithmetic rather than a big-integer
  library.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_reduce_int128.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_reduce_int128
------------------------------------------------------------------------------}

program test_reduce_int128;

{$mode objfpc}{$H+}

uses
  ctypes, hip_bindings, rocprim_types, rocprim_bindings;

procedure CheckHIP(err: hipError_t; const msg: string);
begin
  if err <> hipSuccess then
  begin
    WriteLn('HIP Error: ', err, ' at ', msg);
    Halt(1);
  end;
end;

const
  N = 10;

var
  h_input: array[0..N-1] of int128_t;
  h_output, expected_sum, zero, item: int128_t;
  d_input, d_output: Pointer;
  num_items: csize_t;
  i: Integer;
  stream: hipStream_t;
begin
  WriteLn('Starting rocprim::reduce test for int128_t...');

  num_items := N;
  zero := Int64(0);
  expected_sum := zero;

  { Values above 2^64, so the reduction genuinely exercises both halves. }
  for i := 0 to N - 1 do
  begin
    item.low := QWord(i + 1) * 1000000000000000000;
    item.high := 0;
    h_input[i] := item;
    expected_sum := expected_sum + item;
  end;

  WriteLn(' - Number of items: ', num_items);
  WriteLn(' - First input    : ', h_input[0].ToStr);
  WriteLn(' - Last input     : ', h_input[N-1].ToStr);
  WriteLn(' - CPU reference  : ', expected_sum.ToStr);

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');
  CheckHIP(hipMalloc(@d_input, num_items * SizeOf(int128_t)), 'Allocating d_input');
  CheckHIP(hipMalloc(@d_output, SizeOf(int128_t)), 'Allocating d_output');
  CheckHIP(hipMemcpy(d_input, @h_input[0], num_items * SizeOf(int128_t),
    hipMemcpyHostToDevice), 'Copying h_input to d_input');

  WriteLn(' - Performing reduce on GPU...');
  CheckHIP(rocprim_reduce_int128(PInt128(d_output), PInt128(d_input),
    num_items, zero, stream), 'Performing reduce');

  CheckHIP(hipMemcpy(@h_output, d_output, SizeOf(int128_t),
    hipMemcpyDeviceToHost), 'Copying d_output to h_output');

  WriteLn(' - GPU result     : ', h_output.ToStr);
  WriteLn(' - GPU result hex : ', h_output.ToHex);

  if h_output = expected_sum then
    WriteLn('SUCCESS: GPU result matches the CPU reference.')
  else
  begin
    WriteLn('FAILURE: GPU result does not match the CPU reference.');
    Halt(1);
  end;

  CheckHIP(hipFree(d_input), 'Freeing d_input');
  CheckHIP(hipFree(d_output), 'Freeing d_output');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');
end.
