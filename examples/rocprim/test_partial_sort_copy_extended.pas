{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_partial_sort_copy_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_partial_sort_copy_extended.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_partial_sort_copy_extended
------------------------------------------------------------------------------}

program test_partial_sort_copy_extended;

{$mode objfpc}{$H+}

uses
  ctypes,
  hip_bindings,
  rocprim_bindings,  // Assume this declares rocprim_partial_sort_copy_int with temp/size params
  SysUtils,
  Math;

procedure CheckHIP(err: hipError_t; const msg: string);
begin
  if err <> hipSuccess then
  begin
    WriteLn('HIP Error: ', err, ' at ', msg);
    Halt(1);
  end;
end;

// CPU reference sort for prefix only (bubble for small n; use for verification).
procedure BubbleSortPrefix(var arr: array of cint; middle: cint);
var
  i, j, temp: cint;
begin
  for i := 0 to middle do
    for j := i + 1 to middle do
      if arr[i] > arr[j] then
      begin
        temp := arr[i];
        arr[i] := arr[j];
        arr[j] := temp;
      end;
end;

procedure RunTest(const test_name: string; h_input: array of cint; middle: cint);
var
  size: csize_t;
  h_output, cpu_output: array of cint;
  d_input, d_output, d_temp: Pointer;
  temp_size: csize_t;
  i: cint;
  match: Boolean;
  stream: hipStream_t;
begin
  size := length(h_input);
  SetLength(h_output, size);
  SetLength(cpu_output, size);

  WriteLn(#13#10'=== ', test_name, ' ===');
  Write(' - Input: ');
  for i := 0 to size - 1 do Write(h_input[i], ' ');
  WriteLn;
  WriteLn(' - Middle: ', middle, ' (prefix length: ', middle + 1, ')');

  // CPU reference: Copy full, sort only prefix.
  for i := 0 to size - 1 do cpu_output[i] := h_input[i];
  if middle >= 0 then
    BubbleSortPrefix(cpu_output, middle);
  Write(' - Expected: ');
  for i := 0 to size - 1 do Write(cpu_output[i], ' ');
  WriteLn;

  CheckHIP(hipStreamCreate(@stream), 'Stream Create');
  CheckHIP(hipMalloc(@d_input, size * sizeof(cint)), 'Alloc d_input');
  CheckHIP(hipMalloc(@d_output, size * sizeof(cint)), 'Alloc d_output');

  CheckHIP(hipMemcpy(d_input, @h_input[0], size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD input');

  // Query temp size
  temp_size := 0;
  CheckHIP(rocprim_partial_sort_copy_int(nil, @temp_size, Pointer(d_input), Pointer(d_output), middle, size, stream, True), 'Query temp size');
  WriteLn(' - Temp size: ', temp_size, ' bytes');

  d_temp := nil;
  if temp_size > 0 then
    CheckHIP(hipMalloc(@d_temp, temp_size), 'Alloc d_temp');

  // Execute
  WriteLn(' - Executing on GPU...');
  CheckHIP(rocprim_partial_sort_copy_int(d_temp, @temp_size, Pointer(d_input), Pointer(d_output), middle, size, stream, True), 'GPU partial_sort_copy');

  CheckHIP(hipMemcpy(@h_output[0], d_output, size * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH output');

  Write(' - GPU output: ');
  for i := 0 to size - 1 do Write(h_output[i], ' ');
  WriteLn;

  // Compare
  match := true;
  for i := 0 to size - 1 do
    if h_output[i] <> cpu_output[i] then
    begin
      WriteLn('  Mismatch at ', i, ': GPU=', h_output[i], ' Expected=', cpu_output[i]);
      match := false;
    end;

  if match then
    WriteLn('  SUCCESS!')
  else
    WriteLn('  FAILURE!');

  if assigned(d_temp) then hipFree(d_temp);
  hipFree(d_input);
  hipFree(d_output);
  hipStreamDestroy(stream);
end;

var
  input1: array[0..7] of cint = (6, 3, 5, 4, 1, 8, 2, 7);  // Original unsorted
  input2: array[0..7] of cint = (1, 3, 4, 5, 6, 8, 2, 7);  // Already sorted prefix
  input3: array[0..7] of cint = (8, 7, 6, 5, 4, 3, 2, 1);  // Reversed (full sort for middle=7)
  input4: array[0..7] of cint = (1, 2, 3, 4, 5, 6, 7, 8);  // Sorted, test prefix=1 (middle=0, no change)

begin
  WriteLn('Extended rocprim::partial_sort_copy tests...');

  RunTest('Case 1: Unsorted Prefix (Original)', input1, 4);
  RunTest('Case 2: Already Sorted Prefix', input2, 4);
  RunTest('Case 3: Full Array Sort (middle = size-1)', input3, 7);
  RunTest('Case 4: Minimal Prefix (middle=0, no sort needed)', input4, 0);

  WriteLn(#13#10'All tests complete.');
end.
