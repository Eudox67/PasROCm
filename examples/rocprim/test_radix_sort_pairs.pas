program test_radix_sort_pairs;

{$mode objfpc}{$H+}

uses
  ctypes,
  hip_runtime,
  hip_bindings,
  rocprim_bindings,  // Assume declares rocprim_radix_sort_pairs_int
  SysUtils,
  Math;

const
  hipSuccess = 0;

procedure CheckHIP(err: hipError_t; const msg: string);
begin
  if err <> hipSuccess then
  begin
    WriteLn('HIP Error: ', err, ' at ', msg);
    Halt(1);
  end;
end;

// CPU reference sort for pairs (bubble sort by key, stable).
procedure CPUReferenceSort(var keys, values: array of cint; n: cint);
var
  i, j, temp_key, temp_val: cint;
begin
  for i := 0 to n - 2 do
    for j := i + 1 to n - 1 do
      if keys[i] > keys[j] then
      begin
        // Swap key
        temp_key := keys[i]; keys[i] := keys[j]; keys[j] := temp_key;
        // Swap value (stable)
        temp_val := values[i]; values[i] := values[j]; values[j] := temp_val;
      end;
end;

var
  stream: hipStream_t;  // Shared stream for both tests

// --- Small n Test Procedure ---
procedure TestSmallN;
var
  h_keys_in: array[0..4] of cint = (3, 1, 4, 1, 5);  // Unsorted keys
  h_values_in: array[0..4] of cint = (30, 10, 40, 20, 50);  // Values
  h_keys_out, h_values_out: array[0..4] of cint;
  cpu_keys_out, cpu_values_out: array[0..4] of cint;
  d_keys_in, d_keys_out, d_values_in, d_values_out, d_temp: Pointer;
  num_items, temp_size: csize_t;
  i: cint;
  match_keys, match_values: Boolean;

begin
  WriteLn('Testing rocprim::radix_sort_pairs (Small n=5)...');
  num_items := 5;

  // Setup CPU reference: Copy input, sort.
  for i := 0 to 4 do
  begin
    h_keys_out[i] := h_keys_in[i];
    h_values_out[i] := h_values_in[i];
  end;
  CPUReferenceSort(h_keys_out, h_values_out, num_items);
  // Copy to cpu_* for comparison.
  for i := 0 to 4 do
  begin
    cpu_keys_out[i] := h_keys_out[i];
    cpu_values_out[i] := h_values_out[i];
  end;

  WriteLn(' - Input Keys: 3 1 4 1 5');
  WriteLn(' - Input Values: 30 10 40 20 50');
  WriteLn(' - Expected Keys: ', cpu_keys_out[0], ' ', cpu_keys_out[1], ' ', cpu_keys_out[2], ' ', cpu_keys_out[3], ' ', cpu_keys_out[4]);
  WriteLn(' - Expected Values: ', cpu_values_out[0], ' ', cpu_values_out[1], ' ', cpu_values_out[2], ' ', cpu_values_out[3], ' ', cpu_values_out[4]);

  CheckHIP(hipMalloc(@d_keys_in, num_items * sizeof(cint)), 'Alloc d_keys_in');
  CheckHIP(hipMalloc(@d_values_in, num_items * sizeof(cint)), 'Alloc d_values_in');
  CheckHIP(hipMalloc(@d_keys_out, num_items * sizeof(cint)), 'Alloc d_keys_out');
  CheckHIP(hipMalloc(@d_values_out, num_items * sizeof(cint)), 'Alloc d_values_out');

  CheckHIP(hipMemcpy(d_keys_in, @h_keys_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'HtoD keys_in');
  CheckHIP(hipMemcpy(d_values_in, @h_values_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'HtoD values_in');

  // Initialize outputs to sentinel (-1) for debug.
  for i := 0 to 4 do
  begin
    h_keys_out[i] := -1;
    h_values_out[i] := -1;
  end;
  CheckHIP(hipMemcpy(d_keys_out, @h_keys_out[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Init d_keys_out');
  CheckHIP(hipMemcpy(d_values_out, @h_values_out[0], num_items * sizeof(cint), hipMemcpyHostToDevice), 'Init d_values_out');

  // Query temp
  temp_size := 0;
  CheckHIP(rocprim_radix_sort_pairs_int(nil, @temp_size, Pointer(d_keys_in), Pointer(d_keys_out), Pointer(d_values_in), Pointer(d_values_out), num_items, stream), 'Query temp size');
  WriteLn(' - Temp size: ', temp_size, ' bytes');

  d_temp := nil;
  if temp_size > 0 then
  begin
    CheckHIP(hipMalloc(@d_temp, temp_size), 'Alloc d_temp');
    CheckHIP(hipMemset(d_temp, 0, temp_size), 'Memset temp');
  end;

  // Execute
  WriteLn(' - Executing on GPU...');
  CheckHIP(rocprim_radix_sort_pairs_int(d_temp, @temp_size, Pointer(d_keys_in), Pointer(d_keys_out), Pointer(d_values_in), Pointer(d_values_out), num_items, stream), 'GPU radix_sort_pairs');

  // Sync
  CheckHIP(hipStreamSynchronize(stream), 'Stream Synchronize after execute');

  CheckHIP(hipMemcpy(@h_keys_out[0], d_keys_out, num_items * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH keys_out');
  CheckHIP(hipMemcpy(@h_values_out[0], d_values_out, num_items * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH values_out');

  WriteLn(' - GPU Keys: ', h_keys_out[0], ' ', h_keys_out[1], ' ', h_keys_out[2], ' ', h_keys_out[3], ' ', h_keys_out[4]);
  WriteLn(' - GPU Values: ', h_values_out[0], ' ', h_values_out[1], ' ', h_values_out[2], ' ', h_values_out[3], ' ', h_values_out[4]);

  // Compare
  match_keys := True;
  match_values := True;
  for i := 0 to 4 do
  begin
    if h_keys_out[i] <> cpu_keys_out[i] then
    begin
      WriteLn('  Keys Mismatch at ', i, ': GPU=', h_keys_out[i], ' Expected=', cpu_keys_out[i]);
      match_keys := False;
    end;
    if h_values_out[i] <> cpu_values_out[i] then
    begin
      WriteLn('  Values Mismatch at ', i, ': GPU=', h_values_out[i], ' Expected=', cpu_values_out[i]);
      match_values := False;
    end;
  end;

  if match_keys and match_values then
    WriteLn('SUCCESS: Small n test passed!')
  else
    WriteLn('FAILURE: Small n test failed.');

  if assigned(d_temp) then hipFree(d_temp);
  hipFree(d_keys_in); hipFree(d_values_in); hipFree(d_keys_out); hipFree(d_values_out);

  WriteLn('Small n test complete.');
end;

// --- Large n Test Procedure (Verify rocPRIM Path) ---
procedure TestLargeN(stream: hipStream_t);
var
  large_n: csize_t = 10;  // Small for quick; change to 1000 for performance
  h_keys_large: array of cint;
  h_values_large: array of cint;
  h_keys_out_large, h_values_out_large: array of cint;
  cpu_keys_large, cpu_values_large: array of cint;
  d_keys_large_in, d_keys_large_out, d_values_large_in, d_values_large_out, d_temp_large: Pointer;
  temp_size_large: csize_t;
  i: cint;
  match_keys, match_values: Boolean;

begin
  WriteLn(#13#10'Testing large n (rocPRIM path)...');
  SetLength(h_keys_large, large_n);
  SetLength(h_values_large, large_n);
  SetLength(h_keys_out_large, large_n);
  SetLength(h_values_out_large, large_n);
  SetLength(cpu_keys_large, large_n);
  SetLength(cpu_values_large, large_n);

  // Fill reversed keys, sequential values for stable sort.
  for i := 0 to large_n - 1 do
  begin
    h_keys_large[i] := large_n - i;  // 10 9 ... 1 for n=10
    h_values_large[i] := i * 10 + 10;  // 10, 20, ... 100
  end;

  // CPU reference.
  for i := 0 to large_n - 1 do
  begin
    h_keys_out_large[i] := h_keys_large[i];
    h_values_out_large[i] := h_values_large[i];
  end;
  CPUReferenceSort(h_keys_out_large, h_values_out_large, large_n);
  for i := 0 to large_n - 1 do
  begin
    cpu_keys_large[i] := h_keys_out_large[i];
    cpu_values_large[i] := h_values_out_large[i];
  end;

  WriteLn(' - Large n: ', large_n);
  WriteLn(' - Input Keys (first 5): 10 9 8 7 6 ...');
  WriteLn(' - Expected Keys (first 5): ', cpu_keys_large[0], ' ', cpu_keys_large[1], ' ', cpu_keys_large[2], ' ', cpu_keys_large[3], ' ', cpu_keys_large[4]);

  CheckHIP(hipMalloc(@d_keys_large_in, large_n * sizeof(cint)), 'Alloc d_keys_large_in');
  CheckHIP(hipMalloc(@d_values_large_in, large_n * sizeof(cint)), 'Alloc d_values_large_in');
  CheckHIP(hipMalloc(@d_keys_large_out, large_n * sizeof(cint)), 'Alloc d_keys_large_out');
  CheckHIP(hipMalloc(@d_values_large_out, large_n * sizeof(cint)), 'Alloc d_values_large_out');

  CheckHIP(hipMemcpy(d_keys_large_in, @h_keys_large[0], large_n * sizeof(cint), hipMemcpyHostToDevice), 'HtoD large keys_in');
  CheckHIP(hipMemcpy(d_values_large_in, @h_values_large[0], large_n * sizeof(cint), hipMemcpyHostToDevice), 'HtoD large values_in');

  // Initialize outputs to sentinel (-1).
  for i := 0 to large_n - 1 do
  begin
    h_keys_out_large[i] := -1;
    h_values_out_large[i] := -1;
  end;
  CheckHIP(hipMemcpy(d_keys_large_out, @h_keys_out_large[0], large_n * sizeof(cint), hipMemcpyHostToDevice), 'Init large keys_out');
  CheckHIP(hipMemcpy(d_values_large_out, @h_values_out_large[0], large_n * sizeof(cint), hipMemcpyHostToDevice), 'Init large values_out');

  // Query/alloc temp.
  temp_size_large := 0;
  CheckHIP(rocprim_radix_sort_pairs_int(nil, @temp_size_large, Pointer(d_keys_large_in), Pointer(d_keys_large_out), Pointer(d_values_large_in), Pointer(d_values_large_out), large_n, stream), 'Query large temp');
  WriteLn(' - Large Temp size: ', temp_size_large, ' bytes');

  d_temp_large := nil;
  if temp_size_large > 0 then
  begin
    CheckHIP(hipMalloc(@d_temp_large, temp_size_large), 'Alloc d_temp_large');
    CheckHIP(hipMemset(d_temp_large, 0, temp_size_large), 'Memset large temp');
  end;

  // Execute.
  WriteLn(' - Executing on GPU...');
  CheckHIP(rocprim_radix_sort_pairs_int(d_temp_large, @temp_size_large, Pointer(d_keys_large_in), Pointer(d_keys_large_out), Pointer(d_values_large_in), Pointer(d_values_large_out), large_n, stream), 'GPU large sort');

  // Sync.
  CheckHIP(hipStreamSynchronize(stream), 'Sync large');

  CheckHIP(hipMemcpy(@h_keys_out_large[0], d_keys_large_out, large_n * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH large keys');
  CheckHIP(hipMemcpy(@h_values_out_large[0], d_values_large_out, large_n * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH large values');

  // Check (print first 5 for large n=10).
  WriteLn(' - GPU Large Keys (first 5): ', h_keys_out_large[0], ' ', h_keys_out_large[1], ' ', h_keys_out_large[2], ' ', h_keys_out_large[3], ' ', h_keys_out_large[4]);
  match_keys := True;
  match_values := True;
  for i := 0 to large_n - 1 do
  begin
    if h_keys_out_large[i] <> cpu_keys_large[i] then match_keys := False;
    if h_values_out_large[i] <> cpu_values_large[i] then match_values := False;
  end;

  if match_keys and match_values then
    WriteLn('SUCCESS: Large n test passed!')
  else
    WriteLn('FAILURE: Large n test failed.');

  if assigned(d_temp_large) then hipFree(d_temp_large);
  hipFree(d_keys_large_in); hipFree(d_values_large_in); hipFree(d_keys_large_out); hipFree(d_values_large_out);

  WriteLn('Large n test complete.');
end;

// --- Main --- 
var
  temp_temp: csize_t;  // Dummy for init.

begin
  CheckHIP(hipStreamCreate(@stream), 'Global Stream Create');

  // Small n test.
  TestSmallN;

  // Large n test.
  TestLargeN(stream);

  CheckHIP(hipStreamDestroy(stream), 'Stream Destroy');

  WriteLn('All tests complete.');
end.
