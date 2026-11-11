program test_binary_search;

{$mode objfpc}{$H+}

uses
  ctypes,  // For cint, csize_t
  hip_bindings,  // hipMalloc, hipMemcpy, etc. (assume your wrapper)
  rocprim_bindings,  // rocprim_binary_search_int external
  SysUtils,  // PtrToStr, HexStr
  Math;  // For Min/Max if needed

// Helper to check HIP errors
procedure CheckHIP(err: hipError_t; const msg: string);
begin
  if err <> hipSuccess then
  begin
    WriteLn('HIP Error: ', err, ' at ', msg);
    Halt(1);
  end;
end;

var
  // Host arrays
  h_haystack: array[0..9] of cint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9);  // Sorted
  h_needles: array[0..7] of cint = (0, 2, 12, 4, 14, 6, 8, 10);  // Mixed
  h_output: array[0..7] of csize_t;  // GPU result (0/1 as size_t)
  cpu_output: array[0..7] of csize_t;  // CPU reference

  // Device pointers
  d_haystack: Pointer = nil;
  d_needles: Pointer = nil;
  d_output: Pointer = nil;
  d_temp_storage: Pointer = nil;

  // Sizes
  haystack_size, needles_size, temp_storage_size: csize_t;
  i, j: cint;  // Loop vars

  // Stream
  stream: hipStream_t;

  // Verification
  match: Boolean;

begin
  WriteLn('Starting rocprim::binary_search test on RDNA3 (gfx1100)...');

  haystack_size := Length(h_haystack);
  needles_size := Length(h_needles);

  WriteLn(' - Haystack size: ', haystack_size);
  WriteLn(' - Needles size: ', needles_size);
  
  Write(' - Haystack: ');
  for i := 0 to haystack_size - 1 do Write(h_haystack[i], ' ');
  WriteLn;
  Write(' - Needles: ');
  for i := 0 to needles_size - 1 do Write(h_needles[i], ' ');
  WriteLn;

  // CPU reference (brute force)
  for i := 0 to needles_size - 1 do
  begin
    cpu_output[i] := 0;  // Default not found
    for j := 0 to haystack_size - 1 do
    begin
      if h_needles[i] = h_haystack[j] then
      begin
        cpu_output[i] := 1;  // Found
        break;
      end;
    end;
  end;

  Write(' - Expected (CPU): ');
  for i := 0 to needles_size - 1 do Write(cpu_output[i], ' ');
  WriteLn;

  // Create stream
  CheckHIP(hipStreamCreate( @stream), 'Stream Create');

  // Alloc device memory
  CheckHIP(hipMalloc( @d_haystack, haystack_size * sizeof(cint)), 'Alloc d_haystack');
  CheckHIP(hipMalloc( @d_needles, needles_size * sizeof(cint)), 'Alloc d_needles');
  CheckHIP(hipMalloc( @d_output, needles_size * sizeof(csize_t)), 'Alloc d_output');

  // Copy host to device
  CheckHIP(hipMemcpy(d_haystack, @h_haystack[0], haystack_size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD haystack');
  CheckHIP(hipMemcpy(d_needles, @h_needles[0], needles_size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD needles');

  // Get temp storage size
  temp_storage_size := 0;
  CheckHIP(
    rocprim_binary_search_int(nil, @temp_storage_size, Pointer(d_haystack), Pointer(d_needles), Pointer(d_output), haystack_size, needles_size, stream, True),
    'Get temp storage size'
  );
  WriteLn(' - Temp storage size: ', temp_storage_size, ' bytes');

  // Alloc temp if needed
  if temp_storage_size > 0 then
    CheckHIP(hipMalloc( @d_temp_storage, temp_storage_size), 'Alloc d_temp_storage')
  else
    d_temp_storage := nil;

  // Execute binary_search (via transform in shim)
  WriteLn(' - Executing on GPU (RDNA3)...');
  CheckHIP(
    rocprim_binary_search_int(Pointer(d_temp_storage), @temp_storage_size, Pointer(d_haystack), Pointer(d_needles), Pointer(d_output), haystack_size, needles_size, stream, True),
    'GPU binary_search'
  );

  // Copy GPU result to host
  CheckHIP(hipMemcpy( @h_output[0], d_output, needles_size * sizeof(csize_t), hipMemcpyDeviceToHost), 'DtoH output');

  // Verify
  Write(' - GPU output: ');
  for i := 0 to needles_size - 1 do Write(h_output[i], ' ');
  WriteLn;

  match := True;
  for i := 0 to needles_size - 1 do
  begin
    if h_output[i] <> cpu_output[i] then
    begin
      WriteLn('  Mismatch at index ', i, ': GPU=', h_output[i], ', Expected=', cpu_output[i]);
      match := False;
    end;
  end;

  if match then
    WriteLn('SUCCESS: GPU binary_search matches CPU reference!')
  else
    WriteLn('FAILURE: GPU binary_search incorrect.');

  // Cleanup
  if assigned(d_temp_storage) then hipFree(d_temp_storage);
  hipFree(d_haystack);
  hipFree(d_needles);
  hipFree(d_output);
  hipStreamDestroy(stream);

  WriteLn('Test complete.');
end.
$', HexStr(NativeUInt(d_temp_storage), SizeOf(NativeUInt) * 2));
  CheckHIP(
    rocprim_binary_search_int(d_temp_storage, @temp_storage_size, d_haystack, d_needles, d_output, haystack_size, needles_size, stream, True),
    'Performing binary_search'
  );

  // --- 6. Copy result from Device to Host --- //
  CheckHIP(hipMemcpy(@h_output[0], d_output, needles_size * sizeof(cbool), hipMemcpyDeviceToHost), 'Copying d_output to h_output');

  // --- 7. Verification and Cleanup --- //
  Write(' - GPU output: ');
  for i := 0 to needles_size - 1 do
  begin
    Write(h_output[i], ' ');
  end;
  WriteLn;

  match := true;
  for i := 0 to needles_size - 1 do
  begin
    if h_output[i] <> cpu_output[i] then
    begin
      match := false;
      break;
    end;
  end;

  if match then
    WriteLn('SUCCESS: GPU result matches CPU result.')
  else
    WriteLn('FAILURE: GPU result does not match CPU result.');

  // Free device memory
  CheckHIP(hipFree(d_haystack), 'Freeing d_haystack');
  CheckHIP(hipFree(d_needles), 'Freeing d_needles');
  CheckHIP(hipFree(d_output), 'Freeing d_output');
  CheckHIP(hipFree(d_temp_storage), 'Freeing d_temp_storage');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');

end.
