program test_run_length_encode_non_trivial_runs;

{$mode objfpc}{$H+}

uses
  ctypes,
  hip_bindings,
  rocprim_bindings,
  SysUtils;

procedure CheckHIP(err: hipError_t; const msg: string);
begin
  if err <> hipSuccess then
  begin
    WriteLn('HIP Error: ', err, ' at ', msg);
    Halt(1);
  end;
end;

var
  h_input: array[0..10] of cint = (1, 1, 1, 2, 3, 3, 4, 5, 5, 5, 5);
  h_offsets_output: array[0..10] of cint;
  h_counts_output: array[0..10] of cint;
  h_runs_count: cint;

  cpu_offsets_output: array[0..2] of cint = (0, 4, 7);
  cpu_counts_output: array[0..2] of cint = (3, 2, 4);
  cpu_runs_count: cint = 3;

  d_input: Pointer = nil;
  d_offsets_output: Pointer = nil;
  d_counts_output: Pointer = nil;
  d_runs_count_output: Pointer = nil;
  d_temp_storage: Pointer = nil;

  size, temp_storage_size: csize_t;
  i: Integer;
  stream: hipStream_t;
  match: Boolean;

begin
  WriteLn('Starting rocprim::run_length_encode_non_trivial_runs test...');

  size := Length(h_input);
  WriteLn(' - Input size: ', size);

  Write(' - Input: ');
  for i := 0 to size - 1 do Write(h_input[i], ' ');
  WriteLn;

  Write(' - Expected offsets: ');
  for i := 0 to cpu_runs_count - 1 do Write(cpu_offsets_output[i], ' ');
  WriteLn;

  Write(' - Expected counts: ');
  for i := 0 to cpu_runs_count - 1 do Write(cpu_counts_output[i], ' ');
  WriteLn;

  WriteLn(' - Expected runs count: ', cpu_runs_count);

  CheckHIP(hipStreamCreate(@stream), 'Stream Create');

  CheckHIP(hipMalloc(@d_input, size * sizeof(cint)), 'Alloc d_input');
  // Allocate worst-case output size
  CheckHIP(hipMalloc(@d_offsets_output, size * sizeof(cint)), 'Alloc d_offsets_output');
  CheckHIP(hipMalloc(@d_counts_output, size * sizeof(cint)), 'Alloc d_counts_output');
  CheckHIP(hipMalloc(@d_runs_count_output, sizeof(cint)), 'Alloc d_runs_count_output');

  CheckHIP(hipMemcpy(d_input, @h_input[0], size * sizeof(cint), hipMemcpyHostToDevice), 'HtoD input');

  temp_storage_size := 0;
  CheckHIP(
    rocprim_run_length_encode_non_trivial_runs_int(nil, @temp_storage_size, d_input, d_offsets_output, d_counts_output, d_runs_count_output, size, stream, True),
    'Get temp storage size'
  );
  WriteLn(' - Temp storage size: ', temp_storage_size, ' bytes');

  if temp_storage_size > 0 then
    CheckHIP(hipMalloc(@d_temp_storage, temp_storage_size), 'Alloc d_temp_storage')
  else
    d_temp_storage := nil;

  WriteLn(' - Executing on GPU...');
  CheckHIP(
    rocprim_run_length_encode_non_trivial_runs_int(d_temp_storage, @temp_storage_size, d_input, d_offsets_output, d_counts_output, d_runs_count_output, size, stream, True),
    'GPU run_length_encode_non_trivial_runs'
  );

  CheckHIP(hipMemcpy(@h_runs_count, d_runs_count_output, sizeof(cint), hipMemcpyDeviceToHost), 'DtoH runs_count');
  if h_runs_count > 0 then
  begin
    CheckHIP(hipMemcpy(@h_offsets_output[0], d_offsets_output, h_runs_count * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH offsets_output');
    CheckHIP(hipMemcpy(@h_counts_output[0], d_counts_output, h_runs_count * sizeof(cint), hipMemcpyDeviceToHost), 'DtoH counts_output');
  end;

  WriteLn(' - GPU runs count: ', h_runs_count);
  Write(' - GPU offsets output: ');
  for i := 0 to h_runs_count - 1 do Write(h_offsets_output[i], ' ');
  WriteLn;
  Write(' - GPU counts output: ');
  for i := 0 to h_runs_count - 1 do Write(h_counts_output[i], ' ');
  WriteLn;

  match := (h_runs_count = cpu_runs_count);
  if match then
  begin
    for i := 0 to cpu_runs_count - 1 do
    begin
      if (h_offsets_output[i] <> cpu_offsets_output[i]) or (h_counts_output[i] <> cpu_counts_output[i]) then
      begin
        match := False;
        break;
      end;
    end;
  end;

  if match then
    WriteLn('SUCCESS: GPU result matches CPU result!')
  else
    WriteLn('FAILURE: GPU result does not match CPU result.');

  if assigned(d_temp_storage) then hipFree(d_temp_storage);
  hipFree(d_input);
  hipFree(d_offsets_output);
  hipFree(d_counts_output);
  hipFree(d_runs_count_output);
  hipStreamDestroy(stream);

  WriteLn('Test complete.');
end.
