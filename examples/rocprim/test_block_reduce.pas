{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  Demonstrates rocprim_block_reduce_int: allocates device buffers with HIP, calls the
  primitive through rocprim_bindings, and checks the result.

  Build and run from the repository root:
    fpc @pasrocm.cfg -FEbin examples/rocprim/test_block_reduce.pas
    PASROCM_SHIM_DIR=src/rocprim ./bin/test_block_reduce
------------------------------------------------------------------------------}

program test_block_reduce;

{$mode objfpc}{$H+}

uses
  ctypes, hip_error, hip_types, hip_runtime, rocprim_bindings;

const
  NUM_ITEMS = 1024;
  BLOCK_SIZE = 256;
var
  h_in: array[0..NUM_ITEMS-1] of cint;
  h_out: array[0..((NUM_ITEMS + BLOCK_SIZE - 1) div BLOCK_SIZE)-1] of cint;
  d_in, d_out: Pointer;
  i: integer;
  cpu_sum, gpu_sum: cint;
  stream: hipStream_t;
  status: hipError_t;
  num_blocks: integer;

begin
  num_blocks := (NUM_ITEMS + BLOCK_SIZE - 1) div BLOCK_SIZE;

  // Initialize host data
  writeln('Initializing host data...');
  cpu_sum := 0;
  for i := 0 to NUM_ITEMS - 1 do
  begin
    h_in[i] := i + 1;
    cpu_sum := cpu_sum + h_in[i];
  end;

  // Allocate device memory
  writeln('Allocating device memory...');
  status := hipMalloc(d_in, NUM_ITEMS * sizeof(cint));
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMalloc failed: ', status);
    halt(1);
  end;
  status := hipMalloc(d_out, num_blocks * sizeof(cint));
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMalloc failed: ', status);
    hipFree(d_in);
    halt(1);
  end;

  // Copy data from host to device
  writeln('Copying data from host to device...');
  status := hipMemcpy(d_in, @h_in[0], NUM_ITEMS * sizeof(cint), hipMemcpyHostToDevice);
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMemcpy failed: ', status);
    hipFree(d_in);
    hipFree(d_out);
    halt(1);
  end;

  // Create stream
  status := hipStreamCreate(stream);
  if status <> HIP_SUCCESS then
  begin
    writeln('hipStreamCreate failed: ', status);
    hipFree(d_in);
    hipFree(d_out);
    halt(1);
  end;

  // Execute block reduce
  writeln('Executing block reduce...');
  status := rocprim_block_reduce_int(d_out, d_in, NUM_ITEMS, BLOCK_SIZE, stream);
  if status <> HIP_SUCCESS then
  begin
    writeln('rocprim_block_reduce_int failed: ', status);
    hipStreamDestroy(stream);
    hipFree(d_in);
    hipFree(d_out);
    halt(1);
  end;

  status := hipStreamSynchronize(stream);
  if status <> HIP_SUCCESS then
  begin
    writeln('hipStreamSynchronize failed: ', status);
    hipStreamDestroy(stream);
    hipFree(d_in);
    hipFree(d_out);
    halt(1);
  end;

  // Copy results from device to host
  writeln('Copying results from device to host...');
  status := hipMemcpy(@h_out[0], d_out, num_blocks * sizeof(cint), hipMemcpyDeviceToHost);
  if status <> HIP_SUCCESS then
  begin
    writeln('hipMemcpy failed: ', status);
    hipStreamDestroy(stream);
    hipFree(d_in);
    hipFree(d_out);
    halt(1);
  end;

  // Calculate GPU sum
  gpu_sum := 0;
  for i := 0 to num_blocks - 1 do
  begin
    gpu_sum := gpu_sum + h_out[i];
  end;

  // Verify results
  writeln('Verifying results...');
  writeln('CPU sum: ', cpu_sum);
  writeln('GPU sum: ', gpu_sum);

  if cpu_sum = gpu_sum then
    writeln('Test PASSED')
  else
    writeln('Test FAILED');

  // Clean up
  writeln('Cleaning up...');
  hipStreamDestroy(stream);
  hipFree(d_in);
  hipFree(d_out);
end.
