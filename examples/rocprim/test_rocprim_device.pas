unit test_rocprim_device;

{$mode objfpc}{$H+}

interface

uses
  TestFramework;

procedure RegisterTests;

implementation

uses
  SysUtils, ctypes, hip_bindings, rocprim_bindings;

// Helper procedure to check HIP API call results
procedure CheckHIP(err: hipError_t; const msg: string);
begin
  if err <> hipSuccess then
  begin
    WriteLn('HIP Error: ', err, ' at ', msg);
    Halt(1);
  end;
end;

type
  TDeviceBinarySearchTests = class(TTestCase)
  published
    procedure TestBinarySearchInt;
  end;

procedure RegisterTests;
begin
  RegisterTest(TDeviceBinarySearchTests.Suite);
end;

procedure TDeviceBinarySearchTests.TestBinarySearchInt;
var
  // Host data
  h_haystack: array[0..9] of cint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  h_needles: array[0..7] of cint = (0, 2, 12, 4, 14, 6, 8, 10);
  h_output: array[0..7] of csize_t;
  cpu_output: array[0..7] of csize_t;

  // Device pointers
  d_haystack: Pointer = nil;
  d_needles: Pointer = nil;
  d_output: Pointer = nil;
  d_temp_storage: Pointer = nil;

  // Other variables
  haystack_size: csize_t;
  needles_size: csize_t;
  temp_storage_size: csize_t;
  i: Integer;
  j: Integer;
  stream: hipStream_t;

begin
  haystack_size := Length(h_haystack);
  needles_size := Length(h_needles);

  // Prepare CPU expected output
  for i := 0 to needles_size - 1 do
  begin
    cpu_output[i] := 0; // False
    for j := 0 to haystack_size - 1 do
    begin
      if h_needles[i] = h_haystack[j] then
      begin
        cpu_output[i] := 1; // True
        break;
      end;
    end;
  end;

  CheckHIP(hipStreamCreate(@stream), 'Stream Creation');

  // Allocate device memory
  CheckHIP(hipMalloc(@d_haystack, haystack_size * sizeof(cint)), 'Allocating d_haystack');
  CheckHIP(hipMalloc(@d_needles, needles_size * sizeof(cint)), 'Allocating d_needles');
  CheckHIP(hipMalloc(@d_output, needles_size * sizeof(csize_t)), 'Allocating d_output');

  // Copy data from Host to Device
  CheckHIP(hipMemcpy(d_haystack, @h_haystack[0], haystack_size * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_haystack to d_haystack');
  CheckHIP(hipMemcpy(d_needles, @h_needles[0], needles_size * sizeof(cint), hipMemcpyHostToDevice), 'Copying h_needles to d_needles');

  // Get temporary storage size and allocate
  temp_storage_size := 0;
  CheckHIP(
    rocprim_binary_search_int(nil, @temp_storage_size, d_haystack, d_needles, d_output, haystack_size, needles_size, stream),
    'Getting temporary storage size'
  );
  if temp_storage_size > 0 then
    CheckHIP(hipMalloc(@d_temp_storage, temp_storage_size), 'Allocating d_temp_storage');

  // Perform binary_search on the GPU
  CheckHIP(
    rocprim_binary_search_int(d_temp_storage, @temp_storage_size, d_haystack, d_needles, d_output, haystack_size, needles_size, stream),
    'Performing binary_search'
  );

  // Copy result from Device to Host
  CheckHIP(hipMemcpy(@h_output[0], d_output, needles_size * sizeof(csize_t), hipMemcpyDeviceToHost), 'Copying d_output to h_output');

  // Verification
  for i := 0 to needles_size - 1 do
  begin
    CheckEquals(cpu_output[i], h_output[i], 'Mismatch at index ' + IntToStr(i));
  end;

  // Free device memory
  CheckHIP(hipFree(d_haystack), 'Freeing d_haystack');
  CheckHIP(hipFree(d_needles), 'Freeing d_needles');
  CheckHIP(hipFree(d_output), 'Freeing d_output');
  if d_temp_storage <> nil then
    CheckHIP(hipFree(d_temp_storage), 'Freeing d_temp_storage');
  CheckHIP(hipStreamDestroy(stream), 'Stream Destruction');
end;

end.
