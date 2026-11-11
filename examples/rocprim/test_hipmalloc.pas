program test_hipmalloc;

{$mode objfpc}{$H+}

uses
  ctypes,
  hip_bindings;

// Helper procedure to check HIP API call results
procedure CheckHIP(err: hipError_t; const msg: string);
begin
  if err <> hipSuccess then
  begin
    WriteLn('HIP Error: ', err, ' at ', msg);
    Halt(1);
  end;
end;

var
  d_ptr: Pointer = nil;
  size: csize_t;

begin
  WriteLn('Testing hipMalloc...');
  size := 160;
  WriteLn(' - Allocating ', size, ' bytes...');
  CheckHIP(hipMalloc(@d_ptr, size), 'Allocating memory');
  WriteLn(' - Allocation successful.');
  CheckHIP(hipFree(d_ptr), 'Freeing memory');
  WriteLn(' - Memory freed.');
  WriteLn('Test complete.');
end.
