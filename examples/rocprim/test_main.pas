program test_main;

{$mode objfpc}{$H+}

uses
  Classes,
  TextTestRunner,
  test_rocprim_device in 'test_rocprim_device.pas';

begin
  // Register all tests
  test_rocprim_device.RegisterTests;

  RunRegisteredTests;
end.
