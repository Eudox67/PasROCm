program test_ptr;

{$mode objfpc}{$H+}

uses
  SysUtils;

var
  p: Pointer;
  s: string;

begin
  p := nil;
  s := PtrToStr(p);
  WriteLn(s);
end.
