{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  One job: the SHA256 of a file on disk, as lowercase hex, identical to what
  `sha256sum` prints. It lives in its own unit so the loader's own uses clause
  stays small, and so the hash can be tested on its own.
------------------------------------------------------------------------------}
unit rocprim_shim_hash;

{$mode objfpc}{$H+}

interface

{ Raises EFOpenError if the file cannot be read. System-only file I/O and the
  TSHA256 record: no RTL class library is pulled in. }
function PasROCmShimFileSHA256(const APath: string): string;

implementation

uses
  SysUtils, fpsha256;

function PasROCmShimFileSHA256(const APath: string): string;
var
  Handle: file of Byte;
  Buffer: array[0..65535] of Byte;
  Got: LongInt;
  Hasher: TSHA256;
  Hex: AnsiString;
begin
  AssignFile(Handle, APath);
  Reset(Handle);
  try
    Hasher.Init;
    repeat
      BlockRead(Handle, Buffer, SizeOf(Buffer), Got);
      if Got > 0 then
        Hasher.Update(@Buffer[0], UInt32(Got));
    until Got = 0;
    Hasher.Final;
  finally
    CloseFile(Handle);
  end;
  Hasher.OutputHexa(Hex);
  Result := LowerCase(string(Hex));
end;

end.
