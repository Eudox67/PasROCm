{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.
  Copyright (c) 2024-2026 Eudox67. MIT License (see LICENSE).

  One job: the SHA256 of a file on disk, as lowercase hex, identical to what
  `sha256sum` prints.

  Self-contained on purpose. FPC's fpsha256 lives in fcl-hash, which a stock
  Ubuntu `apt install fpc` does not ship -- measured 2026-11-15, when the new
  example gate failed on every file with "Can't find unit fpsha256". The loader
  is a core unit, so it must compile with nothing but the compiler. SHA256 is
  small, fully specified, and checked here against published vectors and a
  million-byte file (see tests/Test_pasrocm_shim_loader.pas).
------------------------------------------------------------------------------}
unit rocprim_shim_hash;

{$mode objfpc}{$H+}

interface

{ Raises EFOpenError if the file cannot be read. System-only file I/O. }
function PasROCmShimFileSHA256(const APath: string): string;

implementation

uses
  SysUtils;

const
  SHA256_K: array[0..63] of Cardinal = (
    $428a2f98, $71374491, $b5c0fbcf, $e9b5dba5, $3956c25b, $59f111f1, $923f82a4, $ab1c5ed5,
    $d807aa98, $12835b01, $243185be, $550c7dc3, $72be5d74, $80deb1fe, $9bdc06a7, $c19bf174,
    $e49b69c1, $efbe4786, $0fc19dc6, $240ca1cc, $2de92c6f, $4a7484aa, $5cb0a9dc, $76f988da,
    $983e5152, $a831c66d, $b00327c8, $bf597fc7, $c6e00bf3, $d5a79147, $06ca6351, $14292967,
    $27b70a85, $2e1b2138, $4d2c6dfc, $53380d13, $650a7354, $766a0abb, $81c2c92e, $92722c85,
    $a2bfe8a1, $a81a664b, $c24b8b70, $c76c51a3, $d192e819, $d6990624, $f40e3585, $106aa070,
    $19a4c116, $1e376c08, $2748774c, $34b0bcb5, $391c0cb3, $4ed8aa4a, $5b9cca4f, $682e6ff3,
    $748f82ee, $78a5636f, $84c87814, $8cc70208, $90befffa, $a4506ceb, $bef9a3f7, $c67178f2);

type
  TSha256State = array[0..7] of Cardinal;
  TSha256Block = array[0..63] of Byte;

function RotR(X: Cardinal; N: Byte): Cardinal; inline;
begin
  Result := (X shr N) or (X shl (32 - N));
end;

procedure Sha256Block(var State: TSha256State; const Block: TSha256Block);
var
  W: array[0..63] of Cardinal;
  A, B, C, D, E, F, G, H, T1, T2, S0, S1: Cardinal;
  I: Integer;
begin
  for I := 0 to 15 do
    W[I] := (Cardinal(Block[I * 4]) shl 24) or (Cardinal(Block[I * 4 + 1]) shl 16)
         or (Cardinal(Block[I * 4 + 2]) shl 8) or Cardinal(Block[I * 4 + 3]);
  for I := 16 to 63 do
  begin
    S0 := RotR(W[I - 15], 7) xor RotR(W[I - 15], 18) xor (W[I - 15] shr 3);
    S1 := RotR(W[I - 2], 17) xor RotR(W[I - 2], 19) xor (W[I - 2] shr 10);
    W[I] := W[I - 16] + S0 + W[I - 7] + S1;
  end;

  A := State[0]; B := State[1]; C := State[2]; D := State[3];
  E := State[4]; F := State[5]; G := State[6]; H := State[7];

  for I := 0 to 63 do
  begin
    S1 := RotR(E, 6) xor RotR(E, 11) xor RotR(E, 25);
    T1 := H + S1 + ((E and F) xor ((not E) and G)) + SHA256_K[I] + W[I];
    S0 := RotR(A, 2) xor RotR(A, 13) xor RotR(A, 22);
    T2 := S0 + ((A and B) xor (A and C) xor (B and C));
    H := G; G := F; F := E; E := D + T1;
    D := C; C := B; B := A; A := T1 + T2;
  end;

  State[0] := State[0] + A; State[1] := State[1] + B;
  State[2] := State[2] + C; State[3] := State[3] + D;
  State[4] := State[4] + E; State[5] := State[5] + F;
  State[6] := State[6] + G; State[7] := State[7] + H;
end;

function PasROCmShimFileSHA256(const APath: string): string;
const
  HEX: array[0..15] of Char = '0123456789abcdef';
var
  Handle: file of Byte;
  Buffer: array[0..65535] of Byte;
  Block: TSha256Block;
  State: TSha256State;
  Got, Fill, I: LongInt;
  TotalBytes: QWord;
  BitLen: QWord;
  PadLen: Integer;
  Octet: Byte;
begin
  State[0] := $6a09e667; State[1] := $bb67ae85; State[2] := $3c6ef372; State[3] := $a54ff53a;
  State[4] := $510e527f; State[5] := $9b05688c; State[6] := $1f83d9ab; State[7] := $5be0cd19;
  TotalBytes := 0;
  Fill := 0;

  AssignFile(Handle, APath);
  Reset(Handle);
  try
    repeat
      BlockRead(Handle, Buffer, SizeOf(Buffer), Got);
      TotalBytes := TotalBytes + QWord(Got);
      I := 0;
      while I < Got do
      begin
        Block[Fill] := Buffer[I];
        Inc(Fill);
        Inc(I);
        if Fill = 64 then
        begin
          Sha256Block(State, Block);
          Fill := 0;
        end;
      end;
    until Got = 0;
  finally
    CloseFile(Handle);
  end;

  { Padding: 0x80, then zeros, then the 64-bit big-endian bit length. }
  BitLen := TotalBytes * 8;
  Block[Fill] := $80;
  Inc(Fill);
  if Fill > 56 then
  begin
    while Fill < 64 do
    begin
      Block[Fill] := 0;
      Inc(Fill);
    end;
    Sha256Block(State, Block);
    Fill := 0;
  end;
  while Fill < 56 do
  begin
    Block[Fill] := 0;
    Inc(Fill);
  end;
  for I := 7 downto 0 do
  begin
    Block[56 + (7 - I)] := Byte((BitLen shr (I * 8)) and $FF);
  end;
  Sha256Block(State, Block);

  Result := '';
  for I := 0 to 7 do
    for PadLen := 3 downto 0 do
    begin
      Octet := Byte((State[I] shr (PadLen * 8)) and $FF);
      Result := Result + HEX[Octet shr 4] + HEX[Octet and $F];
    end;
end;

end.
