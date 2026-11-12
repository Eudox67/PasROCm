{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.

  Copyright (c) 2024-2026 Eudox67

  MIT License

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.

  This is an unofficial project and is not affiliated with or endorsed by AMD.
------------------------------------------------------------------------------}

{ Specialised HIP host-side helpers. }
{ Specialised HIP host-side helpers. }
unit hip_specialized;

{$MODE OBJFPC}{$H+}
{$PACKRECORDS C}

interface

uses
  hip_common, Math;

type
  THalf = record
    data: UInt16;
  end;
  PHalf = ^THalf;

function hipFloatToHalf(f: Single): THalf;
function hipHalfToFloat(h: THalf): Single;

type
  TBFloat16 = record
    data: UInt16;
  end;
  PBFloat16 = ^TBFloat16;

function hipFloatToBFloat16(f: Single): TBFloat16;
function hipBFloat16ToFloat(b: TBFloat16): Single;

type
  TFP8e4m3fnuz = record
    data: UInt8;
  end;
  PFP8e4m3fnuz = ^TFP8e4m3fnuz;

  TFP8e5m2fnuz = record
    data: UInt8;
  end;
  PFP8e5m2fnuz = ^TFP8e5m2fnuz;

function hipFloatToFP8e4m3fnuz(f: Single): TFP8e4m3fnuz;
function hipFP8e4m3fnuzToFloat(h: TFP8e4m3fnuz): Single;
function hipFloatToFP8e5m2fnuz(f: Single): TFP8e5m2fnuz;
function hipFP8e5m2fnuzToFloat(h: TFP8e5m2fnuz): Single;

function AddHalfs(a, b: THalf): THalf; inline;
function SubHalfs(a, b: THalf): THalf; inline;
function MulHalfs(a, b: THalf): THalf; inline;
function DivHalfs(a, b: THalf): THalf; inline;

function AddBFloat16s(a, b: TBFloat16): TBFloat16; inline;

implementation

// Local stubs for missing intrinsics
function FloatToHalf(f: Single): UInt16;
begin
  Result := 0; // Stub
end;

function HalfToFloat(h: UInt16): Single;
begin
  Result := 0.0; // Stub
end;

function hipFloatToHalf(f: Single): THalf;
begin
  Result.data := FloatToHalf(f);
end;

function hipHalfToFloat(h: THalf): Single;
begin
  Result := HalfToFloat(h.data);
end;

function hipFloatToBFloat16(f: Single): TBFloat16;
var
  u: LongWord absolute f;
begin
  if IsNaN(f) then
  begin
    Result.data := $7FC0;
    Exit;
  end;
  Result.data := (u + $8000) shr 16;
end;

function hipBFloat16ToFloat(b: TBFloat16): Single;
var
  u: LongWord;
  res: Single absolute u;
begin
  u := LongWord(b.data) shl 16;
  Result := res;
end;

function hipFloatToFP8e4m3fnuz(f: Single): TFP8e4m3fnuz;
begin
  Result.data := 0; // Stub
end;

function hipFP8e4m3fnuzToFloat(h: TFP8e4m3fnuz): Single;
begin
  Result := 0.0; // Stub
end;

function hipFloatToFP8e5m2fnuz(f: Single): TFP8e5m2fnuz;
begin
  Result.data := 0; // Stub
end;

function hipFP8e5m2fnuzToFloat(h: TFP8e5m2fnuz): Single;
begin
  Result := 0.0; // Stub
end;

function AddHalfs(a, b: THalf): THalf;
begin
  Result := hipFloatToHalf(hipHalfToFloat(a) + hipHalfToFloat(b));
end;

function SubHalfs(a, b: THalf): THalf;
begin
  Result := hipFloatToHalf(hipHalfToFloat(a) - hipHalfToFloat(b));
end;

function MulHalfs(a, b: THalf): THalf;
begin
  Result := hipFloatToHalf(hipHalfToFloat(a) * hipHalfToFloat(b));
end;

function DivHalfs(a, b: THalf): THalf;
begin
  Result := hipFloatToHalf(hipHalfToFloat(a) / hipHalfToFloat(b));
end;

function AddBFloat16s(a, b: TBFloat16): TBFloat16;
begin
  Result := hipFloatToBFloat16(hipBFloat16ToFloat(a) + hipBFloat16ToFloat(b));
end;

end.
