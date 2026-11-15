{ Types shared by the rocPRIM layer. }
{ Types shared by the rocPRIM layer. }
unit rocprim_types;

{$modeswitch advancedrecords}
{$PACKRECORDS C}

{$MODE OBJFPC}{$H+}
{$modeswitch advancedrecords}

{ Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
  Pascal translation/bindings: Copyright (c) 2024-2026 Eudox67

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
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.

  This file is a Pascal translation of the following C++ header files:
  - rocprim/types.hpp
  - rocprim/types/double_buffer.hpp
  - rocprim/types/future_value.hpp
  - rocprim/types/integer_sequence.hpp
  - rocprim/types/key_value_pair.hpp
  - rocprim/types/tuple.hpp
  - rocprim/types/uninitialized_array.hpp
}

interface

uses
  rocprim_core; //hip_specialized;

type
  empty_type = record
  end;

  identity_decomposer = record
  end;


  THalf = record
    data: UInt16;
  end;
  PHalf = ^THalf;

  TBFloat16 = record
    data: UInt16;
  end;
  PBFloat16 = ^TBFloat16;

  PInteger = ^LongInt;

  // Corresponds to __uint128_t
  uint128_t = record
    low, high: QWord;
  end;

  // Corresponds to __int128_t. The two halves are the ABI layout and must not
  // change; the operators below only add host-side usability. Comparisons treat
  // 'high' as signed and 'low' as unsigned, which is what two's complement over
  // 128 bits means.
  int128_t = record
    low, high: QWord;
    class operator := (v: Int64): int128_t;
    class operator := (v: QWord): int128_t;
    class operator + (a, b: int128_t): int128_t;
    class operator - (a: int128_t): int128_t;
    class operator - (a, b: int128_t): int128_t;
    class operator = (a, b: int128_t): Boolean;
    class operator <> (a, b: int128_t): Boolean;
    class operator < (a, b: int128_t): Boolean;
    class operator > (a, b: int128_t): Boolean;
    class operator <= (a, b: int128_t): Boolean;
    class operator >= (a, b: int128_t): Boolean;
    function IsZero: Boolean;
    function IsNegative: Boolean;
    function ToStr: string;
    function ToHex: string;
  end;

  // lane_mask_type is now based on arch, but for host-side FFI,
  // using the max size is the safest approach.
  lane_mask_type = QWord;

  { Binary operator for reductions/scans (e.g., + for sum).
  Type for function pointers like rocPRIM's BinaryOp. Use LongInt for FPC compatibility. }
  TBinaryOpInt = function(a, b: LongInt): LongInt;

  // HIP vector types are now used directly.
  // These are simplified representations based on hip_vector_types.h
  char2 = record
    x, y: ShortInt;
  end;
  char4 = record
    x, y, z, w: ShortInt;
  end;
  short2 = record
    x, y: SmallInt;
  end;
  short4 = record
    x, y, z, w: SmallInt;
  end;
  int2 = record
    x, y: LongInt;
  end;
  int4 = record
    x, y, z, w: LongInt;
  end;
  longlong2 = record
    x, y: Int64;
  end;
  longlong4 = record
    x, y, z, w: Int64;
  end;

  TDoubleBuffer = record
    buffers: array[0..1] of Pointer;
    selector: Cardinal;
    function current: Pointer;
    function alternate: Pointer;
    procedure swap;
  end;

  TFutureValue = record
    iter: Pointer;
    function get_value: Pointer; // Simplified to return a pointer
  end;

  TKeyValuePair = record
    key: Pointer;   // Using Pointer as a generic type
    value: Pointer; // Using Pointer as a generic type
  end;

  TIntegerSequence = record
    // This is a compile-time construct in C++.
    // In Pascal, you might use a dynamic array or a simple array.
  end;

  TTuple = record
    // A simplified, non-generic tuple. For generic behavior,
    // dynamic arrays or other structures would be used in Pascal.
    item1: Pointer;
    item2: Pointer;
  end;

  TUninitializedArray = record
    // Simplified representation. In a real scenario, you would manage memory manually.
    storage: array of Byte;
  end;

  { Convenience wrappers, so a caller can print a 128-bit value without
    knowing whether it is a record method or a free function. }
  function int128_to_str(const v: int128_t): string;
  function int128_to_hex(const v: int128_t): string;

implementation

uses
  SysUtils;   { IntToHex, for int128_t.ToHex }

{ ---- int128_t: host-side arithmetic over the ABI's two halves -------------- }

class operator int128_t.:= (v: Int64): int128_t;
begin
  Result.low := QWord(v);
  if v < 0 then
    Result.high := QWord(-1)   { sign-extend }
  else
    Result.high := 0;
end;

class operator int128_t.:= (v: QWord): int128_t;
begin
  Result.low := v;
  Result.high := 0;
end;

class operator int128_t.+ (a, b: int128_t): int128_t;
var
  Carry: QWord;
begin
  Result.low := a.low + b.low;
  { Unsigned wrap in the low half is the carry into the high half. }
  if Result.low < a.low then
    Carry := 1
  else
    Carry := 0;
  Result.high := a.high + b.high + Carry;
end;

class operator int128_t.- (a: int128_t): int128_t;
var
  One: int128_t;
begin
  { Two's complement: invert both halves, then add one. }
  Result.low := not a.low;
  Result.high := not a.high;
  One.low := 1;
  One.high := 0;
  Result := Result + One;
end;

class operator int128_t.- (a, b: int128_t): int128_t;
begin
  Result := a + (-b);
end;

class operator int128_t.= (a, b: int128_t): Boolean;
begin
  Result := (a.low = b.low) and (a.high = b.high);
end;

class operator int128_t.<> (a, b: int128_t): Boolean;
begin
  Result := (a.low <> b.low) or (a.high <> b.high);
end;

class operator int128_t.< (a, b: int128_t): Boolean;
begin
  if Int64(a.high) <> Int64(b.high) then
    Exit(Int64(a.high) < Int64(b.high));
  Result := a.low < b.low;
end;

class operator int128_t.> (a, b: int128_t): Boolean;
begin
  Result := b < a;
end;

class operator int128_t.<= (a, b: int128_t): Boolean;
begin
  Result := not (b < a);
end;

class operator int128_t.>= (a, b: int128_t): Boolean;
begin
  Result := not (a < b);
end;

function int128_t.IsZero: Boolean;
begin
  Result := (low = 0) and (high = 0);
end;

function int128_t.IsNegative: Boolean;
begin
  Result := Int64(high) < 0;
end;

{ Divide a 128-bit value by 10 in place, returning the remainder. Schoolbook
  division over four 32-bit limbs: the intermediate (rem shl 32) or limb stays
  below 10 * 2^32, so it fits a QWord and needs no 128-bit division. }
function Div10(var v: int128_t): QWord;
var
  Limbs: array[0..3] of Cardinal;
  Cur, Rem: QWord;
  I: Integer;
begin
  Limbs[0] := Cardinal(v.low and $FFFFFFFF);
  Limbs[1] := Cardinal(v.low shr 32);
  Limbs[2] := Cardinal(v.high and $FFFFFFFF);
  Limbs[3] := Cardinal(v.high shr 32);
  Rem := 0;
  for I := 3 downto 0 do
  begin
    Cur := (Rem shl 32) or Limbs[I];
    Limbs[I] := Cardinal(Cur div 10);
    Rem := Cur mod 10;
  end;
  v.low := QWord(Limbs[1]) shl 32 or Limbs[0];
  v.high := QWord(Limbs[3]) shl 32 or Limbs[2];
  Result := Rem;
end;

function int128_t.ToStr: string;
var
  Work: int128_t;
  Digits: string;
  Neg: Boolean;
begin
  Work := Self;
  Neg := Work.IsNegative;
  if Neg then
    Work := -Work;
  if Work.IsZero then
    Exit('0');
  Digits := '';
  while not Work.IsZero do
    Digits := Chr(Ord('0') + Div10(Work)) + Digits;
  if Neg then
    Result := '-' + Digits
  else
    Result := Digits;
end;

function int128_t.ToHex: string;
var
  S: string;
begin
  S := IntToHex(high, 16) + IntToHex(low, 16);
  while (Length(S) > 1) and (S[1] = '0') do
    Delete(S, 1, 1);
  Result := '0x' + S;
end;

function int128_to_str(const v: int128_t): string;
begin
  Result := v.ToStr;
end;

function int128_to_hex(const v: int128_t): string;
begin
  Result := v.ToHex;
end;



{ TDoubleBuffer }

function TDoubleBuffer.current: Pointer;
begin
  Result := buffers[selector];
end;

function TDoubleBuffer.alternate: Pointer;
begin
  Result := buffers[selector xor 1];
end;

procedure TDoubleBuffer.swap;
begin
  selector := selector xor 1;
end;

{ TFutureValue }

function TFutureValue.get_value: Pointer;
begin
  Result := iter;
end;

end.
