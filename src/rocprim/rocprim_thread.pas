{ rocPRIM thread-level primitives. }
{ rocPRIM thread-level primitives. }
unit rocprim_thread;

{$MODE OBJFPC}{$H+}
{$modeswitch advancedrecords}
{$PACKRECORDS C}

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

  This file is a Pascal translation of the C++ header files in the rocprim/thread directory.
}

interface

uses
  rocprim_core, rocprim_types, math;

type
  TIntegerOp = function(a, b: Integer): Integer;

  TCacheLoadModifier = (
    load_default,
    load_ca,
    load_cg,
    load_cs,
    load_cv,
    load_ldg,
    load_volatile
  );

  TCacheStoreModifier = (
    store_default,
    store_wb,
    store_cg,
    store_cs,
    store_wt,
    store_volatile
  );


  TSum = record  // Advanced record: Methods + constructor
  public
    { Binary sum (32-bit Integer). }
    function DoOp(a, b: Integer): Integer;
    { Binary sum (64-bit Int64). }
    function DoOp(a, b: Int64): Int64;
  end;

  { Min functor: DoOp(a, b) = min(a, b). }
  TMin = record
    function DoOp(a, b: LongInt): LongInt;
    function DoOp(a, b: Int64): Int64;
  end;

  { Max functor: DoOp(a, b) = max(a, b). }
  TMax = record
    function DoOp(a, b: LongInt): LongInt;
    function DoOp(a, b: Int64): Int64;
  end;

// Use TBinaryOpInt for clarity and FPC compatibility
function thread_reduce(input: PInteger; length: LongInt; op: TBinaryOpInt): LongInt;
procedure thread_scan_exclusive(input, output: PInteger; length: LongInt; op: TBinaryOpInt; prefix: LongInt);
procedure thread_scan_inclusive(input, output: PInteger; length: LongInt; op: TBinaryOpInt);

// Lower-level functions with placeholders
procedure thread_load(modifier: TCacheLoadModifier; ptr: Pointer);
procedure thread_store(modifier: TCacheStoreModifier; ptr: Pointer; value: Pointer);
function lower_bound(input: PInteger; num_items, val: LongInt): LongInt;
function upper_bound(input: PInteger; num_items, val: LongInt): LongInt;

implementation

function thread_reduce(input: PInteger; length: LongInt; op: TBinaryOpInt): LongInt;
var
  i: LongInt;
  retval: LongInt;
begin
  if length = 0 then
    Exit(0);
  retval := input[0];
  for i := 1 to length - 1 do
    retval := op(retval, input[i]);
  Result := retval;
end;

procedure thread_scan_exclusive(input, output: PInteger; length: LongInt; op: TBinaryOpInt; prefix: LongInt);
var
  i: LongInt;
  inclusive, exclusive: LongInt;
begin
  if length = 0 then
    Exit;
  exclusive := prefix;
  for i := 0 to length - 1 do
  begin
    inclusive := op(exclusive, input[i]);
    output[i] := exclusive;
    exclusive := inclusive;
  end;
end;

procedure thread_scan_inclusive(input, output: PInteger; length: LongInt; op: TBinaryOpInt);
var
  i: LongInt;
  inclusive: LongInt;
begin
  if length = 0 then
    Exit;
  inclusive := input[0];
  output[0] := inclusive;
  for i := 1 to length - 1 do
  begin
    inclusive := op(inclusive, input[i]);
    output[i] := inclusive;
  end;
end;

procedure thread_load(modifier: TCacheLoadModifier; ptr: Pointer);
begin
  // Placeholder for hardware-specific load
end;

procedure thread_store(modifier: TCacheStoreModifier; ptr: Pointer; value: Pointer);
begin
  // Placeholder for hardware-specific store
end;

function lower_bound(input: PInteger; num_items, val: LongInt): LongInt;
var
  half, retval: LongInt;
begin
  retval := 0;
  while num_items > 0 do
  begin
    half := num_items shr 1;
    if input[retval + half] < val then
    begin
      retval := retval + half + 1;
      num_items := num_items - (half + 1);
    end
    else
    begin
      num_items := half;
    end;
  end;
  Result := retval;
end;

function upper_bound(input: PInteger; num_items, val: LongInt): LongInt;
var
  half, retval: LongInt;
begin
  retval := 0;
  while num_items > 0 do
  begin
    half := num_items shr 1;
    if val < input[retval + half] then
    begin
      num_items := half;
    end
    else
    begin
      retval := retval + half + 1;
      num_items := num_items - (half + 1);
    end;
  end;
  Result := retval;
end;


{ TSum }
function TSum.DoOp(a, b: Integer): Integer;
begin
  Result := a + b;  // Simple sum (CPU impl; could external for GPU)
end;

function TSum.DoOp(a, b: Int64): Int64;
begin
  Result := a + b;  // 64-bit overload
end;

{ TMin }
function TMin.DoOp(a, b: LongInt): LongInt;
begin
  Result := Min(a, b);  // FPC Min (arithmetic min)
end;

function TMin.DoOp(a, b: Int64): Int64;
begin
  Result := Min(a, b);  // 64-bit min
end;

{ TMax }
function TMax.DoOp(a, b: LongInt): LongInt;
begin
  Result := Max(a, b);  // FPC Max
end;

function TMax.DoOp(a, b: Int64): Int64;
begin
  Result := Max(a, b);  // 64-bit max
end;


end.
