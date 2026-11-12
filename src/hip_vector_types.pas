{*
 * hip_vector_types.pas
 *
 * Copyright (c) 2023-2026 Eudox67 - Free Pascal wrappers for HIP.
 * This is an unofficial project and is not affiliated with or endorsed by AMD.
 *
 * Derived from HIP headers, which are:
 * Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *}

{ @abstract(This unit provides vector types for HIP, translated from hip_vector_types.h, and their associated constructor functions.)
  @bold(Group: HIP Vector Types)

  @note(This unit defines vector types that are commonly used in GPU programming.
  These are simple records with fields x, y, z, w as appropriate for the vector dimension.
  Platform-specific differences @(AMD vs. NVIDIA@) are handled via conditionals.
  It also includes the `make_*` helper functions to construct these types, making
  this unit self-contained.)


  See also: hip_types.pas for other core types.

}
{ @bold(Group: 1-component vector types) }
unit hip_vector_types;

{$MODE OBJFPC}{$H+}
{$PACKRECORDS C}  // Ensure C-compatible packing for structs/records

interface



type
  { 1-component signed char vector. }
  char1 = record
    x: Int8;
  end;
  PChar1 = ^char1;

  { 1-component unsigned char vector. }
  uchar1 = record
    x: UInt8;
  end;
  PUChar1 = ^uchar1;

  { 1-component signed short vector. }
  short1 = record
    x: Int16;
  end;
  PShort1 = ^short1;

  { 1-component unsigned short vector. }
  ushort1 = record
    x: UInt16;
  end;
  PUShort1 = ^ushort1;

  { 1-component signed int vector. }
  int1 = record
    x: Int32;
  end;
  PInt1 = ^int1;

  { 1-component unsigned int vector. }
  uint1 = record
    x: UInt32;
  end;
  PUInt1 = ^uint1;

  { 1-component signed long vector. }
  long1 = record
    x: Int64;
  end;
  PLong1 = ^long1;

  { 1-component unsigned long vector. }
  ulong1 = record
    x: UInt64;
  end;
  PULong1 = ^ulong1;

  { 1-component float vector. }
  float1 = record
    x: Single;
  end;
  PFloat1 = ^float1;

  { 1-component double vector. }
  double1 = record
    x: Double;
  end;
  PDouble1 = ^double1;


{ @bold(Group: 2-component vector types) }

type
  { 2-component signed char vector. }
  char2 = record
    x: Int8;
    y: Int8;
  end;
  PChar2 = ^char2;

  { 2-component unsigned char vector. }
  uchar2 = record
    x: UInt8;
    y: UInt8;
  end;
  PUChar2 = ^uchar2;

  { 2-component signed short vector. }
  short2 = record
    x: Int16;
    y: Int16;
  end;
  PShort2 = ^short2;

  { 2-component unsigned short vector. }
  ushort2 = record
    x: UInt16;
    y: UInt16;
  end;
  PUShort2 = ^ushort2;

  { 2-component signed int vector. }
  int2 = record
    x: Int32;
    y: Int32;
  end;
  PInt2 = ^int2;

  { 2-component unsigned int vector. }
  uint2 = record
    x: UInt32;
    y: UInt32;
  end;
  PUInt2 = ^uint2;

  { 2-component signed long vector. }
  long2 = record
    x: Int64;
    y: Int64;
  end;
  PLong2 = ^long2;

  { 2-component unsigned long vector. }
  ulong2 = record
    x: UInt64;
    y: UInt64;
  end;
  PULong2 = ^ulong2;

  { 2-component float vector. }
  float2 = record
    x: Single;
    y: Single;
  end;
  PFloat2 = ^float2;

  { 2-component double vector. }
  double2 = record
    x: Double;
    y: Double;
  end;
  PDouble2 = ^double2;


{ @bold(Group: 3-component vector types) }

type
  { 3-component signed char vector. }
  char3 = record
    x: Int8;
    y: Int8;
    z: Int8;
  end;
  PChar3 = ^char3;

  { 3-component unsigned char vector. }
  uchar3 = record
    x: UInt8;
    y: UInt8;
    z: UInt8;
  end;
  PUChar3 = ^uchar3;

  { 3-component signed short vector. }
  short3 = record
    x: Int16;
    y: Int16;
    z: Int16;
  end;
  PShort3 = ^short3;

  { 3-component unsigned short vector. }
  ushort3 = record
    x: UInt16;
    y: UInt16;
    z: UInt16;
  end;
  PUShort3 = ^ushort3;

  { 3-component signed int vector. }
  int3 = record
    x: Int32;
    y: Int32;
    z: Int32;
  end;
  PInt3 = ^int3;

  { 3-component unsigned int vector. }
  uint3 = record
    x: UInt32;
    y: UInt32;
    z: UInt32;
  end;
  PUInt3 = ^uint3;

  { 3-component signed long vector. }
  long3 = record
    x: Int64;
    y: Int64;
    z: Int64;
  end;
  PLong3 = ^long3;

  { 3-component unsigned long vector. }
  ulong3 = record
    x: UInt64;
    y: UInt64;
    z: UInt64;
  end;
  PULong3 = ^ulong3;

  { 3-component float vector. }
  float3 = record
    x: Single;
    y: Single;
    z: Single;
  end;
  PFloat3 = ^float3;

  { 3-component double vector. }
  double3 = record
    x: Double;
    y: Double;
    z: Double;
  end;
  PDouble3 = ^double3;


{ @bold(Group: 4-component vector types) }

type
  { 4-component signed char vector. }
  char4 = record
    x: Int8;
    y: Int8;
    z: Int8;
    w: Int8;
  end;
  PChar4 = ^char4;

  { 4-component unsigned char vector. }
  uchar4 = record
    x: UInt8;
    y: UInt8;
    z: UInt8;
    w: UInt8;
  end;
  PUChar4 = ^uchar4;

  { 4-component signed short vector. }
  short4 = record
    x: Int16;
    y: Int16;
    z: Int16;
    w: Int16;
  end;
  PShort4 = ^short4;

  { 4-component unsigned short vector. }
  ushort4 = record
    x: UInt16;
    y: UInt16;
    z: UInt16;
    w: UInt16;
  end;
  PUShort4 = ^ushort4;

  { 4-component signed int vector. }
  int4 = record
    x: Int32;
    y: Int32;
    z: Int32;
    w: Int32;
  end;
  PInt4 = ^int4;

  { 4-component unsigned int vector. }
  uint4 = record
    x: UInt32;
    y: UInt32;
    z: UInt32;
    w: UInt32;
  end;
  PUInt4 = ^uint4;

  { 4-component signed long vector. }
  long4 = record
    x: Int64;
    y: Int64;
    z: Int64;
    w: Int64;
  end;
  PLong4 = ^long4;

  { 4-component unsigned long vector. }
  ulong4 = record
    x: UInt64;
    y: UInt64;
    z: UInt64;
    w: UInt64;
  end;
  PULong4 = ^ulong4;

  { 4-component float vector. }
  float4 = record
    x: Single;
    y: Single;
    z: Single;
    w: Single;
  end;
  PFloat4 = ^float4;

  { 4-component double vector. }
  double4 = record
    x: Double;
    y: Double;
    z: Double;
    w: Double;
  end;
  PDouble4 = ^double4;


{ @bold(Group: Vector Type Constructors) }
// These functions correspond to the make_* C++ functions in amd_hip_vector_types.h

function make_uchar1(x: UInt8): uchar1; inline;
function make_uchar2(x, y: UInt8): uchar2; inline;
function make_uchar3(x, y, z: UInt8): uchar3; inline;
function make_uchar4(x, y, z, w: UInt8): uchar4; inline;
function make_char1(x: Int8): char1; inline;
function make_char2(x, y: Int8): char2; inline;
function make_char3(x, y, z: Int8): char3; inline;
function make_char4(x, y, z, w: Int8): char4; inline;
function make_ushort1(x: UInt16): ushort1; inline;
function make_ushort2(x, y: UInt16): ushort2; inline;
function make_ushort3(x, y, z: UInt16): ushort3; inline;
function make_ushort4(x, y, z, w: UInt16): ushort4; inline;
function make_short1(x: Int16): short1; inline;
function make_short2(x, y: Int16): short2; inline;
function make_short3(x, y, z: Int16): short3; inline;
function make_short4(x, y, z, w: Int16): short4; inline;
function make_uint1(x: UInt32): uint1; inline;
function make_uint2(x, y: UInt32): uint2; inline;
function make_uint3(x, y, z: UInt32): uint3; inline;
function make_uint4(x, y, z, w: UInt32): uint4; inline;
function make_int1(x: Int32): int1; inline;
function make_int2(x, y: Int32): int2; inline;
function make_int3(x, y, z: Int32): int3; inline;
function make_int4(x, y, z, w: Int32): int4; inline;
function make_ulong1(x: UInt64): ulong1; inline;
function make_ulong2(x, y: UInt64): ulong2; inline;
function make_ulong3(x, y, z: UInt64): ulong3; inline;
function make_ulong4(x, y, z, w: UInt64): ulong4; inline;
function make_long1(x: Int64): long1; inline;
function make_long2(x, y: Int64): long2; inline;
function make_long3(x, y, z: Int64): long3; inline;
function make_long4(x, y, z, w: Int64): long4; inline;
function make_float1(x: Single): float1; inline;
function make_float2(x, y: Single): float2; inline;
function make_float3(x, y, z: Single): float3; inline;
function make_float4(x, y, z, w: Single): float4; inline;
function make_double1(x: Double): double1; inline;
function make_double2(x, y: Double): double2; inline;
function make_double3(x, y, z: Double): double3; inline;
function make_double4(x, y, z, w: Double): double4; inline;


implementation

{ @bold(Group: Vector Type Constructors Implementation) }
function make_uchar1(x: uint8): uchar1;
begin
  Result.x := x;
end;

function make_uchar2(x, y: uint8): uchar2;
begin
  Result.x := x;
  Result.y := y;
end;

function make_uchar3(x, y, z: uint8): uchar3;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

function make_uchar4(x, y, z, w: uint8): uchar4;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
  Result.w := w;
end;

function make_char1(x: int8): char1;
begin
  Result.x := x;
end;

function make_char2(x, y: int8): char2;
begin
  Result.x := x;
  Result.y := y;
end;

function make_char3(x, y, z: int8): char3;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

function make_char4(x, y, z, w: int8): char4;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
  Result.w := w;
end;

function make_ushort1(x: uint16): ushort1;
begin
  Result.x := x;
end;

function make_ushort2(x, y: uint16): ushort2;
begin
  Result.x := x;
  Result.y := y;
end;

function make_ushort3(x, y, z: uint16): ushort3;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

function make_ushort4(x, y, z, w: uint16): ushort4;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
  Result.w := w;
end;

function make_short1(x: int16): short1;
begin
  Result.x := x;
end;

function make_short2(x, y: int16): short2;
begin
  Result.x := x;
  Result.y := y;
end;

function make_short3(x, y, z: int16): short3;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

function make_short4(x, y, z, w: int16): short4;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
  Result.w := w;
end;

function make_uint1(x: uint32): uint1;
begin
  Result.x := x;
end;

function make_uint2(x, y: uint32): uint2;
begin
  Result.x := x;
  Result.y := y;
end;

function make_uint3(x, y, z: uint32): uint3;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

function make_uint4(x, y, z, w: uint32): uint4;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
  Result.w := w;
end;

function make_int1(x: int32): int1;
begin
  Result.x := x;
end;

function make_int2(x, y: int32): int2;
begin
  Result.x := x;
  Result.y := y;
end;

function make_int3(x, y, z: int32): int3;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

function make_int4(x, y, z, w: int32): int4;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
  Result.w := w;
end;

function make_ulong1(x: uint64): ulong1;
begin
  Result.x := x;
end;

function make_ulong2(x, y: uint64): ulong2;
begin
  Result.x := x;
  Result.y := y;
end;

function make_ulong3(x, y, z: uint64): ulong3;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

function make_ulong4(x, y, z, w: uint64): ulong4;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
  Result.w := w;
end;

function make_long1(x: int64): long1;
begin
  Result.x := x;
end;

function make_long2(x, y: int64): long2;
begin
  Result.x := x;
  Result.y := y;
end;

function make_long3(x, y, z: int64): long3;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

function make_long4(x, y, z, w: int64): long4;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
  Result.w := w;
end;

function make_float1(x: single): float1;
begin
  Result.x := x;
end;

function make_float2(x, y: single): float2;
begin
  Result.x := x;
  Result.y := y;
end;

function make_float3(x, y, z: single): float3;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

function make_float4(x, y, z, w: single): float4;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
  Result.w := w;
end;

function make_double1(x: double): double1;
begin
  Result.x := x;
end;

function make_double2(x, y: double): double2;
begin
  Result.x := x;
  Result.y := y;
end;

function make_double3(x, y, z: double): double3;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

function make_double4(x, y, z, w: double): double4;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
  Result.w := w;
end;

end.
