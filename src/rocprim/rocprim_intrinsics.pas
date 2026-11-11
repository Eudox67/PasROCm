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

unit rocprim_intrinsics;

{< Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
  Pascal translation/bindings: Copyright (c) 2024-2026 Eudox67
  ...
>}

interface

uses
  rocprim_core, rocprim_types;

const
  ROCPRIM_WAVEFRONT_SIZE = 64;

function get_bit(x: Integer; i: Integer): Integer;
function bit_count(x: LongWord): LongWord;
function bit_count_ll(x: QWord): QWord;
function ctz(x: LongWord): LongWord;
function ctz_ll(x: QWord): QWord;

function clz(x: LongWord): LongWord;
function clz_ll(x: QWord): QWord;
function ffs(x: LongInt): LongInt;
function ffs_ll(x: Int64): Int64;

function device_warp_size: Cardinal;
function flat_block_size: Cardinal;
function flat_tile_size: Cardinal;
function lane_id: Cardinal;
function flat_block_thread_id: Cardinal;
function flat_tile_thread_id: Cardinal;
function warp_id: Cardinal;
function flat_block_id: Cardinal;

procedure syncthreads;
procedure wave_barrier;

function masked_bit_count(x: lane_mask_type; add: Cardinal = 0): Cardinal;

// Atomics
function atomic_add(ptr: PLongInt; value: LongInt): LongInt;
function atomic_add(ptr: PInt64; value: Int64): Int64;
function atomic_add(ptr: PSingle; value: Single): Single;
function atomic_add(ptr: PDouble; value: Double): Double;

function atomic_exchange(ptr: PLongInt; value: LongInt): LongInt;
function atomic_exchange(ptr: PInt64; value: Int64): Int64;
function atomic_exchange(ptr: PSingle; value: Single): Single;
function atomic_exchange(ptr: PDouble; value: Double): Double;

function atomic_min(ptr: PLongInt; value: LongInt): LongInt;
function atomic_min(ptr: PInt64; value: Int64): Int64;
function atomic_min(ptr: PSingle; value: Single): Single;
function atomic_min(ptr: PDouble; value: Double): Double;

function atomic_max(ptr: PLongInt; value: LongInt): LongInt;
function atomic_max(ptr: PInt64; value: Int64): Int64;
function atomic_max(ptr: PSingle; value: Single): Single;
function atomic_max(ptr: PDouble; value: Double): Double;

function atomic_cas(ptr: PLongInt; compare: LongInt; value: LongInt): LongInt;
function atomic_cas(ptr: PInt64; compare: Int64; value: Int64): Int64;

function atomic_inc(ptr: PLongWord): LongWord;
function atomic_dec(ptr: PLongWord): LongWord;

implementation

function get_bit(x: Integer; i: Integer): Integer;
begin
  Result := (x shr i) and 1;
end;

function bit_count(x: LongWord): LongWord;
begin
  Result := PopCnt(x);
end;

function bit_count_ll(x: QWord): QWord;
begin
  Result := PopCnt(x);
end;

function ctz(x: LongWord): LongWord;
begin
  Result := BsfDWord(x);
end;

function ctz_ll(x: QWord): QWord;
begin
  Result := BsfQWord(x);
end;

function clz(x: LongWord): LongWord;
begin
  if x = 0 then Result := 32 else Result := 31 - BsrDWord(x);
end;

function clz_ll(x: QWord): QWord;
begin
  if x = 0 then Result := 64 else Result := 63 - BsrQWord(x);
end;

function ffs(x: LongInt): LongInt;
begin
  if x = 0 then Result := 0 else Result := BsfDWord(LongWord(x)) + 1;
end;

function ffs_ll(x: Int64): Int64;
begin
  if x = 0 then Result := 0 else Result := BsfQWord(QWord(x)) + 1;
end;

function device_warp_size: Cardinal;
begin
  Result := ROCPRIM_WAVEFRONT_SIZE;
end;

function flat_block_size: Cardinal;
begin
  Result := 1;
end;

function flat_tile_size: Cardinal;
begin
  Result := flat_block_size;
end;

function lane_id: Cardinal;
begin
  Result := 0;
end;

function flat_block_thread_id: Cardinal;
begin
  Result := 0;
end;

function flat_tile_thread_id: Cardinal;
begin
  Result := flat_block_thread_id;
end;

function warp_id: Cardinal;
begin
  Result := flat_block_thread_id div device_warp_size;
end;

function flat_block_id: Cardinal;
begin
  Result := 0;
end;

procedure syncthreads;
begin
end;

procedure wave_barrier;
begin
end;

function masked_bit_count(x: lane_mask_type; add: Cardinal = 0): Cardinal;
begin
  Result := 0;
end;

// Atomics Stubs
function atomic_add(ptr: PLongInt; value: LongInt): LongInt; begin Result := 0; end;
function atomic_add(ptr: PInt64; value: Int64): Int64; begin Result := 0; end;
function atomic_add(ptr: PSingle; value: Single): Single; begin Result := 0; end;
function atomic_add(ptr: PDouble; value: Double): Double; begin Result := 0; end;

function atomic_exchange(ptr: PLongInt; value: LongInt): LongInt; begin Result := 0; end;
function atomic_exchange(ptr: PInt64; value: Int64): Int64; begin Result := 0; end;
function atomic_exchange(ptr: PSingle; value: Single): Single; begin Result := 0; end;
function atomic_exchange(ptr: PDouble; value: Double): Double; begin Result := 0; end;

function atomic_min(ptr: PLongInt; value: LongInt): LongInt; begin Result := 0; end;
function atomic_min(ptr: PInt64; value: Int64): Int64; begin Result := 0; end;
function atomic_min(ptr: PSingle; value: Single): Single; begin Result := 0; end;
function atomic_min(ptr: PDouble; value: Double): Double; begin Result := 0; end;

function atomic_max(ptr: PLongInt; value: LongInt): LongInt; begin Result := 0; end;
function atomic_max(ptr: PInt64; value: Int64): Int64; begin Result := 0; end;
function atomic_max(ptr: PSingle; value: Single): Single; begin Result := 0; end;
function atomic_max(ptr: PDouble; value: Double): Double; begin Result := 0; end;

function atomic_cas(ptr: PLongInt; compare: LongInt; value: LongInt): LongInt; begin Result := 0; end;
function atomic_cas(ptr: PInt64; compare: Int64; value: Int64): Int64; begin Result := 0; end;

function atomic_inc(ptr: PLongWord): LongWord; begin Result := 0; end;
function atomic_dec(ptr: PLongWord): LongWord; begin Result := 0; end;

end.