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

unit rocprim_warp;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

interface

uses
  hip_runtime, hip_types, hip_error, ctypes, rocprim_core, rocprim_types;

procedure rocprim_set_warp_size(size: cuint); cdecl; external;

// --- Shim Bindings ---

// Warp Reduce
function rocprim_warp_reduce_int8(d_out: PShortInt; d_in: PShortInt; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_reduce_int16(d_out: PSmallInt; d_in: PSmallInt; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_reduce_int32(d_out: PInteger; d_in: PInteger; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_reduce_uint32(d_out: PCardinal; d_in: PCardinal; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_reduce_int64(d_out: PInt64; d_in: PInt64; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_reduce_uint64(d_out: PQWord; d_in: PQWord; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;

// Warp Scan
function rocprim_warp_scan_int8(d_out: PShortInt; d_in: PShortInt; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_scan_int16(d_out: PSmallInt; d_in: PSmallInt; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_scan_int32(d_out: PInteger; d_in: PInteger; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_scan_uint32(d_out: PCardinal; d_in: PCardinal; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_scan_int64(d_out: PInt64; d_in: PInt64; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_scan_uint64(d_out: PQWord; d_in: PQWord; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;

// Warp Sort
function rocprim_warp_sort_int8(d_out: PShortInt; d_in: PShortInt; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_sort_int16(d_out: PSmallInt; d_in: PSmallInt; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_sort_int32(d_out: PInteger; d_in: PInteger; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_sort_uint32(d_out: PCardinal; d_in: PCardinal; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_sort_int64(d_out: PInt64; d_in: PInt64; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_warp_sort_uint64(d_out: PQWord; d_in: PQWord; num_items: size_t; stream: hipStream_t): hipError_t; cdecl; external;


type
  TWarpLoadMethod = (
    warp_load_direct,
    warp_load_striped,
    warp_load_vectorize,
    warp_load_transpose,
    warp_load_default
  );

  TWarpStoreMethod = (
    warp_store_direct,
    warp_store_striped,
    warp_store_vectorize,
    warp_store_transpose,
    warp_store_default
  );

  TWarpExchange = record
    // Bindings pending shim support
  end;

  TWarpLoad = record
    // Bindings pending shim support
  end;

  TWarpReduce = record
    // Use rocprim_warp_reduce_* functions directly
  end;

  TWarpScan = record
    // Use rocprim_warp_scan_* functions directly
  end;

  TWarpSort = record
    // Use rocprim_warp_sort_* functions directly
  end;

  TWarpStore = record
    // Bindings pending shim support
  end;

implementation

end.
