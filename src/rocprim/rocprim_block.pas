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

{ rocPRIM block-level primitives. }
{ rocPRIM block-level primitives. }
unit rocprim_block;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

interface

uses
  ctypes, rocprim_core, rocprim_types, hip_types, hip_error, hip_runtime;

// --- Shim Bindings ---

function rocprim_block_reduce_int(d_out: PInteger; d_in: PInteger; num_items: size_t; block_size: Cardinal; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_block_scan_int(d_out: PInteger; d_in: PInteger; num_items: size_t; block_size: Cardinal; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_block_sort_int(d_out: PInteger; d_in: PInteger; num_items: size_t; block_size: Cardinal; stream: hipStream_t): hipError_t; cdecl; external;
function rocprim_block_histogram_int(d_out: PInteger; d_in: PInteger; num_items: size_t; block_size: Cardinal; bins: Cardinal; stream: hipStream_t): hipError_t; cdecl; external;

type
  TBlockLoadMethod = (
    block_load_direct,
    block_load_striped,
    block_load_vectorize,
    block_load_transpose,
    block_load_warp_transpose,
    block_load_default
  );

  TBlockStoreMethod = (
    block_store_direct,
    block_store_striped,
    block_store_vectorize,
    block_store_transpose,
    block_store_warp_transpose,
    block_store_default
  );

  TBlockAdjacentDifference = record
  end;

  TBlockDiscontinuity = record
  end;

  TBlockExchange = record
  end;

  TBlockHistogram = record
  end;

  TBlockLoad = record
  end;

  TBlockRadixRank = record
  end;

  TBlockRadixSort = record
  end;

  TBlockReduce = record
  end;

  TBlockScan = record
  end;

  TBlockSort = record
  end;

  TBlockStore = record
  end;

implementation

end.
