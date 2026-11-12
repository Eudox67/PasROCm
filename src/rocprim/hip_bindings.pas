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

{ Low-level HIP extern declarations used by the wrappers. }
{ Low-level HIP extern declarations used by the wrappers. }
unit hip_bindings;

interface

uses
  ctypes;

const
  // On Linux, the HIP runtime is typically libamdhip64.so
  libhip = 'libamdhip64.so';

  // HIP API constants
  hipSuccess = 0;

  // Enum for hipMemcpyKind
  hipMemcpyHostToHost = 0;
  hipMemcpyHostToDevice = 1;
  hipMemcpyDeviceToHost = 2;
  hipMemcpyDeviceToDevice = 3;
  hipMemcpyDefault = 4;

type
  hipError_t = cint;
  hipStream_t = Pointer;
  hipMemcpyKind = cint;

// --- HIP Memory Management --- //

function hipMalloc(ptr: PPointer; size: csize_t): hipError_t; cdecl; external libhip;

function hipFree(ptr: Pointer): hipError_t; cdecl; external libhip;

// --- HIP Data Transfer --- //

function hipMemcpy(dst: Pointer; src: Pointer; sizeBytes: csize_t; kind: hipMemcpyKind): hipError_t; cdecl; external libhip;

// --- HIP Stream Management --- //

function hipStreamCreate(stream: PPointer): hipError_t; cdecl; external libhip;

function hipStreamDestroy(stream: Pointer): hipError_t; cdecl; external libhip;

implementation

end.