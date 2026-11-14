{ Pascal facade over the rocPRIM primitives implemented by the rocPRIM shim. }
{ Pascal facade over the rocPRIM primitives implemented by the rocPRIM shim. }
unit rocprim;

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
 
  This file is a Pascal translation of the rocprim.hpp C++ header file.
}

interface

uses
  rocprim_core,
  rocprim_types,
  rocprim_intrinsics,
  rocprim_iterator,
  rocprim_thread,
  rocprim_warp,
  rocprim_block,
  rocprim_device;

function version: Cardinal;

implementation

function version: Cardinal;
begin
  Result := ROCPRIM_VERSION;
end;

end.
