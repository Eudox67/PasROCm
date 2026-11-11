unit rocprim_types;

{$modeswitch advancedrecords}
{$PACKRECORDS C}

{$MODE OBJFPC}{$H+}
{$modeswitch advancedrecords}

{< Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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
>}

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

  // Corresponds to __int128_t
  int128_t = record
    low, high: QWord;
  end;

  // lane_mask_type is now based on arch, but for host-side FFI,
  // using the max size is the safest approach.
  lane_mask_type = QWord;

  { @brief Binary operator for reductions/scans (e.g., + for sum).
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

implementation

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
