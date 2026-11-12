{ rocPRIM iterator adaptors. }
{ rocPRIM iterator adaptors. }
unit rocprim_iterator;

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
  - rocprim/iterator.hpp
  - rocprim/iterator/arg_index_iterator.hpp
  - rocprim/iterator/constant_iterator.hpp
  - rocprim/iterator/counting_iterator.hpp
  - rocprim/iterator/discard_iterator.hpp
  - rocprim/iterator/predicate_iterator.hpp
  - rocprim/iterator/reverse_iterator.hpp
  - rocprim/iterator/texture_cache_iterator.hpp
  - rocprim/iterator/transform_iterator.hpp
  - rocprim/iterator/zip_iterator.hpp
}

interface

uses
  rocprim_core, rocprim_types;

type
  TArgIndexIterator = record
    // Simplified non-generic version
    iterator: Pointer;
    offset: PtrInt;
  end;

  TConstantIterator = record
    // Simplified non-generic version
    value: Pointer;
    index: PtrInt;
  end;

  TCountingIterator = record
    // Simplified non-generic version for Integer
    value: Integer;
  end;

  TDiscardIterator = record
    index: PtrInt;
  end;

  TPredicateIterator = record
    // Placeholder for a more complex structure
    data_iterator: Pointer;
    predicate_iterator: Pointer;
    // Predicate function would be handled differently in Pascal
  end;

  TReverseIterator = record
    // Simplified non-generic version
    source_iterator: Pointer;
  end;

  TTextureCacheIterator = record
    // Highly hardware-specific, requires special implementation
    ptr: Pointer;
    texture_offset: PtrInt;
    texture_object: QWord; // Placeholder for hipTextureObject_t
  end;

  TTransformIterator = record
    // Placeholder for a more complex structure
    iterator: Pointer;
    // Transform function would be handled differently in Pascal
  end;

  TZipIterator = record
    // Placeholder for a more complex structure
    iterator_tuple: TTuple;
  end;

function make_counting_iterator(value: Integer): TCountingIterator;
function make_counting_iterator(value: Int64): TCountingIterator;

function make_constant_iterator(value: Integer): TConstantIterator;
function make_constant_iterator(value: Int64): TConstantIterator;

function make_transform_iterator(it: Pointer; op: Pointer): TTransformIterator;
function make_zip_iterator(iterators: TTuple): TZipIterator;
function make_arg_index_iterator(it: Pointer): TArgIndexIterator;

implementation

function make_counting_iterator(value: Integer): TCountingIterator;
begin
  // Stub
  Result.value := value;
end;

function make_counting_iterator(value: Int64): TCountingIterator;
begin
  // Stub
  Result.value := Integer(value); // Truncation warning, but just a stub
end;

function make_constant_iterator(value: Integer): TConstantIterator;
begin
  // Stub
end;

function make_constant_iterator(value: Int64): TConstantIterator;
begin
  // Stub
end;

function make_transform_iterator(it: Pointer; op: Pointer): TTransformIterator;
begin
  // Stub
end;

function make_zip_iterator(iterators: TTuple): TZipIterator;
begin
  // Stub
end;

function make_arg_index_iterator(it: Pointer): TArgIndexIterator;
begin
  // Stub
end;

end.