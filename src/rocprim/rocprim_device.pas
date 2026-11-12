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

{ rocPRIM device-wide algorithms. }
{ rocPRIM device-wide algorithms. }
unit rocprim_device;

{ Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
  Pascal translation/bindings: Copyright (c) 2024-2026 Eudox67
  ...
}

interface

uses
  rocprim_core, rocprim_types, hip_types, hip_error, hip_runtime;

// Reduce
procedure reduce_get_storage_size(input: PInteger; output: PInteger; size: Cardinal; op: Pointer; stream: Pointer; storage_size: PSizeInt); external;
procedure reduce(storage: Pointer; storage_size: SizeInt; input: PInteger; output: PInteger; size: Cardinal; op: Pointer; stream: Pointer); external;

// Scan
procedure inclusive_scan_get_storage_size(input: PInteger; output: PInteger; size: Cardinal; op: Pointer; stream: Pointer; storage_size: PSizeInt); external;
procedure inclusive_scan(storage: Pointer; storage_size: SizeInt; input: PInteger; output: PInteger; size: Cardinal; op: Pointer; stream: Pointer); external;
procedure exclusive_scan_get_storage_size(input: PInteger; output: PInteger; initial: Integer; size: Cardinal; op: Pointer; stream: Pointer; storage_size: PSizeInt); external;
procedure exclusive_scan(storage: Pointer; storage_size: SizeInt; input: PInteger; output: PInteger; initial: Integer; size: Cardinal; op: Pointer; stream: Pointer); external;

// Scan-by-key
procedure scan_by_key_get_storage_size(keys: PInteger; values: PInteger; size: Cardinal; op: Pointer; stream: Pointer; storage_size: PSizeInt); external;
procedure scan_by_key(storage: Pointer; storage_size: SizeInt; keys: PInteger; values: PInteger; output: PInteger; size: Cardinal; op: Pointer; stream: Pointer); external;

// Radix Sort
procedure radix_sort_keys_get_storage_size(keys: PInteger; size: Cardinal; stream: Pointer; storage_size: PSizeInt); external;
procedure radix_sort_keys(storage: Pointer; storage_size: SizeInt; keys: PInteger; size: Cardinal; stream: Pointer); external;
procedure radix_sort_pairs_get_storage_size(keys: PInteger; values: PPointer; size: Cardinal; stream: Pointer; storage_size: PSizeInt); external;
procedure radix_sort_pairs(storage: Pointer; storage_size: SizeInt; keys: PInteger; values: PPointer; size: Cardinal; stream: Pointer); external;

// Transform
procedure transform(input: PInteger; output: PInteger; size: Cardinal; op: Pointer; stream: Pointer); external;

// Partition
procedure partition_get_storage_size(input: PInteger; flags: PInteger; size: Cardinal; stream: Pointer; storage_size: PSizeInt); external;
procedure partition(storage: Pointer; storage_size: SizeInt; input: PInteger; flags: PInteger; output: PInteger; selected_count_output: PCardinal; size: Cardinal; stream: Pointer); external;

// Select
procedure select_get_storage_size(input: PInteger; flags: PInteger; size: Cardinal; stream: Pointer; storage_size: PSizeInt); external;
procedure select(storage: Pointer; storage_size: SizeInt; input: PInteger; flags: PInteger; output: PInteger; selected_count_output: PCardinal; size: Cardinal; stream: Pointer); external;

// Histogram
procedure histogram_get_storage_size(input: PInteger; size: Cardinal; num_levels: Integer; level_bins: PInteger; stream: Pointer; storage_size: PSizeInt); external;
procedure histogram(storage: Pointer; storage_size: SizeInt; input: PInteger; size: Cardinal; num_levels: Integer; level_bins: PInteger; output: PPointer; stream: Pointer); external;

// Reduce-by-key
procedure reduce_by_key_get_storage_size(keys: PInteger; values: PInteger; size: Cardinal; op: Pointer; stream: Pointer; storage_size: PSizeInt); external;
procedure reduce_by_key(storage: Pointer; storage_size: SizeInt; keys: PInteger; values: PInteger; size: Cardinal; unique_keys_output: PInteger; unique_values_output: PInteger; unique_count_output: PCardinal; op: Pointer; stream: Pointer); external;

// Adjacent Find
procedure adjacent_find(storage: Pointer; var storage_size: size_t; input: Pointer; output: PSizeInt; size: size_t; op: Pointer; stream: Pointer; debug_synchronous: Boolean); external;

// Merge In-place
procedure merge_inplace(storage: Pointer; var storage_size: size_t; data: Pointer; left_size: size_t; right_size: size_t; op: Pointer; stream: Pointer; debug_synchronous: Boolean); external;

// Find End (placeholder)
procedure find_end(storage: Pointer; var storage_size: size_t; haystack: Pointer; haystack_size: size_t; needle: Pointer; needle_size: size_t; output: PSizeInt; op: Pointer; stream: Pointer; debug_synchronous: Boolean); external;

// Find First Of (placeholder)
procedure find_first_of(storage: Pointer; var storage_size: size_t; haystack: Pointer; haystack_size: size_t; needles: Pointer; needles_size: size_t; output: PSizeInt; op: Pointer; stream: Pointer; debug_synchronous: Boolean); external;

// Search (placeholder)
procedure search(storage: Pointer; var storage_size: size_t; haystack: Pointer; haystack_size: size_t; needle: Pointer; needle_size: size_t; output: PSizeInt; op: Pointer; stream: Pointer; debug_synchronous: Boolean); external;

// Search N (placeholder)
procedure search_n(storage: Pointer; var storage_size: size_t; haystack: Pointer; haystack_size: size_t; count: size_t; value: Pointer; output: PSizeInt; op: Pointer; stream: Pointer; debug_synchronous: Boolean); external;

implementation

end.
