unit rocprim_core;

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
  - rocprim/config.hpp
  - rocprim/rocprim_version.hpp
  - rocprim/functional.hpp
  - rocprim/type_traits.hpp
>}

interface

const
  ROCPRIM_VERSION = 70002;
  ROCPRIM_VERSION_MAJOR = 7;
  ROCPRIM_VERSION_MINOR = 0;
  ROCPRIM_VERSION_PATCH = 2;

  ROCPRIM_DEFAULT_MAX_BLOCK_SIZE = 256;
  ROCPRIM_DEFAULT_MIN_WARPS_PER_EU = 1;
  ROCPRIM_NAVI = 1;
  ROCPRIM_ARCH_90a = 910;
  ROCPRIM_WARP_SIZE_32 = 32;
  ROCPRIM_WARP_SIZE_64 = 64;
  ROCPRIM_MAX_WARP_SIZE = ROCPRIM_WARP_SIZE_64;
  ROCPRIM_QUAD_SIZE = 4;
  ROCPRIM_GRID_SIZE_LIMIT = High(LongWord);
  
implementation

end.
