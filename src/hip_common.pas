{*
 * hip_common.pas
 *
 * Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.
 * Pascal translation/bindings: Copyright (c) 2024-2026 Eudox67
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

{ @abstract(This unit provides base macros, platform detection, version information, and math constants for HIP.)
  @bold(Group: Base HIP Definitions)

  @note(This unit combines content from hip_common.h, hip_version.h, and hip_math_constants.h.
  It defines platform macros, version constants, and math constants. Platform detection is handled via FPC directives.
  Define exactly one of HIP_PLATFORM_AMD or HIP_PLATFORM_NVIDIA externally @(e.g., via compiler option -dHIP_PLATFORM_AMD@)
  to match HIP's requirement, or an error will be raised at compile time.)

  See also: hip_types.pas for types that build on these definitions.

}
{ Base macros, platform detection, version and math constants for HIP. }
unit hip_common;

{$MODE OBJFPC}{$H+}
{$PACKRECORDS C}  // Ensure C-compatible packing for any future structs
{$ALIGN 8}        // Ensure 64-bit alignment for compatibility with HIP types

interface


// Platform Detection (translated from hip_common.h)
// Auto-define based on common FPC symbols or environment; can be overridden.
// For AMD/ROCm (clang-based), assume if not NVIDIA.
{$IFNDEF HIP_PLATFORM_NVIDIA}
  {$DEFINE HIP_PLATFORM_AMD}  // Default to AMD if not specified
{$ENDIF}

// Error if both or neither are defined (matches HIP's #error)
{$IF DEFINED(HIP_PLATFORM_AMD) AND DEFINED(HIP_PLATFORM_NVIDIA)}
  {$ERROR "Must define exactly one of HIP_PLATFORM_AMD or HIP_PLATFORM_NVIDIA"}
{$ELSEIF NOT DEFINED(HIP_PLATFORM_AMD) AND NOT DEFINED(HIP_PLATFORM_NVIDIA)}
  {$ERROR "Must define exactly one of HIP_PLATFORM_AMD or HIP_PLATFORM_NVIDIA"}
{$ENDIF}

const
  // Visibility Macros (from hip_common.h; in Pascal, these are for info only, as FPC handles visibility differently)
  { Visibility attribute for public HIP APIs (equivalent to __attribute__((visibility("default"))) in C). }
  HIP_PUBLIC_API = '';  // Placeholder; use for exported functions if building a shared lib

  { Internal exported API visibility (similar to public). }
  HIP_INTERNAL_EXPORTED_API = '';

  // Device Compile Check (from hip_common.h; set to 0 for host-side, as FPC is host-only)
  { Indicates if compiling in device mode (1) or host mode (0). In FPC, this is always 0 unless targeting GPU. }
  __HIP_DEVICE_COMPILE__ = 0;
  
const
  {
    Defines the name of the HIP runtime shared library for the target platform.
    
    This constant is used in all `external` declarations to ensure that the
    Pascal wrappers link to the correct library. It is set conditionally
    based on the target operating system.
  }
  // Library name based on platform and OS
  {$IF DEFINED(HIP_PLATFORM_AMD) AND DEFINED(LINUX)}
    HIP_LIB_NAME = 'libamdhip64.so';    
  {$ELSEIF DEFINED(HIP_PLATFORM_AMD) AND DEFINED(WINDOWS)}
    HIP_LIB_NAME = 'amdhip64.dll'
  {$ELSEIF DEFINED(HIP_PLATFORM_NVIDIA) AND DEFINED(LINUX)}
    HIP_LIB_NAME = 'libcudart.so';
  {$ELSEIF DEFINED(HIP_PLATFORM_NVIDIA) AND DEFINED(WINDOWS)} 
    HIP_LIB_NAME = 'cudart64_130.dll';
  {$ENDIF}
  
  // Platform-specific macros (matching hip_common.h)
  {$IFDEF HIP_PLATFORM_AMD}
    HIP_PLATFORM_AMD = TRUE;
    HIP_PLATFORM_NVIDIA = FALSE;
    PREFIX = 'hip';
  {$ELSE}
    HIP_PLATFORM_AMD = FALSE;
    HIP_PLATFORM_NVIDIA = TRUE;
    PREFIX = 'cu';
  {$ENDIF}
  
  
    
const
  // Architecture Flags (from hip_common.h; defaults to 0 for host-side)
  { @bold(Group: Architecture Feature Flags)
    32-bit atomic support flags (0 = unsupported on host). }
  __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__ = 0;
  __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__ = 0;
  __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__ = 0;
  __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__ = 0;
  __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__ = 0;

const
  { 64-bit atomic support flags. }
  __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__ = 0;
  __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__ = 0;

const
  { Double-precision support. }
  __HIP_ARCH_HAS_DOUBLES__ = 0;

const
  { Warp operations support. }
  __HIP_ARCH_HAS_WARP_VOTE__ = 0;
  __HIP_ARCH_HAS_WARP_BALLOT__ = 0;
  __HIP_ARCH_HAS_WARP_SHUFFLE__ = 0;
  __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__ = 0;

const
  { Synchronization support. }
  __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__ = 0;
  __HIP_ARCH_HAS_SYNC_THREAD_EXT__ = 0;

const
  { Miscellaneous architecture flags. }
  __HIP_ARCH_HAS_SURFACE_FUNCS__ = 0;
  __HIP_ARCH_HAS_3DGRID__ = 0;
  __HIP_ARCH_HAS_DYNAMIC_PARALLEL__ = 0;

const
  // Version Constants (from hip_version.h)
  { @bold(Group: HIP Version Information)
    Major version of the HIP runtime. }
  HIP_VERSION_MAJOR = 6;

  { Minor version of the HIP runtime. }
  HIP_VERSION_MINOR = 3;

  { Patch version of the HIP runtime. }
  HIP_VERSION_PATCH = 42134;

  { Git hash for the HIP version. }
  HIP_VERSION_GITHASH = 'a9a80e791';

  { Build ID for the HIP version. }
  HIP_VERSION_BUILD_ID = 0;

  { Build name for the HIP version. }
  HIP_VERSION_BUILD_NAME = '';

  { Combined HIP version (major * 10000000 + minor * 100000 + patch). }
  HIP_VERSION = (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH);

  { Flag indicating if HIP has precompiled headers support (1 = yes). }
  __HIP_HAS_GET_PCH = 1;


  // Other common macros from hip_common.h (translated to constants where possible)
  { Equivalent to __HIPCC__ in HIP (set to 1 if in HIP compiler mode; here 0 for FPC host). }
  __HIPCC__ = 0;  // FPC is host-side; adjust if you add GPU support  

implementation

end.
