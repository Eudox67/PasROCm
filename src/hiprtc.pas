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

unit hiprtc;

{$mode objfpc}{$H+}
{$linklib amdhip64}          // hipRTC always shipped in libamdhip64.so / .dll
{$linklib hiprtc}            // ROCm >= 6.x moved hipRTC into its own libhiprtc.so

interface

uses
  hip_runtime;                       // for hipDeviceptr_t if needed, plus UIntPtr alias

type
  hiprtcResult      = Integer;
  hiprtcJIT_option  = Integer;
  PhiprtcJIT_option = ^hiprtcJIT_option;
  hiprtcJITInputType= Integer;
  hiprtcProgram     = Pointer;
  hiprtcLinkState   = Pointer;

const
  { hiprtcResult }
  HIPRTC_SUCCESS                                      = 0;
  HIPRTC_ERROR_OUT_OF_MEMORY                          = 1;
  HIPRTC_ERROR_PROGRAM_CREATION_FAILURE               = 2;
  HIPRTC_ERROR_INVALID_INPUT                          = 3;
  HIPRTC_ERROR_INVALID_PROGRAM                        = 4;
  HIPRTC_ERROR_INVALID_OPTION                         = 5;
  HIPRTC_ERROR_COMPILATION                            = 6;
  HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE              = 7;
  HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION  = 8;
  HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION    = 9;
  HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID              = 10;
  HIPRTC_ERROR_INTERNAL_ERROR                         = 11;
  HIPRTC_ERROR_LINKING                                = 100;

  { hiprtcJIT_option }
  HIPRTC_JIT_MAX_REGISTERS                    = 0;
  HIPRTC_JIT_THREADS_PER_BLOCK                = 1;
  HIPRTC_JIT_WALL_TIME                        = 2;
  HIPRTC_JIT_INFO_LOG_BUFFER                  = 3;
  HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES       = 4;
  HIPRTC_JIT_ERROR_LOG_BUFFER                 = 5;
  HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES      = 6;
  HIPRTC_JIT_OPTIMIZATION_LEVEL               = 7;
  HIPRTC_JIT_TARGET_FROM_HIPCONTEXT           = 8;
  HIPRTC_JIT_TARGET                           = 9;
  HIPRTC_JIT_FALLBACK_STRATEGY                = 10;
  HIPRTC_JIT_GENERATE_DEBUG_INFO              = 11;
  HIPRTC_JIT_LOG_VERBOSE                      = 12;
  HIPRTC_JIT_GENERATE_LINE_INFO               = 13;
  HIPRTC_JIT_CACHE_MODE                       = 14;
  HIPRTC_JIT_NEW_SM3X_OPT                     = 15;
  HIPRTC_JIT_FAST_COMPILE                     = 16;
  HIPRTC_JIT_GLOBAL_SYMBOL_NAMES              = 17;
  HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS            = 18;
  HIPRTC_JIT_GLOBAL_SYMBOL_COUNT              = 19;
  HIPRTC_JIT_LTO                              = 20;
  HIPRTC_JIT_FTZ                              = 21;
  HIPRTC_JIT_PREC_DIV                         = 22;
  HIPRTC_JIT_PREC_SQRT                        = 23;
  HIPRTC_JIT_FMA                              = 24;
  HIPRTC_JIT_NUM_OPTIONS                      = 25;
  HIPRTC_JIT_IR_TO_ISA_OPT_EXT                = 10000;
  HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT          = 10001;

  { hiprtcJITInputType }
  HIPRTC_JIT_INPUT_CUBIN                           = 0;
  HIPRTC_JIT_INPUT_PTX                             = 1;
  HIPRTC_JIT_INPUT_FATBINARY                       = 2;
  HIPRTC_JIT_INPUT_OBJECT                          = 3;
  HIPRTC_JIT_INPUT_LIBRARY                         = 4;
  HIPRTC_JIT_INPUT_NVVM                            = 5;
  HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES                = 6;
  HIPRTC_JIT_INPUT_LLVM_BITCODE                    = 100;
  HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE            = 101;
  HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE= 102;

{ ---------- Version / helpers ---------- }
function hiprtcGetErrorString(result: hiprtcResult): PChar; cdecl; external;
function hiprtcVersion(var major, minor: Integer): hiprtcResult; cdecl; external;

{ ---------- Program life-cycle ---------- }
function hiprtcCreateProgram(var prog: hiprtcProgram;
                             const src, name: PChar;
                             numHeaders: Int32;
                             const headers, includeNames: PPChar): hiprtcResult; cdecl; external;
function hiprtcDestroyProgram(var prog: hiprtcProgram): hiprtcResult; cdecl; external;
function hiprtcCompileProgram(prog: hiprtcProgram;
                              numOptions: Int32;
                              const options: PPChar): hiprtcResult; cdecl; external;

{ ---------- Query compiled result ---------- }
function hiprtcGetCodeSize(prog: hiprtcProgram; var codeSize: UIntPtr): hiprtcResult; cdecl; external;
function hiprtcGetCode(prog: hiprtcProgram; code: Pointer): hiprtcResult; cdecl; external;
function hiprtcGetBitcodeSize(prog: hiprtcProgram; var bitcodeSize: UIntPtr): hiprtcResult; cdecl; external;
function hiprtcGetBitcode(prog: hiprtcProgram; bitcode: Pointer): hiprtcResult; cdecl; external;

{ ---------- Log / name expression helpers ---------- }
function hiprtcGetProgramLogSize(prog: hiprtcProgram; var logSize: UIntPtr): hiprtcResult; cdecl; external;
function hiprtcGetProgramLog(prog: hiprtcProgram; log: PChar): hiprtcResult; cdecl; external;
function hiprtcAddNameExpression(prog: hiprtcProgram; const nameExpression: PChar): hiprtcResult; cdecl; external;
function hiprtcGetLoweredName(prog: hiprtcProgram; const nameExpression: PChar; var loweredName: PChar): hiprtcResult; cdecl; external;

{ ---------- Link-time JIT (LTO) ---------- }
function hiprtcLinkCreate(numOptions: UInt32;
                          const options: PhiprtcJIT_option;
                          const optionVals: PPointer;
                          var linkState: hiprtcLinkState): hiprtcResult; cdecl; external;
function hiprtcLinkAddFile(linkState: hiprtcLinkState;
                           inputType: hiprtcJITInputType;
                           const filePath: PChar;
                           numOptions: UInt32;
                           const options: PhiprtcJIT_option;
                           const optionVals: PPointer): hiprtcResult; cdecl; external;
function hiprtcLinkAddData(linkState: hiprtcLinkState;
                           inputType: hiprtcJITInputType;
                           const image: Pointer;
                           imageSize: UIntPtr;
                           const name: PChar;
                           numOptions: UInt32;
                           const options: PhiprtcJIT_option;
                           const optionVals: PPointer): hiprtcResult; cdecl; external;
function hiprtcLinkComplete(linkState: hiprtcLinkState;
                            var binOut: Pointer;
                            var sizeOut: UIntPtr): hiprtcResult; cdecl; external;
function hiprtcLinkDestroy(linkState: hiprtcLinkState): hiprtcResult; cdecl; external;

implementation
end.
