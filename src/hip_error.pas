{*
 * hip_error.pas
 *
 * Copyright (c) 2023-2026 Eudox67 - Free Pascal wrappers for HIP.
 * This is an unofficial project and is not affiliated with or endorsed by AMD.
 *
 * Derived from HIP headers, which are:
 * Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

{ @abstract(This unit provides error codes and related functions for HIP, combining error-related parts from hip_runtime_api.h and hip_rtc.h.)
  @bold(Group: HIP Error Handling)

  @note(Errors are defined as Int32 types with constant values for C compatibility. Use the external functions to get string descriptions.
  Platform-specific library linking is handled via conditionals @(AMD: libamdhip64.so; NVIDIA: libcudart.so—adjust as needed@).)

  See also: hip_runtime_api.pas for API functions that return these errors.

}
{ @bold(Group: Main HIP Error Codes from hip_runtime_api.h) }
unit hip_error;

{$MODE OBJFPC}{$H+}
{$PACKRECORDS C}  // Ensure C-compatible packing (though not needed for enums here)

interface

uses
  hip_common;  // For platform macros and base definitions



type
  { HIP error type (Int32-based for C enum compatibility). }
  THipError  = Int32;
  hipError_t = THipError;

const
  { Success (no error). }
  hipSuccess: Int32 = 0;

  { Invalid value. }
  hipErrorInvalidValue: Int32 = 1;

  { Out of memory. }
  hipErrorOutOfMemory: Int32 = 2;

  { Not initialized. }
  hipErrorNotInitialized: Int32 = 3;

  { Deinitialized. }
  hipErrorDeinitialized: Int32 = 4;

  { Profiler disabled. }
  hipErrorProfilerDisabled: Int32 = 5;

  { Profiler not initialized. }
  hipErrorProfilerNotInitialized: Int32 = 6;

  { Profiler already started. }
  hipErrorProfilerAlreadyStarted: Int32 = 7;

  { Profiler already stopped. }
  hipErrorProfilerAlreadyStopped: Int32 = 8;

  { Invalid configuration. }
  hipErrorInvalidConfiguration: Int32 = 9;

  { Invalid pitch value. }
  hipErrorInvalidPitchValue: Int32 = 12;

  { Invalid symbol. }
  hipErrorInvalidSymbol: Int32 = 13;

  { Invalid device pointer. }
  hipErrorInvalidDevicePointer: Int32 = 17;

  { Invalid texture. }
  hipErrorInvalidTexture: Int32 = 18;

  { Invalid texture binding. }
  hipErrorInvalidTextureBinding: Int32 = 19;

  { Invalid channel descriptor. }
  hipErrorInvalidChannelDescriptor: Int32 = 20;

  { Invalid memcpy direction. }
  hipErrorInvalidMemcpyDirection: Int32 = 21;

  { Address of buffer is invalid. }
  hipErrorAddressOfConstant: Int32 = 22;

  { Texture fetch is not supported. }
  hipErrorTextureFetchFailed: Int32 = 23;

  { Texture is not bound. }
  hipErrorTextureNotBound: Int32 = 24;

  { Synchronization failed. }
  hipErrorSynchronizationError: Int32 = 25;

  { Invalid filter setting. }
  hipErrorInvalidFilterSetting: Int32 = 26;

  { Invalid norm setting. }
  hipErrorInvalidNormSetting: Int32 = 27;

  { Mixed size or type array. }
  hipErrorMixedArrayType: Int32 = 28;

  { Unknown error. }
  hipErrorUnknown: Int32 = 30;

  { Not yet implemented feature. }
  hipErrorNotYetImplemented: Int32 = 31;

  { Memory access fault. }
  hipErrorMemoryValueTooLarge: Int32 = 32;

  { Insufficient driver. }
  hipErrorInsufficientDriver: Int32 = 35;

  { Set on load. }
  hipErrorSetOnActiveProcess: Int32 = 36;

  { No device available. }
  hipErrorNoDevice: Int32 = 38;

  { ECC uncorrectable. }
  hipErrorECCUncorrectable: Int32 = 39;

  { Unsupported limit. }
  hipErrorUnsupportedLimit: Int32 = 40;

  { Peer access unsupported. }
  hipErrorPeerAccessUnsupported: Int32 = 41;

  { Invalid device. }
  hipErrorInvalidDevice: Int32 = 43;

  { Invalid image. }
  hipErrorInvalidImage: Int32 = 44;

  { Invalid context. }
  hipErrorInvalidContext: Int32 = 45;

  { Context already current. }
  hipErrorContextAlreadyCurrent: Int32 = 46;

  { Map failed. }
  hipErrorMapFailed: Int32 = 47;

  { Unmap failed. }
  hipErrorUnmapFailed: Int32 = 48;

  { Array is mapped. }
  hipErrorArrayIsMapped: Int32 = 49;

  { Already mapped. }
  hipErrorAlreadyMapped: Int32 = 50;

  { No binary for GPU. }
  hipErrorNoBinaryForGpu: Int32 = 51;

  { Already acquired. }
  hipErrorAlreadyAcquired: Int32 = 52;

  { Not mapped. }
  hipErrorNotMapped: Int32 = 53;

  { Mapped resource not available for access as an array. }
  hipErrorNotMappedAsArray: Int32 = 54;

  { Mapped resource not available for access as pointer. }
  hipErrorNotMappedAsPointer: Int32 = 55;

  { Uncorrectable ECC error detected. }
  hipErrorECCNotCorrectable: Int32 = 56;

  { Query is not supported on the device. }
  hipErrorNotSupported: Int32 = 57;

  { Device is not ready. }
  hipErrorDeviceNotReady: Int32 = 58;

  { Stream capture failed. }
  hipErrorStreamCaptureUnsupported: Int32 = 900;

  { Stream capture invalid. }
  hipErrorStreamCaptureInvalidated: Int32 = 902;

  { Stream capture merge error. }
  hipErrorStreamCaptureMerge: Int32 = 903;

  { Stream capture unjoined. }
  hipErrorStreamCaptureUnjoined: Int32 = 904;

  { Stream capture unmatched. }
  hipErrorStreamCaptureUnmatched: Int32 = 905;

  { Stream capture isolation error. }
  hipErrorStreamCaptureIsolation: Int32 = 906;

  { Stream capture implicit error. }
  hipErrorStreamCaptureImplicit: Int32 = 907;

  { Captured event overlap. }
  hipErrorCapturedEvent: Int32 = 908;

  { Stream capture wrong thread. }
  hipErrorStreamCaptureWrongThread: Int32 = 909;

  { Timeout occurred. }
  hipErrorTimeout: Int32 = 910;

  { Unknown error (fallback). }
  hipErrorUnknownError: Int32 = 999;

  // ... (Add any additional errors from hip_runtime_api.h if missing; this covers the main ones)


{ @bold(Group: RTC Error Codes from hip_rtc.h) }

type
  { HIP RTC error type (Int32-based for C enum compatibility). }
  THiprtcResult = Int32;

const
  { Success. }
  HIPRTC_SUCCESS: Int32 = 0;

  { Out of memory. }
  HIPRTC_ERROR_OUT_OF_MEMORY: Int32 = 1;

  { Program creation failure. }
  HIPRTC_ERROR_PROGRAM_CREATION_FAILURE: Int32 = 2;

  { Invalid input. }
  HIPRTC_ERROR_INVALID_INPUT: Int32 = 3;

  { Invalid program. }
  HIPRTC_ERROR_INVALID_PROGRAM: Int32 = 4;

  { Invalid option. }
  HIPRTC_ERROR_INVALID_OPTION: Int32 = 5;

  { Compilation error. }
  HIPRTC_ERROR_COMPILATION: Int32 = 6;

  { Builtin operation failure. }
  HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE: Int32 = 7;

  { No name expressions after compilation. }
  HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION: Int32 = 8;

  { No lowered names before compilation. }
  HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION: Int32 = 9;

  { Name expression not valid. }
  HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID: Int32 = 10;

  { Internal error. }
  HIPRTC_ERROR_INTERNAL_ERROR: Int32 = 11;

  { Linking error. }
  HIPRTC_ERROR_LINKING: Int32 = 100;


{ @bold(Group: Error String Functions) }

// External function to get error string for HIP errors
{ Returns a string description of the HIP error code.
  @param(error The error code to describe.)
  @returns(A null-terminated string describing the error.) }
function hipGetErrorString(error: THipError): PChar;
  external HIP_LIB_NAME name PREFIX + 'GetErrorString';

// External function to get error name (more detailed)
{ Returns the name of the HIP error code as a string.
  @param(error The error code to name.)
  @returns(A null-terminated string naming the error.) }
function hipGetErrorName(error: THipError): PChar;
  external HIP_LIB_NAME name PREFIX + 'GetErrorName';
  
// External function for RTC error string
{ Returns a string description of the HIP RTC error code.
  @param(result The RTC error code to describe.)
  @returns(A null-terminated string describing the error.) }
function hiprtcGetErrorString(&result: THiprtcResult): PChar;
  external HIP_LIB_NAME name PREFIX + 'rtcGetErrorString';


implementation

end.
