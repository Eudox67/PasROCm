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

unit hip_error;

{$MODE OBJFPC}{$H+}
{$PACKRECORDS C}  // Ensure C-compatible packing (though not needed for enums here)

interface

uses
  hip_common;  // For platform macros and base definitions

{ @abstract(This unit provides error codes and related functions for HIP, combining error-related parts from hip_runtime_api.h and hip_rtc.h.)
  @groupbegin(HIP Error Handling)

  @note(Errors are defined as Int32 types with constant values for C compatibility. Use the external functions to get string descriptions.
  Platform-specific library linking is handled via conditionals (AMD: libamdhip64.so; NVIDIA: libcudart.so—adjust as needed).)

  @see(hip_runtime_api.pas for API functions that return these errors.)
  @groupend
}

{ @groupbegin(Main HIP Error Codes from hip_runtime_api.h) }

type
  { @brief HIP error type (Int32-based for C enum compatibility). }
  THipError  = Int32;
  hipError_t = THipError;

const
  { @brief Success (no error). }
  hipSuccess: Int32 = 0;

  { @brief Invalid value. }
  hipErrorInvalidValue: Int32 = 1;

  { @brief Out of memory. }
  hipErrorOutOfMemory: Int32 = 2;

  { @brief Not initialized. }
  hipErrorNotInitialized: Int32 = 3;

  { @brief Deinitialized. }
  hipErrorDeinitialized: Int32 = 4;

  { @brief Profiler disabled. }
  hipErrorProfilerDisabled: Int32 = 5;

  { @brief Profiler not initialized. }
  hipErrorProfilerNotInitialized: Int32 = 6;

  { @brief Profiler already started. }
  hipErrorProfilerAlreadyStarted: Int32 = 7;

  { @brief Profiler already stopped. }
  hipErrorProfilerAlreadyStopped: Int32 = 8;

  { @brief Invalid configuration. }
  hipErrorInvalidConfiguration: Int32 = 9;

  { @brief Invalid pitch value. }
  hipErrorInvalidPitchValue: Int32 = 12;

  { @brief Invalid symbol. }
  hipErrorInvalidSymbol: Int32 = 13;

  { @brief Invalid device pointer. }
  hipErrorInvalidDevicePointer: Int32 = 17;

  { @brief Invalid texture. }
  hipErrorInvalidTexture: Int32 = 18;

  { @brief Invalid texture binding. }
  hipErrorInvalidTextureBinding: Int32 = 19;

  { @brief Invalid channel descriptor. }
  hipErrorInvalidChannelDescriptor: Int32 = 20;

  { @brief Invalid memcpy direction. }
  hipErrorInvalidMemcpyDirection: Int32 = 21;

  { @brief Address of buffer is invalid. }
  hipErrorAddressOfConstant: Int32 = 22;

  { @brief Texture fetch is not supported. }
  hipErrorTextureFetchFailed: Int32 = 23;

  { @brief Texture is not bound. }
  hipErrorTextureNotBound: Int32 = 24;

  { @brief Synchronization failed. }
  hipErrorSynchronizationError: Int32 = 25;

  { @brief Invalid filter setting. }
  hipErrorInvalidFilterSetting: Int32 = 26;

  { @brief Invalid norm setting. }
  hipErrorInvalidNormSetting: Int32 = 27;

  { @brief Mixed size or type array. }
  hipErrorMixedArrayType: Int32 = 28;

  { @brief Unknown error. }
  hipErrorUnknown: Int32 = 30;

  { @brief Not yet implemented feature. }
  hipErrorNotYetImplemented: Int32 = 31;

  { @brief Memory access fault. }
  hipErrorMemoryValueTooLarge: Int32 = 32;

  { @brief Insufficient driver. }
  hipErrorInsufficientDriver: Int32 = 35;

  { @brief Set on load. }
  hipErrorSetOnActiveProcess: Int32 = 36;

  { @brief No device available. }
  hipErrorNoDevice: Int32 = 38;

  { @brief ECC uncorrectable. }
  hipErrorECCUncorrectable: Int32 = 39;

  { @brief Unsupported limit. }
  hipErrorUnsupportedLimit: Int32 = 40;

  { @brief Peer access unsupported. }
  hipErrorPeerAccessUnsupported: Int32 = 41;

  { @brief Invalid device. }
  hipErrorInvalidDevice: Int32 = 43;

  { @brief Invalid image. }
  hipErrorInvalidImage: Int32 = 44;

  { @brief Invalid context. }
  hipErrorInvalidContext: Int32 = 45;

  { @brief Context already current. }
  hipErrorContextAlreadyCurrent: Int32 = 46;

  { @brief Map failed. }
  hipErrorMapFailed: Int32 = 47;

  { @brief Unmap failed. }
  hipErrorUnmapFailed: Int32 = 48;

  { @brief Array is mapped. }
  hipErrorArrayIsMapped: Int32 = 49;

  { @brief Already mapped. }
  hipErrorAlreadyMapped: Int32 = 50;

  { @brief No binary for GPU. }
  hipErrorNoBinaryForGpu: Int32 = 51;

  { @brief Already acquired. }
  hipErrorAlreadyAcquired: Int32 = 52;

  { @brief Not mapped. }
  hipErrorNotMapped: Int32 = 53;

  { @brief Mapped resource not available for access as an array. }
  hipErrorNotMappedAsArray: Int32 = 54;

  { @brief Mapped resource not available for access as pointer. }
  hipErrorNotMappedAsPointer: Int32 = 55;

  { @brief Uncorrectable ECC error detected. }
  hipErrorECCNotCorrectable: Int32 = 56;

  { @brief Query is not supported on the device. }
  hipErrorNotSupported: Int32 = 57;

  { @brief Device is not ready. }
  hipErrorDeviceNotReady: Int32 = 58;

  { @brief Stream capture failed. }
  hipErrorStreamCaptureUnsupported: Int32 = 900;

  { @brief Stream capture invalid. }
  hipErrorStreamCaptureInvalidated: Int32 = 902;

  { @brief Stream capture merge error. }
  hipErrorStreamCaptureMerge: Int32 = 903;

  { @brief Stream capture unjoined. }
  hipErrorStreamCaptureUnjoined: Int32 = 904;

  { @brief Stream capture unmatched. }
  hipErrorStreamCaptureUnmatched: Int32 = 905;

  { @brief Stream capture isolation error. }
  hipErrorStreamCaptureIsolation: Int32 = 906;

  { @brief Stream capture implicit error. }
  hipErrorStreamCaptureImplicit: Int32 = 907;

  { @brief Captured event overlap. }
  hipErrorCapturedEvent: Int32 = 908;

  { @brief Stream capture wrong thread. }
  hipErrorStreamCaptureWrongThread: Int32 = 909;

  { @brief Timeout occurred. }
  hipErrorTimeout: Int32 = 910;

  { @brief Unknown error (fallback). }
  hipErrorUnknownError: Int32 = 999;

  // ... (Add any additional errors from hip_runtime_api.h if missing; this covers the main ones)

{ @groupend }

{ @groupbegin(RTC Error Codes from hip_rtc.h) }

type
  { @brief HIP RTC error type (Int32-based for C enum compatibility). }
  THiprtcResult = Int32;

const
  { @brief Success. }
  HIPRTC_SUCCESS: Int32 = 0;

  { @brief Out of memory. }
  HIPRTC_ERROR_OUT_OF_MEMORY: Int32 = 1;

  { @brief Program creation failure. }
  HIPRTC_ERROR_PROGRAM_CREATION_FAILURE: Int32 = 2;

  { @brief Invalid input. }
  HIPRTC_ERROR_INVALID_INPUT: Int32 = 3;

  { @brief Invalid program. }
  HIPRTC_ERROR_INVALID_PROGRAM: Int32 = 4;

  { @brief Invalid option. }
  HIPRTC_ERROR_INVALID_OPTION: Int32 = 5;

  { @brief Compilation error. }
  HIPRTC_ERROR_COMPILATION: Int32 = 6;

  { @brief Builtin operation failure. }
  HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE: Int32 = 7;

  { @brief No name expressions after compilation. }
  HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION: Int32 = 8;

  { @brief No lowered names before compilation. }
  HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION: Int32 = 9;

  { @brief Name expression not valid. }
  HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID: Int32 = 10;

  { @brief Internal error. }
  HIPRTC_ERROR_INTERNAL_ERROR: Int32 = 11;

  { @brief Linking error. }
  HIPRTC_ERROR_LINKING: Int32 = 100;

{ @groupend }

{ @groupbegin(Error String Functions) }

// External function to get error string for HIP errors
{ @brief Returns a string description of the HIP error code.
  @param(error The error code to describe.)
  @returns(A null-terminated string describing the error.) }
function hipGetErrorString(error: THipError): PChar;
  external HIP_LIB_NAME name PREFIX + 'GetErrorString';

// External function to get error name (more detailed)
{ @brief Returns the name of the HIP error code as a string.
  @param(error The error code to name.)
  @returns(A null-terminated string naming the error.) }
function hipGetErrorName(error: THipError): PChar;
  external HIP_LIB_NAME name PREFIX + 'GetErrorName';
  
// External function for RTC error string
{ @brief Returns a string description of the HIP RTC error code.
  @param(result The RTC error code to describe.)
  @returns(A null-terminated string describing the error.) }
function hiprtcGetErrorString(&result: THiprtcResult): PChar;
  external HIP_LIB_NAME name PREFIX + 'rtcGetErrorString';

{ @groupend }

implementation

end.
