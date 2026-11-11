{*
 * hip_runtime_api.pas
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

unit hip_runtime;

{$MODE OBJFPC}{$H+}
{$PACKRECORDS C}  // Ensure C-compatible packing for structs/records
{$linklib amdhip64}

interface

uses
  hip_common,  // For HIP_LIB_NAME, platform macros, and base definitions
  hip_types,   // Foundational unit for all data types, including core types like THipChannelFormatDesc
  hip_error;   // For THipError and error handling

{
  @abstract(This unit provides the main host API signatures for HIP, translated from 
  hip_runtime_api.h and hip_runtime.h)
  @groupbegin(HIP Runtime)

  @note(This is a complete translation of the public API. Handles like streams and
  events are defined as opaque pointers for type-safety. All external functions link
  against the HIP runtime library defined by the HIP_LIB_NAME constant.)

  @see(hip_types.pas for core types; hip_error.pas for error codes.)
  @groupend
}

//Type
  //hipError_t      = Int32;


{ @groupbegin(Flags and Constants) }

const
  HIP_SUCCESS = 0;
  
  { @brief Default stream creation flags. }
  hipStreamDefault = $00;

  { @brief Stream does not implicitly synchronize with null stream. }
  hipStreamNonBlocking = $01;

  { @brief Default event flags. }
  hipEventDefault = $0;

  { @brief Waiting will yield CPU. }
  hipEventBlockingSync = $1;

  { @brief Disable event's timing. }
  hipEventDisableTiming = $2;

  { @brief Event can support IPC. }
  hipEventInterprocess = $4;

  { @brief Disable system fence for event. }
  hipEventDisableSystemFence = $20000000;

  { @brief Use device-scope release for event. }
  hipEventReleaseToDevice = $40000000;

  { @brief Use system-scope release for event. }
  hipEventReleaseToSystem = $80000000;

  { @brief Default host malloc flags. }
  hipHostMallocDefault = $0;
  hipHostAllocDefault = $0;

  { @brief Memory is portable. }
  hipHostMallocPortable = $1;
  hipHostAllocPortable = $1;

  { @brief Memory is mapped. }
  hipHostMallocMapped = $2;
  hipHostAllocMapped = $2;

  { @brief Memory is write-combined. }
  hipHostMallocWriteCombined = $4;
  hipHostAllocWriteCombined = $4;

  { @brief Host memory allocation follows NUMA policy (Linux-specific). }
  hipHostMallocNumaUser = $20000000;

  { @brief Allocate coherent memory. }
  hipHostMallocCoherent = $40000000;

  { @brief Allocate non-coherent memory. }
  hipHostMallocNonCoherent = $80000000;

  { @brief Memory attach global. }
  hipMemAttachGlobal = $01;

  { @brief Memory attach host. }
  hipMemAttachHost = $02;

  { @brief Memory attach single. }
  hipMemAttachSingle = $04;

  { @brief Default device malloc flags. }
  hipDeviceMallocDefault = $0;

  { @brief Fine-grained device malloc. }
  hipDeviceMallocFinegrained = $1;

  { @brief Malloc for HSA signal memory. }
  hipMallocSignalMemory = $2;

  { @brief Uncached device malloc. }
  hipDeviceMallocUncached = $3;

  { @brief Contiguous device malloc. }
  hipDeviceMallocContiguous = $4;

  { @brief Default host register flags. }
  hipHostRegisterDefault = $0;

  { @brief Portable host register. }
  hipHostRegisterPortable = $1;

  { @brief Mapped host register. }
  hipHostRegisterMapped = $2;

  { @brief IO memory host register. }
  hipHostRegisterIoMemory = $4;

  { @brief Read-only host register. }
  hipHostRegisterReadOnly = $08;

  { @brief Coarse-grained external host register. }
  hipExtHostRegisterCoarseGrained = $8;

  { @brief Auto device schedule. }
  hipDeviceScheduleAuto = $0;

  { @brief Spin device schedule. }
  hipDeviceScheduleSpin = $1;

  { @brief Yield device schedule. }
  hipDeviceScheduleYield = $2;

  { @brief Blocking sync device schedule. }
  hipDeviceScheduleBlockingSync = $4;

  { @brief Device schedule mask. }
  hipDeviceScheduleMask = $7;

  { @brief Device map host. }
  hipDeviceMapHost = $8;

  { @brief Device local memory resize to max. }
  hipDeviceLmemResizeToMax = $10;

  { @brief Default array allocation flag. }
  hipArrayDefault = $00;

  { @brief Layered array. }
  hipArrayLayered = $01;

  { @brief Surface load/store array. }
  hipArraySurfaceLoadStore = $02;

  { @brief Cubemap array. }
  hipArrayCubemap = $04;

  { @brief Texture gather array. }
  hipArrayTextureGather = $08;

  { @brief Default occupancy flag. }
  hipOccupancyDefault = $00;

  { @brief Disable caching override for occupancy. }
  hipOccupancyDisableCachingOverride = $01;

  { @brief Multi-device cooperative launch no pre-sync. }
  hipCooperativeLaunchMultiDeviceNoPreSync = $01;

  { @brief Multi-device cooperative launch no post-sync. }
  hipCooperativeLaunchMultiDeviceNoPostSync = $02;

  { @brief CPU device ID. }
  hipCpuDeviceId = -1;

  { @brief Invalid device ID. }
  hipInvalidDeviceId = -2;

  { @brief Any order launch flag. }
  hipExtAnyOrderLaunch = $01;

  { @brief Stream wait value greater than or equal. }
  hipStreamWaitValueGte = $0;

  { @brief Stream wait value equal. }
hipStreamWaitValueEq = $1;

  { @brief Stream wait value AND. }
  hipStreamWaitValueAnd = $2;

  { @brief Stream wait value NOR. }
  hipStreamWaitValueNor = $3;

  { @brief Per-thread stream. }
  hipStreamPerThread: hipStream_t = Pointer(2);

  { @brief Legacy stream. }
  hipStreamLegacy: hipStream_t = Pointer(1);

  { @brief External memory dedicated flag. }
  hipExternalMemoryDedicated = $1;

{ @groupend }

{ @groupbegin(Device Management) }

{
  @brief Explicitly initializes the HIP runtime.
  
  @param(flags Initialization flags (should be 0).)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipInit(flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Init';

{
  @brief Returns the approximate HIP driver version.
  
  @param(driverVersion Pointer to driver version.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipDriverGetVersion(var driverVersion: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DriverGetVersion';

{
  @brief Returns the approximate HIP runtime version.
  
  @param(runtimeVersion Pointer to runtime version.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipRuntimeGetVersion(var runtimeVersion: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'RuntimeGetVersion';

{
  @brief Returns a handle to a compute device.
  
  @param(device Pointer to device handle.)
  @param(ordinal Device ordinal.)
  
  @returns(hipSuccess, hipErrorInvalidDevice)
}
function hipDeviceGet(var device: hipDevice_t; ordinal: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGet';

{
  @brief Returns the compute capability of the device.
  
  @param(major Pointer to major version.)
  @param(minor Pointer to minor version.)
  @param(device Device ordinal.)
  
  @returns(hipSuccess, hipErrorInvalidDevice)
}
function hipDeviceComputeCapability(var major: Int32; var minor: Int32; device: hipDevice_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceComputeCapability';

{
  @brief Returns an identifier string for the device.
  
  @param(name Pointer to name string.)
  @param(len Maximum length of name.)
  @param(device Device ordinal.)
  
  @returns(hipSuccess, hipErrorInvalidDevice)
}
function hipDeviceGetName(name: PChar; len: Int32; device: hipDevice_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGetName';

{
  @brief Returns a UUID for the device (BETA).
  
  @param(uuid Returns the UUID for the device.)
  @param(device The device ordinal.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipDeviceGetUuid(var uuid: THipUUID; device: hipDevice_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGetUuid';

{
  @brief Returns a value for an attribute of the link between two devices.
  
  @param(value Returns the attribute's value.)
  @param(attr The P2P attribute to query.)
  @param(srcDevice The source device of the link.)
  @param(dstDevice The destination device of the link.)
  
  @returns(hipSuccess, hipErrorInvalidDevice)
}
function hipDeviceGetP2PAttribute(var value: Int32; attr: THipDeviceP2PAttr; srcDevice: Int32; dstDevice: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGetP2PAttribute';

{
  @brief Returns a PCI Bus ID string for the device.
  
  @param(pciBusId Pointer to a buffer to store the PCI Bus ID string.)
  @param(len The maximum length of the buffer.)
  @param(device The device ordinal.)
  
  @returns(hipSuccess, hipErrorInvalidDevice)
}
function hipDeviceGetPCIBusId(pciBusId: PChar; len: Int32; device: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGetPCIBusId';

{
  @brief Returns a device handle from a PCI Bus ID string.
  
  @param(device Returns the device handle.)
  @param(pciBusId The PCI Bus ID string.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipDeviceGetByPCIBusId(var device: Int32; pciBusId: PChar): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGetByPCIBusId';

{
  @brief Returns the total amount of memory on the device.
  
  @param(bytes Returns the size of memory in bytes.)
  @param(device The device ordinal.)
  
  @returns(hipSuccess, hipErrorInvalidDevice)
}
function hipDeviceTotalMem(var bytes: SizeUInt; device: hipDevice_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceTotalMem';

{
  @brief Waits for all activity on the current device to complete.
  
  @returns(hipSuccess)
}
function hipDeviceSynchronize: THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceSynchronize';

{
  @brief Resets the current device, destroying all allocations and streams.
  
  @returns(hipSuccess)
}
function hipDeviceReset: THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceReset';

{
  @brief Sets the default device for the calling host thread.
  
  @param(deviceId The device ID to set as current.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorNoDevice)
}
function hipSetDevice(deviceId: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'SetDevice';

{
  @brief Sets the list of valid devices for the application to use.
  
  @param(device_arr An array of device IDs.)
  @param(len The number of devices in the array.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipSetValidDevices(device_arr: PInt32; len: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'SetValidDevices';

{
  @brief Returns the default device ID for the calling host thread.
  
  @param(deviceId Returns the current device ID.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipGetDevice(var deviceId: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetDevice';

{
  @brief Returns the number of compute-capable devices.
  
  @param(count Returns the number of devices.)
  
  @returns(hipSuccess, hipErrorNoDevice)
}
function hipGetDeviceCount(var count: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetDeviceCount';

{
  @brief Queries for a specific device attribute.
  
  @param(pi Returns the attribute's value.)
  @param(attr The attribute to query.)
  @param(deviceId The device to query.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipDeviceGetAttribute(var pi: Int32; attr: THipDeviceAttribute; deviceId: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGetAttribute';

{
  @brief Returns the default memory pool of a device.
  
  @param(mem_pool Returns the default memory pool.)
  @param(device The device index to query.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue, hipErrorNotSupported)
  
  @warning(This API is marked as beta.)
}
function hipDeviceGetDefaultMemPool(var mem_pool: hipMemPool_t; device: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGetDefaultMemPool';

{
  @brief Sets the current memory pool for a device.
  
  @param(device The device index to update.)
  @param(mem_pool The memory pool to set as current.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidDevice, hipErrorNotSupported)
  
  @warning(This API is marked as beta.)
}
function hipDeviceSetMemPool(device: Int32; mem_pool: hipMemPool_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceSetMemPool';

{
  @brief Gets the current memory pool for a device.
  
  @param(mem_pool Returns the current memory pool.)
  @param(device The device index to query.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
  
  @warning(This API is marked as beta.)
}
function hipDeviceGetMemPool(var mem_pool: hipMemPool_t; device: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGetMemPool';

{
  @brief Returns properties for a specific device.
  
  @param(prop A structure to be filled with the device properties.)
  @param(deviceId The device to query.)
  
  @returns(hipSuccess, hipErrorInvalidDevice)
}
function hipGetDeviceProperties(var prop: THipDeviceProp; deviceId: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetDevicePropertiesR0600';

{
  @brief Sets the preferred cache configuration for the current device.
  
  @param(cacheConfig The cache configuration to set.)
  
  @returns(hipSuccess, hipErrorNotInitialized, hipErrorNotSupported)
  
  @note(This is a hint and may be ignored on some architectures.)
}
function hipDeviceSetCacheConfig(cacheConfig: THipFuncCache): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceSetCacheConfig';

{
  @brief Gets the current cache configuration for a device.
  
  @param(cacheConfig Returns the current cache configuration.)
  
  @returns(hipSuccess, hipErrorNotInitialized)
}
function hipDeviceGetCacheConfig(var cacheConfig: THipFuncCache): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGetCacheConfig';

{
  @brief Gets a resource limit for the current device.
  
  @param(pValue Returns the size of the limit in bytes.)
  @param(limit The limit to query.)
  
  @returns(hipSuccess, hipErrorUnsupportedLimit, hipErrorInvalidValue)
}
function hipDeviceGetLimit(var pValue: SizeUInt; limit: THipLimit): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGetLimit';

{
  @brief Sets a resource limit for the current device.
  
  @param(limit The limit to set.)
  @param(value The new size of the limit in bytes.)
  
  @returns(hipSuccess, hipErrorUnsupportedLimit, hipErrorInvalidValue)
}
function hipDeviceSetLimit(limit: THipLimit; value: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceSetLimit';

{
  @brief Gets the shared memory banking configuration for the current device.
  
  @param(pConfig Returns the shared memory configuration.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized)
  
  @note(This is a hint and may be ignored on some architectures.)
}
function hipDeviceGetSharedMemConfig(var pConfig: THipSharedMemConfig): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGetSharedMemConfig';

{
  @brief Sets the shared memory banking configuration for the current device.
  
  @param(config The shared memory configuration to set.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized)
  
  @note(This is a hint and may be ignored on some architectures.)
}
function hipDeviceSetSharedMemConfig(config: THipSharedMemConfig): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceSetSharedMemConfig';

{
  @brief Gets the flags set for the current device.
  
  @param(flags Returns the flags.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipGetDeviceFlags(var flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetDeviceFlags';

{
  @brief Sets flags to control the behavior of the current device.
  
  @param(flags The flags to set.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorSetOnActiveProcess)
}
function hipSetDeviceFlags(flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'SetDeviceFlags';

{
  @brief Chooses a device that best matches the given properties.
  
  @param(device Returns the ID of the chosen device.)
  @param(prop The desired device properties.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipChooseDevice(var device: Int32; const prop: PHipDeviceProp): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ChooseDeviceR0600';

{
  @brief Returns the link type and hop count between two devices.
  
  @param(device1 The first device ordinal.)
  @param(device2 The second device ordinal.)
  @param(linktype Returns the link type.)
  @param(hopcount Returns the hop count.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipExtGetLinkTypeAndHopCount(device1: Int32; device2: Int32; var linktype: UInt32; var hopcount: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ExtGetLinkTypeAndHopCount';
{$ELSE}
  // This is an AMD-specific extension and not available on NVIDIA.
{$ENDIF}

{
  @brief Gets an interprocess memory handle for an existing device memory allocation.
  
  @param(handle Returns the IPC memory handle.)
  @param(devPtr The base pointer of the allocated device memory.)
  
  @returns(hipSuccess, hipErrorInvalidHandle, hipErrorOutOfMemory, hipErrorMapFailed)
}
function hipIpcGetMemHandle(var handle: THipIpcMemHandle; devPtr: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'IpcGetMemHandle';

{
  @brief Opens an interprocess memory handle to get a device pointer.
  
  @param(devPtr Returns the device pointer usable in the local process.)
  @param(handle The IPC memory handle to open.)
  @param(flags Flags for the operation (e.g., hipIpcMemLazyEnablePeerAccess).)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorInvalidDevicePointer)
}
function hipIpcOpenMemHandle(var devPtr: Pointer; handle: THipIpcMemHandle; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'IpcOpenMemHandle';

{
  @brief Closes memory that was opened with hipIpcOpenMemHandle.
  
  @param(devPtr The device pointer returned by hipIpcOpenMemHandle.)
  
  @returns(hipSuccess, hipErrorMapFailed, hipErrorInvalidHandle)
}
function hipIpcCloseMemHandle(devPtr: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'IpcCloseMemHandle';

{
  @brief Gets an opaque interprocess handle for an event.
  
  @param(handle Returns the IPC event handle.)
  @param(event The event to get a handle for.)
  
  @returns(hipSuccess, hipErrorInvalidConfiguration, hipErrorInvalidValue)
}
function hipIpcGetEventHandle(var handle: THipIpcEventHandle; event: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'IpcGetEventHandle';

{
  @brief Opens an interprocess event handle.
  
  @param(event Returns the event handle usable in the local process.)
  @param(handle The IPC event handle to open.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext)
}
function hipIpcOpenEventHandle(var event: hipEvent_t; handle: THipIpcEventHandle): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'IpcOpenEventHandle';

{ @groupend }

{ @groupbegin(Execution Control) }

{
  @brief Sets an attribute for a specific function.
  
  @param(func A pointer to the function.)
  @param(attr The attribute to set.)
  @param(value The value to set for the attribute.)
  
  @returns(hipSuccess, hipErrorInvalidDeviceFunction, hipErrorInvalidValue)
  
  @note(This is a hint and may be ignored on some architectures.)
}
function hipFuncSetAttribute(func: Pointer; attr: THipFuncAttributes; value: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'FuncSetAttribute';

{
  @brief Sets the cache configuration for a specific function.
  
  @param(func A pointer to the function.)
  @param(config The cache configuration to set.)
  
  @returns(hipSuccess, hipErrorNotInitialized)
  
  @note(This is a hint and may be ignored on some architectures.)
}
function hipFuncSetCacheConfig(func: Pointer; config: THipFuncCache): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'FuncSetCacheConfig';

{
  @brief Sets the shared memory configuration for a specific function.
  
  @param(func A pointer to the function.)
  @param(config The shared memory configuration to set.)
  
  @returns(hipSuccess, hipErrorInvalidDeviceFunction, hipErrorInvalidValue)
  
  @note(This is a hint and may be ignored on some architectures.)
}
function hipFuncSetSharedMemConfig(func: Pointer; config: THipSharedMemConfig): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'FuncSetSharedMemConfig';

{ @groupend }

{ @groupbegin(Error Handling) }

{
  @brief Returns the last error from a HIP runtime call and resets it.
  
  @returns(The last error code from the active host thread.)
}
function hipGetLastError: THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetLastError';

{
  @brief Returns the last error from a HIP runtime call and resets it (extended version).
  
  @returns(The last error code from the active host thread.)
}
function hipExtGetLastError: THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ExtGetLastError';

{
  @brief Returns the last error from a HIP runtime call without resetting it.
  
  @returns(The last error code from the active host thread.)
}
function hipPeekAtLastError: THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'PeekAtLastError';

{
  @brief Returns the string name of a HIP error code.
  
  @param(hip_error The error code.)
  
  @returns(A null-terminated string with the error's name.)
}
function hipGetErrorName(hip_error: hip_error.THipError): PChar;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetErrorName';

{
  @brief Returns a descriptive string for a HIP error code.
  
  @param(hipError The error code.)
  
  @returns(A null-terminated string describing the error.)
}
function hipGetErrorString(hipError: hip_error.THipError): PChar;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetErrorString';

{
  @brief Gets the string name of a HIP error code (driver version).
  
  @param(hipError The error code.)
  @param(errorString Returns a pointer to the null-terminated error string.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipDrvGetErrorName(hipError: THipError; var errorString: PChar): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DrvGetErrorName';

{
  @brief Gets a descriptive string for a HIP error code (driver version).
  
  @param(hipError The error code.)
  @param(errorString Returns a pointer to the null-terminated error string.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipDrvGetErrorString(hipError: THipError; var errorString: PChar): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DrvGetErrorString';

{ @groupend }

{ @groupbegin(Stream Management) }

{
  @brief Creates an asynchronous stream.
  
  @param(stream Returns the newly created stream.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipStreamCreate(var stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamCreate';

{
  @brief Creates an asynchronous stream with flags.
  
  @param(stream Returns the newly created stream.)
  @param(flags Flags to control stream creation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipStreamCreateWithFlags(var stream: hipStream_t; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamCreateWithFlags';

{
  @brief Creates an asynchronous stream with a specified priority.
  
  @param(stream Returns the newly created stream.)
  @param(flags Flags to control stream creation.)
  @param(priority The priority of the stream.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipStreamCreateWithPriority(var stream: hipStream_t; flags: UInt32; priority: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamCreateWithPriority';

{
  @brief Returns the range of valid stream priorities.
  
  @param(leastPriority Returns the least priority.)
  @param(greatestPriority Returns the greatest priority.)
  
  @returns(hipSuccess)
}
function hipDeviceGetStreamPriorityRange(var leastPriority: Int32; var greatestPriority: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGetStreamPriorityRange';

{
  @brief Destroys a stream.
  
  @param(stream The stream to destroy.)
  
  @returns(hipSuccess, hipErrorInvalidHandle)
}
function hipStreamDestroy(stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamDestroy';

{
  @brief Queries the status of a stream.
  
  @param(stream The stream to query.)
  
  @returns(hipSuccess if all operations are complete, hipErrorNotReady otherwise.)
}
function hipStreamQuery(stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamQuery';

{
  @brief Waits for all commands in a stream to complete.
  
  @param(stream The stream to synchronize.)
  
  @returns(hipSuccess, hipErrorInvalidHandle)
}
function hipStreamSynchronize(stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamSynchronize';

{
  @brief Makes a stream wait for an event.
  
  @param(stream The stream that will wait.)
  @param(event The event to wait on.)
  @param(flags Reserved for future use, must be 0.)
  
  @returns(hipSuccess, hipErrorInvalidHandle)
}
function hipStreamWaitEvent(stream: hipStream_t; event: hipEvent_t; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamWaitEvent';

{
  @brief Returns the flags associated with a stream.
  
  @param(stream The stream to query.)
  @param(flags Returns the stream's flags.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidHandle)
}
function hipStreamGetFlags(stream: hipStream_t; var flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamGetFlags';

{
  @brief Queries the priority of a stream.
  
  @param(stream The stream to query.)
  @param(priority Returns the stream's priority.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidHandle)
}
function hipStreamGetPriority(stream: hipStream_t; var priority: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamGetPriority';

{
  @brief Gets the device associated with a stream.
  
  @param(stream The stream to query.)
  @param(device Returns the device associated with the stream.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidHandle)
}
function hipStreamGetDevice(stream: hipStream_t; var device: hipDevice_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamGetDevice';

{
  @brief Creates an asynchronous stream with a specified CU mask.
  
  @param(stream Returns the newly created stream.)
  @param(cuMaskSize The size of the CU mask array.)
  @param(cuMask A bit-vector representing the CU mask.)
  
  @returns(hipSuccess, hipErrorInvalidHandle, hipErrorInvalidValue)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipExtStreamCreateWithCUMask(var stream: hipStream_t; cuMaskSize: UInt32; cuMask: PUInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ExtStreamCreateWithCUMask';
{$ELSE}
  // This is an AMD-specific extension.
{$ENDIF}

{
  @brief Gets the CU mask associated with a stream.
  
  @param(stream The stream to query.)
  @param(cuMaskSize The size of the buffer for the CU mask.)
  @param(cuMask Returns the stream's CU mask.)
  
  @returns(hipSuccess, hipErrorInvalidHandle, hipErrorInvalidValue)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipExtStreamGetCUMask(stream: hipStream_t; cuMaskSize: UInt32; cuMask: PUInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ExtStreamGetCUMask';
{$ELSE}
  // This is an AMD-specific extension.
{$ENDIF}

type
  {
    @brief HIP stream callback function pointer type.
  }
  THipStreamCallback = procedure(stream: hipStream_t; status: THipError; userData: Pointer); cdecl;

{
  @brief Adds a callback to be executed on the host after stream completion.
  
  @param(stream The stream to add the callback to.)
  @param(callback The function to call.)
  @param(userData User-specified data to be passed to the callback.)
  @param(flags Reserved for future use, must be 0.)
  
  @returns(hipSuccess, hipErrorInvalidHandle, hipErrorNotSupported)
}
function hipStreamAddCallback(stream: hipStream_t; callback: THipStreamCallback; userData: Pointer; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamAddCallback';

{ @groupend }

{ @groupbegin(Stream Memory Operations) }
{ @abstract(This section describes Stream Memory Wait and Write functions of the HIP runtime API.) }

{
  @brief Enqueues a command to wait on a 32-bit memory location. [BETA]
  
  @param(stream The stream to enqueue the wait on.)
  @param(ptr The device memory address to monitor.)
  @param(value The value to compare against.)
  @param(flags The comparison operation (e.g., hipStreamWaitValueGte).)
  @param(mask A bitmask to apply to the value at ptr before comparison.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta.)
}
function hipStreamWaitValue32(stream: hipStream_t; ptr: Pointer; value: UInt32; flags: UInt32; mask: UInt32 = $FFFFFFFF): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamWaitValue32';

{
  @brief Enqueues a command to wait on a 64-bit memory location. [BETA]
  
  @param(stream The stream to enqueue the wait on.)
  @param(ptr The device memory address to monitor.)
  @param(value The value to compare against.)
  @param(flags The comparison operation.)
  @param(mask A bitmask to apply to the value at ptr before comparison.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta.)
}
function hipStreamWaitValue64(stream: hipStream_t; ptr: Pointer; value: UInt64; flags: UInt32; mask: UInt64 = $FFFFFFFFFFFFFFFF): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamWaitValue64';

{
  @brief Enqueues a command to write a 32-bit value to memory. [BETA]
  
  @param(stream The stream to enqueue the write in.)
  @param(ptr The device memory address to write to.)
  @param(value The 32-bit value to write.)
  @param(flags Reserved for future use.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta.)
}
function hipStreamWriteValue32(stream: hipStream_t; ptr: Pointer; value: UInt32; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamWriteValue32';

{
  @brief Enqueues a command to write a 64-bit value to memory. [BETA]
  
  @param(stream The stream to enqueue the write in.)
  @param(ptr The device memory address to write to.)
  @param(value The 64-bit value to write.)
  @param(flags Reserved for future use.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta.)
}
function hipStreamWriteValue64(stream: hipStream_t; ptr: Pointer; value: UInt64; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamWriteValue64';

{ @groupend }

{ @groupbegin(Event Management) }
{ @abstract(This section describes the event management functions of the HIP runtime API.) }

{
  @brief Creates an event with specified flags.
  
  @param(event Returns the newly created event.)
  @param(flags Flags to control event behavior.)
  
  @returns(hipSuccess, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorLaunchFailure, hipErrorOutOfMemory)
}
function hipEventCreateWithFlags(var event: hipEvent_t; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'EventCreateWithFlags';

{
  @brief Creates an event.
  
  @param(event Returns the newly created event.)
  
  @returns(hipSuccess, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorLaunchFailure, hipErrorOutOfMemory)
}
function hipEventCreate(var event: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'EventCreate';

{
  @brief Records an event in a stream.
  
  @param(event The event to record.)
  @param(stream The stream in which to record the event (nil for default stream).
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized, hipErrorInvalidHandle, hipErrorLaunchFailure)
}
function hipEventRecord(event: hipEvent_t; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'EventRecord';

{
  @brief Destroys an event.
  
  @param(event The event to destroy.)
  
  @returns(hipSuccess, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorLaunchFailure)
}
function hipEventDestroy(event: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'EventDestroy';

{
  @brief Waits for an event to complete.
  
  @param(event The event to wait for.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized, hipErrorInvalidHandle, hipErrorLaunchFailure)
}
function hipEventSynchronize(event: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'EventSynchronize';

{
  @brief Computes the elapsed time between two events.
  
  @param(ms Returns the elapsed time in milliseconds.)
  @param(start The start event.)
  @param(stop The stop event.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotReady, hipErrorInvalidHandle, hipErrorNotInitialized, hipErrorLaunchFailure)
}
function hipEventElapsedTime(var ms: Single; start: hipEvent_t; stop: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'EventElapsedTime';

{
  @brief Queries an event's status.
  
  @param(event The event to query.)
  
  @returns(hipSuccess if completed, hipErrorNotReady otherwise.)
}
function hipEventQuery(event: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'EventQuery';

{ @groupend }

{ @groupbegin(Memory Management) }
{ @abstract(This section describes the memory management functions of the HIP runtime API.) }

{
  @brief Sets an attribute for a specified pointer. [BETA]
  
  @param(value A pointer to the value to set for the attribute.)
  @param(attribute The attribute to set.)
  @param(ptr The device pointer to set attributes for.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
  
  @warning(This API is marked as beta.)
}
function hipPointerSetAttribute(value: Pointer; attribute: THipPointerAttributeEnum; ptr: hipDeviceptr_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'PointerSetAttribute';

{
  @brief Returns attributes for a specified pointer.
  
  @param(attributes A structure to be filled with the pointer's attributes.)
  @param(ptr The pointer to get attributes for.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipPointerGetAttributes(var attributes: THipPointerAttribute; ptr: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'PointerGetAttributes';

{
  @brief Returns a single attribute for a specified pointer. [BETA]
  
  @param(data A pointer to memory where the attribute value will be returned.)
  @param(attribute The attribute to query.)
  @param(ptr The pointer to get the attribute for.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
  
  @warning(This API is marked as beta.)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipPointerGetAttribute(data: Pointer; attribute: THipPointerAttributeEnum; ptr: hipDeviceptr_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'PointerGetAttribute';
{$ELSE}
  // This is an AMD-specific extension.
{$ENDIF}

{
  @brief Returns multiple attributes for a specified pointer (driver version). [BETA]
  
  @param(numAttributes The number of attributes to query.)
  @param(attributes An array of attributes to query.)
  @param(data An array of pointers where the results will be written.)
  @param(ptr The pointer to get attributes for.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
  
  @warning(This API is marked as beta.)
}
function hipDrvPointerGetAttributes(numAttributes: UInt32; attributes: PHipPointerAttributeEnum; data: PPChar; ptr: hipDeviceptr_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DrvPointerGetAttributes';

{ @groupend }


{ @groupbegin(External Resource Interoperability) }
{ @abstract(This section describes external resource interoperability functions.) }

{
  @brief Imports an external semaphore.
  
  @param(extSem_out Returns the imported external semaphore handle.)
  @param(semHandleDesc The semaphore import handle descriptor.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipImportExternalSemaphore(var extSem_out: hipExternalSemaphore_t; const semHandleDesc: PHipExternalSemaphoreHandleDesc): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ImportExternalSemaphore';

{
  @brief Asynchronously signals a set of external semaphore objects.
  
  @param(extSemArray An array of external semaphore handles to signal.)
  @param(paramsArray An array of signal parameters.)
  @param(numExtSems The number of semaphores to signal.)
  @param(stream The stream to enqueue the signal operations in.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipSignalExternalSemaphoresAsync(extSemArray: PhipExternalSemaphore_t; const paramsArray: PHipExternalSemaphoreSignalParams; numExtSems: UInt32; stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'SignalExternalSemaphoresAsync';

{
  @brief Asynchronously waits on a set of external semaphore objects.
  
  @param(extSemArray An array of external semaphore handles to wait on.)
  @param(paramsArray An array of wait parameters.)
  @param(numExtSems The number of semaphores to wait on.)
  @param(stream The stream to enqueue the wait operations in.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipWaitExternalSemaphoresAsync(extSemArray: PhipExternalSemaphore_t; const paramsArray: PHipExternalSemaphoreWaitParams; numExtSems: UInt32; stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'WaitExternalSemaphoresAsync';

{
  @brief Destroys an external semaphore object.
  
  @param(extSem The external semaphore handle to destroy.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipDestroyExternalSemaphore(extSem: hipExternalSemaphore_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DestroyExternalSemaphore';

{
  @brief Imports an external memory object.
  
  @param(extMem_out Returns the imported external memory handle.)
  @param(memHandleDesc The memory import handle descriptor.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipImportExternalMemory(var extMem_out: hipExternalMemory_t; const memHandleDesc: PHipExternalMemoryHandleDesc): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ImportExternalMemory';

{
  @brief Maps a buffer onto an imported external memory object.
  
  @param(devPtr Returns the device pointer to the mapped buffer.)
  @param(extMem The external memory handle.)
  @param(bufferDesc The buffer descriptor.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipExternalMemoryGetMappedBuffer(var devPtr: Pointer; extMem: hipExternalMemory_t; const bufferDesc: PHipExternalMemoryBufferDesc): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ExternalMemoryGetMappedBuffer';

{
  @brief Destroys an external memory object.
  
  @param(extMem The external memory object to destroy.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipDestroyExternalMemory(extMem: hipExternalMemory_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DestroyExternalMemory';

{
  @brief Maps a mipmapped array onto an external memory object.
  
  @param(mipmap Returns the mapped mipmapped array.)
  @param(extMem The external memory handle.)
  @param(mipmapDesc The mipmapped array descriptor.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidResourceHandle)
}
function hipExternalMemoryGetMappedMipmappedArray(var mipmap: hipMipmappedArray_t; extMem: hipExternalMemory_t; const mipmapDesc: PHipExternalMemoryMipmappedArrayDesc): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ExternalMemoryGetMappedMipmappedArray';

{ @groupend }

{ @groupbegin(Memory Allocation) }

{
  @brief Allocates memory on the default device.
  
  @param(ptr Returns a pointer to the allocated memory.)
  @param(size The size of the allocation in bytes.)
  
  @returns(hipSuccess, hipErrorOutOfMemory, hipErrorInvalidValue)
}
function hipMalloc(var ptr: Pointer; size: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Malloc';

{
  @brief Allocates memory on the default device with flags.
  
  @param(ptr Returns a pointer to the allocated memory.)
  @param(sizeBytes The size of the allocation in bytes.)
  @param(flags Flags to control the allocation type.)
  
  @returns(hipSuccess, hipErrorOutOfMemory, hipErrorInvalidValue)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipExtMallocWithFlags(var ptr: Pointer; sizeBytes: SizeUInt; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ExtMallocWithFlags';
{$ELSE}
  // This is an AMD-specific extension.
{$ENDIF}

{ @groupend }

{ @groupbegin(Deprecated Memory Management) }

{
  @brief Allocates pinned host memory [DEPRECATED].

  @param(ptr Returns a pointer to the allocated host memory.)
  @param(size The size of the allocation in bytes.)

  @returns(hipSuccess, hipErrorOutOfMemory)

  @warning(This API is deprecated, use hipHostMalloc instead.)
}
function hipMallocHost(var ptr: Pointer; size: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MallocHost'; deprecated;

{
  @brief Allocates pinned host memory [DEPRECATED].
  
  @param(ptr Returns a pointer to the allocated host memory.)
  @param(size The size of the allocation in bytes.)
  
  @returns(hipSuccess, hipErrorOutOfMemory)
  
  @warning(This API is deprecated, use hipHostMalloc instead.)
}
function hipMemAllocHost(var ptr: Pointer; size: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemAllocHost'; deprecated;

{ @groupend }

{ @groupbegin(Host Memory Management) }

{
  @brief Allocates page-locked (pinned) host memory.
  
  @param(ptr Returns a pointer to the allocated host memory.)
  @param(size The size of the allocation in bytes.)
  @param(flags Flags to control the allocation behavior.)
  
  @returns(hipSuccess, hipErrorOutOfMemory)
}
function hipHostMalloc(var ptr: Pointer; size: SizeUInt; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'HostMalloc';

{
  @brief Allocates page-locked (pinned) host memory.
  
  @param(ptr Returns a pointer to the allocated host memory.)
  @param(size The size of the allocation in bytes.)
  @param(flags Flags to control the allocation behavior.)
  
  @returns(hipSuccess, hipErrorOutOfMemory, hipErrorInvalidValue)
}
function hipHostAlloc(var ptr: Pointer; size: SizeUInt; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'HostAlloc';

{
  @brief Gets the device pointer for a host pointer allocated with hipHostMalloc.
  
  @param(devPtr Returns the device pointer.)
  @param(hstPtr The host pointer.)
  @param(flags Reserved for future use.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorOutOfMemory)
}
function hipHostGetDevicePointer(var devPtr: Pointer; hstPtr: Pointer; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'HostGetDevicePointer';

{
  @brief Returns the flags associated with a host pointer.
  
  @param(flagsPtr Returns the flags used to allocate the memory.)
  @param(hostPtr The host pointer to query.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipHostGetFlags(var flagsPtr: UInt32; hostPtr: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'HostGetFlags';

{
  @brief Registers host memory for device access.
  
  @param(hostPtr The host memory to register.)
  @param(sizeBytes The size of the memory region in bytes.)
  @param(flags Flags to control the registration.)
  
  @returns(hipSuccess, hipErrorOutOfMemory)
}
function hipHostRegister(hostPtr: Pointer; sizeBytes: SizeUInt; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'HostRegister';

{
  @brief Unregisters a region of host memory.
  
  @param(hostPtr The host pointer to unregister.)
  
  @returns(Error code)
}
function hipHostUnregister(hostPtr: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'HostUnregister';

{ @groupend }

{ @groupbegin(Pitched and 3D Memory) }

{
  @brief Allocates pitched linear memory on the device.
  
  @param(ptr Returns a pointer to the allocated memory.)
  @param(pitch Returns the allocated pitch in bytes.)
  @param(width The requested width of the allocation in bytes.)
  @param(height The requested height of the allocation.)
  
  @returns(Error code)
}
function hipMallocPitch(var ptr: Pointer; var pitch: SizeUInt; width: SizeUInt; height: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MallocPitch';

{
  @brief Allocates pitched linear memory on the device.
  
  @param(dptr Returns a device pointer to the allocated memory.)
  @param(pitch Returns the allocated pitch in bytes.)
  @param(widthInBytes The requested width of the allocation in bytes.)
  @param(height The requested height of the allocation.)
  @param(elementSizeBytes The size of each element in bytes (e.g., 4 for float).
  
  @returns(Error code)
}
function hipMemAllocPitch(var dptr: hipDeviceptr_t; var pitch: SizeUInt; widthInBytes: SizeUInt; height: SizeUInt; elementSizeBytes: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemAllocPitch';

{ @groupend }

{ @groupbegin(Memory Freeing) }

{
  @brief Frees memory allocated by hipMalloc.
  
  @param(ptr The pointer to the memory to be freed.)
  
  @returns(hipSuccess, hipErrorInvalidDevicePointer)
}
function hipFree(ptr: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Free';

{
  @brief Frees page-locked host memory.
  
  @param(ptr The pointer to the memory to be freed.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipFreeHost(ptr: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'FreeHost';

{
  @brief Frees page-locked host memory (alias for hipFreeHost).
  
  @param(ptr The pointer to the memory to be freed.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipHostFree(ptr: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'HostFree';

{ @groupend }

{ @groupbegin(Memory Copy) }

{
  @brief Copies data between host and device.
  
  @param(dst The destination pointer.)
  @param(src The source pointer.)
  @param(sizeBytes The size of the copy in bytes.)
  @param(kind The kind of transfer (e.g., HostToDevice).
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorUnknown)
}
function hipMemcpy(dst: Pointer; src: Pointer; sizeBytes: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memcpy';

{
  @brief Copies data on a specific stream.
  
  @param(dst The destination pointer.)
  @param(src The source pointer.)
  @param(sizeBytes The size of the copy in bytes.)
  @param(kind The kind of transfer.)
  @param(stream The stream on which to perform the copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorUnknown, hipErrorContextIsDestroyed)
}
function hipMemcpyWithStream(dst: Pointer; src: Pointer; sizeBytes: SizeUInt; kind: THipMemcpyKind; stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyWithStream';

{
  @brief Copies data from Host to Device.
  
  @param(dst The destination device pointer.)
  @param(src The source host pointer.)
  @param(sizeBytes The size of the copy in bytes.)
  
  @returns(hipSuccess, hipErrorInvalidValue, etc.)
}
function hipMemcpyHtoD(dst: hipDeviceptr_t; src: Pointer; sizeBytes: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyHtoD';

{
  @brief Copies data from Device to Host.
  
  @param(dst The destination host pointer.)
  @param(src The source device pointer.)
  @param(sizeBytes The size of the copy in bytes.)
  
  @returns(hipSuccess, hipErrorInvalidValue, etc.)
}
function hipMemcpyDtoH(dst: Pointer; src: hipDeviceptr_t; sizeBytes: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyDtoH';

{
  @brief Copies data from Device to Device.
  
  @param(dst The destination device pointer.)
  @param(src The source device pointer.)
  @param(sizeBytes The size of the copy in bytes.)
  
  @returns(hipSuccess, hipErrorInvalidValue, etc.)
}
function hipMemcpyDtoD(dst: hipDeviceptr_t; src: hipDeviceptr_t; sizeBytes: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyDtoD';

{
  @brief Asynchronously copies data.
  
  @param(dst The destination pointer.)
  @param(src The source pointer.)
  @param(sizeBytes The size of the copy in bytes.)
  @param(kind The kind of transfer.)
  @param(stream The stream on which to perform the copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorUnknown)
}
function hipMemcpyAsync(dst: Pointer; src: Pointer; sizeBytes: SizeUInt; kind: THipMemcpyKind; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyAsync';

{
  @brief Asynchronously copies data from Host to Device.
  
  @param(dst The destination device pointer.)
  @param(src The source host pointer.)
  @param(sizeBytes The size of the copy in bytes.)
  @param(stream The stream identifier.)
  
  @returns(hipSuccess, hipErrorInvalidValue, etc.)
}
function hipMemcpyHtoDAsync(dst: hipDeviceptr_t; src: Pointer; sizeBytes: SizeUInt; stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyHtoDAsync';

{
  @brief Asynchronously copies data from Device to Host.
  
  @param(dst The destination host pointer.)
  @param(src The source device pointer.)
  @param(sizeBytes The size of the copy in bytes.)
  @param(stream The stream identifier.)
  
  @returns(hipSuccess, hipErrorInvalidValue, etc.)
}
function hipMemcpyDtoHAsync(dst: Pointer; src: hipDeviceptr_t; sizeBytes: SizeUInt; stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyDtoHAsync';

{
  @brief Asynchronously copies data from Device to Device.
  
  @param(dst The destination device pointer.)
  @param(src The source device pointer.)
  @param(sizeBytes The size of the copy in bytes.)
  @param(stream The stream identifier.)
  
  @returns(hipSuccess, hipErrorInvalidValue, etc.)
}
function hipMemcpyDtoDAsync(dst: hipDeviceptr_t; src: hipDeviceptr_t; sizeBytes: SizeUInt; stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyDtoDAsync';

{ @groupend }

{ @groupbegin(Symbol Memory Operations) }

{
  @brief Gets the device pointer associated with a symbol.
  
  @param(devPtr Returns the device pointer.)
  @param(symbol A pointer to the symbol in host memory.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipGetSymbolAddress(var devPtr: Pointer; symbol: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetSymbolAddress';

{
  @brief Gets the size of a symbol.
  
  @param(size Returns the size of the symbol in bytes.)
  @param(symbol A pointer to the symbol in host memory.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipGetSymbolSize(var size: SizeUInt; symbol: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetSymbolSize';

{
  @brief Copies data to a symbol on the device.
  
  @param(symbol The destination device symbol.)
  @param(src The source data pointer.)
  @param(sizeBytes The size of the copy in bytes.)
  @param(offset An offset into the symbol's memory.)
  @param(kind The kind of transfer.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpyToSymbol(symbol: Pointer; src: Pointer; sizeBytes: SizeUInt; offset: SizeUInt = 0; kind: THipMemcpyKind = hipMemcpyHostToDevice): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyToSymbol';

{
  @brief Asynchronously copies data to a symbol on the device.
  
  @param(symbol The destination device symbol.)
  @param(src The source data pointer.)
  @param(sizeBytes The size of the copy in bytes.)
  @param(offset An offset into the symbol's memory.)
  @param(kind The kind of transfer.)
  @param(stream The stream for the asynchronous copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpyToSymbolAsync(symbol: Pointer; src: Pointer; sizeBytes: SizeUInt; offset: SizeUInt; kind: THipMemcpyKind; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyToSymbolAsync';

{
  @brief Copies data from a symbol on the device.
  
  @param(dst The destination data pointer.)
  @param(symbol The source device symbol.)
  @param(sizeBytes The size of the copy in bytes.)
  @param(offset An offset into the symbol's memory.)
  @param(kind The kind of transfer.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpyFromSymbol(dst: Pointer; symbol: Pointer; sizeBytes: SizeUInt; offset: SizeUInt = 0; kind: THipMemcpyKind = hipMemcpyDeviceToHost): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyFromSymbol';

{
  @brief Asynchronously copies data from a symbol on the device.
  
  @param(dst The destination data pointer.)
  @param(symbol The source device symbol.)
  @param(sizeBytes The size of the copy in bytes.)
  @param(offset An offset into the symbol's memory.)
  @param(kind The kind of transfer.)
  @param(stream The stream for the asynchronous copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpyFromSymbolAsync(dst: Pointer; symbol: Pointer; sizeBytes: SizeUInt; offset: SizeUInt; kind: THipMemcpyKind; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyFromSymbolAsync';

{ @groupend }

{ @groupbegin(Memory Set Operations) }

{
  @brief Fills a region of device memory with a byte value.
  
  @param(dst The destination device pointer.)
  @param(value The byte value to set.)
  @param(sizeBytes The size of the memory region in bytes.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized)
}
function hipMemset(dst: Pointer; value: Int32; sizeBytes: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memset';

{
  @brief Fills a memory area with a constant 8-bit value.
  
  @param(dest The destination device pointer.)
  @param(value The byte value to set.)
  @param(count The number of bytes to set.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized)
}
function hipMemsetD8(dest: hipDeviceptr_t; value: Byte; count: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemsetD8';

{
  @brief Asynchronously fills a memory area with a constant 8-bit value.
  
  @param(dest The destination device pointer.)
  @param(value The byte value to set.)
  @param(count The number of bytes to set.)
  @param(stream The stream for the asynchronous operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized)
}
function hipMemsetD8Async(dest: hipDeviceptr_t; value: Byte; count: SizeUInt; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemsetD8Async';

{
  @brief Fills a memory area with a constant 16-bit value.
  
  @param(dest The destination device pointer.)
  @param(value The short value to set.)
  @param(count The number of 16-bit values to set.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized)
}
function hipMemsetD16(dest: hipDeviceptr_t; value: UInt16; count: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemsetD16';

{
  @brief Asynchronously fills a memory area with a constant 16-bit value.
  
  @param(dest The destination device pointer.)
  @param(value The short value to set.)
  @param(count The number of 16-bit values to set.)
  @param(stream The stream for the asynchronous operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized)
}
function hipMemsetD16Async(dest: hipDeviceptr_t; value: UInt16; count: SizeUInt; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemsetD16Async';

{
  @brief Fills a memory area with a constant 32-bit integer value.
  
  @param(dest The destination device pointer.)
  @param(value The integer value to set.)
  @param(count The number of 32-bit values to set.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized)
}
function hipMemsetD32(dest: hipDeviceptr_t; value: Int32; count: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemsetD32';

{
  @brief Asynchronously fills a region of device memory with a byte value.
  
  @param(dst The destination device pointer.)
  @param(value The byte value to set.)
  @param(sizeBytes The size of the memory region in bytes.)
  @param(stream The stream for the asynchronous operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemsetAsync(dst: Pointer; value: Int32; sizeBytes: SizeUInt; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemsetAsync';

{
  @brief Asynchronously fills a memory area with a constant 32-bit integer value.
  
  @param(dst The destination device pointer.)
  @param(value The integer value to set.)
  @param(count The number of 32-bit values to set.)
  @param(stream The stream for the asynchronous operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemsetD32Async(dst: hipDeviceptr_t; value: Int32; count: SizeUInt; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemsetD32Async';

{
  @brief Fills a 2D memory area with a constant value.
  
  @param(dst The destination device pointer.)
  @param(pitch The pitch of the destination memory in bytes.)
  @param(value The byte value to set.)
  @param(width The width of the region to set in bytes.)
  @param(height The height of the region to set.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemset2D(dst: Pointer; pitch: SizeUInt; value: Int32; width: SizeUInt; height: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memset2D';

{
  @brief Asynchronously fills a 2D memory area with a constant value.
  
  @param(dst The destination device pointer.)
  @param(pitch The pitch of the destination memory in bytes.)
  @param(value The byte value to set.)
  @param(width The width of the region to set in bytes.)
  @param(height The height of the region to set.)
  @param(stream The stream for the asynchronous operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemset2DAsync(dst: Pointer; pitch: SizeUInt; value: Int32; width: SizeUInt; height: SizeUInt; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memset2DAsync';

{
  @brief Fills a 3D memory area with a constant value.
  
  @param(pitchedDevPtr A pitched pointer to the destination memory.)
  @param(value The byte value to set.)
  @param(extent The dimensions of the 3D volume to set.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemset3D(pitchedDevPtr: THipPitchedPtr; value: Int32; extent: THipExtent): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memset3D';

{
  @brief Asynchronously fills a 3D memory area with a constant value.
  
  @param(pitchedDevPtr A pitched pointer to the destination memory.)
  @param(value The byte value to set.)
  @param(extent The dimensions of the 3D volume to set.)
  @param(stream The stream for the asynchronous operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemset3DAsync(pitchedDevPtr: THipPitchedPtr; value: Int32; extent: THipExtent; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memset3DAsync';

{ @groupend }

{ @groupbegin(Memory Information) }

{
  @brief Queries the total and free memory on the current device.
  
  @param(free Returns the free memory in bytes.)
  @param(total Returns the total memory in bytes.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue)
}
function hipMemGetInfo(var free: SizeUInt; var total: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemGetInfo';

{
  @brief Gets the size of a memory allocation from a pointer within it.
  
  @param(ptr A pointer to the allocated memory.)
  @param(size Returns the size of the allocation in bytes.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipMemPtrGetInfo(ptr: Pointer; var size: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemPtrGetInfo';
{$ELSE}
  // This is an AMD-specific extension.
{$ENDIF}

{ @groupend }

{ @groupbegin(Array Management) }

{
  @brief Allocates a 1D or 2D array on the device.
  
  @param(array Returns the allocated HIP array.)
  @param(desc The channel format descriptor for the array.)
  @param(width The width of the array.)
  @param(height The height of the array (0 for 1D arrays).
  @param(flags Flags for the allocation.)
  
  @returns(hipSuccess, hipErrorOutOfMemory)
}
function hipMallocArray(var PArr: PHipArray; const desc: PHipChannelFormatDesc;
  width, height: SizeUInt; flags: UInt32 = hipArrayDefault): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MallocArray';
{
  @brief Creates a HIP array object.
  
  @param(pHandle Returns the created array handle.)
  @param(pAllocateArray The descriptor for the array to create.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipArrayCreate(var pHandle: hipArray_t; const pAllocateArray: PHIP_ARRAY_DESCRIPTOR): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ArrayCreate';

{
  @brief Destroys a HIP array object.
  
  @param(array The array to destroy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipArrayDestroy(hArr: hipArray_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ArrayDestroy';

{
  @brief Creates a 3D HIP array object.
  
  @param(array Returns the created 3D array.)
  @param(pAllocateArray The descriptor for the 3D array.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipArray3DCreate(var hArr: hipArray_t; const pAllocateArray: PHIP_ARRAY3D_DESCRIPTOR): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Array3DCreate';

{
  @brief Allocates a 3D pitched memory region on the device.
  
  @param(pitchedDevPtr Returns the pitched pointer to the allocated memory.)
  @param(extent The dimensions of the 3D allocation.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMalloc3D(var pitchedDevPtr: THipPitchedPtr; extent: THipExtent): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Malloc3D';

{
  @brief Frees a HIP array.
  
  @param(array The array to free.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized)
}
function hipFreeArray(hArr: hipArray_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'FreeArray';

{
  @brief Allocates a 3D array on the device.
  
  @param(array Returns the allocated 3D array.)
  @param(desc The channel format descriptor.)
  @param(extent The dimensions of the 3D array.)
  @param(flags Flags for the allocation.)
  
  @returns(hipSuccess, hipErrorOutOfMemory)
}
function hipMalloc3DArray(var hArr: hipArray_t; const desc: PHipChannelFormatDesc; extent: THipExtent; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Malloc3DArray';

{
  @brief Gets information about a specified HIP array.
  
  @param(desc Returns the channel format descriptor.)
  @param(extent Returns the array dimensions.)
  @param(flags Returns the array flags.)
  @param(array The array to query.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidHandle)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipArrayGetInfo(var desc: THipChannelFormatDesc; var extent: THipExtent; var flags: UInt32; hArr: hipArray_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ArrayGetInfo';
{$ELSE}
  // This is an AMD-specific extension.
{$ENDIF}

{
  @brief Gets the descriptor for a 1D or 2D array.
  
  @param(pArrayDescriptor Returns the array descriptor.)
  @param(array The array to query.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidHandle)
}
function hipArrayGetDescriptor(var pArrayDescriptor: THIP_ARRAY_DESCRIPTOR; hArr: hipArray_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ArrayGetDescriptor';

{
  @brief Gets the descriptor for a 3D array.
  
  @param(pArrayDescriptor Returns the 3D array descriptor.)
  @param(array The 3D array to query.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidHandle)
}
function hipArray3DGetDescriptor(var pArrayDescriptor: THIP_ARRAY3D_DESCRIPTOR; hArr: hipArray_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Array3DGetDescriptor';

{ @groupend }

{ @groupbegin(2D and 3D Memory Copy) }

{
  @brief Copies a 2D region of memory.
  
  @param(dst The destination pointer.)
  @param(dpitch The pitch of the destination memory.)
  @param(src The source pointer.)
  @param(spitch The pitch of the source memory.)
  @param(width The width of the copy in bytes.)
  @param(height The height of the copy.)
  @param(kind The kind of transfer.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpy2D(dst: Pointer; dpitch: SizeUInt; src: Pointer; spitch: SizeUInt; width: SizeUInt; height: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memcpy2D';

{
  @brief Copies a 2D region using a parameter structure.
  
  @param(pCopy The parameters for the memory copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpyParam2D(const PCopy: PHipMemcpy2D): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyParam2D';

{
  @brief Asynchronously copies a 2D region using a parameter structure.
  
  @param(pCopy The parameters for the memory copy.)
  @param(stream The stream for the asynchronous operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpyParam2DAsync(const PCopy: PHipMemcpy2D; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyParam2DAsync';

{
  @brief Asynchronously copies a 2D region of memory.
  
  @param(dst The destination pointer.)
  @param(dpitch The pitch of the destination memory.)
  @param(src The source pointer.)
  @param(spitch The pitch of the source memory.)
  @param(width The width of the copy in bytes.)
  @param(height The height of the copy.)
  @param(kind The kind of transfer.)
  @param(stream The stream for the asynchronous operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpy2DAsync(dst: Pointer; dpitch: SizeUInt; src: Pointer; spitch: SizeUInt; width: SizeUInt; height: SizeUInt; kind: THipMemcpyKind; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memcpy2DAsync';

{
  @brief Copies a 2D region to a HIP array.
  
  @param(dst The destination array.)
  @param(wOffset The destination X offset in elements.)
  @param(hOffset The destination Y offset in elements.)
  @param(src The source memory pointer.)
  @param(spitch The pitch of the source memory.)
  @param(width The width of the copy in bytes.)
  @param(height The height of the copy.)
  @param(kind The kind of transfer.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpy2DToArray(dst: hipArray_t; wOffset: SizeUInt; hOffset: SizeUInt; src: Pointer; spitch: SizeUInt; width: SizeUInt; height: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memcpy2DToArray';

{
  @brief Asynchronously copies a 2D region to a HIP array.
  
  @param(dst The destination array.)
  @param(wOffset The destination X offset.)
  @param(hOffset The destination Y offset.)
  @param(src The source memory pointer.)
  @param(spitch The pitch of the source memory.)
  @param(width The width of the copy in bytes.)
  @param(height The height of the copy.)
  @param(kind The kind of transfer.)
  @param(stream The stream for the asynchronous operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpy2DToArrayAsync(dst: hipArray_t; wOffset: SizeUInt; hOffset: SizeUInt; src: Pointer; spitch: SizeUInt; width: SizeUInt; height: SizeUInt; kind: THipMemcpyKind; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memcpy2DToArrayAsync';

{
  @brief Copies from one 2D array to another.
  
  @param(dst The destination array.)
  @param(wOffsetDst The destination X offset.)
  @param(hOffsetDst The destination Y offset.)
  @param(src The source array.)
  @param(wOffsetSrc The source X offset.)
  @param(hOffsetSrc The source Y offset.)
  @param(width The width of the copy in bytes.)
  @param(height The height of the copy.)
  @param(kind The kind of transfer.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpy2DArrayToArray(dst: hipArray_t; wOffsetDst: SizeUInt; hOffsetDst: SizeUInt; src: hipArray_const_t; wOffsetSrc: SizeUInt; hOffsetSrc: SizeUInt; width: SizeUInt; height: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memcpy2DArrayToArray';

{
  @brief Copies data to a 1D array [DEPRECATED].
  
  @param(dst The destination array.)
  @param(wOffset The destination X offset.)
  @param(hOffset The destination Y offset.)
  @param(src The source memory pointer.)
  @param(count The size in bytes to copy.)
  @param(kind The kind of transfer.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is deprecated.)
}
function hipMemcpyToArray(dst: hipArray_t; wOffset: SizeUInt; hOffset: SizeUInt; src: Pointer; count: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyToArray'; deprecated;

{
  @brief Copies data from a 1D array [DEPRECATED].
  
  @param(dst The destination memory pointer.)
  @param(srcArray The source array.)
  @param(wOffset The source X offset.)
  @param(hOffset The source Y offset.)
  @param(count The size in bytes to copy.)
  @param(kind The kind of transfer.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is deprecated.)
}
function hipMemcpyFromArray(dst: Pointer; src: hipArray_const_t; wOffset: SizeUInt; hOffset: SizeUInt; count: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyFromArray'; deprecated;

{
  @brief Copies a 2D region from a HIP array.
  
  @param(dst The destination memory pointer.)
  @param(dpitch The pitch of the destination memory.)
  @param(src The source array.)
  @param(wOffset The source X offset.)
  @param(hOffset The source Y offset.)
  @param(width The width of the copy in bytes.)
  @param(height The height of the copy.)
  @param(kind The kind of transfer.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpy2DFromArray(dst: Pointer; dpitch: SizeUInt; src: hipArray_const_t; wOffset: SizeUInt; hOffset: SizeUInt; width: SizeUInt; height: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memcpy2DFromArray';

{
  @brief Asynchronously copies a 2D region from a HIP array.
  
  @param(dst The destination memory pointer.)
  @param(dpitch The pitch of the destination memory.)
  @param(src The source array.)
  @param(wOffset The source X offset.)
  @param(hOffset The source Y offset.)
  @param(width The width of the copy in bytes.)
  @param(height The height of the copy.)
  @param(kind The kind of transfer.)
  @param(stream The stream for the asynchronous operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpy2DFromArrayAsync(dst: Pointer; dpitch: SizeUInt; src: hipArray_const_t; wOffset: SizeUInt; hOffset: SizeUInt; width: SizeUInt; height: SizeUInt; kind: THipMemcpyKind; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memcpy2DFromArrayAsync';

{
  @brief Copies data from a 1D array to host memory.
  
  @param(dst The destination host pointer.)
  @param(srcArray The source array.)
  @param(srcOffset The offset in the source array in bytes.)
  @param(count The number of bytes to copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpyAtoH(dst: Pointer; src: hipArray_t; srcOffset: SizeUInt; count: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyAtoH';

{
  @brief Copies data from host memory to a 1D array.
  
  @param(dstArray The destination array.)
  @param(dstOffset The offset in the destination array in bytes.)
  @param(srcHost The source host pointer.)
  @param(count The number of bytes to copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpyHtoA(dst: hipArray_t; dstOffset: SizeUInt; const srcHost: Pointer; count: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyHtoA';

{
  @brief Copies a 3D region of memory.
  
  @param(p The parameters for the 3D memory copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpy3D(const p: PHipMemcpy3DParms): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memcpy3D';

{
  @brief Asynchronously copies a 3D region of memory.
  
  @param(p The parameters for the 3D memory copy.)
  @param(stream The stream for the asynchronous operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMemcpy3DAsync(const p: PHipMemcpy3DParms; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'Memcpy3DAsync';

{
  @brief Copies a 3D region of memory (driver version).
  
  @param(pCopy The parameters for the 3D memory copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipDrvMemcpy3D(const pCopy: PHipMemcpy3DParms): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DrvMemcpy3D';
{
  @brief Asynchronously copies a 3D region of memory (driver version).
  
  @param(pCopy The parameters for the 3D memory copy.)
  @param(stream The stream for the asynchronous operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipDrvMemcpy3DAsync(const pCopy: PHipMemcpy3DParms; stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DrvMemcpy3DAsync';

{
  @brief Asynchronously copies from a 1D array to host memory.
  
  @param(dstHost Destination host pointer.)
  @param(srcArray Source array.)
  @param(srcOffset Offset in bytes of source array.)
  @param(ByteCount Size of memory copy in bytes.)
  @param(stream Stream identifier.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized)
  
  @see(hipMemcpyFromArray, hipMemcpyAtoH)
}
function hipMemcpyAtoHAsync(dstHost: Pointer; srchArr: hipArray_t; srcOffset: SizeUInt; ByteCount: SizeUInt; stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyAtoHAsync';

{
  @brief Asynchronously copies from host memory to a 1D array.
  
  @param(dstArray Destination array.)
  @param(dstOffset Offset in bytes of destination array.)
  @param(srcHost Source host pointer.)
  @param(ByteCount Size of memory copy in bytes.)
  @param(stream Stream identifier.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized)
  
  @see(hipMemcpyToArray, hipMemcpyHtoA)
}
function hipMemcpyHtoAAsync(dsthArr: hipArray_t; dstOffset: SizeUInt; const srcHost: Pointer; ByteCount: SizeUInt; stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyHtoAAsync';

{ @groupend }

{ @groupbegin(PeerToPeer Device Memory Access) }
{ @abstract(This section describes the Peer-to-Peer (P2P) device memory access functions of the HIP runtime API.)
  @warning(Peer-to-Peer support is experimental in some versions of HIP.) }

{
  @brief Determines if a device can directly access memory on a peer device.
  
  @param(canAccessPeer Returns 1 if access is possible, 0 otherwise.)
  @param(deviceId The device that would be accessing the memory.)
  @param(peerDeviceId The device where the memory is located.)
  
  @returns(hipSuccess, hipErrorInvalidDevice)
}
function hipDeviceCanAccessPeer(var canAccessPeer: Int32; deviceId: Int32; peerDeviceId: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceCanAccessPeer';

{
  @brief Enables direct access from the current device to a peer device's memory.
  
  @param(peerDeviceId The peer device to enable access to.)
  @param(flags Reserved for future use, must be zero.)
  
  @returns(hipSuccess, hipErrorInvalidDevice, hipErrorPeerAccessAlreadyEnabled)
}
function hipDeviceEnablePeerAccess(peerDeviceId: Int32; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceEnablePeerAccess';

{
  @brief Disables direct access from the current device to a peer device's memory.
  
  @param(peerDeviceId The peer device to disable access to.)
  
  @returns(hipSuccess, hipErrorPeerAccessNotEnabled)
}
function hipDeviceDisablePeerAccess(peerDeviceId: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceDisablePeerAccess';

{
  @brief Gets the base address and size of a memory allocation containing a given pointer.
  
  @param(pbase Returns the base pointer of the allocation.)
  @param(psize Returns the size of the allocation.)
  @param(dptr A device pointer within the allocation.)
  
  @returns(hipSuccess, hipErrorNotFound)
}
function hipMemGetAddressRange(var pbase: hipDeviceptr_t; var psize: SizeUInt; dptr: hipDeviceptr_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemGetAddressRange';

{
  @brief Copies memory between two different devices.
  
  @param(dst The destination device pointer.)
  @param(dstDeviceId The destination device ID.)
  @param(src The source device pointer.)
  @param(srcDeviceId The source device ID.)
  @param(sizeBytes The size of the copy in bytes.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidDevice)
}
function hipMemcpyPeer(dst: Pointer; dstDeviceId: Int32; src: Pointer; srcDeviceId: Int32; sizeBytes: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyPeer';

{
  @brief Asynchronously copies memory between two different devices.
  
  @param(dst The destination device pointer.)
  @param(dstDeviceId The destination device ID.)
  @param(src The source device pointer.)
  @param(srcDevice The source device ID.)
  @param(sizeBytes The size of the copy in bytes.)
  @param(stream The stream for the asynchronous operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidDevice)
}
function hipMemcpyPeerAsync(dst: Pointer; dstDeviceId: Int32; src: Pointer; srcDevice: Int32; sizeBytes: SizeUInt; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemcpyPeerAsync';

{ @groupend }

{ @groupbegin(Context Management - DEPRECATED) }
{ @abstract(This section describes deprecated context management functions. On AMD platforms, device and stream APIs are preferred.)
  @warning(These APIs are deprecated on AMD platforms and map to NVIDIA driver context APIs. Use with caution.) }

const
  HIP_DEPRECATED_MSG = 'This API is deprecated and might not be supported in future releases.';

{
  @brief Creates a context and sets it as current. [DEPRECATED]
  
  @param(ctx Returns the created context.)
  @param(flags Context creation flags.)
  @param(device The device handle.)
  
  @returns(hipSuccess)
}
function hipCtxCreate(var ctx: hipCtx_t; flags: UInt32; device: hipDevice_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxCreate'; deprecated;

{
  @brief Destroys a HIP context. [DEPRECATED]
  
  @param(ctx The context to destroy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipCtxDestroy(ctx: hipCtx_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxDestroy'; deprecated;

{
  @brief Pops the current context from the CPU thread's stack. [DEPRECATED]
  
  @param(ctx Returns the popped context.)
  
  @returns(hipSuccess, hipErrorInvalidContext)
}
function hipCtxPopCurrent(var ctx: hipCtx_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxPopCurrent'; deprecated;

{
  @brief Pushes a context onto the CPU thread's stack. [DEPRECATED]
  
  @param(ctx The context to push.)
  
  @returns(hipSuccess, hipErrorInvalidContext)
}
function hipCtxPushCurrent(ctx: hipCtx_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxPushCurrent'; deprecated;

{
  @brief Binds a HIP context to the calling CPU thread. [DEPRECATED]
  
  @param(ctx The context to set as current.)
  
  @returns(hipSuccess, hipErrorInvalidContext)
}
function hipCtxSetCurrent(ctx: hipCtx_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxSetCurrent'; deprecated;

{
  @brief Returns the HIP context bound to the calling CPU thread. [DEPRECATED]
  
  @param(ctx Returns the current context.)
  
  @returns(hipSuccess, hipErrorInvalidContext)
}
function hipCtxGetCurrent(var ctx: hipCtx_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxGetCurrent'; deprecated;

{
  @brief Returns the device ID for the current context. [DEPRECATED]
  
  @param(device Returns the device from the current context.)
  
  @returns(hipSuccess, hipErrorInvalidContext)
}
function hipCtxGetDevice(var device: hipDevice_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxGetDevice'; deprecated;

{
  @brief Returns the API version used to create a context. [DEPRECATED]
  
  @param(ctx The context to check.)
  @param(apiVersion Returns the API version.)
  
  @returns(hipSuccess)
}
function hipCtxGetApiVersion(ctx: hipCtx_t; var apiVersion: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxGetApiVersion'; deprecated;

{
  @brief Gets the cache configuration for the current context. [DEPRECATED]
  
  @param(cacheConfig Returns the cache configuration.)
  
  @returns(hipSuccess)
}
function hipCtxGetCacheConfig(var cacheConfig: THipFuncCache): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxGetCacheConfig'; deprecated;

{
  @brief Sets the cache configuration for the current context. [DEPRECATED]
  
  @param(cacheConfig The cache configuration to set.)
  
  @returns(hipSuccess)
}
function hipCtxSetCacheConfig(cacheConfig: THipFuncCache): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxSetCacheConfig'; deprecated;

{
  @brief Sets the shared memory configuration for the current context. [DEPRECATED]
  
  @param(config The shared memory configuration to set.)
  
  @returns(hipSuccess)
}
function hipCtxSetSharedMemConfig(config: THipSharedMemConfig): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxSetSharedMemConfig'; deprecated;

{
  @brief Gets the shared memory configuration for the current context. [DEPRECATED]
  
  @param(pConfig Returns the shared memory configuration.)
  
  @returns(hipSuccess)
}
function hipCtxGetSharedMemConfig(var pConfig: THipSharedMemConfig): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxGetSharedMemConfig'; deprecated;

{
  @brief Blocks until the context has completed all preceding tasks. [DEPRECATED]
  
  @returns(hipSuccess)
}
function hipCtxSynchronize(): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxSynchronize'; deprecated;

{
  @brief Returns the flags used for creating the current context. [DEPRECATED]
  
  @param(flags Returns the flags.)
  
  @returns(hipSuccess)
}
function hipCtxGetFlags(var flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxGetFlags'; deprecated;

{
  @brief Enables direct access to memory in a peer context. [DEPRECATED]
  
  @param(peerCtx The peer context.)
  @param(flags Flags (must be 0).)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorPeerAccessAlreadyEnabled)
}
function hipCtxEnablePeerAccess(peerCtx: hipCtx_t; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxEnablePeerAccess'; deprecated;

{
  @brief Disables direct access to memory in a peer context. [DEPRECATED]
  
  @param(peerCtx The peer context to disable.)
  
  @returns(hipSuccess, hipErrorPeerAccessNotEnabled)
}
function hipCtxDisablePeerAccess(peerCtx: hipCtx_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CtxDisablePeerAccess'; deprecated;

{
  @brief Gets the state of the primary context for a device. [DEPRECATED]
  
  @param(dev The device to query.)
  @param(flags Returns the flags of the primary context.)
  @param(active Returns 1 if the context is active, 0 otherwise.)
  
  @returns(hipSuccess)
}
function hipDevicePrimaryCtxGetState(dev: hipDevice_t; var flags: UInt32; var active: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DevicePrimaryCtxGetState'; deprecated;

{
  @brief Releases the primary context on the GPU. [DEPRECATED]
  
  @param(dev The device whose primary context is to be released.)
  
  @returns(hipSuccess)
}
function hipDevicePrimaryCtxRelease(dev: hipDevice_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DevicePrimaryCtxRelease'; deprecated;

{
  @brief Retains the primary context on the GPU, preventing it from being released. [DEPRECATED]
  
  @param(pctx Returns the handle of the primary context.)
  @param(dev The device whose primary context is to be retained.)
  
  @returns(hipSuccess)
}
function hipDevicePrimaryCtxRetain(var pctx: hipCtx_t; dev: hipDevice_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DevicePrimaryCtxRetain'; deprecated;

{
  @brief Resets the primary context on the GPU. [DEPRECATED]
  
  @param(dev The device whose primary context is to be reset.)
  
  @returns(hipSuccess)
}
function hipDevicePrimaryCtxReset(dev: hipDevice_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DevicePrimaryCtxReset'; deprecated;

{
  @brief Sets the flags for the primary context. [DEPRECATED]
  
  @param(dev The device for which the primary context flags are set.)
  @param(flags The new flags for the primary context.)
  
  @returns(hipSuccess, hipErrorContextAlreadyInUse)
}
function hipDevicePrimaryCtxSetFlags(dev: hipDevice_t; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DevicePrimaryCtxSetFlags'; deprecated;

{ @groupend }

{ @groupbegin(Module Management) }
{ @abstract(This section describes the module management functions for loading code objects and launching kernels.) }

{
  @brief Loads a code object from a file into a module.
  
  @param(module Returns the loaded module.)
  @param(fname The filename of the code object to load.)
  
  @returns(hipSuccess, hipErrorFileNotFound, hipErrorInvalidContext)
}
function hipModuleLoad(var module: hipModule_t; fname: PChar): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ModuleLoad';

{
  @brief Unloads a module.
  
  @param(module The module to unload.)
  
  @returns(hipSuccess, hipErrorInvalidResourceHandle)
}
function hipModuleUnload(module: hipModule_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ModuleUnload';

{
  @brief Extracts a function from a module.
  
  @param(function Returns the function handle.)
  @param(module The module to get the function from.)
  @param(kname The name of the function.)
  
  @returns(hipSuccess, hipErrorNotFound)
}
function hipModuleGetFunction(var &function: hipFunction_t; module: hipModule_t; kname: PChar): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ModuleGetFunction';

{
  @brief Finds attributes for a given function.
  
  @param(attr Returns the attributes of the function.)
  @param(func A pointer to the function.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction)
}
function hipFuncGetAttributes(var attr: THipFuncAttributes; func: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'FuncGetAttributes';

{
  @brief Finds a specific attribute for a given function.
  
  @param(value Returns the attribute's value.)
  @param(attrib The attribute to query.)
  @param(hfunc The function to get attributes from.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction)
}
function hipFuncGetAttribute(var value: Int32; attrib: THipFunctionAttribute; hfunc: hipFunction_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'FuncGetAttribute';

{
  @brief Gets a device function handle from a function symbol pointer.
  
  @param(functionPtr Returns the device function handle.)
  @param(symbolPtr A pointer to the device function symbol.)
  
  @returns(hipSuccess, hipErrorInvalidDeviceFunction)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipGetFuncBySymbol(var functionPtr: hipFunction_t; symbolPtr: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetFuncBySymbol';
{$ELSE}
  // This is an AMD-specific extension.
{$ENDIF}

{
  @brief Returns the handle of a texture reference from a module.
  
  @param(texRef Returns the texture reference.)
  @param(hmod The module containing the texture reference.)
  @param(name The name of the texture reference.)
  
  @returns(hipSuccess, hipErrorNotFound)
}
function hipModuleGetTexRef(var texRef: PTextureReference; hmod: hipModule_t; name: PChar): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ModuleGetTexRef';

{
  @brief Loads a module from a memory image.
  
  @param(module Returns the loaded module.)
  @param(image A pointer to the code object data.)
  
  @returns(hipSuccess, hipErrorOutOfMemory, hipErrorNotInitialized)
}
function hipModuleLoadData(var module: hipModule_t; image: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ModuleLoadData';

{
  @brief Loads a module from a memory image with JIT options.
  
  @param(module Returns the loaded module.)
  @param(image A pointer to the code object data.)
  @param(numOptions The number of JIT options.)
  @param(options An array of JIT options.)
  @param(optionValues An array of JIT option values.)
  
  @returns(hipSuccess, hipErrorOutOfMemory, hipErrorNotInitialized)
}
function hipModuleLoadDataEx(var module: hipModule_t; image: Pointer; numOptions: UInt32; options: PHipJitOption; optionValues: PPointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ModuleLoadDataEx';

{
  @brief Launches a kernel from a module.
  
  @param(f The kernel function handle to launch.)
  @param(gridDimX The X grid dimension.)
  @param(gridDimY The Y grid dimension.)
  @param(gridDimZ The Z grid dimension.)
  @param(blockDimX The X block dimension.)
  @param(blockDimY The Y block dimension.)
  @param(blockDimZ The Z block dimension.)
  @param(sharedMemBytes The amount of dynamic shared memory in bytes.)
  @param(stream The stream for the launch.)
  @param(kernelParams An array of pointers to kernel arguments.)
  @param(extra Extra launch options (e.g., HIP_LAUNCH_PARAM_END).)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized)
}
function hipModuleLaunchKernel(f: hipFunction_t; gridDimX: UInt32; gridDimY: UInt32; gridDimZ: UInt32; blockDimX: UInt32; blockDimY: UInt32; blockDimZ: UInt32; sharedMemBytes: UInt32; stream: hipStream_t; kernelParams: PPointer; extra: PPointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ModuleLaunchKernel';

{
  @brief Launches a cooperative kernel from a module.
  
  @param(f The kernel function handle.)
  @param(gridDimX, gridDimY, gridDimZ The grid dimensions.)
  @param(blockDimX, blockDimY, blockDimZ The block dimensions.)
  @param(sharedMemBytes The dynamic shared memory.)
  @param(stream The stream for the launch.)
  @param(kernelParams The kernel arguments.)
  
  @returns(hipSuccess, hipErrorLaunchFailure, etc.)
}
function hipModuleLaunchCooperativeKernel(f: hipFunction_t; gridDimX: UInt32; gridDimY: UInt32; gridDimZ: UInt32; blockDimX: UInt32; blockDimY: UInt32; blockDimZ: UInt32; sharedMemBytes: UInt32; stream: hipStream_t; kernelParams: PPointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ModuleLaunchCooperativeKernel';

{
  @brief Launches cooperative kernels on multiple devices.
  
  @param(launchParamsList A list of launch parameters, one per device.)
  @param(numDevices The size of the launchParamsList array.)
  @param(flags Flags to control launch behavior.)
  
  @returns(hipSuccess, hipErrorLaunchFailure, etc.)
}
function hipModuleLaunchCooperativeKernelMultiDevice(launchParamsList: PHipFunctionLaunchParams; numDevices: UInt32; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ModuleLaunchCooperativeKernelMultiDevice';

{
  @brief Launches a cooperative kernel.
  
  @param(f The kernel function pointer.)
  @param(gridDim The grid dimensions.)
  @param(blockDim The block dimensions.)
  @param(kernelParams The kernel arguments.)
  @param(sharedMemBytes The dynamic shared memory.)
  @param(stream The stream for the launch.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge)
}
function hipLaunchCooperativeKernel(f: Pointer; gridDim: TDim3; blockDim: TDim3; kernelParams: PPointer; sharedMemBytes: UInt32; stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'LaunchCooperativeKernel';

{
  @brief Launches cooperative kernels on multiple devices.
  
  @param(launchParamsList A list of launch parameters.)
  @param(numDevices The number of devices.)
  @param(flags Launch control flags.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge)
}
function hipLaunchCooperativeKernelMultiDevice(launchParamsList: PHipLaunchParams; numDevices: Int32; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'LaunchCooperativeKernelMultiDevice';

{
  @brief Launches kernels on multiple devices with extended controls.
  
  @param(launchParamsList A list of launch parameters.)
  @param(numDevices The number of devices.)
  @param(flags Launch control flags.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipExtLaunchMultiKernelMultiDevice(launchParamsList: PHipLaunchParams; numDevices: Int32; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ExtLaunchMultiKernelMultiDevice';
{$ELSE}
  // This is an AMD-specific extension.
{$ENDIF}

{ @groupend }

{ @groupbegin(Occupancy Calculation) }
{ @abstract(This section describes functions to calculate theoretical kernel occupancy.) }

{
  @brief Determines grid and block sizes for maximum occupancy for a kernel from a module.
  
  @param(gridSize Returns the minimum grid size.)
  @param(blockSize Returns the block size.)
  @param(f The kernel function handle.)
  @param(dynSharedMemPerBlk The dynamic shared memory usage in bytes.)
  @param(blockSizeLimit The maximum block size for the kernel (0 for no limit).)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipModuleOccupancyMaxPotentialBlockSize(var gridSize: Int32; var blockSize: Int32; f: hipFunction_t; dynSharedMemPerBlk: SizeUInt; blockSizeLimit: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ModuleOccupancyMaxPotentialBlockSize';

{
  @brief Determines grid and block sizes for maximum occupancy for a kernel from a module with flags.
  
  @param(gridSize Returns the minimum grid size.)
  @param(blockSize Returns the block size.)
  @param(f The kernel function handle.)
  @param(dynSharedMemPerBlk The dynamic shared memory usage in bytes.)
  @param(blockSizeLimit The maximum block size for the kernel (0 for no limit).)
  @param(flags Extra flags for the occupancy calculation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipModuleOccupancyMaxPotentialBlockSizeWithFlags(var gridSize: Int32; var blockSize: Int32; f: hipFunction_t; dynSharedMemPerBlk: SizeUInt; blockSizeLimit: Int32; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ModuleOccupancyMaxPotentialBlockSizeWithFlags';

{
  @brief Returns the maximum number of active blocks per multiprocessor for a function from a module.
  
  @param(numBlocks Returns the occupancy in number of blocks.)
  @param(f The kernel function handle.)
  @param(blockSize The block size.)
  @param(dynSharedMemPerBlk The dynamic shared memory usage in bytes.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(var numBlocks: Int32; f: hipFunction_t; blockSize: Int32; dynSharedMemPerBlk: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ModuleOccupancyMaxActiveBlocksPerMultiprocessor';

{
  @brief Returns the maximum number of active blocks per multiprocessor for a function from a module with flags.
  
  @param(numBlocks Returns the occupancy in number of blocks.)
  @param(f The kernel function handle.)
  @param(blockSize The block size.)
  @param(dynSharedMemPerBlk The dynamic shared memory usage in bytes.)
  @param(flags Extra flags for the occupancy calculation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(var numBlocks: Int32; f: hipFunction_t; blockSize: Int32; dynSharedMemPerBlk: SizeUInt; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags';

{
  @brief Returns the maximum number of active blocks per multiprocessor for a function pointer.
  
  @param(numBlocks Returns the occupancy in number of blocks.)
  @param(f A pointer to the kernel function.)
  @param(blockSize The block size.)
  @param(dynSharedMemPerBlk The dynamic shared memory usage in bytes.)
  
  @returns(hipSuccess, hipErrorInvalidDeviceFunction, hipErrorInvalidValue)
}
function hipOccupancyMaxActiveBlocksPerMultiprocessor(var numBlocks: Int32; f: Pointer; blockSize: Int32; dynSharedMemPerBlk: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'OccupancyMaxActiveBlocksPerMultiprocessor';

{
  @brief Returns the maximum number of active blocks per multiprocessor for a function pointer with flags.
  
  @param(numBlocks Returns the occupancy in number of blocks.)
  @param(f A pointer to the kernel function.)
  @param(blockSize The block size.)
  @param(dynSharedMemPerBlk The dynamic shared memory usage in bytes.)
  @param(flags Extra flags for the occupancy calculation.)
  
  @returns(hipSuccess, hipErrorInvalidDeviceFunction, hipErrorInvalidValue)
}
function hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(var numBlocks: Int32; f: Pointer; blockSize: Int32; dynSharedMemPerBlk: SizeUInt; flags: UInt32 = hipOccupancyDefault): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'OccupancyMaxActiveBlocksPerMultiprocessorWithFlags';

{
  @brief Determines grid and block sizes for maximum occupancy for a kernel function pointer.
  
  @param(gridSize Returns the minimum grid size.)
  @param(blockSize Returns the block size.)
  @param(f A pointer to the kernel function.)
  @param(dynSharedMemPerBlk The dynamic shared memory usage in bytes.)
  @param(blockSizeLimit The maximum block size for the kernel (0 for no limit).)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipOccupancyMaxPotentialBlockSize(var gridSize: Int32; var blockSize: Int32; f: Pointer; dynSharedMemPerBlk: SizeUInt; blockSizeLimit: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'OccupancyMaxPotentialBlockSize';

{ @groupend }

{ @groupbegin(Profiler Control - DEPRECATED) }
{ @abstract(This section describes deprecated profiler control functions. Use roctracer/rocTX instead.) }

{
  @brief Starts the profiler. [DEPRECATED]
  
  @returns(hipErrorNotSupported)
  
  @warning(This API is deprecated, use roctracer/rocTX instead.)
}
function hipProfilerStart: THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ProfilerStart'; deprecated;

{
  @brief Stops the profiler. [DEPRECATED]
  
  @returns(hipErrorNotSupported)
  
  @warning(This API is deprecated, use roctracer/rocTX instead.)
}
function hipProfilerStop: THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ProfilerStop'; deprecated;

{ @groupend }

{ @groupbegin(Clang Launch API) }
{ @abstract(This section describes the API to support the triple-chevron syntax for kernel launches.) }

{
  @brief Configures a kernel launch.
  
  @param(gridDim The grid dimensions.)
  @param(blockDim The block dimensions.)
  @param(sharedMem The dynamic shared memory size.)
  @param(stream The stream for the launch.)
  
  @returns(hipSuccess, hipErrorNotInitialized, hipErrorInvalidValue)
}
function hipConfigureCall(gridDim: TDim3; blockDim: TDim3; sharedMem: SizeUInt = 0; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ConfigureCall';

{
  @brief Sets a kernel argument for a configured launch.
  
  @param(arg A pointer to the argument data in host memory.)
  @param(size The size of the argument.)
  @param(offset The offset of the argument on the internal argument stack.)
  
  @returns(hipSuccess, hipErrorNotInitialized, hipErrorInvalidValue)
}
function hipSetupArgument(arg: Pointer; size: SizeUInt; offset: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'SetupArgument';

{
  @brief Launches a kernel previously configured with hipConfigureCall.
  
  @param(func The kernel function pointer to launch.)
  
  @returns(hipSuccess, hipErrorNotInitialized, hipErrorInvalidValue)
}
function hipLaunchByPtr(func: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'LaunchByPtr';

{
  @brief Pushes a kernel launch configuration onto a stack.
  
  @param(gridDim The grid dimensions.)
  @param(blockDim The block dimensions.)
  @param(sharedMem The dynamic shared memory size.)
  @param(stream The stream for the launch.)
  
  @returns(hipSuccess, hipErrorNotInitialized, hipErrorInvalidValue)
}
{$IFDEF HIP_PLATFORM_AMD}
function __hipPushCallConfiguration(gridDim: TDim3; blockDim: TDim3; sharedMem: SizeUInt = 0; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name '__hipPushCallConfiguration';
{$ELSE}
  // Internal HIP API not available on NVIDIA.
{$ENDIF}

{
  @brief Pops a kernel launch configuration from the stack.
  
  @param(gridDim Returns the grid dimensions.)
  @param(blockDim Returns the block dimensions.)
  @param(sharedMem Returns the shared memory size.)
  @param(stream Returns the stream.)
  
  @returns(hipSuccess, hipErrorNotInitialized, hipErrorInvalidValue)
}
{$IFDEF HIP_PLATFORM_AMD}
function __hipPopCallConfiguration(var gridDim: TDim3; var blockDim: TDim3; var sharedMem: SizeUInt; var stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name '__hipPopCallConfiguration';
{$ELSE}
  // Internal HIP API not available on NVIDIA.
{$ENDIF}

{
  @brief Launches a kernel with a C-style interface.
  
  @param(function_address The kernel function pointer.)
  @param(numBlocks The number of blocks in the grid.)
  @param(dimBlocks The dimensions of each block.)
  @param(args An array of pointers to the kernel arguments.)
  @param(sharedMemBytes The dynamic shared memory size.)
  @param(stream The stream for the launch.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipLaunchKernel(function_address: Pointer; numBlocks: TDim3; dimBlocks: TDim3; args: PPointer; sharedMemBytes: SizeUInt = 0; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'LaunchKernel';

{
  @brief Enqueues a host function to be called from a stream. [BETA]
  
  @param(stream The stream to enqueue the host function in.)
  @param(fn The host function to call.)
  @param(userData User data to be passed to the host function.)
  
  @returns(hipSuccess, hipErrorInvalidResourceHandle, hipErrorInvalidValue, hipErrorNotSupported)
  
  @warning(This API is marked as beta.)
}
function hipLaunchHostFunc(stream: hipStream_t; fn: THipHostFn; userData: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'LaunchHostFunc';

{
  @brief Copies a 2D region of memory with potentially unaligned source/destination.
  
  @param(pCopy The parameters for the memory copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipDrvMemcpy2DUnaligned(const pCopy: PHipMemcpy2D): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DrvMemcpy2DUnaligned';

{
  @brief Launches a kernel with extended parameters, including start/stop events.
  
  @param(function_address The kernel function pointer.)
  @param(numBlocks The number of blocks in the grid.)
  @param(dimBlocks The dimensions of each block.)
  @param(args An array of pointers to the kernel arguments.)
  @param(sharedMemBytes The dynamic shared memory size.)
  @param(stream The stream for the launch.)
  @param(startEvent An event to record the start time of the kernel.)
  @param(stopEvent An event to record the stop time of the kernel.)
  @param(flags Flags to control the launch behavior.)
  
  @returns(hipSuccess, hipErrorNotInitialized, hipErrorInvalidValue)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipExtLaunchKernel(function_address: Pointer; numBlocks: TDim3; dimBlocks: TDim3; args: PPointer; sharedMemBytes: SizeUInt; stream: hipStream_t; startEvent: hipEvent_t; stopEvent: hipEvent_t; flags: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ExtLaunchKernel';
{$ELSE}
  // This is an AMD-specific extension.
{$ENDIF}

{ @groupend }

{ @groupbegin(Texture Management) }
{ @abstract(This section describes the texture management functions of the HIP runtime API.) }

{
  @brief Creates a texture object.
  
  @param(pTexObject Returns the created texture object.)
  @param(pResDesc The resource descriptor.)
  @param(pTexDesc The texture descriptor.)
  @param(pResViewDesc The resource view descriptor.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported, hipErrorOutOfMemory)
}
function hipCreateTextureObject(var pTexObject: hipTextureObject_t; const pResDesc: PHipResourceDesc; const pTexDesc: PHipTextureDesc; const pResViewDesc: PHipResourceViewDesc): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CreateTextureObject';

{
  @brief Destroys a texture object.
  
  @param(textureObject The texture object to destroy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipDestroyTextureObject(textureObject: hipTextureObject_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DestroyTextureObject';

{
  @brief Gets the channel descriptor of a HIP array.
  
  @param(desc Returns the channel format descriptor.)
  @param(array The array to query.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipGetChannelDesc(var desc: THipChannelFormatDesc; hArr: hipArray_const_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetChannelDesc';

{
  @brief Gets the resource descriptor for a texture object.
  
  @param(pResDesc Returns the resource descriptor.)
  @param(textureObject The texture object to query.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipGetTextureObjectResourceDesc(var pResDesc: THipResourceDesc; textureObject: hipTextureObject_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetTextureObjectResourceDesc';

{
  @brief Gets the resource view descriptor for a texture object.
  
  @param(pResViewDesc Returns the resource view descriptor.)
  @param(textureObject The texture object to query.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipGetTextureObjectResourceViewDesc(var pResViewDesc: THipResourceViewDesc; textureObject: hipTextureObject_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetTextureObjectResourceViewDesc';

{
  @brief Gets the texture descriptor for a texture object.
  
  @param(pTexDesc Returns the texture descriptor.)
  @param(textureObject The texture object to query.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipGetTextureObjectTextureDesc(var pTexDesc: THipTextureDesc; textureObject: hipTextureObject_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetTextureObjectTextureDesc';

{
  @brief Creates a texture object (driver type version).
  
  @param(pTexObject Returns the created texture object.)
  @param(pResDesc The resource descriptor.)
  @param(pTexDesc The texture descriptor.)
  @param(pResViewDesc The resource view descriptor.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipTexObjectCreate(var pTexObject: hipTextureObject_t; const pResDesc: PHipResourceDesc; const pTexDesc: PHipTextureDesc; const pResViewDesc: PHipResourceViewDesc): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexObjectCreate';

{
  @brief Destroys a texture object (driver type version).
  
  @param(texObject The texture object to destroy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipTexObjectDestroy(texObject: hipTextureObject_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexObjectDestroy';

{
  @brief Gets the resource descriptor of a texture object (driver type version).
  
  @param(pResDesc Returns the resource descriptor.)
  @param(texObject The texture object to query.)
  
  @returns(hipSuccess, hipErrorNotSupported, hipErrorInvalidValue)
}
function hipTexObjectGetResourceDesc(var pResDesc: THipResourceDesc; texObject: hipTextureObject_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexObjectGetResourceDesc';

{
  @brief Gets the resource view descriptor of a texture object (driver type version).
  
  @param(pResViewDesc Returns the resource view descriptor.)
  @param(texObject The texture object to query.)
  
  @returns(hipSuccess, hipErrorNotSupported, hipErrorInvalidValue)
}
function hipTexObjectGetResourceViewDesc(var pResViewDesc: THipResourceViewDesc; texObject: hipTextureObject_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexObjectGetResourceViewDesc';

{
  @brief Gets the texture descriptor of a texture object (driver type version).
  
  @param(pTexDesc Returns the texture descriptor.)
  @param(texObject The texture object to query.)
  
  @returns(hipSuccess, hipErrorNotSupported, hipErrorInvalidValue)
}
function hipTexObjectGetTextureDesc(var pTexDesc: THipTextureDesc; texObject: hipTextureObject_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexObjectGetTextureDesc';

{
  @brief Allocates a mipmapped array on the device.
  
  @param(mipmappedArray Returns the allocated mipmapped array.)
  @param(desc The channel format descriptor.)
  @param(extent The dimensions of the base level of the mipmap.)
  @param(numLevels The number of mipmap levels to allocate.)
  @param(flags Flags for the allocation.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorMemoryAllocation)
}
function hipMallocMipmappedArray(var mipmappedArray: hipMipmappedArray_t; const desc: PHipChannelFormatDesc; extent: THipExtent; numLevels: UInt32; flags: UInt32 = 0): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MallocMipmappedArray';

{
  @brief Frees a mipmapped array.
  
  @param(mipmappedArray The mipmapped array to free.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipFreeMipmappedArray(mipmappedArray: hipMipmappedArray_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'FreeMipmappedArray';

{
  @brief Gets a specific mipmap level from a mipmapped array.
  
  @param(levelArray Returns the HIP array for the specified level.)
  @param(mipmappedArray The mipmapped array.)
  @param(level The mipmap level to retrieve.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipGetMipmappedArrayLevel(var levelhArr: hipArray_t; mipmappedArray: hipMipmappedArray_const_t; level: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetMipmappedArrayLevel';

{
  @brief Creates a mipmapped array.
  
  @param(pHandle Returns the created mipmapped array.)
  @param(pMipmappedArrayDesc The descriptor for the mipmapped array.)
  @param(numMipmapLevels The number of mipmap levels.)
  
  @returns(hipSuccess, hipErrorNotSupported, hipErrorInvalidValue)
}
function hipMipmappedArrayCreate(var pHandle: hipMipmappedArray_t; pMipmappedArrayDesc: PHIP_ARRAY3D_DESCRIPTOR; numMipmapLevels: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MipmappedArrayCreate';

{
  @brief Destroys a mipmapped array.
  
  @param(hMipmappedArray The mipmapped array to destroy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMipmappedArrayDestroy(hMipmappedArray: hipMipmappedArray_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MipmappedArrayDestroy';

{
  @brief Gets a specific mipmap level from a mipmapped array.
  
  @param(pLevelArray Returns the array for the specified level.)
  @param(hMipMappedArray The mipmapped array.)
  @param(level The mipmap level.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipMipmappedArrayGetLevel(var pLevelhArr: hipArray_t; hMipMappedArray: hipMipmappedArray_t; level: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MipmappedArrayGetLevel';

{ @groupend }

{ @groupbegin(Deprecated Texture Management) }

{
  @brief Binds a mipmapped array to a texture. [DEPRECATED]
  
  @param(tex A pointer to the texture reference.)
  @param(mipmappedArray The mipmapped array to bind.)
  @param(desc The channel format descriptor.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipBindTextureToMipmappedArray(tex: PTextureReference; mipmappedArray: hipMipmappedArray_const_t; const desc: PHipChannelFormatDesc): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'BindTextureToMipmappedArray'; deprecated;

{
  @brief Gets the texture reference associated with a symbol. [DEPRECATED]
  
  @param(texref Returns the texture reference.)
  @param(symbol A pointer to the symbol.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipGetTextureReference(var texref: PTextureReference; symbol: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetTextureReference'; deprecated;

{
  @brief Gets the border color of a texture reference. [DEPRECATED]
  
  @param(pBorderColor Returns the RGBA border color as a float array.)
  @param(texRef The texture reference.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipTexRefGetBorderColor(pBorderColor: PSingle; texRef: PTextureReference): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefGetBorderColor'; deprecated;

{
  @brief Gets the array bound to a texture reference. [DEPRECATED]
  
  @param(pArray Returns the bound array.)
  @param(texRef The texture reference.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipTexRefGetArray(var phArr: hipArray_t; texRef: PTextureReference): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefGetArray'; deprecated;

{
  @brief Sets the address mode for a texture reference. [DEPRECATED]
  
  @param(texRef The texture reference.)
  @param(dim The dimension (0 for x, 1 for y, 2 for z).)
  @param(am The address mode.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipTexRefSetAddressMode(texRef: PTextureReference; dim: Int32; am: THipTextureAddressMode): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefSetAddressMode'; deprecated;

{
  @brief Binds an array to a texture reference. [DEPRECATED]
  
  @param(tex The texture reference.)
  @param(array The array to bind.)
  @param(flags Flags for the binding.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipTexRefSetArray(tex: PTextureReference; hArr: hipArray_const_t; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefSetArray'; deprecated;

{
  @brief Sets the filter mode for a texture reference. [DEPRECATED]
  
  @param(texRef The texture reference.)
  @param(fm The filter mode.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipTexRefSetFilterMode(texRef: PTextureReference; fm: THipTextureFilterMode): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefSetFilterMode'; deprecated;

{
  @brief Sets the flags for a texture reference. [DEPRECATED]
  
  @param(texRef The texture reference.)
  @param(Flags The flags to set.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipTexRefSetFlags(texRef: PTextureReference; Flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefSetFlags'; deprecated;

{
  @brief Sets the format for a texture reference. [DEPRECATED]
  
  @param(texRef The texture reference.)
  @param(fmt The array format.)
  @param(NumPackedComponents The number of packed components.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipTexRefSetFormat(texRef: PTextureReference; fmt: THipArray_Format; NumPackedComponents: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefSetFormat'; deprecated;

{
  @brief Binds a memory area to a texture. [DEPRECATED]
  
  @param(offset Returns the byte offset.)
  @param(tex The texture reference.)
  @param(devPtr The device pointer to bind.)
  @param(desc The channel format descriptor.)
  @param(size The size of the memory area in bytes.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipBindTexture(var offset: SizeUInt; tex: PTextureReference; devPtr: Pointer; 
  const desc: PHipChannelFormatDesc; size: SizeUInt = High(SizeUInt)): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'BindTexture'; deprecated;

{
  @brief Binds a 2D memory area to a texture. [DEPRECATED]
  
  @param(offset Returns the byte offset.)
  @param(tex The texture reference.)
  @param(devPtr The device pointer to bind.)
  @param(desc The channel format descriptor.)
  @param(width The width of the 2D area.)
  @param(height The height of the 2D area.)
  @param(pitch The pitch of the 2D area.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipBindTexture2D(var offset: SizeUInt; tex: PTextureReference; devPtr: Pointer; 
  const desc: PHipChannelFormatDesc; width, height, pitch: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'BindTexture2D'; deprecated;

{
  @brief Binds a HIP array to a texture. [DEPRECATED]
  
  @param(tex The texture reference.)
  @param(array The array to bind.)
  @param(desc The channel format descriptor.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipBindTextureToArray(tex: PTextureReference; hArr: hipArray_const_t; 
  const desc: PHipChannelFormatDesc): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'BindTextureToArray'; deprecated;

{
  @brief Gets the alignment offset for a texture. [DEPRECATED]
  
  @param(offset Returns the alignment offset in bytes.)
  @param(texref The texture reference.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipGetTextureAlignmentOffset(var offset: SizeUInt; texref: PTextureReference): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetTextureAlignmentOffset'; deprecated;

{
  @brief Unbinds a texture. [DEPRECATED]
  
  @param(tex The texture reference to unbind.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipUnbindTexture(tex: PTextureReference): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'UnbindTexture'; deprecated;

{
  @brief Gets the address bound to a texture reference. [DEPRECATED]
  
  @param(dev_ptr Returns the device pointer.)
  @param(texRef The texture reference.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefGetAddress(var dev_ptr: hipDeviceptr_t; texRef: PTextureReference): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefGetAddress'; deprecated;

{
  @brief Gets the address mode of a texture reference. [DEPRECATED]
  
  @param(pam Returns the address mode.)
  @param(texRef The texture reference.)
  @param(dim The dimension to query.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefGetAddressMode(var pam: THipTextureAddressMode; texRef: PTextureReference; dim: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefGetAddressMode'; deprecated;

{
  @brief Gets the filter mode of a texture reference. [DEPRECATED]
  
  @param(pfm Returns the filter mode.)
  @param(texRef The texture reference.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefGetFilterMode(var pfm: THipTextureFilterMode; texRef: PTextureReference): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefGetFilterMode'; deprecated;

{
  @brief Gets the flags of a texture reference. [DEPRECATED]
  
  @param(pFlags Returns the flags.)
  @param(texRef The texture reference.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefGetFlags(var pFlags: UInt32; texRef: PTextureReference): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefGetFlags'; deprecated;

{
  @brief Gets the format of a texture reference. [DEPRECATED]
  
  @param(pFormat Returns the format.)
  @param(pNumChannels Returns the number of channels.)
  @param(texRef The texture reference.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefGetFormat(var pFormat: THipArray_Format; var pNumChannels: Int32; texRef: PTextureReference): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefGetFormat'; deprecated;

{
  @brief Gets the maximum anisotropy of a texture reference. [DEPRECATED]
  
  @param(pmaxAniso Returns the maximum anisotropy.)
  @param(texRef The texture reference.)
  
  @returns(hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefGetMaxAnisotropy(var pmaxAniso: Int32; texRef: PTextureReference): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefGetMaxAnisotropy'; deprecated;

{
  @brief Gets the mipmap filter mode of a texture reference. [DEPRECATED]
  
  @param(pfm Returns the mipmap filter mode.)
  @param(texRef The texture reference.)
  
  @returns(hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefGetMipmapFilterMode(var pfm: THipTextureFilterMode; texRef: PTextureReference): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefGetMipmapFilterMode'; deprecated;

{
  @brief Gets the mipmap level bias of a texture reference. [DEPRECATED]
  
  @param(pbias Returns the mipmap level bias.)
  @param(texRef The texture reference.)
  
  @returns(hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefGetMipmapLevelBias(var pbias: Single; texRef: PTextureReference): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefGetMipmapLevelBias'; deprecated;

{
  @brief Gets the min and max mipmap level clamps of a texture reference. [DEPRECATED]
  
  @param(pminMipmapLevelClamp Returns the minimum clamp.)
  @param(pmaxMipmapLevelClamp Returns the maximum clamp.)
  @param(texRef The texture reference.)
  
  @returns(hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefGetMipmapLevelClamp(var pminMipmapLevelClamp: Single; var pmaxMipmapLevelClamp: Single; texRef: PTextureReference): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefGetMipmapLevelClamp'; deprecated;

{
  @brief Gets the mipmapped array bound to a texture reference. [DEPRECATED]
  
  @param(pArray Returns the mipmapped array.)
  @param(texRef The texture reference.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefGetMipMappedArray(var pArray: hipMipmappedArray_t; texRef: PTextureReference): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefGetMipMappedArray'; deprecated;

{
  @brief Sets the address for a texture reference. [DEPRECATED]
  
  @param(ByteOffset Returns the byte offset.)
  @param(texRef The texture reference.)
  @param(dptr The device pointer to bind.)
  @param(bytes The size of the memory area.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipTexRefSetAddress(var ByteOffset: SizeUInt; texRef: PTextureReference; dptr: hipDeviceptr_t; bytes: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefSetAddress'; deprecated;

{
  @brief Binds an address as a 2D texture reference. [DEPRECATED]
  
  @param(texRef The texture reference.)
  @param(desc The array descriptor.)
  @param(dptr The device pointer to bind.)
  @param(Pitch The pitch in bytes.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefSetAddress2D(texRef: PTextureReference; const desc: PHIP_ARRAY_DESCRIPTOR; dptr: hipDeviceptr_t; Pitch: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefSetAddress2D'; deprecated;

{
  @brief Sets the maximum anisotropy for a texture reference. [DEPRECATED]
  
  @param(texRef The texture reference.)
  @param(maxAniso The maximum anisotropy value.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefSetMaxAnisotropy(texRef: PTextureReference; maxAniso: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefSetMaxAnisotropy'; deprecated;

{
  @brief Sets the border color for a texture reference. [DEPRECATED]
  
  @param(texRef The texture reference.)
  @param(pBorderColor A pointer to the RGBA border color.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefSetBorderColor(texRef: PTextureReference; pBorderColor: PSingle): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefSetBorderColor'; deprecated;

{
  @brief Sets the mipmap filter mode for a texture reference. [DEPRECATED]
  
  @param(texRef The texture reference.)
  @param(fm The filter mode.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefSetMipmapFilterMode(texRef: PTextureReference; fm: THipTextureFilterMode): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefSetMipmapFilterMode'; deprecated;

{
  @brief Sets the mipmap level bias for a texture reference. [DEPRECATED]
  
  @param(texRef The texture reference.)
  @param(bias The mipmap level bias value.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefSetMipmapLevelBias(texRef: PTextureReference; bias: Single): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefSetMipmapLevelBias'; deprecated;

{
  @brief Sets the mipmap level clamp for a texture reference. [DEPRECATED]
  
  @param(texRef The texture reference.)
  @param(minMipMapLevelClamp The minimum mipmap level clamp.)
  @param(maxMipMapLevelClamp The maximum mipmap level clamp.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipTexRefSetMipmapLevelClamp(texRef: PTextureReference; minMipMapLevelClamp: Single; maxMipMapLevelClamp: Single): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefSetMipmapLevelClamp'; deprecated;

{
  @brief Binds a mipmapped array to a texture reference. [DEPRECATED]
  
  @param(texRef The texture reference.)
  @param(mipmappedArray The mipmapped array to bind.)
  @param(Flags Flags for the binding.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipTexRefSetMipmappedArray(texRef: PTextureReference; mipmappedArray: hipMipmappedArray_t; Flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'TexRefSetMipmappedArray'; deprecated;

{ @groupend }


{ @groupbegin(Callback and Activity APIs) }

{
  @brief Returns the string name of a HIP API from its ID.
  
  @param(id The ID of a HIP API.)
  
  @returns(A null-terminated string with the name of the API.)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipApiName(id: UInt32): PChar;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ApiName';
{$ELSE}
  // This is an AMD-specific extension.
{$ENDIF}

{
  @brief Returns the kernel name from a function handle.
  
  @param(f The function handle.)
  
  @returns(A null-terminated string with the name of the kernel.)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipKernelNameRef(f: hipFunction_t): PChar;
  cdecl; external HIP_LIB_NAME name PREFIX + 'KernelNameRef';
{$ELSE}
  // This is only an AMD-specific extension.
{$ENDIF}

{
  @brief Retrieves the kernel name for a given host function pointer.
  
  @param(hostFunction A pointer to the host-side kernel stub function.)
  @param(stream The stream on which the kernel is executed.)
  
  @returns(A null-terminated string with the name of the kernel.)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipKernelNameRefByPtr(hostFunction: Pointer; stream: hipStream_t): PChar;
  cdecl; external HIP_LIB_NAME name PREFIX + 'KernelNameRefByPtr';
{$ELSE}
  // This is only an AMD-specific extension.
{$ENDIF}

{
  @brief Returns the device ID associated with a stream.
  
  @param(stream The stream to query.)
  
  @returns(The device ID.)
}
{$IFDEF HIP_PLATFORM_AMD}
function hipGetStreamDeviceId(stream: hipStream_t): Int32;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GetStreamDeviceId';
{$ELSE}
  // This is only an AMD-specific extension.
{$ENDIF}

{ @groupend }

{ @groupbegin(Graph Management) }
{ @abstract(This section describes graph management types & functions of the HIP runtime API.)
  @warning(Graph APIs are marked as beta and may have outstanding issues.) }

{
  @brief Begins graph capture on a stream.
  
  @param(stream The stream to initiate capture on.)
  @param(mode The capture mode.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipStreamBeginCapture(stream: hipStream_t; mode: THipStreamCaptureMode): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamBeginCapture';

{
  @brief Ends capture on a stream, returning the captured graph.
  
  @param(stream The stream to end capture on.)
  @param(pGraph Returns the captured graph.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipStreamEndCapture(stream: hipStream_t; var pGraph: hipGraph_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamEndCapture';

{
  @brief Creates a graph.
  
  @param(pGraph Returns the created graph.)
  @param(flags Flags for graph creation (must be 0).)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorMemoryAllocation)
}
function hipGraphCreate(var pGraph: hipGraph_t; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphCreate';

{
  @brief Destroys a graph.
  
  @param(graph The graph to destroy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipGraphDestroy(graph: hipGraph_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphDestroy';

{
  @brief Creates an executable graph from a graph.
  
  @param(pGraphExec Returns the instantiated executable graph.)
  @param(graph The graph to instantiate.)
  @param(pErrorNode Returns the node that caused an error, if any.)
  @param(pLogBuffer A buffer to store log messages.)
  @param(bufferSize The size of the log buffer.)
  
  @returns(hipSuccess, hipErrorOutOfMemory)
}
function hipGraphInstantiate(var pGraphExec: hipGraphExec_t; graph: hipGraph_t; var pErrorNode: hipGraphNode_t; pLogBuffer: PChar; bufferSize: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphInstantiate';

{
  @brief Launches an executable graph in a stream.
  
  @param(graphExec The executable graph to launch.)
  @param(stream The stream in which to launch the graph.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipGraphLaunch(graphExec: hipGraphExec_t; stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphLaunch';

{
  @brief Destroys an executable graph.
  
  @param(graphExec The executable graph to destroy.)
  
  @returns(hipSuccess)
}
function hipGraphExecDestroy(graphExec: hipGraphExec_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecDestroy';

{
  @brief Adds dependency edges to a graph.
  
  @param(graph The graph to which to add dependencies.)
  @param(from An array of nodes that provide the dependencies.)
  @param(to An array of nodes that the dependencies are provided for.)
  @param(numDependencies The number of dependencies to add.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddDependencies(graph: hipGraph_t; const from: PhipGraphNode_t; const into: PhipGraphNode_t; numDependencies: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddDependencies';

{
  @brief Removes dependency edges from a graph.
  
  @param(graph The graph from which to remove dependencies.)
  @param(from An array of nodes that provide the dependencies.)
  @param(to An array of nodes that the dependencies are provided for.)
  @param(numDependencies The number of dependencies to remove.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphRemoveDependencies(graph: hipGraph_t; const from: PhipGraphNode_t; const into: PhipGraphNode_t; numDependencies: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphRemoveDependencies';

{
  @brief Returns a graph's dependency edges.
  
  @param(graph The graph from which to get the edges.)
  @param(from Pointer to an array to return the source nodes of the edges.)
  @param(to Pointer to an array to return the destination nodes of the edges.)
  @param(numEdges On input, the size of the from and to arrays; on output, the number of edges returned.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @note(from and to may both be nil, in which case this function only returns the number of edges.)
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphGetEdges(graph: hipGraph_t; from: PhipGraphNode_t; into: PhipGraphNode_t; var numEdges: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphGetEdges';

{
  @brief Returns a graph's nodes.
  
  @param(graph The graph from which to get the nodes.)
  @param(nodes An array to return the graph's nodes.)
  @param(numNodes On input, the size of the nodes array; on output, the number of nodes returned.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @note(nodes may be nil, in which case this function only returns the number of nodes.)
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphGetNodes(graph: hipGraph_t; nodes: PhipGraphNode_t; var numNodes: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphGetNodes';

{
  @brief Returns a graph's root nodes.
  
  @param(graph The graph from which to get the root nodes.)
  @param(pRootNodes An array to return the graph's root nodes.)
  @param(pNumRootNodes On input, the size of the pRootNodes array; on output, the number of root nodes returned.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @note(pRootNodes may be nil, in which case this function only returns the number of root nodes.)
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphGetRootNodes(graph: hipGraph_t; pRootNodes: PhipGraphNode_t; var pNumRootNodes: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphGetRootNodes';

{
  @brief Returns a node's dependencies.
  
  @param(node The node from which to get the dependencies.)
  @param(pDependencies An array to return the dependencies.)
  @param(pNumDependencies On input, the size of the pDependencies array; on output, the number of dependencies returned.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphNodeGetDependencies(node: hipGraphNode_t; pDependencies: PhipGraphNode_t; var pNumDependencies: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphNodeGetDependencies';

{
  @brief Returns a node's dependent nodes.
  
  @param(node The node from which to get the dependent nodes.)
  @param(pDependentNodes An array to return the dependent nodes.)
  @param(pNumDependentNodes On input, the size of the pDependentNodes array; on output, the number of dependent nodes returned.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphNodeGetDependentNodes(node: hipGraphNode_t; pDependentNodes: PhipGraphNode_t; var pNumDependentNodes: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphNodeGetDependentNodes';

{
  @brief Returns a node's type.
  
  @param(node The node to get the type of.)
  @param(pType Pointer to return the type.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphNodeGetType(node: hipGraphNode_t; var pType: THipGraphNodeType): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphNodeGetType';

{
  @brief Removes a node from the graph.
  
  @param(node The graph node to remove.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphDestroyNode(node: hipGraphNode_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphDestroyNode';

{
  @brief Clones a graph.
  
  @param(pGraphClone Returns the newly created cloned graph.)
  @param(originalGraph The original graph to clone.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorMemoryAllocation)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphClone(var pGraphClone: hipGraph_t; originalGraph: hipGraph_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphClone';

{
  @brief Finds a cloned version of a node in a cloned graph.
  
  @param(pNode Returns the cloned node.)
  @param(originalNode The original node handle.)
  @param(clonedGraph The cloned graph to query.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphNodeFindInClone(var pNode: hipGraphNode_t; originalNode: hipGraphNode_t; clonedGraph: hipGraph_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphNodeFindInClone';

{
  @brief Creates an executable graph from a graph with additional flags.
  
  @param(pGraphExec Pointer to the instantiated executable graph that is created.)
  @param(graph The graph to instantiate.)
  @param(flags Flags to control instantiation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues. It does not currently support any flags.)
}
function hipGraphInstantiateWithFlags(var pGraphExec: hipGraphExec_t; graph: hipGraph_t; flags: UInt64): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphInstantiateWithFlags';
  
{
  @brief Creates an executable graph from a graph with detailed parameters.
  
  @param(pGraphExec Pointer to the instantiated executable graph.)
  @param(graph The graph to instantiate.)
  @param(instantiateParams Graph instantiation parameters.)

  @returns(hipSuccess, hipErrorInvalidValue)

  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphInstantiateWithParams(var pGraphExec: hipGraphExec_t; graph: hipGraph_t; instantiateParams: PHipGraphInstantiateParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphInstantiateWithParams';

{
  @brief Uploads an executable graph's data to the device.
  
  @param(graphExec The executable graph to upload.)
  @param(stream The stream to use for the upload.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphUpload(graphExec: hipGraphExec_t; stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphUpload';

{
  @brief Checks if an executable graph can be updated and performs the update.
  
  @param(hGraphExec The executable graph to update.)
  @param(hGraph The graph containing the updated parameters.)
  @param(hErrorNode_out Returns the node which caused the update to be forbidden.)
  @param(updateResult_out Returns whether the graph update was permitted.)
  
  @returns(hipSuccess, hipErrorGraphExecUpdateFailure)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecUpdate(hGraphExec: hipGraphExec_t; hGraph: hipGraph_t; var hErrorNode_out: hipGraphNode_t; var updateResult_out: THipGraphExecUpdateResult): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecUpdate';

{
  @brief Returns the flags used to instantiate an executable graph.
  
  @param(graphExec The executable graph to get flags from.)
  @param(flags Returns the flags used for instantiation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecGetFlags(graphExec: hipGraphExec_t; var flags: UInt64): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecGetFlags';

{
  @brief Creates a kernel execution node and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created graph node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of nodes that are dependencies for this node.)
  @param(numDependencies The number of dependencies.)
  @param(pNodeParams The parameters for the kernel node.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddKernelNode(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; const pNodeParams: PHipKernelNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddKernelNode';

{
  @brief Gets a kernel node's parameters.
  
  @param(node The node to get parameters from.)
  @param(pNodeParams A pointer to a structure where the parameters will be returned.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphKernelNodeGetParams(node: hipGraphNode_t; var pNodeParams: THipKernelNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphKernelNodeGetParams';

{
  @brief Sets a kernel node's parameters.
  
  @param(node The node to set parameters for.)
  @param(pNodeParams The parameters to set.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphKernelNodeSetParams(node: hipGraphNode_t; const pNodeParams: PHipKernelNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphKernelNodeSetParams';

{
  @brief Sets the parameters for a kernel node in an executable graph.
  
  @param(hGraphExec The executable graph in which the node exists.)
  @param(node The node to set parameters for.)
  @param(pNodeParams The parameters to set.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecKernelNodeSetParams(hGraphExec: hipGraphExec_t; node: hipGraphNode_t; const pNodeParams: PHipKernelNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecKernelNodeSetParams';

{
  @brief Sets an attribute of a kernel node.
  
  @param(hNode The node to set an attribute for.)
  @param(attr The attribute to set.)
  @param(value A pointer to the value of the attribute.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @see(THipKernelNodeAttrID)
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphKernelNodeSetAttribute(hNode: hipGraphNode_t; attr: THipKernelNodeAttrID; const value: PHipKernelNodeAttrValue): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphKernelNodeSetAttribute';

{
  @brief Gets an attribute of a kernel node.
  
  @param(hNode The node to get an attribute from.)
  @param(attr The attribute to get.)
  @param(value A pointer to a location to store the value of the attribute.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @see(THipKernelNodeAttrID)
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphKernelNodeGetAttribute(hNode: hipGraphNode_t; attr: THipKernelNodeAttrID; var value: THipKernelNodeAttrValue): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphKernelNodeGetAttribute';

{
  @brief Copies attributes from a source kernel node to a destination kernel node.
  
  @param(hDst The destination node.)
  @param(hSrc The source node.)
  
  @returns(hipSuccess, hipErrorInvalidContext)
  
  @note(Both nodes must belong to the same context.)
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphKernelNodeCopyAttributes(hSrc: hipGraphNode_t; hDst: hipGraphNode_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphKernelNodeCopyAttributes';
  
// --- Memcpy Node Functions ---

{
  @brief Creates a memcpy node and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created graph node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  @param(pCopyParams The parameters for the memory copy operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddMemcpyNode(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; const pCopyParams: PHipMemcpy3DParms): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddMemcpyNode';

{
  @brief Gets a memcpy node's parameters.
  
  @param(node The node to get parameters from.)
  @param(pNodeParams A pointer to a structure where the parameters will be stored.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphMemcpyNodeGetParams(node: hipGraphNode_t; var pNodeParams: THipMemcpy3DParms): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphMemcpyNodeGetParams';

{
  @brief Sets a memcpy node's parameters.
  
  @param(node The node to set parameters for.)
  @param(pNodeParams The parameters to set for the memcpy operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphMemcpyNodeSetParams(node: hipGraphNode_t; const pNodeParams: PHipMemcpy3DParms): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphMemcpyNodeSetParams';

{
  @brief Sets the parameters for a memcpy node in an executable graph.
  
  @param(hGraphExec The executable graph containing the node.)
  @param(node The node to update.)
  @param(pNodeParams The new parameters for the memcpy operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecMemcpyNodeSetParams(hGraphExec: hipGraphExec_t; node: hipGraphNode_t; var pNodeParams: THipMemcpy3DParms): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecMemcpyNodeSetParams';

{
  @brief Creates a 1D memcpy node and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created graph node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  @param(dst The destination memory address.)
  @param(src The source memory address.)
  @param(count The size of the memory copy in bytes.)
  @param(kind The type of memory copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddMemcpyNode1D(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; dst: Pointer; const src: Pointer; count: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddMemcpyNode1D';

{
  @brief Sets a memcpy node's parameters to perform a 1D copy.
  
  @param(node The node to update.)
  @param(dst The destination memory address.)
  @param(src The source memory address.)
  @param(count The size of the memory copy in bytes.)
  @param(kind The type of memory copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphMemcpyNodeSetParams1D(node: hipGraphNode_t; dst: Pointer; const src: Pointer; count: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphMemcpyNodeSetParams1D';

{
  @brief Sets the parameters for a 1D memcpy node in an executable graph.
  
  @param(hGraphExec The executable graph containing the node.)
  @param(node The node to update.)
  @param(dst The destination memory address.)
  @param(src The source memory address.)
  @param(count The size of the memory copy in bytes.)
  @param(kind The type of memory copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecMemcpyNodeSetParams1D(hGraphExec: hipGraphExec_t; node: hipGraphNode_t; dst: Pointer; const src: Pointer; count: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecMemcpyNodeSetParams1D';

{
  @brief Creates a memcpy node to copy from a symbol and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created graph node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  @param(dst The destination memory address.)
  @param(symbol The source device symbol.)
  @param(count The size of the memory copy in bytes.)
  @param(offset The offset from the start of the symbol in bytes.)
  @param(kind The type of memory copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddMemcpyNodeFromSymbol(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; dst: Pointer; const symbol: Pointer; count: SizeUInt; offset: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddMemcpyNodeFromSymbol';

{
  @brief Sets a memcpy node's parameters to copy from a symbol.
  
  @param(node The node to update.)
  @param(dst The destination memory address.)
  @param(symbol The source device symbol.)
  @param(count The size of the memory copy in bytes.)
  @param(offset The offset from the start of the symbol in bytes.)
  @param(kind The type of memory copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphMemcpyNodeSetParamsFromSymbol(node: hipGraphNode_t; dst: Pointer; const symbol: Pointer; count: SizeUInt; offset: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphMemcpyNodeSetParamsFromSymbol';

{
  @brief Sets the parameters for a memcpy node (from symbol) in an executable graph.
  
  @param(hGraphExec The executable graph containing the node.)
  @param(node The node to update.)
  @param(dst The destination memory address.)
  @param(symbol The source device symbol.)
  @param(count The size of the memory copy in bytes.)
  @param(offset The offset from the start of the symbol in bytes.)
  @param(kind The type of memory copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec: hipGraphExec_t; node: hipGraphNode_t; dst: Pointer; const symbol: Pointer; count: SizeUInt; offset: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecMemcpyNodeSetParamsFromSymbol';

{
  @brief Creates a memcpy node to copy to a symbol and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created graph node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  @param(symbol The destination device symbol.)
  @param(src The source memory address.)
  @param(count The size of the memory copy in bytes.)
  @param(offset The offset from the start of the symbol in bytes.)
  @param(kind The type of memory copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddMemcpyNodeToSymbol(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; const symbol: Pointer; const src: Pointer; count: SizeUInt; offset: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddMemcpyNodeToSymbol';

{
  @brief Sets a memcpy node's parameters to copy to a symbol.
  
  @param(node The node to update.)
  @param(symbol The destination device symbol.)
  @param(src The source memory address.)
  @param(count The size of the memory copy in bytes.)
  @param(offset The offset from the start of the symbol in bytes.)
  @param(kind The type of memory copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphMemcpyNodeSetParamsToSymbol(node: hipGraphNode_t; const symbol: Pointer; const src: Pointer; count: SizeUInt; offset: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphMemcpyNodeSetParamsToSymbol';

{
  @brief Sets the parameters for a memcpy node (to symbol) in an executable graph.
  
  @param(hGraphExec The executable graph containing the node.)
  @param(node The node to update.)
  @param(symbol The destination device symbol.)
  @param(src The source memory address.)
  @param(count The size of the memory copy in bytes.)
  @param(offset The offset from the start of the symbol in bytes.)
  @param(kind The type of memory copy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec: hipGraphExec_t; node: hipGraphNode_t; const symbol: Pointer; const src: Pointer; count: SizeUInt; offset: SizeUInt; kind: THipMemcpyKind): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecMemcpyNodeSetParamsToSymbol';

// --- Memset Node Functions ---

{
  @brief Creates a memset node and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created graph node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  @param(pMemsetParams The parameters for the memset operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddMemsetNode(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; const pMemsetParams: PHipMemsetParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddMemsetNode';

{
  @brief Gets a memset node's parameters.
  
  @param(node The node to get parameters from.)
  @param(pNodeParams A pointer to a structure where the parameters will be stored.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphMemsetNodeGetParams(node: hipGraphNode_t; var pNodeParams: THipMemsetParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphMemsetNodeGetParams';

{
  @brief Sets a memset node's parameters.
  
  @param(node The node to set parameters for.)
  @param(pNodeParams The parameters to set for the memset operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphMemsetNodeSetParams(node: hipGraphNode_t; const pNodeParams: PHipMemsetParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphMemsetNodeSetParams';

{
  @brief Sets the parameters for a memset node in an executable graph.
  
  @param(hGraphExec The executable graph containing the node.)
  @param(node The node to update.)
  @param(pNodeParams The new parameters for the memset operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecMemsetNodeSetParams(hGraphExec: hipGraphExec_t; node: hipGraphNode_t; const pNodeParams: PHipMemsetParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecMemsetNodeSetParams';  

// --- Host Node Functions ---

{
  @brief Creates a host execution node and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created graph node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  @param(pNodeParams The parameters for the host node, including the callback function.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddHostNode(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; const pNodeParams: PHipHostNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddHostNode';

{
  @brief Gets a host node's parameters.
  
  @param(node The node to get parameters from.)
  @param(pNodeParams A pointer to a structure where the parameters will be stored.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphHostNodeGetParams(node: hipGraphNode_t; var pNodeParams: THipHostNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphHostNodeGetParams';

{
  @brief Sets a host node's parameters.
  
  @param(node The node to set parameters for.)
  @param(pNodeParams The parameters to set for the host node.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphHostNodeSetParams(node: hipGraphNode_t; const pNodeParams: PHipHostNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphHostNodeSetParams';

{
  @brief Sets the parameters for a host node in an executable graph.
  
  @param(hGraphExec The executable graph containing the node.)
  @param(node The node to update.)
  @param(pNodeParams The new parameters for the host node.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecHostNodeSetParams(hGraphExec: hipGraphExec_t; node: hipGraphNode_t; const pNodeParams: PHipHostNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecHostNodeSetParams';

// --- Event Node Functions ---

{
  @brief Creates an event record node and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created graph node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  @param(event The event to be recorded by this node.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddEventRecordNode(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; event: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddEventRecordNode';

{
  @brief Returns the event associated with an event record node.
  
  @param(node The event record node to query.)
  @param(event_out Returns the event associated with the node.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphEventRecordNodeGetEvent(node: hipGraphNode_t; var event_out: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphEventRecordNodeGetEvent';

{
  @brief Sets the event for an event record node.
  
  @param(node The event record node to update.)
  @param(event The new event to associate with the node.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphEventRecordNodeSetEvent(node: hipGraphNode_t; event: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphEventRecordNodeSetEvent';

{
  @brief Sets the event for an event record node in an executable graph.
  
  @param(hGraphExec The executable graph containing the node.)
  @param(hNode The node to update.)
  @param(event The new event to associate with the node.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecEventRecordNodeSetEvent(hGraphExec: hipGraphExec_t; hNode: hipGraphNode_t; event: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecEventRecordNodeSetEvent';

{
  @brief Creates an event wait node and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created graph node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  @param(event The event this node will wait on.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddEventWaitNode(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; event: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddEventWaitNode';

{
  @brief Returns the event associated with an event wait node.
  
  @param(node The event wait node to query.)
  @param(event_out Returns the event associated with the node.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphEventWaitNodeGetEvent(node: hipGraphNode_t; var event_out: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphEventWaitNodeGetEvent';

{
  @brief Sets the event for an event wait node.
  
  @param(node The event wait node to update.)
  @param(event The new event for the node to wait on.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphEventWaitNodeSetEvent(node: hipGraphNode_t; event: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphEventWaitNodeSetEvent';

{
  @brief Sets the event for an event wait node in an executable graph.
  
  @param(hGraphExec The executable graph containing the node.)
  @param(hNode The node to update.)
  @param(event The new event for the node to wait on.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecEventWaitNodeSetEvent(hGraphExec: hipGraphExec_t; hNode: hipGraphNode_t; event: hipEvent_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecEventWaitNodeSetEvent';

// --- Child Graph Node Functions ---

{
  @brief Creates a child graph node and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created graph node.)
  @param(graph The parent graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  @param(childGraph The graph to clone into this node.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddChildGraphNode(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; childGraph: hipGraph_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddChildGraphNode';

{
  @brief Gets a handle to the embedded graph of a child graph node.
  
  @param(node The child graph node to query.)
  @param(pGraph Returns a pointer to the embedded graph.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphChildGraphNodeGetGraph(node: hipGraphNode_t; var pGraph: hipGraph_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphChildGraphNodeGetGraph';

{
  @brief Sets the child graph for a node in an executable graph.
  
  @param(hGraphExec The executable graph containing the node.)
  @param(node The node to update.)
  @param(childGraph The new child graph with updated parameters.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecChildGraphNodeSetParams(hGraphExec: hipGraphExec_t; node: hipGraphNode_t; childGraph: hipGraph_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecChildGraphNodeSetParams';
 
// --- Empty Node Functions ---

{
  @brief Creates an empty node and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created empty node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @note(An empty node is useful as a join point for dependencies.)
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddEmptyNode(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddEmptyNode';

// --- Memory Allocation and Free Node Functions ---

{
  @brief Creates a memory allocation node and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created memory allocation node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  @param(pNodeParams The parameters for the memory allocation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddMemAllocNode(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; var pNodeParams: THipMemAllocNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddMemAllocNode';

{
  @brief Returns the parameters for a memory allocation node.
  
  @param(node The memory allocation node to query.)
  @param(pNodeParams A pointer to a structure where the parameters will be stored.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphMemAllocNodeGetParams(node: hipGraphNode_t; var pNodeParams: THipMemAllocNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphMemAllocNodeGetParams';

{
  @brief Creates a memory free node and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created memory free node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  @param(dev_ptr The device pointer to be freed by this node.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddMemFreeNode(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; dev_ptr: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddMemFreeNode';

{
  @brief Returns the device pointer from a memory free node.
  
  @param(node The memory free node to query.)
  @param(dev_ptr Returns the device pointer to be freed.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphMemFreeNodeGetParams(node: hipGraphNode_t; var dev_ptr: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphMemFreeNodeGetParams';

// --- External Semaphore Node Functions ---

{
  @brief Creates an external semaphore signal node and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created graph node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  @param(nodeParams The parameters for the external semaphore signal operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddExternalSemaphoresSignalNode(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; const nodeParams: PHipExternalSemaphoreSignalNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddExternalSemaphoresSignalNode';

{
  @brief Gets the parameters of an external semaphore signal node.
  
  @param(hNode The node to get parameters from.)
  @param(params_out A pointer to a structure where the parameters will be stored.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExternalSemaphoresSignalNodeGetParams(hNode: hipGraphNode_t; var params_out: THipExternalSemaphoreSignalNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExternalSemaphoresSignalNodeGetParams';
  
{
  @brief Sets the parameters for an external semaphore signal node.
  
  @param(hNode The node to set parameters for.)
  @param(nodeParams The parameters to set.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExternalSemaphoresSignalNodeSetParams(hNode: hipGraphNode_t; const nodeParams: PHipExternalSemaphoreSignalNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExternalSemaphoresSignalNodeSetParams';

{
  @brief Sets the parameters for an external semaphore signal node in an executable graph.
  
  @param(hGraphExec The executable graph containing the node.)
  @param(hNode The node to update.)
  @param(nodeParams The new parameters for the operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec: hipGraphExec_t; hNode: hipGraphNode_t; const nodeParams: PHipExternalSemaphoreSignalNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecExternalSemaphoresSignalNodeSetParams';

{
  @brief Creates an external semaphore wait node and adds it to a graph.
  
  @param(pGraphNode Returns the pointer to the created graph node.)
  @param(graph The graph to which the node will be added.)
  @param(pDependencies An array of dependency nodes.)
  @param(numDependencies The number of dependencies.)
  @param(nodeParams The parameters for the external semaphore wait operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphAddExternalSemaphoresWaitNode(var pGraphNode: hipGraphNode_t; graph: hipGraph_t; const pDependencies: PhipGraphNode_t; numDependencies: SizeUInt; const nodeParams: PHipExternalSemaphoreWaitNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphAddExternalSemaphoresWaitNode';

{
  @brief Gets the parameters of an external semaphore wait node.
  
  @param(hNode The node to get parameters from.)
  @param(params_out A pointer to a structure where the parameters will be stored.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExternalSemaphoresWaitNodeGetParams(hNode: hipGraphNode_t; var params_out: THipExternalSemaphoreWaitNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExternalSemaphoresWaitNodeGetParams';

{
  @brief Sets the parameters for an external semaphore wait node.
  
  @param(hNode The node to set parameters for.)
  @param(nodeParams The parameters to set.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExternalSemaphoresWaitNodeSetParams(hNode: hipGraphNode_t; const nodeParams: PHipExternalSemaphoreWaitNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExternalSemaphoresWaitNodeSetParams';
  
{
  @brief Sets the parameters for an external semaphore wait node in an executable graph.
  
  @param(hGraphExec The executable graph containing the node.)
  @param(hNode The node to update.)
  @param(nodeParams The new parameters for the operation.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec: hipGraphExec_t; hNode: hipGraphNode_t; const nodeParams: PHipExternalSemaphoreWaitNodeParams): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphExecExternalSemaphoresWaitNodeSetParams';
    
// --- Stream Capture Functions ---

{
  @brief Begins graph capture on a stream, adding to an existing graph.
  
  @param(stream The stream to initiate capture on.)
  @param(graph The graph to capture into.)
  @param(dependencies An array of nodes that are dependencies for the first captured node.)
  @param(dependencyData Optional array of data associated with each dependency (currently unsupported).)
  @param(numDependencies The number of dependencies.)
  @param(mode Controls the interaction of this capture sequence with other API calls.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(The `dependencyData` parameter is not currently supported and must be nil. This API is marked as beta.)
}
function hipStreamBeginCaptureToGraph(stream: hipStream_t; graph: hipGraph_t; const dependencies: PhipGraphNode_t; const dependencyData: Pointer; numDependencies: SizeUInt; mode: THipStreamCaptureMode): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamBeginCaptureToGraph';

{
  @brief Gets the capture status of a stream.
  
  @param(stream The stream to query.)
  @param(pCaptureStatus Returns the current capture status of the stream.)
  @param(pId Returns the unique ID of the capture sequence.)
  
  @returns(hipSuccess, hipErrorStreamCaptureImplicit)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipStreamGetCaptureInfo(stream: hipStream_t; var pCaptureStatus: THipStreamCaptureStatus; var pId: UInt64): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamGetCaptureInfo';

{
  @brief Gets extended capture information for a stream.
  
  @param(stream The stream to query.)
  @param(captureStatus_out Returns the current capture status.)
  @param(id_out Returns the unique ID of the capture sequence.)
  @param(graph_out Returns the graph being captured into.)
  @param(dependencies_out Returns a pointer to an array of dependency nodes.)
  @param(numDependencies_out Returns the size of the dependency array.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorStreamCaptureImplicit)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipStreamGetCaptureInfo_v2(stream: hipStream_t; var captureStatus_out: THipStreamCaptureStatus; var id_out: UInt64; var graph_out: hipGraph_t; var dependencies_out: PhipGraphNode_t; var numDependencies_out: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamGetCaptureInfo_v2';

{
  @brief Queries if a stream is actively capturing.
  
  @param(stream The stream to query.)
  @param(pCaptureStatus Returns the current capture status of the stream.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorStreamCaptureImplicit)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipStreamIsCapturing(stream: hipStream_t; var pCaptureStatus: THipStreamCaptureStatus): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamIsCapturing';

{
  @brief Updates the set of dependencies in a capturing stream.
  
  @param(stream The stream under capture.)
  @param(dependencies A pointer to an array of nodes to add or replace as dependencies.)
  @param(numDependencies The size of the dependency array.)
  @param(flags A flag indicating how to update the dependency set (add or replace).)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorIllegalState)
  
  @see(THipStreamUpdateCaptureDependenciesFlags)
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipStreamUpdateCaptureDependencies(stream: hipStream_t; dependencies: PhipGraphNode_t; numDependencies: SizeUInt; flags: UInt32 = 0): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'StreamUpdateCaptureDependencies';

{
  @brief Swaps the stream capture mode of the calling thread.
  
  @param(mode On input, a pointer to the new mode; on output, it is overwritten with the previous mode.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipThreadExchangeStreamCaptureMode(var mode: THipStreamCaptureMode): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'ThreadExchangeStreamCaptureMode';
  
// --- Graph Utility and User Object Functions ---

{
  @brief Writes a DOT file describing the graph's structure for visualization.
  
  @param(graph The graph to print.)
  @param(path The path to the output .dot file.)
  @param(flags Flags from hipGraphDebugDotFlags to control the level of detail.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorOperatingSystem)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphDebugDotPrint(graph: hipGraph_t; const path: PChar; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphDebugDotPrint';

{
  @brief Enables or disables a node in an executable graph.
  
  @param(hGraphExec The executable graph in which the node exists.)
  @param(hNode The node from the original graph to enable or disable.)
  @param(isEnabled Set to 1 to enable the node, 0 to disable it.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @note(Currently only kernel, memset, and memcpy nodes are supported.)
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphNodeSetEnabled(hGraphExec: hipGraphExec_t; hNode: hipGraphNode_t; isEnabled: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphNodeSetEnabled';

{
  @brief Queries whether a node in an executable graph is enabled.
  
  @param(hGraphExec The executable graph to query.)
  @param(hNode The node from the original graph to check.)
  @param(isEnabled Returns 1 if the node is enabled, 0 if disabled.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @note(Currently only kernel, memset, and memcpy nodes are supported.)
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphNodeGetEnabled(hGraphExec: hipGraphExec_t; hNode: hipGraphNode_t; var isEnabled: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphNodeGetEnabled';
  
{
  @brief Gets memory attributes related to graphs for a specific device.
  
  @param(device The device to query.)
  @param(attr The memory attribute to get.)
  @param(value A pointer to a location to store the attribute's value.)
  
  @returns(hipSuccess, hipErrorInvalidDevice)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipDeviceGetGraphMemAttribute(device: Int32; attr: THipGraphMemAttributeType; value: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGetGraphMemAttribute';

{
  @brief Sets memory attributes related to graphs for a specific device.
  
  @param(device The device to set the attribute for.)
  @param(attr The memory attribute to set.)
  @param(value A pointer to the attribute's new value.)
  
  @returns(hipSuccess, hipErrorInvalidDevice)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipDeviceSetGraphMemAttribute(device: Int32; attr: THipGraphMemAttributeType; value: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceSetGraphMemAttribute';

{
  @brief Frees unused memory allocated for graphs on a specific device back to the OS.
  
  @param(device The device on which to trim graph memory.)
  
  @returns(hipSuccess, hipErrorInvalidDevice)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipDeviceGraphMemTrim(device: Int32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DeviceGraphMemTrim';
  
{
  @brief Creates a user object to manage the lifetime of a custom resource.
  
  @param(object_out Returns the created user object.)
  @param(ptr A pointer to the custom resource to be managed.)
  @param(destroy The destructor callback function to be called when the object's refcount reaches zero.)
  @param(initialRefcount The initial reference count for the object.)
  @param(flags Flags for the user object (e.g., hipUserObjectNoDestructorSync).)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipUserObjectCreate(var object_out: hipUserObject_t; ptr: Pointer; destroy: THipHostFn; initialRefcount: UInt32; flags: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'UserObjectCreate';

{
  @brief Decrements the reference count of a user object.
  
  @param(object The user object to release.)
  @param(count The number of references to release (defaults to 1).)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipUserObjectRelease(Aobject: hipUserObject_t; count: UInt32 = 1): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'UserObjectRelease';

{
  @brief Increments the reference count of a user object.
  
  @param(object The user object to retain.)
  @param(count The number of references to add (defaults to 1).)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipUserObjectRetain(Aobject: hipUserObject_t; count: UInt32 = 1): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'UserObjectRetain';

{
  @brief Associates a user object with a graph, incrementing its reference count.
  
  @param(graph The graph that will retain the user object.)
  @param(object The user object to be retained by the graph.)
  @param(count The number of references for the graph to hold (defaults to 1).)
  @param(flags Flags for the operation (e.g., hipGraphUserObjectMove).
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphRetainUserObject(graph: hipGraph_t; Aobject: hipUserObject_t; count: UInt32 = 1; flags: UInt32 = 0): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphRetainUserObject';

{
  @brief Releases a graph's reference to a user object.
  
  @param(graph The graph that is releasing the user object.)
  @param(object The user object to be released.)
  @param(count The number of references for the graph to release (defaults to 1).
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @warning(This API is marked as beta and may have outstanding issues.)
}
function hipGraphReleaseUserObject(graph: hipGraph_t; Aobject: hipUserObject_t; count: UInt32 = 1): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphReleaseUserObject';

{ @groupend }


{ @groupbegin(Virtual Memory Management) }
{
  @abstract(This section describes virtual memory management functions.)
  
  @note(These APIs are implemented on Linux and may be under development on Windows.)
  @warning(These APIs are marked as beta and may have outstanding issues.)
}

{
  @brief Frees an address range reservation made via hipMemAddressReserve.
  
  @param(devPtr The starting address of the range to free.)
  @param(size The size of the range in bytes.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemAddressFree(devPtr: Pointer; size: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemAddressFree';

{
  @brief Reserves a virtual address range.
  
  @param(ptr Returns the starting address of the reserved range.)
  @param(size The size of the reservation in bytes.)
  @param(alignment The requested alignment of the address. Must be a power of two.)
  @param(addr A requested starting address for the range. Can be nil.)
  @param(flags Reserved for future use, must be zero.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemAddressReserve(var ptr: Pointer; size: SizeUInt; alignment: SizeUInt; addr: Pointer; flags: UInt64): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemAddressReserve';

{
  @brief Creates a physical memory allocation (handle).
  
  @param(handle Returns the handle for the created memory allocation.)
  @param(size The size of the allocation in bytes.)
  @param(prop The properties of the allocation (e.g., location, type).
  @param(flags Reserved for future use, must be zero.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemCreate(var handle: hipMemGenericAllocationHandle_t; size: SizeUInt; const prop: PHipMemAllocationProp; flags: UInt64): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemCreate';

{
  @brief Exports a physical memory allocation to a shareable handle.
  
  @param(shareableHandle A pointer to the location to store the OS-specific shareable handle.)
  @param(handle The memory allocation handle to export.)
  @param(handleType The type of shareable handle to create (e.g., POSIX file descriptor).
  @param(flags Reserved for future use, must be zero.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemExportToShareableHandle(shareableHandle: Pointer; handle: hipMemGenericAllocationHandle_t; handleType: THipMemAllocationHandleType; flags: UInt64): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemExportToShareableHandle';

{
  @brief Gets the access flags for a given memory location and virtual address.
  
  @param(flags Returns the access flags for the specified location.)
  @param(location The location (e.g., a specific device) to query.)
  @param(ptr The virtual address to check.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemGetAccess(var flags: UInt64; const location: PHipMemLocation; ptr: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemGetAccess';

{
  @brief Calculates the required granularity for a memory allocation.
  
  @param(granularity Returns the allocation granularity in bytes.)
  @param(prop The properties of the allocation.)
  @param(option Determines whether to return the minimal or recommended granularity.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemGetAllocationGranularity(var granularity: SizeUInt; const prop: PHipMemAllocationProp; option: THipMemAllocationGranularityFlags): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemGetAllocationGranularity';

{
  @brief Retrieves the property structure for a given memory allocation handle.
  
  @param(prop Returns the properties of the specified handle.)
  @param(handle The memory allocation handle to query.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemGetAllocationPropertiesFromHandle(var prop: THipMemAllocationProp; handle: hipMemGenericAllocationHandle_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemGetAllocationPropertiesFromHandle';

{
  @brief Imports a physical memory allocation from a shareable handle.
  
  @param(handle Returns the imported memory allocation handle.)
  @param(osHandle The OS-specific shareable handle representing the memory.)
  @param(shHandleType The type of the shareable handle being imported.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemImportFromShareableHandle(var handle: hipMemGenericAllocationHandle_t; osHandle: Pointer; shHandleType: THipMemAllocationHandleType): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemImportFromShareableHandle';

{
  @brief Maps a physical memory allocation to a reserved virtual address range.
  
  @param(ptr The starting virtual address where the memory will be mapped.)
  @param(size The size of the mapping in bytes.)
  @param(offset The offset into the physical memory allocation (must be zero).
  @param(handle The physical memory allocation handle to map.)
  @param(flags Reserved for future use, must be zero.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemMap(ptr: Pointer; size: SizeUInt; offset: SizeUInt; handle: hipMemGenericAllocationHandle_t; flags: UInt64): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemMap';

{
  @brief Asynchronously maps or unmaps subregions of sparse HIP arrays.
  
  @param(mapInfoList A list of mapping operations to perform.)
  @param(count The number of operations in the list.)
  @param(stream The stream on which to perform the operations.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemMapArrayAsync(mapInfoList: PHipArrayMapInfo; count: UInt32; stream: hipStream_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemMapArrayAsync';

{
  @brief Releases a physical memory allocation handle.
  
  @param(handle The memory allocation handle to release.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemRelease(handle: hipMemGenericAllocationHandle_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemRelease';

{
  @brief Returns the allocation handle for a given virtual address.
  
  @param(handle Returns the handle representing the memory at the specified address.)
  @param(addr The virtual address to look up.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemRetainAllocationHandle(var handle: hipMemGenericAllocationHandle_t; addr: Pointer): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemRetainAllocationHandle';

{
  @brief Sets the access permissions for a virtual address range on one or more devices.
  
  @param(ptr The starting virtual address of the range.)
  @param(size The size of the range in bytes.)
  @param(desc An array of access descriptors, one for each location.)
  @param(count The number of descriptors in the array.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemSetAccess(ptr: Pointer; size: SizeUInt; const desc: PHipMemAccessDesc; count: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemSetAccess';

{
  @brief Unmaps a memory range that was previously mapped with hipMemMap.
  
  @param(ptr The starting virtual address of the range to unmap.)
  @param(size The size of the range to unmap in bytes.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorNotSupported)
}
function hipMemUnmap(ptr: Pointer; size: SizeUInt): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'MemUnmap';

{ @groupend }


{ @groupbegin(OpenGL Interoperability) }
{ @abstract(This section describes OpenGL and graphics interoperability functions.) }

{
  @brief Maps graphics resources for access by HIP.
  
  @param(count The number of resources to map.)
  @param(resources A pointer to the resources to map.)
  @param(stream The stream for synchronization.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorInvalidResourceHandle)
}
function hipGraphicsMapResources(count: Int32; resources: PHipGraphicsResource; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphicsMapResources';

{
  @brief Unmaps graphics resources.
  
  @param(count The number of resources to unmap.)
  @param(resources A pointer to the resources to unmap.)
  @param(stream The stream for synchronization.)
  
  @returns(hipSuccess, hipErrorInvalidValue, hipErrorContextIsDestroyed)
}
function hipGraphicsUnmapResources(count: Int32; resources: PHipGraphicsResource; stream: hipStream_t = nil): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphicsUnmapResources';

{
  @brief Gets a HIP array through which to access a subresource of a mapped graphics resource.
  
  @param(array Returns a pointer to the HIP array for accessing the subresource.)
  @param(resource The mapped graphics resource to access.)
  @param(arrayIndex The array index for layered resources.)
  @param(mipLevel The mipmap level for the subresource to access.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
  
  @note(For non-layered resources, arrayIndex must be 0. For non-mipmapped resources, mipLevel must be 0.)
}
function hipGraphicsSubResourceGetMappedArray(var hArr: hipArray_t; resource: hipGraphicsResource_t; arrayIndex: UInt32; mipLevel: UInt32): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphicsSubResourceGetMappedArray';

{
  @brief Gets the device pointer and size for a mapped graphics resource.
  
  @param(devPtr Returns the device pointer for accessing the resource.)
  @param(size Returns the size of the accessible buffer in bytes.)
  @param(resource The mapped graphics resource to access.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipGraphicsResourceGetMappedPointer(var devPtr: Pointer; var size: SizeUInt; resource: hipGraphicsResource_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphicsResourceGetMappedPointer';

{
  @brief Unregisters a graphics resource.
  
  @param(resource The graphics resource to unregister.)
  
  @returns(hipSuccess)
  
  @note(The resource should be unmapped before being unregistered.)
}
function hipGraphicsUnregisterResource(resource: hipGraphicsResource_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'GraphicsUnregisterResource';

{ @groupend }

{ @groupbegin(Surface Management) }
{ @abstract(This section describes surface object functions.)
  @note(These APIs are under development in some versions of HIP.) }

{
  @brief Creates a surface object.
  
  @param(pSurfObject Returns the created surface object.)
  @param(pResDesc The resource descriptor for the surface.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipCreateSurfaceObject(var pSurfObject: hipSurfaceObject_t; const pResDesc: PHipResourceDesc): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'CreateSurfaceObject';

{
  @brief Destroys a surface object.
  
  @param(surfaceObject The surface object to destroy.)
  
  @returns(hipSuccess, hipErrorInvalidValue)
}
function hipDestroySurfaceObject(surfaceObject: hipSurfaceObject_t): THipError;
  cdecl; external HIP_LIB_NAME name PREFIX + 'DestroySurfaceObject';

{ @groupend }


implementation

{ @brief [DEPRECATED] Convenience function for binding texture to mipmapped array. 
  This attempts to auto-detect the channel format from the base mipmap level.
  
  @note This is a Pascal convenience wrapper, not a C API function.
  @param(tex [in] Texture reference to bind)
  @param(mipmappedArray [in] Mipmapped array to bind to)
  @see(hip_deprecated.txt) Use hipTexObjectCreate instead }
function hipBindTextureToMipmappedArray(tex: PTextureReference; 
  mipmappedArray: hipMipmappedArray_const_t): THipError; deprecated;
var
  desc: THipChannelFormatDesc;
  levelArr: hipArray_t;
  err: THipError;
begin
  // Get the base level (level 0) array
  err := hipGetMipmappedArrayLevel(levelArr, mipmappedArray, 0);
  if err <> hipSuccess then
  begin
    Result := err;
    Exit;
  end;
  
  // Query channel descriptor from base level
  err := hipGetChannelDesc(desc, levelArr);
  if err = hipSuccess then
    // Bind using detected descriptor by calling the external C function
    Result := hipBindTextureToArray(tex, levelArr, @desc)
  else
    Result := err;
end;

end.
