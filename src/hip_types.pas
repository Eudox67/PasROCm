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

{*
 * hip_Types.pas
 *
 * Copyright (c) 2023-2026 Eudox67 - Free Pascal wrappers for HIP.
 * Derived from HIP headers, Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc.
 *
 * This unit provides a comprehensive set of core Types, enums, and structs for HIP.
 * It combines definitions from driver_Types.h, channel_descriptor.h, library_Types.h,
 * surface_Types.h, texture_Types.h, hip_version.h, and the Type definitions
 * from hip_runtime_api.h to create a single, foundational Types unit.
 *}

{ Core HIP handle, enum and record types. }
{ Core HIP handle, enum and record types. }
unit hip_types;

{$MODE OBJFPC}{$H+}
{$PACKRECORDS C}

interface

uses
    hip_common, hip_error, hip_vector_types;

// Opaque handle definitions
Type
  // Core Handles
  THipCtx = record end;
  hipCtx_t = ^THipCtx;
  
  { @bold(Group: Stream, Event, and Callback Types) }
  { Opaque handle for a HIP Stream. }
  THipStream = record end;
  hipStream_t = ^THipStream;
  { Opaque handle for a HIP Event. }
  THipEvent = record end;
  hipEvent_t = ^THipEvent;
  
  { HIP stream callback function pointer type. }
  THipStreamCallback = procedure(stream: hipStream_t; status: THipError; userData: Pointer); cdecl;  
  
  THipModule = record end;
  hipModule_t = ^THipModule;
  
  THipFunction = record end;
  hipFunction_t = ^THipFunction;

  // Graph Handles
  THipGraph = record end;
  hipGraph_t = ^THipGraph;
  
  THipGraphNode = record end;
  hipGraphNode_t = ^THipGraphNode;
  PhipGraphNode_t = ^hipGraphNode_t;
  
  THipGraphExec = record end;
  hipGraphExec_t = ^THipGraphExec;

  // Memory and Resource Handles
  THipMemGenericAllocationHandle = record end;
  hipMemGenericAllocationHandle_t = ^THipMemGenericAllocationHandle;
  
  THipMemPool = record end;
  hipMemPool_t = ^THipMemPool;
  
  THipUserObject = record end;
  hipUserObject_t = ^THipUserObject;
  
  THipExternalMemory = record end;
  hipExternalMemory_t = ^THipExternalMemory;
  
  THipExternalSemaphore = record end;
  hipExternalSemaphore_t = ^THipExternalSemaphore;
  PhipExternalSemaphore_t = ^hipExternalSemaphore_t;
  
  THipGraphicsResource = record end;
  hipGraphicsResource_t = ^THipGraphicsResource;
  PHipGraphicsResource = ^hipGraphicsResource_t;

  // because of the general use of size_t in HIP and ROCm
  size_t = sizeUInt;
  PSize_t = ^size_t;
    
  { Pointer-to-pointer types for blas, solver, etc. }
  PPSingle             = ^PSingle;
  PPDouble             = ^PDouble;
  PPPSingle            = ^PPSingle;
  PPPDouble            = ^PPDouble;
  PPPointer            = ^PPointer;

Type
  {
    HIP limit (enum as Int32).
  }
  THipLimit = Int32;

const
  hipLimitStackSize = $00;
  hipLimitPrintfFifoSize = $01;
  hipLimitMallocHeapSize = $02;
  hipLimitRange = 3;


{ @bold(Group: IPC Handle Types) }
const
  { Size of the IPC handle. Essential for record layout. }
  HIP_IPC_HANDLE_SIZE = 64;

Type
  { HIP IPC memory handle. }
  THipIpcMemHandle = record
    reserved: array[0..HIP_IPC_HANDLE_SIZE - 1] of Char;
  end;

  { HIP IPC event handle. }
  THipIpcEventHandle = record
    reserved: array[0..HIP_IPC_HANDLE_SIZE - 1] of Char;
  end;

{ @bold(Group: Core Execution and Device Types) }
Type
  { HIP device (integer alias). }
  hipDevice_t = Int32;

  { Grid and block dimensions struct. }
  dim3 = record
    x: UInt32;
    y: UInt32;
    z: UInt32;
  end;
  
  TDim3 = dim3; // Alias for consistency

  { HIP device architecture features (bitfield struct). }
  THipDeviceArch = record
    case Byte of
      0: ( // Full 32-bit value
        Value: UInt32;
      );
      1: ( // Individual bit fields
        hasGlobalInt32Atomics: 0..1;      // bit 0
        hasGlobalFloatAtomicExch: 0..1;    // bit 1
        hasSharedInt32Atomics: 0..1;       // bit 2
        hasSharedFloatAtomicExch: 0..1;    // bit 3
        hasFloatAtomicAdd: 0..1;           // bit 4
        hasGlobalInt64Atomics: 0..1;       // bit 5
        hasSharedInt64Atomics: 0..1;       // bit 6
        hasDoubles: 0..1;                  // bit 7
        hasWarpVote: 0..1;                 // bit 8
        hasWarpBallot: 0..1;               // bit 9
        hasWarpShuffle: 0..1;              // bit 10
        hasFunnelShift: 0..1;              // bit 11
        hasThreadFenceSystem: 0..1;        // bit 12
        hasSyncThreadsExt: 0..1;           // bit 13
        hasSurfaceFuncs: 0..1;             // bit 14
        has3dGrid: 0..1;                   // bit 15
        hasDynamicParallelism: 0..1;       // bit 16
        { Remaining unused bits can be accessed via Value }
      );
  end;
  { HIP UUID (16-byte array). }
  THipUUID = record
    bytes: array[0..15] of AnsiChar;
  end;

  { HIP device properties. }
  THipDeviceProp = record
    name: array[0..255] of AnsiChar;
    uuid: THipUUID;
    luid: array[0..7] of AnsiChar;
    luidDeviceNodeMask: UInt32;
    totalGlobalMem: SizeUInt;
    sharedMemPerBlock: SizeUInt;
    regsPerBlock: Int32;
    warpSize: Int32;
    memPitch: SizeUInt;
    maxThreadsPerBlock: Int32;
    maxThreadsDim: array[0..2] of Int32;
    maxGridSize: array[0..2] of Int32;
    clockRate: Int32;
    totalConstMem: SizeUInt;
    major: Int32;
    minor: Int32;
    textureAlignment: SizeUInt;
    texturePitchAlignment: SizeUInt;
    deviceOverlap: Int32;
    multiProcessorCount: Int32;
    kernelExecTimeoutEnabled: Int32;
    integrated: Int32;
    canMapHostMemory: Int32;
    computeMode: Int32;
    maxTexture1D: Int32;
    maxTexture1DMipmap: Int32;
    maxTexture1DLinear: Int32;
    maxTexture2D: array[0..1] of Int32;
    maxTexture2DMipmap: array[0..1] of Int32;
    maxTexture2DLinear: array[0..2] of Int32;
    maxTexture2DGather: array[0..1] of Int32;
    maxTexture3D: array[0..2] of Int32;
    maxTexture3DAlt: array[0..2] of Int32;
    maxTextureCubemap: Int32;
    maxTexture1DLayered: array[0..1] of Int32;
    maxTexture2DLayered: array[0..2] of Int32;
    maxTextureCubemapLayered: array[0..1] of Int32;
    maxSurface1D: Int32;
    maxSurface2D: array[0..1] of Int32;
    maxSurface3D: array[0..2] of Int32;
    maxSurface1DLayered: array[0..1] of Int32;
    maxSurface2DLayered: array[0..2] of Int32;
    maxSurfaceCubemap: Int32;
    maxSurfaceCubemapLayered: array[0..1] of Int32;
    surfaceAlignment: SizeUInt;
    concurrentKernels: Int32;
    ECCEnabled: Int32;
    pciBusID: Int32;
    pciDeviceID: Int32;
    pciDomainID: Int32;
    tccDriver: Int32;
    asyncEngineCount: Int32;
    unifiedAddressing: Int32;
    memoryClockRate: Int32;
    memoryBusWidth: Int32;
    l2CacheSize: Int32;
    persistingL2CacheMaxSize: Int32;
    maxThreadsPerMultiProcessor: Int32;
    streamPrioritiesSupported: Int32;
    globalL1CacheSupported: Int32;
    localL1CacheSupported: Int32;
    sharedMemPerMultiprocessor: SizeUInt;
    regsPerMultiprocessor: Int32;
    managedMemory: Int32;
    isMultiGpuBoard: Int32;
    multiGpuBoardGroupID: Int32;
    hostNativeAtomicSupported: Int32;
    singleToDoublePrecisionPerfRatio: Int32;
    pageableMemoryAccess: Int32;
    concurrentManagedAccess: Int32;
    computePreemptionSupported: Int32;
    canUseHostPointerForRegisteredMem: Int32;
    cooperativeLaunch: Int32;
    cooperativeMultiDeviceLaunch: Int32;
    sharedMemPerBlockOptin: SizeUInt;
    pageableMemoryAccessUsesHostPageTables: Int32;
    directManagedMemAccessFromHost: Int32;
    maxBlocksPerMultiProcessor: Int32;
    accessPolicyMaxWindowSize: Int32;
    reservedSharedMemPerBlock: SizeUInt;
    hostRegisterSupported: Int32;
    sparseHipArraySupported: Int32;
    hostRegisterReadOnlySupported: Int32;
    timelineSemaphoreInteropSupported: Int32;
    memoryPoolsSupported: Int32;
    gpuDirectRDMASupported: Int32;
    gpuDirectRDMAFlushWritesOptions: UInt32;
    gpuDirectRDMAWritesOrdering: Int32;
    memoryPoolSupportedHandleTypes: UInt32;
    deferredMappingHipArraySupported: Int32;
    ipcEventSupported: Int32;
    clusterLaunch: Int32;
    unifiedFunctionPointers: Int32;
    reserved: array[0..62] of Int32;
    hipReserved: array[0..31] of Int32;
    gcnArchName: array[0..255] of AnsiChar;
    maxSharedMemoryPerMultiProcessor: SizeUInt;
    clockInstructionRate: Int32;
    arch: THipDeviceArch;
    hdpMemFlushCntl: ^UInt32;
    hdpRegFlushCntl: ^UInt32;
    cooperativeMultiDeviceUnmatchedFunc: Int32;
    cooperativeMultiDeviceUnmatchedGridDim: Int32;
    cooperativeMultiDeviceUnmatchedBlockDim: Int32;
    cooperativeMultiDeviceUnmatchedSharedMem: Int32;
    isLargeBar: Int32;
    asicRevision: Int32;
  end;

  hipDeviceProp_t = THipDeviceProp;
  PHipDeviceProp = ^THipDeviceProp;


{ @bold(Group: Function and Kernel Attributes) }
Type
  { HIP function attributes. }
  THipFuncAttributes = record
    binaryVersion: Int32;
    cacheModeCA: Int32;
    constSizeBytes: SizeUInt;
    localSizeBytes: SizeUInt;
    maxDynamicSharedSizeBytes: Int32;
    maxThreadsPerBlock: Int32;
    numRegs: Int32;
    preferredShmemCarveout: Int32;
    ptxVersion: Int32;
    sharedSizeBytes: SizeUInt;
  end;

  { HIP function attribute enum (from driver_Types.h). }
  THipFunctionAttribute = Type Integer;
  
const
  hipFuncAttributeMaxThreadsPerBlock = 0;
  hipFuncAttributeSharedSizeBytes = 1;
  hipFuncAttributeConstSizeBytes = 2;
  hipFuncAttributeLocalSizeBytes = 3;
  hipFuncAttributeNumRegs = 4;
  hipFuncAttributePtxVersion = 5;
  hipFuncAttributeBinaryVersion = 6;
  hipFuncAttributeCacheModeCA = 7;
  hipFuncAttributeMaxDynamicSharedSizeBytes = 8;
  hipFuncAttributePreferredSharedMemoryCarveout = 9;
  hipFuncAttributeMax = 10;

Type
  { HIP function cache preference. }
  THipFuncCache = Type Integer;
  
const
  hipFuncCachePreferNone = 0;
  hipFuncCachePreferShared = 1;
  hipFuncCachePreferL1 = 2;
  hipFuncCachePreferEqual = 3;

Type
  { HIP shared memory configuration. }
  THipSharedMemConfig = Type Integer;
  
const
  hipSharedMemBankSizeDefault = 0;
  hipSharedMemBankSizeFourByte = 1;
  hipSharedMemBankSizeEightByte = 2;

Type
  { JIT compilation options. }
  THipJitOption = Type Integer;
  PHipJitOption = ^THipJitOption;
const
  hipJitOptionMaxRegisters = 0;
  hipJitOptionThreadsPerBlock = 1;
  hipJitOptionWallTime = 2;
  hipJitOptionInfoLogBuffer = 3;
  hipJitOptionInfoLogBufferSizeBytes = 4;
  hipJitOptionErrorLogBuffer = 5;
  hipJitOptionErrorLogBufferSizeBytes = 6;
  hipJitOptionOptimizationLevel = 7;
  hipJitOptionTargetFromContext = 8;
  hipJitOptionTarget = 9;
  hipJitOptionFallbackStrategy = 10;
  hipJitOptionGenerateDebugInfo = 11;
  hipJitOptionLogVerbose = 12;
  hipJitOptionGenerateLineInfo = 13;
  hipJitOptionCacheMode = 14;
  hipJitOptionNumOptions = 15;

{ @bold(Group: Channel and Format Types) }
Type
  { HIP channel format kind. }
  THipChannelFormatKind = Type Integer;
  
const
  hipChannelFormatKindSigned   = 0;
  hipChannelFormatKindUnsigned = 1;
  hipChannelFormatKindFloat    = 2;
  hipChannelFormatKindNone     = 3;

Type
  { HIP channel format descriptor. }
  THipChannelFormatDesc = record
    x: Int32;
    y: Int32;
    z: Int32;
    w: Int32;
    f: THipChannelFormatKind;
  end;
  
  PHipChannelFormatDesc = ^THipChannelFormatDesc;

{ @bold(Group: Array and Mipmap Types) }
Type
  THipArray = record end;
  hipArray_t = ^THipArray;
  hipArray_const_t = ^THipArray;

  { HIP array format. }
  THipArray_Format = Type Integer;

  THipMipmappedArray_st = record
    data: Pointer;
    desc: THipChannelFormatDesc;
    &Type: UInt32;
    width: UInt32;
    height: UInt32;
    depth: UInt32;
    min_mipmap_level: UInt32;
    max_mipmap_level: UInt32;
    flags: UInt32;
    format: THipArray_Format;
    num_channels: UInt32;
  end;
  
  hipMipmappedArray_t = ^THipMipmappedArray_st;
  hipMipmappedArray_const_t = ^THipMipmappedArray_st;


const
  HIP_AD_FORMAT_UNSIGNED_INT8  = $01;
  HIP_AD_FORMAT_UNSIGNED_INT16 = $02;
  HIP_AD_FORMAT_UNSIGNED_INT32 = $03;
  HIP_AD_FORMAT_SIGNED_INT8    = $08;
  HIP_AD_FORMAT_SIGNED_INT16   = $09;
  HIP_AD_FORMAT_SIGNED_INT32   = $0A;
  HIP_AD_FORMAT_HALF           = $10;
  HIP_AD_FORMAT_FLOAT          = $20;

Type
  { HIP array descriptor. }
  THIP_ARRAY_DESCRIPTOR = record
    Width: SizeUInt;
    Height: SizeUInt;
    Format: THipArray_Format;
    NumChannels: UInt32;
  end;
  PHIP_ARRAY_DESCRIPTOR = ^THIP_ARRAY_DESCRIPTOR;

  { HIP 3D array descriptor. }
  THIP_ARRAY3D_DESCRIPTOR = record
    Width: SizeUInt;
    Height: SizeUInt;
    Depth: SizeUInt;
    Format: THipArray_Format;
    NumChannels: UInt32;
    Flags: UInt32;
  end;
  
  PHIP_ARRAY3D_DESCRIPTOR = ^THIP_ARRAY3D_DESCRIPTOR;

{ @bold(Group: Resource and View Types) }
Type
  { HIP resource Type. }
  THipResourceType = Type Integer;
  
const
  hipResourceTypeArray          = $00;
  hipResourceTypeMipmappedArray = $01;
  hipResourceTypeLinear         = $02;
  hipResourceTypePitch2D        = $03;

Type
  THipPresourceType = Type Integer;
  
const
  HIP_RESOURCE_TYPE_ARRAY           = $00;
  HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = $01;
  HIP_RESOURCE_TYPE_LINEAR          = $02;
  HIP_RESOURCE_TYPE_PITCH2D         = $03;

Type
  { HIP resource view format. }
  THipResourceViewFormat = Type Integer;
  
const
  hipResViewFormatNone = $00;
  hipResViewFormatUnsignedChar1 = $01;
  hipResViewFormatUnsignedChar2 = $02;
  hipResViewFormatUnsignedChar4 = $03;
  hipResViewFormatSignedChar1 = $04;
  hipResViewFormatSignedChar2 = $05;
  hipResViewFormatSignedChar4 = $06;
  hipResViewFormatUnsignedShort1 = $07;
  hipResViewFormatUnsignedShort2 = $08;
  hipResViewFormatUnsignedShort4 = $09;
  hipResViewFormatSignedShort1 = $0A;
  hipResViewFormatSignedShort2 = $0B;
  hipResViewFormatSignedShort4 = $0C;
  hipResViewFormatUnsignedInt1 = $0D;
  hipResViewFormatUnsignedInt2 = $0E;
  hipResViewFormatUnsignedInt4 = $0F;
  hipResViewFormatSignedInt1 = $10;
  hipResViewFormatSignedInt2 = $11;
  hipResViewFormatSignedInt4 = $12;
  hipResViewFormatHalf1 = $13;
  hipResViewFormatHalf2 = $14;
  hipResViewFormatHalf4 = $15;
  hipResViewFormatFloat1 = $16;
  hipResViewFormatFloat2 = $17;
  hipResViewFormatFloat4 = $18;
  hipResViewFormatUnsignedBlockCompressed1 = $19;
  hipResViewFormatUnsignedBlockCompressed2 = $1A;
  hipResViewFormatUnsignedBlockCompressed3 = $1B;
  hipResViewFormatUnsignedBlockCompressed4 = $1C;
  hipResViewFormatSignedBlockCompressed4 = $1D;
  hipResViewFormatUnsignedBlockCompressed5 = $1E;
  hipResViewFormatSignedBlockCompressed5 = $1F;
  hipResViewFormatUnsignedBlockCompressed6H = $20;
  hipResViewFormatSignedBlockCompressed6H = $21;
  hipResViewFormatUnsignedBlockCompressed7 = $22;

type
  { HIP resource descriptor. }
  PHipArray = ^THipArray;     // THipArray defined above

  PHipMipmappedArray = ^THipMipmappedArray;
  THipMipmappedArray = record end;  // Forward declaration

  THipResourceDesc = record
    resType: THipResourceType;
    case Integer of
      0: ( // hipArray
        hipArray: PHipArray;
      );
      1: ( // mipmapped array
        mipmap: PHipMipmappedArray;
      );
      2: ( // linear memory
        linear: record
          devPtr: Pointer;
          desc: THipChannelFormatDesc;
          sizeInBytes: SizeUInt;
        end;
      );
      3: ( // pitch 2D
        pitch2D: record
          devPtr: Pointer;
          desc: THipChannelFormatDesc;
          width: SizeUInt;
          height: SizeUInt;
          pitchInBytes: SizeUInt;
        end;
      );
  end;

  PHipResourceDesc = ^THipResourceDesc;

  { HIP resource view descriptor. }
  THipResourceViewDesc = record
    format: THipResourceViewFormat;
    width: SizeUInt;
    height: SizeUInt;
    depth: SizeUInt;
    firstMipmapLevel: UInt32;
    lastMipmapLevel: UInt32;
    firstLayer: UInt32;
    lastLayer: UInt32;
  end;
  
  PHipResourceViewDesc = ^THipResourceViewDesc;

{ @bold(Group: Memory and Copy Types) }
Type
  hipDeviceptr_t = Pointer;
  PhipDeviceptr_t = ^hipDeviceptr_t;
  hipDeviceptr_array = PhipDeviceptr_t;


  { HIP memory Types. }
  THipMemoryType = Type Integer;
  
const
  hipMemoryTypeHost = 0;
  hipMemoryTypeDevice = 1;
  hipMemoryTypeManaged = 2;
  hipMemoryTypeUnified = 2; // Alias for managed

Type
  { HIP memcpy kind. }
  THipMemcpyKind = Type Integer;
  
const
  hipMemcpyHostToHost = 0;
  hipMemcpyHostToDevice = 1;
  hipMemcpyDeviceToHost = 2;
  hipMemcpyDeviceToDevice = 3;
  hipMemcpyDefault = 4;
  hipMemcpyDeviceToDeviceNoCU = 1024;

Type
  { HIP pointer attributes. }
  THipPointerAttribute = record
    memoryType: THipMemoryType;
    device: Int32;
    devicePointer: Pointer;
    hostPointer: Pointer;
    isManaged: Int32;
    allocationFlags: UInt32;
  end;

type
  THipMemcpy2D = record
    { @name Source Parameters }
    srcXInBytes: SizeUInt;      // Source X offset in bytes
    srcY: SizeUInt;             // Source Y offset
    srcMemoryType: THipMemoryType; // Source memory type (hipMemoryTypeHost, hipMemoryTypeDevice)
    srcHost: Pointer;           // Source host pointer (const void* in C)
    srcDevice: Pointer;         // Source device pointer (hipDeviceptr_t)
    srcArray: hipArray_t;       // Source array handle (if srcMemoryType is array)
    srcPitch: SizeUInt;         // Source pitch in bytes

    { @name Destination Parameters }
    dstXInBytes: SizeUInt;      // Destination X offset in bytes
    dstY: SizeUInt;             // Destination Y offset
    dstMemoryType: THipMemoryType; // Destination memory type
    dstHost: Pointer;           // Destination host pointer (void* in C)
    dstDevice: Pointer;         // Destination device pointer (hipDeviceptr_t)
    dstArray: hipArray_t;       // Destination array handle (if dstMemoryType is array)
    dstPitch: SizeUInt;         // Destination pitch in bytes

    { @name Transfer Region }
    WidthInBytes: SizeUInt;     // Width of 2D region in bytes
    Height: SizeUInt;           // Height of 2D region in elements
  end;

  PHipMemcpy2D = ^THipMemcpy2D;

  { HIP pointer attribute enum. }
  THipPointerAttributeEnum = Type Integer;
  PHipPointerAttributeEnum = ^THipPointerAttributeEnum;
  
const
  { The context on which a pointer was allocated
    @warning Not supported in HIP }
  HIP_POINTER_ATTRIBUTE_CONTEXT = 1;

  { Memory type describing location of a pointer }
  HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = 2;

  { Address at which the pointer is allocated on device }
  HIP_POINTER_ATTRIBUTE_DEVICE_POINTER = 3;

  { Address at which the pointer is allocated on host }
  HIP_POINTER_ATTRIBUTE_HOST_POINTER = 4;

  { A pair of tokens for use with Linux kernel interface
    @warning Not supported in HIP }
  HIP_POINTER_ATTRIBUTE_P2P_TOKENS = 5;

  { Synchronize every synchronous memory operation initiated on this region }
  HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6;

  { Unique ID for an allocated memory region }
  HIP_POINTER_ATTRIBUTE_BUFFER_ID = 7;

  { Indicates if the pointer points to managed memory }
  HIP_POINTER_ATTRIBUTE_IS_MANAGED = 8;

  { Device ordinal of a device on which a pointer was allocated or registered }
  HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9;

  { If this pointer maps to an allocation suitable for hipIpcGetMemHandle
    @warning Not supported in HIP }
  HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = 10;

  { Starting address for this requested pointer }
  HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11;

  { Size of the address range for this requested pointer }
  HIP_POINTER_ATTRIBUTE_RANGE_SIZE = 12;

  { Tells if this pointer is in a valid address range that is mapped to a backing allocation }
  HIP_POINTER_ATTRIBUTE_MAPPED = 13;

  { Bitmask of allowed hipmemAllocationHandleType for this allocation
    @warning Not supported in HIP }
  HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14;

  { Returns if the memory referenced by this pointer can be used with the GPUDirect RDMA API
    @warning Not supported in HIP }
  HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15;

  { Returns the access flags the device associated with for the corresponding memory referenced by the pointer }
  HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16;

  { Returns the mempool handle for the allocation if it was allocated from a mempool
    @warning Not supported in HIP }
  HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17;
{ @bold(Group: Device Attributes and Other Enums) }

type
  {
    HIP device attribute (enum as Int32 with constants).
  }
  THipDeviceAttribute = Int32;

const
  // CUDA Compatible Range (0 - 9999)
  HIP_DEVICE_ATTRIBUTE_CUDA_COMPATIBLE_BEGIN = 0;
  HIP_DEVICE_ATTRIBUTE_ECC_ENABLED = HIP_DEVICE_ATTRIBUTE_CUDA_COMPATIBLE_BEGIN;
  HIP_DEVICE_ATTRIBUTE_ACCESS_POLICY_MAX_WINDOW_SIZE = 1;
  HIP_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 2;
  HIP_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 3;
  HIP_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 4;
  HIP_DEVICE_ATTRIBUTE_CLOCK_RATE = 5;
  HIP_DEVICE_ATTRIBUTE_COMPUTE_MODE = 6;
  HIP_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 7;
  HIP_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 8;
  HIP_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 9;
  HIP_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 10;
  HIP_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 11;
  HIP_DEVICE_ATTRIBUTE_DEVICE_OVERLAP = 12;
  HIP_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 13;
  HIP_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 14;
  HIP_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 15;
  HIP_DEVICE_ATTRIBUTE_INTEGRATED = 16;
  HIP_DEVICE_ATTRIBUTE_IS_MULTI_GPU_BOARD = 17;
  HIP_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 18;
  HIP_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 19;
  HIP_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 20;
  HIP_DEVICE_ATTRIBUTE_LUID = 21;
  HIP_DEVICE_ATTRIBUTE_LUID_DEVICE_NODE_MASK = 22;
  HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 23;
  HIP_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 24;
  HIP_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTI_PROCESSOR = 25;
  HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 26;
  HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 27;
  HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 28;
  HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 29;
  HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 30;
  HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 31;
  HIP_DEVICE_ATTRIBUTE_MAX_SURFACE_1D = 32;
  HIP_DEVICE_ATTRIBUTE_MAX_SURFACE_1D_LAYERED = 33;
  HIP_DEVICE_ATTRIBUTE_MAX_SURFACE_2D = 34;
  HIP_DEVICE_ATTRIBUTE_MAX_SURFACE_2D_LAYERED = 35;
  HIP_DEVICE_ATTRIBUTE_MAX_SURFACE_3D = 36;
  HIP_DEVICE_ATTRIBUTE_MAX_SURFACE_CUBEMAP = 37;
  HIP_DEVICE_ATTRIBUTE_MAX_SURFACE_CUBEMAP_LAYERED = 38;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_1D_WIDTH = 39;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_1D_LAYERED = 40;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_1D_LINEAR = 41;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_1D_MIPMAP = 42;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_2D_WIDTH = 43;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_2D_HEIGHT = 44;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_2D_GATHER = 45;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_2D_LAYERED = 46;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_2D_LINEAR = 47;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_2D_MIPMAP = 48;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_3D_WIDTH = 49;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_3D_HEIGHT = 50;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_3D_DEPTH = 51;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_3D_ALT = 52;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_CUBEMAP = 53;
  HIP_DEVICE_ATTRIBUTE_MAX_TEXTURE_CUBEMAP_LAYERED = 54;
  HIP_DEVICE_ATTRIBUTE_MAX_THREADS_DIM = 55;
  HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 56;
  HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTI_PROCESSOR = 57;
  HIP_DEVICE_ATTRIBUTE_MAX_PITCH = 58;
  HIP_DEVICE_ATTRIBUTE_MEMORY_BUS_WIDTH = 59;
  HIP_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 60;
  HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 61;
  HIP_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 62;
  HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 63;
  HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 64;
  HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 65;
  HIP_DEVICE_ATTRIBUTE_PCI_BUS_ID = 66;
  HIP_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 67;
  HIP_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 68;
  HIP_DEVICE_ATTRIBUTE_PERSISTING_L2_CACHE_MAX_SIZE = 69;
  HIP_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 70;
  HIP_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEM_PER_BLOCK = 71;
  HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 72;
  HIP_DEVICE_ATTRIBUTE_SHARED_MEM_PER_BLOCK_OPTIN = 73;
  HIP_DEVICE_ATTRIBUTE_SHARED_MEM_PER_MULTIPROCESSOR = 74;
  HIP_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 75;
  HIP_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 76;
  HIP_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 77;
  HIP_DEVICE_ATTRIBUTE_TCC_DRIVER = 78;
  HIP_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 79;
  HIP_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 80;
  HIP_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 81;
  HIP_DEVICE_ATTRIBUTE_TOTAL_GLOBAL_MEM = 82;
  HIP_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 83;
  HIP_DEVICE_ATTRIBUTE_WARP_SIZE = 84;
  HIP_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 85;
  HIP_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 86;
  HIP_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 87;
  HIP_DEVICE_ATTRIBUTE_MEMORY_POOL_SUPPORTED_HANDLE_TYPES = 88;
  HIP_DEVICE_ATTRIBUTE_CUDA_COMPATIBLE_END = 9999;

  // AMD-Specific Range (10000 - 19999)
  HIP_DEVICE_ATTRIBUTE_AMD_SPECIFIC_BEGIN = 10000;
  HIP_DEVICE_ATTRIBUTE_CLOCK_INSTRUCTION_RATE = HIP_DEVICE_ATTRIBUTE_AMD_SPECIFIC_BEGIN;
  HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 10001;
  HIP_DEVICE_ATTRIBUTE_HDP_MEM_FLUSH_CNTL = 10002;
  HIP_DEVICE_ATTRIBUTE_HDP_REG_FLUSH_CNTL = 10003;
  HIP_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_UNMATCHED_FUNC = 10004;
  HIP_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_UNMATCHED_GRID_DIM = 10005;
  HIP_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_UNMATCHED_BLOCK_DIM = 10006;
  HIP_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_UNMATCHED_SHARED_MEM = 10007;
  HIP_DEVICE_ATTRIBUTE_IS_LARGE_BAR = 10008;
  HIP_DEVICE_ATTRIBUTE_ASIC_REVISION = 10009;
  HIP_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE = 10010;
  HIP_DEVICE_ATTRIBUTE_IMAGE_SUPPORT = 10011;
  HIP_DEVICE_ATTRIBUTE_PHYSICAL_MULTIPROCESSOR_COUNT = 10012;
  HIP_DEVICE_ATTRIBUTE_FINE_GRAIN_SUPPORT = 10013;
  HIP_DEVICE_ATTRIBUTE_WALL_CLOCK_RATE = 10014;
  HIP_DEVICE_ATTRIBUTE_AMD_SPECIFIC_END = 19999;

  // Vendor-Specific Range (20000+)
  HIP_DEVICE_ATTRIBUTE_VENDOR_SPECIFIC_BEGIN = 20000;Type

  {
    HIP device P2P attribute (enum as Int32).
  }
  THipDeviceP2PAttr = Int32;

const
  HIP_DEV_P2P_ATTR_PERFORMANCE_RANK = 0;
  HIP_DEV_P2P_ATTR_ACCESS_SUPPORTED = 1;
  HIP_DEV_P2P_ATTR_NATIVE_ATOMIC_SUPPORTED = 2;
  HIP_DEV_P2P_ATTR_HIP_ARRAY_ACCESS_SUPPORTED = 3;

Type
  {
    HIP driver proc address query result (enum as Int32).
  }
  THipDriverProcAddressQueryResult = Int32;

const
  HIP_GET_PROC_ADDRESS_SUCCESS = 0;
  HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1;
  HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2;

Type
  {
    HIP compute mode (enum as Int32).
  }
  THipComputeMode = Int32;

const
  hipComputeModeDefault = 0;
  hipComputeModeExclusive = 1;
  hipComputeModeProhibited = 2;
  hipComputeModeExclusiveProcess = 3;

Type
  {
    HIP flush GPU direct RDMA writes options (enum as Int32).
  }
  THipFlushGPUDirectRDMAWritesOptions = Int32;

const
  hipFlushGPUDirectRDMAWritesOptionHost = 1 shl 0;
  hipFlushGPUDirectRDMAWritesOptionMemOps = 1 shl 1;

Type
  {
    HIP GPU direct RDMA writes ordering (enum as Int32).
  }
  THipGPUDirectRDMAWritesOrdering = Int32;

const
  hipGPUDirectRDMAWritesOrderingNone = 0;
  hipGPUDirectRDMAWritesOrderingOwner = 100;
  hipGPUDirectRDMAWritesOrderingAllDevices = 200;


Type
  { HIP pitched pointer. }
  THipPitchedPtr = record
    ptr: Pointer;
    pitch: SizeUInt;
    xsize: SizeUInt;
    ysize: SizeUInt;
  end;

  { HIP extent (dimensions). }
  THipExtent = record
    width: SizeUInt;
    height: SizeUInt;
    depth: SizeUInt;
  end;

  { HIP position. }
  THipPos = record
    x: SizeUInt;
    y: SizeUInt;
    z: SizeUInt;
  end;

  { HIP memcpy 3D parameters. }
  THipMemcpy3DParms = record
    srcArray: hipArray_t;
    srcPos: THipPos;
    srcPtr: THipPitchedPtr;
    dstArray: hipArray_t;
    dstPos: THipPos;
    dstPtr: THipPitchedPtr;
    extent: THipExtent;
    kind: THipMemcpyKind;
  end;
  
  PHipMemcpy3DParms = ^THipMemcpy3DParms;

{ @bold(Group: Library and Data Types from library_Types.h) }
Type
  { HIP data Type. }
  HipDataType = Type Integer;
  
const
  HIP_R_32F = 0;
  HIP_R_64F = 1;
  HIP_R_16F = 2;
  HIP_R_8I = 3;
  HIP_C_32F = 4;
  HIP_C_64F = 5;
  HIP_C_16F = 6;
  HIP_C_8I = 7;
  HIP_R_8U = 8;
  HIP_C_8U = 9;
  HIP_R_32I = 10;
  HIP_C_32I = 11;
  HIP_R_32U = 12;
  HIP_C_32U = 13;
  HIP_R_16BF = 14;
  HIP_C_16BF = 15;
  HIP_R_4I = 16;
  HIP_C_4I = 17;
  HIP_R_4U = 18;
  HIP_C_4U = 19;
  HIP_R_16I = 20;
  HIP_C_16I = 21;
  HIP_R_16U = 22;
  HIP_C_16U = 23;
  HIP_R_64I = 24;
  HIP_C_64I = 25;
  HIP_R_64U = 26;
  HIP_C_64U = 27;
  HIP_R_8F_E4M3 = 28;
  HIP_R_8F_E5M2 = 29;
  HIP_R_8F_E4M3_FNUZ = 1000;
  HIP_R_8F_E5M2_FNUZ = 1001;

Type
  { HIP library property Type. }
  THipLibraryPropertyType = Type Integer;
  
const
  HIP_LIBRARY_MAJOR_VERSION = 0;
  HIP_LIBRARY_MINOR_VERSION = 1;
  HIP_LIBRARY_PATCH_LEVEL   = 2;

{ @bold(Group: Surface and Texture Types) }
Type
  THipSurface = record end;
  hipSurfaceObject_t = ^THipSurface;

  TSurfaceReference = record
    surfaceObject: hipSurfaceObject_t;
  end;

  THipSurfaceBoundaryMode = Type Integer;
  
const
  hipBoundaryModeZero  = 0;
  hipBoundaryModeTrap  = 1;
  hipBoundaryModeClamp = 2;

Type
  THipTexture = record end;
  hipTextureObject_t = ^THipTexture;

  THipTextureAddressMode = Type Integer;
  
const
  hipAddressModeWrap   = 0;
  hipAddressModeClamp  = 1;
  hipAddressModeMirror = 2;
  hipAddressModeBorder = 3;

Type
  THipTextureFilterMode = Type Integer;
  
const
  hipFilterModePoint  = 0;
  hipFilterModeLinear = 1;

Type
  THipTextureReadMode = Type Integer;
  
const
  hipReadModeElementType       = 0;
  hipReadModeNormalizedFloat = 1;

Type
  TTextureReference = record
    normalized: Int32;
    readMode: THipTextureReadMode;
    filterMode: THipTextureFilterMode;
    addressMode: array[0..2] of THipTextureAddressMode;
    channelDesc: THipChannelFormatDesc;
    sRGB: Int32;
    maxAnisotropy: UInt32;
    mipmapFilterMode: THipTextureFilterMode;
    mipmapLevelBias: Single;
    minMipmapLevelClamp: Single;
    maxMipmapLevelClamp: Single;
    textureObject: hipTextureObject_t;
    numChannels: Int32;
    format: THipArray_Format;
  end;
  
  PTextureReference = ^TTextureReference;
  hipTexRef = PTextureReference;

  THipTextureDesc = record
    addressMode: array[0..2] of THipTextureAddressMode;
    filterMode: THipTextureFilterMode;
    readMode: THipTextureReadMode;
    sRGB: Int32;
    borderColor: array[0..3] of Single;
    normalizedCoords: Int32;
    maxAnisotropy: UInt32;
    mipmapFilterMode: THipTextureFilterMode;
    mipmapLevelBias: Single;
    minMipmapLevelClamp: Single;
    maxMipmapLevelClamp: Single;
  end;
  
  PHipTextureDesc = ^THipTextureDesc;

{ @bold(Group: Graph Management Types) }
Type
  { HIP graph node Types. }
  THipGraphNodeType = Type Integer;
  
const
  hipGraphNodeTypeKernel = 0;
  hipGraphNodeTypeMemcpy = 1;
  hipGraphNodeTypeMemset = 2;
  hipGraphNodeTypeHost = 3;
  hipGraphNodeTypeGraph = 4;
  hipGraphNodeTypeEmpty = 5;
  hipGraphNodeTypeWaitEvent = 6;
  hipGraphNodeTypeEventRecord = 7;
  hipGraphNodeTypeExtSemaphoreSignal = 8;
  hipGraphNodeTypeExtSemaphoreWait = 9;
  hipGraphNodeTypeMemAlloc = 10;
  hipGraphNodeTypeMemFree = 11;
  hipGraphNodeTypeCount = 12;

Type
  { Graph memory attribute types }
  THipGraphMemAttributeType = type Integer;

const
  HIP_GRAPH_MEM_ATTR_CONVENTIONAL = 0;
  HIP_GRAPH_MEM_ATTR_HANDLE = 1;
  HIP_GRAPH_MEM_ATTR_NUM_BUFFERS = 2;
  HIP_GRAPH_MEM_ATTR_CONVENTIONAL_RESERVED = 3;
  HIP_GRAPH_MEM_ATTR_HANDLE_RESERVED = 4;
  HIP_GRAPH_MEM_ATTR_NUM_BUFFERS_RESERVED = 5;
  HIP_GRAPH_MEM_ATTR_HANDLE_TYPES = 6;
  HIP_GRAPH_MEM_ATTR_BUFFER_PROPS = 7;

Type
  { Parameters for a kernel node in a HIP graph. }
  THipKernelNodeParams = record
    func: Pointer;
    gridDim: TDim3;
    blockDim: TDim3;
    sharedMemBytes: UInt32;
    kernelParams: PPointer; // Pointer to array of argument pointers
    extra: PPointer;        // Pointer to array of extra option pointers
  end;
  
  PHipKernelNodeParams = ^THipKernelNodeParams;

  { Parameters for a memset node in a HIP graph. }
  THipMemsetParams = record
    dst: Pointer;
    value: UInt32;
    pitch: SizeUInt;
    elementSize: SizeUInt; // 1, 2, or 4 bytes
    width: SizeUInt;
    height: SizeUInt;
  end;
  
  PHipMemsetParams = ^THipMemsetParams;

  { Host callback function pointer Type. }
  THipHostFn = procedure(userData: Pointer); cdecl;

  { Parameters for a host node in a HIP graph. }
  THipHostNodeParams = record
    fn: THipHostFn;
    userData: Pointer;
  end;
  
  PHipHostNodeParams = ^THipHostNodeParams;

  { Parameters for an external semaphore signal node. }
  THipExternalSemaphoreSignalNodeParams = record
    extSemArray: PhipExternalSemaphore_t;
    paramsArray: Pointer; // PHipExternalSemaphoreSignalParams
    numExtSems: UInt32;
  end;
  
  PHipExternalSemaphoreSignalNodeParams = ^THipExternalSemaphoreSignalNodeParams;

  { Parameters for an external semaphore wait node. }
  THipExternalSemaphoreWaitNodeParams = record
    extSemArray: PhipExternalSemaphore_t;
    paramsArray: Pointer; // PHipExternalSemaphoreWaitParams
    numExtSems: UInt32;
  end;
  
  PHipExternalSemaphoreWaitNodeParams = ^THipExternalSemaphoreWaitNodeParams;

type
  { Simple integer-based type aliases for C enums }
  THipMemLocationType = type Integer;
  THipMemAccessFlags = type Integer;
  THipMemAllocationType = type Integer;
  THipMemAllocationHandleType = type Integer;
  THipGraphExecUpdateResult = type Integer;
  { Attribute IDs for a kernel node. }
  THipKernelNodeAttrID = Type Integer;
//  THipGraphNodeType = type Integer;

  { Pointer types (if needed) }
  PHipMemLocationType = ^THipMemLocationType;
  PHipMemAccessFlags = ^THipMemAccessFlags;
  PHipMemAllocationType = ^THipMemAllocationType;
  PHipMemAllocationHandleType = ^THipMemAllocationHandleType;
  PHipGraphExecUpdateResult = ^THipGraphExecUpdateResult;
  PHipGraphNodeType = ^THipGraphNodeType;
  PHipKernelNodeAttrID = ^THipKernelNodeAttrID;

const
  { HIP memory location type constants }
  hipMemLocationTypeInvalid = 0;
  hipMemLocationTypeDevice = 1;

  { HIP memory access flags constants }
  hipMemAccessFlagsProtNone = 0;
  hipMemAccessFlagsProtRead = 1;
  hipMemAccessFlagsProtReadWrite = 3;

  { HIP memory allocation type constants }
  hipMemAllocationTypeInvalid = $0;
  hipMemAllocationTypePinned = $1;
  hipMemAllocationTypeMax = $7FFFFFFF;

  { HIP memory allocation handle type constants }
  hipMemHandleTypeNone = $0;
  hipMemHandleTypePosixFileDescriptor = $1;
  hipMemHandleTypeWin32 = $2;
  hipMemHandleTypeWin32Kmt = $4;

type
  THipMemLocation = record
    &type: THipMemLocationType;
    id: Int32;
  end;

  THipMemPoolProps = record
    allocType: THipMemAllocationType;
    handleType: THipMemAllocationHandleType;
    location: THipMemLocation;
    win32SecurityAttributes: Pointer;
    reserved: array[0..63] of Byte;
  end;

  { Parameters for a memory allocation node in a HIP graph. }
  THipMemAllocNodeParams = record
    poolProps: THipMemPoolProps;
    dptr: ^Pointer;
    size: SizeUInt;
    bytesize: SizeUInt;
  end;
  PHipMemAllocNodeParams = ^THipMemAllocNodeParams;

  { Graph instantiation parameters. }
  THipGraphInstantiateParams = record
    flags: UInt64;
    stream: hipStream_t;
    errNode_out: hipGraphNode_t;
    result_out: THipGraphExecUpdateResult;
  end;
  
  PHipGraphInstantiateParams = ^THipGraphInstantiateParams;

const

  hipKernelNodeAttributeAccessPolicyWindow = 1;
  hipKernelNodeAttributeCooperative = 2;

type
  THipAccessProperty = type integer;

const
  hipAccessPropertyNormal     = 0;
  hipAccessPropertyStreaming  = 1;
  hipAccessPropertyPersisting = 2;

type
  { Define the nested record type first }
  THipAccessPolicyWindow = record
    base_ptr: Pointer;
    num_bytes: SizeUInt;
    hitRatio: Single;
    hitProp: THipAccessProperty;
    missProp: THipAccessProperty;
  end;

  { Value for a kernel node attribute. Union handled by variant record. }
  THipKernelNodeAttrValue = record
    case Integer of
      0: (accessPolicyWindow: THipAccessPolicyWindow);
      1: (cooperative: Int32);
  end;

  PHipKernelNodeAttrValue = ^THipKernelNodeAttrValue;

{ @bold(Group: Module, Launch, and Sparse Memory Types) }

type
  { Launch parameters for a cooperative kernel. }
  THipLaunchParams = record
    func: Pointer;
    gridDim: TDim3;
    blockDim: TDim3;
    sharedMem: SizeUInt;
    stream: hipStream_t;
    args: PPointer;
  end;
  PHipLaunchParams = ^THipLaunchParams;

  { (Identical to THipLaunchParams) Launch parameters for a cooperative kernel from a module. }
  THipFunctionLaunchParams = THipLaunchParams;
  PHipFunctionLaunchParams = ^THipFunctionLaunchParams;

  {
    Specifies the type of sparse subresource for array mapping.
    @seeAlso(THipArrayMapInfo)
  }
  THipArraySparseSubresourceType = type Integer;

const
  { The subresource is a sparse mipmap level. }
  hipArraySparseSubresourceTypeSparseLevel = 0;
  { The subresource is the mipmap tail. }
  hipArraySparseSubresourceTypeMiptail     = 1;

type
  {
    Defines a sparse memory mapping operation for a HIP array or mipmapped array.
    This record is used with the hipMemMapArrayAsync function.
  }
  { Define the nested record types first }
  THipArraySparseLevel = record
    level: UInt32;
    layer: UInt32;
    offsetX: UInt32;
    offsetY: UInt32;
    offsetZ: UInt32;
    extentWidth: UInt32;
    extentHeight: UInt32;
    extentDepth: UInt32;
  end;

  THipArrayMiptail = record
    layer: UInt32;
    offset: UInt64;
    size: UInt64;
  end;

  THipArrayMapInfo = record
    { The type of the resource (e.g., hipResourceTypeArray). }
    resourceType: THipResourceType;

    { A pointer to the resource, which can be either a hipArray_t or hipMipmappedArray_t. }
    resource: Pointer;

    { The type of the subresource to be mapped, which determines which part of the union is valid. }
    subresourceType: THipArraySparseSubresourceType;

    { A variant record that holds the specific subresource information. }

    { The type of memory usage for this mapping (e.g., hipMemoryTypeDevice). }
    memUsage: THipMemoryType;

    { Flags for the mapping operation (must be zero). }
    flags: UInt32;

    { The device pointer to the memory to be mapped. }
    devicePtr: hipDeviceptr_t;

    { The memory pool from which the memory was allocated. }
    memPool: hipMemPool_t;

    { The offset within the physical memory allocation. }
    offset: UInt64;

    case Integer of
      hipArraySparseSubresourceTypeSparseLevel: (
        { Describes a specific sparse mipmap level to map. }
        sparseLevel: THipArraySparseLevel
      );
      hipArraySparseSubresourceTypeMiptail: (
        { Describes the mipmap tail to map. }
        miptail: THipArrayMiptail
      );

  end;

  PHipArrayMapInfo = ^THipArrayMapInfo;

{ @bold(Group: Stream Capture Types) }
Type
  { Stream capture status. }
  THipStreamCaptureStatus = Type Integer;
  
const
  hipStreamCaptureStatusNone = 0;
  hipStreamCaptureStatusActive = 1;
  hipStreamCaptureStatusInvalidated = 2;

type
  { Stream capture mode. }
  THipStreamCaptureMode = Type Integer;
  
const
  hipStreamCaptureModeGlobal = 0;
  hipStreamCaptureModeThreadLocal = 1;
  hipStreamCaptureModeRelaxed = 2;

type
  { Flags for updating stream capture dependencies. }
  THipStreamUpdateCaptureDependenciesFlags = Type Integer;
  
const
  hipStreamAddCaptureDependencies = 0;
  hipStreamSetCaptureDependencies = 1;
  

{ @bold(Group: Graph Execution and Update Types) }
//Type
  { Result of a graph execution update. }
//  THipGraphExecUpdateResult = Type Integer;
  
const
  hipGraphExecUpdateSuccess = 0;
  hipGraphExecUpdateError = 1;
  hipGraphExecUpdateErrorTopologyChanged = 2;
  hipGraphExecUpdateErrorNodeTypeChanged = 3;
  hipGraphExecUpdateErrorFunctionChanged = 4;
  hipGraphExecUpdateErrorParametersChanged = 5;
  hipGraphExecUpdateErrorNotSupported = 6;
  hipGraphExecUpdateErrorUnsupportedFunctionChange = 7;
  hipGraphExecUpdateErrorAttributesChanged = 8;

{ @bold(Group: External Resource Interop Types) }
Type
  { External memory handle Types. }
  THipExternalMemoryHandleType = Type Integer;
  
const
  hipExternalMemoryHandleTypeOpaqueFd = 1;
  hipExternalMemoryHandleTypeOpaqueWin32 = 2;
  hipExternalMemoryHandleTypeOpaqueWin32Kmt = 3;
  hipExternalMemoryHandleTypeD3D12Heap = 4;
  hipExternalMemoryHandleTypeD3D12Resource = 5;
  hipExternalMemoryHandleTypeD3D11Resource = 6;
  hipExternalMemoryHandleTypeD3D11ResourceKmt = 7;

type
  TWin32Handle = record
    handle: Pointer;
    name:   Pointer;
  end;

  THipExternalMemoryHandle = record
  case Integer of
    0: (fd: Int32);
    1: (win32: TWin32Handle);
  end;

  { External memory handle descriptor. }
  THipExternalMemoryHandleDesc = record
    &Type: THipExternalMemoryHandleType;
    handle:THipExternalMemoryHandle;
    size: UInt64;
    flags: UInt32;
  end;
  PHipExternalMemoryHandleDesc = ^THipExternalMemoryHandleDesc;

  { External memory buffer descriptor. }
  THipExternalMemoryBufferDesc = record
    offset: UInt64;
    size: UInt64;
    flags: UInt32;
  end;
  
  PHipExternalMemoryBufferDesc = ^THipExternalMemoryBufferDesc;

  { External memory mipmapped array descriptor. }
  THipExternalMemoryMipmappedArrayDesc = record
    offset: UInt64;
    formatDesc: THipChannelFormatDesc;
    extent: THipExtent;
    flags: UInt32;
    numLevels: UInt32;
  end;
  PHipExternalMemoryMipmappedArrayDesc = ^THipExternalMemoryMipmappedArrayDesc;

  { External semaphore handle Types. }
  THipExternalSemaphoreHandleType = Type Integer;
  
const
  hipExternalSemaphoreHandleTypeOpaqueFd = 1;
  hipExternalSemaphoreHandleTypeOpaqueWin32 = 2;
  hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3;
  hipExternalSemaphoreHandleTypeD3D12Fence = 4;
  hipExternalSemaphoreHandleTypeD3D11Fence = 5;
  hipExternalSemaphoreHandleTypeNvSciSync = 6;
  hipExternalSemaphoreHandleTypeKeyedMutex = 7;
  hipExternalSemaphoreHandleTypeKeyedMutexKmt = 8;
  hipExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9;
  hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10;

type
  THipExternalSemaphoreHandle = record
  case Integer of
    0: (fd: Int32);
    1: (win32: TWin32Handle);
  end;

  { External semaphore handle descriptor. }
  THipExternalSemaphoreHandleDesc = record
    &Type: THipExternalSemaphoreHandleType;
    handle: THipExternalSemaphoreHandle;
    flags: UInt32;
  end;
  
  PHipExternalSemaphoreHandleDesc = ^THipExternalSemaphoreHandleDesc;

  { External semaphore signal parameters. }
  THipExternalSemaphoreSignalParams = record
    params: record
      fence: record value: UInt64; end;
      nvSciSync: record fence: Pointer; end;
      keyedMutex: record key: UInt64; end;
    end;
    flags: UInt32;
    reserved: array[0..15] of UInt32;
  end;
  
  PHipExternalSemaphoreSignalParams = ^THipExternalSemaphoreSignalParams;

  { External semaphore wait parameters. }
  THipExternalSemaphoreWaitParams = record
    params: record
      fence: record value: UInt64; end;
      nvSciSync: record fence: Pointer; end;
      keyedMutex: record key: UInt64; timeoutMs: UInt32; end;
    end;
    flags: UInt32;
    reserved: array[0..11] of UInt32;
  end;
  
  PHipExternalSemaphoreWaitParams = ^THipExternalSemaphoreWaitParams;

{ @bold(Group: Virtual Memory Management Types) }
//Type
//  { Memory allocation handle Types. }
//  THipMemAllocationHandleType = Type Integer;
//  
//const
//  hipMemHandleTypeNone = 0;
//  hipMemHandleTypePosixFileDescriptor = 1;
//  hipMemHandleTypeWin32 = 2;
//  hipMemHandleTypeWin32Kmt = 3;

//type
//  { Memory allocation Types. }
//  THipMemAllocationType = Type Integer;
//
//const
//  hipMemAllocationTypeInvalid = 0;
//  hipMemAllocationTypePinned = 1;
//
//  { Memory location Types. }
//  THipMemLocationType = Type Integer;
  
//const
//  hipMemLocationTypeInvalid = 0;
//  hipMemLocationTypeDevice = 1;
//
//  { Memory location descriptor. }
//  THipMemLocation = record
//    &Type: THipMemLocationType;
//    id: Int32;
//  end;
  
  PHipMemLocation = ^THipMemLocation;

  { Memory allocation properties. }
  THipMemAllocationProp = record
    &type: THipMemAllocationType;
    requestedHandleTypes: THipMemAllocationHandleType;
    location: THipMemLocation;
    win32HandleMetaData: Pointer;
    allocFlags: record
      compressionType: Byte;
      gpuDirectRDMACapable: Byte;
      reserved: array[0..5] of UInt16;
    end;
  end;
  
  PHipMemAllocationProp = ^THipMemAllocationProp;

  { Memory allocation granularity flags. }
  THipMemAllocationGranularityFlags = Type Integer;
  
const
  hipMemAllocationGranularityMinimum = 0;
  hipMemAllocationGranularityRecommended = 1;

type
  { Memory access descriptor. }
  THipMemAccessDesc = record
    location: THipMemLocation;
    flags: THipMemAccessFlags;
  end;
  PHipMemAccessDesc = ^THipMemAccessDesc;

implementation

end.
