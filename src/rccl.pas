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

{ Bindings for RCCL collective communication primitives. }
{ Bindings for RCCL collective communication primitives. }
unit rccl;

{$mode objfpc}{$H+}
{$linklib rccl}

interface

uses
  hip_runtime, hip_types;

(*----------------------------------------------------
   Types and opaque handles
----------------------------------------------------*)
type
  ncclComm_t            = Pointer;
  ncclUniqueId          = record
    internal: array[0..127] of Char;   // 128-byte opaque
  end;
  
  ncclConfig_t          = record
    size                 : UIntPtr;
    magic                : UInt32;
    version              : UInt32;
    blocking             : Int32;
    cgaClusterSize       : Int32;
    minCTAs              : Int32;
    maxCTAs              : Int32;
    netName              : PChar;
    splitShare           : Int32;
  end;
  
  ncclRedOp_t           = Int32;
  ncclDataType_t        = Int32;
  ncclResult_t          = Int32;
  ncclScalarResidence_t = Int32;
  mscclAlgoHandle_t     = Int32;

(*----------------------------------------------------
   Result codes
----------------------------------------------------*)
const
  ncclSuccess                 = 0;
  ncclUnhandledCudaError      = 1;
  ncclSystemError             = 2;
  ncclInternalError           = 3;
  ncclInvalidArgument         = 4;
  ncclInvalidUsage            = 5;
  ncclRemoteError             = 6;
  ncclInProgress              = 7;

(*----------------------------------------------------
   Reduction ops
----------------------------------------------------*)
const
  ncclSum        = 0;
  ncclProd       = 1;
  ncclMax        = 2;
  ncclMin        = 3;
  ncclAvg        = 4;
  ncclNumOps     = 5;

(*----------------------------------------------------
   Data types
----------------------------------------------------*)
const
  ncclInt8       = 0;
  ncclUint8      = 1;
  ncclInt32      = 2;
  ncclUint32     = 3;
  ncclInt64      = 4;
  ncclUint64     = 5;
  ncclFloat16    = 6;
  ncclFloat32    = 7;
  ncclFloat64    = 8;
  ncclBfloat16   = 9;
  ncclFp8E4M3    = 10;
  ncclFp8E5M2    = 11;
  ncclNumTypes   = 12;

(*----------------------------------------------------
   Scalar residence
----------------------------------------------------*)
const
  ncclScalarDevice        = 0;
  ncclScalarHostImmediate = 1;

(*----------------------------------------------------
   Config initializer
----------------------------------------------------*)
const
  NCCL_CONFIG_INITIALIZER : ncclConfig_t = (
    size        : SizeOf(ncclConfig_t);
    magic       : $cafebeef;
    version     : 22105;
    blocking    : -2147483648;
    cgaClusterSize: -2147483648;
    minCTAs     : -2147483648;
    maxCTAs     : -2147483648;
    netName     : nil;
    splitShare  : -2147483648
  );

(*----------------------------------------------------
   Memory & version
----------------------------------------------------*)
function ncclMemAlloc(var ptr: Pointer; size: UIntPtr): ncclResult_t; cdecl; external;
function ncclMemFree(ptr: Pointer): ncclResult_t; cdecl; external;
function ncclGetVersion(var version: Integer): ncclResult_t; cdecl; external;
function ncclGetUniqueId(var id: ncclUniqueId): ncclResult_t; cdecl; external;

(*----------------------------------------------------
   Communicator life-cycle
----------------------------------------------------*)
function ncclCommInitRankConfig(var comm: ncclComm_t; nranks: Int32; const commId: ncclUniqueId; rank: Int32; const config: ncclConfig_t): ncclResult_t; cdecl; external;
function ncclCommInitRank(var comm: ncclComm_t; nranks: Int32; const commId: ncclUniqueId; rank: Int32): ncclResult_t; cdecl; external;
function ncclCommInitAll(var comm: ncclComm_t; ndev: Int32; const devlist: PInt32): ncclResult_t; cdecl; external;
function ncclCommFinalize(comm: ncclComm_t): ncclResult_t; cdecl; external;
function ncclCommDestroy(comm: ncclComm_t): ncclResult_t; cdecl; external;
function ncclCommAbort(comm: ncclComm_t): ncclResult_t; cdecl; external;
function ncclCommSplit(comm: ncclComm_t; color, key: Int32; var newcomm: ncclComm_t; const config: ncclConfig_t): ncclResult_t; cdecl; external;

(*----------------------------------------------------
   Error helpers
----------------------------------------------------*)
function ncclGetErrorString(result: ncclResult_t): PChar; cdecl; external;
function ncclGetLastError(comm: ncclComm_t): PChar; cdecl; external;
function ncclCommGetAsyncError(comm: ncclComm_t; var asyncError: ncclResult_t): ncclResult_t; cdecl; external;

(*----------------------------------------------------
   Communicator queries
----------------------------------------------------*)
function ncclCommCount(const comm: ncclComm_t; var count: Int32): ncclResult_t; cdecl; external;
function ncclCommCuDevice(const comm: ncclComm_t; var device: Int32): ncclResult_t; cdecl; external;
function ncclCommUserRank(const comm: ncclComm_t; var rank: Int32): ncclResult_t; cdecl; external;

(*----------------------------------------------------
   Memory registration
----------------------------------------------------*)
function ncclCommRegister(const comm: ncclComm_t; buff: Pointer; size: UIntPtr; var handle: Pointer): ncclResult_t; cdecl; external;
function ncclCommDeregister(const comm: ncclComm_t; handle: Pointer): ncclResult_t; cdecl; external;

(*----------------------------------------------------
   Reduction-op creation / destruction
----------------------------------------------------*)
function ncclRedOpCreatePreMulSum(var op: ncclRedOp_t; scalar: Pointer; datatype: ncclDataType_t; residence: ncclScalarResidence_t; comm: ncclComm_t): ncclResult_t; cdecl; external;
function ncclRedOpDestroy(op: ncclRedOp_t; comm: ncclComm_t): ncclResult_t; cdecl; external;

(*----------------------------------------------------
   Collective operations
----------------------------------------------------*)
function ncclReduce(const sendbuff: hipDeviceptr_t; recvbuff: hipDeviceptr_t; count: UIntPtr; datatype: ncclDataType_t; op: ncclRedOp_t; root: Int32; comm: ncclComm_t; stream: Pointer): ncclResult_t; cdecl; external;
function ncclBcast(buff: hipDeviceptr_t; count: UIntPtr; datatype: ncclDataType_t; root: Int32; comm: ncclComm_t; stream: Pointer): ncclResult_t; cdecl; external;
function ncclBroadcast(const sendbuff: hipDeviceptr_t; recvbuff: hipDeviceptr_t; count: UIntPtr; datatype: ncclDataType_t; root: Int32; comm: ncclComm_t; stream: Pointer): ncclResult_t; cdecl; external;
function ncclAllReduce(const sendbuff: hipDeviceptr_t; recvbuff: hipDeviceptr_t; count: UIntPtr; datatype: ncclDataType_t; op: ncclRedOp_t; comm: ncclComm_t; stream: Pointer): ncclResult_t; cdecl; external;
function ncclReduceScatter(const sendbuff: hipDeviceptr_t; recvbuff: hipDeviceptr_t; recvcount: UIntPtr; datatype: ncclDataType_t; op: ncclRedOp_t; comm: ncclComm_t; stream: Pointer): ncclResult_t; cdecl; external;
function ncclAllGather(const sendbuff: hipDeviceptr_t; recvbuff: hipDeviceptr_t; sendcount: UIntPtr; datatype: ncclDataType_t; comm: ncclComm_t; stream: Pointer): ncclResult_t; cdecl; external;
function ncclSend(const sendbuff: hipDeviceptr_t; count: UIntPtr; datatype: ncclDataType_t; peer: Int32; comm: ncclComm_t; stream: Pointer): ncclResult_t; cdecl; external;
function ncclRecv(recvbuff: hipDeviceptr_t; count: UIntPtr; datatype: ncclDataType_t; peer: Int32; comm: ncclComm_t; stream: Pointer): ncclResult_t; cdecl; external;
function ncclGather(const sendbuff: hipDeviceptr_t; recvbuff: hipDeviceptr_t; sendcount: UIntPtr; datatype: ncclDataType_t; root: Int32; comm: ncclComm_t; stream: Pointer): ncclResult_t; cdecl; external;
function ncclScatter(const sendbuff: hipDeviceptr_t; recvbuff: hipDeviceptr_t; recvcount: UIntPtr; datatype: ncclDataType_t; root: Int32; comm: ncclComm_t; stream: Pointer): ncclResult_t; cdecl; external;
function ncclAllToAll(const sendbuff: hipDeviceptr_t; recvbuff: hipDeviceptr_t; count: UIntPtr; datatype: ncclDataType_t; comm: ncclComm_t; stream: Pointer): ncclResult_t; cdecl; external;
function ncclAllToAllv(const sendbuff: hipDeviceptr_t; const sendcounts, sdispls: PUIntPtr; recvbuff: hipDeviceptr_t; const recvcounts, rdispls: PUIntPtr; datatype: ncclDataType_t; comm: ncclComm_t; stream: Pointer): ncclResult_t; cdecl; external;

(*----------------------------------------------------
   MSCCL extensions
----------------------------------------------------*)
function mscclLoadAlgo(const filename: PChar; var handle: mscclAlgoHandle_t; rank: Int32): ncclResult_t; cdecl; external;
function mscclRunAlgo(const sendBuff: hipDeviceptr_t; const sendcounts, sdispls: PUIntPtr; recvBuff: hipDeviceptr_t; const recvcounts, rdispls: PUIntPtr; count: UIntPtr; datatype: ncclDataType_t; root, peer: Int32; op: ncclRedOp_t; algo: mscclAlgoHandle_t; comm: ncclComm_t; stream: Pointer): ncclResult_t; cdecl; external;
function mscclUnloadAlgo(algo: mscclAlgoHandle_t): ncclResult_t; cdecl; external;

(*----------------------------------------------------
   Group semantics
----------------------------------------------------*)
function ncclGroupStart: ncclResult_t; cdecl; external;
function ncclGroupEnd: ncclResult_t; cdecl; external;

implementation
end.
