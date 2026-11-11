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

unit miopen;

{$mode objfpc}{$H+}
{$linklib MIOpen}
{$DEFINE MIOPEN_BETA_API}  // uncomment to use the BETA API's
//{$DEFINE ROCm_6.x}
//{$DEFINE ROCm_7.x}

{$IF DEFINED(ROCm_6)}
  {$DEFINE BACKEND}
{$ELSEIF DEFINED(ROCm_7)}
  {$DEFINE BACKEND}
{$ENDIF}

interface

uses
   hip_types, hip_error, hip_runtime, ctypes;

const
  {$IFDEF MSWINDOWS}
  LIBMIOPEN = 'MIOpen.dll';
  {$ENDIF}
  {$IFDEF LINUX}
  LIBMIOPEN = 'MIOpen';
  {$ENDIF}
  {$IFDEF DARWIN} // For macOS ??
  LIBMIOPEN = 'libMIOpen.dylib';
  {$ENDIF}



{$I miopen_defs.inc}

{ === API FUNCTIONS === }

{ Handle }
function miopenGetErrorString(const error: miopenStatus_t): PChar; cdecl; external;
function miopenGetVersion(var major, minor, patch: size_t): miopenStatus_t; cdecl; external;
function miopenCreate(var handle: miopenHandle_t): miopenStatus_t; cdecl; external;
function miopenCreateWithStream(var handle: miopenHandle_t; const stream: miopenAcceleratorQueue_t): miopenStatus_t; cdecl; external;
function miopenDestroy(const handle: miopenHandle_t): miopenStatus_t; cdecl; external;
function miopenSetStream(const handle: miopenHandle_t; const streamID: miopenAcceleratorQueue_t): miopenStatus_t; cdecl; external;
function miopenGetStream(const handle: miopenHandle_t; var streamID: miopenAcceleratorQueue_t): miopenStatus_t; cdecl; external;
function miopenSetAllocator(const handle: miopenHandle_t; const allocator: miopenAllocatorFunction; const deallocator: miopenDeallocatorFunction; const allocatorContext: Pointer): miopenStatus_t; cdecl; external;
function miopenGetKernelTime(const handle: miopenHandle_t; var time: Single): miopenStatus_t; cdecl; external;
function miopenEnableProfiling(const handle: miopenHandle_t; const enable: Boolean): miopenStatus_t; cdecl; external;

{ Tensor Descriptor }
function miopenCreateTensorDescriptor(var tensorDesc: miopenTensorDescriptor_t): miopenStatus_t; cdecl; external;
function miopenSet4dTensorDescriptor(const tensorDesc: miopenTensorDescriptor_t; const dataType: miopenDataType_t; n, c, h, w: Integer): miopenStatus_t; cdecl; external;
function miopenSet4dTensorDescriptorEx(const tensorDesc: miopenTensorDescriptor_t; const dataType: miopenDataType_t; n, c, h, w: Integer; nStride, cStride, hStride, wStride: Integer): miopenStatus_t; cdecl; external;
function miopenGet4dTensorDescriptor(const tensorDesc: miopenTensorDescriptor_t; var dataType: miopenDataType_t; var n, c, h, w: Integer; var nStride, cStride, hStride, wStride: Integer): miopenStatus_t; cdecl; external;
function miopenSetTensorDescriptor(const tensorDesc: miopenTensorDescriptor_t; const dataType: miopenDataType_t; nbDims: Integer; const dimsA, stridesA: PInteger): miopenStatus_t; cdecl; external;
function miopenGetTensorDescriptor(const tensorDesc: miopenTensorDescriptor_t; var dataType: miopenDataType_t; dimsA, stridesA: PInteger): miopenStatus_t; cdecl; external;
function miopenGetTensorDescriptorSize(const tensorDesc: miopenTensorDescriptor_t; var size: Integer): miopenStatus_t; cdecl; external;
function miopenDestroyTensorDescriptor(const tensorDesc: miopenTensorDescriptor_t): miopenStatus_t; cdecl; external;
function miopenGetTensorNumBytes(const tensorDesc: miopenTensorDescriptor_t; var numBytes: size_t): miopenStatus_t; cdecl; external;
function miopenSetNdTensorDescriptorWithLayout(const tensorDesc: miopenTensorDescriptor_t; const dataType: miopenDataType_t; const tensorLayout: miopenTensorLayout_t; const lens: PInteger; num_lens: Integer): miopenStatus_t; cdecl; external;

{$IFDEF MIOPEN_BETA_API}
// Note: Need to define Psize_t = ^size_t;
function miopenSetTensorDescriptorV2(const tensorDesc: miopenTensorDescriptor_t; const dataType: miopenDataType_t; nbDims: Integer; const dimsA, stridesA: Psize_t): miopenStatus_t; cdecl; external;
function miopenSetTensorCastType(const tensorDesc: miopenTensorDescriptor_t; const cast_type: miopenDataType_t): miopenStatus_t; cdecl; external;
{$ENDIF}

function miopenCreateSeqTensorDescriptor(var tensorDesc: miopenSeqTensorDescriptor_t): miopenStatus_t; cdecl; external;
function miopenDestroySeqTensorDescriptor(const tensorDesc: miopenSeqTensorDescriptor_t): miopenStatus_t; cdecl; external;

{ Tensor Operations }
function miopenOpTensor(const handle: miopenHandle_t; const tensorOp: miopenTensorOp_t; const alpha1: Pointer; const aDesc: miopenTensorDescriptor_t; const A: Pointer; const alpha2: Pointer; const bDesc: miopenTensorDescriptor_t; const B: Pointer; const beta: Pointer; const cDesc: miopenTensorDescriptor_t; C: Pointer): miopenStatus_t; cdecl; external;
function miopenSetTensor(const handle: miopenHandle_t; const yDesc: miopenTensorDescriptor_t; y: Pointer; const alpha: Pointer): miopenStatus_t; cdecl; external;
function miopenScaleTensor(const handle: miopenHandle_t; const yDesc: miopenTensorDescriptor_t; y: Pointer; const alpha: Pointer): miopenStatus_t; cdecl; external;
function miopenTransformTensor(const handle: miopenHandle_t; const alpha: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const beta: Pointer; const yDesc: miopenTensorDescriptor_t; y: Pointer): miopenStatus_t; cdecl; external;

{ Convolution }
function miopenCreateConvolutionDescriptor(var convDesc: miopenConvolutionDescriptor_t): miopenStatus_t; cdecl; external;
function miopenInitConvolutionDescriptor(const convDesc: miopenConvolutionDescriptor_t; const c_mode: miopenConvolutionMode_t; pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w: Integer): miopenStatus_t; cdecl; external;
function miopenGetConvolutionDescriptor(const convDesc: miopenConvolutionDescriptor_t; var c_mode: miopenConvolutionMode_t; var pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w: Integer): miopenStatus_t; cdecl; external;
function miopenGetConvolutionForwardOutputDim(const convDesc: miopenConvolutionDescriptor_t; const inputTensorDesc, filterDesc: miopenTensorDescriptor_t; var n, c, h, w: Integer): miopenStatus_t; cdecl; external;
function miopenDestroyConvolutionDescriptor(const convDesc: miopenConvolutionDescriptor_t): miopenStatus_t; cdecl; external;
function miopenConvolutionForwardGetWorkSpaceSize(const handle: miopenHandle_t; const wDesc, xDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const yDesc: miopenTensorDescriptor_t; var workSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenFindConvolutionForwardAlgorithm(const handle: miopenHandle_t; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const wDesc: miopenTensorDescriptor_t; const w: Pointer; const convDesc: miopenConvolutionDescriptor_t; const yDesc: miopenTensorDescriptor_t; y: Pointer; const requestAlgoCount: Integer; var returnedAlgoCount: Integer; perfResults: PmiopenConvAlgoPerf_t; workSpace: Pointer; workSpaceSize: size_t; exhaustiveSearch: Boolean): miopenStatus_t; cdecl; external;
function miopenConvolutionForward(const handle: miopenHandle_t; const alpha: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const wDesc: miopenTensorDescriptor_t; const w: Pointer; const convDesc: miopenConvolutionDescriptor_t; const algo: miopenConvFwdAlgorithm_t; const beta: Pointer; const yDesc: miopenTensorDescriptor_t; y: Pointer; workSpace: Pointer; workSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenConvolutionForwardBias(const handle: miopenHandle_t; const alpha: Pointer; const bDesc: miopenTensorDescriptor_t; const b, beta: Pointer; const yDesc: miopenTensorDescriptor_t; y: Pointer): miopenStatus_t; cdecl; external;
function miopenConvolutionBackwardDataGetWorkSpaceSize(const handle: miopenHandle_t; const dyDesc, wDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const dxDesc: miopenTensorDescriptor_t; var workSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenFindConvolutionBackwardDataAlgorithm(const handle: miopenHandle_t; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const wDesc: miopenTensorDescriptor_t; const w: Pointer; const convDesc: miopenConvolutionDescriptor_t; const dxDesc: miopenTensorDescriptor_t; dx: Pointer; const requestAlgoCount: Integer; var returnedAlgoCount: Integer; perfResults: PmiopenConvAlgoPerf_t; workSpace: Pointer; workSpaceSize: size_t; exhaustiveSearch: Boolean): miopenStatus_t; cdecl; external;
function miopenConvolutionBackwardData(const handle: miopenHandle_t; const alpha: Pointer; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const wDesc: miopenTensorDescriptor_t; const w: Pointer; const convDesc: miopenConvolutionDescriptor_t; const algo: miopenConvBwdDataAlgorithm_t; const beta: Pointer; const dxDesc: miopenTensorDescriptor_t; dx: Pointer; workSpace: Pointer; workSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenConvolutionBackwardWeightsGetWorkSpaceSize(const handle: miopenHandle_t; const dyDesc, xDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const dwDesc: miopenTensorDescriptor_t; var workSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenFindConvolutionBackwardWeightsAlgorithm(const handle: miopenHandle_t; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const convDesc: miopenConvolutionDescriptor_t; const dwDesc: miopenTensorDescriptor_t; dw: Pointer; const requestAlgoCount: Integer; var returnedAlgoCount: Integer; perfResults: PmiopenConvAlgoPerf_t; workSpace: Pointer; workSpaceSize: size_t; exhaustiveSearch: Boolean): miopenStatus_t; cdecl; external;
function miopenConvolutionBackwardWeights(const handle: miopenHandle_t; const alpha: Pointer; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const convDesc: miopenConvolutionDescriptor_t; const algo: miopenConvBwdWeightsAlgorithm_t; const beta: Pointer; const dwDesc: miopenTensorDescriptor_t; dw: Pointer; workSpace: Pointer; workSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenConvolutionBackwardBias(const handle: miopenHandle_t; const alpha: Pointer; const dyDesc: miopenTensorDescriptor_t; const dy, beta: Pointer; const dbDesc: miopenTensorDescriptor_t; db: Pointer): miopenStatus_t; cdecl; external;

{ N-Dimensional Convolution Functions }
function miopenInitConvolutionNdDescriptor(convDesc: miopenConvolutionDescriptor_t; spatialDim: Integer; const padA, strideA, dilationA: PInteger; c_mode: miopenConvolutionMode_t): miopenStatus_t; cdecl; external;
function miopenGetConvolutionSpatialDim(convDesc: miopenConvolutionDescriptor_t; var spatialDim: Integer): miopenStatus_t; cdecl; external;
function miopenGetConvolutionNdDescriptor(convDesc: miopenConvolutionDescriptor_t; requestedSpatialDim: Integer; var spatialDim: Integer; padA, strideA, dilationA: PInteger; var c_mode: miopenConvolutionMode_t): miopenStatus_t; cdecl; external;
function miopenGetConvolutionNdForwardOutputDim(convDesc: miopenConvolutionDescriptor_t; const inputTensorDesc, filterDesc: miopenTensorDescriptor_t; var nDim: Integer; outputTensorDimA: PInteger): miopenStatus_t; cdecl; external;

{ Convolution Group Count Functions }
function miopenSetConvolutionGroupCount(convDesc: miopenConvolutionDescriptor_t; groupCount: Integer): miopenStatus_t; cdecl; external;
function miopenGetConvolutionGroupCount(convDesc: miopenConvolutionDescriptor_t; var groupCount: Integer): miopenStatus_t; cdecl; external;

{ Transpose Convolution Padding Functions }
function miopenSetTransposeConvOutputPadding(convDesc: miopenConvolutionDescriptor_t; adj_h, adj_w: Integer): miopenStatus_t; cdecl; external;
function miopenSetTransposeConvNdOutputPadding(convDesc: miopenConvolutionDescriptor_t; spatialDim: Integer; const adjA: PInteger): miopenStatus_t; cdecl; external;

{ Convolution Attribute Functions }
function miopenSetConvolutionAttribute(convDesc: miopenConvolutionDescriptor_t; attr: miopenConvolutionAttrib_t; value: Integer): miopenStatus_t; cdecl; external;
function miopenGetConvolutionAttribute(convDesc: miopenConvolutionDescriptor_t; attr: miopenConvolutionAttrib_t; var value: Integer): miopenStatus_t; cdecl; external;


{ Convolution Solution/Immediate API (Forward) }
function miopenConvolutionForwardGetSolutionCount(handle: miopenHandle_t; const wDesc, xDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const yDesc: miopenTensorDescriptor_t; var solutionCount: size_t): miopenStatus_t; cdecl; external;
function miopenConvolutionForwardGetSolution(handle: miopenHandle_t; const wDesc, xDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const yDesc: miopenTensorDescriptor_t; const maxSolutionCount: size_t; var solutionCount: size_t; solutions: PmiopenConvSolution_t): miopenStatus_t; cdecl; external;
function miopenConvolutionForwardGetSolutionWorkspaceSize(handle: miopenHandle_t; const wDesc, xDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const yDesc: miopenTensorDescriptor_t; const solution_id: UInt64; var workSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenConvolutionForwardCompileSolution(handle: miopenHandle_t; const wDesc, xDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const yDesc: miopenTensorDescriptor_t; const solution_id: UInt64): miopenStatus_t; cdecl; external;
function miopenConvolutionForwardImmediate(handle: miopenHandle_t; const wDesc: miopenTensorDescriptor_t; const w: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const convDesc: miopenConvolutionDescriptor_t; const yDesc: miopenTensorDescriptor_t; y: Pointer; workSpace: Pointer; workSpaceSize: size_t; const solution_id: UInt64): miopenStatus_t; cdecl; external;

{ Convolution Solution/Immediate API (Backward Data) }
function miopenConvolutionBackwardDataGetSolutionCount(handle: miopenHandle_t; const dyDesc, wDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const dxDesc: miopenTensorDescriptor_t; var solutionCount: size_t): miopenStatus_t; cdecl; external;
function miopenConvolutionBackwardDataGetSolution(handle: miopenHandle_t; const dyDesc, wDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const dxDesc: miopenTensorDescriptor_t; const maxSolutionCount: size_t; var solutionCount: size_t; solutions: PmiopenConvSolution_t): miopenStatus_t; cdecl; external;
function miopenConvolutionBackwardDataGetSolutionWorkspaceSize(handle: miopenHandle_t; const dyDesc, wDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const dxDesc: miopenTensorDescriptor_t; const solution_id: UInt64; var workSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenConvolutionBackwardDataCompileSolution(handle: miopenHandle_t; const dyDesc, wDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const dxDesc: miopenTensorDescriptor_t; const solution_id: UInt64): miopenStatus_t; cdecl; external;
function miopenConvolutionBackwardDataImmediate(handle: miopenHandle_t; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const wDesc: miopenTensorDescriptor_t; const w: Pointer; const convDesc: miopenConvolutionDescriptor_t; const dxDesc: miopenTensorDescriptor_t; dx: Pointer; workSpace: Pointer; workSpaceSize: size_t; const solution_id: UInt64): miopenStatus_t; cdecl; external;

{ Convolution Solution/Immediate API (Backward Weights) }
function miopenConvolutionBackwardWeightsGetSolutionCount(handle: miopenHandle_t; const dyDesc, xDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const dwDesc: miopenTensorDescriptor_t; var solutionCount: size_t): miopenStatus_t; cdecl; external;
function miopenConvolutionBackwardWeightsGetSolution(handle: miopenHandle_t; const dyDesc, xDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const dwDesc: miopenTensorDescriptor_t; const maxSolutionCount: size_t; var solutionCount: size_t; solutions: PmiopenConvSolution_t): miopenStatus_t; cdecl; external;
function miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(handle: miopenHandle_t; const dyDesc, xDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const dwDesc: miopenTensorDescriptor_t; const solution_id: UInt64; var workSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenConvolutionBackwardWeightsCompileSolution(handle: miopenHandle_t; const dyDesc, xDesc: miopenTensorDescriptor_t; const convDesc: miopenConvolutionDescriptor_t; const dwDesc: miopenTensorDescriptor_t; const solution_id: UInt64): miopenStatus_t; cdecl; external;
function miopenConvolutionBackwardWeightsImmediate(handle: miopenHandle_t; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const convDesc: miopenConvolutionDescriptor_t; const dwDesc: miopenTensorDescriptor_t; dw: Pointer; workSpace: Pointer; workSpaceSize: size_t; const solution_id: UInt64): miopenStatus_t; cdecl; external;


{ Activation }
function miopenCreateActivationDescriptor(var activDesc: miopenActivationDescriptor_t): miopenStatus_t; cdecl; external;
function miopenSetActivationDescriptor(const activDesc: miopenActivationDescriptor_t; const mode: miopenActivationMode_t; activAlpha, activBeta, activGamma: Double): miopenStatus_t; cdecl; external;
function miopenGetActivationDescriptor(const activDesc: miopenActivationDescriptor_t; var mode: miopenActivationMode_t; var activAlpha, activBeta, activGamma: Double): miopenStatus_t; cdecl; external;
function miopenActivationForward(const handle: miopenHandle_t; const activDesc: miopenActivationDescriptor_t; const alpha: Pointer; const xDesc: miopenTensorDescriptor_t; const x, beta: Pointer; const yDesc: miopenTensorDescriptor_t; y: Pointer): miopenStatus_t; cdecl; external;
function miopenActivationBackward(const handle: miopenHandle_t; const activDesc: miopenActivationDescriptor_t; const alpha: Pointer; const yDesc: miopenTensorDescriptor_t; const y: Pointer; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const beta: Pointer; const dxDesc: miopenTensorDescriptor_t; dx: Pointer): miopenStatus_t; cdecl; external;
function miopenDestroyActivationDescriptor(const activDesc: miopenActivationDescriptor_t): miopenStatus_t; cdecl; external;

{ Pooling }
function miopenCreatePoolingDescriptor(var poolDesc: miopenPoolingDescriptor_t): miopenStatus_t; cdecl; external;
function miopenSet2dPoolingDescriptor(const poolDesc: miopenPoolingDescriptor_t; const mode: miopenPoolingMode_t; windowHeight, windowWidth, pad_h, pad_w, stride_h, stride_w: Integer): miopenStatus_t; cdecl; external;
function miopenGet2dPoolingDescriptor(const poolDesc: miopenPoolingDescriptor_t; var mode: miopenPoolingMode_t; var windowHeight, windowWidth, pad_h, pad_w, stride_h, stride_w: Integer): miopenStatus_t; cdecl; external;
function miopenGetPoolingForwardOutputDim(const poolDesc: miopenPoolingDescriptor_t; const tensorDesc: miopenTensorDescriptor_t; var n, c, h, w: Integer): miopenStatus_t; cdecl; external;
function miopenPoolingGetWorkSpaceSize(const yDesc: miopenTensorDescriptor_t; var workSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenPoolingForward(const handle: miopenHandle_t; const poolDesc: miopenPoolingDescriptor_t; const alpha: Pointer; const xDesc: miopenTensorDescriptor_t; const x, beta: Pointer; const yDesc: miopenTensorDescriptor_t; y: Pointer; do_backward: Boolean; workSpace: Pointer; workSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenPoolingBackward(const handle: miopenHandle_t; const poolDesc: miopenPoolingDescriptor_t; const alpha: Pointer; const yDesc: miopenTensorDescriptor_t; const y: Pointer; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const beta: Pointer; const dxDesc: miopenTensorDescriptor_t; dx: Pointer; workSpace: Pointer): miopenStatus_t; cdecl; external;
function miopenDestroyPoolingDescriptor(const poolDesc: miopenPoolingDescriptor_t): miopenStatus_t; cdecl; external;
function miopenSetPoolingIndexType(poolDesc: miopenPoolingDescriptor_t; index_type: miopenIndexType_t): miopenStatus_t; cdecl; external;
function miopenGetPoolingIndexType(poolDesc: miopenPoolingDescriptor_t; var index_type: miopenIndexType_t): miopenStatus_t; cdecl; external;
function miopenSetPoolingWorkSpaceIndexMode(poolDesc: miopenPoolingDescriptor_t; workspace_index: miopenPoolingWorkspaceIndexMode_t): miopenStatus_t; cdecl; external;
function miopenGetPoolingWorkSpaceIndexMode(poolDesc: miopenPoolingDescriptor_t; var workspace_index: miopenPoolingWorkspaceIndexMode_t): miopenStatus_t; cdecl; external;
function miopenSetNdPoolingDescriptor(poolDesc: miopenPoolingDescriptor_t; const mode: miopenPoolingMode_t; nbDims: Integer; const windowDimA, padA, stridesA: PInteger): miopenStatus_t; cdecl; external;
function miopenGetNdPoolingDescriptor(const poolDesc: miopenPoolingDescriptor_t; nbDimsRequested: Integer; var mode: miopenPoolingMode_t; var nbDims: Integer; windowDimA, padA, stridesA: PInteger): miopenStatus_t; cdecl; external;
function miopenGetPoolingNdForwardOutputDim(const poolDesc: miopenPoolingDescriptor_t; const tensorDesc: miopenTensorDescriptor_t; dims: Integer; tensorDimArr: PInteger): miopenStatus_t; cdecl; external;
function miopenPoolingGetWorkSpaceSizeV2(const poolDesc: miopenPoolingDescriptor_t; const yDesc: miopenTensorDescriptor_t; var workSpaceSize: size_t): miopenStatus_t; cdecl; external;

{ LRN (Local Response Normalization) API }
function miopenCreateLRNDescriptor(var lrnDesc: miopenLRNDescriptor_t): miopenStatus_t; cdecl; external;
function miopenSetLRNDescriptor(const lrnDesc: miopenLRNDescriptor_t; mode: miopenLRNMode_t; lrnN: Cardinal; lrnAlpha, lrnBeta, lrnK: Double): miopenStatus_t; cdecl; external;
function miopenGetLRNDescriptor(const lrnDesc: miopenLRNDescriptor_t; var mode: miopenLRNMode_t; var lrnN: Cardinal; var lrnAlpha, lrnBeta, lrnK: Double): miopenStatus_t; cdecl; external;
function miopenLRNGetWorkSpaceSize(const yDesc: miopenTensorDescriptor_t; var workSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenLRNForward(handle: miopenHandle_t; const lrnDesc: miopenLRNDescriptor_t; const alpha: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const beta: Pointer; const yDesc: miopenTensorDescriptor_t; y: Pointer; do_backward: Boolean; workSpace: Pointer): miopenStatus_t; cdecl; external;
function miopenLRNBackward(handle: miopenHandle_t; const lrnDesc: miopenLRNDescriptor_t; const alpha: Pointer; const yDesc: miopenTensorDescriptor_t; const y: Pointer; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const beta: Pointer; const dxDesc: miopenTensorDescriptor_t; dx: Pointer; const workSpace: Pointer): miopenStatus_t; cdecl; external;
function miopenDestroyLRNDescriptor(lrnDesc: miopenLRNDescriptor_t): miopenStatus_t; cdecl; external;

{ Softmax }
function miopenSoftmaxForward_V2(const handle: miopenHandle_t; const alpha: Pointer; const xDesc: miopenTensorDescriptor_t; const x, beta: Pointer; const yDesc: miopenTensorDescriptor_t; y: Pointer; const algorithm: miopenSoftmaxAlgorithm_t; const mode: miopenSoftmaxMode_t): miopenStatus_t; cdecl; external;
function miopenSoftmaxBackward_V2(const handle: miopenHandle_t; const alpha: Pointer; const yDesc: miopenTensorDescriptor_t; const y: Pointer; const dyDesc: miopenTensorDescriptor_t; const dy, beta: Pointer; const dxDesc: miopenTensorDescriptor_t; dx: Pointer; const algorithm: miopenSoftmaxAlgorithm_t; const mode: miopenSoftmaxMode_t): miopenStatus_t; cdecl; external;
function miopenSoftmaxForward(const handle: miopenHandle_t; const alpha: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const beta: Pointer; const yDesc: miopenTensorDescriptor_t; y: Pointer): miopenStatus_t; cdecl; external;
function miopenSoftmaxBackward(const handle: miopenHandle_t; const alpha: Pointer; const yDesc: miopenTensorDescriptor_t; const y: Pointer; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const beta: Pointer; const dxDesc: miopenTensorDescriptor_t; dx: Pointer): miopenStatus_t; cdecl; external;

{ Batch Normalization }
function miopenDeriveBNTensorDescriptor(const derivedBnDesc, xDesc: miopenTensorDescriptor_t; const bn_mode: miopenBatchNormMode_t): miopenStatus_t; cdecl; external;
function miopenBatchNormalizationForwardTraining(const handle: miopenHandle_t; const bn_mode: miopenBatchNormMode_t; const alpha, beta: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const yDesc: miopenTensorDescriptor_t; y: Pointer; const bnScaleBiasMeanVarDesc: miopenTensorDescriptor_t; bnScale, bnBias: Pointer; expAvgFactor: Double; resultRunningMean, resultRunningVariance: Pointer; epsilon: Double; resultSaveMean, resultSaveInvVariance: Pointer): miopenStatus_t; cdecl; external;
function miopenBatchNormalizationForwardInference(const handle: miopenHandle_t; const bn_mode: miopenBatchNormMode_t; const alpha, beta: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const yDesc: miopenTensorDescriptor_t; y: Pointer; const bnScaleBiasMeanVarDesc: miopenTensorDescriptor_t; const bnScale, bnBias, estimatedMean, estimatedVariance: Pointer; epsilon: Double): miopenStatus_t; cdecl; external;
function miopenBatchNormalizationBackward(const handle: miopenHandle_t; const bn_mode: miopenBatchNormMode_t; const alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const dxDesc: miopenTensorDescriptor_t; dx: Pointer; const bnScaleBiasDiffDesc: miopenTensorDescriptor_t; const bnScale: Pointer; resultBnScaleDiff, resultBnBiasDiff: Pointer; epsilon: Double; const savedMean, savedInvVariance: Pointer): miopenStatus_t; cdecl; external;
function miopenBatchNormalizationForwardTraining_V2(const handle: miopenHandle_t; const bn_mode: miopenBatchNormMode_t; const alpha, beta: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const yDesc: miopenTensorDescriptor_t; y: Pointer; const scaleDesc, biasVarDesc, savedMeanDesc, savedVarDesc: miopenTensorDescriptor_t; const bnScale, bnBias: Pointer; expAvgFactor: Double; resultRunningMean, resultRunningVariance: Pointer; epsilon: Double; resultSaveMean, resultSaveInvVariance: Pointer): miopenStatus_t; cdecl; external;
function miopenBatchNormalizationForwardInference_V2(const handle: miopenHandle_t; const bn_mode: miopenBatchNormMode_t; const alpha, beta: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const yDesc: miopenTensorDescriptor_t; y: Pointer; const scaleDesc, biasDesc, estMeanDesc, estVarianceDesc: miopenTensorDescriptor_t; const bnScale, bnBias, estimatedMean, estimatedVariance: Pointer; epsilon: Double): miopenStatus_t; cdecl; external;
function miopenBatchNormalizationBackward_V2(const handle: miopenHandle_t; const bn_mode: miopenBatchNormMode_t; const alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const dxDesc: miopenTensorDescriptor_t; dx: Pointer; const scaleDesc, biasDesc, savedMeanDesc, savedVarDesc: miopenTensorDescriptor_t; const bnScale: Pointer; resultBnScaleDiff, resultBnBiasDiff: Pointer; epsilon: Double; const savedMean, savedInvVariance: Pointer): miopenStatus_t; cdecl; external;


{ Dropout }
function miopenCreateDropoutDescriptor(var dropoutDesc: miopenDropoutDescriptor_t): miopenStatus_t; cdecl; external;
function miopenDestroyDropoutDescriptor(const dropoutDesc: miopenDropoutDescriptor_t): miopenStatus_t; cdecl; external;
function miopenDropoutGetReserveSpaceSize(const xDesc: miopenTensorDescriptor_t; var reserveSpaceSizeInBytes: size_t): miopenStatus_t; cdecl; external;
function miopenDropoutGetStatesSize(const handle: miopenHandle_t; var stateSizeInBytes: size_t): miopenStatus_t; cdecl; external;
function miopenSetDropoutDescriptor(const dropoutDesc: miopenDropoutDescriptor_t; const handle: miopenHandle_t; dropout: Single; states: Pointer; stateSizeInBytes: size_t; seed: UInt64; use_mask, state_evo: Boolean; rng_mode: miopenRNGType_t): miopenStatus_t; cdecl; external;
function miopenDropoutForward(const handle: miopenHandle_t; const dropoutDesc: miopenDropoutDescriptor_t; const noise_shape, xDesc: miopenTensorDescriptor_t; const x: Pointer; const yDesc: miopenTensorDescriptor_t; y, reserveSpace: Pointer; reserveSpaceSizeInBytes: size_t): miopenStatus_t; cdecl; external;
function miopenDropoutBackward(const handle: miopenHandle_t; const dropoutDesc: miopenDropoutDescriptor_t; const noise_shape, dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const dxDesc: miopenTensorDescriptor_t; dx, reserveSpace: Pointer; reserveSpaceSizeInBytes: size_t): miopenStatus_t; cdecl; external;

{ === MIOpen Fusion API === }

{ --- Fusion Plan Management --- }

function miopenCreateFusionPlan(var fusePlanDesc: miopenFusionPlanDescriptor_t; const fuseDirection: miopenFusionDirection_t; const inputDesc: miopenTensorDescriptor_t): miopenStatus_t; cdecl; external;
function miopenDestroyFusionPlan(fusePlanDesc: miopenFusionPlanDescriptor_t): miopenStatus_t; cdecl; external;
function miopenCompileFusionPlan(handle: miopenHandle_t; fusePlanDesc: miopenFusionPlanDescriptor_t): miopenStatus_t; cdecl; external;
function miopenFusionPlanGetOp(fusePlanDesc: miopenFusionPlanDescriptor_t; const op_idx: Integer; var op: miopenFusionOpDescriptor_t): miopenStatus_t; cdecl; external;
function miopenFusionPlanGetWorkSpaceSize(handle: miopenHandle_t; fusePlanDesc: miopenFusionPlanDescriptor_t; var workSpaceSize: size_t; algo: miopenConvFwdAlgorithm_t): miopenStatus_t; cdecl; external;
function miopenFusionPlanConvolutionGetAlgo(fusePlanDesc: miopenFusionPlanDescriptor_t; const requestAlgoCount: Integer; var returnedAlgoCount: Integer; returnedAlgos: PmiopenConvFwdAlgorithm_t): miopenStatus_t; cdecl; external;
function miopenFusionPlanConvolutionSetAlgo(fusePlanDesc: miopenFusionPlanDescriptor_t; algo: miopenConvFwdAlgorithm_t): miopenStatus_t; cdecl; external;

{ --- Operator Creation --- }

// Convolution Op
function miopenCreateOpConvForward(fusePlanDesc: miopenFusionPlanDescriptor_t; var convOp: miopenFusionOpDescriptor_t; convDesc: miopenConvolutionDescriptor_t; const wDesc: miopenTensorDescriptor_t): miopenStatus_t; cdecl; external;

// Activation Ops
function miopenCreateOpActivationForward(fusePlanDesc: miopenFusionPlanDescriptor_t; var activFwdOp: miopenFusionOpDescriptor_t; mode: miopenActivationMode_t): miopenStatus_t; cdecl; external;
function miopenCreateOpActivationBackward(fusePlanDesc: miopenFusionPlanDescriptor_t; var activBwdOp: miopenFusionOpDescriptor_t; mode: miopenActivationMode_t): miopenStatus_t; cdecl; external;

// Bias Op
function miopenCreateOpBiasForward(fusePlanDesc: miopenFusionPlanDescriptor_t; var biasOp: miopenFusionOpDescriptor_t; const bDesc: miopenTensorDescriptor_t): miopenStatus_t; cdecl; external;

// Batch Normalization Ops
function miopenCreateOpBatchNormInference(fusePlanDesc: miopenFusionPlanDescriptor_t; var bnOp: miopenFusionOpDescriptor_t; const bn_mode: miopenBatchNormMode_t; const bnScaleBiasMeanVarDesc: miopenTensorDescriptor_t): miopenStatus_t; cdecl; external;
function miopenCreateOpBatchNormForward(fusePlanDesc: miopenFusionPlanDescriptor_t; var bnFwdOp: miopenFusionOpDescriptor_t; const bn_mode: miopenBatchNormMode_t; runningMeanVariance: Boolean): miopenStatus_t; cdecl; external;
function miopenCreateOpBatchNormBackward(fusePlanDesc: miopenFusionPlanDescriptor_t; var bnBwdOp: miopenFusionOpDescriptor_t; const bn_mode: miopenBatchNormMode_t): miopenStatus_t; cdecl; external;

{ --- Operator Arguments Management --- }

function miopenCreateOperatorArgs(var args: miopenOperatorArgs_t): miopenStatus_t; cdecl; external;
function miopenDestroyOperatorArgs(args: miopenOperatorArgs_t): miopenStatus_t; cdecl; external;

{ --- Setting Operator Arguments --- }

// Convolution Args
function miopenSetOpArgsConvForward(args: miopenOperatorArgs_t; const convOp: miopenFusionOpDescriptor_t; const alpha, beta: Pointer; const w: Pointer): miopenStatus_t; cdecl; external;

// Activation Args
function miopenSetOpArgsActivForward(args: miopenOperatorArgs_t; const activFwdOp: miopenFusionOpDescriptor_t; const alpha, beta: Pointer; activAlpha, activBeta, activGamma: Double): miopenStatus_t; cdecl; external;
function miopenSetOpArgsActivBackward(args: miopenOperatorArgs_t; const activBwdOp: miopenFusionOpDescriptor_t; const alpha, beta: Pointer; const y, reserved: Pointer; activAlpha, activBeta, activGamma: Double): miopenStatus_t; cdecl; external;

// Batch Normalization Args
function miopenSetOpArgsBatchNormInference(args: miopenOperatorArgs_t; const bnOp: miopenFusionOpDescriptor_t; const alpha, beta: Pointer; const bnScale, bnBias, estimatedMean, estimatedVariance: Pointer; epsilon: Double): miopenStatus_t; cdecl; external;
function miopenSetOpArgsBatchNormForward(args: miopenOperatorArgs_t; const bnOp: miopenFusionOpDescriptor_t; const alpha, beta: Pointer; const bnScale, bnBias: Pointer; savedMean, savedInvVariance, runningMean, runningVariance: Pointer; expAvgFactor, epsilon: Double): miopenStatus_t; cdecl; external;
function miopenSetOpArgsBatchNormBackward(args: miopenOperatorArgs_t; const bnOp: miopenFusionOpDescriptor_t; const alpha, beta: Pointer; const x, bnScale, bnBias: Pointer; resultBnScaleDiff, resultBnBiasDiff: Pointer; const savedMean, savedInvVariance: Pointer): miopenStatus_t; cdecl; external;

// Bias Forward Args
function miopenSetOpArgsBiasForward(args: miopenOperatorArgs_t; const biasOp: miopenFusionOpDescriptor_t; const alpha, beta: Pointer; const bias: Pointer): miopenStatus_t; cdecl; external;

{ --- Execution --- }

function miopenExecuteFusionPlan(const handle: miopenHandle_t; const fusePlanDesc: miopenFusionPlanDescriptor_t; const inputDesc: miopenTensorDescriptor_t; const input: Pointer; const outputDesc: miopenTensorDescriptor_t; output: Pointer; args: miopenOperatorArgs_t): miopenStatus_t; cdecl; external;

{ --- Legacy Fused Function --- }

function miopenConvolutionBiasActivationForward(handle: miopenHandle_t; const alpha1: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const wDesc: miopenTensorDescriptor_t; const w: Pointer; const convDesc: miopenConvolutionDescriptor_t; algo: miopenConvFwdAlgorithm_t; workspace: Pointer; workspaceSizeInBytes: size_t; const alpha2: Pointer; const zDesc: miopenTensorDescriptor_t; const z: Pointer; const biasDesc: miopenTensorDescriptor_t; const bias: Pointer; const activationDesc: miopenActivationDescriptor_t; const yDesc: miopenTensorDescriptor_t; y: Pointer): miopenStatus_t; cdecl; external;

{$IFDEF BACKEND}

{ Backend API Functions }
function miopenBackendCreateDescriptor(descriptorType: miopenBackendDescriptorType_t; var descriptor: miopenBackendDescriptor_t): miopenStatus_t; cdecl; external LIBMIOPEN;
function miopenBackendSetAttribute(descriptor: miopenBackendDescriptor_t; attributeName: miopenBackendAttributeName_t; attributeType: miopenBackendAttributeType_t; elementCount: Int64; arrayOfElements: Pointer): miopenStatus_t; cdecl; external LIBMIOPEN;
function miopenBackendFinalize(descriptor: miopenBackendDescriptor_t): miopenStatus_t; cdecl; external LIBMIOPEN;
function miopenBackendGetAttribute(descriptor: miopenBackendDescriptor_t; attributeName: miopenBackendAttributeName_t; attributeType: miopenBackendAttributeType_t; requestedElementCount: Int64; var elementCount: Int64; arrayOfElements: Pointer): miopenStatus_t; cdecl; external LIBMIOPEN;
function miopenBackendExecute(handle: miopenHandle_t; executionPlan: miopenBackendDescriptor_t; variantPack: miopenBackendDescriptor_t): miopenStatus_t; cdecl; external LIBMIOPEN;
function miopenBackendDestroyDescriptor(descriptor: miopenBackendDescriptor_t): miopenStatus_t; cdecl; external LIBMIOPEN;
function miopenBackendInitialize(descriptor: miopenBackendDescriptor_t; descriptorType: miopenBackendDescriptorType_t; sizeInBytes: size_t): miopenStatus_t; cdecl; external LIBMIOPEN;

{$ENDIF}

{ === MIOpen RNN (Recurrent Neural Network) API === }

{ --- RNN Descriptor Management --- }

function miopenCreateRNNDescriptor(var rnnDesc: miopenRNNDescriptor_t): miopenStatus_t; cdecl; external;
function miopenDestroyRNNDescriptor(rnnDesc: miopenRNNDescriptor_t): miopenStatus_t; cdecl; external;

function miopenSetRNNDescriptor(rnnDesc: miopenRNNDescriptor_t; const hsize: Integer; const nlayers: Integer; inMode: miopenRNNInputMode_t; direction: miopenRNNDirectionMode_t; rnnMode: miopenRNNMode_t; biasMode: miopenRNNBiasMode_t; algo: miopenRNNAlgo_t; dataType: miopenDataType_t): miopenStatus_t; cdecl; external;
function miopenSetRNNDescriptor_V2(rnnDesc: miopenRNNDescriptor_t; const hsize: Integer; const nlayers: Integer; dropoutDesc: miopenDropoutDescriptor_t; inMode: miopenRNNInputMode_t; direction: miopenRNNDirectionMode_t; rnnMode: miopenRNNMode_t; biasMode: miopenRNNBiasMode_t; algo: miopenRNNAlgo_t; dataType: miopenDataType_t): miopenStatus_t; cdecl; external;

function miopenGetRNNDescriptor(rnnDesc: miopenRNNDescriptor_t; var rnnMode: miopenRNNMode_t; var algoMode: miopenRNNAlgo_t; var inputMode: miopenRNNInputMode_t; var dirMode: miopenRNNDirectionMode_t; var biasMode: miopenRNNBiasMode_t; var hiddenSize: Integer; var layer: Integer): miopenStatus_t; cdecl; external;
function miopenGetRNNDescriptor_V2(rnnDesc: miopenRNNDescriptor_t; var hiddenSize: Integer; var layer: Integer; var dropoutDesc: miopenDropoutDescriptor_t; var inputMode: miopenRNNInputMode_t; var dirMode: miopenRNNDirectionMode_t; var rnnMode: miopenRNNMode_t; var biasMode: miopenRNNBiasMode_t; var algoMode: miopenRNNAlgo_t; var dataType: miopenDataType_t): miopenStatus_t; cdecl; external;

{ --- RNN Sequence Tensor Management --- }

function miopenSetRNNDataSeqTensorDescriptor(seqTensorDesc: miopenSeqTensorDescriptor_t; dataType: miopenDataType_t; layout: miopenRNNBaseLayout_t; maxSequenceLen: Integer; batchSize: Integer; vectorSize: Integer; const sequenceLenArray: PInteger; paddingMarker: Pointer): miopenStatus_t; cdecl; external;
function miopenGetRNNDataSeqTensorDescriptor(seqTensorDesc: miopenSeqTensorDescriptor_t; var dataType: miopenDataType_t; var layout: miopenRNNBaseLayout_t; var maxSequenceLen: Integer; var batchSize: Integer; var vectorSize: Integer; sequenceLenArrayLimit: Integer; sequenceLenArray: PInteger; var paddingMarker: Pointer): miopenStatus_t; cdecl; external;

{ --- RNN Padding Mode --- }

function miopenSetRNNPaddingMode(rnnDesc: miopenRNNDescriptor_t; paddingMode: miopenRNNPaddingMode_t): miopenStatus_t; cdecl; external;
function miopenGetRNNPaddingMode(rnnDesc: miopenRNNDescriptor_t; var paddingMode: miopenRNNPaddingMode_t): miopenStatus_t; cdecl; external;

{ --- RNN Size Queries --- }

function miopenGetRNNWorkspaceSize(handle: miopenHandle_t; const rnnDesc: miopenRNNDescriptor_t; const sequenceLen: Integer; const xDesc: PmiopenTensorDescriptor_t; var numBytes: size_t): miopenStatus_t; cdecl; external;
function miopenGetRNNTrainingReserveSize(handle: miopenHandle_t; rnnDesc: miopenRNNDescriptor_t; const sequenceLen: Integer; const xDesc: PmiopenTensorDescriptor_t; var numBytes: size_t): miopenStatus_t; cdecl; external;
function miopenGetRNNTempSpaceSizes(handle: miopenHandle_t; rnnDesc: miopenRNNDescriptor_t; xDesc: miopenSeqTensorDescriptor_t; fwdMode: miopenRNNFWDMode_t; var workSpaceSize: size_t; var reserveSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenGetRNNParamsSize(handle: miopenHandle_t; rnnDesc: miopenRNNDescriptor_t; xDesc: miopenTensorDescriptor_t; var numBytes: size_t; dtype: miopenDataType_t): miopenStatus_t; cdecl; external;
function miopenGetRNNParamsDescriptor(handle: miopenHandle_t; rnnDesc: miopenRNNDescriptor_t; xDesc: miopenTensorDescriptor_t; wDesc: miopenTensorDescriptor_t; dtype: miopenDataType_t): miopenStatus_t; cdecl; external;
function miopenGetRNNInputTensorSize(handle: miopenHandle_t; rnnDesc: miopenRNNDescriptor_t; const seqLen: Integer; xDesc: PmiopenTensorDescriptor_t; var numBytes: size_t): miopenStatus_t; cdecl; external;
function miopenGetRNNHiddenTensorSize(handle: miopenHandle_t; rnnDesc: miopenRNNDescriptor_t; const seqLen: Integer; xDesc: PmiopenTensorDescriptor_t; var numBytes: size_t): miopenStatus_t; cdecl; external;
function miopenGetRNNLayerParamSize(handle: miopenHandle_t; rnnDesc: miopenRNNDescriptor_t; const layer: Integer; xDesc: miopenTensorDescriptor_t; const paramID: Integer; var numBytes: size_t): miopenStatus_t; cdecl; external;
function miopenGetRNNLayerBiasSize(handle: miopenHandle_t; rnnDesc: miopenRNNDescriptor_t; const layer: Integer; const biasID: Integer; var numBytes: size_t): miopenStatus_t; cdecl; external;

{ --- RNN Parameter Access --- }

function miopenGetRNNLayerParam(handle: miopenHandle_t; rnnDesc: miopenRNNDescriptor_t; const layer: Integer; xDesc: miopenTensorDescriptor_t; wDesc: miopenTensorDescriptor_t; const w: Pointer; const paramID: Integer; paramDesc: miopenTensorDescriptor_t; layerParam: Pointer): miopenStatus_t; cdecl; external;
function miopenGetRNNLayerBias(handle: miopenHandle_t; rnnDesc: miopenRNNDescriptor_t; const layer: Integer; xDesc: miopenTensorDescriptor_t; wDesc: miopenTensorDescriptor_t; const w: Pointer; const biasID: Integer; biasDesc: miopenTensorDescriptor_t; layerBias: Pointer): miopenStatus_t; cdecl; external;
function miopenGetRNNLayerParamOffset(rnnDesc: miopenRNNDescriptor_t; const layer: Integer; xDesc: miopenTensorDescriptor_t; const paramID: Integer; paramDesc: miopenTensorDescriptor_t; var layerParamOffset: size_t): miopenStatus_t; cdecl; external;
function miopenGetRNNLayerBiasOffset(rnnDesc: miopenRNNDescriptor_t; const layer: Integer; xDesc: miopenTensorDescriptor_t; const biasID: Integer; biasDesc: miopenTensorDescriptor_t; var layerBiasOffset: size_t): miopenStatus_t; cdecl; external;
function miopenSetRNNLayerParam(handle: miopenHandle_t; rnnDesc: miopenRNNDescriptor_t; const layer: Integer; xDesc: miopenTensorDescriptor_t; wDesc: miopenTensorDescriptor_t; w: Pointer; const paramID: Integer; paramDesc: miopenTensorDescriptor_t; const layerParam: Pointer): miopenStatus_t; cdecl; external;
function miopenSetRNNLayerBias(handle: miopenHandle_t; rnnDesc: miopenRNNDescriptor_t; const layer: Integer; xDesc: miopenTensorDescriptor_t; wDesc: miopenTensorDescriptor_t; w: Pointer; const biasID: Integer; biasDesc: miopenTensorDescriptor_t; const layerBias: Pointer): miopenStatus_t; cdecl; external;

{ --- RNN Execution (Modern API with Sequence Tensors) --- }

function miopenRNNForward(handle: miopenHandle_t; const rnnDesc: miopenRNNDescriptor_t; fwdMode: miopenRNNFWDMode_t; const xDesc: miopenSeqTensorDescriptor_t; const x: Pointer; const hDesc: miopenTensorDescriptor_t; const hx: Pointer; hy: Pointer; const cDesc: miopenTensorDescriptor_t; const cx: Pointer; cy: Pointer; const yDesc: miopenSeqTensorDescriptor_t; y: Pointer; const w: Pointer; weightSpaceSize: size_t; workSpace: Pointer; workSpaceNumBytes: size_t; reserveSpace: Pointer; reserveSpaceNumBytes: size_t): miopenStatus_t; cdecl; external;
function miopenRNNBackwardSeqData(handle: miopenHandle_t; const rnnDesc: miopenRNNDescriptor_t; const yDesc: miopenSeqTensorDescriptor_t; const y, dy: Pointer; const hDesc: miopenTensorDescriptor_t; const hx, dhy: Pointer; dhx: Pointer; const cDesc: miopenTensorDescriptor_t; const cx, dcy: Pointer; dcx: Pointer; const xDesc: miopenSeqTensorDescriptor_t; dx: Pointer; const w: Pointer; weightSpaceSize: size_t; workSpace: Pointer; workSpaceNumBytes: size_t; reserveSpace: Pointer; reserveSpaceNumBytes: size_t): miopenStatus_t; cdecl; external;
function miopenRNNBackwardWeightsSeqTensor(handle: miopenHandle_t; const rnnDesc: miopenRNNDescriptor_t; const xDesc: miopenSeqTensorDescriptor_t; const x: Pointer; const hDesc: miopenTensorDescriptor_t; const hx: Pointer; const yDesc: miopenSeqTensorDescriptor_t; const y: Pointer; dw: Pointer; weightSpaceSize: size_t; workSpace: Pointer; workSpaceNumBytes: size_t; const reserveSpace: Pointer; reserveSpaceNumBytes: size_t): miopenStatus_t; cdecl; external;

{ --- RNN Execution (Legacy API with Tensor Descriptor Arrays) --- }

function miopenRNNForwardTraining(handle: miopenHandle_t; const rnnDesc: miopenRNNDescriptor_t; const sequenceLen: Integer; const xDesc: PmiopenTensorDescriptor_t; const x: Pointer; const hxDesc: miopenTensorDescriptor_t; const hx: Pointer; const cxDesc: miopenTensorDescriptor_t; const cx: Pointer; const wDesc: miopenTensorDescriptor_t; const w: Pointer; const yDesc: PmiopenTensorDescriptor_t; y: Pointer; const hyDesc: miopenTensorDescriptor_t; hy: Pointer; const cyDesc: miopenTensorDescriptor_t; cy: Pointer; workSpace: Pointer; workSpaceNumBytes: size_t; reserveSpace: Pointer; reserveSpaceNumBytes: size_t): miopenStatus_t; cdecl; external;
function miopenRNNBackwardData(handle: miopenHandle_t; const rnnDesc: miopenRNNDescriptor_t; const sequenceLen: Integer; const yDesc: PmiopenTensorDescriptor_t; const y: Pointer; const dyDesc: PmiopenTensorDescriptor_t; const dy: Pointer; const dhyDesc: miopenTensorDescriptor_t; const dhy: Pointer; const dcyDesc: miopenTensorDescriptor_t; const dcy: Pointer; const wDesc: miopenTensorDescriptor_t; const w: Pointer; const hxDesc: miopenTensorDescriptor_t; const hx: Pointer; const cxDesc: miopenTensorDescriptor_t; const cx: Pointer; const dxDesc: PmiopenTensorDescriptor_t; dx: Pointer; const dhxDesc: miopenTensorDescriptor_t; dhx: Pointer; const dcxDesc: miopenTensorDescriptor_t; dcx: Pointer; workSpace: Pointer; workSpaceNumBytes: size_t; reserveSpace: Pointer; reserveSpaceNumBytes: size_t): miopenStatus_t; cdecl; external;
function miopenRNNBackwardWeights(handle: miopenHandle_t; const rnnDesc: miopenRNNDescriptor_t; const sequenceLen: Integer; const xDesc: PmiopenTensorDescriptor_t; const x: Pointer; const hxDesc: miopenTensorDescriptor_t; const hx: Pointer; const yDesc: PmiopenTensorDescriptor_t; const y: Pointer; const dwDesc: miopenTensorDescriptor_t; dw: Pointer; workSpace: Pointer; workSpaceNumBytes: size_t; const reserveSpace: Pointer; reserveSpaceNumBytes: size_t): miopenStatus_t; cdecl; external;
function miopenRNNForwardInference(handle: miopenHandle_t; rnnDesc: miopenRNNDescriptor_t; const sequenceLen: Integer; const xDesc: PmiopenTensorDescriptor_t; const x: Pointer; const hxDesc: miopenTensorDescriptor_t; const hx: Pointer; const cxDesc: miopenTensorDescriptor_t; const cx: Pointer; const wDesc: miopenTensorDescriptor_t; const w: Pointer; const yDesc: PmiopenTensorDescriptor_t; y: Pointer; const hyDesc: miopenTensorDescriptor_t; hy: Pointer; const cyDesc: miopenTensorDescriptor_t; cy: Pointer; workSpace: Pointer; workSpaceNumBytes: size_t): miopenStatus_t; cdecl; external;

{ === MIOpen Loss Function API === }

{ --- CTC Loss API --- }

function miopenCreateCTCLossDescriptor(var ctcLossDesc: miopenCTCLossDescriptor_t): miopenStatus_t; cdecl; external;
function miopenDestroyCTCLossDescriptor(ctcLossDesc: miopenCTCLossDescriptor_t): miopenStatus_t; cdecl; external;
function miopenSetCTCLossDescriptor(ctcLossDesc: miopenCTCLossDescriptor_t; dataType: miopenDataType_t; const blank_label_id: Integer; apply_softmax_layer: Boolean): miopenStatus_t; cdecl; external;
function miopenGetCTCLossDescriptor(ctcLossDesc: miopenCTCLossDescriptor_t; var dataType: miopenDataType_t; var blank_label_id: Integer; var apply_softmax_layer: Boolean): miopenStatus_t; cdecl; external;
function miopenGetCTCLossWorkspaceSize(handle: miopenHandle_t; const probsDesc: miopenTensorDescriptor_t; const gradientsDesc: miopenTensorDescriptor_t; const labels: PInteger; const labelLengths: PInteger; const inputLengths: PInteger; algo: miopenCTCLossAlgo_t; const ctcLossDesc: miopenCTCLossDescriptor_t; var workSpaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenCTCLoss(handle: miopenHandle_t; const probsDesc: miopenTensorDescriptor_t; const probs: Pointer; const labels: PInteger; const labelLengths: PInteger; const inputLengths: PInteger; losses: Pointer; const gradientsDesc: miopenTensorDescriptor_t; gradients: Pointer; algo: miopenCTCLossAlgo_t; const ctcLossDesc: miopenCTCLossDescriptor_t; workSpace: Pointer; workSpaceSize: size_t): miopenStatus_t; cdecl; external;


{$IFDEF MIOPEN_BETA_API}
{ --- SoftMarginLoss API --- }
function miopenGetSoftMarginLossForwardWorkspaceSize(handle: miopenHandle_t; inputDesc: miopenTensorDescriptor_t; targetDesc: miopenTensorDescriptor_t; outputDesc: miopenTensorDescriptor_t; reduction: miopenLossReductionMode_t; var sizeInBytes: size_t): miopenStatus_t; cdecl; external;
function miopenSoftMarginLossForward(handle: miopenHandle_t; inputDesc: miopenTensorDescriptor_t; const input: Pointer; targetDesc: miopenTensorDescriptor_t; const target: Pointer; outputDesc: miopenTensorDescriptor_t; output: Pointer; reduction: miopenLossReductionMode_t; workspace: Pointer = nil; workspaceSizeInBytes: size_t = 0): miopenStatus_t; cdecl; external;
function miopenSoftMarginLossBackward(handle: miopenHandle_t; inputDesc: miopenTensorDescriptor_t; const input: Pointer; targetDesc: miopenTensorDescriptor_t; const target: Pointer; doutputDesc: miopenTensorDescriptor_t; const doutput: Pointer; dinputDesc: miopenTensorDescriptor_t; dinput: Pointer; reduction: miopenLossReductionMode_t): miopenStatus_t; cdecl; external;

{ --- MultiMarginLoss API --- }
function miopenGetMultiMarginLossForwardWorkspaceSize(handle: miopenHandle_t; inputDesc: miopenTensorDescriptor_t; targetDesc: miopenTensorDescriptor_t; weightDesc: miopenTensorDescriptor_t; outputDesc: miopenTensorDescriptor_t; p: Int64; margin: Single; reduction: miopenLossReductionMode_t; var sizeInBytes: size_t): miopenStatus_t; cdecl; external;
function miopenMultiMarginLossForward(handle: miopenHandle_t; inputDesc: miopenTensorDescriptor_t; const input: Pointer; targetDesc: miopenTensorDescriptor_t; const target: Pointer; weightDesc: miopenTensorDescriptor_t; const weight: Pointer; outputDesc: miopenTensorDescriptor_t; output: Pointer; p: Int64; margin: Single; reduction: miopenLossReductionMode_t; workspace: Pointer; workspaceSizeInBytes: size_t): miopenStatus_t; cdecl; external;
{$ENDIF}

{ ReduceTensor }
function miopenCreateReduceTensorDescriptor(var reduceTensorDesc: miopenReduceTensorDescriptor_t): miopenStatus_t; cdecl; external;
function miopenDestroyReduceTensorDescriptor(const reduceTensorDesc: miopenReduceTensorDescriptor_t): miopenStatus_t; cdecl; external;
function miopenSetReduceTensorDescriptor(const reduceTensorDesc: miopenReduceTensorDescriptor_t; reduceTensorOp: miopenReduceTensorOp_t; reduceTensorCompType: miopenDataType_t; reduceTensorNanOpt: miopenNanPropagation_t; reduceTensorIndices: miopenReduceTensorIndices_t; reduceTensorIndicesType: miopenIndicesType_t): miopenStatus_t; cdecl; external;
function miopenGetReduceTensorDescriptor(const reduceTensorDesc: miopenReduceTensorDescriptor_t; var reduceTensorOp: miopenReduceTensorOp_t; var reduceTensorCompType: miopenDataType_t; var reduceTensorNanOpt: miopenNanPropagation_t; var reduceTensorIndices: miopenReduceTensorIndices_t; var reduceTensorIndicesType: miopenIndicesType_t): miopenStatus_t; cdecl; external;
function miopenGetReductionIndicesSize(const handle: miopenHandle_t; const reduceTensorDesc: miopenReduceTensorDescriptor_t; const aDesc: miopenTensorDescriptor_t; const cDesc: miopenTensorDescriptor_t; var sizeInBytes: size_t): miopenStatus_t; cdecl; external;
function miopenGetReductionWorkspaceSize(const handle: miopenHandle_t; const reduceTensorDesc: miopenReduceTensorDescriptor_t; const aDesc: miopenTensorDescriptor_t; const cDesc: miopenTensorDescriptor_t; var sizeInBytes: size_t): miopenStatus_t; cdecl; external;
function miopenReduceTensor(const handle: miopenHandle_t; const reduceTensorDesc: miopenReduceTensorDescriptor_t; indices: Pointer; indicesSizeInBytes: size_t; workspace: Pointer; workspaceSizeInBytes: size_t; const alpha: Pointer; const aDesc: miopenTensorDescriptor_t; const A: Pointer; const beta: Pointer; const cDesc: miopenTensorDescriptor_t; C: Pointer): miopenStatus_t; cdecl; external;

{ === MIOpen Find 2.0 API === }

{ --- Problem and Descriptor Creation --- }

function miopenCreateConvProblem(var problem: miopenProblem_t; operatorDesc: miopenConvolutionDescriptor_t; direction: miopenProblemDirection_t): miopenStatus_t; cdecl; external;
function miopenCreateMhaProblem(var problem: miopenProblem_t; operatorDesc: miopenMhaDescriptor_t; direction: miopenProblemDirection_t): miopenStatus_t; cdecl; external;

function miopenCreateMhaDescriptor(var mhaDesc: miopenMhaDescriptor_t): miopenStatus_t; cdecl; external;
function miopenSetMhaDescriptor(mhaDesc: miopenMhaDescriptor_t; scale: Single): miopenStatus_t; cdecl; external;
function miopenGetMhaDescriptor(mhaDesc: miopenMhaDescriptor_t; var scale: Single): miopenStatus_t; cdecl; external;

function miopenCreateSoftmaxDescriptor(var softmaxDesc: miopenSoftmaxDescriptor_t): miopenStatus_t; cdecl; external;
function miopenSetSoftmaxDescriptor(softmaxDesc: miopenSoftmaxDescriptor_t; alpha: Single; beta: Single; algorithm: miopenSoftmaxAlgorithm_t; mode: miopenSoftmaxMode_t): miopenStatus_t; cdecl; external;
function miopenGetSoftmaxDescriptor(const softmaxDesc: miopenSoftmaxDescriptor_t; var alpha: Single; var beta: Single; var algorithm: miopenSoftmaxAlgorithm_t; var mode: miopenSoftmaxMode_t): miopenStatus_t; cdecl; external;

function miopenDestroyProblem(const problem: miopenProblem_t): miopenStatus_t; cdecl; external;
function miopenSetProblemTensorDescriptor(const problem: miopenProblem_t; id: miopenTensorArgumentId_t; const descriptor: miopenTensorDescriptor_t): miopenStatus_t; cdecl; external;


{ --- Find Options Management --- }

function miopenCreateFindOptions(var options: miopenFindOptions_t): miopenStatus_t; cdecl; external;
function miopenDestroyFindOptions(const options: miopenFindOptions_t): miopenStatus_t; cdecl; external;

function miopenSetFindOptionTuning(const options: miopenFindOptions_t; value: Integer): miopenStatus_t; cdecl; external;
function miopenSetFindOptionResultsOrder(const options: miopenFindOptions_t; value: miopenFindResultsOrder_t): miopenStatus_t; cdecl; external;
function miopenSetFindOptionWorkspaceLimit(const options: miopenFindOptions_t; value: size_t): miopenStatus_t; cdecl; external;
function miopenSetFindOptionPreallocatedWorkspace(const options: miopenFindOptions_t; buffer: Pointer; size: size_t): miopenStatus_t; cdecl; external;
function miopenSetFindOptionPreallocatedTensor(const options: miopenFindOptions_t; id: miopenTensorArgumentId_t; buffer: Pointer): miopenStatus_t; cdecl; external;
function miopenSetFindOptionAttachBinaries(const options: miopenFindOptions_t; attach: Cardinal): miopenStatus_t; cdecl; external;


{ --- Solution Search and Execution --- }

// Note: You will need a pointer-to-solution-handle type
// PmiopenSolution_t = ^miopenSolution_t;
function miopenFindSolutions(handle: miopenHandle_t; const problem: miopenProblem_t; const options: miopenFindOptions_t; solutions: PmiopenSolution_t; var numSolutions: size_t; maxSolutions: size_t): miopenStatus_t; cdecl; external;

// Note: You will need the record and pointer type for miopenTensorArgument_t
function miopenRunSolution(handle: miopenHandle_t; const solution: miopenSolution_t; nInputs: size_t; const tensors: PmiopenTensorArgument_t; workspace: Pointer; workspaceSize: size_t): miopenStatus_t; cdecl; external;


{ --- Solution Management --- }

function miopenDestroySolution(const solution: miopenSolution_t): miopenStatus_t; cdecl; external;
function miopenLoadSolution(var solution: miopenSolution_t; const data: PAnsiChar; size: size_t): miopenStatus_t; cdecl; external;
function miopenSaveSolution(const solution: miopenSolution_t; data: PAnsiChar): miopenStatus_t; cdecl; external;

function miopenGetSolutionSize(const solution: miopenSolution_t; var size: size_t): miopenStatus_t; cdecl; external;
function miopenGetSolutionWorkspaceSize(const solution: miopenSolution_t; var workspaceSize: size_t): miopenStatus_t; cdecl; external;
function miopenGetSolutionTime(const solution: miopenSolution_t; var time: Single): miopenStatus_t; cdecl; external;
function miopenGetSolutionSolverId(const solution: miopenSolution_t; var solverId: UInt64): miopenStatus_t; cdecl; external;
function miopenGetSolverIdConvAlgorithm(solverId: UInt64; var result: miopenConvAlgorithm_t): miopenStatus_t; cdecl; external;


{ --- Beta API Problem Creation --- }

{$IFDEF MIOPEN_BETA_API}

function miopenCreateActivationProblem(var problem: miopenProblem_t; operatorDesc: miopenActivationDescriptor_t; direction: miopenProblemDirection_t): miopenStatus_t; cdecl; external;
function miopenCreateBatchnormProblem(var problem: miopenProblem_t; mode: miopenBatchNormMode_t; runningMeanVariance: Boolean; direction: miopenProblemDirection_t): miopenStatus_t; cdecl; external;
function miopenFuseProblems(const problem1: miopenProblem_t; const problem2: miopenProblem_t): miopenStatus_t; cdecl; external;
function miopenCreateBiasProblem(var problem: miopenProblem_t; direction: miopenProblemDirection_t): miopenStatus_t; cdecl; external;
function miopenCreateSoftmaxProblem(var problem: miopenProblem_t; operatorDesc: miopenSoftmaxDescriptor_t; direction: miopenProblemDirection_t): miopenStatus_t; cdecl; external;
function miopenGLUForward(handle: miopenHandle_t; const inputDesc: miopenTensorDescriptor_t; const input: Pointer; const outputDesc: miopenTensorDescriptor_t; output: Pointer; const dim: Cardinal): miopenStatus_t; cdecl; external;
function miopenGLUBackward(handle: miopenHandle_t; const inputDesc: miopenTensorDescriptor_t; const input: Pointer; const outputGradDesc: miopenTensorDescriptor_t; const outputGrad: Pointer; const inputGradDesc: miopenTensorDescriptor_t; inputGrad: Pointer; const dim: Cardinal): miopenStatus_t; cdecl; external;


{ --- Beta Extras --- }

function miopenLayerNormForward(handle: miopenHandle_t; mode: miopenNormMode_t; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const weightDesc: miopenTensorDescriptor_t; const weight: Pointer; const biasDesc: miopenTensorDescriptor_t; const bias: Pointer; epsilon: Single; normalized_dim: Int32; const yDesc: miopenTensorDescriptor_t; y: Pointer; const meanDesc: miopenTensorDescriptor_t; mean: Pointer; const rstdDesc: miopenTensorDescriptor_t; rstd: Pointer): miopenStatus_t; cdecl; external;
function miopenAddLayerNormForward(handle: miopenHandle_t; mode: miopenNormMode_t; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const x2Desc: miopenTensorDescriptor_t; const x2: Pointer; const weightDesc: miopenTensorDescriptor_t; const weight: Pointer; const biasDesc: miopenTensorDescriptor_t; const bias: Pointer; epsilon: Single; normalized_dim: Int32; const yDesc: miopenTensorDescriptor_t; y: Pointer; const meanDesc: miopenTensorDescriptor_t; mean: Pointer; const rstdDesc: miopenTensorDescriptor_t; rstd: Pointer): miopenStatus_t; cdecl; external;
function miopenT5LayerNormForward(handle: miopenHandle_t; mode: miopenNormMode_t; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const weightDesc: miopenTensorDescriptor_t; const weight: Pointer; epsilon: Single; const yDesc: miopenTensorDescriptor_t; y: Pointer; const rstdDesc: miopenTensorDescriptor_t; rstd: Pointer): miopenStatus_t; cdecl; external;
function miopenGetT5LayerNormBackwardWorkspaceSize(handle: miopenHandle_t; mode: miopenNormMode_t; const xDesc: miopenTensorDescriptor_t; const dyDesc: miopenTensorDescriptor_t;  const weightDesc: miopenTensorDescriptor_t; const rstdDesc: miopenTensorDescriptor_t; const dxDesc: miopenTensorDescriptor_t; const dwDesc: miopenTensorDescriptor_t; var sizeInBytes: size_t): miopenStatus_t; cdecl; external;
function miopenT5LayerNormBackward(handle: miopenHandle_t; mode: miopenNormMode_t; workspace: Pointer; workspaceSizeInBytes: size_t; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const weightDesc: miopenTensorDescriptor_t; const weight: Pointer; const rstdDesc: miopenTensorDescriptor_t; const rstd: Pointer; const dxDesc: miopenTensorDescriptor_t; dx: Pointer; const dwDesc: miopenTensorDescriptor_t; dw: Pointer): miopenStatus_t; cdecl; external;
function miopenCatForward(handle: miopenHandle_t; const xCount: Int32; const xDescs: PmiopenTensorDescriptor_t; const xs: PPointer; const yDesc: miopenTensorDescriptor_t; y: Pointer; const dim: Int32): miopenStatus_t; cdecl; external;
function miopenGLUForward(handle: miopenHandle_t; const inputDesc: miopenTensorDescriptor_t; const input: Pointer; const outputDesc: miopenTensorDescriptor_t; output: Pointer; const dim: Int32): miopenStatus_t; cdecl; external;
function miopenGLUBackward(handle: miopenHandle_t; const inputDesc: miopenTensorDescriptor_t; const input: Pointer; const outputGradDesc: miopenTensorDescriptor_t; const outputGrad: Pointer; const inputGradDesc: miopenTensorDescriptor_t; inputGrad: Pointer; const dim: Int32): miopenStatus_t; cdecl; external;
function miopenGroupNormForward(handle: miopenHandle_t; mode: miopenNormMode_t; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const weightDesc: miopenTensorDescriptor_t; const weight: Pointer; const biasDesc: miopenTensorDescriptor_t; const bias: Pointer; const groups: UInt64; epsilon: Single; const yDesc: miopenTensorDescriptor_t; y: Pointer; const meanDesc: miopenTensorDescriptor_t; mean: Pointer; const rstdDesc: miopenTensorDescriptor_t; rstd: Pointer): miopenStatus_t; cdecl; external;
function miopenGetReduceCalculationWorkspaceSize(handle: miopenHandle_t; const xDesc: miopenTensorDescriptor_t; const dim: Int32; const reduceCalculationOp: miopenReduceCalculationOp_t; const reduceDesc: miopenTensorDescriptor_t; var sizeInBytes: size_t): miopenStatus_t; cdecl; external;
function miopenReduceCalculationForward(handle: miopenHandle_t; nanPropagation: miopenReduceCalculationNanPropagation_t; workspace: Pointer; workspaceSizeInBytes: size_t; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const dim: Int32; const reduceCalculationOp: miopenReduceCalculationOp_t; const reduceDesc: miopenTensorDescriptor_t; y: Pointer): miopenStatus_t; cdecl; external;
function miopenReduceExtremeForward(handle: miopenHandle_t; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const dim: Int32; const reduceExtremeOp: miopenReduceExtremeOp_t; const yDesc: miopenTensorDescriptor_t; y: Pointer; const indiceDesc: miopenTensorDescriptor_t; indice: Pointer): miopenStatus_t; cdecl; external;
function miopenFusedAdam(
    handle: miopenHandle_t;
    const paramDesc: miopenTensorDescriptor_t;
    param: Pointer;
    const gradDesc: miopenTensorDescriptor_t;
    const grad: Pointer;
    const expAvgDesc: miopenTensorDescriptor_t;
    expAvg: Pointer;
    const expAvgSqDesc: miopenTensorDescriptor_t;
    expAvgSq: Pointer;
    const maxExpAvgSqDesc: miopenTensorDescriptor_t;
    maxExpAvgSq: Pointer;
    const stateStepDesc: miopenTensorDescriptor_t;
    stateStep: Pointer;
    const state_step: Cardinal; // C unsigned int -> Pascal Cardinal
    const lr: Single;           // C float -> Pascal Single (by value)
    const beta1: Single;        // C float -> Pascal Single (by value)
    const beta2: Single;        // C float -> Pascal Single (by value)
    const weight_decay: Single; // C float -> Pascal Single (by value)
    const eps: Single;          // C float -> Pascal Single (by value)
    amsgrad: Byte;              // C bool -> Pascal Byte (1 byte)
    maximize: Byte;             // C bool -> Pascal Byte (1 byte)
    adamw: Byte;                // C bool -> Pascal Byte (1 byte)
    const gradScaleDesc: miopenTensorDescriptor_t;
    const gradScale: Pointer;
    const foundInfDesc: miopenTensorDescriptor_t;
    const foundInf: Pointer
): miopenStatus_t; cdecl; external;

function miopenGetGetitemWorkspaceSize(
    handle: miopenHandle_t;
    indexCount: UInt32;
    indexDescs: PmiopenTensorDescriptor_t; // Pointer to an array of index descriptors
    var sizeInBytes: size_t
): miopenStatus_t; cdecl; external;

function miopenGetitemBackward(
    handle: miopenHandle_t;
    workspace: Pointer;
    workspaceSizeInBytes: size_t;
    const dyDesc: miopenTensorDescriptor_t;
    const dy: Pointer;
    indexCount: UInt32;                     // Correct Order
    indexDescs: PmiopenTensorDescriptor_t;  // Correct Order
    indexs: PPointer;                       // Correct Order
    const dxDesc: miopenTensorDescriptor_t;   // Correct Order
    dx: Pointer;                            // Correct Order
    const errorDesc: miopenTensorDescriptor_t;// Correct Order
    error: Pointer;                         // Correct Order
    dimCount: UInt32;                       // Correct Order
    dims: PInt32;                           // Correct Order
    sliceCount: UInt32;                     // Correct Order
    slices: PInt32;                         // Correct Order
    offset: UInt32                          // Correct Order
): miopenStatus_t; cdecl; external;

function miopenRoPEForward(handle: miopenHandle_t; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const cosDesc: miopenTensorDescriptor_t; const cosData: Pointer; const sinDesc: miopenTensorDescriptor_t; const sinData: Pointer; const yDesc: miopenTensorDescriptor_t; y: Pointer): miopenStatus_t; cdecl; external;
function miopenRoPEBackward(handle: miopenHandle_t; const dyDesc: miopenTensorDescriptor_t; const dy: Pointer; const cosDesc: miopenTensorDescriptor_t; const cosData: Pointer; const sinDesc: miopenTensorDescriptor_t; const sinData: Pointer; const dxDesc: miopenTensorDescriptor_t; dx: Pointer): miopenStatus_t; cdecl; external;
{ keepdim maps to a C++ bool (1 byte, True = 1). Do not use LongBool here:
  FPC's LongBool(True) is -1 and MIOpen reads that as False. }
function miopenKthvalueForward(handle: miopenHandle_t; const xDesc: miopenTensorDescriptor_t; const x: Pointer; const valuesDesc: miopenTensorDescriptor_t; values: Pointer; const indicesDesc: miopenTensorDescriptor_t; indices: Pointer; const k: size_t; const dim: Int32; keepdim: Boolean): miopenStatus_t; cdecl; external;
function miopenGetPReLUBackwardWorkspaceSize(handle: miopenHandle_t; const inputDesc: miopenTensorDescriptor_t; const weightDesc: miopenTensorDescriptor_t; var sizeInBytes: size_t): miopenStatus_t; cdecl; external;
function miopenPReLUBackward(handle: miopenHandle_t; workspace: Pointer; workspaceSizeInBytes: size_t; const inputDesc: miopenTensorDescriptor_t; const input: Pointer; const weightDesc: miopenTensorDescriptor_t; const weight: Pointer; const doutputDesc: miopenTensorDescriptor_t; const doutput: Pointer; const dinputDesc: miopenTensorDescriptor_t; dinput: Pointer; const dweightDesc: miopenTensorDescriptor_t; dweight: Pointer): miopenStatus_t; cdecl; external;

{$ENDIF}


implementation


end.
