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

type
  miopenAcceleratorQueue_t = hipStream_t;
  PPAnsiChar = ^PAnsiChar;
  size_t  = csize_t;
  Psize_t = ^size_t;
  
  
  // Phantom records to represent the opaque C structs.
  TmiopenHandle                 = record end;
  TmiopenFusionOpDescriptor     = record end;
  TmiopenTensorDescriptor       = record end;
  TmiopenSeqTensorDescriptor    = record end;
  TmiopenConvolutionDescriptor  = record end;
  TmiopenPoolingDescriptor      = record end;
  TmiopenLRNDescriptor          = record end;
  TmiopenActivationDescriptor   = record end;
  TmiopenRNNDescriptor          = record end;
  TmiopenCTCLossDescriptor      = record end;
  TmiopenDropoutDescriptor      = record end;
  TmiopenReduceTensorDescriptor = record end;
  TmiopenMhaDescriptor          = record end;
  TmiopenSoftmaxDescriptor      = record end;
  TmiopenFusionPlanDescriptor   = record end;
  TmiopenOperatorDescriptor     = record end;
  TmiopenOperatorArgs           = record end;
  TmiopenProblem                = record end;
  TmiopenFindOptions            = record end;
  TmiopenSolution               = record end;
  TmiopenBackendDescriptor      = record end;
  TmiopenTensorArgument_t       = record end;

  // C-style type aliases as typed pointers
  miopenHandle_t                  = ^TmiopenHandle;
  miopenFusionOpDescriptor_t      = ^TmiopenFusionOpDescriptor;
  miopenTensorDescriptor_t        = ^TmiopenTensorDescriptor;
  miopenSeqTensorDescriptor_t     = ^TmiopenSeqTensorDescriptor;
  miopenConvolutionDescriptor_t   = ^TmiopenConvolutionDescriptor;
  miopenPoolingDescriptor_t       = ^TmiopenPoolingDescriptor;
  miopenLRNDescriptor_t           = ^TmiopenLRNDescriptor;
  miopenActivationDescriptor_t    = ^TmiopenActivationDescriptor;
  miopenRNNDescriptor_t           = ^TmiopenRNNDescriptor;
  miopenCTCLossDescriptor_t       = ^TmiopenCTCLossDescriptor;
  miopenDropoutDescriptor_t       = ^TmiopenDropoutDescriptor;
  miopenReduceTensorDescriptor_t  = ^TmiopenReduceTensorDescriptor;
  miopenMhaDescriptor_t           = ^TmiopenMhaDescriptor;
  miopenSoftmaxDescriptor_t       = ^TmiopenSoftmaxDescriptor;
  miopenFusionPlanDescriptor_t    = ^TmiopenFusionPlanDescriptor;
  miopenOperatorDescriptor_t      = ^TmiopenOperatorDescriptor;
  miopenOperatorArgs_t            = ^TmiopenOperatorArgs;
  miopenProblem_t                 = ^TmiopenProblem;
  miopenFindOptions_t             = ^TmiopenFindOptions;
  miopenSolution_t                = ^TmiopenSolution;
  miopenBackendDescriptor_t       = ^TmiopenBackendDescriptor;

  // The original miopen.h uses MIOPEN_DECLARE_OBJECT which likely translates to a pointer type.
  MIOPEN_DECLARE_OBJECT           = Pointer;

  miopenStatus_t                  = Integer;
  miopenDataType_t                = Integer;
  miopenTensorLayout_t            = Integer;
  miopenIndexType_t               = Integer;
  miopenTensorOp_t                = Integer;
  miopenConvolutionMode_t         = Integer;
  miopenPaddingMode_t             = Integer;
  miopenPoolingMode_t             = Integer;
  miopenPoolingWorkspaceIndexMode_t = Integer;
  miopenLRNMode_t                 = Integer;
  miopenNormMode_t                = Integer;
  miopenBatchNormMode_t           = Integer;
  miopenActivationMode_t          = Integer;
  miopenSoftmaxAlgorithm_t        = Integer;
  miopenSoftmaxMode_t             = Integer;
  miopenReduceTensorOp_t          = Integer;
  miopenNanPropagation_t          = Integer;
  miopenReduceTensorIndices_t     = Integer;
  miopenIndicesType_t             = Integer;
  miopenConvolutionAttrib_t       = Integer;
  miopenConvFwdAlgorithm_t        = Integer;
  miopenConvBwdWeightsAlgorithm_t = Integer;
  miopenConvBwdDataAlgorithm_t    = Integer;
  miopenConvAlgorithm_t           = Integer;
  miopenFusionDirection_t         = Integer;
  miopenRNNMode_t                 = Integer;
  miopenRNNInputMode_t            = Integer;
  miopenRNNAlgo_t                 = Integer;
  miopenRNNDirectionMode_t        = Integer;
  miopenRNNBiasMode_t             = Integer;
  miopenRNNGEMMalgoMode_t         = Integer;
  miopenRNNPaddingMode_t          = Integer;
  miopenRNNFWDMode_t              = Integer;
  miopenRNNBaseLayout_t           = Integer;
  miopenLossReductionMode_t 	  = Integer;
  miopenCTCLossAlgo_t             = Integer;
  miopenRNGType_t                 = Integer;
   // Backend API Types
  miopenBackendDescriptorType_t   = Integer;
  miopenBackendAttributeName_t    = Integer;
  miopenBackendAttributeType_t    = Integer;
  miopenBackendHeurMode_t         = Integer;

  miopenFindResultsOrder_t        = Integer;
  miopenF8RoundingMode_t          = Integer;
  miopenProblemDirection_t        = Integer;
  miopenTensorArgumentId_t        = Integer;

  miopenReduceExtremeOp_t         = Integer;
  miopenReduceCalculationOp_t     = Integer;
  miopenReduceCalculationNanPropagation_t = Integer; // Same underlying values as miopenNanPropagation_t
  miopenRoPEMode_t                = Integer;
  miopenRNNRoPEScalingType_t      = Integer;


  miopenAllocatorFunction = function(context: Pointer; sizeBytes: size_t): Pointer; cdecl;
  miopenDeallocatorFunction = procedure(context: Pointer; memory: Pointer); cdecl;
  
  PmiopenTensorDescriptor_t = ^miopenTensorDescriptor_t;
  PmiopenConvFwdAlgorithm_t = ^miopenConvFwdAlgorithm_t;
  PmiopenSolution_t = ^miopenSolution_t;

  PmiopenTensorArgument_t = ^miopenTensorArgument_t;
  miopenTensorArgument_t = record
    id: miopenTensorArgumentId_t;
    descriptor: ^miopenTensorDescriptor_t;
    buffer: Pointer;
  end;  

{$PACKRECORDS C}
  PmiopenConvAlgoPerf_t = ^miopenConvAlgoPerf_t;
  miopenConvAlgoPerf_t = record
    algo: Integer;             // fwd_algo (offset 0)
    bwd_weights_algo: Integer; // bwd_weights_algo (offset 4)
    bwd_data_algo: Integer;    // bwd_data_algo (offset 8)
    time: Single;              // offset 12
    memory: size_t;            // offset 16
  end;
{$PACKRECORDS DEFAULT}

  PmiopenConvSolution_t = ^miopenConvSolution_t;
  miopenConvSolution_t = record
    time: Single;
    workspace_size: size_t;
    solution_id: UInt64;
    algorithm: miopenConvAlgorithm_t;
  end;


{ miopenStatus_t }
const
  miopenStatusSuccess              = 0;
  miopenStatusNotInitialized       = 1;
  miopenStatusInvalidValue         = 2;
  miopenStatusBadParm              = 3;
  miopenStatusAllocFailed          = 4;
  miopenStatusInternalError        = 5;
  miopenStatusNotImplemented       = 6;
  miopenStatusUnknownError         = 7;
  miopenStatusUnsupportedOp        = 8;
  miopenStatusGpuOperationsSkipped = 9;
  miopenStatusVersionMismatch      = 10;

{ miopenDataType_t }
const
  miopenHalf     = 0;
  miopenFloat    = 1;
  miopenInt32    = 2;
  miopenInt8     = 3;
  miopenBFloat16 = 5;
  miopenDouble   = 6;
  miopenFloat8   = 7;
  miopenBFloat8  = 8;
  miopenInt64    = 9;

{ miopenConvolutionMode_t }
const
  miopenConvolution = 0;
  miopenTranspose   = 1;
  miopenGroupConv   = 2;
  miopenDepthwise   = 3;

{ miopenConvolutionAttrib_t }
const
  miopenConvolutionAttribFP16Math = 0;
  miopenConvolutionAttribInt8Math = 1;

{ miopenActivationMode_t }
const
  miopenActivationPASTHRU     = 0; // Also known as Identity
  miopenActivationLOGISTIC    = 1; // This is the constant for Sigmoid
  miopenActivationTANH        = 2;
  miopenActivationRELU        = 3;
  miopenActivationSOFTRELU    = 4; // Also known as SmoothReLU
  miopenActivationABS         = 5;
  miopenActivationPOWER       = 6;
  miopenActivationCLIPPEDRELU = 7;
  miopenActivationLEAKYRELU   = 8;
  miopenActivationELU         = 9;
  miopenActivationGELU        = 10;

{ miopenTensorLayout_t }
const
  miopenTensorNCHW   = 0;
  miopenTensorNHWC   = 1;
  miopenTensorCHWN   = 2;
  miopenTensorNCHWc4 = 3;
  miopenTensorNCHWc8 = 4;
  miopenTensorCHWNc4 = 5;
  miopenTensorCHWNc8 = 6;
  miopenTensorNCDHW  = 7;
  miopenTensorNDHWC  = 8;

{ miopenPaddingMode_t }
const
  miopenPaddingDefault = 0;
  miopenPaddingSame    = 1;
  miopenPaddingValid   = 2;

{ miopenIndexType_t }
const
  miopenIndexUint8  = 0;
  miopenIndexUint16 = 1;
  miopenIndexUint32 = 2;
  miopenIndexUint64 = 3;

{ miopenTensorOp_t }
const
  miopenTensorOpAdd = 0;
  miopenTensorOpMul = 1;
  miopenTensorOpMin = 2;
  miopenTensorOpMax = 3;

{ miopenConvolutionAttrib_t }
const
  MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL = 0;
  MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC = 1;
{$IFDEF MIOPEN_BETA_API}
  MIOPEN_CONVOLUTION_ATTRIB_FP8_ROUNDING_MODE = 2;
{$ENDIF}

{ miopenReduceTensorOp_t }
const
  MIOPEN_REDUCE_TENSOR_ADD   = 0;
  MIOPEN_REDUCE_TENSOR_MUL   = 1;
  MIOPEN_REDUCE_TENSOR_MIN   = 2;
  MIOPEN_REDUCE_TENSOR_MAX   = 3;
  MIOPEN_REDUCE_TENSOR_AMAX  = 4;
  MIOPEN_REDUCE_TENSOR_AVG   = 5;
  MIOPEN_REDUCE_TENSOR_NORM1 = 6;
  MIOPEN_REDUCE_TENSOR_NORM2 = 7;

{ miopenNanPropagation_t }
const
  MIOPEN_NOT_PROPAGATE_NAN = 0;
  MIOPEN_PROPAGATE_NAN     = 1;

{ miopenReduceTensorIndices_t }
const
  MIOPEN_REDUCE_TENSOR_NO_INDICES        = 0;
  MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES = 1;

{ miopenIndicesType_t }
const
  MIOPEN_32BIT_INDICES = 0;
  MIOPEN_64BIT_INDICES = 1;
  MIOPEN_16BIT_INDICES = 2;
  MIOPEN_8BIT_INDICES  = 3;

{$IFDEF MIOPEN_BETA_API}
{ miopenNormMode_t }
const
  MIOPEN_ELEMENTWISE_AFFINE             = 0;
  MIOPEN_WEIGHT_BIAS                    = 1;
  MIOPEN_ELEMENTWISE_AFFINE_FUSED_ADD   = 2;
  MIOPEN_WEIGHT_BIAS_FUSED_ADD          = 3;
  MIOPEN_ELEMENTWISE_AFFINE_T5          = 4;
  MIOPEN_WEIGHT_BIAS_T5                 = 5;
{$ENDIF}

{ miopenRNGType_t }
const
  MIOPEN_RNG_PSEUDO_XORWOW = 0;

{ miopenProblemDirection_t }
const
  miopenProblemDirectionForward         = 0;
  miopenProblemDirectionBackward        = 1;
  miopenProblemDirectionBackwardWeights = 2;
{$IFDEF MIOPEN_BETA_API}
  miopenProblemDirectionInference       = 4;
{$ENDIF}

{ miopenMhaMask_t }
const
  miopenMhaMaskNone   = 0;
  miopenMhaMaskCausal = 1;

{ miopenPoolingWorkspaceIndexMode_t }
const
  miopenPoolingWorkspaceIndexMode_REUSE = 0; // The default mode
  miopenPoolingWorkspaceIndexMode_MASK  = 1;

{ miopenFindResultsOrder_t }
const
  miopenFindResultsOrderByTime   = 0; // Order results by time
  miopenFindResultsOrderByMemory = 1; // Order results by memory requirements

{ miopenConvFwdAlgorithm_t }
const
  miopenConvolutionFwdAlgoGEMM         = 0; // GEMM variant
  miopenConvolutionFwdAlgoDirect       = 1; // Direct convolutions
  miopenConvolutionFwdAlgoFFT          = 2; // Fast Fourier Transform indirect convolutions
  miopenConvolutionFwdAlgoWinograd     = 3; // Winograd indirect convolutions
  // Note: Value 4 is skipped in the C enum
  miopenConvolutionFwdAlgoImplicitGEMM = 5; // Implicit GEMM convolutions

{ miopenConvBwdDataAlgorithm_t }
const
  miopenConvolutionBwdDataAlgoGEMM         = 0; // GEMM variant
  miopenConvolutionBwdDataAlgoDirect       = 1; // Direct convolutions
  miopenConvolutionBwdDataAlgoFFT          = 2; // Fast Fourier Transform indirect convolutions
  miopenConvolutionBwdDataAlgoWinograd     = 3; // Winograd indirect convolutions
  miopenTransposeBwdDataAlgoGEMM           = 4; // Deprecated Transpose GEMM variant legacy, ToBe Removed
  miopenConvolutionBwdDataAlgoImplicitGEMM = 5; // Implicit GEMM convolutions

{ miopenConvBwdWeightsAlgorithm_t }
const
  miopenConvolutionBwdWeightsAlgoGEMM         = 0; // GEMM variant
  miopenConvolutionBwdWeightsAlgoDirect       = 1; // Direct convolution algorithm
  // Note: Value 2 is skipped in the C enum
  miopenConvolutionBwdWeightsAlgoWinograd     = 3; // Winograd convolutions
  // Note: Value 4 is skipped in the C enum
  miopenConvolutionBwdWeightsAlgoImplicitGEMM = 5; // Implicit GEMM convolutions

// Optional: Constants for the generic miopenConvAlgorithm_t type
// (These are likely the base values used in miopenConvAlgoPerf_t when the specific direction isn't known)
{ miopenConvAlgorithm_t }
const
  miopenConvolutionAlgoGEMM         = 0; // GEMM variant
  miopenConvolutionAlgoDirect       = 1; // Direct convolutions
  miopenConvolutionAlgoFFT          = 2; // Fast Fourier Transform indirect convolutions
  miopenConvolutionAlgoWinograd     = 3; // Winograd indirect convolutions
  // Note: Value 4 is skipped in the C enum
  miopenConvolutionAlgoImplicitGEMM = 5; // Implicit GEMM convolutions

{ miopenLRNMode_t }
const
  miopenLRNWithinChannel = 0; { Normalize within channel }
  miopenLRNCrossChannel  = 1; { Normalize cross channel }
  
{ miopenRNNMode_t }
const
  miopenRNNRELU  = 0; // RNN with ReLU activation
  miopenRNNTANH  = 1; // RNN with tanh activation
  miopenLSTM     = 2; // LSTM
  miopenGRU      = 3; // GRU

{ miopenRNNAlgo_t }
const
  miopenRNNdefault     = 0; // Use dedicated gate-operation kernel for LSTM and fundamental algorithm for vanilla RNN & GRU
  miopenRNNfundamental = 1; // Function by basic tensor operations, supported for vanilla RNN, LSTM, GRU

{ miopenRNNInputMode_t }
const
  miopenRNNlinear = 0; // Matrix multiplication at the input of the first layer
  miopenRNNskip   = 1; // No operation is performed at the input of the first layer.

{ miopenRNNDirectionMode_t }
const
  miopenRNNunidirection = 0; // Forward in time only.
  miopenRNNbidirection  = 1; // Forward and backwards in time.

{ miopenRNNBiasMode_t }
const
  miopenRNNNoBias  = 0; // No Biases will be applied to GEMM operations
  miopenRNNwithBias = 1; // Biases will be applied to GEMM operations

{ miopenRNNPaddingMode_t }
const
  miopenRNNIONotPadded = 0; // Not padded data at RNN input/output
  miopenRNNIOWithPadding = 1; // Padded data at RNN input/output

{ miopenRNNGEMMalgoMode_t }
const
  miopenRNNAlgoGEMM = 0;

{ miopenRNNFWDMode_t }
const
  miopenRNNTraining  = 0;
  miopenRNNInference = 1;

{ miopenRNNBaseLayout_t }
const
  miopenRNNDataUnknownLayout     = 0;
  miopenRNNDataSeqMajorNotPadded = 1;
  miopenRNNDataSeqMajorPadded    = 2;
  miopenRNNDataBatchMajorPadded  = 3;
    
{ miopenLossReductionMode_t }
const
  MIOPEN_LOSS_REDUCTION_NONE  = 0; { output tensor elements are not reduced }
  MIOPEN_LOSS_REDUCTION_SUM   = 1; { output tensor elements are summed up }
  MIOPEN_LOSS_REDUCTION_MEAN  = 2; { output tensor elements are summed up and divided with total number of elements to get mean value }

{ miopenCTCLossAlgo_t }
const
  MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC = 0;

{ miopenPoolingMode_t }
const
  miopenPoolingMax              = 0;
  miopenPoolingAverage          = 1;
  miopenPoolingAverageInclusive = 2;

{ miopenFusionDirection_t }  
const
  miopenVerticalFusion   = 0;  
  miopenHorizontalFusion = 1;    

{ miopenSoftmaxAlgorithm_t }
const
  MIOPEN_SOFTMAX_FAST     = 0;
  MIOPEN_SOFTMAX_ACCURATE = 1;
  MIOPEN_SOFTMAX_LOG      = 2;

{ miopenSoftmaxMode_t }
const
  MIOPEN_SOFTMAX_MODE_INSTANCE = 0;
  MIOPEN_SOFTMAX_MODE_CHANNEL  = 1;

{ miopenBatchNormMode_t }
const
  miopenBNPerActivation = 0;
  miopenBNSpatial       = 1;
  
{ miopenTensorArgumentId_t }
const
  miopenTensorArgumentIdInvalid = 0;
  miopenTensorConvolutionX      = 1;
  miopenTensorConvolutionW      = 2;
  miopenTensorConvolutionY      = 3;

  miopenTensorMhaK                  = 4;
  miopenTensorMhaQ                  = 5;
  miopenTensorMhaV                  = 6;
  miopenTensorMhaDescaleK           = 7;
  miopenTensorMhaDescaleQ           = 8;
  miopenTensorMhaDescaleV           = 9;
  miopenTensorMhaDescaleS           = 10;
  miopenTensorMhaScaleS             = 11;
  miopenTensorMhaScaleO             = 12;
  miopenTensorMhaDropoutProbability = 13;
  miopenTensorMhaDropoutSeed        = 14;
  miopenTensorMhaDropoutOffset      = 15;
  miopenTensorMhaO                  = 16;
  miopenTensorMhaAmaxO              = 17;
  miopenTensorMhaAmaxS              = 18;
  miopenTensorMhaM                  = 19;
  miopenTensorMhaZInv               = 20;
  miopenTensorMhaDO                 = 21;
  miopenTensorMhaDescaleO           = 22;
  miopenTensorMhaDescaleDO          = 23;
  miopenTensorMhaDescaleDS          = 24;
  miopenTensorMhaScaleDS            = 25;
  miopenTensorMhaScaleDQ            = 26;
  miopenTensorMhaScaleDK            = 27;
  miopenTensorMhaScaleDV            = 28;
  miopenTensorMhaDQ                 = 29;
  miopenTensorMhaDK                 = 30;
  miopenTensorMhaDV                 = 31;
  miopenTensorMhaAmaxDQ             = 32;
  miopenTensorMhaAmaxDK             = 33;
  miopenTensorMhaAmaxDV             = 34;
  miopenTensorMhaAmaxDS             = 35;
  miopenTensorMhaBias               = 36;

{$IFDEF MIOPEN_BETA_API}
  miopenTensorActivationX                = 37;
  miopenTensorActivationY                = 38;
  miopenTensorActivationDX               = 39;
  miopenTensorActivationDY               = 40;
  miopenTensorBiasX                      = 41;
  miopenTensorBiasY                      = 42;
  miopenTensorBias                       = 43;
  miopenTensorSoftmaxX                   = 44;
  miopenTensorSoftmaxY                   = 45;
  miopenTensorSoftmaxDX                  = 46;
  miopenTensorSoftmaxDY                  = 47;
  miopenTensorBatchnormX                 = 48;
  miopenTensorBatchnormY                 = 49;
  miopenTensorBatchnormRunningMean       = 50;
  miopenTensorBatchnormRunningVariance   = 51;
  miopenTensorBatchnormSavedMean         = 52;
  miopenTensorBatchnormSavedVariance     = 53;
  miopenTensorBatchnormScale             = 54;
  miopenTensorBatchnormScaleDiff         = 55;
  miopenTensorBatchnormEstimatedMean     = 56;
  miopenTensorBatchnormEstimatedVariance = 57;
  miopenTensorBatchnormBias              = 58;
  miopenTensorBatchnormBiasDiff          = 59;
  miopenTensorBatchnormDX                = 60;
  miopenTensorBatchnormDY                = 61;
{$ENDIF}

  // Bitmask for scalar arguments
  miopenTensorArgumentIsScalar = 1 shl 31;

  miopenTensorMhaMask = miopenTensorArgumentIsScalar or 1;
{$IFDEF MIOPEN_BETA_API}
  miopenScalarBatchnormExpAvgFactor = miopenTensorArgumentIsScalar or 2;
  miopenScalarBatchnormEpsilon      = miopenTensorArgumentIsScalar or 3;
{$ENDIF}
  
  
{ miopenBackendDescriptorType_t }
const
  MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR        = 0;
  MIOPEN_BACKEND_ENGINE_DESCRIPTOR             = 1;
  MIOPEN_BACKEND_ENGINECFG_DESCRIPTOR          = 2;
  MIOPEN_BACKEND_ENGINEHEUR_DESCRIPTOR         = 3;
  MIOPEN_BACKEND_EXECUTION_PLAN_DESCRIPTOR     = 4;
  MIOPEN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR  = 5;
  MIOPEN_BACKEND_KNOB_CHOICE_DESCRIPTOR        = 6;
  MIOPEN_BACKEND_KNOB_INFO_DESCRIPTOR          = 7;
  MIOPEN_BACKEND_LAYOUT_INFO_DESCRIPTOR        = 8;
  MIOPEN_BACKEND_MATMUL_DESCRIPTOR             = 9;
  MIOPEN_BACKEND_OPERATION_CONCAT_DESCRIPTOR   = 10;
  MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR = 11;
  MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR = 12;
  MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR = 13;
  MIOPEN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR = 14;
  MIOPEN_BACKEND_OPERATION_MATMUL_DESCRIPTOR   = 15;
  MIOPEN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR = 16;
  MIOPEN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR = 17;
  MIOPEN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR = 18;
  MIOPEN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR = 19;
  MIOPEN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR = 20;
  MIOPEN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR = 21;
  MIOPEN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR  = 22;
  MIOPEN_BACKEND_OPERATION_RNG_DESCRIPTOR      = 23;
  MIOPEN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR   = 24;
  MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR     = 25;
  MIOPEN_BACKEND_POINTWISE_DESCRIPTOR          = 26;
  MIOPEN_BACKEND_REDUCTION_DESCRIPTOR          = 27;
  MIOPEN_BACKEND_RESAMPLE_DESCRIPTOR           = 28;
  MIOPEN_BACKEND_RNG_DESCRIPTOR                = 29;
  MIOPEN_BACKEND_TENSOR_DESCRIPTOR             = 30;
  MIOPEN_BACKEND_VARIANT_PACK_DESCRIPTOR       = 31;

{$IFDEF MIOPEN_BETA_API}

{ miopenBackendAttributeName_t }
const
  MIOPEN_ATTR_POINTWISE_MODE                  = 0;
  MIOPEN_ATTR_POINTWISE_MATH_PREC             = 1;
  MIOPEN_ATTR_POINTWISE_NAN_PROPAGATION       = 2;
  MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP       = 3;
  MIOPEN_ATTR_POINTWISE_RELU_UPPER_CLIP       = 4;
  MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE = 5;
  MIOPEN_ATTR_POINTWISE_ELU_ALPHA             = 6;
  MIOPEN_ATTR_POINTWISE_SOFTPLUS_BETA         = 7;
  MIOPEN_ATTR_POINTWISE_SWISH_BETA            = 8;
  MIOPEN_ATTR_POINTWISE_AXIS                  = 9;

  MIOPEN_ATTR_CONVOLUTION_COMP_TYPE      = 100;
  MIOPEN_ATTR_CONVOLUTION_CONV_MODE      = 101;
  MIOPEN_ATTR_CONVOLUTION_DILATIONS      = 102;
  MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES = 103;
  MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS  = 104;
  MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS   = 105;
  MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS   = 106;

  MIOPEN_ATTR_ENGINEHEUR_MODE            = 200;
  MIOPEN_ATTR_ENGINEHEUR_OPERATION_GRAPH = 201;
  MIOPEN_ATTR_ENGINEHEUR_RESULTS         = 202;
  MIOPEN_ATTR_ENGINEHEUR_SM_COUNT_TARGET = 203;

  MIOPEN_ATTR_ENGINECFG_ENGINE            = 300;
  MIOPEN_ATTR_ENGINECFG_INTERMEDIATE_INFO = 301;
  MIOPEN_ATTR_ENGINECFG_KNOB_CHOICES      = 302;

  MIOPEN_ATTR_EXECUTION_PLAN_HANDLE                     = 400;
  MIOPEN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG              = 401;
  MIOPEN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE             = 402;
  MIOPEN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS = 403;
  MIOPEN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS = 404;
  MIOPEN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION        = 405;

  MIOPEN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID            = 500;
  MIOPEN_ATTR_INTERMEDIATE_INFO_SIZE                 = 501;
  MIOPEN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS  = 502;
  MIOPEN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES = 503;

  MIOPEN_ATTR_KNOB_CHOICE_KNOB_TYPE  = 600;
  MIOPEN_ATTR_KNOB_CHOICE_KNOB_VALUE = 601;

  MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA        = 700;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA         = 701;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC    = 702;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W            = 703;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X            = 704;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y            = 705;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA       = 706;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA        = 707;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC   = 708;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W           = 709;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX          = 710;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY          = 711;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA     = 712;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA      = 713;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC = 714;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW        = 715;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X         = 716;
  MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY        = 717;
  MIOPEN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR          = 750;
  MIOPEN_ATTR_OPERATION_POINTWISE_XDESC                  = 751;
  MIOPEN_ATTR_OPERATION_POINTWISE_BDESC                  = 752;
  MIOPEN_ATTR_OPERATION_POINTWISE_YDESC                  = 753;
  MIOPEN_ATTR_OPERATION_POINTWISE_ALPHA1                 = 754;
  MIOPEN_ATTR_OPERATION_POINTWISE_ALPHA2                 = 755;
  MIOPEN_ATTR_OPERATION_POINTWISE_DXDESC                 = 756;
  MIOPEN_ATTR_OPERATION_POINTWISE_DYDESC                 = 757;
  MIOPEN_ATTR_OPERATION_POINTWISE_TDESC                  = 758;

  MIOPEN_ATTR_OPERATION_GENSTATS_MODE      = 770;
  MIOPEN_ATTR_OPERATION_GENSTATS_MATH_PREC = 771;
  MIOPEN_ATTR_OPERATION_GENSTATS_XDESC     = 772;
  MIOPEN_ATTR_OPERATION_GENSTATS_SUMDESC   = 773;
  MIOPEN_ATTR_OPERATION_GENSTATS_SQSUMDESC = 774;

  MIOPEN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE                = 780;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC                 = 781;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC                = 782;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC             = 783;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC                = 784;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC                 = 785;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC    = 786;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC     = 787;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC = 788;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC  = 789;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC           = 790;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC        = 791;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC             = 792;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC              = 793;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC          = 794;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC              = 795;
  MIOPEN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC   = 796;

  MIOPEN_ATTR_OPERATIONGRAPH_HANDLE              = 800;
  MIOPEN_ATTR_OPERATIONGRAPH_OPS                 = 801;
  MIOPEN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = 802;

  MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT       = 900;
  MIOPEN_ATTR_TENSOR_DATA_TYPE            = 901;
  MIOPEN_ATTR_TENSOR_DIMENSIONS           = 902;
  MIOPEN_ATTR_TENSOR_STRIDES              = 903;
  MIOPEN_ATTR_TENSOR_VECTOR_COUNT         = 904;
  MIOPEN_ATTR_TENSOR_VECTORIZED_DIMENSION = 905;
  MIOPEN_ATTR_TENSOR_UNIQUE_ID            = 906;
  MIOPEN_ATTR_TENSOR_IS_VIRTUAL           = 907;
  MIOPEN_ATTR_TENSOR_IS_BY_VALUE          = 908;
  MIOPEN_ATTR_TENSOR_REORDERING_MODE      = 909;
  MIOPEN_ATTR_TENSOR_RAGGED_OFFSET_DESC   = 910;

  MIOPEN_ATTR_VARIANT_PACK_UNIQUE_IDS    = 1000;
  MIOPEN_ATTR_VARIANT_PACK_DATA_POINTERS = 1001;
  MIOPEN_ATTR_VARIANT_PACK_INTERMEDIATES = 1002;
  MIOPEN_ATTR_VARIANT_PACK_WORKSPACE     = 1003;

  MIOPEN_ATTR_LAYOUT_INFO_TENSOR_UID = 1100;
  MIOPEN_ATTR_LAYOUT_INFO_TYPES      = 1101;

  MIOPEN_ATTR_KNOB_INFO_TYPE          = 1200;
  MIOPEN_ATTR_KNOB_INFO_MAXIMUM_VALUE = 1201;
  MIOPEN_ATTR_KNOB_INFO_MINIMUM_VALUE = 1202;
  MIOPEN_ATTR_KNOB_INFO_STRIDE        = 1203;

  MIOPEN_ATTR_ENGINE_OPERATION_GRAPH = 1300;
  MIOPEN_ATTR_ENGINE_GLOBAL_INDEX    = 1301;
  MIOPEN_ATTR_ENGINE_KNOB_INFO       = 1302;
  MIOPEN_ATTR_ENGINE_NUMERICAL_NOTE  = 1303;
  MIOPEN_ATTR_ENGINE_LAYOUT_INFO     = 1304;
  MIOPEN_ATTR_ENGINE_BEHAVIOR_NOTE   = 1305;
  MIOPEN_ATTR_ENGINE_SM_COUNT_TARGET = 1306;

  MIOPEN_ATTR_MATMUL_COMP_TYPE     = 1500;
  MIOPEN_ATTR_MATMUL_PADDING_VALUE = 1501;

  MIOPEN_ATTR_OPERATION_MATMUL_ADESC                           = 1520;
  MIOPEN_ATTR_OPERATION_MATMUL_BDESC                           = 1521;
  MIOPEN_ATTR_OPERATION_MATMUL_CDESC                           = 1522;
  MIOPEN_ATTR_OPERATION_MATMUL_DESC                            = 1523;
  MIOPEN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT = 1524;
  MIOPEN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC            = 1525;
  MIOPEN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC            = 1526;
  MIOPEN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC            = 1527;

  MIOPEN_ATTR_REDUCTION_OPERATOR  = 1600;
  MIOPEN_ATTR_REDUCTION_COMP_TYPE = 1601;

  MIOPEN_ATTR_OPERATION_REDUCTION_XDESC = 1610;
  MIOPEN_ATTR_OPERATION_REDUCTION_YDESC = 1611;
  MIOPEN_ATTR_OPERATION_REDUCTION_DESC  = 1612;

  MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC        = 1620;
  MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC        = 1621;
  MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC      = 1622;
  MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC    = 1623;
  MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC           = 1624;
  MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC          = 1625;
  MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC   = 1626;
  MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC    = 1627;
  MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC = 1628;
  MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC  = 1629;
  MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS          = 1630;

  MIOPEN_ATTR_RESAMPLE_MODE            = 1700;
  MIOPEN_ATTR_RESAMPLE_COMP_TYPE       = 1701;
  MIOPEN_ATTR_RESAMPLE_SPATIAL_DIMS    = 1702;
  MIOPEN_ATTR_RESAMPLE_POST_PADDINGS   = 1703;
  MIOPEN_ATTR_RESAMPLE_PRE_PADDINGS    = 1704;
  MIOPEN_ATTR_RESAMPLE_STRIDES         = 1705;
  MIOPEN_ATTR_RESAMPLE_WINDOW_DIMS     = 1706;
  MIOPEN_ATTR_RESAMPLE_NAN_PROPAGATION = 1707;
  MIOPEN_ATTR_RESAMPLE_PADDING_MODE    = 1708;

  MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_XDESC   = 1710;
  MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_YDESC   = 1711;
  MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC = 1712;
  MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA   = 1713;
  MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_BETA    = 1714;
  MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_DESC    = 1716;

  MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC  = 1720;
  MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC  = 1721;
  MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC = 1722;
  MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA   = 1723;
  MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_BETA    = 1724;
  MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_DESC    = 1725;
  MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_XDESC   = 1726;
  MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_YDESC   = 1727;

  MIOPEN_ATTR_OPERATION_CONCAT_AXIS          = 1800;
  MIOPEN_ATTR_OPERATION_CONCAT_INPUT_DESCS   = 1801;
  MIOPEN_ATTR_OPERATION_CONCAT_INPLACE_INDEX = 1802;
  MIOPEN_ATTR_OPERATION_CONCAT_OUTPUT_DESC   = 1803;

  MIOPEN_ATTR_OPERATION_SIGNAL_MODE     = 1900;
  MIOPEN_ATTR_OPERATION_SIGNAL_FLAGDESC = 1901;
  MIOPEN_ATTR_OPERATION_SIGNAL_VALUE    = 1902;
  MIOPEN_ATTR_OPERATION_SIGNAL_XDESC    = 1903;
  MIOPEN_ATTR_OPERATION_SIGNAL_YDESC    = 1904;

  MIOPEN_ATTR_OPERATION_NORM_FWD_MODE                     = 2000;
  MIOPEN_ATTR_OPERATION_NORM_FWD_PHASE                    = 2001;
  MIOPEN_ATTR_OPERATION_NORM_FWD_XDESC                    = 2002;
  MIOPEN_ATTR_OPERATION_NORM_FWD_MEAN_DESC                = 2003;
  MIOPEN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC        = 2004;
  MIOPEN_ATTR_OPERATION_NORM_FWD_SCALE_DESC               = 2005;
  MIOPEN_ATTR_OPERATION_NORM_FWD_BIAS_DESC                = 2006;
  MIOPEN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC             = 2007;
  MIOPEN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC      = 2008;
  MIOPEN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC  = 2009;
  MIOPEN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC   = 2010;
  MIOPEN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC = 2011;
  MIOPEN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC  = 2012;
  MIOPEN_ATTR_OPERATION_NORM_FWD_YDESC                    = 2013;
  MIOPEN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS          = 2014;

  MIOPEN_ATTR_OPERATION_NORM_BWD_MODE              = 2100;
  MIOPEN_ATTR_OPERATION_NORM_BWD_XDESC             = 2101;
  MIOPEN_ATTR_OPERATION_NORM_BWD_MEAN_DESC         = 2102;
  MIOPEN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC = 2103;
  MIOPEN_ATTR_OPERATION_NORM_BWD_DYDESC            = 2104;
  MIOPEN_ATTR_OPERATION_NORM_BWD_SCALE_DESC        = 2105;
  MIOPEN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC      = 2106;
  MIOPEN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC       = 2107;
  MIOPEN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC        = 2108;
  MIOPEN_ATTR_OPERATION_NORM_BWD_DXDESC            = 2109;
  MIOPEN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS   = 2110;

  MIOPEN_ATTR_OPERATION_RESHAPE_XDESC = 2200;
  MIOPEN_ATTR_OPERATION_RESHAPE_YDESC = 2201;

  MIOPEN_ATTR_RNG_DISTRIBUTION                   = 2300;
  MIOPEN_ATTR_RNG_NORMAL_DIST_MEAN               = 2301;
  MIOPEN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION = 2302;
  MIOPEN_ATTR_RNG_UNIFORM_DIST_MAXIMUM           = 2303;
  MIOPEN_ATTR_RNG_UNIFORM_DIST_MINIMUM           = 2304;
  MIOPEN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY     = 2305;

  MIOPEN_ATTR_OPERATION_RNG_YDESC       = 2310;
  MIOPEN_ATTR_OPERATION_RNG_SEED        = 2311;
  MIOPEN_ATTR_OPERATION_RNG_DESC        = 2312;
  MIOPEN_ATTR_OPERATION_RNG_OFFSET_DESC = 2313;

{$ENDIF}

{ miopenBackendAttributeType_t }
const
  MIOPEN_TYPE_HANDLE              = 0;  // miopenHandle_t
  MIOPEN_TYPE_DATA_TYPE           = 1;  // miopenDataType_t
  MIOPEN_TYPE_BOOLEAN             = 2;  // bool
  MIOPEN_TYPE_INT64               = 3;  // int64_t
  MIOPEN_TYPE_FLOAT               = 4;  // float
  MIOPEN_TYPE_DOUBLE              = 5;  // double
  MIOPEN_TYPE_VOID_PTR            = 6;  // void *
  MIOPEN_TYPE_CONVOLUTION_MODE    = 7;  // miopenConvolutionMode_t
  MIOPEN_TYPE_HEUR_MODE           = 8;  // miopenBackendHeurMode_t
  MIOPEN_TYPE_KNOB_TYPE           = 9;  // miopenBackendKnobType_t 
  MIOPEN_TYPE_NAN_PROPAGATION     = 10; // miopenNanPropagation_t 
  MIOPEN_TYPE_NUMERICAL_NOTE      = 11; // miopenBackendNumericalNote_t 
  MIOPEN_TYPE_LAYOUT_TYPE         = 12; // miopenBackendLayoutType_t 
  MIOPEN_TYPE_ATTRIB_NAME         = 13; // miopenBackendAttributeName_t
  MIOPEN_TYPE_POINTWISE_MODE      = 14; // miopenPointwiseMode_t 
  MIOPEN_TYPE_BACKEND_DESCRIPTOR  = 15; // miopenBackendDescriptor_t
  MIOPEN_TYPE_GENSTATS_MODE       = 16; // miopenGenStatsMode_t 
  MIOPEN_TYPE_BN_FINALIZE_STATS_MODE = 17; // miopenBnFinalizeStatsMode_t 
  MIOPEN_TYPE_REDUCTION_OPERATOR_TYPE = 18; // miopenReduceTensorOp_t
  MIOPEN_TYPE_BEHAVIOR_NOTE       = 19; // miopenBackendBehaviorNote_t 
  MIOPEN_TYPE_TENSOR_REORDERING_MODE = 20; // miopenBackendTensorReordering_t 
  MIOPEN_TYPE_RESAMPLE_MODE       = 21; // miopenResampleMode_t 
  MIOPEN_TYPE_PADDING_MODE        = 22; // miopenPaddingMode_t
  MIOPEN_TYPE_INT32               = 23; // int32_t
  MIOPEN_TYPE_CHAR                = 24; // char
  MIOPEN_TYPE_SIGNAL_MODE         = 25; // miopenSignalMode_t 
  MIOPEN_TYPE_FRACTION            = 26; // miopenFraction_t 
  MIOPEN_TYPE_NORM_MODE           = 27; // miopenNormMode_t
  MIOPEN_TYPE_NORM_FWD_PHASE      = 28; 
  MIOPEN_TYPE_RNG_DISTRIBUTION    = 29; 
  
{ miopenBackendHeurMode_t }
const
  MIOPEN_HEUR_MODE_INSTANT   = 0;
  MIOPEN_HEUR_MODE_B         = 1;
  MIOPEN_HEUR_MODE_FALLBACK  = 2;
  MIOPEN_HEUR_MODE_A         = 3;
  MIOPEN_HEUR_MODES_COUNT    = 4;
  
{ miopenF8RoundingMode_t }
const
  miopenF8RoundingModeStandard   = 0;
  miopenF8RoundingModeStochastic = 1;

{ miopenReduceExtremeOp_t }
const
  MIOPEN_REDUCE_EXTREME_ARGMIN = 1; {< the operation is getting the minimum index of the reduced elements }
  MIOPEN_REDUCE_EXTREME_ARGMAX = 2; {< the operation is getting the maximum index of the reduced elements }
  MIOPEN_REDUCE_EXTREME_MIN = 3;    {< the operation is getting the minimum value and index of the reduced elements }
  MIOPEN_REDUCE_EXTREME_MAX = 4;    {< the operation is getting the maximum value and index of the reduced elements }

{ miopenReduceCalculationOp_t }
const
  MIOPEN_REDUCE_CALCULATION_PROD = 1; {< the operation is multiplying the values of the reduced elements }
  MIOPEN_REDUCE_CALCULATION_SUM = 2;  {< the operation is adding the values of the reduced elements }

{ miopenReduceCalculationNanPropagation_t }
const
  MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN = 0; {< does not propagate Nan number }
  MIOPEN_REDUCE_CALCULATION_PROPAGATE_NAN = 1;     {< propagate the Nan number by the Reduction operation }

{ miopenRoPEMode_t }
const
  // You need to find the actual integer values for these in your miopen.h
  // Placeholder values used here - REPLACE with actual values from miopen.h
  miopenRoPEModeRotate = 0; // Example value
  miopenRoPEModeStatic = 1; // Example value
  // Add other modes as defined in the header

{ miopenRNNRoPEScalingType_t }
const
  // You need to find the actual integer values for these in your miopen.h
  // Placeholder values used here - REPLACE with actual values from miopen.h
  miopenRNNRoPEScalingTypeNone = 0;      // Example value
  miopenRNNRoPEScalingTypeDynamic = 1;   // Example value
  // Add other scaling types as defined in the header

{ miopenNormMode_t }
const
  // You need to find the actual integer values for these in your miopen.h
  // Placeholder values used here - REPLACE with actual values from miopen.h
  miopenNormModeLayerNorm = 0;      // Example value
  miopenNormModeGroupNorm = 1;      // Example value
  miopenNormModeT5LayerNorm = 2;    // Example value
  miopenNormModeAddLayerNorm = 3;   // Example value
  // Add other norm modes as defined in the header




