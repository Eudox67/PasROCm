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

unit rocblas;

{$mode objfpc}{$H+}
{$linklib rocblas}

interface

uses
  hip_runtime, hip_types;

type
  { Opaque handles }
  rocblas_handle = Pointer;

  { Integer types }
  rocblas_int    = Int32;
  rocblas_int64  = Int64;
  rocblas_stride = Int64;

  { Floating point types }
  rocblas_half = record
    data: UInt16;
  end;
  
  rocblas_bfloat16 = record
    data: UInt16;
  end;
  
  rocblas_float_complex = record
    x, y: Single;
  end;
  
  rocblas_double_complex = record
    x, y: Double;
  end;

  { Pointer types for stronger type checking }
  Procblas_int	       = ^rocblas_int;
  Prochalf             = ^rocblas_half;
  Procbfloat16         = ^rocblas_bfloat16;
  Procfloat_complex    = ^rocblas_float_complex;
  Procdouble_complex   = ^rocblas_double_complex;

  { Pointer-to-pointer types for batched functions }
  PPSingle             = ^PSingle;
  PPDouble             = ^PDouble;
  PPPSingle            = ^PPSingle;
  PPPDouble            = ^PPDouble;
  PProchalf            = ^Prochalf;
  PPProcbfloat16       = ^Procbfloat16;
  PPProcfloat_complex  = ^Procfloat_complex;
  PPProcdouble_complex = ^Procdouble_complex;
  PPPointer            = ^PPointer;
    
  { Enum definitions }
//  rocblas_operation    = (rocblas_operation_none = 111, rocblas_operation_transpose, rocblas_operation_conjugate_transpose);
//  rocblas_fill         = (rocblas_fill_upper = 121, rocblas_fill_lower, rocblas_fill_full);
//  rocblas_diagonal     = (rocblas_diagonal_non_unit = 131, rocblas_diagonal_unit);
//  rocblas_side         = (rocblas_side_left = 141, rocblas_side_right, rocblas_side_both );
//  rocblas_status       = (rocblas_status_success, rocblas_status_invalid_handle, rocblas_status_not_implemented, rocblas_status_invalid_pointer, rocblas_status_invalid_size, rocblas_status_memory_error, rocblas_status_internal_error, rocblas_status_perf_degraded, rocblas_status_size_query_mismatch, rocblas_status_size_increased, rocblas_status_size_unchanged, rocblas_status_invalid_value, rocblas_status_continue, rocblas_status_check_numerics_fail, rocblas_status_excluded_from_build);
//  rocblas_pointer_mode = (rocblas_pointer_mode_host, rocblas_pointer_mode_device);

  rocblas_status       = Int32;
  rocblas_operation    = Int32;
  rocblas_fill         = Int32;
  rocblas_diagonal     = Int32;
  rocblas_side         = Int32;
  rocblas_pointer_mode = Int32;
  rocblas_datatype     = Int32;
  rocblas_gemm_algo    = Int32;
  
  rocblas_layer_mode_flags   = UInt32;
  rocblas_storev             = Int32;
  rocblas_direct             = Int32;
  rocblas_eorder             = Int32;
  rocblas_evect              = Int32;
  rocblas_erange             = Int32;
  rocblas_eform              = Int32;
  rocblas_svect              = Int32; 
  rocblas_workmode           = Int32; 
  rocblas_srange             = Int32;

  { rocblas_srange }
const
  { NOTE: the rocSOLVER enumeration constants below (srange, svect, workmode,
    eform, eorder, evect, erange) MUST match rocsolver/rocsolver-extra-types.h
    exactly. They are passed straight through to the C library, which validates
    them against its own enum ranges and returns rocblas_status_invalid_value
    (11) for anything else. }
  rocblas_srange_all   : rocblas_srange = 261;
  rocblas_srange_value : rocblas_srange = 262;
  rocblas_srange_index : rocblas_srange = 263;
    
  { rocblas_svect } 
const
  rocblas_svect_all       : rocblas_svect = 191;
  rocblas_svect_singular  : rocblas_svect = 192;
  rocblas_svect_overwrite : rocblas_svect = 193;
  rocblas_svect_none      : rocblas_svect = 194;

  { rocblas_gemm_algo }
const
  rocblas_gemm_algo_standard                    = 0;
  rocblas_gemm_algo_solution_index              = 1;
  rocblas_gemm_algo_algorithm                   = 2;
  rocblas_gemm_algo_solution_index_deprecated   = 3;  

  { rocblas_workmode } 
const
  rocblas_outofplace : rocblas_workmode = 201;
  rocblas_inplace    : rocblas_workmode = 202;  

  { rocblas_eform }
const
  rocblas_eform_ax      : rocblas_eform = 221; //  A*x = lambda*B*x
  rocblas_eform_abx     : rocblas_eform = 222; // A*B*x = lambda*x
  rocblas_eform_bax     : rocblas_eform = 223; // B*A*x = lambda*x

  { rocblas_layer_mode_flags (Bitmask) }
const
  rocblas_layer_mode_none         : rocblas_layer_mode_flags = 0;
  rocblas_layer_mode_log_trace    : rocblas_layer_mode_flags = 1;
  rocblas_layer_mode_log_bench    : rocblas_layer_mode_flags = 2;
  rocblas_layer_mode_log_profile  : rocblas_layer_mode_flags = 4;

  { rocblas_workmode }
const
  rocblas_workmode_device = 0;
  rocblas_workmode_host   = 1;


{ rocblas_storev }
const
  rocblas_storev_column_wise : rocblas_storev = 181;
  rocblas_storev_row_wise    : rocblas_storev = 182;

{ rocblas_direct }
const
  rocblas_forward_direction  : rocblas_direct = 171;
  rocblas_backward_direction : rocblas_direct = 172;
//  {$ENDIF}
//{$ENDIF}

  { rocblas_eorder }
const
  rocblas_eorder_entire : rocblas_eorder = 242;
  rocblas_eorder_blocks : rocblas_eorder = 241;

  { rocblas_evect }
const
  rocblas_evect_none        : rocblas_evect = 213;
  rocblas_evect_original    : rocblas_evect = 211;
  rocblas_evect_tridiagonal : rocblas_evect = 212;

  { rocblas_erange }
const
  rocblas_erange_all   : rocblas_erange = 231;
  rocblas_erange_value : rocblas_erange = 232;
  rocblas_erange_index : rocblas_erange = 233;

  { rocblas_status }
const
  rocblas_status_success                = 0;
  rocblas_status_invalid_handle         = 1;
  rocblas_status_not_implemented        = 2;
  rocblas_status_invalid_pointer        = 3;
  rocblas_status_invalid_size           = 4;
  rocblas_status_memory_error           = 5;
  rocblas_status_internal_error         = 6;
  rocblas_status_perf_degraded          = 7;
  rocblas_status_size_query_mismatch    = 8;
  rocblas_status_size_increased         = 9;
  rocblas_status_size_unchanged         = 10;
  rocblas_status_invalid_value          = 11;
  rocblas_status_continue               = 12;
  rocblas_status_check_numerics_fail    = 13;
  rocblas_status_excluded_from_build    = 14;

  { rocblas_operation }
const
  rocblas_operation_none                = 111;
  rocblas_operation_transpose           = 112;
  rocblas_operation_conjugate_transpose = 113;

  { rocblas_fill }
const
  rocblas_fill_upper = 121;
  rocblas_fill_lower = 122;
  rocblas_fill_full  = 123;

  { rocblas_diagonal }
const
  rocblas_diagonal_non_unit = 131;
  rocblas_diagonal_unit     = 132;

  { rocblas_side }
const
  rocblas_side_left  = 141;
  rocblas_side_right = 142;
  rocblas_side_both  = 143;

  { rocblas_pointer_mode }
const
     rocblas_pointer_mode_host   = 0;
  rocblas_pointer_mode_device = 1;

  { rocblas_datatype for gemm_ex (FP16/BF16 storage, FP32 accumulate) }
const
  rocblas_datatype_f16_r  = 150;
  rocblas_datatype_f32_r  = 151;
  rocblas_datatype_bf16_r = 168;  { rocblas-types.h: 168 in ROCm 7.0.2 and 10.1; was 160=i8_r }

{ ============================================================================================ }
{   Handle management functions                                                                }
{ ============================================================================================ }

function rocblas_create_handle(var handle: rocblas_handle): rocblas_status; cdecl; external;
function rocblas_destroy_handle(handle: rocblas_handle): rocblas_status; cdecl; external;
function rocblas_set_stream(handle: rocblas_handle; stream: Pointer): rocblas_status; cdecl; external;
function rocblas_get_stream(handle: rocblas_handle; var stream: Pointer): rocblas_status; cdecl; external;
function rocblas_set_pointer_mode(handle: rocblas_handle; mode: rocblas_pointer_mode): rocblas_status; cdecl; external;
function rocblas_get_pointer_mode(handle: rocblas_handle; var mode: rocblas_pointer_mode): rocblas_status; cdecl; external;

{ ============================================================================================ }
{  Device Memory Allocation Functions                                                          }
{ ============================================================================================ }

function rocblas_set_workspace(handle: rocblas_handle; workspace: Pointer; size_in_bytes: size_t): rocblas_status; cdecl; external;
function rocblas_start_device_memory_size_query(handle: rocblas_handle): rocblas_status; cdecl; external;
function rocblas_stop_device_memory_size_query(handle: rocblas_handle; var size_in_bytes: size_t): rocblas_status; cdecl; external;
function rocblas_is_managing_device_memory(handle: rocblas_handle; var is_managing: Boolean): rocblas_status; cdecl; external;

{ ============================================================================================ }
{   Level 1 BLAS functions                                                                     }
{ ============================================================================================ }

{ axpy: y := alpha * x + y }
function rocblas_haxpy(handle: rocblas_handle; n: rocblas_int; const alpha, x: Prochalf; incx: rocblas_int; y: Prochalf; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_saxpy(handle: rocblas_handle; n: rocblas_int; const alpha, x: PSingle; incx: rocblas_int; y: PSingle; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_daxpy(handle: rocblas_handle; n: rocblas_int; const alpha, x: PDouble; incx: rocblas_int; y: PDouble; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_caxpy(handle: rocblas_handle; n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int; y: Procfloat_complex; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zaxpy(handle: rocblas_handle; n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int; y: Procdouble_complex; incy: rocblas_int): rocblas_status; cdecl; external;

{ axpy_batched }
function rocblas_saxpy_batched(handle: rocblas_handle; n: rocblas_int; const alpha: PSingle; x: PPSingle; incx: rocblas_int; y: PPSingle; incy: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_daxpy_batched(handle: rocblas_handle; n: rocblas_int; const alpha: PDouble; x: PPDouble; incx: rocblas_int; y: PPDouble; incy: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_caxpy_batched(handle: rocblas_handle; n: rocblas_int; const alpha: Procfloat_complex; x: PPProcfloat_complex; incx: rocblas_int; y: PPProcfloat_complex; incy: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zaxpy_batched(handle: rocblas_handle; n: rocblas_int; const alpha: Procdouble_complex; x: PPProcdouble_complex; incx: rocblas_int; y: PPProcdouble_complex; incy: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ axpy_strided_batched }
function rocblas_saxpy_strided_batched(handle: rocblas_handle; n: rocblas_int; const alpha: PSingle; const x: PSingle; incx: rocblas_int; stride_x: rocblas_stride; y: PSingle; incy: rocblas_int; stride_y: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_daxpy_strided_batched(handle: rocblas_handle; n: rocblas_int; const alpha: PDouble; const x: PDouble; incx: rocblas_int; stride_x: rocblas_stride; y: PDouble; incy: rocblas_int; stride_y: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_caxpy_strided_batched(handle: rocblas_handle; n: rocblas_int; const alpha: Procfloat_complex; const x: Procfloat_complex; incx: rocblas_int; stride_x: rocblas_stride; y: Procfloat_complex; incy: rocblas_int; stride_y: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zaxpy_strided_batched(handle: rocblas_handle; n: rocblas_int; const alpha: Procdouble_complex; const x: Procdouble_complex; incx: rocblas_int; stride_x: rocblas_stride; y: Procdouble_complex; incy: rocblas_int; stride_y: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ axpy 64-bit }
function rocblas_haxpy_64(handle: rocblas_handle; n: Int64; const alpha, x: Prochalf; incx: Int64; y: Prochalf; incy: Int64): rocblas_status; cdecl; external;
function rocblas_saxpy_64(handle: rocblas_handle; n: Int64; const alpha, x: PSingle; incx: Int64; y: PSingle; incy: Int64): rocblas_status; cdecl; external;
function rocblas_daxpy_64(handle: rocblas_handle; n: Int64; const alpha, x: PDouble; incx: Int64; y: PDouble; incy: Int64): rocblas_status; cdecl; external;
function rocblas_caxpy_64(handle: rocblas_handle; n: Int64; const alpha, x: Procfloat_complex; incx: Int64; y: Procfloat_complex; incy: Int64): rocblas_status; cdecl; external;
function rocblas_zaxpy_64(handle: rocblas_handle; n: Int64; const alpha, x: Procdouble_complex; incx: Int64; y: Procdouble_complex; incy: Int64): rocblas_status; cdecl; external;

{ copy: y := x }
function rocblas_scopy(handle: rocblas_handle; n: rocblas_int; const x: PSingle; incx: rocblas_int; y: PSingle; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dcopy(handle: rocblas_handle; n: rocblas_int; const x: PDouble; incx: rocblas_int; y: PDouble; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ccopy(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; y: Procfloat_complex; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zcopy(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; y: Procdouble_complex; incy: rocblas_int): rocblas_status; cdecl; external;

{ copy_batched }
function rocblas_scopy_batched(handle: rocblas_handle; n: rocblas_int; const x: PPSingle; incx: rocblas_int; y: PPSingle; incy: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dcopy_batched(handle: rocblas_handle; n: rocblas_int; const x: PPDouble; incx: rocblas_int; y: PPDouble; incy: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ccopy_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcfloat_complex; incx: rocblas_int; y: PPProcfloat_complex; incy: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zcopy_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcdouble_complex; incx: rocblas_int; y: PPProcdouble_complex; incy: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ copy_strided_batched }
function rocblas_scopy_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: PSingle; incx: rocblas_int; stridex: rocblas_stride; y: PSingle; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dcopy_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: PDouble; incx: rocblas_int; stridex: rocblas_stride; y: PDouble; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ccopy_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zcopy_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ copy 64-bit }
function rocblas_scopy_64(handle: rocblas_handle; n: Int64; const x: PSingle; incx: Int64; y: PSingle; incy: Int64): rocblas_status; cdecl; external;
function rocblas_dcopy_64(handle: rocblas_handle; n: Int64; const x: PDouble; incx: Int64; y: PDouble; incy: Int64): rocblas_status; cdecl; external;
function rocblas_ccopy_64(handle: rocblas_handle; n: Int64; const x: Procfloat_complex; incx: Int64; y: Procfloat_complex; incy: Int64): rocblas_status; cdecl; external;
function rocblas_zcopy_64(handle: rocblas_handle; n: Int64; const x: Procdouble_complex; incx: Int64; y: Procdouble_complex; incy: Int64): rocblas_status; cdecl; external;

{ dot: result = x * y }
function rocblas_sdot(handle: rocblas_handle; n: rocblas_int; const x: PSingle; incx: rocblas_int; const y: PSingle; incy: rocblas_int; result: PSingle): rocblas_status; cdecl; external;
function rocblas_ddot(handle: rocblas_handle; n: rocblas_int; const x: PDouble; incx: rocblas_int; const y: PDouble; incy: rocblas_int; result: PDouble): rocblas_status; cdecl; external;
function rocblas_hdot(handle: rocblas_handle; n: rocblas_int; const x: Prochalf; incx: rocblas_int; const y: Prochalf; incy: rocblas_int; result: Prochalf): rocblas_status; cdecl; external;
function rocblas_bfdot(handle: rocblas_handle; n: rocblas_int; const x: Procbfloat16; incx: rocblas_int; const y: Procbfloat16; incy: rocblas_int; result: Procbfloat16): rocblas_status; cdecl; external;
function rocblas_cdotu(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; const y: Procfloat_complex; incy: rocblas_int; result: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zdotu(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; const y: Procdouble_complex; incy: rocblas_int; result: Procdouble_complex): rocblas_status; cdecl; external;
function rocblas_cdotc(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; const y: Procfloat_complex; incy: rocblas_int; result: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zdotc(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; const y: Procdouble_complex; incy: rocblas_int; result: Procdouble_complex): rocblas_status; cdecl; external;

{ dot_batched }
function rocblas_sdot_batched(handle: rocblas_handle; n: rocblas_int; const x: PPSingle; incx: rocblas_int; const y: PPSingle; incy: rocblas_int; batch_count: rocblas_int; result: PSingle): rocblas_status; cdecl; external;
function rocblas_ddot_batched(handle: rocblas_handle; n: rocblas_int; const x: PPDouble; incx: rocblas_int; const y: PPDouble; incy: rocblas_int; batch_count: rocblas_int; result: PDouble): rocblas_status; cdecl; external;
function rocblas_cdotu_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcfloat_complex; incx: rocblas_int; const y: PPProcfloat_complex; incy: rocblas_int; batch_count: rocblas_int; result: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zdotu_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcdouble_complex; incx: rocblas_int; const y: PPProcdouble_complex; incy: rocblas_int; batch_count: rocblas_int; result: Procdouble_complex): rocblas_status; cdecl; external;
function rocblas_cdotc_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcfloat_complex; incx: rocblas_int; const y: PPProcfloat_complex; incy: rocblas_int; batch_count: rocblas_int; result: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zdotc_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcdouble_complex; incx: rocblas_int; const y: PPProcdouble_complex; incy: rocblas_int; batch_count: rocblas_int; result: Procdouble_complex): rocblas_status; cdecl; external;

{ dot_strided_batched }
function rocblas_sdot_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: PSingle; incx: rocblas_int; stridex: rocblas_stride; const y: PSingle; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int; result: PSingle): rocblas_status; cdecl; external;
function rocblas_ddot_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: PDouble; incx: rocblas_int; stridex: rocblas_stride; const y: PDouble; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int; result: PDouble): rocblas_status; cdecl; external;
function rocblas_cdotu_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int; result: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zdotu_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int; result: Procdouble_complex): rocblas_status; cdecl; external;
function rocblas_cdotc_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int; result: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zdotc_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int; result: Procdouble_complex): rocblas_status; cdecl; external;

{ dot 64-bit }
function rocblas_sdot_64(handle: rocblas_handle; n: Int64; const x: PSingle; incx: Int64; const y: PSingle; incy: Int64; result: PSingle): rocblas_status; cdecl; external;
function rocblas_ddot_64(handle: rocblas_handle; n: Int64; const x: PDouble; incx: Int64; const y: PDouble; incy: Int64; result: PDouble): rocblas_status; cdecl; external;
function rocblas_hdot_64(handle: rocblas_handle; n: Int64; const x: Prochalf; incx: Int64; const y: Prochalf; incy: Int64; result: Prochalf): rocblas_status; cdecl; external;
function rocblas_bfdot_64(handle: rocblas_handle; n: Int64; const x: Procbfloat16; incx: Int64; const y: Procbfloat16; incy: Int64; result: Procbfloat16): rocblas_status; cdecl; external;
function rocblas_cdotu_64(handle: rocblas_handle; n: Int64; const x: Procfloat_complex; incx: Int64; const y: Procfloat_complex; incy: Int64; result: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zdotu_64(handle: rocblas_handle; n: Int64; const x: Procdouble_complex; incx: Int64; const y: Procdouble_complex; incy: Int64; result: Procdouble_complex): rocblas_status; cdecl; external;
function rocblas_cdotc_64(handle: rocblas_handle; n: Int64; const x: Procfloat_complex; incx: Int64; const y: Procfloat_complex; incy: Int64; result: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zdotc_64(handle: rocblas_handle; n: Int64; const x: Procdouble_complex; incx: Int64; const y: Procdouble_complex; incy: Int64; result: Procdouble_complex): rocblas_status; cdecl; external;

{ nrm2: result = ||x|| }
function rocblas_snrm2(handle: rocblas_handle; n: rocblas_int; const x: PSingle; incx: rocblas_int; result: PSingle): rocblas_status; cdecl; external;
function rocblas_dnrm2(handle: rocblas_handle; n: rocblas_int; const x: PDouble; incx: rocblas_int; result: PDouble): rocblas_status; cdecl; external;
function rocblas_scnrm2(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; result: PSingle): rocblas_status; cdecl; external;
function rocblas_dznrm2(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; result: PDouble): rocblas_status; cdecl; external;

{ nrm2_batched }
function rocblas_snrm2_batched(handle: rocblas_handle; n: rocblas_int; const x: PPSingle; incx: rocblas_int; batch_count: rocblas_int; results: PSingle): rocblas_status; cdecl; external;
function rocblas_dnrm2_batched(handle: rocblas_handle; n: rocblas_int; const x: PPDouble; incx: rocblas_int; batch_count: rocblas_int; results: PDouble): rocblas_status; cdecl; external;
function rocblas_scnrm2_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcfloat_complex; incx: rocblas_int; batch_count: rocblas_int; results: PSingle): rocblas_status; cdecl; external;
function rocblas_dznrm2_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcdouble_complex; incx: rocblas_int; batch_count: rocblas_int; results: PDouble): rocblas_status; cdecl; external;

{ nrm2_strided_batched }
function rocblas_snrm2_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: PSingle; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; results: PSingle): rocblas_status; cdecl; external;
function rocblas_dnrm2_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: PDouble; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; results: PDouble): rocblas_status; cdecl; external;
function rocblas_scnrm2_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; results: PSingle): rocblas_status; cdecl; external;
function rocblas_dznrm2_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; results: PDouble): rocblas_status; cdecl; external;


{ nrm2 64-bit }
function rocblas_snrm2_64(handle: rocblas_handle; n: Int64; const x: PSingle; incx: Int64; result: PSingle): rocblas_status; cdecl; external;
function rocblas_dnrm2_64(handle: rocblas_handle; n: Int64; const x: PDouble; incx: Int64; result: PDouble): rocblas_status; cdecl; external;
function rocblas_scnrm2_64(handle: rocblas_handle; n: Int64; const x: Procfloat_complex; incx: Int64; result: PSingle): rocblas_status; cdecl; external;
function rocblas_dznrm2_64(handle: rocblas_handle; n: Int64; const x: Procdouble_complex; incx: Int64; result: PDouble): rocblas_status; cdecl; external;

{ scal: x := alpha * x }
function rocblas_sscal(handle: rocblas_handle; n: rocblas_int; const alpha, x: PSingle; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dscal(handle: rocblas_handle; n: rocblas_int; const alpha, x: PDouble; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cscal(handle: rocblas_handle; n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zscal(handle: rocblas_handle; n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csscal(handle: rocblas_handle; n: rocblas_int; const alpha: PSingle; x: Procfloat_complex; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zdscal(handle: rocblas_handle; n: rocblas_int; const alpha: PDouble; x: Procdouble_complex; incx: rocblas_int): rocblas_status; cdecl; external;

{ scal_batched }
function rocblas_sscal_batched(handle: rocblas_handle; n: rocblas_int; const alpha: PSingle; x: PPSingle; incx: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dscal_batched(handle: rocblas_handle; n: rocblas_int; const alpha: PDouble; x: PPDouble; incx: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cscal_batched(handle: rocblas_handle; n: rocblas_int; const alpha: Procfloat_complex; x: PPProcfloat_complex; incx: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zscal_batched(handle: rocblas_handle; n: rocblas_int; const alpha: Procdouble_complex; x: PPProcdouble_complex; incx: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csscal_batched(handle: rocblas_handle; n: rocblas_int; const alpha: PSingle; x: PPProcfloat_complex; incx: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zdscal_batched(handle: rocblas_handle; n: rocblas_int; const alpha: PDouble; x: PPProcdouble_complex; incx: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ scal_strided_batched }
function rocblas_sscal_strided_batched(handle: rocblas_handle; n: rocblas_int; const alpha: PSingle; x: PSingle; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dscal_strided_batched(handle: rocblas_handle; n: rocblas_int; const alpha: PDouble; x: PDouble; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cscal_strided_batched(handle: rocblas_handle; n: rocblas_int; const alpha: Procfloat_complex; x: Procfloat_complex; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zscal_strided_batched(handle: rocblas_handle; n: rocblas_int; const alpha: Procdouble_complex; x: Procdouble_complex; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csscal_strided_batched(handle: rocblas_handle; n: rocblas_int; const alpha: PSingle; x: Procfloat_complex; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zdscal_strided_batched(handle: rocblas_handle; n: rocblas_int; const alpha: PDouble; x: Procdouble_complex; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ scal 64-bit }
function rocblas_sscal_64(handle: rocblas_handle; n: Int64; const alpha, x: PSingle; incx: Int64): rocblas_status; cdecl; external;
function rocblas_dscal_64(handle: rocblas_handle; n: Int64; const alpha, x: PDouble; incx: Int64): rocblas_status; cdecl; external;
function rocblas_cscal_64(handle: rocblas_handle; n: Int64; const alpha, x: Procfloat_complex; incx: Int64): rocblas_status; cdecl; external;
function rocblas_zscal_64(handle: rocblas_handle; n: Int64; const alpha, x: Procdouble_complex; incx: Int64): rocblas_status; cdecl; external;
function rocblas_csscal_64(handle: rocblas_handle; n: Int64; const alpha: PSingle; x: Procfloat_complex; incx: Int64): rocblas_status; cdecl; external;
function rocblas_zdscal_64(handle: rocblas_handle; n: Int64; const alpha: PDouble; x: Procdouble_complex; incx: Int64): rocblas_status; cdecl; external;

{ swap: y := x; x := y }
function rocblas_sswap(handle: rocblas_handle; n: rocblas_int; x: PSingle; incx: rocblas_int; y: PSingle; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dswap(handle: rocblas_handle; n: rocblas_int; x: PDouble; incx: rocblas_int; y: PDouble; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cswap(handle: rocblas_handle; n: rocblas_int; x: Procfloat_complex; incx: rocblas_int; y: Procfloat_complex; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zswap(handle: rocblas_handle; n: rocblas_int; x: Procdouble_complex; incx: rocblas_int; y: Procdouble_complex; incy: rocblas_int): rocblas_status; cdecl; external;

{ swap_batched }
function rocblas_sswap_batched(handle: rocblas_handle; n: rocblas_int; x: PPSingle; incx: rocblas_int; y: PPSingle; incy: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dswap_batched(handle: rocblas_handle; n: rocblas_int; x: PPDouble; incx: rocblas_int; y: PPDouble; incy: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cswap_batched(handle: rocblas_handle; n: rocblas_int; x: PPProcfloat_complex; incx: rocblas_int; y: PPProcfloat_complex; incy: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zswap_batched(handle: rocblas_handle; n: rocblas_int; x: PPProcdouble_complex; incx: rocblas_int; y: PPProcdouble_complex; incy: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ swap_strided_batched }
function rocblas_sswap_strided_batched(handle: rocblas_handle; n: rocblas_int; x: PSingle; incx: rocblas_int; stridex: rocblas_stride; y: PSingle; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dswap_strided_batched(handle: rocblas_handle; n: rocblas_int; x: PDouble; incx: rocblas_int; stridex: rocblas_stride; y: PDouble; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cswap_strided_batched(handle: rocblas_handle; n: rocblas_int; x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zswap_strided_batched(handle: rocblas_handle; n: rocblas_int; x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ swap 64-bit }
function rocblas_sswap_64(handle: rocblas_handle; n: Int64; x: PSingle; incx: Int64; y: PSingle; incy: Int64): rocblas_status; cdecl; external;
function rocblas_dswap_64(handle: rocblas_handle; n: Int64; x: PDouble; incx: Int64; y: PDouble; incy: Int64): rocblas_status; cdecl; external;
function rocblas_cswap_64(handle: rocblas_handle; n: Int64; x: Procfloat_complex; incx: Int64; y: Procfloat_complex; incy: Int64): rocblas_status; cdecl; external;
function rocblas_zswap_64(handle: rocblas_handle; n: Int64; x: Procdouble_complex; incx: Int64; y: Procdouble_complex; incy: Int64): rocblas_status; cdecl; external;

{ asum: result = sum of magnitudes of elements of x }
function rocblas_sasum(handle: rocblas_handle; n: rocblas_int; const x: PSingle; incx: rocblas_int; result: PSingle): rocblas_status; cdecl; external;
function rocblas_dasum(handle: rocblas_handle; n: rocblas_int; const x: PDouble; incx: rocblas_int; result: PDouble): rocblas_status; cdecl; external;
function rocblas_scasum(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; result: PSingle): rocblas_status; cdecl; external;
function rocblas_dzasum(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; result: PDouble): rocblas_status; cdecl; external;

{ asum_batched }
function rocblas_sasum_batched(handle: rocblas_handle; n: rocblas_int; const x: PPSingle; incx: rocblas_int; batch_count: rocblas_int; results: PSingle): rocblas_status; cdecl; external;
function rocblas_dasum_batched(handle: rocblas_handle; n: rocblas_int; const x: PPDouble; incx: rocblas_int; batch_count: rocblas_int; results: PDouble): rocblas_status; cdecl; external;
function rocblas_scasum_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcfloat_complex; incx: rocblas_int; batch_count: rocblas_int; results: PSingle): rocblas_status; cdecl; external;
function rocblas_dzasum_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcdouble_complex; incx: rocblas_int; batch_count: rocblas_int; results: PDouble): rocblas_status; cdecl; external;

{ asum_strided_batched }
function rocblas_sasum_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: PSingle; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; results: PSingle): rocblas_status; cdecl; external;
function rocblas_dasum_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: PDouble; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; results: PDouble): rocblas_status; cdecl; external;
function rocblas_scasum_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; results: PSingle): rocblas_status; cdecl; external;
function rocblas_dzasum_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; results: PDouble): rocblas_status; cdecl; external;

{ asum 64-bit }
function rocblas_sasum_64(handle: rocblas_handle; n: Int64; const x: PSingle; incx: Int64; result: PSingle): rocblas_status; cdecl; external;
function rocblas_dasum_64(handle: rocblas_handle; n: Int64; const x: PDouble; incx: Int64; result: PDouble): rocblas_status; cdecl; external;
function rocblas_scasum_64(handle: rocblas_handle; n: Int64; const x: Procfloat_complex; incx: Int64; result: PSingle): rocblas_status; cdecl; external;
function rocblas_dzasum_64(handle: rocblas_handle; n: Int64; const x: Procdouble_complex; incx: Int64; result: PDouble): rocblas_status; cdecl; external;


{ amax/amin: result = index of max/min absolute value }

{ amax }
function rocblas_isamax(handle: rocblas_handle; n: rocblas_int; const x: PSingle; incx: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_idamax(handle: rocblas_handle; n: rocblas_int; const x: PDouble; incx: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_icamax(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_izamax(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; result: PInt32): rocblas_status; cdecl; external;

{ amax_batched }
function rocblas_isamax_batched(handle: rocblas_handle; n: rocblas_int; const x: PPSingle; incx: rocblas_int; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_idamax_batched(handle: rocblas_handle; n: rocblas_int; const x: PPDouble; incx: rocblas_int; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_icamax_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcfloat_complex; incx: rocblas_int; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_izamax_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcdouble_complex; incx: rocblas_int; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;

{ amax_strided_batched }
function rocblas_isamax_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: PSingle; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_idamax_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: PDouble; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_icamax_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_izamax_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;

{ amax 64-bit }
function rocblas_isamax_64(handle: rocblas_handle; n: Int64; const x: PSingle; incx: Int64; result: PInt64): rocblas_status; cdecl; external;
function rocblas_idamax_64(handle: rocblas_handle; n: Int64; const x: PDouble; incx: Int64; result: PInt64): rocblas_status; cdecl; external;
function rocblas_icamax_64(handle: rocblas_handle; n: Int64; const x: Procfloat_complex; incx: Int64; result: PInt64): rocblas_status; cdecl; external;
function rocblas_izamax_64(handle: rocblas_handle; n: Int64; const x: Procdouble_complex; incx: Int64; result: PInt64): rocblas_status; cdecl; external;

{ amin }
function rocblas_isamin(handle: rocblas_handle; n: rocblas_int; const x: PSingle; incx: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_idamin(handle: rocblas_handle; n: rocblas_int; const x: PDouble; incx: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_icamin(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_izamin(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; result: PInt32): rocblas_status; cdecl; external;

{ amin_batched }
function rocblas_isamin_batched(handle: rocblas_handle; n: rocblas_int; const x: PPSingle; incx: rocblas_int; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_idamin_batched(handle: rocblas_handle; n: rocblas_int; const x: PPDouble; incx: rocblas_int; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_icamin_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcfloat_complex; incx: rocblas_int; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_izamin_batched(handle: rocblas_handle; n: rocblas_int; const x: PPProcdouble_complex; incx: rocblas_int; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;

{ amin_strided_batched }
function rocblas_isamin_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: PSingle; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_idamin_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: PDouble; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_icamin_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;
function rocblas_izamin_strided_batched(handle: rocblas_handle; n: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int; result: PInt32): rocblas_status; cdecl; external;

{ amin 64-bit }
function rocblas_isamin_64(handle: rocblas_handle; n: Int64; const x: PSingle; incx: Int64; result: PInt64): rocblas_status; cdecl; external;
function rocblas_idamin_64(handle: rocblas_handle; n: Int64; const x: PDouble; incx: Int64; result: PInt64): rocblas_status; cdecl; external;
function rocblas_icamin_64(handle: rocblas_handle; n: Int64; const x: Procfloat_complex; incx: Int64; result: PInt64): rocblas_status; cdecl; external;
function rocblas_izamin_64(handle: rocblas_handle; n: Int64; const x: Procdouble_complex; incx: Int64; result: PInt64): rocblas_status; cdecl; external;

{ rot / rotg / rotm / rotmg  }

{ rot scalar }
function rocblas_srot(handle: rocblas_handle; n: rocblas_int; x: PSingle; incx: rocblas_int; y: PSingle; incy: rocblas_int; const c, s: PSingle): rocblas_status; cdecl; external;
function rocblas_drot(handle: rocblas_handle; n: rocblas_int; x: PDouble; incx: rocblas_int; y: PDouble; incy: rocblas_int; const c, s: PDouble): rocblas_status; cdecl; external;
function rocblas_crot(handle: rocblas_handle; n: rocblas_int; x: Procfloat_complex; incx: rocblas_int; y: Procfloat_complex; incy: rocblas_int; const c: PSingle; const s: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_csrot(handle: rocblas_handle; n: rocblas_int; x: Procfloat_complex; incx: rocblas_int; y: Procfloat_complex; incy: rocblas_int; const c, s: PSingle): rocblas_status; cdecl; external;
function rocblas_zrot(handle: rocblas_handle; n: rocblas_int; x: Procdouble_complex; incx: rocblas_int; y: Procdouble_complex; incy: rocblas_int; const c: PDouble; const s: Procdouble_complex): rocblas_status; cdecl; external;
function rocblas_zdrot(handle: rocblas_handle; n: rocblas_int; x: Procdouble_complex; incx: rocblas_int; y: Procdouble_complex; incy: rocblas_int; const c, s: PDouble): rocblas_status; cdecl; external;

{ rot scalar 64-bit }
function rocblas_srot_64(handle: rocblas_handle; n: Int64; x: PSingle; incx: Int64; y: PSingle; incy: Int64; const c, s: PSingle): rocblas_status; cdecl; external;
function rocblas_drot_64(handle: rocblas_handle; n: Int64; x: PDouble; incx: Int64; y: PDouble; incy: Int64; const c, s: PDouble): rocblas_status; cdecl; external;
function rocblas_crot_64(handle: rocblas_handle; n: Int64; x: Procfloat_complex; incx: Int64; y: Procfloat_complex; incy: Int64; const c: PSingle; const s: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_csrot_64(handle: rocblas_handle; n: Int64; x: Procfloat_complex; incx: Int64; y: Procfloat_complex; incy: Int64; const c, s: PSingle): rocblas_status; cdecl; external;
function rocblas_zrot_64(handle: rocblas_handle; n: Int64; x: Procdouble_complex; incx: Int64; y: Procdouble_complex; incy: Int64; const c: PDouble; const s: Procdouble_complex): rocblas_status; cdecl; external;
function rocblas_zdrot_64(handle: rocblas_handle; n: Int64; x: Procdouble_complex; incx: Int64; y: Procdouble_complex; incy: Int64; const c, s: PDouble): rocblas_status; cdecl; external;

{ rot batched }
function rocblas_srot_batched(handle: rocblas_handle; n: rocblas_int; x: PPSingle; incx: rocblas_int; y: PPSingle; incy: rocblas_int; const c, s: PSingle; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_drot_batched(handle: rocblas_handle; n: rocblas_int; x: PPDouble; incx: rocblas_int; y: PPDouble; incy: rocblas_int; const c, s: PDouble; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_crot_batched(handle: rocblas_handle; n: rocblas_int; x: PPProcfloat_complex; incx: rocblas_int; y: PPProcfloat_complex; incy: rocblas_int; const c: PSingle; const s: Procfloat_complex; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csrot_batched(handle: rocblas_handle; n: rocblas_int; x: PPProcfloat_complex; incx: rocblas_int; y: PPProcfloat_complex; incy: rocblas_int; const c, s: PSingle; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zrot_batched(handle: rocblas_handle; n: rocblas_int; x: PPProcdouble_complex; incx: rocblas_int; y: PPProcdouble_complex; incy: rocblas_int; const c: PDouble; const s: Procdouble_complex; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zdrot_batched(handle: rocblas_handle; n: rocblas_int; x: PPProcdouble_complex; incx: rocblas_int; y: PPProcdouble_complex; incy: rocblas_int; const c, s: PDouble; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ rot batched 64-bit }
function rocblas_srot_batched_64(handle: rocblas_handle; n: Int64; x: PPSingle; incx: Int64; y: PPSingle; incy: Int64; const c, s: PSingle; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_drot_batched_64(handle: rocblas_handle; n: Int64; x: PPDouble; incx: Int64; y: PPDouble; incy: Int64; const c, s: PDouble; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_crot_batched_64(handle: rocblas_handle; n: Int64; x: PPProcfloat_complex; incx: Int64; y: PPProcfloat_complex; incy: Int64; const c: PSingle; const s: Procfloat_complex; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csrot_batched_64(handle: rocblas_handle; n: Int64; x: PPProcfloat_complex; incx: Int64; y: PPProcfloat_complex; incy: Int64; const c, s: PSingle; batch_count: Int64): rocblas_status; cdecl; external;

{ rot strided_batched }
function rocblas_srot_strided_batched(handle: rocblas_handle; n: rocblas_int; x: PSingle; incx: rocblas_int; stride_x: rocblas_stride; y: PSingle; incy: rocblas_int; stride_y: rocblas_stride; const c, s: PSingle; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_drot_strided_batched(handle: rocblas_handle; n: rocblas_int; x: PDouble; incx: rocblas_int; stride_x: rocblas_stride; y: PDouble; incy: rocblas_int; stride_y: rocblas_stride; const c, s: PDouble; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_crot_strided_batched(handle: rocblas_handle; n: rocblas_int; x: Procfloat_complex; incx: rocblas_int; stride_x: rocblas_stride; y: Procfloat_complex; incy: rocblas_int; stride_y: rocblas_stride; const c: PSingle; const s: Procfloat_complex; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csrot_strided_batched(handle: rocblas_handle; n: rocblas_int; x: Procfloat_complex; incx: rocblas_int; stride_x: rocblas_stride; y: Procfloat_complex; incy: rocblas_int; stride_y: rocblas_stride; const c, s: PSingle; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zrot_strided_batched(handle: rocblas_handle; n: rocblas_int; x: Procdouble_complex; incx: rocblas_int; stride_x: rocblas_stride; y: Procdouble_complex; incy: rocblas_int; stride_y: rocblas_stride; const c: PDouble; const s: Procdouble_complex; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zdrot_strided_batched(handle: rocblas_handle; n: rocblas_int; x: Procdouble_complex; incx: rocblas_int; stride_x: rocblas_stride; y: Procdouble_complex; incy: rocblas_int; stride_y: rocblas_stride; const c, s: PDouble; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ rot strided_batched 64-bit }
function rocblas_srot_strided_batched_64(handle: rocblas_handle; n: Int64; x: PSingle; incx: Int64; stride_x: rocblas_stride; y: PSingle; incy: Int64; stride_y: rocblas_stride; const c, s: PSingle; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_drot_strided_batched_64(handle: rocblas_handle; n: Int64; x: PDouble; incx: Int64; stride_x: rocblas_stride; y: PDouble; incy: Int64; stride_y: rocblas_stride; const c, s: PDouble; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_crot_strided_batched_64(handle: rocblas_handle; n: Int64; x: Procfloat_complex; incx: Int64; stride_x: rocblas_stride; y: Procfloat_complex; incy: Int64; stride_y: rocblas_stride; const c: PSingle; const s: Procfloat_complex; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csrot_strided_batched_64(handle: rocblas_handle; n: Int64; x: Procfloat_complex; incx: Int64; stride_x: rocblas_stride; y: Procfloat_complex; incy: Int64; stride_y: rocblas_stride; const c, s: PSingle; batch_count: Int64): rocblas_status; cdecl; external;

{ rotg scalar }
function rocblas_srotg(handle: rocblas_handle; a, b, c, s: PSingle): rocblas_status; cdecl; external;
function rocblas_drotg(handle: rocblas_handle; a, b, c, s: PDouble): rocblas_status; cdecl; external;
function rocblas_crotg(handle: rocblas_handle; a, b: Procfloat_complex; c: PSingle; s: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zrotg(handle: rocblas_handle; a, b: Procdouble_complex; c: PDouble; s: Procdouble_complex): rocblas_status; cdecl; external;

{ rotg scalar 64-bit }
function rocblas_srotg_64(handle: rocblas_handle; a, b, c, s: PSingle): rocblas_status; cdecl; external;
function rocblas_drotg_64(handle: rocblas_handle; a, b, c, s: PDouble): rocblas_status; cdecl; external;
function rocblas_crotg_64(handle: rocblas_handle; a, b: Procfloat_complex; c: PSingle; s: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zrotg_64(handle: rocblas_handle; a, b: Procdouble_complex; c: PDouble; s: Procdouble_complex): rocblas_status; cdecl; external;

{ rotg batched }
function rocblas_srotg_batched(handle: rocblas_handle; a, b, c, s: PPSingle; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_drotg_batched(handle: rocblas_handle; a, b, c, s: PPDouble; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_crotg_batched(handle: rocblas_handle; a, b: PPProcfloat_complex; c: PPSingle; s: PPProcfloat_complex; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zrotg_batched(handle: rocblas_handle; a, b: PPProcdouble_complex; c: PPDouble; s: PPProcdouble_complex; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ rotg batched 64-bit }
function rocblas_srotg_batched_64(handle: rocblas_handle; a, b, c, s: PPSingle; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_drotg_batched_64(handle: rocblas_handle; a, b, c, s: PPDouble; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_crotg_batched_64(handle: rocblas_handle; a, b: PPProcfloat_complex; c: PPSingle; s: PPProcfloat_complex; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zrotg_batched_64(handle: rocblas_handle; a, b: PPProcdouble_complex; c: PPDouble; s: PPProcdouble_complex; batch_count: Int64): rocblas_status; cdecl; external;

{ rotg strided_batched }
function rocblas_srotg_strided_batched(handle: rocblas_handle; a: PSingle; stride_a: rocblas_stride; b: PSingle; stride_b: rocblas_stride; c: PSingle; stride_c: rocblas_stride; s: PSingle; stride_s: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_drotg_strided_batched(handle: rocblas_handle; a: PDouble; stride_a: rocblas_stride; b: PDouble; stride_b: rocblas_stride; c: PDouble; stride_c: rocblas_stride; s: PDouble; stride_s: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_crotg_strided_batched(handle: rocblas_handle; a: Procfloat_complex; stride_a: rocblas_stride; b: Procfloat_complex; stride_b: rocblas_stride; c: PSingle; stride_c: rocblas_stride; s: Procfloat_complex; stride_s: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zrotg_strided_batched(handle: rocblas_handle; a: Procdouble_complex; stride_a: rocblas_stride; b: Procdouble_complex; stride_b: rocblas_stride; c: PDouble; stride_c: rocblas_stride; s: Procdouble_complex; stride_s: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ rotg strided_batched 64-bit }
function rocblas_srotg_strided_batched_64(handle: rocblas_handle; a: PSingle; stride_a: rocblas_stride; b: PSingle; stride_b: rocblas_stride; c: PSingle; stride_c: rocblas_stride; s: PSingle; stride_s: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_drotg_strided_batched_64(handle: rocblas_handle; a: PDouble; stride_a: rocblas_stride; b: PDouble; stride_b: rocblas_stride; c: PDouble; stride_c: rocblas_stride; s: PDouble; stride_s: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_crotg_strided_batched_64(handle: rocblas_handle; a: Procfloat_complex; stride_a: rocblas_stride; b: Procfloat_complex; stride_b: rocblas_stride; c: PSingle; stride_c: rocblas_stride; s: Procfloat_complex; stride_s: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zrotg_strided_batched_64(handle: rocblas_handle; a: Procdouble_complex; stride_a: rocblas_stride; b: Procdouble_complex; stride_b: rocblas_stride; c: PDouble; stride_c: rocblas_stride; s: Procdouble_complex; stride_s: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ rotm scalar }
function rocblas_srotm(handle: rocblas_handle; n: rocblas_int; x: PSingle; incx: rocblas_int; y: PSingle; incy: rocblas_int; const param: PSingle): rocblas_status; cdecl; external;
function rocblas_drotm(handle: rocblas_handle; n: rocblas_int; x: PDouble; incx: rocblas_int; y: PDouble; incy: rocblas_int; const param: PDouble): rocblas_status; cdecl; external;

{ rotm scalar 64-bit }
function rocblas_srotm_64(handle: rocblas_handle; n: Int64; x: PSingle; incx: Int64; y: PSingle; incy: Int64; const param: PSingle): rocblas_status; cdecl; external;
function rocblas_drotm_64(handle: rocblas_handle; n: Int64; x: PDouble; incx: Int64; y: PDouble; incy: Int64; const param: PDouble): rocblas_status; cdecl; external;

{ rotm batched }
function rocblas_srotm_batched(handle: rocblas_handle; n: rocblas_int; x: PPSingle; incx: rocblas_int; y: PPSingle; incy: rocblas_int; const param: PPSingle; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_drotm_batched(handle: rocblas_handle; n: rocblas_int; x: PPDouble; incx: rocblas_int; y: PPDouble; incy: rocblas_int; const param: PPDouble; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ rotm batched 64-bit }
function rocblas_srotm_batched_64(handle: rocblas_handle; n: Int64; x: PPSingle; incx: Int64; y: PPSingle; incy: Int64; const param: PPSingle; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_drotm_batched_64(handle: rocblas_handle; n: Int64; x: PPDouble; incx: Int64; y: PPDouble; incy: Int64; const param: PPDouble; batch_count: Int64): rocblas_status; cdecl; external;

{ rotm strided_batched }
function rocblas_srotm_strided_batched(handle: rocblas_handle; n: rocblas_int; x: PSingle; incx: rocblas_int; stride_x: rocblas_stride; y: PSingle; incy: rocblas_int; stride_y: rocblas_stride; const param: PSingle; stride_param: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_drotm_strided_batched(handle: rocblas_handle; n: rocblas_int; x: PDouble; incx: rocblas_int; stride_x: rocblas_stride; y: PDouble; incy: rocblas_int; stride_y: rocblas_stride; const param: PDouble; stride_param: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ rotm strided_batched 64-bit }
function rocblas_srotm_strided_batched_64(handle: rocblas_handle; n: Int64; x: PSingle; incx: Int64; stride_x: rocblas_stride; y: PSingle; incy: Int64; stride_y: rocblas_stride; const param: PSingle; stride_param: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_drotm_strided_batched_64(handle: rocblas_handle; n: Int64; x: PDouble; incx: Int64; stride_x: rocblas_stride; y: PDouble; incy: Int64; stride_y: rocblas_stride; const param: PDouble; stride_param: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ rotmg scalar }
function rocblas_srotmg(handle: rocblas_handle; d1, d2, x1: PSingle; const y1: PSingle; param: PSingle): rocblas_status; cdecl; external;
function rocblas_drotmg(handle: rocblas_handle; d1, d2, x1: PDouble; const y1: PDouble; param: PDouble): rocblas_status; cdecl; external;

{ rotmg scalar 64-bit }
function rocblas_srotmg_64(handle: rocblas_handle; d1, d2, x1: PSingle; const y1: PSingle; param: PSingle): rocblas_status; cdecl; external;
function rocblas_drotmg_64(handle: rocblas_handle; d1, d2, x1: PDouble; const y1: PDouble; param: PDouble): rocblas_status; cdecl; external;

{ rotmg batched }
function rocblas_srotmg_batched(handle: rocblas_handle; d1, d2, x1: PPSingle; const y1: PPSingle; param: PPSingle; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_drotmg_batched(handle: rocblas_handle; d1, d2, x1: PPDouble; const y1: PPDouble; param: PPDouble; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ rotmg batched 64-bit }
function rocblas_srotmg_batched_64(handle: rocblas_handle; d1, d2, x1: PPSingle; const y1: PPSingle; param: PPSingle; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_drotmg_batched_64(handle: rocblas_handle; d1, d2, x1: PPDouble; const y1: PPDouble; param: PPDouble; batch_count: Int64): rocblas_status; cdecl; external;

{ rotmg strided_batched }
function rocblas_srotmg_strided_batched(handle: rocblas_handle; d1: PSingle; stride_d1: rocblas_stride; d2: PSingle; stride_d2: rocblas_stride; x1: PSingle; stride_x1: rocblas_stride; const y1: PSingle; stride_y1: rocblas_stride; param: PSingle; stride_param: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_drotmg_strided_batched(handle: rocblas_handle; d1: PDouble; stride_d1: rocblas_stride; d2: PDouble; stride_d2: rocblas_stride; x1: PDouble; stride_x1: rocblas_stride; const y1: PDouble; stride_y1: rocblas_stride; param: PDouble; stride_param: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ rotmg strided_batched 64-bit }
function rocblas_srotmg_strided_batched_64(handle: rocblas_handle; d1: PSingle; stride_d1: rocblas_stride; d2: PSingle; stride_d2: rocblas_stride; x1: PSingle; stride_x1: rocblas_stride; const y1: PSingle; stride_y1: rocblas_stride; param: PSingle; stride_param: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_drotmg_strided_batched_64(handle: rocblas_handle; d1: PDouble; stride_d1: rocblas_stride; d2: PDouble; stride_d2: rocblas_stride; x1: PDouble; stride_x1: rocblas_stride; const y1: PDouble; stride_y1: rocblas_stride; param: PDouble; stride_param: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;


{ ============================================================================================ }
{   Level 2 BLAS functions                                                                     }
{ ============================================================================================ }

{ gemv: y := alpha*A*x + beta*y, or y := alpha*A**T*x + beta*y, or y := alpha*A**H*x + beta*y }
function rocblas_sgemv(handle: rocblas_handle; trans: rocblas_operation; m, n: rocblas_int; const alpha, A: PSingle; lda: rocblas_int; const x: PSingle; incx: rocblas_int; const beta: PSingle; y: PSingle; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dgemv(handle: rocblas_handle; trans: rocblas_operation; m, n: rocblas_int; const alpha, A: PDouble; lda: rocblas_int; const x: PDouble; incx: rocblas_int; const beta: PDouble; y: PDouble; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgemv(handle: rocblas_handle; trans: rocblas_operation; m, n: rocblas_int; const alpha, A: Procfloat_complex; lda: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; const beta: Procfloat_complex; y: Procfloat_complex; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgemv(handle: rocblas_handle; trans: rocblas_operation; m, n: rocblas_int; const alpha, A: Procdouble_complex; lda: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; const beta: Procdouble_complex; y: Procdouble_complex; incy: rocblas_int): rocblas_status; cdecl; external;

{ gemv 64-bit }
function rocblas_sgemv_64(handle: rocblas_handle; trans: rocblas_operation; m, n: Int64; const alpha, A: PSingle; lda: Int64; const x: PSingle; incx: Int64; const beta: PSingle; y: PSingle; incy: Int64): rocblas_status; cdecl; external;
function rocblas_dgemv_64(handle: rocblas_handle; trans: rocblas_operation; m, n: Int64; const alpha, A: PDouble; lda: Int64; const x: PDouble; incx: Int64; const beta: PDouble; y: PDouble; incy: Int64): rocblas_status; cdecl; external;
function rocblas_cgemv_64(handle: rocblas_handle; trans: rocblas_operation; m, n: Int64; const alpha, A: Procfloat_complex; lda: Int64; const x: Procfloat_complex; incx: Int64; const beta: Procfloat_complex; y: Procfloat_complex; incy: Int64): rocblas_status; cdecl; external;
function rocblas_zgemv_64(handle: rocblas_handle; trans: rocblas_operation; m, n: Int64; const alpha, A: Procdouble_complex; lda: Int64; const x: Procdouble_complex; incx: Int64; const beta: Procdouble_complex; y: Procdouble_complex; incy: Int64): rocblas_status; cdecl; external;

{ gemv batched }
function rocblas_sgemv_batched(handle: rocblas_handle; trans: rocblas_operation; m, n: rocblas_int; const alpha: PSingle; const A: PPSingle; lda: rocblas_int; const x: PPSingle; incx: rocblas_int; const beta: PSingle; y: PPSingle; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dgemv_batched(handle: rocblas_handle; trans: rocblas_operation; m, n: rocblas_int; const alpha: PDouble; const A: PPDouble; lda: rocblas_int; const x: PPDouble; incx: rocblas_int; const beta: PDouble; y: PPDouble; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgemv_batched(handle: rocblas_handle; trans: rocblas_operation; m, n: rocblas_int; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: rocblas_int; const x: PPProcfloat_complex; incx: rocblas_int; const beta: Procfloat_complex; y: PPProcfloat_complex; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgemv_batched(handle: rocblas_handle; trans: rocblas_operation; m, n: rocblas_int; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: rocblas_int; const x: PPProcdouble_complex; incx: rocblas_int; const beta: Procdouble_complex; y: PPProcdouble_complex; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ gemv batched 64-bit }
function rocblas_sgemv_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n: Int64; const alpha: PSingle; const A: PPSingle; lda: Int64; const x: PPSingle; incx: Int64; const beta: PSingle; y: PPSingle; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dgemv_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n: Int64; const alpha: PDouble; const A: PPDouble; lda: Int64; const x: PPDouble; incx: Int64; const beta: PDouble; y: PPDouble; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_cgemv_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n: Int64; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: Int64; const x: PPProcfloat_complex; incx: Int64; const beta: Procfloat_complex; y: PPProcfloat_complex; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zgemv_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n: Int64; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: Int64; const x: PPProcdouble_complex; incx: Int64; const beta: Procdouble_complex; y: PPProcdouble_complex; incy, batch_count: Int64): rocblas_status; cdecl; external;

{ gemv strided batched }
function rocblas_sgemv_strided_batched(handle: rocblas_handle; trans: rocblas_operation; m, n: rocblas_int; const alpha, A: PSingle; lda: rocblas_int; strideA: rocblas_stride; const x: PSingle; incx: rocblas_int; stridex: rocblas_stride; const beta: PSingle; y: PSingle; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dgemv_strided_batched(handle: rocblas_handle; trans: rocblas_operation; m, n: rocblas_int; const alpha, A: PDouble; lda: rocblas_int; strideA: rocblas_stride; const x: PDouble; incx: rocblas_int; stridex: rocblas_stride; const beta: PDouble; y: PDouble; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgemv_strided_batched(handle: rocblas_handle; trans: rocblas_operation; m, n: rocblas_int; const alpha, A: Procfloat_complex; lda: rocblas_int; strideA: rocblas_stride; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; const beta: Procfloat_complex; y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgemv_strided_batched(handle: rocblas_handle; trans: rocblas_operation; m, n: rocblas_int; const alpha, A: Procdouble_complex; lda: rocblas_int; strideA: rocblas_stride; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; const beta: Procdouble_complex; y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ gemv strided batched 64-bit }
function rocblas_sgemv_strided_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n: Int64; const alpha, A: PSingle; lda: Int64; strideA: rocblas_stride; const x: PSingle; incx: Int64; stridex: rocblas_stride; const beta: PSingle; y: PSingle; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dgemv_strided_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n: Int64; const alpha, A: PDouble; lda: Int64; strideA: rocblas_stride; const x: PDouble; incx: Int64; stridex: rocblas_stride; const beta: PDouble; y: PDouble; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_cgemv_strided_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n: Int64; const alpha, A: Procfloat_complex; lda: Int64; strideA: rocblas_stride; const x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; const beta: Procfloat_complex; y: Procfloat_complex; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zgemv_strided_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n: Int64; const alpha, A: Procdouble_complex; lda: Int64; strideA: rocblas_stride; const x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; const beta: Procdouble_complex; y: Procdouble_complex; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ gbmv scalar }
function rocblas_sgbmv(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: rocblas_int; const alpha: PSingle; const A: PSingle; lda: rocblas_int; const x: PSingle; incx: rocblas_int; const beta: PSingle; y: PSingle; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dgbmv(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: rocblas_int; const alpha: PDouble; const A: PDouble; lda: rocblas_int; const x: PDouble; incx: rocblas_int; const beta: PDouble; y: PDouble; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgbmv(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; const beta: Procfloat_complex; y: Procfloat_complex; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgbmv(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; const beta: Procdouble_complex; y: Procdouble_complex; incy: rocblas_int): rocblas_status; cdecl; external;

{ gbmv scalar 64-bit }
function rocblas_sgbmv_64(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: Int64; const alpha: PSingle; const A: PSingle; lda: Int64; const x: PSingle; incx: Int64; const beta: PSingle; y: PSingle; incy: Int64): rocblas_status; cdecl; external;
function rocblas_dgbmv_64(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: Int64; const alpha: PDouble; const A: PDouble; lda: Int64; const x: PDouble; incx: Int64; const beta: PDouble; y: PDouble; incy: Int64): rocblas_status; cdecl; external;
function rocblas_cgbmv_64(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; const x: Procfloat_complex; incx: Int64; const beta: Procfloat_complex; y: Procfloat_complex; incy: Int64): rocblas_status; cdecl; external;
function rocblas_zgbmv_64(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; const x: Procdouble_complex; incx: Int64; const beta: Procdouble_complex; y: Procdouble_complex; incy: Int64): rocblas_status; cdecl; external;

{ gbmv batched }
function rocblas_sgbmv_batched(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: rocblas_int; const alpha: PSingle; const A: PPSingle; lda: rocblas_int; const x: PPSingle; incx: rocblas_int; const beta: PSingle; y: PPSingle; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dgbmv_batched(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: rocblas_int; const alpha: PDouble; const A: PPDouble; lda: rocblas_int; const x: PPDouble; incx: rocblas_int; const beta: PDouble; y: PPDouble; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgbmv_batched(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: rocblas_int; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: rocblas_int; const x: PPProcfloat_complex; incx: rocblas_int; const beta: Procfloat_complex; y: PPProcfloat_complex; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgbmv_batched(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: rocblas_int; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: rocblas_int; const x: PPProcdouble_complex; incx: rocblas_int; const beta: Procdouble_complex; y: PPProcdouble_complex; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ gbmv batched 64-bit }
function rocblas_sgbmv_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: Int64; const alpha: PSingle; const A: PPSingle; lda: Int64; const x: PPSingle; incx: Int64; const beta: PSingle; y: PPSingle; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dgbmv_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: Int64; const alpha: PDouble; const A: PPDouble; lda: Int64; const x: PPDouble; incx: Int64; const beta: PDouble; y: PPDouble; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_cgbmv_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: Int64; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: Int64; const x: PPProcfloat_complex; incx: Int64; const beta: Procfloat_complex; y: PPProcfloat_complex; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zgbmv_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: Int64; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: Int64; const x: PPProcdouble_complex; incx: Int64; const beta: Procdouble_complex; y: PPProcdouble_complex; incy, batch_count: Int64): rocblas_status; cdecl; external;

{ gbmv strided_batched }
function rocblas_sgbmv_strided_batched(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: rocblas_int; const alpha: PSingle; const A: PSingle; lda: rocblas_int; stride_A: rocblas_stride; const x: PSingle; incx: rocblas_int; stride_x: rocblas_stride; const beta: PSingle; y: PSingle; incy: rocblas_int; stride_y: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dgbmv_strided_batched(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: rocblas_int; const alpha: PDouble; const A: PDouble; lda: rocblas_int; stride_A: rocblas_stride; const x: PDouble; incx: rocblas_int; stride_x: rocblas_stride; const beta: PDouble; y: PDouble; incy: rocblas_int; stride_y: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgbmv_strided_batched(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; stride_A: rocblas_stride; const x: Procfloat_complex; incx: rocblas_int; stride_x: rocblas_stride; const beta: Procfloat_complex; y: Procfloat_complex; incy: rocblas_int; stride_y: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgbmv_strided_batched(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; stride_A: rocblas_stride; const x: Procdouble_complex; incx: rocblas_int; stride_x: rocblas_stride; const beta: Procdouble_complex; y: Procdouble_complex; incy: rocblas_int; stride_y: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ gbmv strided_batched 64-bit }
function rocblas_sgbmv_strided_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: Int64; const alpha: PSingle; const A: PSingle; lda: Int64; stride_A: rocblas_stride; const x: PSingle; incx: Int64; stride_x: rocblas_stride; const beta: PSingle; y: PSingle; incy: Int64; stride_y: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dgbmv_strided_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: Int64; const alpha: PDouble; const A: PDouble; lda: Int64; stride_A: rocblas_stride; const x: PDouble; incx: Int64; stride_x: rocblas_stride; const beta: PDouble; y: PDouble; incy: Int64; stride_y: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_cgbmv_strided_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; stride_A: rocblas_stride; const x: Procfloat_complex; incx: Int64; stride_x: rocblas_stride; const beta: Procfloat_complex; y: Procfloat_complex; incy: Int64; stride_y: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zgbmv_strided_batched_64(handle: rocblas_handle; trans: rocblas_operation; m, n, kl, ku: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; stride_A: rocblas_stride; const x: Procdouble_complex; incx: Int64; stride_x: rocblas_stride; const beta: Procdouble_complex; y: Procdouble_complex; incy: Int64; stride_y: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ hemv: y := alpha*A*x + beta*y }
function rocblas_chemv(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: Procfloat_complex; lda: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; const beta: Procfloat_complex; y: Procfloat_complex; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhemv(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: Procdouble_complex; lda: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; const beta: Procdouble_complex; y: Procdouble_complex; incy: rocblas_int): rocblas_status; cdecl; external;

{ hemv 64-bit }
function rocblas_chemv_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: Procfloat_complex; lda: Int64; const x: Procfloat_complex; incx: Int64; const beta: Procfloat_complex; y: Procfloat_complex; incy: Int64): rocblas_status; cdecl; external;
function rocblas_zhemv_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: Procdouble_complex; lda: Int64; const x: Procdouble_complex; incx: Int64; const beta: Procdouble_complex; y: Procdouble_complex; incy: Int64): rocblas_status; cdecl; external;

{ hemv batched }
function rocblas_chemv_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: rocblas_int; const x: PPProcfloat_complex; incx: rocblas_int; const beta: Procfloat_complex; y: PPProcfloat_complex; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhemv_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: rocblas_int; const x: PPProcdouble_complex; incx: rocblas_int; const beta: Procdouble_complex; y: PPProcdouble_complex; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_chemv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: Int64; const x: PPProcfloat_complex; incx: Int64; const beta: Procfloat_complex; y: PPProcfloat_complex; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zhemv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: Int64; const x: PPProcdouble_complex; incx: Int64; const beta: Procdouble_complex; y: PPProcdouble_complex; incy, batch_count: Int64): rocblas_status; cdecl; external;

{ hemv strided batched }
function rocblas_chemv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: Procfloat_complex; lda: rocblas_int; strideA: rocblas_stride; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; const beta: Procfloat_complex; y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhemv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: Procdouble_complex; lda: rocblas_int; strideA: rocblas_stride; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; const beta: Procdouble_complex; y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_chemv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: Procfloat_complex; lda: Int64; strideA: rocblas_stride; const x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; const beta: Procfloat_complex; y: Procfloat_complex; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zhemv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: Procdouble_complex; lda: Int64; strideA: rocblas_stride; const x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; const beta: Procdouble_complex; y: Procdouble_complex; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ hbmv: y := alpha*A*x + beta*y }
function rocblas_chbmv(handle: rocblas_handle; uplo: rocblas_fill; n, k: rocblas_int; const alpha, A: Procfloat_complex; lda: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; const beta: Procfloat_complex; y: Procfloat_complex; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhbmv(handle: rocblas_handle; uplo: rocblas_fill; n, k: rocblas_int; const alpha, A: Procdouble_complex; lda: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; const beta: Procdouble_complex; y: Procdouble_complex; incy: rocblas_int): rocblas_status; cdecl; external;

{ hbmv 64-bit }
function rocblas_chbmv_64(handle: rocblas_handle; uplo: rocblas_fill; n, k: Int64; const alpha, A: Procfloat_complex; lda: Int64; const x: Procfloat_complex; incx: Int64; const beta: Procfloat_complex; y: Procfloat_complex; incy: Int64): rocblas_status; cdecl; external;
function rocblas_zhbmv_64(handle: rocblas_handle; uplo: rocblas_fill; n, k: Int64; const alpha, A: Procdouble_complex; lda: Int64; const x: Procdouble_complex; incx: Int64; const beta: Procdouble_complex; y: Procdouble_complex; incy: Int64): rocblas_status; cdecl; external;

{ hbmv batched }
function rocblas_chbmv_batched(handle: rocblas_handle; uplo: rocblas_fill; n, k: rocblas_int; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: rocblas_int; const x: PPProcfloat_complex; incx: rocblas_int; const beta: Procfloat_complex; y: PPProcfloat_complex; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhbmv_batched(handle: rocblas_handle; uplo: rocblas_fill; n, k: rocblas_int; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: rocblas_int; const x: PPProcdouble_complex; incx: rocblas_int; const beta: Procdouble_complex; y: PPProcdouble_complex; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ hbmv batched 64-bit }
function rocblas_chbmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n, k: Int64; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: Int64; const x: PPProcfloat_complex; incx: Int64; const beta: Procfloat_complex; y: PPProcfloat_complex; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zhbmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n, k: Int64; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: Int64; const x: PPProcdouble_complex; incx: Int64; const beta: Procdouble_complex; y: PPProcdouble_complex; incy, batch_count: Int64): rocblas_status; cdecl; external;

{ hbmv strided batched }
function rocblas_chbmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n, k: rocblas_int; const alpha, A: Procfloat_complex; lda: rocblas_int; strideA: rocblas_stride; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; const beta: Procfloat_complex; y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhbmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n, k: rocblas_int; const alpha, A: Procdouble_complex; lda: rocblas_int; strideA: rocblas_stride; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; const beta: Procdouble_complex; y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ hbmv strided batched 64-bit }
function rocblas_chbmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n, k: Int64; const alpha, A: Procfloat_complex; lda: Int64; strideA: rocblas_stride; const x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; const beta: Procfloat_complex; y: Procfloat_complex; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zhbmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n, k: Int64; const alpha, A: Procdouble_complex; lda: Int64; strideA: rocblas_stride; const x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; const beta: Procdouble_complex; y: Procdouble_complex; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ hpmv: y := alpha*A*x + beta*y }
function rocblas_chpmv(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, AP: Procfloat_complex; const x: Procfloat_complex; incx: rocblas_int; const beta: Procfloat_complex; y: Procfloat_complex; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhpmv(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, AP: Procdouble_complex; const x: Procdouble_complex; incx: rocblas_int; const beta: Procdouble_complex; y: Procdouble_complex; incy: rocblas_int): rocblas_status; cdecl; external;

{ hpmv 64-bit }
function rocblas_chpmv_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, AP: Procfloat_complex; const x: Procfloat_complex; incx: Int64; const beta: Procfloat_complex; y: Procfloat_complex; incy: Int64): rocblas_status; cdecl; external;
function rocblas_zhpmv_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, AP: Procdouble_complex; const x: Procdouble_complex; incx: Int64; const beta: Procdouble_complex; y: Procdouble_complex; incy: Int64): rocblas_status; cdecl; external;

{ hpmv batched }
function rocblas_chpmv_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procfloat_complex; const AP, x: PPProcfloat_complex; incx: rocblas_int; const beta: Procfloat_complex; y: PPProcfloat_complex; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhpmv_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procdouble_complex; const AP, x: PPProcdouble_complex; incx: rocblas_int; const beta: Procdouble_complex; y: PPProcdouble_complex; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ hpmv batched 64-bit }
function rocblas_chpmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procfloat_complex; const AP, x: PPProcfloat_complex; incx: Int64; const beta: Procfloat_complex; y: PPProcfloat_complex; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zhpmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procdouble_complex; const AP, x: PPProcdouble_complex; incx: Int64; const beta: Procdouble_complex; y: PPProcdouble_complex; incy, batch_count: Int64): rocblas_status; cdecl; external;

{ hpmv strided batched }
function rocblas_chpmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, AP: Procfloat_complex; strideA: rocblas_stride; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; const beta: Procfloat_complex; y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhpmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, AP: Procdouble_complex; strideA: rocblas_stride; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; const beta: Procdouble_complex; y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_chpmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, AP: Procfloat_complex; strideA: rocblas_stride; const x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; const beta: Procfloat_complex; y: Procfloat_complex; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zhpmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, AP: Procdouble_complex; strideA: rocblas_stride; const x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; const beta: Procdouble_complex; y: Procdouble_complex; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ symv: y := alpha*A*x + beta*y }
function rocblas_ssymv(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: PSingle; lda: rocblas_int; const x: PSingle; incx: rocblas_int; const beta: PSingle; y: PSingle; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsymv(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: PDouble; lda: rocblas_int; const x: PDouble; incx: rocblas_int; const beta: PDouble; y: PDouble; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csymv(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: Procfloat_complex; lda: rocblas_int; const x: Procfloat_complex; incx: rocblas_int; const beta: Procfloat_complex; y: Procfloat_complex; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsymv(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: Procdouble_complex; lda: rocblas_int; const x: Procdouble_complex; incx: rocblas_int; const beta: Procdouble_complex; y: Procdouble_complex; incy: rocblas_int): rocblas_status; cdecl; external;

{ symv 64-bit }
function rocblas_ssymv_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: PSingle; lda: Int64; const x: PSingle; incx: Int64; const beta: PSingle; y: PSingle; incy: Int64): rocblas_status; cdecl; external;
function rocblas_dsymv_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: PDouble; lda: Int64; const x: PDouble; incx: Int64; const beta: PDouble; y: PDouble; incy: Int64): rocblas_status; cdecl; external;
function rocblas_csymv_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: Procfloat_complex; lda: Int64; const x: Procfloat_complex; incx: Int64; const beta: Procfloat_complex; y: Procfloat_complex; incy: Int64): rocblas_status; cdecl; external;
function rocblas_zsymv_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: Procdouble_complex; lda: Int64; const x: Procdouble_complex; incx: Int64; const beta: Procdouble_complex; y: Procdouble_complex; incy: Int64): rocblas_status; cdecl; external;

{ symv batched }
function rocblas_ssymv_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PSingle; const A: PPSingle; lda: rocblas_int; const x: PPSingle; incx: rocblas_int; const beta: PSingle; y: PPSingle; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsymv_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PDouble; const A: PPDouble; lda: rocblas_int; const x: PPDouble; incx: rocblas_int; const beta: PDouble; y: PPDouble; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csymv_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: rocblas_int; const x: PPProcfloat_complex; incx: rocblas_int; const beta: Procfloat_complex; y: PPProcfloat_complex; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsymv_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: rocblas_int; const x: PPProcdouble_complex; incx: rocblas_int; const beta: Procdouble_complex; y: PPProcdouble_complex; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ symv batched 64-bit }
function rocblas_ssymv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PSingle; const A: PPSingle; lda: Int64; const x: PPSingle; incx: Int64; const beta: PSingle; y: PPSingle; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsymv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PDouble; const A: PPDouble; lda: Int64; const x: PPDouble; incx: Int64; const beta: PDouble; y: PPDouble; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csymv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: Int64; const x: PPProcfloat_complex; incx: Int64; const beta: Procfloat_complex; y: PPProcfloat_complex; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zsymv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: Int64; const x: PPProcdouble_complex; incx: Int64; const beta: Procdouble_complex; y: PPProcdouble_complex; incy, batch_count: Int64): rocblas_status; cdecl; external;

{ symv strided batched }
function rocblas_ssymv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: PSingle; lda: rocblas_int; strideA: rocblas_stride; const x: PSingle; incx: rocblas_int; stridex: rocblas_stride; const beta: PSingle; y: PSingle; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsymv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: PDouble; lda: rocblas_int; strideA: rocblas_stride; const x: PDouble; incx: rocblas_int; stridex: rocblas_stride; const beta: PDouble; y: PDouble; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csymv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: Procfloat_complex; lda: rocblas_int; strideA: rocblas_stride; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; const beta: Procfloat_complex; y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsymv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: Procdouble_complex; lda: rocblas_int; strideA: rocblas_stride; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; const beta: Procdouble_complex; y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ symv strided batched 64-bit }
function rocblas_ssymv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: PSingle; lda: Int64; strideA: rocblas_stride; const x: PSingle; incx: Int64; stridex: rocblas_stride; const beta: PSingle; y: PSingle; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsymv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: PDouble; lda: Int64; strideA: rocblas_stride; const x: PDouble; incx: Int64; stridex: rocblas_stride; const beta: PDouble; y: PDouble; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csymv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: Procfloat_complex; lda: Int64; strideA: rocblas_stride; const x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; const beta: Procfloat_complex; y: Procfloat_complex; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zsymv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: Procdouble_complex; lda: Int64; strideA: rocblas_stride; const x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; const beta: Procdouble_complex; y: Procdouble_complex; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ sbmv: y := alpha*A*x + beta*y }
function rocblas_ssbmv(handle: rocblas_handle; uplo: rocblas_fill; n, k: rocblas_int; const alpha, A: PSingle; lda: rocblas_int; const x: PSingle; incx: rocblas_int; const beta: PSingle; y: PSingle; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsbmv(handle: rocblas_handle; uplo: rocblas_fill; n, k: rocblas_int; const alpha, A: PDouble; lda: rocblas_int; const x: PDouble; incx: rocblas_int; const beta: PDouble; y: PDouble; incy: rocblas_int): rocblas_status; cdecl; external;

{ sbmv 64-bit }
function rocblas_ssbmv_64(handle: rocblas_handle; uplo: rocblas_fill; n, k: Int64; const alpha, A: PSingle; lda: Int64; const x: PSingle; incx: Int64; const beta: PSingle; y: PSingle; incy: Int64): rocblas_status; cdecl; external;
function rocblas_dsbmv_64(handle: rocblas_handle; uplo: rocblas_fill; n, k: Int64; const alpha, A: PDouble; lda: Int64; const x: PDouble; incx: Int64; const beta: PDouble; y: PDouble; incy: Int64): rocblas_status; cdecl; external;

{ sbmv batched }
function rocblas_ssbmv_batched(handle: rocblas_handle; uplo: rocblas_fill; n, k: rocblas_int; const alpha: PSingle; const A: PPSingle; lda: rocblas_int; const x: PPSingle; incx: rocblas_int; const beta: PSingle; y: PPSingle; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsbmv_batched(handle: rocblas_handle; uplo: rocblas_fill; n, k: rocblas_int; const alpha: PDouble; const A: PPDouble; lda: rocblas_int; const x: PPDouble; incx: rocblas_int; const beta: PDouble; y: PPDouble; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ssbmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n, k: Int64; const alpha: PSingle; const A: PPSingle; lda: Int64; const x: PPSingle; incx: Int64; const beta: PSingle; y: PPSingle; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsbmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n, k: Int64; const alpha: PDouble; const A: PPDouble; lda: Int64; const x: PPDouble; incx: Int64; const beta: PDouble; y: PPDouble; incy, batch_count: Int64): rocblas_status; cdecl; external;

{ sbmv strided batched }
function rocblas_ssbmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n, k: rocblas_int; const alpha, A: PSingle; lda: rocblas_int; strideA: rocblas_stride; const x: PSingle; incx: rocblas_int; stridex: rocblas_stride; const beta: PSingle; y: PSingle; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsbmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n, k: rocblas_int; const alpha, A: PDouble; lda: rocblas_int; strideA: rocblas_stride; const x: PDouble; incx: rocblas_int; stridex: rocblas_stride; const beta: PDouble; y: PDouble; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ssbmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n, k: Int64; const alpha, A: PSingle; lda: Int64; strideA: rocblas_stride; const x: PSingle; incx: Int64; stridex: rocblas_stride; const beta: PSingle; y: PSingle; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsbmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n, k: Int64; const alpha, A: PDouble; lda: Int64; strideA: rocblas_stride; const x: PDouble; incx: Int64; stridex: rocblas_stride; const beta: PDouble; y: PDouble; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ spmv scalar }
function rocblas_sspmv(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: PSingle; const x: PSingle; incx: rocblas_int; const beta: PSingle; y: PSingle; incy: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dspmv(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: PDouble; const x: PDouble; incx: rocblas_int; const beta: PDouble; y: PDouble; incy: rocblas_int): rocblas_status; cdecl; external;

{ spmv scalar 64-bit }
function rocblas_sspmv_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: PSingle; const x: PSingle; incx: Int64; const beta: PSingle; y: PSingle; incy: Int64): rocblas_status; cdecl; external;
function rocblas_dspmv_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: PDouble; const x: PDouble; incx: Int64; const beta: PDouble; y: PDouble; incy: Int64): rocblas_status; cdecl; external;

{ spmv batched }
function rocblas_sspmv_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PSingle; const A, x: PPSingle; incx: rocblas_int; const beta: PSingle; y: PPSingle; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dspmv_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PDouble; const A, x: PPDouble; incx: rocblas_int; const beta: PDouble; y: PPDouble; incy, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ spmv batched 64-bit }
function rocblas_sspmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PSingle; const A, x: PPSingle; incx: Int64; const beta: PSingle; y: PPSingle; incy, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dspmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PDouble; const A, x: PPDouble; incx: Int64; const beta: PDouble; y: PPDouble; incy, batch_count: Int64): rocblas_status; cdecl; external;

{ spmv strided-batched }
function rocblas_sspmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: PSingle; strideA: rocblas_stride; const x: PSingle; incx: rocblas_int; stridex: rocblas_stride; const beta: PSingle; y: PSingle; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dspmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, A: PDouble; strideA: rocblas_stride; const x: PDouble; incx: rocblas_int; stridex: rocblas_stride; const beta: PDouble; y: PDouble; incy: rocblas_int; stridey: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ spmv strided-batched 64-bit }
function rocblas_sspmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: PSingle; strideA: rocblas_stride; const x: PSingle; incx: Int64; stridex: rocblas_stride; const beta: PSingle; y: PSingle; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dspmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, A: PDouble; strideA: rocblas_stride; const x: PDouble; incx: Int64; stridex: rocblas_stride; const beta: PDouble; y: PDouble; incy: Int64; stridey: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ tbmv scalar }
function rocblas_stbmv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PSingle; lda: rocblas_int; x: PSingle; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtbmv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PDouble; lda: rocblas_int; x: PDouble; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctbmv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: Procfloat_complex; lda: rocblas_int; x: Procfloat_complex; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztbmv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: Procdouble_complex; lda: rocblas_int; x: Procdouble_complex; incx: rocblas_int): rocblas_status; cdecl; external;

{ tbmv 64-bit scalar }
function rocblas_stbmv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: PSingle; lda: Int64; x: PSingle; incx: Int64): rocblas_status; cdecl; external;
function rocblas_dtbmv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: PDouble; lda: Int64; x: PDouble; incx: Int64): rocblas_status; cdecl; external;
function rocblas_ctbmv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: Procfloat_complex; lda: Int64; x: Procfloat_complex; incx: Int64): rocblas_status; cdecl; external;
function rocblas_ztbmv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: Procdouble_complex; lda: Int64; x: Procdouble_complex; incx: Int64): rocblas_status; cdecl; external;

{ tbmv batched }
function rocblas_stbmv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PPSingle; lda: rocblas_int; x: PPSingle; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtbmv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PPDouble; lda: rocblas_int; x: PPDouble; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctbmv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PPProcfloat_complex; lda: rocblas_int; x: PPProcfloat_complex; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztbmv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PPProcdouble_complex; lda: rocblas_int; x: PPProcdouble_complex; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ tbmv strided batched }
function rocblas_stbmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PSingle; lda: rocblas_int; strideA: rocblas_stride; x: PSingle; incx: rocblas_int; strideX: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtbmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PDouble; lda: rocblas_int; strideA: rocblas_stride; x: PDouble; incx: rocblas_int; strideX: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctbmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: Procfloat_complex; lda: rocblas_int; strideA: rocblas_stride; x: Procfloat_complex; incx: rocblas_int; strideX: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztbmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: Procdouble_complex; lda: rocblas_int; strideA: rocblas_stride; x: Procdouble_complex; incx: rocblas_int; strideX: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ trmv scalar }
function rocblas_strmv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PSingle; lda: rocblas_int; x: PSingle; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtrmv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PDouble; lda: rocblas_int; x: PDouble; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctrmv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: Procfloat_complex; lda: rocblas_int; x: Procfloat_complex; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztrmv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: Procdouble_complex; lda: rocblas_int; x: Procdouble_complex; incx: rocblas_int): rocblas_status; cdecl; external;

{ trmv 64-bit scalar }
function rocblas_strmv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PSingle; lda: Int64; x: PSingle; incx: Int64): rocblas_status; cdecl; external;
function rocblas_dtrmv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PDouble; lda: Int64; x: PDouble; incx: Int64): rocblas_status; cdecl; external;
function rocblas_ctrmv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: Procfloat_complex; lda: Int64; x: Procfloat_complex; incx: Int64): rocblas_status; cdecl; external;
function rocblas_ztrmv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: Procdouble_complex; lda: Int64; x: Procdouble_complex; incx: Int64): rocblas_status; cdecl; external;

{ trmv batched }
function rocblas_strmv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PPSingle; lda: rocblas_int; x: PPSingle; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtrmv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PPDouble; lda: rocblas_int; x: PPDouble; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctrmv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PPProcfloat_complex; lda: rocblas_int; x: PPProcfloat_complex; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztrmv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PPProcdouble_complex; lda: rocblas_int; x: PPProcdouble_complex; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
{ trmv 64-bit batched }
function rocblas_strmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PPSingle; lda: Int64; x: PPSingle; incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtrmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PPDouble; lda: Int64; x: PPDouble; incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctrmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PPProcfloat_complex; lda: Int64; x: PPProcfloat_complex; incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztrmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PPProcdouble_complex; lda: Int64; x: PPProcdouble_complex; incx, batch_count: Int64): rocblas_status; cdecl; external;

{ trmv strided-batched }
function rocblas_strmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PSingle; lda: rocblas_int; stride_A: rocblas_stride; x: PSingle; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtrmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PDouble; lda: rocblas_int; stride_A: rocblas_stride; x: PDouble; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctrmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: Procfloat_complex; lda: rocblas_int; stride_A: rocblas_stride; x: Procfloat_complex; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztrmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: Procdouble_complex; lda: rocblas_int; stride_A: rocblas_stride; x: Procdouble_complex; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ trmv 64-bit strided-batched }
function rocblas_strmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PSingle; lda: Int64; stride_A: rocblas_stride; x: PSingle; incx: Int64; stride_x: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtrmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PDouble; lda: Int64; stride_A: rocblas_stride; x: PDouble; incx: Int64; stride_x: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctrmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: Procfloat_complex; lda: Int64; stride_A: rocblas_stride; x: Procfloat_complex; incx: Int64; stride_x: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztrmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: Procdouble_complex; lda: Int64; stride_A: rocblas_stride; x: Procdouble_complex; incx: Int64; stride_x: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ tpmv: Triangular packed matrix-vector multiplication}
{ tpmv scalar }
function rocblas_stpmv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PSingle; x: PSingle; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtpmv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PDouble; x: PDouble; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctpmv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: Procfloat_complex; x: Procfloat_complex; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztpmv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: Procdouble_complex; x: Procdouble_complex; incx: rocblas_int): rocblas_status; cdecl; external;

{ tpmv 64-bit scalar }
function rocblas_stpmv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PSingle; x: PSingle; incx: Int64): rocblas_status; cdecl; external;
function rocblas_dtpmv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PDouble; x: PDouble; incx: Int64): rocblas_status; cdecl; external;
function rocblas_ctpmv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: Procfloat_complex; x: Procfloat_complex; incx: Int64): rocblas_status; cdecl; external;
function rocblas_ztpmv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: Procdouble_complex; x: Procdouble_complex; incx: Int64): rocblas_status; cdecl; external;

{ tpmv batched }
function rocblas_stpmv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PPSingle; x: PPSingle; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtpmv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PPDouble; x: PPDouble; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctpmv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PPProcfloat_complex; x: PPProcfloat_complex; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztpmv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PPProcdouble_complex; x: PPProcdouble_complex; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ tpmv batched 64-bit }
function rocblas_stpmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PPSingle; x: PPSingle; incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtpmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PPDouble; x: PPDouble; incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctpmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PPProcfloat_complex; x: PPProcfloat_complex; incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztpmv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PPProcdouble_complex; x: PPProcdouble_complex; incx, batch_count: Int64): rocblas_status; cdecl; external;

{ tpmv strided-batched }
function rocblas_stpmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PSingle; stride_A: rocblas_stride; x: PSingle; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtpmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PDouble; stride_A: rocblas_stride; x: PDouble; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctpmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: Procfloat_complex; stride_A: rocblas_stride; x: Procfloat_complex; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztpmv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: Procdouble_complex; stride_A: rocblas_stride; x: Procdouble_complex; incx: rocblas_int; stride_x: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ tpmv strided-batched 64-bit }
function rocblas_stpmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PSingle; stride_A: rocblas_stride; x: PSingle; incx: Int64; stride_x: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtpmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PDouble; stride_A: rocblas_stride; x: PDouble; incx: Int64; stride_x: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctpmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: Procfloat_complex; stride_A: rocblas_stride; x: Procfloat_complex; incx: Int64; stride_x: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztpmv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: Procdouble_complex; stride_A: rocblas_stride; x: Procdouble_complex; incx: Int64; stride_x: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ trsv: A*x = b, or A**T*x = b, or A**H*x = b }
function rocblas_strsv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PSingle; lda: rocblas_int; x: PSingle; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtrsv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PDouble; lda: rocblas_int; x: PDouble; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctrsv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: Procfloat_complex; lda: rocblas_int; x: Procfloat_complex; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztrsv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: Procdouble_complex; lda: rocblas_int; x: Procdouble_complex; incx: rocblas_int): rocblas_status; cdecl; external;

{ trsv 64-bit }
function rocblas_strsv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PSingle; lda: Int64; x: PSingle; incx: Int64): rocblas_status; cdecl; external;
function rocblas_dtrsv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PDouble; lda: Int64; x: PDouble; incx: Int64): rocblas_status; cdecl; external;
function rocblas_ctrsv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: Procfloat_complex; lda: Int64; x: Procfloat_complex; incx: Int64): rocblas_status; cdecl; external;
function rocblas_ztrsv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: Procdouble_complex; lda: Int64; x: Procdouble_complex; incx: Int64): rocblas_status; cdecl; external;

{ trsv batched }
function rocblas_strsv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PPSingle; lda: rocblas_int; x: PPSingle; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtrsv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PPDouble; lda: rocblas_int; x: PPDouble; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctrsv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PPProcfloat_complex; lda: rocblas_int; x: PPProcfloat_complex; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztrsv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PPProcdouble_complex; lda: rocblas_int; x: PPProcdouble_complex; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ trsv batched 64-bit }
function rocblas_strsv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PPSingle; lda: Int64; x: PPSingle; incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtrsv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PPDouble; lda: Int64; x: PPDouble; incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctrsv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PPProcfloat_complex; lda: Int64; x: PPProcfloat_complex; incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztrsv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PPProcdouble_complex; lda: Int64; x: PPProcdouble_complex; incx, batch_count: Int64): rocblas_status; cdecl; external;

{ trsv strided batched }
function rocblas_strsv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PSingle; lda: rocblas_int; strideA: rocblas_stride; x: PSingle; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtrsv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: PDouble; lda: rocblas_int; strideA: rocblas_stride; x: PDouble; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctrsv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: Procfloat_complex; lda: rocblas_int; strideA: rocblas_stride; x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztrsv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const A: Procdouble_complex; lda: rocblas_int; strideA: rocblas_stride; x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ trsv strided batched 64-bit }
function rocblas_strsv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PSingle; lda: Int64; strideA: rocblas_stride; x: PSingle; incx: Int64; stridex: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtrsv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: PDouble; lda: Int64; strideA: rocblas_stride; x: PDouble; incx: Int64; stridex: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctrsv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: Procfloat_complex; lda: Int64; strideA: rocblas_stride; x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztrsv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const A: Procdouble_complex; lda: Int64; strideA: rocblas_stride; x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ tpsv: A*x = b, or A**T*x = b, or A**H*x = b }
function rocblas_stpsv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const AP, x: PSingle; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtpsv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const AP, x: PDouble; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctpsv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const AP, x: Procfloat_complex; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztpsv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const AP, x: Procdouble_complex; incx: rocblas_int): rocblas_status; cdecl; external;

{ tpsv 64-bit }
function rocblas_stpsv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const AP, x: PSingle; incx: Int64): rocblas_status; cdecl; external;
function rocblas_dtpsv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const AP, x: PDouble; incx: Int64): rocblas_status; cdecl; external;
function rocblas_ctpsv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const AP, x: Procfloat_complex; incx: Int64): rocblas_status; cdecl; external;
function rocblas_ztpsv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const AP, x: Procdouble_complex; incx: Int64): rocblas_status; cdecl; external;

{ tpsv batched }
function rocblas_stpsv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const AP, x: PPSingle; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtpsv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const AP, x: PPDouble; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctpsv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const AP, x: PPProcfloat_complex; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztpsv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const AP, x: PPProcdouble_complex; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ tpsv batched 64-bit }
function rocblas_stpsv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const AP, x: PPPSingle; incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtpsv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const AP, x: PPDouble;  incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctpsv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const AP, x: PPProcfloat_complex;  incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztpsv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const AP, x: PPProcdouble_complex; incx, batch_count: Int64): rocblas_status; cdecl; external;

{ tpsv strided batched }
function rocblas_stpsv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const AP: PSingle; strideA: rocblas_stride; x: PSingle; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtpsv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const AP: PDouble; strideA: rocblas_stride; x: PDouble; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctpsv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const AP: Procfloat_complex; strideA: rocblas_stride; x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztpsv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: rocblas_int; const AP: Procdouble_complex; strideA: rocblas_stride; x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ tpsv strided batched 64-bit }
function rocblas_stpsv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const AP: PSingle; strideA: rocblas_stride; x: PSingle; incx: Int64; stridex: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtpsv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const AP: PDouble; strideA: rocblas_stride; x: PDouble; incx: Int64; stridex: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctpsv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const AP: Procfloat_complex; strideA: rocblas_stride; x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztpsv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n: Int64; const AP: Procdouble_complex; strideA: rocblas_stride; x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ tbsv: A*x = b, or A**T*x = b, or A**H*x = b }
function rocblas_stbsv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PSingle; lda: rocblas_int; x: PSingle; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtbsv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PDouble; lda: rocblas_int; x: PDouble; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctbsv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: Procfloat_complex; lda: rocblas_int; x: Procfloat_complex; incx: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztbsv(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: Procdouble_complex; lda: rocblas_int; x: Procdouble_complex; incx: rocblas_int): rocblas_status; cdecl; external;

{ tbsv 64-bit }
function rocblas_stbsv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: PSingle; lda: Int64; x: PSingle; incx: Int64): rocblas_status; cdecl; external;
function rocblas_dtbsv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: PDouble; lda: Int64; x: PDouble; incx: Int64): rocblas_status; cdecl; external;
function rocblas_ctbsv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: Procfloat_complex; lda: Int64; x: Procfloat_complex; incx: Int64): rocblas_status; cdecl; external;
function rocblas_ztbsv_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: Procdouble_complex; lda: Int64; x: Procdouble_complex; incx: Int64): rocblas_status; cdecl; external;

{ tbsv batched }
function rocblas_stbsv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PPSingle; lda: rocblas_int; x: PPSingle; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtbsv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PPDouble; lda: rocblas_int; x: PPDouble; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctbsv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PPProcfloat_complex; lda: rocblas_int; x: PPProcfloat_complex; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztbsv_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PPProcdouble_complex; lda: rocblas_int; x: PPProcdouble_complex; incx, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ tbsv batched 64-bit }
function rocblas_stbsv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: PPSingle; lda: Int64; x: PPSingle; incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtbsv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: PPDouble; lda: Int64; x: PPDouble; incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctbsv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: PPProcfloat_complex; lda: Int64; x: PPProcfloat_complex; incx, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztbsv_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: PPProcdouble_complex; lda: Int64; x: PPProcdouble_complex; incx, batch_count: Int64): rocblas_status; cdecl; external;

{ tbsv strided batched }
function rocblas_stbsv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PSingle; lda: rocblas_int; strideA: rocblas_stride; x: PSingle; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtbsv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: PDouble; lda: rocblas_int; strideA: rocblas_stride; x: PDouble; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctbsv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: Procfloat_complex; lda: rocblas_int; strideA: rocblas_stride; x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztbsv_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: rocblas_int; const A: Procdouble_complex; lda: rocblas_int; strideA: rocblas_stride; x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ tbsv strided batched 64-bit }
function rocblas_stbsv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: PSingle; lda: Int64; strideA: rocblas_stride; x: PSingle; incx: Int64; stridex: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtbsv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: PDouble; lda: Int64; strideA: rocblas_stride; x: PDouble; incx: Int64; stridex: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctbsv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: Procfloat_complex; lda: Int64; strideA: rocblas_stride; x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztbsv_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; n, k: Int64; const A: Procdouble_complex; lda: Int64; strideA: rocblas_stride; x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ ger: A := alpha*x*y**T + A }
function rocblas_sger(handle: rocblas_handle; m, n: rocblas_int; const alpha, x: PSingle; incx: rocblas_int; const y: PSingle; incy: rocblas_int; A: PSingle; lda: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dger(handle: rocblas_handle; m, n: rocblas_int; const alpha, x: PDouble; incx: rocblas_int; const y: PDouble; incy: rocblas_int; A: PDouble; lda: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgeru(handle: rocblas_handle; m, n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int; const y: Procfloat_complex; incy: rocblas_int; A: Procfloat_complex; lda: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgeru(handle: rocblas_handle; m, n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int; const y: Procdouble_complex; incy: rocblas_int; A: Procdouble_complex; lda: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgerc(handle: rocblas_handle; m, n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int; const y: Procfloat_complex; incy: rocblas_int; A: Procfloat_complex; lda: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgerc(handle: rocblas_handle; m, n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int; const y: Procdouble_complex; incy: rocblas_int; A: Procdouble_complex; lda: rocblas_int): rocblas_status; cdecl; external;

{ ger 64-bit }
function rocblas_sger_64(handle: rocblas_handle; m, n: Int64; const alpha, x: PSingle; incx: Int64; const y: PSingle; incy: Int64; A: PSingle; lda: Int64): rocblas_status; cdecl; external;
function rocblas_dger_64(handle: rocblas_handle; m, n: Int64; const alpha, x: PDouble; incx: Int64; const y: PDouble; incy: Int64; A: PDouble; lda: Int64): rocblas_status; cdecl; external;
function rocblas_cgeru_64(handle: rocblas_handle; m, n: Int64; const alpha, x: Procfloat_complex; incx: Int64; const y: Procfloat_complex; incy: Int64; A: Procfloat_complex; lda: Int64): rocblas_status; cdecl; external;
function rocblas_zgeru_64(handle: rocblas_handle; m, n: Int64; const alpha, x: Procdouble_complex; incx: Int64; const y: Procdouble_complex; incy: Int64; A: Procdouble_complex; lda: Int64): rocblas_status; cdecl; external;
function rocblas_cgerc_64(handle: rocblas_handle; m, n: Int64; const alpha, x: Procfloat_complex; incx: Int64; const y: Procfloat_complex; incy: Int64; A: Procfloat_complex; lda: Int64): rocblas_status; cdecl; external;
function rocblas_zgerc_64(handle: rocblas_handle; m, n: Int64; const alpha, x: Procdouble_complex; incx: Int64; const y: Procdouble_complex; incy: Int64; A: Procdouble_complex; lda: Int64): rocblas_status; cdecl; external;

{ ger batched }
function rocblas_sger_batched(handle: rocblas_handle; m, n: rocblas_int; const alpha: PSingle; const x: PPSingle; incx: rocblas_int; const y: PPSingle; incy: rocblas_int; A: PPSingle; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dger_batched(handle: rocblas_handle; m, n: rocblas_int; const alpha: PDouble; const x: PPDouble; incx: rocblas_int; const y: PPDouble; incy: rocblas_int; A: PPDouble; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgeru_batched(handle: rocblas_handle; m, n: rocblas_int; const alpha: Procfloat_complex; const x: PPProcfloat_complex; incx: rocblas_int; const y: PPProcfloat_complex; incy: rocblas_int; A: PPProcfloat_complex; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgeru_batched(handle: rocblas_handle; m, n: rocblas_int; const alpha: Procdouble_complex; const x: PPProcdouble_complex; incx: rocblas_int; const y: PPProcdouble_complex; incy: rocblas_int; A: PPProcdouble_complex; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgerc_batched(handle: rocblas_handle; m, n: rocblas_int; const alpha: Procfloat_complex; const x: PPProcfloat_complex; incx: rocblas_int; const y: PPProcfloat_complex; incy: rocblas_int; A: PPProcfloat_complex; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgerc_batched(handle: rocblas_handle; m, n: rocblas_int; const alpha: Procdouble_complex; const x: PPProcdouble_complex; incx: rocblas_int; const y: PPProcdouble_complex; incy: rocblas_int; A: PPProcdouble_complex; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ ger batched 64-bit }
function rocblas_sger_batched_64(handle: rocblas_handle; m, n: Int64; const alpha: PSingle; const x: PPSingle; incx: Int64; const y: PPSingle; incy: Int64; A: PPSingle; lda, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dger_batched_64(handle: rocblas_handle; m, n: Int64; const alpha: PDouble; const x: PPDouble; incx: Int64; const y: PPDouble; incy: Int64; A: PPDouble; lda, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_cgeru_batched_64(handle: rocblas_handle; m, n: Int64; const alpha: Procfloat_complex; const x: PPProcfloat_complex; incx: Int64; const y: PPProcfloat_complex; incy: Int64; A: PPProcfloat_complex; lda, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zgeru_batched_64(handle: rocblas_handle; m, n: Int64; const alpha: Procdouble_complex; const x: PPProcdouble_complex; incx: Int64; const y: PPProcdouble_complex; incy: Int64; A: PPProcdouble_complex; lda, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_cgerc_batched_64(handle: rocblas_handle; m, n: Int64; const alpha: Procfloat_complex; const x: PPProcfloat_complex; incx: Int64; const y: PPProcfloat_complex; incy: Int64; A: PPProcfloat_complex; lda, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zgerc_batched_64(handle: rocblas_handle; m, n: Int64; const alpha: Procdouble_complex; const x: PPProcdouble_complex; incx: Int64; const y: PPProcdouble_complex; incy: Int64; A: PPProcdouble_complex; lda, batch_count: Int64): rocblas_status; cdecl; external;

{ ger strided batched }
function rocblas_sger_strided_batched(handle: rocblas_handle; m, n: rocblas_int; const alpha, x: PSingle; incx: rocblas_int; stridex: rocblas_stride; const y: PSingle; incy: rocblas_int; stridey: rocblas_stride; A: PSingle; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dger_strided_batched(handle: rocblas_handle; m, n: rocblas_int; const alpha, x: PDouble; incx: rocblas_int; stridex: rocblas_stride; const y: PDouble; incy: rocblas_int; stridey: rocblas_stride; A: PDouble; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgeru_strided_batched(handle: rocblas_handle; m, n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; A: Procfloat_complex; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgeru_strided_batched(handle: rocblas_handle; m, n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; A: Procdouble_complex; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgerc_strided_batched(handle: rocblas_handle; m, n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; A: Procfloat_complex; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgerc_strided_batched(handle: rocblas_handle; m, n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; A: Procdouble_complex; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ ger strided batched 64-bit }
function rocblas_sger_strided_batched_64(handle: rocblas_handle; m, n: Int64; const alpha, x: PSingle; incx: Int64; stridex: rocblas_stride; const y: PSingle; incy: Int64; stridey: rocblas_stride; A: PSingle; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dger_strided_batched_64(handle: rocblas_handle; m, n: Int64; const alpha, x: PDouble; incx: Int64; stridex: rocblas_stride; const y: PDouble; incy: Int64; stridey: rocblas_stride; A: PDouble; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_cgeru_strided_batched_64(handle: rocblas_handle; m, n: Int64; const alpha, x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; const y: Procfloat_complex; incy: Int64; stridey: rocblas_stride; A: Procfloat_complex; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zgeru_strided_batched_64(handle: rocblas_handle; m, n: Int64; const alpha, x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; const y: Procdouble_complex; incy: Int64; stridey: rocblas_stride; A: Procdouble_complex; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_cgerc_strided_batched_64(handle: rocblas_handle; m, n: Int64; const alpha, x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; const y: Procfloat_complex; incy: Int64; stridey: rocblas_stride; A: Procfloat_complex; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zgerc_strided_batched_64(handle: rocblas_handle; m, n: Int64; const alpha, x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; const y: Procdouble_complex; incy: Int64; stridey: rocblas_stride; A: Procdouble_complex; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ her: A := alpha*x*x**H + A }
function rocblas_cher(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PSingle; const x: Procfloat_complex; incx: rocblas_int; A: Procfloat_complex; lda: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zher(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PDouble; const x: Procdouble_complex; incx: rocblas_int; A: Procdouble_complex; lda: rocblas_int): rocblas_status; cdecl; external;

{ her 64-bit }
function rocblas_cher_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PSingle; const x: Procfloat_complex; incx: Int64; A: Procfloat_complex; lda: Int64): rocblas_status; cdecl; external;
function rocblas_zher_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PDouble; const x: Procdouble_complex; incx: Int64; A: Procdouble_complex; lda: Int64): rocblas_status; cdecl; external;

{ her batched }
function rocblas_cher_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PSingle; const x: PPProcfloat_complex; incx: rocblas_int; A: PPProcfloat_complex; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zher_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PDouble; const x: PPProcdouble_complex; incx: rocblas_int; A: PPProcdouble_complex; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cher_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PSingle; const x: PPProcfloat_complex; incx: Int64; A: PPProcfloat_complex; lda, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zher_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PDouble; const x: PPProcdouble_complex; incx: Int64; A: PPProcdouble_complex; lda, batch_count: Int64): rocblas_status; cdecl; external;

{ her strided batched }
function rocblas_cher_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PSingle; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; A: Procfloat_complex; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zher_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PDouble; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; A: Procdouble_complex; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cher_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PSingle; const x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; A: Procfloat_complex; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zher_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PDouble; const x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; A: Procdouble_complex; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ hpr: A := alpha*x*x**H + A }
function rocblas_chpr(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PSingle; const x: Procfloat_complex; incx: rocblas_int; AP: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zhpr(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PDouble; const x: Procdouble_complex; incx: rocblas_int; AP: Procdouble_complex): rocblas_status; cdecl; external;

{ hpr 64-bit }
function rocblas_chpr_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PSingle; const x: Procfloat_complex; incx: Int64; AP: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zhpr_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PDouble; const x: Procdouble_complex; incx: Int64; AP: Procdouble_complex): rocblas_status; cdecl; external;

{ hpr batched }
function rocblas_chpr_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PSingle; const x: PPProcfloat_complex; incx: rocblas_int; AP: PPProcfloat_complex; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhpr_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PDouble; const x: PPProcdouble_complex; incx: rocblas_int; AP: PPProcdouble_complex; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_chpr_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PSingle; const x: PPProcfloat_complex; incx: Int64; AP: PPProcfloat_complex; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zhpr_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PDouble; const x: PPProcdouble_complex; incx: Int64; AP: PPProcdouble_complex; batch_count: Int64): rocblas_status; cdecl; external;

{ hpr strided batched }
function rocblas_chpr_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PSingle; const x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; AP: Procfloat_complex; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhpr_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PDouble; const x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; AP: Procdouble_complex; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_chpr_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PSingle; const x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; AP: Procfloat_complex; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zhpr_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PDouble; const x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; AP: Procdouble_complex; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ syr: A := alpha*x*x**T + A }
function rocblas_ssyr(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PSingle; incx: rocblas_int; A: PSingle; lda: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsyr(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PDouble; incx: rocblas_int; A: PDouble; lda: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csyr(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int; A: Procfloat_complex; lda: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsyr(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int; A: Procdouble_complex; lda: rocblas_int): rocblas_status; cdecl; external;

{ syr 64-bit }
function rocblas_ssyr_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PSingle; incx: Int64; A: PSingle; lda: Int64): rocblas_status; cdecl; external;
function rocblas_dsyr_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PDouble; incx: Int64; A: PDouble; lda: Int64): rocblas_status; cdecl; external;
function rocblas_csyr_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procfloat_complex; incx: Int64; A: Procfloat_complex; lda: Int64): rocblas_status; cdecl; external;
function rocblas_zsyr_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procdouble_complex; incx: Int64; A: Procdouble_complex; lda: Int64): rocblas_status; cdecl; external;

{ syr batched }
function rocblas_ssyr_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PSingle; const x: PPSingle; incx: rocblas_int; A: PPSingle; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsyr_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PDouble; const x: PPDouble; incx: rocblas_int; A: PPDouble; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csyr_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procfloat_complex; const x: PPProcfloat_complex; incx: rocblas_int; A: PPProcfloat_complex; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsyr_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procdouble_complex; const x: PPProcdouble_complex; incx: rocblas_int; A: PPProcdouble_complex; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ syr batched 64-bit }
function rocblas_ssyr_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PSingle; const x: PPSingle; incx: Int64; A: PPSingle; lda, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsyr_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PDouble; const x: PPDouble; incx: Int64; A: PPDouble; lda, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csyr_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procfloat_complex; const x: PPProcfloat_complex; incx: Int64; A: PPProcfloat_complex; lda, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zsyr_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procdouble_complex; const x: PPProcdouble_complex; incx: Int64; A: PPProcdouble_complex; lda, batch_count: Int64): rocblas_status; cdecl; external;

{ syr strided batched }
function rocblas_ssyr_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PSingle; incx: rocblas_int; stridex: rocblas_stride; A: PSingle; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsyr_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PDouble; incx: rocblas_int; stridex: rocblas_stride; A: PDouble; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csyr_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; A: Procfloat_complex; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsyr_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; A: Procdouble_complex; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ syr strided batched 64-bit }
function rocblas_ssyr_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PSingle; incx: Int64; stridex: rocblas_stride; A: PSingle; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsyr_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PDouble; incx: Int64; stridex: rocblas_stride; A: PDouble; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csyr_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; A: Procfloat_complex; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zsyr_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; A: Procdouble_complex; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ spr: A := alpha*x*x**T + A }
function rocblas_sspr(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PSingle; incx: rocblas_int; AP: PSingle): rocblas_status; cdecl; external;
function rocblas_dspr(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PDouble; incx: rocblas_int; AP: PDouble): rocblas_status; cdecl; external;

{ spr 64-bit }
function rocblas_sspr_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PSingle; incx: Int64; AP: PSingle): rocblas_status; cdecl; external;
function rocblas_dspr_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PDouble; incx: Int64; AP: PDouble): rocblas_status; cdecl; external;

{ spr batched }
function rocblas_sspr_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PSingle; const x: PPSingle; incx: rocblas_int; AP: PPSingle; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dspr_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PDouble; const x: PPDouble; incx: rocblas_int; AP: PPDouble; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_sspr_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PSingle; const x: PPSingle; incx: Int64; AP: PPSingle; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dspr_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PDouble; const x: PPDouble; incx: Int64; AP: PPDouble; batch_count: Int64): rocblas_status; cdecl; external;

{ spr strided batched }
function rocblas_sspr_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PSingle; incx: rocblas_int; stridex: rocblas_stride; AP: PSingle; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dspr_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PDouble; incx: rocblas_int; stridex: rocblas_stride; AP: PDouble; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_sspr_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PSingle; incx: Int64; stridex: rocblas_stride; AP: PSingle; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dspr_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PDouble; incx: Int64; stridex: rocblas_stride; AP: PDouble; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;



{ her2: A := alpha*x*y**H + conj(alpha)*y*x**H + A }
function rocblas_cher2(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int; const y: Procfloat_complex; incy: rocblas_int; A: Procfloat_complex; lda: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zher2(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int; const y: Procdouble_complex; incy: rocblas_int; A: Procdouble_complex; lda: rocblas_int): rocblas_status; cdecl; external;

{ her2 64-bit }
function rocblas_cher2_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procfloat_complex; incx: Int64; const y: Procfloat_complex; incy: Int64; A: Procfloat_complex; lda: Int64): rocblas_status; cdecl; external;
function rocblas_zher2_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procdouble_complex; incx: Int64; const y: Procdouble_complex; incy: Int64; A: Procdouble_complex; lda: Int64): rocblas_status; cdecl; external;

{ her2 batched }
function rocblas_cher2_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procfloat_complex; const x: PPProcfloat_complex; incx: rocblas_int; const y: PPProcfloat_complex; incy: rocblas_int; A: PPProcfloat_complex; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zher2_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procdouble_complex; const x: PPProcdouble_complex; incx: rocblas_int; const y: PPProcdouble_complex; incy: rocblas_int; A: PPProcdouble_complex; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ her2 batched 64-bit }
function rocblas_cher2_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procfloat_complex; const x: PPProcfloat_complex; incx: Int64; const y: PPProcfloat_complex; incy: Int64; A: PPProcfloat_complex; lda, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zher2_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procdouble_complex; const x: PPProcdouble_complex; incx: Int64; const y: PPProcdouble_complex; incy: Int64; A: PPProcdouble_complex; lda, batch_count: Int64): rocblas_status; cdecl; external;

{ her2 strided batched }
function rocblas_cher2_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; A: Procfloat_complex; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zher2_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; A: Procdouble_complex; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ her2 strided batched 64-bit }
function rocblas_cher2_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; const y: Procfloat_complex; incy: Int64; stridey: rocblas_stride; A: Procfloat_complex; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zher2_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; const y: Procdouble_complex; incy: Int64; stridey: rocblas_stride; A: Procdouble_complex; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ hpr2: A := alpha*x*y**H + conj(alpha)*y*x**H + A }
function rocblas_chpr2(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int; const y: Procfloat_complex; incy: rocblas_int; AP: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zhpr2(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int; const y: Procdouble_complex; incy: rocblas_int; AP: Procdouble_complex): rocblas_status; cdecl; external;

{ hpr2 64-bit }
function rocblas_chpr2_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procfloat_complex; incx: Int64; const y: Procfloat_complex; incy: Int64; AP: Procfloat_complex): rocblas_status; cdecl; external;
function rocblas_zhpr2_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procdouble_complex; incx: Int64; const y: Procdouble_complex; incy: Int64; AP: Procdouble_complex): rocblas_status; cdecl; external;

{ hpr2 batched }
function rocblas_chpr2_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procfloat_complex; const x: PPProcfloat_complex; incx: rocblas_int; const y: PPProcfloat_complex; incy: rocblas_int; AP: PPProcfloat_complex; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhpr2_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procdouble_complex; const x: PPProcdouble_complex; incx: rocblas_int; const y: PPProcdouble_complex; incy: rocblas_int; AP: PPProcdouble_complex; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ hpr2 batched 64-bit }
function rocblas_chpr2_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procfloat_complex; const x: PPProcfloat_complex; incx: Int64; const y: PPProcfloat_complex; incy: Int64; AP: PPProcfloat_complex; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zhpr2_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procdouble_complex; const x: PPProcdouble_complex; incx: Int64; const y: PPProcdouble_complex; incy: Int64; AP: PPProcdouble_complex; batch_count: Int64): rocblas_status; cdecl; external;

{ hpr2 strided batched }
function rocblas_chpr2_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; AP: Procfloat_complex; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhpr2_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; AP: Procdouble_complex; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ hpr2 strided batched 64-bit }
function rocblas_chpr2_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; const y: Procfloat_complex; incy: Int64; stridey: rocblas_stride; AP: Procfloat_complex; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zhpr2_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; const y: Procdouble_complex; incy: Int64; stridey: rocblas_stride; AP: Procdouble_complex; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ spr2: A := alpha*x*y**T + alpha*y*x**T + A }
function rocblas_sspr2(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PSingle; incx: rocblas_int; const y: PSingle; incy: rocblas_int; AP: PSingle): rocblas_status; cdecl; external;
function rocblas_dspr2(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PDouble; incx: rocblas_int; const y: PDouble; incy: rocblas_int; AP: PDouble): rocblas_status; cdecl; external;

{ spr2 64-bit }
function rocblas_sspr2_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PSingle; incx: Int64; const y: PSingle; incy: Int64; AP: PSingle): rocblas_status; cdecl; external;
function rocblas_dspr2_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PDouble; incx: Int64; const y: PDouble; incy: Int64; AP: PDouble): rocblas_status; cdecl; external;

{ spr2 batched }
function rocblas_sspr2_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PSingle; const x: PPSingle; incx: rocblas_int; const y: PPSingle; incy: rocblas_int; AP: PPSingle; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dspr2_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PDouble; const x: PPDouble; incx: rocblas_int; const y: PPDouble; incy: rocblas_int; AP: PPDouble; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ spr2 batched 64-bit }
function rocblas_sspr2_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PSingle; const x: PPSingle; incx: Int64; const y: PPSingle; incy: Int64; AP: PPSingle; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dspr2_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PDouble; const x: PPDouble; incx: Int64; const y: PPDouble; incy: Int64; AP: PPDouble; batch_count: Int64): rocblas_status; cdecl; external;

{ spr2 strided batched }
function rocblas_sspr2_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PSingle; incx: rocblas_int; stridex: rocblas_stride; const y: PSingle; incy: rocblas_int; stridey: rocblas_stride; AP: PSingle; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dspr2_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PDouble; incx: rocblas_int; stridex: rocblas_stride; const y: PDouble; incy: rocblas_int; stridey: rocblas_stride; AP: PDouble; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ spr2 strided batched 64-bit }
function rocblas_sspr2_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PSingle; incx: Int64; stridex: rocblas_stride; const y: PSingle; incy: Int64; stridey: rocblas_stride; AP: PSingle; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dspr2_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PDouble; incx: Int64; stridex: rocblas_stride; const y: PDouble; incy: Int64; stridey: rocblas_stride; AP: PDouble; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ syr2: A := alpha*x*y**T + alpha*y*x**T + A }
function rocblas_ssyr2(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PSingle; incx: rocblas_int; const y: PSingle; incy: rocblas_int; A: PSingle; lda: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsyr2(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PDouble; incx: rocblas_int; const y: PDouble; incy: rocblas_int; A: PDouble; lda: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csyr2(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int; const y: Procfloat_complex; incy: rocblas_int; A: Procfloat_complex; lda: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsyr2(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int; const y: Procdouble_complex; incy: rocblas_int; A: Procdouble_complex; lda: rocblas_int): rocblas_status; cdecl; external;

{ syr2 64-bit }
function rocblas_ssyr2_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PSingle; incx: Int64; const y: PSingle; incy: Int64; A: PSingle; lda: Int64): rocblas_status; cdecl; external;
function rocblas_dsyr2_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PDouble; incx: Int64; const y: PDouble; incy: Int64; A: PDouble; lda: Int64): rocblas_status; cdecl; external;
function rocblas_csyr2_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procfloat_complex; incx: Int64; const y: Procfloat_complex; incy: Int64; A: Procfloat_complex; lda: Int64): rocblas_status; cdecl; external;
function rocblas_zsyr2_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procdouble_complex; incx: Int64; const y: Procdouble_complex; incy: Int64; A: Procdouble_complex; lda: Int64): rocblas_status; cdecl; external;

{ syr2 batched }
function rocblas_ssyr2_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PSingle; const x: PPSingle; incx: rocblas_int; const y: PPSingle; incy: rocblas_int; A: PPSingle; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsyr2_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: PDouble; const x: PPDouble; incx: rocblas_int; const y: PPDouble; incy: rocblas_int; A: PPDouble; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csyr2_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procfloat_complex; const x: PPProcfloat_complex; incx: rocblas_int; const y: PPProcfloat_complex; incy: rocblas_int; A: PPProcfloat_complex; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsyr2_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha: Procdouble_complex; const x: PPProcdouble_complex; incx: rocblas_int; const y: PPProcdouble_complex; incy: rocblas_int; A: PPProcdouble_complex; lda, batch_count: rocblas_int): rocblas_status; cdecl; external;

{ syr2 batched 64-bit }
function rocblas_ssyr2_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PSingle; const x: PPSingle; incx: Int64; const y: PPSingle; incy: Int64; A: PPSingle; lda, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsyr2_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: PDouble; const x: PPDouble; incx: Int64; const y: PPDouble; incy: Int64; A: PPDouble; lda, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csyr2_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procfloat_complex; const x: PPProcfloat_complex; incx: Int64; const y: PPProcfloat_complex; incy: Int64; A: PPProcfloat_complex; lda, batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zsyr2_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha: Procdouble_complex; const x: PPProcdouble_complex; incx: Int64; const y: PPProcdouble_complex; incy: Int64; A: PPProcdouble_complex; lda, batch_count: Int64): rocblas_status; cdecl; external;

{ syr2 strided batched }
function rocblas_ssyr2_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PSingle; incx: rocblas_int; stridex: rocblas_stride; const y: PSingle; incy: rocblas_int; stridey: rocblas_stride; A: PSingle; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsyr2_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: PDouble; incx: rocblas_int; stridex: rocblas_stride; const y: PDouble; incy: rocblas_int; stridey: rocblas_stride; A: PDouble; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csyr2_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procfloat_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procfloat_complex; incy: rocblas_int; stridey: rocblas_stride; A: Procfloat_complex; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsyr2_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; n: rocblas_int; const alpha, x: Procdouble_complex; incx: rocblas_int; stridex: rocblas_stride; const y: Procdouble_complex; incy: rocblas_int; stridey: rocblas_stride; A: Procdouble_complex; lda: rocblas_int; strideA: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ syr2 strided batched 64-bit }
function rocblas_ssyr2_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PSingle; incx: Int64; stridex: rocblas_stride; const y: PSingle; incy: Int64; stridey: rocblas_stride; A: PSingle; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsyr2_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: PDouble; incx: Int64; stridex: rocblas_stride; const y: PDouble; incy: Int64; stridey: rocblas_stride; A: PDouble; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csyr2_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procfloat_complex; incx: Int64; stridex: rocblas_stride; const y: Procfloat_complex; incy: Int64; stridey: rocblas_stride; A: Procfloat_complex; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zsyr2_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; n: Int64; const alpha, x: Procdouble_complex; incx: Int64; stridex: rocblas_stride; const y: Procdouble_complex; incy: Int64; stridey: rocblas_stride; A: Procdouble_complex; lda: Int64; strideA: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ ============================================================================================ }
{   Level 3 BLAS functions                                                                     }
{ ============================================================================================ }

{ ---------- GEMM ---------- }
{ scalar }
function rocblas_sgemm(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: PSingle; const A: PSingle; lda: rocblas_int; const B: PSingle; ldb: rocblas_int; const beta: PSingle; C: PSingle; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dgemm(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: PDouble; const A: PDouble; lda: rocblas_int; const B: PDouble; ldb: rocblas_int; const beta: PDouble; C: PDouble; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgemm(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; const B: Procfloat_complex; ldb: rocblas_int; const beta: Procfloat_complex; C: Procfloat_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgemm(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; const B: Procdouble_complex; ldb: rocblas_int; const beta: Procdouble_complex; C: Procdouble_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit scalar }
function rocblas_sgemm_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: PSingle; const A: PSingle; lda: Int64; const B: PSingle; ldb: Int64; const beta: PSingle; C: PSingle; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_dgemm_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: PDouble; const A: PDouble; lda: Int64; const B: PDouble; ldb: Int64; const beta: PDouble; C: PDouble; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_cgemm_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; const B: Procfloat_complex; ldb: Int64; const beta: Procfloat_complex; C: Procfloat_complex; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_zgemm_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; const B: Procdouble_complex; ldb: Int64; const beta: Procdouble_complex; C: Procdouble_complex; ldc: Int64): rocblas_status; cdecl; external;

{ ---------- GEMM_EX (mixed-precision, separate input/output buffers) ---------- }
{ NOTE: ROCm >= 6.4/7.x signature includes a SEPARATE output matrix D with its own ldd.
  Older bindings that omitted d/d_type/ldd silently shifted every argument from C onward
  by 3 slots -> garbage results (e.g. FP16 matmul returned 0.0889 instead of 0.5).
  The actual function has 23 params; the C macro wrapper appends obsolete
  workspace_size/workspace as variadic args that are ignored by the callee.
  We omit those trailing params — x86-64 ABI ignores extra caller-passed values. }
function rocblas_gemm_ex(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: Pointer; const A: Pointer; a_type: rocblas_datatype; lda: rocblas_int; const B: Pointer; b_type: rocblas_datatype; ldb: rocblas_int; const beta: Pointer; const C: Pointer; c_type: rocblas_datatype; ldc: rocblas_int; D: Pointer; d_type: rocblas_datatype; ldd: rocblas_int; compute_type: rocblas_datatype; algo: rocblas_gemm_algo; solution_index: Integer; flags: Cardinal): rocblas_status; cdecl; external;
function rocblas_gemm_ex_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: Pointer; const A: Pointer; a_type: rocblas_datatype; lda: Int64; const B: Pointer; b_type: rocblas_datatype; ldb: Int64; const beta: Pointer; const C: Pointer; c_type: rocblas_datatype; ldc: Int64; D: Pointer; d_type: rocblas_datatype; ldd: Int64; compute_type: rocblas_datatype; algo: rocblas_gemm_algo; solution_index: Integer; flags: Cardinal): rocblas_status; cdecl; external;
function rocblas_gemm_batched_ex(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: Pointer; const A: Pointer; a_type: rocblas_datatype; lda: rocblas_int; const B: Pointer; b_type: rocblas_datatype; ldb: rocblas_int; const beta: Pointer; const C: Pointer; c_type: rocblas_datatype; ldc: rocblas_int; D: Pointer; d_type: rocblas_datatype; ldd: rocblas_int; batch_count: rocblas_int; compute_type: rocblas_datatype; algo: rocblas_gemm_algo; solution_index: Integer; flags: Cardinal): rocblas_status; cdecl; external;

{ Strided batched GEMM: each batch matrix is at base + batch*stride.
  This is the natural primitive for attention (Q@K^T, Attn@V) — the
  batch stride is just S*head_dim elements, no pointer arrays needed. }
function rocblas_gemm_strided_batched_ex(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: Pointer; const A: Pointer; a_type: rocblas_datatype; lda: rocblas_int; stride_a: rocblas_stride; const B: Pointer; b_type: rocblas_datatype; ldb: rocblas_int; stride_b: rocblas_stride; const beta: Pointer; const C: Pointer; c_type: rocblas_datatype; ldc: rocblas_int; stride_c: rocblas_stride; D: Pointer; d_type: rocblas_datatype; ldd: rocblas_int; stride_d: rocblas_stride; batch_count: rocblas_int; compute_type: rocblas_datatype; algo: rocblas_gemm_algo; solution_index: Integer; flags: Cardinal): rocblas_status; cdecl; external;

function rocblas_gemm_strided_batched_ex_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: Pointer; const A: Pointer; a_type: rocblas_datatype; lda: Int64; stride_a: rocblas_stride; const B: Pointer; b_type: rocblas_datatype; ldb: Int64; stride_b: rocblas_stride; const beta: Pointer; const C: Pointer; c_type: rocblas_datatype; ldc: Int64; stride_c: rocblas_stride; D: Pointer; d_type: rocblas_datatype; ldd: Int64; stride_d: rocblas_stride; batch_count: Int64; compute_type: rocblas_datatype; algo: rocblas_gemm_algo; solution_index: Integer; flags: Cardinal): rocblas_status; cdecl; external;

function rocblas_gemm_batched_ex_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: Pointer; const A: Pointer; a_type: rocblas_datatype; lda: Int64; const B: Pointer; b_type: rocblas_datatype; ldb: Int64; const beta: Pointer; const C: Pointer; c_type: rocblas_datatype; ldc: Int64; D: Pointer; d_type: rocblas_datatype; ldd: Int64; batch_count: Int64; compute_type: rocblas_datatype; algo: rocblas_gemm_algo; solution_index: Integer; flags: Cardinal): rocblas_status; cdecl; external;

{ batched }
function rocblas_sgemm_batched(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: PSingle; const A: PPSingle; lda: rocblas_int; const B: PPSingle; ldb: rocblas_int; const beta: PSingle; C: PPSingle; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dgemm_batched(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: PDouble; const A: PPDouble; lda: rocblas_int; const B: PPDouble; ldb: rocblas_int; const beta: PDouble; C: PPDouble; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgemm_batched(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: rocblas_int; const B: PPProcfloat_complex; ldb: rocblas_int; const beta: Procfloat_complex; C: PPProcfloat_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgemm_batched(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: rocblas_int; const B: PPProcdouble_complex; ldb: rocblas_int; const beta: Procdouble_complex; C: PPProcdouble_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit batched }
function rocblas_sgemm_batched_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: PSingle; const A: PPSingle; lda: Int64; const B: PPSingle; ldb: Int64; const beta: PSingle; C: PPSingle; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dgemm_batched_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: PDouble; const A: PPDouble; lda: Int64; const B: PPDouble; ldb: Int64; const beta: PDouble; C: PPDouble; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_cgemm_batched_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: Int64; const B: PPProcfloat_complex; ldb: Int64; const beta: Procfloat_complex; C: PPProcfloat_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zgemm_batched_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: Int64; const B: PPProcdouble_complex; ldb: Int64; const beta: Procdouble_complex; C: PPProcdouble_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;

{ strided-batched }
function rocblas_sgemm_strided_batched(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: PSingle; const A: PSingle; lda: rocblas_int; stride_A: rocblas_stride; const B: PSingle; ldb: rocblas_int; stride_B: rocblas_stride; const beta: PSingle; C: PSingle; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dgemm_strided_batched(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: PDouble; const A: PDouble; lda: rocblas_int; stride_A: rocblas_stride; const B: PDouble; ldb: rocblas_int; stride_B: rocblas_stride; const beta: PDouble; C: PDouble; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_cgemm_strided_batched(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; stride_A: rocblas_stride; const B: Procfloat_complex; ldb: rocblas_int; stride_B: rocblas_stride; const beta: Procfloat_complex; C: Procfloat_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zgemm_strided_batched(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; stride_A: rocblas_stride; const B: Procdouble_complex; ldb: rocblas_int; stride_B: rocblas_stride; const beta: Procdouble_complex; C: Procdouble_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit strided-batched }
function rocblas_sgemm_strided_batched_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: PSingle; const A: PSingle; lda: Int64; stride_A: rocblas_stride; const B: PSingle; ldb: Int64; stride_B: rocblas_stride; const beta: PSingle; C: PSingle; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dgemm_strided_batched_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: PDouble; const A: PDouble; lda: Int64; stride_A: rocblas_stride; const B: PDouble; ldb: Int64; stride_B: rocblas_stride; const beta: PDouble; C: PDouble; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_cgemm_strided_batched_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; stride_A: rocblas_stride; const B: Procfloat_complex; ldb: Int64; stride_B: rocblas_stride; const beta: Procfloat_complex; C: Procfloat_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zgemm_strided_batched_64(handle: rocblas_handle; transA, transB: rocblas_operation; m, n, k: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; stride_A: rocblas_stride; const B: Procdouble_complex; ldb: Int64; stride_B: rocblas_stride; const beta: Procdouble_complex; C: Procdouble_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ ---------- HEMM ---------- }
{ scalar }
function rocblas_chemm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; const B: Procfloat_complex; ldb: rocblas_int; const beta: Procfloat_complex; C: Procfloat_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhemm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; const B: Procdouble_complex; ldb: rocblas_int; const beta: Procdouble_complex; C: Procdouble_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit scalar }
function rocblas_chemm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; const B: Procfloat_complex; ldb: Int64; const beta: Procfloat_complex; C: Procfloat_complex; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_zhemm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; const B: Procdouble_complex; ldb: Int64; const beta: Procdouble_complex; C: Procdouble_complex; ldc: Int64): rocblas_status; cdecl; external;

{ batched }
function rocblas_chemm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: rocblas_int; const B: PPProcfloat_complex; ldb: rocblas_int; const beta: Procfloat_complex; C: PPProcfloat_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhemm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: rocblas_int; const B: PPProcdouble_complex; ldb: rocblas_int; const beta: Procdouble_complex; C: PPProcdouble_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit batched }
function rocblas_chemm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: Int64; const B: PPProcfloat_complex; ldb: Int64; const beta: Procfloat_complex; C: PPProcfloat_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zhemm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: Int64; const B: PPProcdouble_complex; ldb: Int64; const beta: Procdouble_complex; C: PPProcdouble_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;

{ strided-batched }
function rocblas_chemm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; stride_A: rocblas_stride; const B: Procfloat_complex; ldb: rocblas_int; stride_B: rocblas_stride; const beta: Procfloat_complex; C: Procfloat_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zhemm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; stride_A: rocblas_stride; const B: Procdouble_complex; ldb: rocblas_int; stride_B: rocblas_stride; const beta: Procdouble_complex; C: Procdouble_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit strided-batched }
function rocblas_chemm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; stride_A: rocblas_stride; const B: Procfloat_complex; ldb: Int64; stride_B: rocblas_stride; const beta: Procfloat_complex; C: Procfloat_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zhemm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; stride_A: rocblas_stride; const B: Procdouble_complex; ldb: Int64; stride_B: rocblas_stride; const beta: Procdouble_complex; C: Procdouble_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ ---------- HERK ---------- }
{ scalar }
function rocblas_cherk(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PSingle; const A: Procfloat_complex; lda: rocblas_int; const beta: PSingle; C: Procfloat_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zherk(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PDouble; const A: Procdouble_complex; lda: rocblas_int; const beta: PDouble; C: Procdouble_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit scalar }
function rocblas_cherk_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PSingle; const A: Procfloat_complex; lda: Int64; const beta: PSingle; C: Procfloat_complex; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_zherk_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PDouble; const A: Procdouble_complex; lda: Int64; const beta: PDouble; C: Procdouble_complex; ldc: Int64): rocblas_status; cdecl; external;

{ batched }
function rocblas_cherk_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PSingle; const A: PPProcfloat_complex; lda: rocblas_int; const beta: PSingle; C: PPProcfloat_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zherk_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PDouble; const A: PPProcdouble_complex; lda: rocblas_int; const beta: PDouble; C: PPProcdouble_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit batched }
function rocblas_cherk_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PSingle; const A: PPProcfloat_complex; lda: Int64; const beta: PSingle; C: PPProcfloat_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zherk_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PDouble; const A: PPProcdouble_complex; lda: Int64; const beta: PDouble; C: PPProcdouble_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;

{ strided-batched }
function rocblas_cherk_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PSingle; const A: Procfloat_complex; lda: rocblas_int; stride_A: rocblas_stride; const beta: PSingle; C: Procfloat_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zherk_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PDouble; const A: Procdouble_complex; lda: rocblas_int; stride_A: rocblas_stride; const beta: PDouble; C: Procdouble_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit strided-batched }
function rocblas_cherk_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PSingle; const A: Procfloat_complex; lda: Int64; stride_A: rocblas_stride; const beta: PSingle; C: Procfloat_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zherk_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PDouble; const A: Procdouble_complex; lda: Int64; stride_A: rocblas_stride; const beta: PDouble; C: Procdouble_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ ---------- HER2K ---------- }
{ scalar }
function rocblas_cher2k(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; const B: Procfloat_complex; ldb: rocblas_int; const beta: PSingle; C: Procfloat_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zher2k(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; const B: Procdouble_complex; ldb: rocblas_int; const beta: PDouble; C: Procdouble_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit scalar }
function rocblas_cher2k_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; const B: Procfloat_complex; ldb: Int64; const beta: PSingle; C: Procfloat_complex; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_zher2k_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; const B: Procdouble_complex; ldb: Int64; const beta: PDouble; C: Procdouble_complex; ldc: Int64): rocblas_status; cdecl; external;

{ batched }
function rocblas_cher2k_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: rocblas_int; const B: PPProcfloat_complex; ldb: rocblas_int; const beta: PSingle; C: PPProcfloat_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zher2k_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: rocblas_int; const B: PPProcdouble_complex; ldb: rocblas_int; const beta: PDouble; C: PPProcdouble_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit batched }
function rocblas_cher2k_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: Int64; const B: PPProcfloat_complex; ldb: Int64; const beta: PSingle; C: PPProcfloat_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zher2k_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: Int64; const B: PPProcdouble_complex; ldb: Int64; const beta: PDouble; C: PPProcdouble_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;

{ strided-batched }
function rocblas_cher2k_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; stride_A: rocblas_stride; const B: Procfloat_complex; ldb: rocblas_int; stride_B: rocblas_stride; const beta: PSingle; C: Procfloat_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zher2k_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; stride_A: rocblas_stride; const B: Procdouble_complex; ldb: rocblas_int; stride_B: rocblas_stride; const beta: PDouble; C: Procdouble_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit strided-batched }
function rocblas_cher2k_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; stride_A: rocblas_stride; const B: Procfloat_complex; ldb: Int64; stride_B: rocblas_stride; const beta: PSingle; C: Procfloat_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zher2k_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; stride_A: rocblas_stride; const B: Procdouble_complex; ldb: Int64; stride_B: rocblas_stride; const beta: PDouble; C: Procdouble_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ ---------- SYMM ---------- }
{ scalar }
function rocblas_ssymm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: PSingle; const A: PSingle; lda: rocblas_int; const B: PSingle; ldb: rocblas_int; const beta: PSingle; C: PSingle; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsymm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: PDouble; const A: PDouble; lda: rocblas_int; const B: PDouble; ldb: rocblas_int; const beta: PDouble; C: PDouble; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csymm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; const B: Procfloat_complex; ldb: rocblas_int; const beta: Procfloat_complex; C: Procfloat_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsymm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; const B: Procdouble_complex; ldb: rocblas_int; const beta: Procdouble_complex; C: Procdouble_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit scalar }
function rocblas_ssymm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: PSingle; const A: PSingle; lda: Int64; const B: PSingle; ldb: Int64; const beta: PSingle; C: PSingle; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_dsymm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: PDouble; const A: PDouble; lda: Int64; const B: PDouble; ldb: Int64; const beta: PDouble; C: PDouble; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_csymm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; const B: Procfloat_complex; ldb: Int64; const beta: Procfloat_complex; C: Procfloat_complex; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_zsymm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; const B: Procdouble_complex; ldb: Int64; const beta: Procdouble_complex; C: Procdouble_complex; ldc: Int64): rocblas_status; cdecl; external;

{ batched }
function rocblas_ssymm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: PSingle; const A: PPSingle; lda: rocblas_int; const B: PPSingle; ldb: rocblas_int; const beta: PSingle; C: PPSingle; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsymm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: PDouble; const A: PPDouble; lda: rocblas_int; const B: PPDouble; ldb: rocblas_int; const beta: PDouble; C: PPDouble; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csymm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: rocblas_int; const B: PPProcfloat_complex; ldb: rocblas_int; const beta: Procfloat_complex; C: PPProcfloat_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsymm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: rocblas_int; const B: PPProcdouble_complex; ldb: rocblas_int; const beta: Procdouble_complex; C: PPProcdouble_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit batched }
function rocblas_ssymm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: PSingle; const A: PPSingle; lda: Int64; const B: PPSingle; ldb: Int64; const beta: PSingle; C: PPSingle; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsymm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: PDouble; const A: PPDouble; lda: Int64; const B: PPDouble; ldb: Int64; const beta: PDouble; C: PPDouble; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csymm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: Int64; const B: PPProcfloat_complex; ldb: Int64; const beta: Procfloat_complex; C: PPProcfloat_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zsymm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: Int64; const B: PPProcdouble_complex; ldb: Int64; const beta: Procdouble_complex; C: PPProcdouble_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;

{ strided-batched }
function rocblas_ssymm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: PSingle; const A: PSingle; lda: rocblas_int; stride_A: rocblas_stride; const B: PSingle; ldb: rocblas_int; stride_B: rocblas_stride; const beta: PSingle; C: PSingle; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsymm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: PDouble; const A: PDouble; lda: rocblas_int; stride_A: rocblas_stride; const B: PDouble; ldb: rocblas_int; stride_B: rocblas_stride; const beta: PDouble; C: PDouble; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csymm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; stride_A: rocblas_stride; const B: Procfloat_complex; ldb: rocblas_int; stride_B: rocblas_stride; const beta: Procfloat_complex; C: Procfloat_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsymm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; stride_A: rocblas_stride; const B: Procdouble_complex; ldb: rocblas_int; stride_B: rocblas_stride; const beta: Procdouble_complex; C: Procdouble_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit strided-batched }
function rocblas_ssymm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: PSingle; const A: PSingle; lda: Int64; stride_A: rocblas_stride; const B: PSingle; ldb: Int64; stride_B: rocblas_stride; const beta: PSingle; C: PSingle; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsymm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: PDouble; const A: PDouble; lda: Int64; stride_A: rocblas_stride; const B: PDouble; ldb: Int64; stride_B: rocblas_stride; const beta: PDouble; C: PDouble; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csymm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; stride_A: rocblas_stride; const B: Procfloat_complex; ldb: Int64; stride_B: rocblas_stride; const beta: Procfloat_complex; C: Procfloat_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zsymm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; m, n: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; stride_A: rocblas_stride; const B: Procdouble_complex; ldb: Int64; stride_B: rocblas_stride; const beta: Procdouble_complex; C: Procdouble_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ ---------- SYRK ---------- }
{ scalar }
function rocblas_ssyrk(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PSingle; const A: PSingle; lda: rocblas_int; const beta: PSingle; C: PSingle; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsyrk(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PDouble; const A: PDouble; lda: rocblas_int; const beta: PDouble; C: PDouble; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csyrk(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; const beta: Procfloat_complex; C: Procfloat_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsyrk(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; const beta: Procdouble_complex; C: Procdouble_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit scalar }
function rocblas_ssyrk_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PSingle; const A: PSingle; lda: Int64; const beta: PSingle; C: PSingle; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_dsyrk_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PDouble; const A: PDouble; lda: Int64; const beta: PDouble; C: PDouble; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_csyrk_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; const beta: Procfloat_complex; C: Procfloat_complex; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_zsyrk_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; const beta: Procdouble_complex; C: Procdouble_complex; ldc: Int64): rocblas_status; cdecl; external;

{ batched }
function rocblas_ssyrk_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PSingle; const A: PPSingle; lda: rocblas_int; const beta: PSingle; C: PPSingle; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsyrk_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PDouble; const A: PPDouble; lda: rocblas_int; const beta: PDouble; C: PPDouble; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csyrk_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: rocblas_int; const beta: Procfloat_complex; C: PPProcfloat_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsyrk_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: rocblas_int; const beta: Procdouble_complex; C: PPProcdouble_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit batched }
function rocblas_ssyrk_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PSingle; const A: PPSingle; lda: Int64; const beta: PSingle; C: PPSingle; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsyrk_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PDouble; const A: PPDouble; lda: Int64; const beta: PDouble; C: PPDouble; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csyrk_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: Int64; const beta: Procfloat_complex; C: PPProcfloat_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zsyrk_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: Int64; const beta: Procdouble_complex; C: PPProcdouble_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;

{ strided-batched }
function rocblas_ssyrk_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PSingle; const A: PSingle; lda: rocblas_int; stride_A: rocblas_stride; const beta: PSingle; C: PSingle; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsyrk_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PDouble; const A: PDouble; lda: rocblas_int; stride_A: rocblas_stride; const beta: PDouble; C: PDouble; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csyrk_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; stride_A: rocblas_stride; const beta: Procfloat_complex; C: Procfloat_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsyrk_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; stride_A: rocblas_stride; const beta: Procdouble_complex; C: Procdouble_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit strided-batched }
function rocblas_ssyrk_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PSingle; const A: PSingle; lda: Int64; stride_A: rocblas_stride; const beta: PSingle; C: PSingle; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsyrk_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PDouble; const A: PDouble; lda: Int64; stride_A: rocblas_stride; const beta: PDouble; C: PDouble; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csyrk_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; stride_A: rocblas_stride; const beta: Procfloat_complex; C: Procfloat_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zsyrk_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; stride_A: rocblas_stride; const beta: Procdouble_complex; C: Procdouble_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ ---------- SYR2K ---------- }
{ scalar }
function rocblas_ssyr2k(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PSingle; const A: PSingle; lda: rocblas_int; const B: PSingle; ldb: rocblas_int; const beta: PSingle; C: PSingle; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsyr2k(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PDouble; const A: PDouble; lda: rocblas_int; const B: PDouble; ldb: rocblas_int; const beta: PDouble; C: PDouble; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csyr2k(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; const B: Procfloat_complex; ldb: rocblas_int; const beta: Procfloat_complex; C: Procfloat_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsyr2k(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; const B: Procdouble_complex; ldb: rocblas_int; const beta: Procdouble_complex; C: Procdouble_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit scalar }
function rocblas_ssyr2k_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PSingle; const A: PSingle; lda: Int64; const B: PSingle; ldb: Int64; const beta: PSingle; C: PSingle; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_dsyr2k_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PDouble; const A: PDouble; lda: Int64; const B: PDouble; ldb: Int64; const beta: PDouble; C: PDouble; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_csyr2k_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; const B: Procfloat_complex; ldb: Int64; const beta: Procfloat_complex; C: Procfloat_complex; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_zsyr2k_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; const B: Procdouble_complex; ldb: Int64; const beta: Procdouble_complex; C: Procdouble_complex; ldc: Int64): rocblas_status; cdecl; external;

{ batched }
function rocblas_ssyr2k_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PSingle; const A: PPSingle; lda: rocblas_int; const B: PPSingle; ldb: rocblas_int; const beta: PSingle; C: PPSingle; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsyr2k_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PDouble; const A: PPDouble; lda: rocblas_int; const B: PPDouble; ldb: rocblas_int; const beta: PDouble; C: PPDouble; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csyr2k_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: rocblas_int; const B: PPProcfloat_complex; ldb: rocblas_int; const beta: Procfloat_complex; C: PPProcfloat_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsyr2k_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: rocblas_int; const B: PPProcdouble_complex; ldb: rocblas_int; const beta: Procdouble_complex; C: PPProcdouble_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ 64-bit batched }
function rocblas_ssyr2k_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PSingle; const A: PPSingle; lda: Int64; const B: PPSingle; ldb: Int64; const beta: PSingle; C: PPSingle; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsyr2k_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PDouble; const A: PPDouble; lda: Int64; const B: PPDouble; ldb: Int64; const beta: PDouble; C: PPDouble; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csyr2k_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: Int64; const B: PPProcfloat_complex; ldb: Int64; const beta: Procfloat_complex; C: PPProcfloat_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zsyr2k_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: Int64; const B: PPProcdouble_complex; ldb: Int64; const beta: Procdouble_complex; C: PPProcdouble_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;

{ strided-batched }
function rocblas_ssyr2k_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PSingle; const A: PSingle; lda: rocblas_int; stride_A: rocblas_stride; const B: PSingle; ldb: rocblas_int; stride_B: rocblas_stride; const beta: PSingle; C: PSingle; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dsyr2k_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: PDouble; const A: PDouble; lda: rocblas_int; stride_A: rocblas_stride; const B: PDouble; ldb: rocblas_int; stride_B: rocblas_stride; const beta: PDouble; C: PDouble; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_csyr2k_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; stride_A: rocblas_stride; const B: Procfloat_complex; ldb: rocblas_int; stride_B: rocblas_stride; const beta: Procfloat_complex; C: Procfloat_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_zsyr2k_strided_batched(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; stride_A: rocblas_stride; const B: Procdouble_complex; ldb: rocblas_int; stride_B: rocblas_stride; const beta: Procdouble_complex; C: Procdouble_complex; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ 64-bit strided-batched }
function rocblas_ssyr2k_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PSingle; const A: PSingle; lda: Int64; stride_A: rocblas_stride; const B: PSingle; ldb: Int64; stride_B: rocblas_stride; const beta: PSingle; C: PSingle; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dsyr2k_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: PDouble; const A: PDouble; lda: Int64; stride_A: rocblas_stride; const B: PDouble; ldb: Int64; stride_B: rocblas_stride; const beta: PDouble; C: PDouble; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_csyr2k_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; stride_A: rocblas_stride; const B: Procfloat_complex; ldb: Int64; stride_B: rocblas_stride; const beta: Procfloat_complex; C: Procfloat_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_zsyr2k_strided_batched_64(handle: rocblas_handle; uplo: rocblas_fill; transA: rocblas_operation; n, k: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; stride_A: rocblas_stride; const B: Procdouble_complex; ldb: Int64; stride_B: rocblas_stride; const beta: Procdouble_complex; C: Procdouble_complex; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ ---------- TRMM ---------- }
{ scalar }
function rocblas_strmm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m: rocblas_int; n: rocblas_int; const alpha: PSingle; const A: PSingle; lda: rocblas_int; const B: PSingle; ldb: rocblas_int; C: PSingle; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtrmm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m: rocblas_int; n: rocblas_int; const alpha: PDouble; const A: PDouble; lda: rocblas_int; const B: PDouble; ldb: rocblas_int; C: PDouble; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctrmm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m: rocblas_int; n: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; const B: Procfloat_complex; ldb: rocblas_int; C: Procfloat_complex; ldc: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztrmm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m: rocblas_int; n: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; const B: Procdouble_complex; ldb: rocblas_int; C: Procdouble_complex; ldc: rocblas_int): rocblas_status; cdecl; external;

{ 64-bit scalar }
function rocblas_strmm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: PSingle; const A: hipDeviceptr_t; lda: Int64; const B: hipDeviceptr_t; ldb: Int64; C: hipDeviceptr_t; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_dtrmm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: PDouble; const A: hipDeviceptr_t; lda: Int64; const B: hipDeviceptr_t; ldb: Int64; C: hipDeviceptr_t; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_ctrmm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: Procfloat_complex; const A: hipDeviceptr_t; lda: Int64; const B: hipDeviceptr_t; ldb: Int64; C: hipDeviceptr_t; ldc: Int64): rocblas_status; cdecl; external;
function rocblas_ztrmm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: Procdouble_complex; const A: hipDeviceptr_t; lda: Int64; const B: hipDeviceptr_t; ldb: Int64; C: hipDeviceptr_t; ldc: Int64): rocblas_status; cdecl; external;

{ 32-bit batched }
function rocblas_strmm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: PSingle; const A: PPSingle; lda: rocblas_int; const B: PPSingle; ldb: rocblas_int; C: PPSingle; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtrmm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: PDouble; const A: PPDouble; lda: rocblas_int; const B: PPDouble; ldb: rocblas_int; C: PPDouble; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctrmm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: rocblas_int; const B: PPProcfloat_complex; ldb: rocblas_int; C: PPProcfloat_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztrmm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: rocblas_int; const B: PPProcdouble_complex; ldb: rocblas_int; C: PPProcdouble_complex; ldc: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ 64-bit batched }
function rocblas_strmm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: PSingle; const A: PPSingle; lda: Int64; const B: PPSingle; ldb: Int64; C: PPSingle; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtrmm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: PDouble; const A: PPDouble; lda: Int64; const B: PPDouble; ldb: Int64; C: PPDouble; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctrmm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: Int64; const B: PPProcfloat_complex; ldb: Int64; C: PPProcfloat_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztrmm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: Int64; const B: PPProcdouble_complex; ldb: Int64; C: PPProcdouble_complex; ldc: Int64; batch_count: Int64): rocblas_status; cdecl; external;

{ 32-bit strided-batched }
function rocblas_strmm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: PSingle; const A: hipDeviceptr_t; lda: rocblas_int; stride_A: rocblas_stride; const B: hipDeviceptr_t; ldb: rocblas_int; stride_B: rocblas_stride; C: hipDeviceptr_t; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtrmm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: PDouble; const A: hipDeviceptr_t; lda: rocblas_int; stride_A: rocblas_stride; const B: hipDeviceptr_t; ldb: rocblas_int; stride_B: rocblas_stride; C: hipDeviceptr_t; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctrmm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: Procfloat_complex; const A: hipDeviceptr_t; lda: rocblas_int; stride_A: rocblas_stride; const B: hipDeviceptr_t; ldb: rocblas_int; stride_B: rocblas_stride; C: hipDeviceptr_t; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztrmm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: Procdouble_complex; const A: hipDeviceptr_t; lda: rocblas_int; stride_A: rocblas_stride; const B: hipDeviceptr_t; ldb: rocblas_int; stride_B: rocblas_stride; C: hipDeviceptr_t; ldc: rocblas_int; stride_C: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ 64-bit strided-batched }
function rocblas_strmm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: PSingle; const A: hipDeviceptr_t; lda: Int64; stride_A: rocblas_stride; const B: hipDeviceptr_t; ldb: Int64; stride_B: rocblas_stride; C: hipDeviceptr_t; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtrmm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: PDouble; const A: hipDeviceptr_t; lda: Int64; stride_A: rocblas_stride; const B: hipDeviceptr_t; ldb: Int64; stride_B: rocblas_stride; C: hipDeviceptr_t; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctrmm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: Procfloat_complex; const A: hipDeviceptr_t; lda: Int64; stride_A: rocblas_stride; const B: hipDeviceptr_t; ldb: Int64; stride_B: rocblas_stride; C: hipDeviceptr_t; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztrmm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: Procdouble_complex; const A: hipDeviceptr_t; lda: Int64; stride_A: rocblas_stride; const B: hipDeviceptr_t; ldb: Int64; stride_B: rocblas_stride; C: hipDeviceptr_t; ldc: Int64; stride_C: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;

{ ---------- TRSM ---------- }
{ scalar }
function rocblas_strsm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: PSingle; const A: PSingle; lda: rocblas_int; B: PSingle; ldb: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtrsm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: PDouble; const A: PDouble; lda: rocblas_int; B: PDouble; ldb: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctrsm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; B: Procfloat_complex; ldb: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztrsm(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; B: Procdouble_complex; ldb: rocblas_int): rocblas_status; cdecl; external;

{ 64-bit scalar }
function rocblas_strsm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: PSingle; const A: hipDeviceptr_t; lda: Int64; B: hipDeviceptr_t; ldb: Int64): rocblas_status; cdecl; external;
function rocblas_dtrsm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: PDouble; const A: hipDeviceptr_t; lda: Int64; B: hipDeviceptr_t; ldb: Int64): rocblas_status; cdecl; external;
function rocblas_ctrsm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: Procfloat_complex; const A: hipDeviceptr_t; lda: Int64; B: hipDeviceptr_t; ldb: Int64): rocblas_status; cdecl; external;
function rocblas_ztrsm_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: Procdouble_complex; const A: hipDeviceptr_t; lda: Int64; B: hipDeviceptr_t; ldb: Int64): rocblas_status; cdecl; external;

{ batched }
function rocblas_strsm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: PSingle; const A: PPSingle; lda: rocblas_int; B: PPSingle; ldb: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtrsm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: PDouble; const A: PPDouble; lda: rocblas_int; B: PPDouble; ldb: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctrsm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: rocblas_int; B: PPProcfloat_complex; ldb: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztrsm_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: rocblas_int; B: PPProcdouble_complex; ldb: rocblas_int; batch_count: rocblas_int): rocblas_status; cdecl; external;

{ 64-bit batched }
function rocblas_strsm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: PSingle; const A: PPSingle; lda: Int64; B: PPSingle; ldb: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtrsm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: PDouble; const A: PPDouble; lda: Int64; B: PPDouble; ldb: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctrsm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: Procfloat_complex; const A: PPProcfloat_complex; lda: Int64; B: PPProcfloat_complex; ldb: Int64; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztrsm_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: Procdouble_complex; const A: PPProcdouble_complex; lda: Int64; B: PPProcdouble_complex; ldb: Int64; batch_count: Int64): rocblas_status; cdecl; external;

{ strided-batched }
function rocblas_strsm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: PSingle; const A: PSingle; lda: rocblas_int; stride_A: rocblas_stride; B: PSingle; ldb: rocblas_int; stride_B: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_dtrsm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: PDouble; const A: PDouble; lda: rocblas_int; stride_A: rocblas_stride; B: PDouble; ldb: rocblas_int; stride_B: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ctrsm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: rocblas_int; stride_A: rocblas_stride; B: Procfloat_complex; ldb: rocblas_int; stride_B: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
function rocblas_ztrsm_strided_batched(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: rocblas_int; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: rocblas_int; stride_A: rocblas_stride; B: Procdouble_complex; ldb: rocblas_int; stride_B: rocblas_stride; batch_count: rocblas_int): rocblas_status; cdecl; external;
{ 64-bit strided-batched }
function rocblas_strsm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: PSingle; const A: PSingle; lda: Int64; stride_A: rocblas_stride; B: PSingle; ldb: Int64; stride_B: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_dtrsm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: PDouble; const A: PDouble; lda: Int64; stride_A: rocblas_stride; B: PDouble; ldb: Int64; stride_B: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ctrsm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: Procfloat_complex; const A: Procfloat_complex; lda: Int64; stride_A: rocblas_stride; B: Procfloat_complex; ldb: Int64; stride_B: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;
function rocblas_ztrsm_strided_batched_64(handle: rocblas_handle; side: rocblas_side; uplo: rocblas_fill; transA: rocblas_operation; diag: rocblas_diagonal; m, n: Int64; const alpha: Procdouble_complex; const A: Procdouble_complex; lda: Int64; stride_A: rocblas_stride; B: Procdouble_complex; ldb: Int64; stride_B: rocblas_stride; batch_count: Int64): rocblas_status; cdecl; external;


implementation



end.
