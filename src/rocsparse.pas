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

unit rocsparse;

{$mode objfpc}{$H+}
{$linklib rocsparse}

interface

uses
  hip_runtime, hip_types, ctypes;

//
// rocsparse-types.h
//

type
  {$if defined(rocsparse_ILP64)}
  rocsparse_int = int64;
  {$else}
  rocsparse_int = int32;
  {$endif}

  rocsparse_handle = type Pointer;
  rocsparse_mat_descr = type Pointer;
  rocsparse_hyb_mat = type Pointer;
  rocsparse_mat_info = type Pointer;
  rocsparse_color_info = type Pointer;

  // Generic API
  rocsparse_spvec_descr = type Pointer;
  rocsparse_const_spvec_descr = type Pointer;
  rocsparse_spmat_descr = type Pointer;
  rocsparse_const_spmat_descr = type Pointer;
  rocsparse_dnvec_descr = type Pointer;
  rocsparse_const_dnvec_descr = type Pointer;
  rocsparse_dnmat_descr = type Pointer;
  rocsparse_const_dnmat_descr = type Pointer;

  Procsparse_handle = ^rocsparse_handle;
  Procsparse_mat_descr = ^rocsparse_mat_descr;
  Procsparse_mat_info = ^rocsparse_mat_info;
  Procsparse_hyb_mat = ^rocsparse_hyb_mat;
  Procsparse_pointer_mode = ^rocsparse_pointer_mode;
  Procsparse_int = ^rocsparse_int;
  Procsparse_float_complex = ^rocsparse_float_complex;
  Procsparse_double_complex = ^rocsparse_double_complex;
  PhipStream_t = ^hipStream_t;

  rocsparse_operation = (
    rocsparse_operation_none = 111,
    rocsparse_operation_transpose = 112,
    rocsparse_operation_conjugate_transpose = 113
  );

  rocsparse_index_base = (
    rocsparse_index_base_zero = 0,
    rocsparse_index_base_one = 1
  );

  rocsparse_matrix_type = (
    rocsparse_matrix_type_general = 0,
    rocsparse_matrix_type_symmetric = 1,
    rocsparse_matrix_type_hermitian = 2,
    rocsparse_matrix_type_triangular = 3
  );

  rocsparse_diag_type = (
    rocsparse_diag_type_non_unit = 0,
    rocsparse_diag_type_unit = 1
  );

  rocsparse_fill_mode = (
    rocsparse_fill_mode_lower = 0,
    rocsparse_fill_mode_upper = 1
  );

  rocsparse_storage_mode = (
    rocsparse_storage_mode_sorted = 0,
    rocsparse_storage_mode_unsorted = 1
  );

  rocsparse_action = (
    rocsparse_action_symbolic = 0,
    rocsparse_action_numeric = 1
  );

  rocsparse_direction = (
    rocsparse_direction_row = 0,
    rocsparse_direction_column = 1
  );

  rocsparse_hyb_partition = (
    rocsparse_hyb_partition_auto = 0,
    rocsparse_hyb_partition_user = 1,
    rocsparse_hyb_partition_max = 2
  );

  rocsparse_analysis_policy = (
    rocsparse_analysis_policy_reuse = 0,
    rocsparse_analysis_policy_force = 1
  );

  rocsparse_solve_policy = (
    rocsparse_solve_policy_auto = 0
  );

  rocsparse_pointer_mode = (
    rocsparse_pointer_mode_host = 0,
    rocsparse_pointer_mode_device = 1
  );

  rocsparse_layer_mode = (
    rocsparse_layer_mode_none = $0,
    rocsparse_layer_mode_log_trace = $1,
    rocsparse_layer_mode_log_bench = $2,
    rocsparse_layer_mode_log_debug = $4
  );

  rocsparse_status = (
    rocsparse_status_success = 0,
    rocsparse_status_invalid_handle = 1,
    rocsparse_status_not_implemented = 2,
    rocsparse_status_invalid_pointer = 3,
    rocsparse_status_invalid_size = 4,
    rocsparse_status_memory_error = 5,
    rocsparse_status_internal_error = 6,
    rocsparse_status_invalid_value = 7,
    rocsparse_status_arch_mismatch = 8,
    rocsparse_status_zero_pivot = 9,
    rocsparse_status_not_initialized = 10,
    rocsparse_status_type_mismatch = 11,
    rocsparse_status_requires_sorted_storage = 12
  );

  rocsparse_indextype = (
    rocsparse_indextype_u16 = 1,
    rocsparse_indextype_i32 = 2,
    rocsparse_indextype_i64 = 3
  );

  rocsparse_datatype = (
    rocsparse_datatype_f16_r = 150,
    rocsparse_datatype_f32_r = 151,
    rocsparse_datatype_f64_r = 152,
    rocsparse_datatype_f32_c = 154,
    rocsparse_datatype_f64_c = 155,
    rocsparse_datatype_i8_r = 160,
    rocsparse_datatype_u8_r = 161,
    rocsparse_datatype_i32_r = 162,
    rocsparse_datatype_u32_r = 163
  );

  rocsparse_format = (
      rocsparse_format_coo = 0,
      rocsparse_format_coo_aos = 1,
      rocsparse_format_csr = 2,
      rocsparse_format_csc = 3,
      rocsparse_format_ell = 4,
      rocsparse_format_bell = 5
  );

  rocsparse_order = (
      rocsparse_order_row = 0,
      rocsparse_order_column = 1
  );

  rocsparse_spmat_attribute = (
      rocsparse_spmat_fill_mode = 0,
      rocsparse_spmat_diag_type = 1,
      rocsparse_spmat_matrix_type = 2,
      rocsparse_spmat_storage_mode = 3
  );

  rocsparse_spmv_alg = (
      rocsparse_spmv_alg_default = 0,
      rocsparse_spmv_alg_coo = 1,
      rocsparse_spmv_alg_csr_adaptive = 2,
      rocsparse_spmv_alg_csr_stream = 3,
      rocsparse_spmv_alg_ell = 4,
      rocsparse_spmv_alg_coo_atomic = 5,
      rocsparse_spmv_alg_bsr = 6,
      rocsparse_spmv_alg_csr_lrb = 7
  );

  rocsparse_spsv_alg = (
      rocsparse_spsv_alg_default = 0
  );

  rocsparse_spsm_alg = (
      rocsparse_spsm_alg_default = 0
  );

  rocsparse_spgemm_alg = (
      rocsparse_spgemm_alg_default = 0
  );

//
// rocsparse-complex-types.h
//
  rocsparse_float_complex = record
    x: Single;
    y: Single;
  end;

  rocsparse_double_complex = record
    x: Double;
    y: Double;
  end;

//
// rocsparse-auxiliary.h
//
function rocsparse_create_handle(handle: Procsparse_handle): rocsparse_status; cdecl; external;
function rocsparse_destroy_handle(handle: rocsparse_handle): rocsparse_status; cdecl; external;
function rocsparse_set_stream(handle: rocsparse_handle; stream: hipStream_t): rocsparse_status; cdecl; external;
function rocsparse_get_stream(handle: rocsparse_handle; stream: PhipStream_t): rocsparse_status; cdecl; external;
function rocsparse_set_pointer_mode(handle: rocsparse_handle; mode: rocsparse_pointer_mode): rocsparse_status; cdecl; external;
function rocsparse_get_pointer_mode(handle: rocsparse_handle; mode: Procsparse_pointer_mode): rocsparse_status; cdecl; external;
function rocsparse_get_version(handle: rocsparse_handle; version: PInteger): rocsparse_status; cdecl; external;
function rocsparse_get_git_rev(handle: rocsparse_handle; rev: PChar): rocsparse_status; cdecl; external;
function rocsparse_create_mat_descr(descr: Procsparse_mat_descr): rocsparse_status; cdecl; external;
function rocsparse_destroy_mat_descr(descr: rocsparse_mat_descr): rocsparse_status; cdecl; external;
function rocsparse_copy_mat_descr(dest: rocsparse_mat_descr; const src: rocsparse_mat_descr): rocsparse_status; cdecl; external;
function rocsparse_set_mat_index_base(descr: rocsparse_mat_descr; base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_get_mat_index_base(const descr: rocsparse_mat_descr): rocsparse_index_base; cdecl; external;
function rocsparse_set_mat_type(descr: rocsparse_mat_descr; atype: rocsparse_matrix_type): rocsparse_status; cdecl; external;
function rocsparse_get_mat_type(const descr: rocsparse_mat_descr): rocsparse_matrix_type; cdecl; external;
function rocsparse_set_mat_fill_mode(descr: rocsparse_mat_descr; fill_mode: rocsparse_fill_mode): rocsparse_status; cdecl; external;
function rocsparse_get_mat_fill_mode(const descr: rocsparse_mat_descr): rocsparse_fill_mode; cdecl; external;
function rocsparse_set_mat_diag_type(descr: rocsparse_mat_descr; diag_type: rocsparse_diag_type): rocsparse_status; cdecl; external;
function rocsparse_get_mat_diag_type(const descr: rocsparse_mat_descr): rocsparse_diag_type; cdecl; external;
function rocsparse_set_mat_storage_mode(descr: rocsparse_mat_descr; storage_mode: rocsparse_storage_mode): rocsparse_status; cdecl; external;
function rocsparse_get_mat_storage_mode(const descr: rocsparse_mat_descr): rocsparse_storage_mode; cdecl; external;
function rocsparse_create_mat_info(info: Procsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_copy_mat_info(dest: rocsparse_mat_info; const src: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_destroy_mat_info(info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_create_hyb_mat(hyb: Procsparse_hyb_mat): rocsparse_status; cdecl; external;
function rocsparse_destroy_hyb_mat(hyb: rocsparse_hyb_mat): rocsparse_status; cdecl; external;

//
// rocsparse_axpyi.h
//
function rocsparse_saxpyi(handle: rocsparse_handle; nnz: rocsparse_int; alpha: PSingle; x_val: PSingle; x_ind: Procsparse_int; y: PSingle; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_daxpyi(handle: rocsparse_handle; nnz: rocsparse_int; alpha: PDouble; x_val: PDouble; x_ind: Procsparse_int; y: PDouble; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_caxpyi(handle: rocsparse_handle; nnz: rocsparse_int; alpha: Procsparse_float_complex; x_val: Procsparse_float_complex; x_ind: Procsparse_int; y: Procsparse_float_complex; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_zaxpyi(handle: rocsparse_handle; nnz: rocsparse_int; alpha: Procsparse_double_complex; x_val: Procsparse_double_complex; x_ind: Procsparse_int; y: Procsparse_double_complex; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;

//
// rocsparse_doti.h
//
function rocsparse_sdoti(handle: rocsparse_handle; nnz: rocsparse_int; x_val: PSingle; x_ind: Procsparse_int; y: PSingle; result: PSingle; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_ddoti(handle: rocsparse_handle; nnz: rocsparse_int; x_val: PDouble; x_ind: Procsparse_int; y: PDouble; result: PDouble; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_cdoti(handle: rocsparse_handle; nnz: rocsparse_int; x_val: Procsparse_float_complex; x_ind: Procsparse_int; y: Procsparse_float_complex; result: Procsparse_float_complex; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_zdoti(handle: rocsparse_handle; nnz: rocsparse_int; x_val: Procsparse_double_complex; x_ind: Procsparse_int; y: Procsparse_double_complex; result: Procsparse_double_complex; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;

//
// rocsparse_dotci.h
//
function rocsparse_cdotci(handle: rocsparse_handle; nnz: rocsparse_int; x_val: Procsparse_float_complex; x_ind: Procsparse_int; y: Procsparse_float_complex; result: Procsparse_float_complex; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_zdotci(handle: rocsparse_handle; nnz: rocsparse_int; x_val: Procsparse_double_complex; x_ind: Procsparse_int; y: Procsparse_double_complex; result: Procsparse_double_complex; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;

//
// rocsparse_gthr.h
//
function rocsparse_sgthr(handle: rocsparse_handle; nnz: rocsparse_int; y: PSingle; x_val: PSingle; x_ind: Procsparse_int; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_dgthr(handle: rocsparse_handle; nnz: rocsparse_int; y: PDouble; x_val: PDouble; x_ind: Procsparse_int; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_cgthr(handle: rocsparse_handle; nnz: rocsparse_int; y: Procsparse_float_complex; x_val: Procsparse_float_complex; x_ind: Procsparse_int; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_zgthr(handle: rocsparse_handle; nnz: rocsparse_int; y: Procsparse_double_complex; x_val: Procsparse_double_complex; x_ind: Procsparse_int; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;

//
// rocsparse_gthrz.h
//
function rocsparse_sgthrz(handle: rocsparse_handle; nnz: rocsparse_int; y: PSingle; x_val: PSingle; x_ind: Procsparse_int; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_dgthrz(handle: rocsparse_handle; nnz: rocsparse_int; y: PDouble; x_val: PDouble; x_ind: Procsparse_int; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_cgthrz(handle: rocsparse_handle; nnz: rocsparse_int; y: Procsparse_float_complex; x_val: Procsparse_float_complex; x_ind: Procsparse_int; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_zgthrz(handle: rocsparse_handle; nnz: rocsparse_int; y: Procsparse_double_complex; x_val: Procsparse_double_complex; x_ind: Procsparse_int; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;

//
// rocsparse_roti.h
//
function rocsparse_sroti(handle: rocsparse_handle; nnz: rocsparse_int; x_val: PSingle; x_ind: Procsparse_int; y: PSingle; c: PSingle; s: PSingle; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_droti(handle: rocsparse_handle; nnz: rocsparse_int; x_val: PDouble; x_ind: Procsparse_int; y: PDouble; c: PDouble; s: PDouble; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;

//
// rocsparse_sctr.h
//
function rocsparse_ssctr(handle: rocsparse_handle; nnz: rocsparse_int; x_val: PSingle; x_ind: Procsparse_int; y: PSingle; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_dsctr(handle: rocsparse_handle; nnz: rocsparse_int; x_val: PDouble; x_ind: Procsparse_int; y: PDouble; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_csctr(handle: rocsparse_handle; nnz: rocsparse_int; x_val: Procsparse_float_complex; x_ind: Procsparse_int; y: Procsparse_float_complex; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;
function rocsparse_zsctr(handle: rocsparse_handle; nnz: rocsparse_int; x_val: Procsparse_double_complex; x_ind: Procsparse_int; y: Procsparse_double_complex; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;

//
// Level 2
//

//
// rocsparse_bsrmv.h
//
function rocsparse_sbsrmv_analysis(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_dbsrmv_analysis(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_cbsrmv_analysis(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_zbsrmv_analysis(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_sbsrmv(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; x: PSingle; beta: PSingle; y: PSingle): rocsparse_status; cdecl; external;
function rocsparse_dbsrmv(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; x: PDouble; beta: PDouble; y: PDouble): rocsparse_status; cdecl; external;
function rocsparse_cbsrmv(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; x: Procsparse_float_complex; beta: Procsparse_float_complex; y: Procsparse_float_complex): rocsparse_status; cdecl; external;
function rocsparse_zbsrmv(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; x: Procsparse_double_complex; beta: Procsparse_double_complex; y: Procsparse_double_complex): rocsparse_status; cdecl; external;
function rocsparse_bsrmv_clear(handle: rocsparse_handle; info: rocsparse_mat_info): rocsparse_status; cdecl; external;

//
// rocsparse_bsrsv.h
//
function rocsparse_bsrsv_zero_pivot(handle: rocsparse_handle; info: rocsparse_mat_info; position: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_sbsrsv_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_dbsrsv_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_cbsrsv_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_zbsrsv_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_sbsrsv_analysis(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dbsrsv_analysis(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_cbsrsv_analysis(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zbsrsv_analysis(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_bsrsv_clear(handle: rocsparse_handle; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_sbsrsv_solve(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nnzb: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; x: PSingle; y: PSingle; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dbsrsv_solve(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nnzb: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; x: PDouble; y: PDouble; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_cbsrsv_solve(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; x: Procsparse_float_complex; y: Procsparse_float_complex; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zbsrsv_solve(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; x: Procsparse_double_complex; y: Procsparse_double_complex; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_bsrxmv.h
//
function rocsparse_sbsrxmv(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; size_of_mask: rocsparse_int; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_mask_ptr: Procsparse_int; bsr_row_ptr: Procsparse_int; bsr_end_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; x: PSingle; beta: PSingle; y: PSingle): rocsparse_status; cdecl; external;
function rocsparse_dbsrxmv(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; size_of_mask: rocsparse_int; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_mask_ptr: Procsparse_int; bsr_row_ptr: Procsparse_int; bsr_end_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; x: PDouble; beta: PDouble; y: PDouble): rocsparse_status; cdecl; external;
function rocsparse_cbsrxmv(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; size_of_mask: rocsparse_int; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_mask_ptr: Procsparse_int; bsr_row_ptr: Procsparse_int; bsr_end_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; x: Procsparse_float_complex; beta: Procsparse_float_complex; y: Procsparse_float_complex): rocsparse_status; cdecl; external;
function rocsparse_zbsrxmv(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; size_of_mask: rocsparse_int; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_mask_ptr: Procsparse_int; bsr_row_ptr: Procsparse_int; bsr_end_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; x: Procsparse_double_complex; beta: Procsparse_double_complex; y: Procsparse_double_complex): rocsparse_status; cdecl; external;

//
// rocsparse_coomv.h
//
function rocsparse_scoomv(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; coo_val: PSingle; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int; x: PSingle; beta: PSingle; y: PSingle): rocsparse_status; cdecl; external;
function rocsparse_dcoomv(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; coo_val: PDouble; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int; x: PDouble; beta: PDouble; y: PDouble): rocsparse_status; cdecl; external;
function rocsparse_ccoomv(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; coo_val: Procsparse_float_complex; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int; x: Procsparse_float_complex; beta: Procsparse_float_complex; y: Procsparse_float_complex): rocsparse_status; cdecl; external;
function rocsparse_zcoomv(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; coo_val: Procsparse_double_complex; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int; x: Procsparse_double_complex; beta: Procsparse_double_complex; y: Procsparse_double_complex): rocsparse_status; cdecl; external;

//
// rocsparse_csritsv.h
//
function rocsparse_csritsv_zero_pivot(handle: rocsparse_handle; descr: rocsparse_mat_descr; info: rocsparse_mat_info; position: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_scsritsv_buffer_size(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_dcsritsv_buffer_size(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_ccsritsv_buffer_size(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_zcsritsv_buffer_size(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_scsritsv_analysis(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dcsritsv_analysis(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_ccsritsv_analysis(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zcsritsv_analysis(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_csritsv_clear(handle: rocsparse_handle; descr: rocsparse_mat_descr; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_scsritsv_solve(handle: rocsparse_handle; host_nmaxiter: Procsparse_int; host_tol: PSingle; host_history: PSingle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; x: PSingle; y: PSingle; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dcsritsv_solve(handle: rocsparse_handle; host_nmaxiter: Procsparse_int; host_tol: PDouble; host_history: PDouble; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; x: PDouble; y: PDouble; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_ccsritsv_solve(handle: rocsparse_handle; host_nmaxiter: Procsparse_int; host_tol: PSingle; host_history: PSingle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; x: Procsparse_float_complex; y: Procsparse_float_complex; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zcsritsv_solve(handle: rocsparse_handle; host_nmaxiter: Procsparse_int; host_tol: PDouble; host_history: PDouble; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; x: Procsparse_double_complex; y: Procsparse_double_complex; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_csrmv.h
//
function rocsparse_scsrmv_analysis(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_dcsrmv_analysis(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_ccsrmv_analysis(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_zcsrmv_analysis(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_csrmv_clear(handle: rocsparse_handle; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_scsrmv(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; x: PSingle; beta: PSingle; y: PSingle): rocsparse_status; cdecl; external;
function rocsparse_dcsrmv(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; x: PDouble; beta: PDouble; y: PDouble): rocsparse_status; cdecl; external;
function rocsparse_ccsrmv(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; x: Procsparse_float_complex; beta: Procsparse_float_complex; y: Procsparse_float_complex): rocsparse_status; cdecl; external;
function rocsparse_zcsrmv(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; x: Procsparse_double_complex; beta: Procsparse_double_complex; y: Procsparse_double_complex): rocsparse_status; cdecl; external;

//
// rocsparse_csrsv.h
//
function rocsparse_csrsv_zero_pivot(handle: rocsparse_handle; descr: rocsparse_mat_descr; info: rocsparse_mat_info; position: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_scsrsv_buffer_size(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_dcsrsv_buffer_size(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_ccsrsv_buffer_size(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_zcsrsv_buffer_size(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_scsrsv_analysis(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dcsrsv_analysis(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_ccsrsv_analysis(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zcsrsv_analysis(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_csrsv_clear(handle: rocsparse_handle; descr: rocsparse_mat_descr; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_scsrsv_solve(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; x: PSingle; y: PSingle; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dcsrsv_solve(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; x: PDouble; y: PDouble; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_ccsrsv_solve(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; x: Procsparse_float_complex; y: Procsparse_float_complex; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zcsrsv_solve(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; x: Procsparse_double_complex; y: Procsparse_double_complex; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_ellmv.h
//
function rocsparse_sellmv(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; ell_val: PSingle; ell_col_ind: Procsparse_int; ell_width: rocsparse_int; x: PSingle; beta: PSingle; y: PSingle): rocsparse_status; cdecl; external;
function rocsparse_dellmv(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; ell_val: PDouble; ell_col_ind: Procsparse_int; ell_width: rocsparse_int; x: PDouble; beta: PDouble; y: PDouble): rocsparse_status; cdecl; external;
function rocsparse_cellmv(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; ell_val: Procsparse_float_complex; ell_col_ind: Procsparse_int; ell_width: rocsparse_int; x: Procsparse_float_complex; beta: Procsparse_float_complex; y: Procsparse_float_complex): rocsparse_status; cdecl; external;
function rocsparse_zellmv(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; ell_val: Procsparse_double_complex; ell_col_ind: Procsparse_int; ell_width: rocsparse_int; x: Procsparse_double_complex; beta: Procsparse_double_complex; y: Procsparse_double_complex): rocsparse_status; cdecl; external;

//
// rocsparse_gebsrmv.h
//
function rocsparse_sgebsrmv(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; row_block_dim: rocsparse_int; col_block_dim: rocsparse_int; x: PSingle; beta: PSingle; y: PSingle): rocsparse_status; cdecl; external;
function rocsparse_dgebsrmv(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; row_block_dim: rocsparse_int; col_block_dim: rocsparse_int; x: PDouble; beta: PDouble; y: PDouble): rocsparse_status; cdecl; external;
function rocsparse_cgebsrmv(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; row_block_dim: rocsparse_int; col_block_dim: rocsparse_int; x: Procsparse_float_complex; beta: Procsparse_float_complex; y: Procsparse_float_complex): rocsparse_status; cdecl; external;
function rocsparse_zgebsrmv(handle: rocsparse_handle; dir: rocsparse_direction; trans: rocsparse_operation; mb: rocsparse_int; nb: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; row_block_dim: rocsparse_int; col_block_dim: rocsparse_int; x: Procsparse_double_complex; beta: Procsparse_double_complex; y: Procsparse_double_complex): rocsparse_status; cdecl; external;

//
// rocsparse_gemvi.h
//
function rocsparse_sgemvi_buffer_size(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_dgemvi_buffer_size(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_cgemvi_buffer_size(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_zgemvi_buffer_size(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_sgemvi(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; alpha: PSingle; A: PSingle; lda: rocsparse_int; nnz: rocsparse_int; x_val: PSingle; x_ind: Procsparse_int; beta: PSingle; y: PSingle; idx_base: rocsparse_index_base; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dgemvi(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; alpha: PDouble; A: PDouble; lda: rocsparse_int; nnz: rocsparse_int; x_val: PDouble; x_ind: Procsparse_int; beta: PDouble; y: PDouble; idx_base: rocsparse_index_base; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_cgemvi(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; alpha: Procsparse_float_complex; A: Procsparse_float_complex; lda: rocsparse_int; nnz: rocsparse_int; x_val: Procsparse_float_complex; x_ind: Procsparse_int; beta: Procsparse_float_complex; y: Procsparse_float_complex; idx_base: rocsparse_index_base; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zgemvi(handle: rocsparse_handle; trans: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; alpha: Procsparse_double_complex; A: Procsparse_double_complex; lda: rocsparse_int; nnz: rocsparse_int; x_val: Procsparse_double_complex; x_ind: Procsparse_int; beta: Procsparse_double_complex; y: Procsparse_double_complex; idx_base: rocsparse_index_base; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_hybmv.h
//
function rocsparse_shybmv(handle: rocsparse_handle; trans: rocsparse_operation; alpha: PSingle; descr: rocsparse_mat_descr; hyb: rocsparse_hyb_mat; x: PSingle; beta: PSingle; y: PSingle): rocsparse_status; cdecl; external;
function rocsparse_dhybmv(handle: rocsparse_handle; trans: rocsparse_operation; alpha: PDouble; descr: rocsparse_mat_descr; hyb: rocsparse_hyb_mat; x: PDouble; beta: PDouble; y: PDouble): rocsparse_status; cdecl; external;
function rocsparse_chybmv(handle: rocsparse_handle; trans: rocsparse_operation; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; hyb: rocsparse_hyb_mat; x: Procsparse_float_complex; beta: Procsparse_float_complex; y: Procsparse_float_complex): rocsparse_status; cdecl; external;
function rocsparse_zhybmv(handle: rocsparse_handle; trans: rocsparse_operation; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; hyb: rocsparse_hyb_mat; x: Procsparse_double_complex; beta: Procsparse_double_complex; y: Procsparse_double_complex): rocsparse_status; cdecl; external;

//
// Level 3
//

//
// rocsparse_bsrmm.h
//
function rocsparse_sbsrmm(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_B: rocsparse_operation; mb: rocsparse_int; n: rocsparse_int; kb: rocsparse_int; nnzb: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; B: PSingle; ldb: rocsparse_int; beta: PSingle; C: PSingle; ldc: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_dbsrmm(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_B: rocsparse_operation; mb: rocsparse_int; n: rocsparse_int; kb: rocsparse_int; nnzb: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; B: PDouble; ldb: rocsparse_int; beta: PDouble; C: PDouble; ldc: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_cbsrmm(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_B: rocsparse_operation; mb: rocsparse_int; n: rocsparse_int; kb: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; B: Procsparse_float_complex; ldb: rocsparse_int; beta: Procsparse_float_complex; C: Procsparse_float_complex; ldc: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_zbsrmm(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_B: rocsparse_operation; mb: rocsparse_int; n: rocsparse_int; kb: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; B: Procsparse_double_complex; ldb: rocsparse_int; beta: Procsparse_double_complex; C: Procsparse_double_complex; ldc: rocsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_bsrsm.h
//
function rocsparse_bsrsm_zero_pivot(handle: rocsparse_handle; info: rocsparse_mat_info; position: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_sbsrsm_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_X: rocsparse_operation; mb: rocsparse_int; nrhs: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_dbsrsm_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_X: rocsparse_operation; mb: rocsparse_int; nrhs: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_cbsrsm_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_X: rocsparse_operation; mb: rocsparse_int; nrhs: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_zbsrsm_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_X: rocsparse_operation; mb: rocsparse_int; nrhs: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_sbsrsm_analysis(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_X: rocsparse_operation; mb: rocsparse_int; nrhs: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dbsrsm_analysis(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_X: rocsparse_operation; mb: rocsparse_int; nrhs: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_cbsrsm_analysis(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_X: rocsparse_operation; mb: rocsparse_int; nrhs: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zbsrsm_analysis(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_X: rocsparse_operation; mb: rocsparse_int; nrhs: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_bsrsm_clear(handle: rocsparse_handle; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_sbsrsm_solve(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_X: rocsparse_operation; mb: rocsparse_int; nrhs: rocsparse_int; nnzb: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; B: PSingle; ldb: rocsparse_int; X: PSingle; ldx: rocsparse_int; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dbsrsm_solve(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_X: rocsparse_operation; mb: rocsparse_int; nrhs: rocsparse_int; nnzb: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; B: PDouble; ldb: rocsparse_int; X: PDouble; ldx: rocsparse_int; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_cbsrsm_solve(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_X: rocsparse_operation; mb: rocsparse_int; nrhs: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; B: Procsparse_float_complex; ldb: rocsparse_int; X: Procsparse_float_complex; ldx: rocsparse_int; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zbsrsm_solve(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_X: rocsparse_operation; mb: rocsparse_int; nrhs: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; B: Procsparse_double_complex; ldb: rocsparse_int; X: Procsparse_double_complex; ldx: rocsparse_int; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_csrmm.h
//
function rocsparse_scsrmm(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; k: rocsparse_int; nnz: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: PSingle; ldb: rocsparse_int; beta: PSingle; C: PSingle; ldc: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_dcsrmm(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; k: rocsparse_int; nnz: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: PDouble; ldb: rocsparse_int; beta: PDouble; C: PDouble; ldc: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_ccsrmm(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; k: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: Procsparse_float_complex; ldb: rocsparse_int; beta: Procsparse_float_complex; C: Procsparse_float_complex; ldc: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_zcsrmm(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; k: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: Procsparse_double_complex; ldb: rocsparse_int; beta: Procsparse_double_complex; C: Procsparse_double_complex; ldc: rocsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_csrsm.h
//
function rocsparse_csrsm_zero_pivot(handle: rocsparse_handle; info: rocsparse_mat_info; position: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_scsrsm_buffer_size(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; nrhs: rocsparse_int; nnz: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: PSingle; ldb: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_dcsrsm_buffer_size(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; nrhs: rocsparse_int; nnz: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: PDouble; ldb: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_ccsrsm_buffer_size(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; nrhs: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: Procsparse_float_complex; ldb: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_zcsrsm_buffer_size(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; nrhs: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: Procsparse_double_complex; ldb: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_scsrsm_analysis(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; nrhs: rocsparse_int; nnz: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: PSingle; ldb: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dcsrsm_analysis(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; nrhs: rocsparse_int; nnz: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: PDouble; ldb: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_ccsrsm_analysis(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; nrhs: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: Procsparse_float_complex; ldb: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zcsrsm_analysis(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; nrhs: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: Procsparse_double_complex; ldb: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_csrsm_clear(handle: rocsparse_handle; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_scsrsm_solve(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; nrhs: rocsparse_int; nnz: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: PSingle; ldb: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dcsrsm_solve(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; nrhs: rocsparse_int; nnz: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: PDouble; ldb: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_ccsrsm_solve(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; nrhs: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: Procsparse_float_complex; ldb: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zcsrsm_solve(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; nrhs: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; B: Procsparse_double_complex; ldb: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_gebsrmm.h
//
function rocsparse_sgebsrmm(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_B: rocsparse_operation; mb: rocsparse_int; n: rocsparse_int; kb: rocsparse_int; nnzb: rocsparse_int; alpha: PSingle; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; row_block_dim: rocsparse_int; col_block_dim: rocsparse_int; B: PSingle; ldb: rocsparse_int; beta: PSingle; C: PSingle; ldc: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_dgebsrmm(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_B: rocsparse_operation; mb: rocsparse_int; n: rocsparse_int; kb: rocsparse_int; nnzb: rocsparse_int; alpha: PDouble; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; row_block_dim: rocsparse_int; col_block_dim: rocsparse_int; B: PDouble; ldb: rocsparse_int; beta: PDouble; C: PDouble; ldc: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_cgebsrmm(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_B: rocsparse_operation; mb: rocsparse_int; n: rocsparse_int; kb: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_float_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; row_block_dim: rocsparse_int; col_block_dim: rocsparse_int; B: Procsparse_float_complex; ldb: rocsparse_int; beta: Procsparse_float_complex; C: Procsparse_float_complex; ldc: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_zgebsrmm(handle: rocsparse_handle; dir: rocsparse_direction; trans_A: rocsparse_operation; trans_B: rocsparse_operation; mb: rocsparse_int; n: rocsparse_int; kb: rocsparse_int; nnzb: rocsparse_int; alpha: Procsparse_double_complex; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; row_block_dim: rocsparse_int; col_block_dim: rocsparse_int; B: Procsparse_double_complex; ldb: rocsparse_int; beta: Procsparse_double_complex; C: Procsparse_double_complex; ldc: rocsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_gemmi.h
//
function rocsparse_sgemmi(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; k: rocsparse_int; nnz: rocsparse_int; alpha: PSingle; A: PSingle; lda: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; beta: PSingle; C: PSingle; ldc: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_dgemmi(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; k: rocsparse_int; nnz: rocsparse_int; alpha: PDouble; A: PDouble; lda: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; beta: PDouble; C: PDouble; ldc: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_cgemmi(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; k: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_float_complex; A: Procsparse_float_complex; lda: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; beta: Procsparse_float_complex; C: Procsparse_float_complex; ldc: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_zgemmi(handle: rocsparse_handle; trans_A: rocsparse_operation; trans_B: rocsparse_operation; m: rocsparse_int; n: rocsparse_int; k: rocsparse_int; nnz: rocsparse_int; alpha: Procsparse_double_complex; A: Procsparse_double_complex; lda: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; beta: Procsparse_double_complex; C: Procsparse_double_complex; ldc: rocsparse_int): rocsparse_status; cdecl; external;

//
// Conversion
//

//
// rocsparse_csr2coo.h
//
function rocsparse_csr2coo(handle: rocsparse_handle; csr_row_ptr: Procsparse_int; nnz: rocsparse_int; m: rocsparse_int; coo_row_ind: Procsparse_int; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;

//
// rocsparse_coo2csr.h
//
function rocsparse_coo2csr(handle: rocsparse_handle; coo_row_ind: Procsparse_int; nnz: rocsparse_int; m: rocsparse_int; csr_row_ptr: Procsparse_int; idx_base: rocsparse_index_base): rocsparse_status; cdecl; external;

//
// rocsparse_nnz.h
//
function rocsparse_snnz(handle: rocsparse_handle; dir: rocsparse_direction; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; A: PSingle; ld: rocsparse_int; nnz_per_row_columns: Procsparse_int; nnz_total_dev_host_ptr: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_dnnz(handle: rocsparse_handle; dir: rocsparse_direction; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; A: PDouble; ld: rocsparse_int; nnz_per_row_columns: Procsparse_int; nnz_total_dev_host_ptr: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_cnnz(handle: rocsparse_handle; dir: rocsparse_direction; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; A: Procsparse_float_complex; ld: rocsparse_int; nnz_per_row_columns: Procsparse_int; nnz_total_dev_host_ptr: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_znnz(handle: rocsparse_handle; dir: rocsparse_direction; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; A: Procsparse_double_complex; ld: rocsparse_int; nnz_per_row_columns: Procsparse_int; nnz_total_dev_host_ptr: Procsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_csr2csc.h
//
function rocsparse_csr2csc_buffer_size(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; copy_values: rocsparse_action; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_scsr2csc(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; csc_val: PSingle; csc_row_ind: Procsparse_int; csc_col_ptr: Procsparse_int; copy_values: rocsparse_action; idx_base: rocsparse_index_base; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dcsr2csc(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; csc_val: PDouble; csc_row_ind: Procsparse_int; csc_col_ptr: Procsparse_int; copy_values: rocsparse_action; idx_base: rocsparse_index_base; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_ccsr2csc(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; csc_val: Procsparse_float_complex; csc_row_ind: Procsparse_int; csc_col_ptr: Procsparse_int; copy_values: rocsparse_action; idx_base: rocsparse_index_base; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zcsr2csc(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; csc_val: Procsparse_double_complex; csc_row_ind: Procsparse_int; csc_col_ptr: Procsparse_int; copy_values: rocsparse_action; idx_base: rocsparse_index_base; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_dense2coo.h
//
function rocsparse_sdense2coo(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; A: PSingle; ld: rocsparse_int; nnz_per_rows: Procsparse_int; coo_val: PSingle; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_ddense2coo(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; A: PDouble; ld: rocsparse_int; nnz_per_rows: Procsparse_int; coo_val: PDouble; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_cdense2coo(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; A: Procsparse_float_complex; ld: rocsparse_int; nnz_per_rows: Procsparse_int; coo_val: Procsparse_float_complex; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_zdense2coo(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; A: Procsparse_double_complex; ld: rocsparse_int; nnz_per_rows: Procsparse_int; coo_val: Procsparse_double_complex; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_dense2csr.h
//
function rocsparse_sdense2csr(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; A: PSingle; ld: rocsparse_int; nnz_per_rows: Procsparse_int; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_ddense2csr(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; A: PDouble; ld: rocsparse_int; nnz_per_rows: Procsparse_int; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_cdense2csr(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; A: Procsparse_float_complex; ld: rocsparse_int; nnz_per_rows: Procsparse_int; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_zdense2csr(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; A: Procsparse_double_complex; ld: rocsparse_int; nnz_per_rows: Procsparse_int; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_csr2dense.h
//
function rocsparse_scsr2dense(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; A: PSingle; ld: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_dcsr2dense(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; A: PDouble; ld: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_ccsr2dense(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; A: Procsparse_float_complex; ld: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_zcsr2dense(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; A: Procsparse_double_complex; ld: rocsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_coo2dense.h
//
function rocsparse_scoo2dense(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; coo_val: PSingle; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int; A: PSingle; ld: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_dcoo2dense(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; coo_val: PDouble; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int; A: PDouble; ld: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_ccoo2dense(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; coo_val: Procsparse_float_complex; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int; A: Procsparse_float_complex; ld: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_zcoo2dense(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; coo_val: Procsparse_double_complex; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int; A: Procsparse_double_complex; ld: rocsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_csc2dense.h
//
function rocsparse_scsc2dense(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; csc_val: PSingle; csc_col_ptr: Procsparse_int; csc_row_ind: Procsparse_int; A: PSingle; ld: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_dcsc2dense(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; csc_val: PDouble; csc_col_ptr: Procsparse_int; csc_row_ind: Procsparse_int; A: PDouble; ld: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_ccsc2dense(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; csc_val: Procsparse_float_complex; csc_col_ptr: Procsparse_int; csc_row_ind: Procsparse_int; A: Procsparse_float_complex; ld: rocsparse_int): rocsparse_status; cdecl; external;
function rocsparse_zcsc2dense(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; csc_val: Procsparse_double_complex; csc_col_ptr: Procsparse_int; csc_row_ind: Procsparse_int; A: Procsparse_double_complex; ld: rocsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_csr2bsr.h
//
function rocsparse_csr2bsr_nnz(handle: rocsparse_handle; dir: rocsparse_direction; m: rocsparse_int; n: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; block_dim: rocsparse_int; bsr_descr: rocsparse_mat_descr; bsr_row_ptr: Procsparse_int; bsr_nnz: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_scsr2bsr(handle: rocsparse_handle; dir: rocsparse_direction; m: rocsparse_int; n: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; block_dim: rocsparse_int; bsr_descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_dcsr2bsr(handle: rocsparse_handle; dir: rocsparse_direction; m: rocsparse_int; n: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; block_dim: rocsparse_int; bsr_descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_ccsr2bsr(handle: rocsparse_handle; dir: rocsparse_direction; m: rocsparse_int; n: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; block_dim: rocsparse_int; bsr_descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_zcsr2bsr(handle: rocsparse_handle; dir: rocsparse_direction; m: rocsparse_int; n: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; block_dim: rocsparse_int; bsr_descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_bsr2csr.h
//
function rocsparse_sbsr2csr(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nb: rocsparse_int; bsr_descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_dbsr2csr(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nb: rocsparse_int; bsr_descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_cbsr2csr(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nb: rocsparse_int; bsr_descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_zbsr2csr(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nb: rocsparse_int; bsr_descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_csr2ell.h
//
function rocsparse_csr2ell_width(handle: rocsparse_handle; m: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_row_ptr: Procsparse_int; ell_descr: rocsparse_mat_descr; ell_width: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_scsr2ell(handle: rocsparse_handle; m: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; ell_descr: rocsparse_mat_descr; ell_width: rocsparse_int; ell_val: PSingle; ell_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_dcsr2ell(handle: rocsparse_handle; m: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; ell_descr: rocsparse_mat_descr; ell_width: rocsparse_int; ell_val: PDouble; ell_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_ccsr2ell(handle: rocsparse_handle; m: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; ell_descr: rocsparse_mat_descr; ell_width: rocsparse_int; ell_val: Procsparse_float_complex; ell_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_zcsr2ell(handle: rocsparse_handle; m: rocsparse_int; csr_descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; ell_descr: rocsparse_mat_descr; ell_width: rocsparse_int; ell_val: Procsparse_double_complex; ell_col_ind: Procsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_ell2csr.h
//
function rocsparse_ell2csr_nnz(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; ell_descr: rocsparse_mat_descr; ell_width: rocsparse_int; ell_col_ind: Procsparse_int; csr_descr: rocsparse_mat_descr; csr_row_ptr: Procsparse_int; csr_nnz: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_sell2csr(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; ell_descr: rocsparse_mat_descr; ell_width: rocsparse_int; ell_val: PSingle; ell_col_ind: Procsparse_int; csr_descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_dell2csr(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; ell_descr: rocsparse_mat_descr; ell_width: rocsparse_int; ell_val: PDouble; ell_col_ind: Procsparse_int; csr_descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_cell2csr(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; ell_descr: rocsparse_mat_descr; ell_width: rocsparse_int; ell_val: Procsparse_float_complex; ell_col_ind: Procsparse_int; csr_descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_zell2csr(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; ell_descr: rocsparse_mat_descr; ell_width: rocsparse_int; ell_val: Procsparse_double_complex; ell_col_ind: Procsparse_int; csr_descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_csr2hyb.h
//
function rocsparse_scsr2hyb(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; hyb: rocsparse_hyb_mat; user_ell_width: rocsparse_int; partition_type: rocsparse_hyb_partition): rocsparse_status; cdecl; external;
function rocsparse_dcsr2hyb(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; hyb: rocsparse_hyb_mat; user_ell_width: rocsparse_int; partition_type: rocsparse_hyb_partition): rocsparse_status; cdecl; external;
function rocsparse_ccsr2hyb(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; hyb: rocsparse_hyb_mat; user_ell_width: rocsparse_int; partition_type: rocsparse_hyb_partition): rocsparse_status; cdecl; external;
function rocsparse_zcsr2hyb(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; hyb: rocsparse_hyb_mat; user_ell_width: rocsparse_int; partition_type: rocsparse_hyb_partition): rocsparse_status; cdecl; external;

//
// rocsparse_hyb2csr.h
//
function rocsparse_hyb2csr_buffer_size(handle: rocsparse_handle; descr: rocsparse_mat_descr; hyb: rocsparse_hyb_mat; csr_row_ptr: Procsparse_int; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_shyb2csr(handle: rocsparse_handle; descr: rocsparse_mat_descr; hyb: rocsparse_hyb_mat; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dhyb2csr(handle: rocsparse_handle; descr: rocsparse_mat_descr; hyb: rocsparse_hyb_mat; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_chyb2csr(handle: rocsparse_handle; descr: rocsparse_mat_descr; hyb: rocsparse_hyb_mat; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zhyb2csr(handle: rocsparse_handle; descr: rocsparse_mat_descr; hyb: rocsparse_hyb_mat; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_csrsort.h
//
function rocsparse_csrsort_buffer_size(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_csrsort(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; perm: Procsparse_int; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_coosort.h
//
function rocsparse_coosort_buffer_size(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_coosort_by_row(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int; perm: Procsparse_int; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_coosort_by_column(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; coo_row_ind: Procsparse_int; coo_col_ind: Procsparse_int; perm: Procsparse_int; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse-extra.h
//
function rocsparse_create_identity_permutation(handle: rocsparse_handle; n: rocsparse_int; p: Procsparse_int): rocsparse_status; cdecl; external;

//
// rocsparse_cscsort.h
//
function rocsparse_cscsort_buffer_size(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; csc_col_ptr: Procsparse_int; csc_row_ind: Procsparse_int; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_cscsort(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csc_col_ptr: Procsparse_int; csc_row_ind: Procsparse_int; perm: Procsparse_int; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_prune_csr2csr_by_percentage.h
//
function rocsparse_sprune_csr2csr_by_percentage_buffer_size(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz_A: rocsparse_int; csr_descr_A: rocsparse_mat_descr; csr_val_A: PSingle; csr_row_ptr_A: Procsparse_int; csr_col_ind_A: Procsparse_int; percentage: Single; csr_descr_C: rocsparse_mat_descr; csr_val_C: PSingle; csr_row_ptr_C: Procsparse_int; csr_col_ind_C: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_dprune_csr2csr_by_percentage_buffer_size(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz_A: rocsparse_int; csr_descr_A: rocsparse_mat_descr; csr_val_A: PDouble; csr_row_ptr_A: Procsparse_int; csr_col_ind_A: Procsparse_int; percentage: Double; csr_descr_C: rocsparse_mat_descr; csr_val_C: PDouble; csr_row_ptr_C: Procsparse_int; csr_col_ind_C: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_sprune_csr2csr_nnz_by_percentage(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz_A: rocsparse_int; csr_descr_A: rocsparse_mat_descr; csr_val_A: PSingle; csr_row_ptr_A: Procsparse_int; csr_col_ind_A: Procsparse_int; percentage: Single; csr_descr_C: rocsparse_mat_descr; csr_row_ptr_C: Procsparse_int; nnz_total_dev_host_ptr: Procsparse_int; info: rocsparse_mat_info; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dprune_csr2csr_nnz_by_percentage(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz_A: rocsparse_int; csr_descr_A: rocsparse_mat_descr; csr_val_A: PDouble; csr_row_ptr_A: Procsparse_int; csr_col_ind_A: Procsparse_int; percentage: Double; csr_descr_C: rocsparse_mat_descr; csr_row_ptr_C: Procsparse_int; nnz_total_dev_host_ptr: Procsparse_int; info: rocsparse_mat_info; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_sprune_csr2csr_by_percentage(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz_A: rocsparse_int; csr_descr_A: rocsparse_mat_descr; csr_val_A: PSingle; csr_row_ptr_A: Procsparse_int; csr_col_ind_A: Procsparse_int; percentage: Single; csr_descr_C: rocsparse_mat_descr; csr_val_C: PSingle; csr_row_ptr_C: Procsparse_int; csr_col_ind_C: Procsparse_int; info: rocsparse_mat_info; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dprune_csr2csr_by_percentage(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz_A: rocsparse_int; csr_descr_A: rocsparse_mat_descr; csr_val_A: PDouble; csr_row_ptr_A: Procsparse_int; csr_col_ind_A: Procsparse_int; percentage: Double; csr_descr_C: rocsparse_mat_descr; csr_val_C: PDouble; csr_row_ptr_C: Procsparse_int; csr_col_ind_C: Procsparse_int; info: rocsparse_mat_info; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_prune_csr2csr.h
//
function rocsparse_sprune_csr2csr_buffer_size(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz_A: rocsparse_int; csr_descr_A: rocsparse_mat_descr; csr_val_A: PSingle; csr_row_ptr_A: Procsparse_int; csr_col_ind_A: Procsparse_int; threshold: PSingle; csr_descr_C: rocsparse_mat_descr; csr_val_C: PSingle; csr_row_ptr_C: Procsparse_int; csr_col_ind_C: Procsparse_int; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_dprune_csr2csr_buffer_size(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz_A: rocsparse_int; csr_descr_A: rocsparse_mat_descr; csr_val_A: PDouble; csr_row_ptr_A: Procsparse_int; csr_col_ind_A: Procsparse_int; threshold: PDouble; csr_descr_C: rocsparse_mat_descr; csr_val_C: PDouble; csr_row_ptr_C: Procsparse_int; csr_col_ind_C: Procsparse_int; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_sprune_csr2csr_nnz(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz_A: rocsparse_int; csr_descr_A: rocsparse_mat_descr; csr_val_A: PSingle; csr_row_ptr_A: Procsparse_int; csr_col_ind_A: Procsparse_int; threshold: PSingle; csr_descr_C: rocsparse_mat_descr; csr_row_ptr_C: Procsparse_int; nnz_total_dev_host_ptr: Procsparse_int; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dprune_csr2csr_nnz(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz_A: rocsparse_int; csr_descr_A: rocsparse_mat_descr; csr_val_A: PDouble; csr_row_ptr_A: Procsparse_int; csr_col_ind_A: Procsparse_int; threshold: PDouble; csr_descr_C: rocsparse_mat_descr; csr_row_ptr_C: Procsparse_int; nnz_total_dev_host_ptr: Procsparse_int; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_sprune_csr2csr(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz_A: rocsparse_int; csr_descr_A: rocsparse_mat_descr; csr_val_A: PSingle; csr_row_ptr_A: Procsparse_int; csr_col_ind_A: Procsparse_int; threshold: PSingle; csr_descr_C: rocsparse_mat_descr; csr_val_C: PSingle; csr_row_ptr_C: Procsparse_int; csr_col_ind_C: Procsparse_int; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dprune_csr2csr(handle: rocsparse_handle; m: rocsparse_int; n: rocsparse_int; nnz_A: rocsparse_int; csr_descr_A: rocsparse_mat_descr; csr_val_A: PDouble; csr_row_ptr_A: Procsparse_int; csr_col_ind_A: Procsparse_int; threshold: PDouble; csr_descr_C: rocsparse_mat_descr; csr_val_C: PDouble; csr_row_ptr_C: Procsparse_int; csr_col_ind_C: Procsparse_int; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// Preconditioners
//

//
// rocsparse_bsric0.h
//
function rocsparse_bsric0_zero_pivot(handle: rocsparse_handle; info: rocsparse_mat_info; position: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_sbsric0_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_dbsric0_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_cbsric0_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_zbsric0_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_sbsric0_analysis(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dbsric0_analysis(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_cbsric0_analysis(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zbsric0_analysis(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_bsric0_clear(handle: rocsparse_handle; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_sbsric0(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dbsric0(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_cbsric0(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zbsric0(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_bsrilu0.h
//
function rocsparse_bsrilu0_zero_pivot(handle: rocsparse_handle; info: rocsparse_mat_info; position: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_sbsrilu0_numeric_boost(handle: rocsparse_handle; info: rocsparse_mat_info; enable_boost: cint; boost_tol: PSingle; boost_val: PSingle): rocsparse_status; cdecl; external;
function rocsparse_dbsrilu0_numeric_boost(handle: rocsparse_handle; info: rocsparse_mat_info; enable_boost: cint; boost_tol: PDouble; boost_val: PDouble): rocsparse_status; cdecl; external;
function rocsparse_cbsrilu0_numeric_boost(handle: rocsparse_handle; info: rocsparse_mat_info; enable_boost: cint; boost_tol: PSingle; boost_val: Procsparse_float_complex): rocsparse_status; cdecl; external;
function rocsparse_zbsrilu0_numeric_boost(handle: rocsparse_handle; info: rocsparse_mat_info; enable_boost: cint; boost_tol: PDouble; boost_val: Procsparse_double_complex): rocsparse_status; cdecl; external;
function rocsparse_sbsrilu0_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_dbsrilu0_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_cbsrilu0_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_zbsrilu0_buffer_size(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_sbsrilu0_analysis(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dbsrilu0_analysis(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_cbsrilu0_analysis(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zbsrilu0_analysis(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_bsrilu0_clear(handle: rocsparse_handle; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_sbsrilu0(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PSingle; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dbsrilu0(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: PDouble; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_cbsrilu0(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_float_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zbsrilu0(handle: rocsparse_handle; dir: rocsparse_direction; mb: rocsparse_int; nnzb: rocsparse_int; descr: rocsparse_mat_descr; bsr_val: Procsparse_double_complex; bsr_row_ptr: Procsparse_int; bsr_col_ind: Procsparse_int; block_dim: rocsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_csric0.h
//
function rocsparse_csric0_zero_pivot(handle: rocsparse_handle; info: rocsparse_mat_info; position: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_scsric0_buffer_size(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_dcsric0_buffer_size(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_ccsric0_buffer_size(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_zcsric0_buffer_size(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_scsric0_analysis(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dcsric0_analysis(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_ccsric0_analysis(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zcsric0_analysis(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_csric0_clear(handle: rocsparse_handle; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_scsric0(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dcsric0(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_ccsric0(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zcsric0(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;

//
// rocsparse_csrilu0.h
//
function rocsparse_csrilu0_zero_pivot(handle: rocsparse_handle; info: rocsparse_mat_info; position: Procsparse_int): rocsparse_status; cdecl; external;
function rocsparse_scsrilu0_buffer_size(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_dcsrilu0_buffer_size(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_ccsrilu0_buffer_size(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_zcsrilu0_buffer_size(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; buffer_size: PSize_t): rocsparse_status; cdecl; external;
function rocsparse_scsrilu0_analysis(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dcsrilu0_analysis(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_ccsrilu0_analysis(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zcsrilu0_analysis(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; analysis: rocsparse_analysis_policy; solve: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_csrilu0_clear(handle: rocsparse_handle; info: rocsparse_mat_info): rocsparse_status; cdecl; external;
function rocsparse_scsrilu0(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PSingle; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_dcsrilu0(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: PDouble; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_ccsrilu0(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_float_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;
function rocsparse_zcsrilu0(handle: rocsparse_handle; m: rocsparse_int; nnz: rocsparse_int; descr: rocsparse_mat_descr; csr_val: Procsparse_double_complex; csr_row_ptr: Procsparse_int; csr_col_ind: Procsparse_int; info: rocsparse_mat_info; policy: rocsparse_solve_policy; temp_buffer: Pointer): rocsparse_status; cdecl; external;

implementation

end.
