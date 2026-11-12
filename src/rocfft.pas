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

{ Bindings for rocFFT plans and real/complex transforms. }
{ Bindings for rocFFT plans and real/complex transforms. }
unit rocfft;

{$mode objfpc}{$H+}
{$linklib rocfft}


interface

uses
  hip_runtime;

type
  rocfft_plan                   = Pointer;
  rocfft_plan_description       = Pointer;
  rocfft_execution_info         = Pointer;
  rocfft_field                  = Pointer;
  rocfft_brick                  = Pointer;

  rocfft_status                 = Integer;
  rocfft_transform_type         = Integer;
  rocfft_precision              = Integer;
  rocfft_result_placement       = Integer;
  rocfft_array_type             = Integer;
  rocfft_comm_type              = Integer;

const
  rocfft_status_success               = 0;
  rocfft_status_failure               = 1;
  rocfft_status_invalid_arg_value     = 2;
  rocfft_status_invalid_dimensions    = 3;
  rocfft_status_invalid_array_type    = 4;
  rocfft_status_invalid_strides       = 5;
  rocfft_status_invalid_distance      = 6;
  rocfft_status_invalid_offset        = 7;
  rocfft_status_invalid_work_buffer   = 8;

  { NOTE ON NORMALIZATION:
    rocFFT follows the convention of many high-performance FFT libraries:
    - Forward transforms are unscaled.
    - Inverse transforms are unscaled (i.e. they are not multiplied by 1/N).
    The user is responsible for scaling the result of an inverse transform by 1/N.
  }
  rocfft_transform_type_complex_forward  = 0;
  rocfft_transform_type_complex_inverse  = 1;
  rocfft_transform_type_real_forward     = 2;
  rocfft_transform_type_real_inverse     = 3;

  rocfft_precision_single = 0;
  rocfft_precision_double = 1;
  rocfft_precision_half   = 2;

  rocfft_placement_inplace      = 0;
  rocfft_placement_notinplace   = 1;

  rocfft_array_type_complex_interleaved   = 0;
  rocfft_array_type_complex_planar        = 1;
  rocfft_array_type_real                  = 2;
  rocfft_array_type_hermitian_interleaved = 3;
  rocfft_array_type_hermitian_planar      = 4;
  rocfft_array_type_unset                 = 5;

  rocfft_comm_none = 0;
  rocfft_comm_mpi  = 1;

{ ---------- Library life-cycle ---------- }
function rocfft_setup: rocfft_status; cdecl; external;
function rocfft_cleanup: rocfft_status; cdecl; external;

{ ---------- Plan creation / destruction ---------- }
function rocfft_plan_create(var plan: rocfft_plan; placement: rocfft_result_placement; transform_type: rocfft_transform_type; precision: rocfft_precision; dimensions: UIntPtr; const lengths: PUIntPtr; number_of_transforms: UIntPtr; description: rocfft_plan_description): rocfft_status; cdecl; external;
function rocfft_plan_destroy(plan: rocfft_plan): rocfft_status; cdecl; external;

{ ---------- Execution ---------- }
function rocfft_execute(const plan: rocfft_plan; in_buffer: PPointer; out_buffer: PPointer; info: rocfft_execution_info): rocfft_status; cdecl; external;

{ ---------- Plan description ---------- }
function rocfft_plan_description_create(var description: rocfft_plan_description): rocfft_status; cdecl; external;
function rocfft_plan_description_destroy(description: rocfft_plan_description): rocfft_status; cdecl; external;
function rocfft_plan_description_set_scale_factor(description: rocfft_plan_description; const scale_factor: Double): rocfft_status; cdecl; external;
function rocfft_plan_description_set_data_layout(description: rocfft_plan_description; in_array_type, out_array_type: rocfft_array_type; const in_offsets, out_offsets: PUIntPtr; in_strides_size: UIntPtr; const in_strides: PUIntPtr; in_distance: UIntPtr; out_strides_size: UIntPtr; const out_strides: PUIntPtr; out_distance: UIntPtr): rocfft_status; cdecl; external;
function rocfft_plan_description_set_comm(description: rocfft_plan_description; comm_type: rocfft_comm_type; comm_handle: Pointer): rocfft_status; cdecl; external;

{ ---------- Execution info ---------- }
function rocfft_execution_info_create(var info: rocfft_execution_info): rocfft_status; cdecl; external;
function rocfft_execution_info_destroy(info: rocfft_execution_info): rocfft_status; cdecl; external;
function rocfft_execution_info_set_work_buffer(info: rocfft_execution_info; work_buffer: Pointer; size_in_bytes: UIntPtr): rocfft_status; cdecl; external;
function rocfft_execution_info_set_stream(info: rocfft_execution_info; stream: Pointer): rocfft_status; cdecl; external;
function rocfft_execution_info_set_load_callback(info: rocfft_execution_info; cb_functions, cb_data: PPointer; shared_mem_bytes: UIntPtr): rocfft_status; cdecl; external;
function rocfft_execution_info_set_store_callback(info: rocfft_execution_info; cb_functions, cb_data: PPointer; shared_mem_bytes: UIntPtr): rocfft_status; cdecl; external;

{ ---------- Work buffer query ---------- }
function rocfft_plan_get_work_buffer_size(const plan: rocfft_plan; var size_in_bytes: UIntPtr): rocfft_status; cdecl; external;

{ ---------- Field / brick helpers ---------- }
function rocfft_field_create(var field: rocfft_field): rocfft_status; cdecl; external;
function rocfft_field_destroy(field: rocfft_field): rocfft_status; cdecl; external;
function rocfft_brick_create(var brick: rocfft_brick; const field_lower, field_upper, brick_stride: PUIntPtr; dim: UIntPtr; deviceID: Integer): rocfft_status; cdecl; external;
function rocfft_brick_destroy(brick: rocfft_brick): rocfft_status; cdecl; external;
function rocfft_field_add_brick(field: rocfft_field; brick: rocfft_brick): rocfft_status; cdecl; external;
function rocfft_plan_description_add_infield(description: rocfft_plan_description; field: rocfft_field): rocfft_status; cdecl; external;
function rocfft_plan_description_add_outfield(description: rocfft_plan_description; field: rocfft_field): rocfft_status; cdecl; external;

{ ---------- Cache management ---------- }
function rocfft_cache_serialize(var buffer: Pointer; var buffer_len_bytes: UIntPtr): rocfft_status; cdecl; external;
function rocfft_cache_buffer_free(buffer: Pointer): rocfft_status; cdecl; external;
function rocfft_cache_deserialize(const buffer: Pointer; buffer_len_bytes: UIntPtr): rocfft_status; cdecl; external;

{ ---------- Version ---------- }
function rocfft_get_version_string(buf: PChar; len: UIntPtr): rocfft_status; cdecl; external;

implementation
end.
