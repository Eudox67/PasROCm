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

{ Bindings for rocRAND generators and distributions. }
{ Bindings for rocRAND generators and distributions. }
unit rocrand;

{$mode objfpc}{$H+}
{$linklib rocrand}

interface

uses
  hip_runtime, hip_types;

type
  rocrand_generator             = Pointer;
  rocrand_discrete_distribution = Pointer;

  Puint4 = ^uint4;
  uint4 = record
    x, y, z, w: UInt32;
  end;

  Phalf2 = ^half2;
  half2 = record
    packed_value: UInt32;
  end;

  rocrand_status                = Integer;
  rocrand_rng_type              = Integer;
  rocrand_ordering              = Integer;
  rocrand_direction_vector_set  = Integer;

const
  rocrand_status_success                    = 0;
  rocrand_status_version_mismatch           = 100;
  rocrand_status_not_created                = 101;
  rocrand_status_allocation_failed          = 102;
  rocrand_status_type_error                 = 103;
  rocrand_status_out_of_range               = 104;
  rocrand_status_length_not_multiple        = 105;
  rocrand_status_double_precision_required  = 106;
  rocrand_status_launch_failure             = 107;
  rocrand_status_internal_error             = 108;

  rocrand_rng_pseudo_default                = 400;
  rocrand_rng_pseudo_xorwow                 = 401;
  rocrand_rng_pseudo_mrg32k3a               = 402;
  rocrand_rng_pseudo_mtgp32                 = 403;
  rocrand_rng_pseudo_philox4_32_10          = 404;
  rocrand_rng_pseudo_mrg31k3p               = 405;
  rocrand_rng_pseudo_lfsr113                = 406;
  rocrand_rng_pseudo_mt19937                = 407;
  rocrand_rng_pseudo_threefry2_32_20        = 408;
  rocrand_rng_pseudo_threefry2_64_20        = 409;
  rocrand_rng_pseudo_threefry4_32_20        = 410;
  rocrand_rng_pseudo_threefry4_64_20        = 411;
  rocrand_rng_quasi_default                 = 500;
  rocrand_rng_quasi_sobol32                 = 501;
  rocrand_rng_quasi_scrambled_sobol32       = 502;
  rocrand_rng_quasi_sobol64                 = 504;
  rocrand_rng_quasi_scrambled_sobol64       = 505;

  rocrand_ordering_pseudo_best              = 100;
  rocrand_ordering_pseudo_default           = 101;
  rocrand_ordering_pseudo_seeded            = 102;
  rocrand_ordering_pseudo_legacy            = 103;
  rocrand_ordering_pseudo_dynamic           = 104;
  rocrand_ordering_quasi_default            = 201;

  rocrand_direction_vectors_32_joekuo6              = 101;
  rocrand_scrambled_direction_vectors_32_joekuo6    = 102;
  rocrand_direction_vectors_64_joekuo6              = 103;
  rocrand_scrambled_direction_vectors_64_joekuo6    = 104;

{ ---------- Generator life-cycle ---------- }
function rocrand_create_generator(var generator: rocrand_generator; rng_type: rocrand_rng_type): rocrand_status; cdecl; external;
function rocrand_create_generator_host(var generator: rocrand_generator; rng_type: rocrand_rng_type): rocrand_status; cdecl; external;
function rocrand_create_generator_host_blocking(var generator: rocrand_generator; rng_type: rocrand_rng_type): rocrand_status; cdecl; external;
function rocrand_destroy_generator(generator: rocrand_generator): rocrand_status; cdecl; external;

{ ---------- Seed / stream / offset ---------- }
function rocrand_initialize_generator(generator: rocrand_generator): rocrand_status; cdecl; external;
function rocrand_set_stream(generator: rocrand_generator; const stream: Pointer): rocrand_status; cdecl; external;
function rocrand_set_seed(generator: rocrand_generator; seed: UInt64): rocrand_status; cdecl; external;
function rocrand_set_seed_uint4(generator: rocrand_generator; const seed: uint4): rocrand_status; cdecl; external; // caller must pack a uint4
function rocrand_set_offset(generator: rocrand_generator; offset: UInt64): rocrand_status; cdecl; external;
function rocrand_set_ordering(generator: rocrand_generator; order: rocrand_ordering): rocrand_status; cdecl; external;
function rocrand_set_quasi_random_generator_dimensions(generator: rocrand_generator; dimensions: UInt32): rocrand_status; cdecl; external;

{ ---------- Generator queries ---------- }
function rocrand_get_version(var version: Integer): rocrand_status; cdecl; external;

{ ---------- Discrete distributions ---------- }
function rocrand_create_poisson_distribution(lambda: Double; var discrete_distribution: rocrand_discrete_distribution): rocrand_status; cdecl; external;
function rocrand_create_discrete_distribution(const probabilities: PDouble; size: UInt32; offset: UInt32; var discrete_distribution: rocrand_discrete_distribution): rocrand_status; cdecl; external;
function rocrand_destroy_discrete_distribution(discrete_distribution: rocrand_discrete_distribution): rocrand_status; cdecl; external;

{ ---------- Direction-vector helpers ---------- }
function rocrand_get_direction_vectors32(var vectors: PUInt32; vset: rocrand_direction_vector_set): rocrand_status; cdecl; external;
function rocrand_get_direction_vectors64(var vectors: PUInt64; vset: rocrand_direction_vector_set): rocrand_status; cdecl; external;
function rocrand_get_scramble_constants32(var constants: PUInt32): rocrand_status; cdecl; external;
function rocrand_get_scramble_constants64(var constants: PUInt64): rocrand_status; cdecl; external;

{ ---------- Host output generators ---------- }
function rocrand_generate(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr): rocrand_status; cdecl; external;
function rocrand_generate_long_long(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr): rocrand_status; cdecl; external;
function rocrand_generate_char(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr): rocrand_status; cdecl; external;
function rocrand_generate_short(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr): rocrand_status; cdecl; external;
function rocrand_generate_uniform(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr): rocrand_status; cdecl; external;
function rocrand_generate_uniform_double(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr): rocrand_status; cdecl; external;
function rocrand_generate_uniform_half(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr): rocrand_status; cdecl; external;
function rocrand_generate_normal(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr; mean: Single; stddev: Single): rocrand_status; cdecl; external;
function rocrand_generate_normal_double(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr; mean: Double; stddev: Double): rocrand_status; cdecl; external;
function rocrand_generate_normal_half(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr; mean: half2; stddev: half2): rocrand_status; cdecl; external;
function rocrand_generate_log_normal(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr; mean: Single; stddev: Single): rocrand_status; cdecl; external;
function rocrand_generate_log_normal_double(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr; mean: Double; stddev: Double): rocrand_status; cdecl; external;
function rocrand_generate_log_normal_half(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr; mean: half2; stddev: half2): rocrand_status; cdecl; external;
function rocrand_generate_poisson(generator: rocrand_generator; output: hipDeviceptr_t; n: UIntPtr; lambda: Double): rocrand_status; cdecl; external;

implementation

end.
