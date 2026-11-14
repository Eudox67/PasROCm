cmake_minimum_required(VERSION 3.19)   # string(JSON) reads ci/gfx-targets.json

# Auto-detect a ROCm root unless ROCM_ROOT is given. Preference order matches
# build.sh: /opt/rocm first, then the newest /opt/rocm-<version> install.
#   cmake -DGPU_TARGET=gfx1100 -DROCM_ROOT=/opt/rocm-6.4.4 ..
if(NOT DEFINED ROCM_ROOT)
    if(EXISTS "/opt/rocm/bin/hipcc" OR EXISTS "/opt/rocm/lib/llvm/bin/amdclang++")
        set(ROCM_ROOT "/opt/rocm")
    else()
        file(GLOB _rocm_dirs LIST_DIRECTORIES true /opt/rocm-[0-9]*)
        list(SORT _rocm_dirs COMPARE NATURAL ORDER DESCENDING)
        if(_rocm_dirs)
            list(GET _rocm_dirs 0 ROCM_ROOT)
        endif()
    endif()
endif()
if(NOT DEFINED ROCM_ROOT)
    message(FATAL_ERROR "No ROCm found under /opt; pass -DROCM_ROOT=/path/to/rocm")
endif()
message(STATUS "Using ROCm: ${ROCM_ROOT}")

# ---------------------------------------------------------------------------
# One target per build. The GPU target decides the wavefront width, which is
# injected as a compile definition -- it is never guessed from preprocessor
# macros (DESIGN.md section 12.3). The mapping lives in exactly one file:
# ci/gfx-targets.json, shared with build.sh and the CI planner.
# ---------------------------------------------------------------------------
if(NOT DEFINED GPU_TARGET)
    message(FATAL_ERROR "Pass -DGPU_TARGET=gfxNNNN (one target per build; see ci/gfx-targets.json)")
endif()
if(GPU_TARGET MATCHES ";")
    message(FATAL_ERROR "GPU_TARGET must be a single target, got '${GPU_TARGET}'")
endif()

set(_target_table "${CMAKE_CURRENT_SOURCE_DIR}/../../ci/gfx-targets.json")
if(NOT EXISTS "${_target_table}")
    message(FATAL_ERROR "Missing target table: ${_target_table}")
endif()
file(READ "${_target_table}" _target_json)

string(JSON _wavefront ERROR_VARIABLE _json_error
       GET "${_target_json}" targets "${GPU_TARGET}" wavefront)
if(_json_error)
    message(FATAL_ERROR "Unknown GPU_TARGET '${GPU_TARGET}' (not in ci/gfx-targets.json)")
endif()

string(JSON _width_count ERROR_VARIABLE _json_error LENGTH
       "${_target_json}" targets "${GPU_TARGET}" widths)
set(_widths_mask 0)
if(NOT _json_error)
    math(EXPR _width_last "${_width_count} - 1")
    foreach(_index RANGE 0 ${_width_last})
        string(JSON _width GET "${_target_json}" targets "${GPU_TARGET}" widths ${_index})
        if(_width STREQUAL "32")
            math(EXPR _widths_mask "${_widths_mask} | 1")
        elseif(_width STREQUAL "64")
            math(EXPR _widths_mask "${_widths_mask} | 2")
        endif()
    endforeach()
endif()
if(_widths_mask EQUAL 0)
    message(FATAL_ERROR "ci/gfx-targets.json has no usable widths for '${GPU_TARGET}'")
endif()
message(STATUS "Target ${GPU_TARGET}: wavefront ${_wavefront}, widths mask ${_widths_mask}")

set(ROCM_VERSION "unknown" CACHE STRING "ROCm version stamped into the shim")

# The shim is HIP/C++, so the compiler must be hipcc or amdclang++ (a plain g++
# rejects --rocm-path). Honour an explicit -DCMAKE_CXX_COMPILER if given.
if(NOT CMAKE_CXX_COMPILER)
    if(EXISTS "${ROCM_ROOT}/bin/hipcc")
        set(CMAKE_CXX_COMPILER "${ROCM_ROOT}/bin/hipcc")
    elseif(EXISTS "${ROCM_ROOT}/lib/llvm/bin/amdclang++")
        set(CMAKE_CXX_COMPILER "${ROCM_ROOT}/lib/llvm/bin/amdclang++")
    else()
        message(FATAL_ERROR "Neither hipcc nor amdclang++ found under ${ROCM_ROOT}")
    endif()
endif()
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")

# hipcc injects -x hip itself; a bare amdclang++ needs it explicitly.
set(_hip_language_flags "")
if(NOT CMAKE_CXX_COMPILER MATCHES "hipcc")
    set(_hip_language_flags "-x hip")
endif()

project(pasrocm_shim_${GPU_TARGET} CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(pasrocm_shim_${GPU_TARGET} SHARED
    rocprim_shim.cpp
    rocprim_shim_math_kernels.cpp
)

target_compile_options(pasrocm_shim_${GPU_TARGET} PRIVATE
    --rocm-path=${ROCM_ROOT}
    --offload-arch=${GPU_TARGET}
    ${_hip_language_flags}
    -fvisibility=hidden
)

# SONAME is the on-disk file name, deliberately without a .so.N suffix: the
# Pascal loader resolves the library by exact name, so a versioned soname would
# diverge from it. ABI breaks are gated by ci/check-abi.sh --strict and the
# package version instead.
target_link_options(pasrocm_shim_${GPU_TARGET} PRIVATE
    "LINKER:-soname,libpasrocm_shim_${GPU_TARGET}.so"
    # -fvisibility=hidden does not hide __global__ kernels (HIP gives their host
    # stubs default visibility so the runtime can register them), so the version
    # script is what closes the dynamic symbol table to the 149 public names.
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/rocprim_shim.map"
)

# Only the table may select wave64: clang accepts -mwavefrontsize64 on gfx1100
# and silently folds the device width to 64, so this must never be a knob.
if(_wavefront EQUAL 64)
    target_compile_options(pasrocm_shim_${GPU_TARGET} PRIVATE -mwavefrontsize64)
endif()

target_compile_definitions(pasrocm_shim_${GPU_TARGET} PRIVATE
    PASROCM_GPU_TARGET="${GPU_TARGET}"
    PASROCM_WAVEFRONT_SIZE=${_wavefront}
    PASROCM_WAVEFRONT_WIDTHS_MASK=${_widths_mask}
    PASROCM_ROCM_VERSION="${ROCM_VERSION}"
)

# HIP headers come only from the selected ROCm install. Never add a local copy of
# the hip/ headers: mixing versions with the installed hip/amd_detail/*.h
# internals does not compile (ROCm 7.x reported 'unknown type name
# hipDriverEntryPointQueryResult').
target_include_directories(pasrocm_shim_${GPU_TARGET} PRIVATE ${ROCM_ROOT}/include)

find_library(AMD_HIP64 amdhip64
    PATHS ${ROCM_ROOT}/lib ${ROCM_ROOT}/lib64
    NO_DEFAULT_PATH
    REQUIRED)
target_link_libraries(pasrocm_shim_${GPU_TARGET} PRIVATE ${AMD_HIP64})

set_target_properties(pasrocm_shim_${GPU_TARGET} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "pasrocm_shim_${GPU_TARGET}"
)

install(TARGETS pasrocm_shim_${GPU_TARGET}
    LIBRARY DESTINATION lib
)
