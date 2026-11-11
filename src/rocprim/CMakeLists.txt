cmake_minimum_required(VERSION 3.18)

# Auto-detect a ROCm root unless ROCM_ROOT is given. Preference order matches
# build.sh: /opt/rocm first, then the newest /opt/rocm-<version> install.
#   cmake -DROCM_ROOT=/opt/rocm-6.3.2 ..
if(NOT DEFINED ROCM_ROOT)
    if(EXISTS "/opt/rocm/bin/hipcc")
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

# The shim is HIP/C++, so hipcc must be the compiler; --rocm-path below is a
# clang flag and g++ rejects it.
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${ROCM_ROOT}/bin/hipcc")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --rocm-path=${ROCM_ROOT}")

project(rocprim_shim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${ROCM_ROOT}/include)
# HIP headers come only from the selected ROCm install. Never add a local copy of
# the hip/ headers: mixing versions with the installed hip/amd_detail/*.h
# internals does not compile (ROCm 7.x reported 'unknown type name
# hipDriverEntryPointQueryResult').

add_library(rocprim_shim SHARED rocprim_shim.cpp rocprim_shim_math_kernels.cpp)

find_library(AMD_HIP64 amdhip64
    PATHS ${ROCM_ROOT}/lib ${ROCM_ROOT}/lib64
    NO_DEFAULT_PATH
    REQUIRED)
target_link_libraries(rocprim_shim PRIVATE ${AMD_HIP64})

set_target_properties(rocprim_shim PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS rocprim_shim
    LIBRARY DESTINATION lib
)
