cmake_minimum_required(VERSION 3.16)
project(sparse_matmul_lib)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(NPUSLIM_CXX11_ABI "0" CACHE STRING "Value for _GLIBCXX_USE_CXX11_ABI")

# SOC_VERSION must be passed by the build system (detected via acl.get_soc_name()).
# No default — if not set, the build should not proceed.
if(NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
    message(FATAL_ERROR
        "SOC_VERSION is not set. "
        "The build system should detect it via 'python3 -c \"import acl; print(acl.get_soc_name())\"' "
        "and pass -DSOC_VERSION=<result> to cmake."
    )
endif()

# ASCEND_CANN_PACKAGE_PATH must be set via ASCEND_HOME_PATH env var.
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE STRING "CANN package path")
if(NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}")
    message(FATAL_ERROR
        "ASCEND_HOME_PATH is not set or does not exist. "
        "Please source the CANN environment (set_env.sh) before building."
    )
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CSRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# --- Compile AscendC kernel ---
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found in CANN package")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ascendc_kernels_${RUN_MODE} STATIC
    ${CSRC_DIR}/matmul_sparse_custom_kernel.cpp
)
ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
    ASCENDC_DUMP HAVE_WORKSPACE HAVE_TILING
)

# --- Host wrapper shared library ---
add_library(sparse_matmul SHARED
    ${CSRC_DIR}/sparse_matmul_host.cpp
    ${CSRC_DIR}/matmul_sparse_custom_tiling.cpp
)

target_compile_options(sparse_matmul PRIVATE -O2 -std=c++17 -fPIC)
target_compile_definitions(sparse_matmul PRIVATE
    SOC_VERSION="${SOC_VERSION}"
    _GLIBCXX_USE_CXX11_ABI=${NPUSLIM_CXX11_ABI}
)

set(CANN_INCLUDE_DIRS
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include/adv_api
)

target_include_directories(sparse_matmul PRIVATE
    ${CANN_INCLUDE_DIRS}
    ${CSRC_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

target_link_libraries(sparse_matmul PRIVATE
    ascendc_kernels_${RUN_MODE}
    tiling_api register platform ascendalog dl
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64/libascendcl.so
)

if(NOT DEFINED TORCH_NPU_HOME OR TORCH_NPU_HOME STREQUAL "")
    message(FATAL_ERROR "TORCH_NPU_HOME is required for sparse_matmul TaskQueue")
endif()
if(NOT EXISTS "${TORCH_NPU_HOME}/include" OR NOT EXISTS "${TORCH_NPU_HOME}/lib/libtorch_npu.so")
    message(FATAL_ERROR "TORCH_NPU_HOME does not point to a torch_npu package: ${TORCH_NPU_HOME}")
endif()

find_package(Torch REQUIRED)

target_include_directories(sparse_matmul PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_HOME}/include
)
target_link_libraries(sparse_matmul PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_NPU_HOME}/lib/libtorch_npu.so
)
target_link_directories(sparse_matmul PRIVATE
    ${TORCH_NPU_HOME}/lib
)
set(NPUSLIM_TASKQUEUE_RPATH
    "$ORIGIN;${TORCH_NPU_HOME}/lib;${TORCH_INSTALL_PREFIX}/lib"
)

set_target_properties(sparse_matmul PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "sparse_matmul"
    BUILD_RPATH "${NPUSLIM_TASKQUEUE_RPATH}"
    INSTALL_RPATH "${NPUSLIM_TASKQUEUE_RPATH}"
)
