# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18.0)
project(a5_ccu_urma_route_probe)
if(NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}/include/hcomm/ccu/ccu_launch.h")
    message(FATAL_ERROR "This branch requires public CCU headers (CANN 9.2/9.3); use the original branch for 9.1")
endif()

set(HCCL_CUSTOM_OPS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(HCCL_CUSTOM_OPS_INCLUDE_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_ccu
    # CANN Toolkit头文件
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
)

# Host 动态链接库
add_library(a5_ccu_urma_route_probe SHARED)
target_include_directories(a5_ccu_urma_route_probe PRIVATE
    ${HCCL_CUSTOM_OPS_INCLUDE_LIST}
)
target_compile_options(a5_ccu_urma_route_probe PRIVATE
    -U_FORTIFY_SOURCE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g -O0
    -std=c++14
    -fstack-protector-all
)
target_compile_definitions(a5_ccu_urma_route_probe PRIVATE
    -DHOST_COMPILE
)
target_link_options(a5_ccu_urma_route_probe PRIVATE
    -Wl,-z,defs
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)
target_link_directories(a5_ccu_urma_route_probe PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
target_link_libraries(a5_ccu_urma_route_probe PRIVATE
    hcomm
    c_sec
    ascendcl
    ${CMAKE_DL_LIBS}
)
install(TARGETS a5_ccu_urma_route_probe
    LIBRARY DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH}
    COMPONENT hccl
)
install(FILES
    inc/a5_ccu_urma_route_probe.h
    inc/a5_uvs_source_route_provider.h
    DESTINATION ${CUSTOM_OPS_OPP_INC_PATH}
    COMPONENT hccl
)

# 添加子目录
add_subdirectory(op_host)
add_subdirectory(op_kernel_ccu)

option(A5_CCU_BUILD_HOST_TESTS "Test public API host lifecycle without NPU" OFF)
if(A5_CCU_BUILD_HOST_TESTS)
    enable_testing()
    add_executable(a5_ccu_public_host_test
        ../../tests/cpp/test_a5_ccu_public_api.cpp op_host/public_ccu.cc)
    target_include_directories(a5_ccu_public_host_test SYSTEM PRIVATE ${HCCL_CUSTOM_OPS_INCLUDE_LIST})
    target_compile_options(a5_ccu_public_host_test PRIVATE -std=c++14 -Wall -Wextra -Werror -UNDEBUG)
    target_link_directories(a5_ccu_public_host_test PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
    target_link_libraries(a5_ccu_public_host_test PRIVATE hcomm c_sec ascendcl ${CMAKE_DL_LIBS})
    add_test(NAME public_ccu_host COMMAND a5_ccu_public_host_test)
    add_executable(a5_ccu_public_local_copy_probe testcase/local_copy_probe.cc
        op_host/public_ccu.cc op_kernel_ccu/grouped_local_copy_kernel.cc)
    target_include_directories(a5_ccu_public_local_copy_probe SYSTEM PRIVATE ${HCCL_CUSTOM_OPS_INCLUDE_LIST})
    target_compile_options(a5_ccu_public_local_copy_probe PRIVATE -std=c++14 -Wall -Wextra -Werror)
    target_link_directories(a5_ccu_public_local_copy_probe PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
    target_link_libraries(a5_ccu_public_local_copy_probe PRIVATE hccl hcomm c_sec ascendcl msprofiler ${CMAKE_DL_LIBS})
endif()
