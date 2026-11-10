cmake_minimum_required(VERSION 3.18)
project(a5_ccu_l0_l1_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT IS_DIRECTORY "${ASCEND_CANN_PACKAGE_PATH}/include/hcomm/ccu")
  message(FATAL_ERROR "Select an explicit CANN 9.2/9.3 installation")
endif()
add_executable(a5_ccu_l0_l1_probe main.cc)
target_compile_options(a5_ccu_l0_l1_probe PRIVATE -O2 -Wall -Wextra -Werror)
target_include_directories(a5_ccu_l0_l1_probe SYSTEM PRIVATE
  ${ASCEND_CANN_PACKAGE_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
  ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm
  ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm/ccu
  ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu)
foreach(component hccl hcomm ascendcl c_sec)
  find_library(${component}_LIB NAMES ${component}
    PATHS ${ASCEND_CANN_PACKAGE_PATH}/lib64 NO_DEFAULT_PATH REQUIRED)
  target_link_libraries(a5_ccu_l0_l1_probe PRIVATE ${${component}_LIB})
endforeach()
target_link_libraries(a5_ccu_l0_l1_probe PRIVATE ${CMAKE_DL_LIBS} pthread)
set_target_properties(a5_ccu_l0_l1_probe PROPERTIES BUILD_RPATH "${ASCEND_CANN_PACKAGE_PATH}/lib64")
