set(op_kernel_ccu_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/route_kernel.cc
)

target_sources(a5_ccu_urma_route_probe PRIVATE
    ${op_kernel_ccu_src_list}
)
