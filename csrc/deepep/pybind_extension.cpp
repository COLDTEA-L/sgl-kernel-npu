#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>

#include "deep_ep.hpp"
#include "config.hpp"
#include "event.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME deep_ep_cpp
#endif

namespace py = pybind11;

TORCH_LIBRARY_FRAGMENT(deep_ep, m)
{
    m.def(
        "ccu_urma_prepared_multipath_alltoall(Tensor send_data, Tensor(a!) recv_data, int plan_handle) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(deep_ep, PrivateUse1, m)
{
    m.impl("ccu_urma_prepared_multipath_alltoall",
           TORCH_FN(deep_ep::ccu_urma_prepared_multipath_alltoall_op));
}

TORCH_LIBRARY_IMPL(deep_ep, Meta, m)
{
    m.impl("ccu_urma_prepared_multipath_alltoall",
           TORCH_FN(deep_ep::ccu_urma_prepared_multipath_alltoall_meta));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(), py::arg("num_sms") = 20,
             py::arg("num_max_nvl_chunked_send_tokens") = 6, py::arg("num_max_nvl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6, py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &deep_ep::get_low_latency_rdma_size_hint);

    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, std::string>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("get_notify_send_data", &deep_ep::Buffer::get_notify_send_data)
        .def("hccl_all2_all_ccu", &deep_ep::Buffer::hccl_all2_all_ccu)
        .def("explicit_multipath_all2all_ccu", &deep_ep::Buffer::explicit_multipath_all2all_ccu,
             py::arg("send_data"), py::arg("plan_id"), py::arg("path_weights"),
             "Graphable A5 CCU+URMA AllToAll using a pre-provisioned explicit path plan")
        .def("ccu_urma_multiroute_write", &deep_ep::Buffer::ccu_urma_multiroute_write)
        .def("ccu_urma_multiroute_alltoall", &deep_ep::Buffer::ccu_urma_multiroute_alltoall)
        .def("ccu_urma_multiroute_alltoall_out", &deep_ep::Buffer::ccu_urma_multiroute_alltoall_out)
        .def("ccu_urma_explicit_multipath_alltoall",
             &deep_ep::Buffer::ccu_urma_explicit_multipath_alltoall,
             py::arg("send_data"), py::arg("relay_manifest"),
             py::arg("direct_route"), py::arg("path_weights"))
        .def("ccu_urma_explicit_multipath_alltoall_out",
             &deep_ep::Buffer::ccu_urma_explicit_multipath_alltoall_out,
             py::arg("send_data"), py::arg("recv_data"),
             py::arg("relay_manifest"), py::arg("direct_route"),
             py::arg("path_weights"))
        .def("prepare_ccu_urma_explicit_multipath_plan",
             &deep_ep::Buffer::prepare_ccu_urma_explicit_multipath_plan,
             py::arg("plan_id"), py::arg("relay_manifest"),
             py::arg("direct_route"), py::arg("path_weights"),
             "Prepare explicit CommLinks, Channels and a CCU kernel outside graph execution")
        .def("ccu_urma_prepared_multipath_alltoall",
             &deep_ep::Buffer::ccu_urma_prepared_multipath_alltoall,
             py::arg("send_data"), py::arg("plan_handle"),
             "Launch a previously prepared explicit multipath CCU plan")
        .def("ccu_urma_prepared_multipath_alltoall_out",
             &deep_ep::Buffer::ccu_urma_prepared_multipath_alltoall_out,
             py::arg("send_data"), py::arg("recv_data"), py::arg("plan_handle"),
             "Out variant of the prepared explicit multipath CCU plan")
        .def("all2_all_detour_io_die", &deep_ep::Buffer::all2_all_detour_io_die)
        .def("clean_low_latency_buffer", &deep_ep::Buffer::clean_low_latency_buffer)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("notify_verify", &deep_ep::Buffer::notify_verify)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine)
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine)
        .def("low_latency_dispatch", &deep_ep::Buffer::low_latency_dispatch)
        .def("low_latency_combine", &deep_ep::Buffer::low_latency_combine)
        .def("fused_deep_moe", &deep_ep::Buffer::fused_deep_moe, py::arg("x"), py::arg("expert_ids"),
             py::arg("gmm1_permuted_weight"), py::arg("gmm1_permuted_weight_scale"), py::arg("gmm2_weight"),
             py::arg("gmm2_weight_scale"), py::arg("expert_scales_optional"),
             py::arg("num_max_dispatch_tokens_per_rank"), py::arg("num_experts"), py::arg("quant_mode"),
             py::arg("profile_enable") = false)
        .def("begin_profile", &deep_ep::Buffer::begin_profile, py::arg("num_profile_skip_launches"),
             py::arg("num_profile_active_launches"), py::arg("profile_trace_dir") = "")
        .def("end_profile", &deep_ep::Buffer::end_profile)
        .def("dispatch_ffn_combine", &deep_ep::Buffer::dispatch_ffn_combine);
}
