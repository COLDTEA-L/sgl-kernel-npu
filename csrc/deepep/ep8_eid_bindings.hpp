#pragma once
#include "ep8_eid_host.hpp"
#include "ep8_relay_planner_bindings.hpp"
#include "deep_ep.hpp"
#include <torch_npu/csrc/core/npu/NPUStream.h>
namespace deep_ep::ep8 {
void SubmitEidTransfer(const at::Tensor&,const at::Tensor&,int64_t,int64_t,HcclComm,const Plan&,U);
void SynchronizeEidPlan(int64_t);
class NpuEidRuntime : public EidRuntime {
public:
    NpuEidRuntime(int rank,Buffer& buffer,int64_t handle)
        :rank_(rank),handle_(handle),comm_(buffer.ep8_pipeline_comm()),stream_(c10_npu::getCurrentNPUStream()) {
        TORCH_CHECK(buffer.rank==rank && handle>0,"invalid EID attachment");
    }
    void Check(const at::Tensor& t) override {
        TORCH_CHECK(t.device()==stream_.device() && c10_npu::getCurrentNPUStream()==stream_,"EID host requires original caller stream/device");
    }
    void Sync() override {stream_.synchronize();SynchronizeEidPlan(handle_);}
    void Transfer(const at::Tensor& src,const at::Tensor& dst,const Plan& plan,U rb) override {
        SubmitEidTransfer(src,dst,handle_,rank_,comm_,plan,rb);
    }
private:
    int rank_;int64_t handle_;HcclComm comm_;c10_npu::NPUStream stream_;
};
inline void BindEid(pybind11::module_& m) {
    namespace py=pybind11;
    m.attr("ep8_eid_host_abi")=1;
    py::class_<EidPrepared,std::shared_ptr<EidPrepared>>(m,"Ep8EidPrepared")
        .def_readonly("row_bytes",&EidPrepared::rowBytes)
        .def_property_readonly("uses_relay",&EidPrepared::UsesRelay)
        .def_property_readonly("chunks",[](const EidPrepared&){return 0;})
        .def("describe",[](const EidPrepared& p){return py::make_tuple(Scalars(p.plan.counts),Scalars(p.plan.relays),Scalars(p.plan.moved),Rows(p.plan.actions));});
    py::class_<EidHost,std::shared_ptr<EidHost>>(m,"Ep8EidHost")
        .def("prepare",[](EidHost& h,const py::object& c,const at::Tensor& send){auto matrix=HostCounts(c);py::gil_scoped_release release;return h.Prepare(matrix,send);},py::keep_alive<0,1>())
        .def("execute",&EidHost::Execute,py::call_guard<py::gil_scoped_release>())
        .def("run",[](EidHost& h,const py::object& c,const at::Tensor& send,const at::Tensor& recv){auto matrix=HostCounts(c);py::gil_scoped_release release;return h.Run(matrix,send,recv);})
        .def("close",&EidHost::Close,py::call_guard<py::gil_scoped_release>())
        .def("stats",[](const EidHost& h){auto s=h.Stats();py::dict d;
            d["generation"]=s[0];d["tensor_allocations"]=s[1];d["event_allocations"]=s[2];d["reserved_bytes"]=s[3];
            d["poisoned"]=bool(s[4]);d["zero_backing_bytes"]=s[5];d["closed"]=bool(s[6]);return d;})
        .def("payload_stats",[](const EidHost& h){py::list out;for(const auto& p:h.PayloadStats()){py::dict d;
            d["device"]=p.device;d["dtype"]=p.dtype;d["row_shape"]=p.shape;d["row_bytes"]=p.rowBytes;d["reserved_bytes"]=0;out.append(d);}return out;});
    m.def("ep8_eid_create",[](const Planner& planner,int rank,Buffer& buffer,int64_t handle,U maxPayloads){
        return std::make_shared<EidHost>(planner,rank,std::make_shared<NpuEidRuntime>(rank,buffer,handle),maxPayloads);
    },py::arg("planner"),py::arg("rank"),py::arg("buffer"),py::arg("handle"),py::arg("max_payloads")=4,py::keep_alive<0,3>());
}
}
