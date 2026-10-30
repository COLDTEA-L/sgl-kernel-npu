#pragma once
#include "ep8_pipeline_npu.hpp"
#include "ep8_relay_planner_bindings.hpp"

namespace deep_ep::ep8 {
// Explicit CPU-only test runtime: capture actual C++ driver submissions for an
// asynchronous eight-rank DAG simulator. It never simulates NPU elapsed time.
struct PipelineTrace {
    int op, lane; size_t event = 0;
    at::Tensor src, dst;
    Matrix counts{}, storage{};
    bool hasStorage = false;
    CopySpans copies;
};
class TracePipelineRuntime : public PipelineRuntime {
public:
    std::vector<PipelineTrace> trace;
    int active = 0, failAfter = -1;
    U syncs = 0;
    void Check(const at::Tensor& t) override { TORCH_CHECK(t.device().is_cpu() && active == 0, "trace requires CPU caller lane"); }
    void Sync() override { ++syncs; }
    void Lane(int lane) override { active = lane; }
    void Events(size_t) override {}
    void Record(size_t e, int l) override { PipelineTrace t{0, l}; t.event = e; trace.push_back(std::move(t)); }
    void Wait(size_t e, int l) override { PipelineTrace t{1, l}; t.event = e; trace.push_back(std::move(t)); }
    void Retain(const at::Tensor&, int) override {}
    void Copy(const at::Tensor& src, const at::Tensor& dst, const CopySpans& c) override {
        if (c.spans.empty()) return;
        PipelineTrace t{2, active}; t.src = src; t.dst = dst; t.copies = c; trace.push_back(std::move(t));
    }
    void Transfer(const at::Tensor& src, const at::Tensor& dst, const Matrix& c, const Matrix* storage, U) override {
        TORCH_CHECK(failAfter < 0 || --failAfter != 0, "injected pipeline submission failure");
        // Mirror vendor DisjointAllToAllVBuffers, including zero-row backing.
        auto address = [](const at::Tensor& t) {
            return reinterpret_cast<uintptr_t>(t.numel() ? t.data_ptr() : t.storage().data_ptr().get());
        };
        auto capacity = [](const at::Tensor& t) { return t.numel() ? t.nbytes() : t.storage().nbytes(); };
        auto a = address(src), b = address(dst), an = capacity(src), bn = capacity(dst);
        TORCH_CHECK(a && b && an && bn && (a <= b ? an <= b-a : bn <= a-b),
                    "trace vendor buffer capacity overlap (including zero rows)");
        PipelineTrace t{3, active}; t.src = src; t.dst = dst; t.counts = c;
        if (storage) { t.hasStorage = true; t.storage = *storage; }
        trace.push_back(std::move(t));
    }
};
inline void BindPipeline(pybind11::module_& m) {
    namespace py = pybind11;
    m.attr("ep8_pipeline_host_abi") = 3;
    py::class_<PipelinePrepared, std::shared_ptr<PipelinePrepared>>(m, "Ep8PipelinePrepared")
        .def_readonly("row_bytes", &PipelinePrepared::rowBytes)
        .def_property_readonly("uses_relay", &PipelinePrepared::UsesRelay)
        .def_property_readonly("chunks", [](const PipelinePrepared& p) { return p.layout.chunks.size(); })
        .def("copy_report", [](const PipelinePrepared& p) {
            py::dict d;
            d["bypassed_spans"] = p.bypassSpans; d["bypassed_bytes"] = p.bypassBytes;
            std::array<U, 3> spans{}, bytes{};
            for (const auto& c : p.layout.chunks) for (int i = 0; i < 3; ++i)
                for (const auto& v : c.localCopies[i].spans) { ++spans[i]; bytes[i] += v[2]; }
            // Rows are useful across dtype/width; bytes above use actual rowBytes.
            d["original_spans"] = spans; d["original_rows"] = bytes;
            return d;
        })
        .def("describe", [](const PipelinePrepared& p) {
            return py::make_tuple(Scalars(p.plan.counts), Scalars(p.plan.relays), Scalars(p.plan.moved), Rows(p.plan.actions));
        });
    py::class_<PipelineHost, std::shared_ptr<PipelineHost>>(m, "Ep8PipelineHost")
        .def("attach_lane", [](PipelineHost& h, int lane, Buffer& b, int64_t handle) {
            auto rt = dynamic_cast<NpuPipelineRuntime*>(h.Runtime().get());
            TORCH_CHECK(rt, "lane attachment only applies to NPU runtime"); rt->Attach(lane, b, handle);
        }, py::keep_alive<1, 3>())
        .def("prepare", [](PipelineHost& h, const py::object& counts, const at::Tensor& send) {
            Matrix c = HostCounts(counts); py::gil_scoped_release release; return h.Prepare(c, send);
        }, py::keep_alive<0, 1>())
        .def("execute", &PipelineHost::Execute, py::call_guard<py::gil_scoped_release>())
        .def("run", [](PipelineHost& h, const py::object& counts, const at::Tensor& send, const at::Tensor& recv) {
            Matrix c = HostCounts(counts); py::gil_scoped_release release; return h.Run(c, send, recv);
        })
        .def("close", &PipelineHost::Close, py::call_guard<py::gil_scoped_release>())
        .def("stats", [](const PipelineHost& h) {
            auto v = h.Stats(); py::dict d;
            d["generation"] = v[0]; d["tensor_allocations"] = v[1]; d["event_allocations"] = v[2];
            d["reserved_bytes"] = v[3]; d["poisoned"] = bool(v[4]);
            d["zero_backing_bytes"] = v[5]; d["closed"] = bool(v[6]); return d;
        })
        .def("payload_stats", [](const PipelineHost& h) {
            py::list result;
            for (const auto& p : h.PayloadStats()) {
                py::dict d;
                d["device"] = p.device.str(); d["dtype"] = c10::toString(p.dtype); d["row_shape"] = p.shape;
                d["row_bytes"] = p.rowBytes; d["reserved_bytes"] = p.stats[3];
                d["tensor_allocations"] = p.stats[1]; d["generation"] = p.stats[0]; result.append(d);
            }
            return result;
        })
        .def("take_trace", [](PipelineHost& h) {
            auto rt = dynamic_cast<TracePipelineRuntime*>(h.Runtime().get()); TORCH_CHECK(rt, "CPU trace only");
            py::list out;
            for (auto& t : rt->trace) {
                if (t.op <= 1) out.append(py::make_tuple(t.op, t.lane, t.event));
                else if (t.op == 2) out.append(py::make_tuple(t.op, t.lane, t.src, t.dst, Rows(t.copies.spans)));
                else out.append(py::make_tuple(t.op, t.lane, t.src, t.dst, Scalars(t.counts),
                                               t.hasStorage ? py::object(Scalars(t.storage)) : py::none()));
            }
            rt->trace.clear(); return out;
        })
        .def("trace_fail_after", [](PipelineHost& h, int transfers) {
            auto rt = dynamic_cast<TracePipelineRuntime*>(h.Runtime().get()); TORCH_CHECK(rt, "CPU trace only");
            rt->failAfter = transfers;
        });
    m.def("ep8_pipeline_create", [](const Planner& planner, int rank, U chunk, U maxChunks, U limit, bool cpuTrace, bool bypassCopies, U maxPayloads) {
        std::shared_ptr<PipelineRuntime> rt;
        if (cpuTrace) rt = std::make_shared<TracePipelineRuntime>();
        else rt = std::make_shared<NpuPipelineRuntime>(rank);
        return std::make_shared<PipelineHost>(planner, rank, chunk, maxChunks, limit, std::move(rt), bypassCopies, maxPayloads);
    }, py::arg("planner"), py::arg("rank"), py::arg("chunk_bytes") = 8 << 20,
       py::arg("max_chunks") = 128, py::arg("max_scratch_bytes") = 1 << 30, py::arg("cpu_trace") = false, py::arg("bypass_copies") = true, py::arg("max_payloads") = 4);
}
} // namespace deep_ep::ep8
