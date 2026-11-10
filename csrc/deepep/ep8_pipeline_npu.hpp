#pragma once
#include "ep8_pipeline_host.hpp"
#include "deep_ep.hpp"
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

namespace deep_ep::ep8 {
void SubmitPipelineTransfer(const at::Tensor&, const at::Tensor&, int64_t, int64_t,
                           const uint64_t*, uint64_t, HcclComm, const uint64_t*);
class NpuPipelineRuntime : public PipelineRuntime {
public:
    explicit NpuPipelineRuntime(int rank) : rank_(rank) {}
    void Attach(int lane, Buffer& buffer, int64_t handle) {
        TORCH_CHECK(lane >= 0 && lane < 3 && !streams_[lane] && buffer.rank == rank_ && handle > 0,
                    "invalid native pipeline lane attachment");
        auto stream = c10_npu::getCurrentNPUStream();
        auto comm = buffer.ep8_pipeline_comm();
        for (int i = 0; i < 3; ++i) if (streams_[i]) {
            TORCH_CHECK(*streams_[i] != stream && streams_[i]->device() == stream.device() && comms_[i] != comm,
                        "pipeline requires distinct streams/communicators on one NPU");
        }
        streams_[lane] = stream; comms_[lane] = comm; handles_[lane] = handle;
    }
    void Check(const at::Tensor& t) override {
        TORCH_CHECK(streams_[0] && streams_[1] && streams_[2], "attach all three pipeline lanes first");
        TORCH_CHECK(t.device() == streams_[0]->device() && c10_npu::getCurrentNPUStream() == *streams_[0],
                    "native pipeline must run on its original caller stream/device");
    }
    void Sync() override { for (auto& s : streams_) if (s) s->synchronize(); }
    void Lane(int lane) override {
        TORCH_CHECK(streams_[lane], "missing pipeline stream");
        c10_npu::setCurrentNPUStream(*streams_[lane]); active_ = lane;
    }
    void Events(size_t count) override { while (events_.size() < count) events_.push_back(std::make_unique<c10_npu::NPUEvent>()); }
    void Record(size_t event, int lane) override { events_.at(event)->record(*streams_[lane]); }
    void Wait(size_t event, int lane) override { events_.at(event)->block(*streams_[lane]); }
    void Retain(const at::Tensor& t, int lane) override {
        if (t.storage().nbytes()) c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), *streams_[lane]);
    }
    void Copy(const at::Tensor& src, const at::Tensor& dst, const CopySpans& spans) override {
        if (!spans.spans.empty()) CopySpansOut(src, dst, spans);
    }
    void Transfer(const at::Tensor& src, const at::Tensor& dst, const Matrix& counts, const Matrix* storage, U rb) override {
        // SubmitPipelineTransfer uses stream() (with queue drain), never
        // stream(false); queued Torch copies must precede raw CCU submissions.
        SubmitPipelineTransfer(src, dst, handles_[active_], rank_, counts.data(), rb,
                               comms_[active_], storage ? storage->data() : nullptr);
    }
private:
    int rank_, active_ = 0;
    std::array<std::optional<c10_npu::NPUStream>, 3> streams_;
    std::array<HcclComm, 3> comms_{};
    std::array<int64_t, 3> handles_{};
    std::vector<std::unique_ptr<c10_npu::NPUEvent>> events_;
};
} // namespace deep_ep::ep8
