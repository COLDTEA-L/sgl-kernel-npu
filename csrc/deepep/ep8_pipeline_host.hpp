#pragma once
#include "ep8_copy_spans.hpp"
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>

namespace deep_ep::ep8 {
// Shared execution driver: real NPU runtime and deferred CPU trace tests use
// exactly the same dependency order, layouts, view binding and workspace pool.
struct PipelineRuntime {
    virtual ~PipelineRuntime() = default;
    virtual void Check(const at::Tensor&) = 0;
    virtual void Sync() = 0;
    virtual void Lane(int) = 0;
    virtual void Events(size_t) = 0;
    virtual void Record(size_t, int) = 0;
    virtual void Wait(size_t, int) = 0;
    virtual void Retain(const at::Tensor&, int) = 0;
    virtual void Copy(const at::Tensor&, const at::Tensor&, const CopySpans&) = 0;
    virtual void Transfer(const at::Tensor&, const at::Tensor&, const Matrix&, const Matrix*, U) = 0;
};
struct PipelineCopyBinding {
    // -1 means stage through workspace. Nonnegative values are row offsets.
    int64_t pack = -1, forward = -1, unpack = -1;
};
struct PipelinePrepared {
    Plan plan;
    Layout layout;
    std::vector<ChunkViews> views;
    std::vector<PipelineCopyBinding> copyBindings;
    std::array<U, 3> bypassSpans{}, bypassBytes{};
    U sendRows = 0, recvRows = 0, generation = 0, rowBytes = 0;
    const void* owner = nullptr;
    bool tracked = false;
    bool UsesRelay() const { return !layout.chunks.empty(); }
};

class PipelinePayloadHost {
public:
    PipelinePayloadHost(const Planner& planner, int rank, U chunkBytes, U maxChunks, U maxScratch,
                 std::shared_ptr<PipelineRuntime> runtime, bool bypassCopies = true)
        : planner_(planner), rank_(rank), rowBytes_(planner.RowBytes()), chunkBytes_(chunkBytes),
          maxChunks_(maxChunks), rt_(std::move(runtime)), bypassCopies_(bypassCopies) {
        TORCH_CHECK(rank >= 0 && rank < 8 && chunkBytes && chunkBytes <= Max && maxChunks && maxChunks <= Max && maxScratch,
                    "invalid native pipeline configuration");
        limits_.fill(maxScratch);
    }
    ~PipelinePayloadHost() {
        try { Close(); } catch (const std::exception& e) {
            std::fprintf(stderr, "EP8_CPP_CLOSE_FAILED: %s\n", e.what());
        }
    }
    std::shared_ptr<PipelinePrepared> Prepare(const Matrix& counts, const at::Tensor& example) {
        auto lock = Lock(); Check(example);
        auto p = Build(counts, example);
        return p;
    }
    void Execute(const std::shared_ptr<PipelinePrepared>& p, const at::Tensor& send, const at::Tensor& recv) {
        auto lock = Lock(); Check(send); Validate(p, send, recv); Submit(*p, send, recv);
    }
    // One boundary for dynamic plan + layout + pooled views + actual submission.
    // Timings are host-only phase durations, not device completion or profiling.
    std::array<double, 3> Run(const Matrix& counts, const at::Tensor& send, const at::Tensor& recv) {
        auto lock = Lock(); Check(send);
        if (!dynamic_) dynamic_ = std::make_shared<PipelinePrepared>();
        auto p = Build(counts, send, dynamic_);
        Validate(p, send, recv);
        auto start = Clock::now(); Submit(*p, send, recv);
        return {lastPlanUs_, lastLayoutUs_, Micros(start)};
    }
    void Close() {
        auto lock = Lock();
        if (closed_) return;
        rt_->Sync();  // all lanes, also after a partially submitted failure
        Invalidate(); dynamic_.reset(); arena_.clear(); zeroView_ = at::Tensor(); zeroRecvView_ = at::Tensor();
        zero_ = at::Tensor(); zeroRecv_ = at::Tensor(); reservedBytes_ = 0;
        closed_ = true;
    }
    std::array<U, 7> Stats() const {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        TORCH_CHECK(lock.owns_lock(), "cannot inspect native pipeline during host submission");
        return {generation_, allocations_, events_, reservedBytes_, poisoned_ ? 1u : 0u,
                zero_.defined() ? static_cast<U>(zero_.nbytes() + zeroRecv_.nbytes()) : 0u, closed_ ? 1u : 0u};
    }
    void SetLimits(const std::array<U, 8>& limits) { limits_ = limits; }
    std::array<U, 8> ReservedPerRank() const {
        std::array<U, 8> result{};
        if (!closed_) for (int r = 0; r < 8; ++r) for (U n : capacities_[r]) result[r] += n * rowBytes_ * 2;
        return result;
    }
private:
    using Clock = std::chrono::steady_clock;
    static double Micros(Clock::time_point start) {
        return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    }
    std::unique_lock<std::mutex> Lock() {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        TORCH_CHECK(lock.owns_lock(), "native pipeline does not allow concurrent host submissions");
        return lock;
    }
    void Check(const at::Tensor& t) {
        TORCH_CHECK(!closed_ && !poisoned_, "native pipeline is closed or failed; recreate context");
        rt_->Check(t);
        TORCH_CHECK(t.dim() >= 1 && t.is_contiguous(), "invalid pipeline tensor shape/strides");
        TORCH_CHECK(IsCcuPayload(t.scalar_type()), "unsupported pipeline dtype");
        U bytes = t.element_size();
        for (int i = 1; i < t.dim(); ++i) {
            TORCH_CHECK(t.size(i) > 0 && static_cast<U>(t.size(i)) <= Max / bytes, "invalid row shape");
            bytes *= t.size(i);
        }
        TORCH_CHECK(bytes == rowBytes_, "pipeline row_bytes mismatch");
        if (zero_.defined()) {
            TORCH_CHECK(t.device() == zero_.device() && t.scalar_type() == zero_.scalar_type() &&
                        t.sizes().slice(1) == zero_.sizes().slice(1), "pipeline payload signature changed");
        } else {
            auto sizes = t.sizes().vec(); sizes[0] = 1;
            zero_ = at::empty(sizes, t.options());
            // Vendor validates disjoint backing CAPACITIES even for zero rows.
            // Keep separate send/receive sentinels, never one shared empty view.
            zeroRecv_ = at::empty(sizes, t.options());
            zeroView_ = zero_.narrow(0, 0, 0); zeroRecvView_ = zeroRecv_.narrow(0, 0, 0);
            rt_->Retain(zero_, 0); rt_->Retain(zeroRecv_, 0);
        }
    }
    std::shared_ptr<PipelinePrepared> Build(const Matrix& counts, const at::Tensor& example,
                                            std::shared_ptr<PipelinePrepared> reuse = {}) {
        auto begin = Clock::now();
        auto p = reuse ? reuse : std::make_shared<PipelinePrepared>();
        bool tracked = p->tracked;
        planner_.MakeInto(counts, p->plan);
        lastPlanUs_ = Micros(begin); begin = Clock::now();
        p->sendRows = p->recvRows = 0;
        for (int k = 0; k < 8; ++k) { p->sendRows += counts[rank_ * 8 + k]; p->recvRows += counts[k * 8 + rank_]; }
        TORCH_CHECK(p->sendRows == static_cast<U>(example.size(0)), "send rows differ from current matrix");
        planner_.BuildInto(counts, p->plan.relays, p->plan.moved, chunkBytes_, maxChunks_, rank_, p->layout, true);
        DescribeCopies(*p);
        if (p->UsesRelay()) {
            Reserve(p->layout, example); Bind(*p);
            if (!tracked) {
                live_.erase(std::remove_if(live_.begin(), live_.end(), [](const auto& w) { return w.expired(); }), live_.end());
                live_.push_back(p); p->tracked = true;
            }
        }
        p->rowBytes = rowBytes_;
        p->owner = this;
        lastLayoutUs_ = Micros(begin);
        return p;
    }
    void Invalidate() {
        for (auto& weak : live_) if (auto p = weak.lock()) { p->views.clear(); p->generation = 0; }
    }
    void Reserve(const Layout& layout, const at::Tensor& example) {
        Sizes exact = capacities_, rounded{};
        bool growth = arena_.empty();
        for (int r = 0; r < 8; ++r) for (int i = 0; i < 4; ++i) {
            exact[r][i] = std::max({exact[r][i], layout.pools[r][i], U{1}});
            growth = growth || exact[r][i] != capacities_[r][i];
            U n = 1; while (n < exact[r][i] && n <= Max / 2) n *= 2;
            rounded[r][i] = n < exact[r][i] ? exact[r][i] : n;
        }
        auto fits = [&](const Sizes& caps) {
            for (int r = 0; r < 8; ++r) {
                U total = 0;
                for (U n : caps[r]) { if (n > Max - total) return false; total += n; }
                if (total > limits_[r] / rowBytes_ / 2) return false;
            }
            return true;
        };
        TORCH_CHECK(fits(exact), "native pipeline high-water workspace exceeds max_scratch_bytes on a rank");
        if (growth) {
            auto chosen = fits(rounded) ? rounded : exact;
            rt_->Sync(); Invalidate(); arena_.clear();
            std::vector<at::Tensor> fresh;
            for (int slot = 0; slot < 2; ++slot) for (U n : chosen[rank_]) {
                auto shape = example.sizes().vec(); shape[0] = n;
                auto t = at::empty(shape, example.options());
                // Register each persistent allocation once, not every execution.
                rt_->Retain(t, 1); rt_->Retain(t, 2);
                fresh.push_back(std::move(t));
            }
            arena_ = std::move(fresh); capacities_ = chosen;
            ++generation_; allocations_ += 8; reservedBytes_ = 0;
            for (auto& t : arena_) reservedBytes_ += t.nbytes();
        }
        size_t needed = 1 + 2 * layout.chunks.size();
        rt_->Events(needed); events_ = std::max<U>(events_, needed);
    }
    void DescribeCopies(PipelinePrepared& p) {
        p.copyBindings.resize(p.layout.chunks.size());
        p.bypassSpans = {}; p.bypassBytes = {};
        for (size_t j = 0; j < p.layout.chunks.size(); ++j) {
            const auto& c = p.layout.chunks[j]; auto& b = p.copyBindings[j]; b = {};
            if (!bypassCopies_) continue;
            const auto& pack = c.localCopies[0].spans;
            const auto& forward = c.localCopies[1].spans;
            const auto& unpack = c.localCopies[2].spans;
            // Only full, contiguous, correctly ordered intervals can alias.
            // Coalescing has already merged exactly adjacent source/dest spans.
            if (pack.size() == 1 && pack[0][1] == 0 && pack[0][2] == c.sizes[rank_][0]) b.pack = pack[0][0];
            if (forward.size() == 1 && forward[0][0] == 0 && forward[0][1] == 0 &&
                forward[0][2] == c.sizes[rank_][1] && forward[0][2] == c.sizes[rank_][2]) b.forward = 0;
            if (unpack.size() == 1 && unpack[0][0] == 0 && unpack[0][2] == c.sizes[rank_][3]) b.unpack = unpack[0][1];
            const std::array<int64_t, 3> offsets{b.pack, b.forward, b.unpack};
            for (int i = 0; i < 3; ++i) if (offsets[i] >= 0) {
                ++p.bypassSpans[i]; p.bypassBytes[i] += c.localCopies[i].spans[0][2] * rowBytes_;
            }
        }
    }
    void Bind(PipelinePrepared& p) {
        p.views.resize(p.layout.chunks.size());
        for (size_t j = 0; j < p.layout.chunks.size(); ++j) {
            const auto& sizes = p.layout.chunks[j].sizes[rank_];
            const auto& b = p.copyBindings[j]; auto& views = p.views[j];
            for (int i = 0; i < 4; ++i) {
                if ((i == 0 && b.pack >= 0) || (i == 2 && b.forward >= 0) || (i == 3 && b.unpack >= 0)) {
                    views[i] = at::Tensor(); continue; // bound to current payload at submission
                }
                if (!sizes[i]) { views[i] = i % 2 ? zeroRecvView_ : zeroView_; continue; }
                // Reuse unchanged workspace views on a dynamic call. Never
                // mutate a TensorImpl that queued work may still reference.
                if (p.generation == generation_ && views[i].defined() &&
                    views[i].size(0) == static_cast<int64_t>(sizes[i])) continue;
                views[i] = arena_[(j % 2) * 4 + i].narrow(0, 0, sizes[i]);
            }
        }
        p.generation = generation_;
    }
    void Validate(const std::shared_ptr<PipelinePrepared>& p, const at::Tensor& send, const at::Tensor& recv) {
        TORCH_CHECK(p && p->owner == this, "prepared pipeline belongs to another context");
        TORCH_CHECK(send.size(0) == static_cast<int64_t>(p->sendRows) && recv.dim() == send.dim() &&
                    recv.size(0) == static_cast<int64_t>(p->recvRows) && recv.is_contiguous() &&
                    send.sizes().slice(1) == recv.sizes().slice(1) && send.device() == recv.device() &&
                    send.scalar_type() == recv.scalar_type(), "pipeline payload differs from prepared matrix");
        if (send.numel() && recv.numel()) {
            auto a = reinterpret_cast<uintptr_t>(send.data_ptr()), b = reinterpret_cast<uintptr_t>(recv.data_ptr());
            TORCH_CHECK(a <= b ? b - a >= send.nbytes() : a - b >= recv.nbytes(), "pipeline buffers overlap");
        }
        if (p->UsesRelay() && p->generation != generation_) Bind(*p);
    }
    void Submit(PipelinePrepared& p, const at::Tensor& src, const at::Tensor& dst) {
        // Keep real backing for zero-row calls; no per-iteration sentinel allocation.
        auto send = src.storage().nbytes() ? src : zeroView_;
        auto recv = dst.storage().nbytes() ? dst : zeroRecvView_;
        struct Restore { PipelineRuntime& rt; ~Restore() { try { rt.Lane(0); } catch (...) {} } } restore{*rt_};
        int batch = -1;
        try {
            rt_->Lane(0);
            if (!p.UsesRelay()) { rt_->Transfer(send, recv, p.plan.counts, nullptr, rowBytes_); return; }
            rt_->Record(0, 0); rt_->Wait(0, 1); rt_->Wait(0, 2);
            rt_->Retain(send, 1); rt_->Retain(recv, 2);
            rt_->Transfer(send, recv, p.layout.direct, &p.plan.counts, rowBytes_);
            for (size_t j = 0; j < p.layout.chunks.size(); ++j) {
                batch = static_cast<int>(j);
                const auto& chunk = p.layout.chunks[j]; const auto& v = p.views[j];
                size_t ready = 1 + 2*j, done = ready + 1;
                rt_->Lane(1);
                if (j >= 2) rt_->Wait(done - 4, 1);
                const auto& b = p.copyBindings[j];
                auto firstSend = b.pack >= 0 ? send.narrow(0, b.pack, chunk.sizes[rank_][0]) : v[0];
                if (b.pack < 0) rt_->Copy(send, firstSend, chunk.localCopies[0]);
                rt_->Transfer(firstSend, v[1], chunk.first, nullptr, rowBytes_);
                rt_->Record(ready, 1);
                rt_->Lane(2); rt_->Wait(ready, 2);
                auto secondSend = b.forward >= 0 ? v[1] : v[2];
                auto secondRecv = b.unpack >= 0 ? recv.narrow(0, b.unpack, chunk.sizes[rank_][3]) : v[3];
                if (b.forward < 0) rt_->Copy(v[1], secondSend, chunk.localCopies[1]);
                rt_->Transfer(secondSend, secondRecv, chunk.second, nullptr, rowBytes_);
                if (b.unpack < 0) rt_->Copy(secondRecv, recv, chunk.localCopies[2]);
                rt_->Record(done, 2);
            }
            rt_->Wait(2 * p.layout.chunks.size(), 0);
        } catch (...) {
            poisoned_ = true;
            std::fprintf(stderr, "EP8_CPP_SUBMIT_FAILED rank=%d batch=%d generation=%lu; context poisoned, no retry\n",
                         rank_, batch, static_cast<unsigned long>(generation_));
            throw;
        }
    }
    Planner planner_;
    int rank_;
    U rowBytes_, chunkBytes_, maxChunks_;
    std::array<U, 8> limits_{};
    std::shared_ptr<PipelineRuntime> rt_;
    mutable std::mutex mutex_;
    at::Tensor zero_, zeroView_, zeroRecv_, zeroRecvView_;
    std::shared_ptr<PipelinePrepared> dynamic_;
    std::vector<at::Tensor> arena_;
    Sizes capacities_{};
    std::vector<std::weak_ptr<PipelinePrepared>> live_;
    U generation_ = 0, allocations_ = 0, events_ = 0, reservedBytes_ = 0;
    double lastPlanUs_ = 0, lastLayoutUs_ = 0;
    bool closed_ = false, poisoned_ = false, bypassCopies_ = true;
};
// One communication context, multiple typed payload workspaces. The lane-0 join
// in Submit orders consecutive directions, including shared event and CCU handle
// reuse. Profile selection never creates a communicator/channel/kernel or syncs.
class PipelineHost {
public:
    struct PayloadInfo {
        c10::Device device;
        at::ScalarType dtype;
        std::vector<int64_t> shape;
        U rowBytes;
        std::array<U, 7> stats;
    };
    PipelineHost(const Planner& planner, int rank, U chunkBytes, U maxChunks, U maxScratch,
                 std::shared_ptr<PipelineRuntime> runtime, bool bypassCopies = true, U maxPayloads = 4)
        : planner_(planner), rank_(rank), chunkBytes_(chunkBytes), maxChunks_(maxChunks),
          limit_(maxScratch), rt_(std::move(runtime)), bypassCopies_(bypassCopies), maxPayloads_(maxPayloads) {
        TORCH_CHECK(rank >= 0 && rank < 8 && chunkBytes && chunkBytes <= Max &&
                    maxChunks && maxChunks <= Max && maxScratch && maxScratch <= Max && maxPayloads && rt_,
                    "invalid native pipeline configuration");
    }
    ~PipelineHost() {
        try { Close(); } catch (const std::exception& e) {
            std::fprintf(stderr, "EP8_CPP_CLOSE_FAILED: %s\n", e.what());
        }
    }
    std::shared_ptr<PipelinePrepared> Prepare(const Matrix& counts, const at::Tensor& example) {
        auto lock = Lock(); Check(); auto& p = Select(example); Budget(p);
        return p.host->Prepare(counts, example);
    }
    void Execute(const std::shared_ptr<PipelinePrepared>& prepared, const at::Tensor& send, const at::Tensor& recv) {
        auto lock = Lock(); Check();
        TORCH_CHECK(prepared, "null prepared pipeline");
        for (auto& p : payloads_) if (prepared->owner == p.host.get()) {
            Budget(p); p.host->Execute(prepared, send, recv); return;
        }
        TORCH_CHECK(false, "prepared pipeline belongs to another context");
    }
    std::array<double, 3> Run(const Matrix& counts, const at::Tensor& send, const at::Tensor& recv) {
        auto lock = Lock(); Check(); auto& p = Select(send); Budget(p);
        return p.host->Run(counts, send, recv);
    }
    void Close() {
        auto lock = Lock();
        if (closed_) return;
        for (auto& p : payloads_) p.host->Close();
        closed_ = true;
    }
    std::array<U, 7> Stats() const {
        auto lock = Lock(); std::array<U, 7> result{};
        for (const auto& p : payloads_) {
            auto s = p.host->Stats();
            for (int i : {0, 1, 3, 5}) result[i] += s[i];
            // Events belong to the shared runtime, not to individual profiles.
            result[2] = std::max(result[2], s[2]); result[4] |= s[4];
        }
        result[6] = closed_; return result;
    }
    std::vector<PayloadInfo> PayloadStats() const {
        auto lock = Lock(); std::vector<PayloadInfo> result;
        for (const auto& p : payloads_) result.push_back({p.device, p.dtype, p.shape, p.rowBytes, p.host->Stats()});
        return result;
    }
    const std::shared_ptr<PipelineRuntime>& Runtime() const { return rt_; }
private:
    struct Payload {
        c10::Device device;
        at::ScalarType dtype;
        std::vector<int64_t> shape;
        U rowBytes;
        std::unique_ptr<PipelinePayloadHost> host;
    };
    std::unique_lock<std::mutex> Lock() const {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        TORCH_CHECK(lock.owns_lock(), "native pipeline does not allow concurrent host submissions");
        return lock;
    }
    void Check() const {
        TORCH_CHECK(!closed_, "native pipeline is closed; recreate context");
        for (const auto& p : payloads_) TORCH_CHECK(!p.host->Stats()[4], "native pipeline failed; recreate context");
    }
    Payload& Select(const at::Tensor& t) {
        rt_->Check(t);
        TORCH_CHECK(t.dim() >= 1 && t.is_contiguous() && IsCcuPayload(t.scalar_type()), "invalid pipeline payload");
        for (auto& p : payloads_) if (p.device == t.device() && p.dtype == t.scalar_type() &&
                                       t.sizes().slice(1) == at::IntArrayRef(p.shape)) return p;
        U bytes = t.element_size();
        for (int i = 1; i < t.dim(); ++i) {
            TORCH_CHECK(t.size(i) > 0 && static_cast<U>(t.size(i)) <= Max / bytes, "invalid row shape");
            bytes *= t.size(i);
        }
        TORCH_CHECK(payloads_.size() < maxPayloads_, "pipeline max_payloads exceeded; reuse existing payload signatures or increase max_payloads");
        payloads_.push_back({t.device(), t.scalar_type(), t.sizes().slice(1).vec(), bytes,
            std::make_unique<PipelinePayloadHost>(planner_.WithRowBytes(bytes), rank_, chunkBytes_, maxChunks_,
                                                  limit_, rt_, bypassCopies_)});
        return payloads_.back();
    }
    void Budget(Payload& selected) {
        std::array<U, 8> remaining; remaining.fill(limit_);
        for (const auto& p : payloads_) if (&p != &selected) {
            auto reserved = p.host->ReservedPerRank();
            for (int r = 0; r < 8; ++r) {
                TORCH_CHECK(reserved[r] <= remaining[r], "pipeline aggregate workspace budget exceeded");
                remaining[r] -= reserved[r];
            }
        }
        selected.host->SetLimits(remaining);
    }
    Planner planner_;
    int rank_;
    U chunkBytes_, maxChunks_, limit_;
    std::shared_ptr<PipelineRuntime> rt_;
    bool bypassCopies_, closed_ = false;
    U maxPayloads_;
    mutable std::mutex mutex_;
    std::vector<Payload> payloads_;
};
} // namespace deep_ep::ep8
