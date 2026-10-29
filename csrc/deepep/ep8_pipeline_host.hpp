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
struct PipelinePrepared {
    Plan plan;
    Layout layout;
    std::vector<ChunkViews> views;
    U sendRows = 0, recvRows = 0, generation = 0;
    const void* owner = nullptr;
    bool UsesRelay() const { return !layout.chunks.empty(); }
};

class PipelineHost {
public:
    PipelineHost(const Planner& planner, int rank, U chunkBytes, U maxChunks, U maxScratch,
                 std::shared_ptr<PipelineRuntime> runtime)
        : planner_(planner), rank_(rank), rowBytes_(planner.RowBytes()), chunkBytes_(chunkBytes),
          maxChunks_(maxChunks), limit_(maxScratch), rt_(std::move(runtime)) {
        TORCH_CHECK(rank >= 0 && rank < 8 && chunkBytes && chunkBytes <= Max && maxChunks && maxChunks <= Max && maxScratch,
                    "invalid native pipeline configuration");
    }
    ~PipelineHost() {
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
        auto p = Build(counts, send);
        Validate(p, send, recv);
        auto start = Clock::now(); Submit(*p, send, recv);
        return {lastPlanUs_, lastLayoutUs_, Micros(start)};
    }
    void Close() {
        auto lock = Lock();
        if (closed_) return;
        rt_->Sync();  // all lanes, also after a partially submitted failure
        Invalidate(); arena_.clear(); zero_ = at::Tensor(); reservedBytes_ = 0;
        closed_ = true;
    }
    std::array<U, 7> Stats() const {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        TORCH_CHECK(lock.owns_lock(), "cannot inspect native pipeline during host submission");
        return {generation_, allocations_, events_, reservedBytes_, poisoned_ ? 1u : 0u,
                zero_.defined() ? static_cast<U>(zero_.nbytes()) : 0u, closed_ ? 1u : 0u};
    }
    const std::shared_ptr<PipelineRuntime>& Runtime() const { return rt_; }
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
        TORCH_CHECK(t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat ||
                    t.scalar_type() == at::kChar || t.scalar_type() == at::kInt, "unsupported pipeline dtype");
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
            rt_->Retain(zero_, 0);
        }
    }
    std::shared_ptr<PipelinePrepared> Build(const Matrix& counts, const at::Tensor& example) {
        auto begin = Clock::now();
        auto p = std::make_shared<PipelinePrepared>();
        p->plan = planner_.Make(counts);
        lastPlanUs_ = Micros(begin); begin = Clock::now();
        p->owner = this;
        for (int k = 0; k < 8; ++k) { p->sendRows += counts[rank_ * 8 + k]; p->recvRows += counts[k * 8 + rank_]; }
        TORCH_CHECK(p->sendRows == static_cast<U>(example.size(0)), "send rows differ from current matrix");
        p->layout = planner_.Build(counts, p->plan.relays, p->plan.moved, chunkBytes_, maxChunks_, rank_);
        if (p->UsesRelay()) {
            Reserve(p->layout, example); Bind(*p);
            live_.erase(std::remove_if(live_.begin(), live_.end(), [](const auto& w) { return w.expired(); }), live_.end());
            live_.push_back(p);
        }
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
            for (auto& row : caps) {
                U total = 0;
                for (U n : row) { if (n > Max - total) return false; total += n; }
                if (total > limit_ / rowBytes_ / 2) return false;
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
    void Bind(PipelinePrepared& p) {
        std::vector<std::array<U, 4>> sizes;
        for (const auto& c : p.layout.chunks) sizes.push_back(c.sizes[rank_]);
        p.views = BindWorkspaceViews(arena_, p.layout.pools[rank_], sizes).second;
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
        auto send = src.storage().nbytes() ? src : zero_.narrow(0, 0, 0);
        auto recv = dst.storage().nbytes() ? dst : zero_.narrow(0, 0, 0);
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
                rt_->Copy(send, v[0], chunk.localCopies[0]);
                rt_->Transfer(v[0], v[1], chunk.first, nullptr, rowBytes_);
                rt_->Record(ready, 1);
                rt_->Lane(2); rt_->Wait(ready, 2);
                rt_->Copy(v[1], v[2], chunk.localCopies[1]);
                rt_->Transfer(v[2], v[3], chunk.second, nullptr, rowBytes_);
                rt_->Copy(v[3], recv, chunk.localCopies[2]);
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
    U rowBytes_, chunkBytes_, maxChunks_, limit_;
    std::shared_ptr<PipelineRuntime> rt_;
    mutable std::mutex mutex_;
    at::Tensor zero_;
    std::vector<at::Tensor> arena_;
    Sizes capacities_{};
    std::vector<std::weak_ptr<PipelinePrepared>> live_;
    U generation_ = 0, allocations_ = 0, events_ = 0, reservedBytes_ = 0;
    double lastPlanUs_ = 0, lastLayoutUs_ = 0;
    bool closed_ = false, poisoned_ = false;
};
} // namespace deep_ep::ep8
