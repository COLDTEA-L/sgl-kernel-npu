#pragma once
#include "ep8_relay_planner.hpp"
#include <cmath>
#include <string>

namespace deep_ep::ep8 {
// A conservative admission policy, not an optimal routing solver or a latency
// guarantee. One hottest flow, six relays, four split candidates each. Rejected
// calls return a compact decision and never construct a layout/workspace.
struct FastDecision {
    Matrix counts{};
    int flow = -1, relay = -1;
    U rows = 0, chunkBytes = 0, evaluations = 0;
    double directUs = 0, relayUs = 0, savingUs = 0;
    std::string reason = "no_remote_traffic";
    bool UsesRelay() const { return relay >= 0; }
};

class FastSelector {
public:
    FastSelector(const std::vector<std::vector<U>>& allowed, U rowBytes,
                 const std::vector<U>& capacity, U chunkBytes, double linkGBps,
                 double extraUs, double minSavingUs)
        : rowBytes_(rowBytes), baseChunk_(chunkBytes), bytesPerUs_(linkGBps * 1000.),
          extraUs_(extraUs), minSavingUs_(minSavingUs) {
        // Reuse topology, integer and capacity checks, once at construction.
        Planner check(allowed, rowBytes, 1, 0, 1, capacity);
        if (!chunkBytes || chunkBytes > Max || !std::isfinite(bytesPerUs_) || bytesPerUs_ <= 0 ||
            !std::isfinite(extraUs) || extraUs < 0 || !std::isfinite(minSavingUs) || minSavingUs < 0)
            throw std::invalid_argument("invalid fast selector cost configuration");
        capacity_.fill(Max / rowBytes);
        for (size_t k = 0; k < capacity.size(); ++k) capacity_[k] = capacity[k] / rowBytes;
        for (size_t f = 0; f < 64; ++f) for (U k : allowed[f]) masks_[f] |= 1u << k;
    }

    FastDecision Select(const Matrix& counts) const {
        CheckCounts(counts, rowBytes_);
        FastDecision out; out.counts = counts;
        U peak = 0, other = 0;
        for (size_t f = 0; f < 64; ++f) {
            if (f / 8 == f % 8) continue;
            U n = counts[f];
            if (n > peak) { other = peak; peak = n; out.flow = static_cast<int>(f); }
            else other = std::max(other, n);
        }
        out.directUs = static_cast<double>(peak) * rowBytes_ / bytesPerUs_;
        if (!peak) return out;
        // Gap to the second peak limits global benefit. Even with empty relay
        // legs, a two-batch pipeline's optimistic optimum is 0.6*peak, so its
        // saving cannot exceed 0.4*peak. Reject before enumerating any relay.
        const long double optimisticSaving = std::min(static_cast<long double>(peak - other),
                                                       static_cast<long double>(peak) * 2 / 5);
        if (optimisticSaving * rowBytes_ / bytesPerUs_ <= extraUs_ + minSavingUs_) {
            out.reason = "global_gap_below_cost"; return out;
        }
        const size_t f = out.flow, s = f / 8, d = f % 8;
        out.reason = "no_admissible_relay";
        const U baseRows = std::max<U>(1, baseChunk_ / rowBytes_);
        double bestTime = out.directUs - minSavingUs_;
        for (size_t k = 0; k < 8; ++k) {
            if (!(masks_[f] & (1u << k))) continue;
            const U h = std::max(counts[s * 8 + k], counts[k * 8 + d]);
            if (h >= peak) continue;
            const U gap = peak - h;
            // With b equal chunks the relay path proxy is h+x+ceil(x/b).
            // Balancing it against peak-x gives x=(peak-h)*b/(2*b+1).
            // Also test the base-chunk regime boundaries. All arithmetic is
            // bounded before multiplying; no floating point row selection.
            const std::array<U, 4> candidates{gap / 3, (gap / 5) * 2 + (gap % 5) * 2 / 5,
                                             baseRows, baseRows > Max / 2 ? Max : 2 * baseRows};
            for (U x : candidates) {
                ++out.evaluations;
                x = std::min({x, peak, capacity_[k]});
                if (!x) continue;
                // At most two batches; grow chunk size for a larger admitted
                // hotspot rather than paying per-batch submission indefinitely.
                const U chunk = std::max(baseRows, x / 2 + x % 2);
                const long double relayRows = static_cast<long double>(h) + x + std::min(x, chunk);
                const long double tail = std::max({static_cast<long double>(other),
                                                   static_cast<long double>(peak - x), relayRows});
                const double time = static_cast<double>(tail * rowBytes_ / bytesPerUs_) + extraUs_;
                if (time < bestTime) {
                    bestTime = time; out.relay = static_cast<int>(k); out.rows = x;
                    // chunk may exceed a tiny payload; cap before byte conversion.
                    out.chunkBytes = std::min(chunk, x) * rowBytes_;
                }
            }
        }
        if (out.UsesRelay()) {
            out.relayUs = bestTime; out.savingUs = out.directUs - bestTime;
            out.reason = "single_hotspot_admitted";
        }
        return out;
    }

    Plan Materialize(const FastDecision& decision) const {
        // Re-evaluate a compact decision so it cannot be reused with a selector
        // having different topology/cost/capacity. Not used on the native path.
        auto checked = Select(decision.counts);
        if (checked.flow != decision.flow || checked.relay != decision.relay || checked.rows != decision.rows ||
            checked.chunkBytes != decision.chunkBytes)
            throw std::invalid_argument("decision belongs to a different fast selector");
        Plan p; p.counts = decision.counts; p.relays.fill(-1); p.evaluations = decision.evaluations;
        for (size_t f = 0; f < 64; ++f) p.loads[f] = f / 8 == f % 8 ? 0 : p.counts[f];
        if (decision.UsesRelay()) {
            size_t f = decision.flow, s = f / 8, d = f % 8, k = decision.relay;
            U x = decision.rows;
            const U before = *std::max_element(p.loads.begin(), p.loads.end());
            p.relays[f] = k; p.moved[f] = x; p.loads[f] -= x;
            p.loads[s * 8 + k] += x; p.loads[k * 8 + d] += x;
            const U after = *std::max_element(p.loads.begin(), p.loads.end());
            p.actions.push_back({s, d, k, x, before - after});
        }
        return p;
    }
private:
    U rowBytes_, baseChunk_;
    double bytesPerUs_, extraUs_, minSavingUs_;
    std::array<unsigned, 64> masks_{};
    std::array<U, 8> capacity_{};
};
} // namespace deep_ep::ep8
