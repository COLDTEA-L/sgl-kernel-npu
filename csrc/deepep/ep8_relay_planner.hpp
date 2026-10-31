#pragma once

// Host-only EP8 planner. No HCCL, Torch, communicator, or device state.
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace deep_ep::ep8 {
using U = uint64_t;
using Matrix = std::array<U, 64>;
using Relays = std::array<int, 64>;
using Action = std::array<U, 5>;
using Sizes = std::array<std::array<U, 4>, 8>;
using Span = std::array<U, 3>;  // source row, destination row, row count
constexpr U Max = static_cast<U>(std::numeric_limits<int64_t>::max());

struct CopySpans {
    std::vector<Span> spans;
    CopySpans() = default;
    explicit CopySpans(std::vector<Span> values) : spans(std::move(values)) {}
    void Coalesce() {
        std::sort(spans.begin(), spans.end());
        size_t used = 0;
        for (const auto value : spans) {
            if (used && spans[used - 1][0] + spans[used - 1][2] == value[0] &&
                spans[used - 1][1] + spans[used - 1][2] == value[1]) {
                spans[used - 1][2] += value[2];
            } else {
                spans[used++] = value;
            }
        }
        spans.resize(used);
    }
};

struct Plan {
    Matrix counts{}, moved{}, loads{};
    Relays relays{};
    std::vector<Action> actions;
    U evaluations = 0;
    bool exhausted = false;
};

struct Chunk {
    Matrix first{}, second{};
    std::vector<std::array<U, 10>> pieces;
    Sizes sizes{};
    std::array<CopySpans, 3> localCopies;  // pack, forward, unpack
};

struct Layout {
    U chunkRows = 0;
    Matrix direct{};
    std::vector<std::array<U, 7>> directPieces;
    std::vector<Chunk> chunks;
    Sizes pools{};
};

inline U CheckCounts(const Matrix& counts, U rowBytes) {
    if (rowBytes == 0 || rowBytes > Max) throw std::invalid_argument("invalid row_bytes");
    U total = 0;
    for (U n : counts) {
        if (n > Max - total) throw std::invalid_argument("total rows overflow int64");
        total += n;
    }
    if (total > Max / rowBytes) throw std::invalid_argument("total bytes overflow int64");
    return total;
}

inline std::pair<Matrix, Matrix> Offsets(const Matrix& c) {
    Matrix src{}, dst{};
    std::array<U, 8> sends{}, recvs{};
    for (size_t f = 0; f < 64; ++f) {
        src[f] = sends[f / 8]; dst[f] = recvs[f % 8];
        sends[f / 8] += c[f]; recvs[f % 8] += c[f];
    }
    return {src, dst};
}

class Planner {
public:
    Planner(const std::vector<std::vector<U>>& allowed, U rowBytes, U minChunk,
            U minGain, U maxActions, const std::vector<U>& capacity)
        : rowBytes_(rowBytes), minGain_(minGain), maxActions_(maxActions),
          minChunkBytes_(minChunk), capacityBytes_(capacity) {
        if (!rowBytes || rowBytes > Max || !minChunk || minChunk > Max ||
            minGain > Max || maxActions > 56 || allowed.size() != 64 ||
            (!capacity.empty() && capacity.size() != 8))
            throw std::invalid_argument("invalid EP8 planner configuration");
        chunkRows_ = minChunk / rowBytes + (minChunk % rowBytes != 0);
        capacity_.fill(Max);
        for (size_t k = 0; k < capacity.size(); ++k) {
            if (capacity[k] > Max) throw std::invalid_argument("invalid relay capacity");
            capacity_[k] = capacity[k] / rowBytes;
        }
        for (size_t f = 0; f < 64; ++f) {
            for (U k : allowed[f]) {
                if (k >= 8 || k == f / 8 || k == f % 8 || f / 8 == f % 8 ||
                    (masks_[f] & (1u << k)))
                    throw std::invalid_argument("invalid or duplicate participant relay");
                masks_[f] |= 1u << k;
            }
        }
    }

    U RowBytes() const { return rowBytes_; }
    Planner WithDenseReady(U divisor) const {
        if (divisor < 2 || divisor > 1024) throw std::invalid_argument("dense split divisor must be 2..1024");
        Planner out=*this; out.denseDivisor_=divisor; return out;
    }
    U DenseDivisor() const { return denseDivisor_; }
    // Preserve byte thresholds rather than scaling already rounded row counts.
    Planner WithRowBytes(U bytes) const {
        if (!bytes || bytes > Max) throw std::invalid_argument("invalid EP8 row bytes");
        Planner out = *this;
        out.rowBytes_ = bytes;
        out.chunkRows_ = minChunkBytes_ / bytes + (minChunkBytes_ % bytes != 0);
        out.capacity_.fill(Max);
        for (size_t k = 0; k < capacityBytes_.size(); ++k) out.capacity_[k] = capacityBytes_[k] / bytes;
        return out;
    }

    Plan Make(const Matrix& counts) const {
        Plan p; MakeInto(counts, p); return p;
    }

    // Reuse host storage, but recompute every decision for the actual matrix.
    void MakeInto(const Matrix& counts, Plan& p) const {
        CheckCounts(counts, rowBytes_);
        p.counts = counts; p.relays.fill(-1); p.moved.fill(0);
        p.actions.clear(); p.evaluations = 0; p.exhausted = false;
        for (size_t f = 0; f < 64; ++f) p.loads[f] = f / 8 == f % 8 ? 0 : counts[f];
        if (denseDivisor_) { MakeDense(p); return; }
        std::array<U, 8> used{};
        p.actions.reserve(56);
        for (U iteration = 0; iteration < maxActions_; ++iteration) {
            bool found = false;
            U bestLoad = 0, bestGain = 0, bestX = 0;
            size_t bestF = 0, bestK = 0;
            // Ascending (flow, relay) order preserves the reference tie break.
            // A fixed-array scan replaces Python lists, sorting and generators.
            for (size_t f = 0; f < 64; ++f) {
                U a = p.loads[f];
                if (!counts[f] || !masks_[f] || p.relays[f] >= 0 || (found && a < bestLoad)) continue;
                for (size_t k = 0; k < 8; ++k) {
                    if (!(masks_[f] & (1u << k))) continue;
                    ++p.evaluations;
                    U h = std::max(p.loads[f / 8 * 8 + k], p.loads[k * 8 + f % 8]);
                    if (a <= h) continue;
                    U x = std::min({counts[f], (a - h) / 2, capacity_[k] - used[k]});
                    U gain = a - std::max(a - x, h + x);
                    if (x < chunkRows_ || !gain || gain * rowBytes_ < minGain_) continue;
                    if (!found || a > bestLoad || (a == bestLoad && gain > bestGain)) {
                        found = true; bestLoad = a; bestGain = gain; bestX = x; bestF = f; bestK = k;
                    }
                }
            }
            if (!found) { p.exhausted = true; break; }
            p.relays[bestF] = static_cast<int>(bestK); p.moved[bestF] = bestX;
            used[bestK] += bestX; p.loads[bestF] -= bestX;
            p.loads[bestF / 8 * 8 + bestK] += bestX;
            p.loads[bestK * 8 + bestF % 8] += bestX;
            p.actions.push_back({bestF / 8, bestF % 8, bestK, bestX, bestGain});
        }
    }

    Layout Build(const Matrix& counts, const Relays& relays, const Matrix& moved,
                 U chunkBytes, U maxChunks, int rank = -1) const {
        Layout out; BuildInto(counts, relays, moved, chunkBytes, maxChunks, rank, out, false); return out;
    }

    void BuildInto(const Matrix& counts, const Relays& relays, const Matrix& moved,
                   U chunkBytes, U maxChunks, int rank, Layout& out, bool compact) const {
        CheckCounts(counts, rowBytes_);
        if (!chunkBytes || chunkBytes > Max || !maxChunks || maxChunks > Max || rank < -1 || rank >= 8)
            throw std::invalid_argument("invalid chunk bounds");
        out.chunkRows = std::max<U>(1, chunkBytes / rowBytes_);
        out.directPieces.clear(); out.pools = {};
        U maxMoved = 0;
        for (size_t f = 0; f < 64; ++f) {
            int k = relays[f];
            if (moved[f] > counts[f] || (moved[f] && (k < 0 || k >= 8 || !(masks_[f] & (1u << k)))) ||
                (!moved[f] && k != -1)) throw std::invalid_argument("invalid relay assignment");
            out.direct[f] = counts[f] - moved[f]; maxMoved = std::max(maxMoved, moved[f]);
        }
        U numChunks = maxMoved / out.chunkRows + (maxMoved % out.chunkRows != 0);
        if (numChunks > maxChunks) throw std::invalid_argument("relay chunk count exceeds max_chunks");
        auto [src, dst] = Offsets(counts);
        if (!compact) {
            auto [ds, dr] = Offsets(out.direct);
            for (size_t f = 0; f < 64; ++f) if (out.direct[f])
                out.directPieces.push_back({f / 8, f % 8, out.direct[f], src[f], dst[f], ds[f], dr[f]});
        }
        out.chunks.resize(numChunks);
        for (U j = 0; j < numChunks; ++j) {
            auto& chunk = out.chunks[j];
            chunk.first.fill(0); chunk.second.fill(0); chunk.sizes = {}; chunk.pieces.clear();
            for (auto& copies : chunk.localCopies) copies.spans.clear();
            std::array<std::array<U, 10>, 56> pieces;
            size_t pieceCount = 0;
            for (size_t f = 0; f < 64; ++f) {
                if (moved[f] <= j * out.chunkRows) continue;
                U n = std::min(out.chunkRows, moved[f] - j * out.chunkRows);
                U s = f / 8, d = f % 8, k = static_cast<U>(relays[f]);
                U offset = out.direct[f] + j * out.chunkRows;
                pieces[pieceCount++] = {s, d, k, n, src[f] + offset, dst[f] + offset, 0, 0, 0, 0};
                chunk.first[s * 8 + k] += n; chunk.second[k * 8 + d] += n;
            }
            auto [fs, fr] = Offsets(chunk.first); auto [ss, sr] = Offsets(chunk.second);
            Matrix fu{}, su{};
            for (size_t index = 0; index < pieceCount; ++index) {
                auto& p = pieces[index];
                U a = p[0] * 8 + p[2], b = p[2] * 8 + p[1];
                p[6] = fs[a] + fu[a]; p[7] = fr[a] + fu[a];
                p[8] = ss[b] + su[b]; p[9] = sr[b] + su[b];
                fu[a] += p[3]; su[b] += p[3];
                if (rank >= 0) {
                    if (p[0] == static_cast<U>(rank)) chunk.localCopies[0].spans.push_back({p[4], p[6], p[3]});
                    if (p[2] == static_cast<U>(rank)) chunk.localCopies[1].spans.push_back({p[7], p[8], p[3]});
                    if (p[1] == static_cast<U>(rank)) chunk.localCopies[2].spans.push_back({p[9], p[5], p[3]});
                }
            }
            if (!compact) chunk.pieces.assign(pieces.begin(), pieces.begin() + pieceCount);
            if (rank >= 0) for (auto& copies : chunk.localCopies) copies.Coalesce();
            for (size_t f = 0; f < 64; ++f) {
                chunk.sizes[f / 8][0] += chunk.first[f];
                chunk.sizes[f % 8][1] += chunk.first[f];
                chunk.sizes[f / 8][2] += chunk.second[f];
                chunk.sizes[f % 8][3] += chunk.second[f];
            }
            for (size_t r = 0; r < 8; ++r) for (size_t i = 0; i < 4; ++i)
                out.pools[r][i] = std::max(out.pools[r][i], chunk.sizes[r][i]);
        }
    }
private:
    // One delayed relay action, triggered by excess over the source row mean.
    // Include self in the /8 mean; physical edge loads exclude self. No time
    // estimate or helper-card burden predicate. At most 56*6 candidates.
    void MakeDense(Plan& p) const {
        p.exhausted=true;
        if (!maxActions_) return;
        U peak=0, second=0;
        std::array<U,8> base{};
        for(size_t f=0;f<64;++f) {
            base[f/8]+=p.counts[f];
            const U n=p.loads[f];
            if(n>peak) {second=peak;peak=n;} else second=std::max(second,n);
        }
        for(auto& n:base) n/=8;
        U bestExcess=0,bestAfter=peak,bestX=0;size_t bestF=0,bestK=0;
        for(size_t f=0;f<64;++f) {
            if(f/8==f%8 || p.counts[f]<=base[f/8]) continue;
            const U excess=p.counts[f]-base[f/8];
            if(bestX && excess<bestExcess) continue;
            const U requested=excess/denseDivisor_;
            if(requested<chunkRows_) continue;
            for(size_t k=0;k<8;++k) {
                if(!(masks_[f]&(1u<<k))) continue;
                ++p.evaluations;
                const U x=std::min(requested,capacity_[k]);
                const size_t first=f/8*8+k, last=k*8+f%8;
                // Reject, do not force a split that grows the global peak.
                // Subtraction before addition also prevents integer overflow.
                if(x<chunkRows_ || x>peak-p.loads[first] || x>peak-p.loads[last]) continue;
                const U other=p.loads[f]==peak ? second : peak;
                const U after=std::max({other,p.loads[f]-x,p.loads[first]+x,p.loads[last]+x});
                if((peak-after)*rowBytes_<minGain_) continue;
                if(!bestX || excess>bestExcess || (excess==bestExcess && after<bestAfter)) {
                    bestExcess=excess;bestAfter=after;bestX=x;bestF=f;bestK=k;
                }
            }
        }
        if(!bestX) return;
        p.relays[bestF]=static_cast<int>(bestK);p.moved[bestF]=bestX;
        p.loads[bestF]-=bestX;p.loads[bestF/8*8+bestK]+=bestX;p.loads[bestK*8+bestF%8]+=bestX;
        p.actions.push_back({bestF/8,bestF%8,bestK,bestX,peak-bestAfter});p.exhausted=false;
    }
    U denseDivisor_=0;
    U rowBytes_, chunkRows_, minGain_, maxActions_, minChunkBytes_;
    std::vector<U> capacityBytes_;
    std::array<unsigned, 64> masks_{};
    std::array<U, 8> capacity_{};
};
} // namespace deep_ep::ep8
