/**
 * compact_test.cpp — Regression test for the compact() rebuild-path
 * corruption bug: if the >=50%-deleted rebuild in compact() ever finds
 * itself unable to reinsert a survivor into the fresh graph, *this must be
 * left exactly as it was before compact() was called (still fragmented,
 * still correct), not half-cleared with a stale count_/num_deleted_.
 *
 * The failure is forced deterministically by shrinking max_elements_ on the
 * live object immediately before calling compact(): the rebuild copies
 * max_elements_ into the fresh graph's config, so a capacity too small for
 * the survivor set makes rebuilt.insert() return UINT32_MAX partway through
 * — the same failure compact() must already tolerate from a degenerate
 * input vector, but reachable here without depending on quantization or
 * normalization edge cases.
 *
 * Standalone (no Node/N-API): compiled and run directly by `npm test` in
 * this package, alongside the N-API-bound checks in test.js.
 */

#include <cassert>
#include <cstdio>
#include <vector>

// Test-only white-box access: this test needs to read count_/deleted_/etc.
// directly (to prove they're untouched after a failed compact()) and to
// mutate max_elements_ (to force that failure deterministically), neither
// of which the public API exposes. Confined to this translation unit only.
#define private public
#include "../src/float_hnsw.hpp"
#include "../src/uint8_float_hnsw.hpp"
#undef private

using pikelet::wasm::DistanceMetric;
using pikelet::wasm::FloatHNSW;
using pikelet::wasm::FloatHNSWConfig;
using pikelet::wasm::Uint8FloatHNSW;
using pikelet::wasm::Uint8FloatHNSWConfig;

namespace {

constexpr size_t kDims = 4;
constexpr size_t kCount = 10;

std::vector<float> makeVector(size_t seed) {
    std::vector<float> v(kDims);
    for (size_t d = 0; d < kDims; d++) {
        v[d] = static_cast<float>((seed * 7 + d * 3) % 11) - 5.0f;
    }
    return v;
}

void testFloatCompactFailureLeavesObjectUntouched() {
    FloatHNSWConfig config;
    config.M = 4;
    config.ef_construction = 32;
    config.ef_search = 32;
    config.max_elements = kCount;
    config.metric = DistanceMetric::L2;

    FloatHNSW idx(kDims, config);
    std::vector<std::vector<float>> vecs;
    for (size_t i = 0; i < kCount; i++) {
        vecs.push_back(makeVector(i));
        const uint32_t id = idx.insert(vecs.back().data());
        assert(id == i);
    }

    // Delete >= half so compact() takes the rebuild path.
    for (size_t i = 0; i < 6; i++) idx.mark_delete(static_cast<uint32_t>(i));
    assert(idx.num_deleted_ == 6);
    const size_t countBefore = idx.count_;
    const size_t deletedBefore = idx.num_deleted_;
    const size_t verticesBefore = idx.vectors_.size();
    const size_t deletedFlagsBefore = idx.deleted_.size();

    // Force the mid-rebuild insert() failure: shrink capacity below the
    // surviving count (4 survivors), so rebuilt.insert() hits its very
    // first guard (count_ >= max_elements_) partway through reinsertion.
    idx.max_elements_ = 2;

    std::vector<uint32_t> out_map;
    idx.compact(out_map);

    // On failure, out_map must be cleared (the documented failure signal)...
    assert(out_map.empty());
    // ...and *this must be byte-for-byte the pre-compact object: this is
    // the bug this test guards against, where the old graph had already
    // been swapped away to empty vectors before the rebuild loop ran.
    assert(idx.count_ == countBefore);
    assert(idx.num_deleted_ == deletedBefore);
    assert(idx.vectors_.size() == verticesBefore);
    assert(idx.deleted_.size() == deletedFlagsBefore);

    // The still-fragmented-but-intact index must remain usable: a
    // surviving vector (id 7) must still find itself.
    auto results = idx.search(vecs[7].data(), 1);
    assert(!results.empty());
    assert(results[0].first == 7);

    std::printf("  ok: FloatHNSW compact() failure leaves object untouched\n");
}

void testUint8CompactFailureLeavesObjectUntouched() {
    Uint8FloatHNSWConfig config;
    config.M = 4;
    config.ef_construction = 32;
    config.ef_search = 32;
    config.max_elements = kCount;
    config.metric = DistanceMetric::L2;

    Uint8FloatHNSW idx(kDims, config);
    std::vector<std::vector<float>> vecs;
    for (size_t i = 0; i < kCount; i++) {
        vecs.push_back(makeVector(i));
        const uint32_t id = idx.insert(vecs.back().data());
        assert(id == i);
    }

    for (size_t i = 0; i < 6; i++) idx.mark_delete(static_cast<uint32_t>(i));
    assert(idx.num_deleted_ == 6);
    const size_t countBefore = idx.count_;
    const size_t deletedBefore = idx.num_deleted_;
    const size_t qdataBefore = idx.qdata_.size();
    const size_t deletedFlagsBefore = idx.deleted_.size();

    idx.max_elements_ = 2;

    std::vector<uint32_t> out_map;
    idx.compact(out_map);

    assert(out_map.empty());
    assert(idx.count_ == countBefore);
    assert(idx.num_deleted_ == deletedBefore);
    assert(idx.qdata_.size() == qdataBefore);
    assert(idx.deleted_.size() == deletedFlagsBefore);

    auto results = idx.search(vecs[7].data(), 1);
    assert(!results.empty());
    assert(results[0].first == 7);

    std::printf("  ok: Uint8FloatHNSW compact() failure leaves object untouched\n");
}

}  // namespace

int main() {
    testFloatCompactFailureLeavesObjectUntouched();
    testUint8CompactFailureLeavesObjectUntouched();
    std::printf("compact_test: all checks passed.\n");
    return 0;
}
