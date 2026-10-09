// BERT-style encoder forward (MiniLM-L6, or any model of the same shape
// with up to MAXL layers) over the block-affine u8 weight blob emitted by
// export_encoder_blob.py — the inline-encoder (kind 3) kernel.
// Float weight tensors are never materialized: every GEMV dequantizes
// inside the dot product, from u8 or u4 block-affine weights. LayerNorm, softmax, GELU, biases, and residuals
// run in f32. Every projection runs over all tokens at once, TOKEN_TILE
// tokens per pass over the weights (see gemm_tile), with bit-identical
// results to a token-at-a-time GEMV; the FFN intermediate is materialized
// for the whole sequence. Attention is direct (seq <= MAXSEQ, the score
// tile is KBs) and O(seq^2) per head per layer, so MAXSEQ trades
// embedding latency for window width — pinned to P (the position-embedding
// table size, 512), since a wider window has no positions to encode past
// that anyway.
//
// Layout constants mirror the exporter exactly; offsets are running sums
// in the same emit order. Everything is 16-byte aligned by construction.
// The layer count and the weight format are read at runtime: the host
// passes the declaration's L and bits, having checked the blob length
// against them.

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

namespace {

constexpr int V = 30522, P = 512, T = 2, D = 384, F = 1536, MAXL = 12;
constexpr int H = 12, HD = 32;
// Weight formats: u8 in blocks of 64, or u4 in blocks of 32 (two weights
// per byte; see gemm_tile). Each block carries an f32 scale and offset.
constexpr int B8 = 64, B4 = 32;
constexpr int MAXNB = F / B4;  // most blocks per row (F-wide, u4)
constexpr int MAXSEQ = 512;
constexpr float LN_EPS = 1e-12f;

struct QuantMat {
    const uint8_t* q;
    const float* s;
    const float* o;
};

struct Layer {
    QuantMat wq, wk, wv, wo, wu, wd;
    const float *bq, *bk, *bv, *bo, *bu, *bd;
    const float *ln1_g, *ln1_b, *ln2_g, *ln2_b;
};

struct Layout {
    QuantMat word, pos, type;
    const float *embln_g, *embln_b;
    Layer layer[MAXL];
};

const uint8_t* cursor_base;
size_t cursor;

const uint8_t* take(size_t bytes) {
    const uint8_t* p = cursor_base + cursor;
    cursor += bytes;
    return p;
}

QuantMat take_quant(int rows, int cols, bool q4) {
    QuantMat m;
    const int nb = cols / (q4 ? B4 : B8);
    m.q = take(q4 ? (size_t)rows * cols / 2 : (size_t)rows * cols);
    m.s = (const float*)take((size_t)rows * nb * 4);
    m.o = (const float*)take((size_t)rows * nb * 4);
    return m;
}

void fill_layout(const uint8_t* blob, int layers, bool q4, Layout& lay) {
    cursor_base = blob;
    cursor = 0;
    lay.word = take_quant(V, D, q4);
    lay.pos = take_quant(P, D, q4);
    lay.type = take_quant(T, D, q4);
    lay.embln_g = (const float*)take(D * 4);
    lay.embln_b = (const float*)take(D * 4);
    for (int i = 0; i < layers; i++) {
        Layer& ly = lay.layer[i];
        ly.wq = take_quant(D, D, q4); ly.bq = (const float*)take(D * 4);
        ly.wk = take_quant(D, D, q4); ly.bk = (const float*)take(D * 4);
        ly.wv = take_quant(D, D, q4); ly.bv = (const float*)take(D * 4);
        ly.wo = take_quant(D, D, q4); ly.bo = (const float*)take(D * 4);
        ly.ln1_g = (const float*)take(D * 4);
        ly.ln1_b = (const float*)take(D * 4);
        ly.wu = take_quant(F, D, q4); ly.bu = (const float*)take(F * 4);
        ly.wd = take_quant(D, F, q4); ly.bd = (const float*)take(D * 4);
        ly.ln2_g = (const float*)take(D * 4);
        ly.ln2_b = (const float*)take(D * 4);
    }
}

// The weight at index c (0..31) of a u4 block: the block's 16 bytes hold
// weight j in the low nibble of byte j and weight j+16 in the high nibble,
// so one AND and one shift unpack a whole block into two u8x16 vectors.
inline int u4_at(const uint8_t* block, int c) {
    return c < 16 ? (block[c] & 0x0F) : (block[c - 16] >> 4);
}

#ifdef __wasm_simd128__
// Widen 16 u8 weights to f32 and accumulate them against each tile
// token's 16 activations at xoff.
template <int NT>
inline void accumulate16(v128_t bytes, const float* X, int cols, int xoff, v128_t* vacc) {
    const v128_t lo16 = wasm_u16x8_extend_low_u8x16(bytes);
    const v128_t hi16 = wasm_u16x8_extend_high_u8x16(bytes);
    const v128_t w0 = wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(lo16));
    const v128_t w1 = wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(lo16));
    const v128_t w2 = wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(hi16));
    const v128_t w3 = wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(hi16));
    for (int t = 0; t < NT; t++) {
        const float* xb = X + t * cols + xoff;
        vacc[t] = wasm_f32x4_add(vacc[t], wasm_f32x4_mul(w0, wasm_v128_load(xb)));
        vacc[t] = wasm_f32x4_add(vacc[t], wasm_f32x4_mul(w1, wasm_v128_load(xb + 4)));
        vacc[t] = wasm_f32x4_add(vacc[t], wasm_f32x4_mul(w2, wasm_v128_load(xb + 8)));
        vacc[t] = wasm_f32x4_add(vacc[t], wasm_f32x4_mul(w3, wasm_v128_load(xb + 12)));
    }
}

#endif

// Fused block-affine GEMM over a tile of NT tokens: Y[t] = A X[t] + bias,
// A u8 (Q4 false) or u4 (Q4 true, scalar builds only), rows x cols; X and
// Y token-major (stride cols / rows). Each 16-weight chunk is widened to
// f32 once and applied to every token in the tile, so a layer's weights
// stream through memory once per NT tokens instead of once per token. Per
// token the arithmetic and its order are exactly the one-token GEMV's, so
// results are bit-identical whatever the tile size.
template <int NT, bool Q4>
void gemm_tile(const QuantMat& m, int rows, int cols, const float* X, const float* bias, float* Y) {
    constexpr int BS = Q4 ? B4 : B8;
    const int nblocks = cols / BS;
    const size_t rowBytes = Q4 ? (size_t)cols / 2 : (size_t)cols;
    float xsums[NT][MAXNB];
    for (int t = 0; t < NT; t++) {
        for (int b = 0; b < nblocks; b++) {
            float s = 0.f;
            for (int c = 0; c < BS; c++) s += X[t * cols + b * BS + c];
            xsums[t][b] = s;
        }
    }
#ifdef __wasm_simd128__
    // u4 SIMD runs through gemm_q4 instead (see there); this tile loop is
    // u8-only under SIMD.
    static_assert(!Q4, "u4 SIMD GEMM is gemm_q4");
    for (int r = 0; r < rows; r++) {
        const uint8_t* arow = m.q + r * rowBytes;
        const float* srow = m.s + (size_t)r * nblocks;
        const float* orow = m.o + (size_t)r * nblocks;
        float acc[NT];
        for (int t = 0; t < NT; t++) acc[t] = bias ? bias[r] : 0.f;
        for (int b = 0; b < nblocks; b++) {
            v128_t vacc[NT];
            for (int t = 0; t < NT; t++) vacc[t] = wasm_f32x4_const_splat(0.f);
            for (int c = 0; c < BS; c += 16) {
                accumulate16<NT>(wasm_v128_load(arow + b * BS + c), X, cols, b * BS + c, vacc);
            }
            for (int t = 0; t < NT; t++) {
                const float dot = wasm_f32x4_extract_lane(vacc[t], 0) + wasm_f32x4_extract_lane(vacc[t], 1)
                    + wasm_f32x4_extract_lane(vacc[t], 2) + wasm_f32x4_extract_lane(vacc[t], 3);
                acc[t] += srow[b] * dot + orow[b] * xsums[t][b];
            }
        }
        for (int t = 0; t < NT; t++) Y[t * rows + r] = acc[t];
    }
#else
    for (int t = 0; t < NT; t++) {
        const float* x = X + t * cols;
        for (int r = 0; r < rows; r++) {
            const uint8_t* arow = m.q + r * rowBytes;
            float acc = bias ? bias[r] : 0.f;
            for (int b = 0; b < nblocks; b++) {
                float dot = 0.f;
                for (int c = 0; c < BS; c++) {
                    const int w = Q4 ? u4_at(arow + b * (BS / 2), c) : arow[b * BS + c];
                    dot += (float)w * x[b * BS + c];
                }
                acc += m.s[(size_t)r * nblocks + b] * dot + m.o[(size_t)r * nblocks + b] * xsums[t][b];
            }
            Y[t * rows + r] = acc;
        }
    }
#endif
}

#ifndef TOKEN_TILE
#define TOKEN_TILE 4
#endif

// Y = A X + bias for all seq tokens: full tiles, then 4-, 2- and 1-token
// tiles for the remainder.
#ifdef __wasm_simd128__
#ifndef Q4_GROUP
#define Q4_GROUP 4
#endif
// Dot products of one dequantized f32 weight row against NT tokens.
template <int NT>
inline void dot_rows(const float* w, const float* X, int cols, float* out) {
    v128_t acc[NT];
    for (int t = 0; t < NT; t++) acc[t] = wasm_f32x4_const_splat(0.f);
    for (int c = 0; c < cols; c += 4) {
        const v128_t wv = wasm_v128_load(w + c);
        for (int t = 0; t < NT; t++) acc[t] = wasm_f32x4_add(acc[t], wasm_f32x4_mul(wv, wasm_v128_load(X + t * cols + c)));
    }
    for (int t = 0; t < NT; t++) {
        out[t] = wasm_f32x4_extract_lane(acc[t], 0) + wasm_f32x4_extract_lane(acc[t], 1)
            + wasm_f32x4_extract_lane(acc[t], 2) + wasm_f32x4_extract_lane(acc[t], 3);
    }
}

// u4 GEMM over the whole sequence: each weight row is dequantized once
// into an f32 row that stays in L1, then dotted with every token, so the
// unpack and dequantization are paid once per row per query rather than
// once per tile.
void gemm_q4(const QuantMat& m, int rows, int cols, const float* X, int seq, const float* bias, float* Y) {
    alignas(16) float wrow[F];
    const int nblocks = cols / B4;
    float dots[Q4_GROUP];
    for (int r = 0; r < rows; r++) {
        const uint8_t* arow = m.q + (size_t)r * (cols / 2);
        const float* srow = m.s + (size_t)r * nblocks;
        const float* orow = m.o + (size_t)r * nblocks;
        for (int b = 0; b < nblocks; b++) {
            const v128_t packed = wasm_v128_load(arow + b * (B4 / 2));
            const v128_t scale = wasm_f32x4_splat(srow[b]);
            const v128_t offset = wasm_f32x4_splat(orow[b]);
            const v128_t halves[2] = { wasm_v128_and(packed, wasm_u8x16_splat(0x0F)), wasm_u8x16_shr(packed, 4) };
            for (int h = 0; h < 2; h++) {
                const v128_t lo16 = wasm_u16x8_extend_low_u8x16(halves[h]);
                const v128_t hi16 = wasm_u16x8_extend_high_u8x16(halves[h]);
                float* dst = wrow + b * B4 + h * 16;
                wasm_v128_store(dst, wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(lo16)), scale), offset));
                wasm_v128_store(dst + 4, wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(lo16)), scale), offset));
                wasm_v128_store(dst + 8, wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(hi16)), scale), offset));
                wasm_v128_store(dst + 12, wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(hi16)), scale), offset));
            }
        }
        const float bb = bias ? bias[r] : 0.f;
        int t = 0;
        for (; t + Q4_GROUP <= seq; t += Q4_GROUP) {
            dot_rows<Q4_GROUP>(wrow, X + t * cols, cols, dots);
            for (int i = 0; i < Q4_GROUP; i++) Y[(t + i) * rows + r] = bb + dots[i];
        }
        for (; t < seq; t++) {
            dot_rows<1>(wrow, X + t * cols, cols, dots);
            Y[t * rows + r] = bb + dots[0];
        }
    }
}
#endif

template <bool Q4>
void gemm_seq(const QuantMat& m, int rows, int cols, const float* X, int seq, const float* bias, float* Y) {
    int t = 0;
    for (; t + TOKEN_TILE <= seq; t += TOKEN_TILE) gemm_tile<TOKEN_TILE, Q4>(m, rows, cols, X + t * cols, bias, Y + t * rows);
    for (; t + 4 <= seq; t += 4) gemm_tile<4, Q4>(m, rows, cols, X + t * cols, bias, Y + t * rows);
    for (; t + 2 <= seq; t += 2) gemm_tile<2, Q4>(m, rows, cols, X + t * cols, bias, Y + t * rows);
    for (; t < seq; t++) gemm_tile<1, Q4>(m, rows, cols, X + t * cols, bias, Y + t * rows);
}

// The blob's weight format, set per encoder_forward call (single query at
// a time, like the layout cursor).
bool g_q4;

void gemm(const QuantMat& m, int rows, int cols, const float* X, int seq, const float* bias, float* Y) {
#ifdef __wasm_simd128__
    if (g_q4) gemm_q4(m, rows, cols, X, seq, bias, Y);
#else
    if (g_q4) gemm_seq<true>(m, rows, cols, X, seq, bias, Y);
#endif
    else gemm_seq<false>(m, rows, cols, X, seq, bias, Y);
}

// Dequantize one D-wide row into f32 (embedding gather).
void dequant_row(const QuantMat& m, int row, float* out) {
    const int bs = g_q4 ? B4 : B8;
    const int nb = D / bs;
    const uint8_t* q = m.q + (size_t)row * (g_q4 ? D / 2 : D);
    const float* s = m.s + (size_t)row * nb;
    const float* o = m.o + (size_t)row * nb;
    for (int b = 0; b < nb; b++) {
        for (int c = 0; c < bs; c++) {
            const int w = g_q4 ? u4_at(q + b * (bs / 2), c) : q[b * bs + c];
            out[b * bs + c] = (float)w * s[b] + o[b];
        }
    }
}

void layernorm(float* x, const float* g, const float* bta) {
    float mu = 0.f;
    for (int d = 0; d < D; d++) mu += x[d];
    mu /= D;
    float var = 0.f;
    for (int d = 0; d < D; d++) { const float dv = x[d] - mu; var += dv * dv; }
    var /= D;
    const float inv = 1.f / sqrtf(var + LN_EPS);
    for (int d = 0; d < D; d++) x[d] = (x[d] - mu) * inv * g[d] + bta[d];
}

inline float gelu(float v) {
    return 0.5f * v * (1.f + erff(v * 0.70710678f));
}

// Working buffers (BSS; single query at a time).
float bx[MAXSEQ * D];
float bq_[MAXSEQ * D];
float bk_[MAXSEQ * D];
float bv_[MAXSEQ * D];
float bctx[MAXSEQ * D];
float btmp[D];
float bh[MAXSEQ * F];
float bscores[MAXSEQ];

} // namespace

extern "C" {

// ids: i32 token ids. outHidden: seq*D floats (final hidden states).
// dbgStages: null, or (1+layers)*seq*D floats — after-embedding-LN plus
// each layer's output, for stage parity against the torch references.
// layers: the blob's layer count, 1..MAXL. bits: its weight format, 8
// (u8, blocks of 64) or 4 (u4, blocks of 32).
// Returns seq on success, negative on error.
int encoder_forward(const uint8_t* blob, const int* ids, int seq, float* outHidden, float* dbgStages,
                    int layers, int bits) {
    if (seq < 1 || seq > MAXSEQ) return -1;
    if (layers < 1 || layers > MAXL) return -3;
    if (bits != 8 && bits != 4) return -4;
    g_q4 = bits == 4;
    Layout lay;
    fill_layout(blob, layers, g_q4, lay);

    for (int t = 0; t < seq; t++) {
        const int id = ids[t];
        if (id < 0 || id >= V) return -2;
        float* x = bx + t * D;
        dequant_row(lay.word, id, x);
        dequant_row(lay.pos, t, btmp);
        for (int d = 0; d < D; d++) x[d] += btmp[d];
        dequant_row(lay.type, 0, btmp);
        for (int d = 0; d < D; d++) x[d] += btmp[d];
        layernorm(x, lay.embln_g, lay.embln_b);
    }
    if (dbgStages) {
        for (int i = 0; i < seq * D; i++) dbgStages[i] = bx[i];
    }

    const float invSqrtHd = 1.f / sqrtf((float)HD);
    for (int li = 0; li < layers; li++) {
        const Layer& ly = lay.layer[li];
        gemm(ly.wq, D, D, bx, seq, ly.bq, bq_);
        gemm(ly.wk, D, D, bx, seq, ly.bk, bk_);
        gemm(ly.wv, D, D, bx, seq, ly.bv, bv_);
        for (int h = 0; h < H; h++) {
            const int off = h * HD;
            for (int ti = 0; ti < seq; ti++) {
                const float* qv = bq_ + ti * D + off;
                float maxScore = -1e30f;
                for (int tj = 0; tj < seq; tj++) {
                    const float* kv = bk_ + tj * D + off;
                    float s = 0.f;
                    for (int d = 0; d < HD; d++) s += qv[d] * kv[d];
                    s *= invSqrtHd;
                    bscores[tj] = s;
                    if (s > maxScore) maxScore = s;
                }
                float denom = 0.f;
                for (int tj = 0; tj < seq; tj++) {
                    bscores[tj] = expf(bscores[tj] - maxScore);
                    denom += bscores[tj];
                }
                float* out = bctx + ti * D + off;
                for (int d = 0; d < HD; d++) out[d] = 0.f;
                for (int tj = 0; tj < seq; tj++) {
                    const float a = bscores[tj] / denom;
                    const float* vv = bv_ + tj * D + off;
                    for (int d = 0; d < HD; d++) out[d] += a * vv[d];
                }
            }
        }
        // bq_ is dead once attention has run; it holds the projections.
        gemm(ly.wo, D, D, bctx, seq, ly.bo, bq_);
        for (int t = 0; t < seq; t++) {
            float* x = bx + t * D;
            for (int d = 0; d < D; d++) x[d] += bq_[t * D + d];
            layernorm(x, ly.ln1_g, ly.ln1_b);
        }
        gemm(ly.wu, F, D, bx, seq, ly.bu, bh);
        for (int i = 0; i < seq * F; i++) bh[i] = gelu(bh[i]);
        gemm(ly.wd, D, F, bh, seq, ly.bd, bq_);
        for (int t = 0; t < seq; t++) {
            float* x = bx + t * D;
            for (int d = 0; d < D; d++) x[d] += bq_[t * D + d];
            layernorm(x, ly.ln2_g, ly.ln2_b);
        }
        if (dbgStages) {
            float* dst = dbgStages + (1 + li) * seq * D;
            for (int i = 0; i < seq * D; i++) dst[i] = bx[i];
        }
    }

    for (int i = 0; i < seq * D; i++) outHidden[i] = bx[i];
    return seq;
}

} // extern "C"
