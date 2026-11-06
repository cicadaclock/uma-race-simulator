// .NET Framework System.Random (src/rng.rs NetRandom) and the race's wrapper around it (RaceRandom).
//
// Words 0..55: seed array, 56: inext, 57: inextp, 58 / 59: draw count lo / hi. Two places for them:
// * with NET_RANDOM_IN_STATE defined, the race state's rng field (race_state.glsl, include it first): the stream
//   argument `s` is ignored and every call uses this invocation's lane;
// * otherwise a storage buffer at NET_RANDOM_BINDING, one stream per race: word w of stream s is
//   rng_words[w * rng_stride + s], so the streams of a wave are adjacent. Set rng_stride (the stream count) in
//   main().
//
// Include after exact.glsl.

#ifndef UMA_NET_RANDOM_GLSL
#define UMA_NET_RANDOM_GLSL

const uint NET_RANDOM_WORDS = 60u;

#ifdef NET_RANDOM_IN_STATE
int rng_get(uint s, uint w) { return int(r_rng(w)); }
void rng_set(uint s, uint w, int v) { set_r_rng(w, uint(v)); }
#else
#ifndef NET_RANDOM_BINDING
#error "define NET_RANDOM_BINDING (or NET_RANDOM_IN_STATE) before including net_random.glsl"
#endif
layout(std430, binding = NET_RANDOM_BINDING) buffer NetRandomState { int rng_words[]; };
uint rng_stride;
int rng_get(uint s, uint w) { return rng_words[w * rng_stride + s]; }
void rng_set(uint s, uint w, int v) { rng_words[w * rng_stride + s] = v; }
#endif

const int NET_MBIG = 2147483647;
const int NET_MSEED = 161803398;

// new Random(seed). Integer arithmetic wraps exactly like Rust's wrapping_add / wrapping_sub.
void net_random_init(uint s, int seed) {
    for (uint w = 0u; w < NET_RANDOM_WORDS; w++) rng_set(s, w, 0);
    int subtraction = seed == int(0x80000000u) ? NET_MBIG : abs(seed);
    int mj = NET_MSEED - subtraction;
    rng_set(s, 55u, mj);
    int mk = 1;
    uint ii = 0u;
    for (int k = 1; k < 55; k++) {
        ii += 21u;
        if (ii >= 55u) ii -= 55u;
        rng_set(s, ii, mk);
        mk = mj - mk;
        if (mk < 0) mk += NET_MBIG;
        mj = rng_get(s, ii);
    }
    for (int k = 1; k < 5; k++) {
        for (uint i = 1u; i < 56u; i++) {
            uint n = i + 30u;
            if (n >= 55u) n -= 55u;
            int v = rng_get(s, i) - rng_get(s, 1u + n);
            if (v < 0) v += NET_MBIG;
            rng_set(s, i, v);
        }
    }
    rng_set(s, 56u, 0);
    rng_set(s, 57u, 21);
}

int net_random_internal_sample(uint s) {
    int a = rng_get(s, 56u) + 1;
    if (a >= 56) a = 1;
    int b = rng_get(s, 57u) + 1;
    if (b >= 56) b = 1;
    int r = rng_get(s, uint(a)) - rng_get(s, uint(b));
    if (r == NET_MBIG) r -= 1;
    if (r < 0) r += NET_MBIG;
    rng_set(s, uint(a), r);
    rng_set(s, 56u, a);
    rng_set(s, 57u, b);
    uint lo = uint(rng_get(s, 58u)) + 1u;
    rng_set(s, 58u, int(lo));
    if (lo == 0u) rng_set(s, 59u, rng_get(s, 59u) + 1);
    return r;
}

uint64_t net_random_draws(uint s) { return uint64_t(uint(rng_get(s, 58u))) | (uint64_t(uint(rng_get(s, 59u))) << 32); }

// NextDouble() = sample * (1.0 / int.MaxValue), with the constant's exact bits.
double net_random_next_double(uint s) {
    return double(net_random_internal_sample(s)) * packDouble2x32(uvec2(0x00200000u, 0x3E000000u));
}

// Next(min, max), max exclusive (min <= max is the caller's responsibility).
int net_random_next_range(uint s, int mn, int mx) {
    int64_t range = int64_t(mx) - int64_t(mn);
    if (range <= int64_t(NET_MBIG)) {
        // Truncation; the product is in [0, range).
        return int(net_random_next_double(s) * double(range)) + mn;
    }
    // GetSampleForLargeRange: never hit by race code.
    int r = net_random_internal_sample(s);
    bool neg = (net_random_internal_sample(s) % 2) == 0;
    double rd = double(neg ? -r : r);
    double d = div_rn64(rd + (double(NET_MBIG) - 1.0lf), 2.0lf * double(NET_MBIG) - 1.0lf);
    return int(int64_t(d * double(range)) + int64_t(mn));
}

// RaceRandom
int race_random_int_range(uint s, int mn, int mx) { return net_random_next_range(s, mn, mx); }
int race_random_int(uint s, int mx) { return net_random_next_range(s, 0, mx); }

// (float)NextDouble() * |max - min| + min (product and sum in f64, one rounding), clamped to [min, max].
float race_random_float_range(uint s, float mn, float mx) {
    double d = net_random_next_double(s);
    float f = float(double(float(d)) * double(abs(mx - mn)) + double(mn));
    return mn > f ? mn : (f > mx ? mx : f);
}
float race_random_float(uint s, float mx) { return race_random_float_range(s, 0.0, mx); }

#endif
