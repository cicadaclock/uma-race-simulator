// Bit-exact float helpers for the GPU port.
//
// * The src/sim/util.rs helpers with the game's semantics (NaN, signed zero, out-of-range casts).
// * Correctly rounded division and square root in f32 and f64, for every input. Vulkan only promises 2.5 ulp for
//   `/` (AMD's driver misrounds about 30% of f32 and 26% of f64 quotients); Rust and the game round correctly, so
//   every `/` and `sqrt` in ported code goes through div_rn32 / div_rn64 / sqrt_rn32 / sqrt_rn64. The hardware
//   result is only the starting guess.
//
// Requirements, all checked by gpu_selftest on the device:
// * The module is patched by src/gpu/spirv.rs (NoContraction on every float op), so `a * b + c` is never fused.
// * f32/f64 add, sub, mul and conversions round to nearest even.
// Vulkan lets fma() round twice, so the host probes it (Gpu::new) and sets UMA_FMA32_FUSED / UMA_FMA64_FUSED. With a
// fused fma the residuals below take one instruction; without, they are computed exactly in f64 (for f32) or with
// Dekker's two-product (for f64).
//
// Porting rule: write each expression with the same f32 / f64 casts as the Rust, and never use a float literal
// that is not exactly representable (use uintBitsToFloat with the Rust constant's bits instead).

#ifndef UMA_EXACT_GLSL
#define UMA_EXACT_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Set by the host from its fma probe (src/gpu/context.rs).
layout(constant_id = 0) const bool UMA_FMA32_FUSED = false;
layout(constant_id = 1) const bool UMA_FMA64_FUSED = false;

// ---------------------------------------------------------------------------------------------------------------
// Bit-level helpers. They work on the encoding, so they behave the same on devices that flush denormals.

bool is_finite32(float x) { return (floatBitsToUint(x) & 0x7f800000u) != 0x7f800000u; }
bool is_nan32(float x) { return (floatBitsToUint(x) & 0x7fffffffu) > 0x7f800000u; }
bool is_finite64(double x) { return (unpackDouble2x32(x).y & 0x7ff00000u) != 0x7ff00000u; }
bool is_nan64(double x) { return (doubleBitsToUint64(x) & 0x7fffffffffffffffUL) > 0x7ff0000000000000UL; }
// Biased exponent field: 0 for zero and subnormals, 255 / 2047 for Inf and NaN.
uint exp_field32(float x) { return (floatBitsToUint(x) >> 23) & 0xffu; }
uint exp_field64(double x) { return (unpackDouble2x32(x).y >> 20) & 0x7ffu; }
bool sign_bit32(float x) { return (floatBitsToUint(x) >> 31) != 0u; }
bool sign_bit64(double x) { return (unpackDouble2x32(x).y >> 31) != 0u; }
float neg_bits32(float x) { return uintBitsToFloat(floatBitsToUint(x) ^ 0x80000000u); }
double neg_bits64(double x) { return uint64BitsToDouble(doubleBitsToUint64(x) ^ 0x8000000000000000UL); }

// The adjacent value toward +Inf / -Inf of a finite x.
float next_up32(float x) {
    uint u = floatBitsToUint(x);
    if ((u & 0x7fffffffu) == 0u) return uintBitsToFloat(1u);
    return uintBitsToFloat((u >> 31) == 0u ? u + 1u : u - 1u);
}
float next_down32(float x) { return neg_bits32(next_up32(neg_bits32(x))); }
double next_up64(double x) {
    uint64_t u = doubleBitsToUint64(x);
    if ((u & 0x7fffffffffffffffUL) == 0UL) return uint64BitsToDouble(1UL);
    return uint64BitsToDouble((u >> 63) == 0UL ? u + 1UL : u - 1UL);
}
double next_down64(double x) { return neg_bits64(next_up64(neg_bits64(x))); }

// ---------------------------------------------------------------------------------------------------------------
// Special values, decided on the bits so they never depend on the hardware's / or sqrt.

bool is_zero32(float x) { return (floatBitsToUint(x) & 0x7fffffffu) == 0u; }
bool is_zero64(double x) { return (doubleBitsToUint64(x) & 0x7fffffffffffffffUL) == 0UL; }
float signed32(bool neg, uint magnitude) { return uintBitsToFloat(magnitude | (neg ? 0x80000000u : 0u)); }
double signed64(bool neg, uint64_t magnitude) { return uint64BitsToDouble(magnitude | (neg ? 0x8000000000000000UL : 0UL)); }

// a / b when a or b is zero, Inf or NaN (IEEE 754).
float div_special32(float a, float b) {
    bool neg = sign_bit32(a) != sign_bit32(b);
    if (is_nan32(a) || is_nan32(b) || (is_zero32(a) && is_zero32(b)) || (!is_finite32(a) && !is_finite32(b))) return uintBitsToFloat(0x7fc00000u);
    if (!is_finite32(a) || is_zero32(b)) return signed32(neg, 0x7f800000u);
    return signed32(neg, 0u);
}
double div_special64(double a, double b) {
    bool neg = sign_bit64(a) != sign_bit64(b);
    if (is_nan64(a) || is_nan64(b) || (is_zero64(a) && is_zero64(b)) || (!is_finite64(a) && !is_finite64(b))) return uint64BitsToDouble(0x7ff8000000000000UL);
    if (!is_finite64(a) || is_zero64(b)) return signed64(neg, 0x7ff0000000000000UL);
    return signed64(neg, 0UL);
}
// sqrt(x) when x is NaN, ±0, negative or +Inf.
float sqrt_special32(float x) { return (is_nan32(x) || is_zero32(x) || !sign_bit32(x)) ? x : uintBitsToFloat(0x7fc00000u); }
double sqrt_special64(double x) { return (is_nan64(x) || is_zero64(x) || !sign_bit64(x)) ? x : uint64BitsToDouble(0x7ff8000000000000UL); }

// Whether a guess is usable: finite, same sign, within 8 ulps of a reference.
bool near32(float q, float g) {
    if (!is_finite32(q) || sign_bit32(q) != sign_bit32(g)) return false;
    uint u = floatBitsToUint(q), v = floatBitsToUint(g);
    return (u > v ? u - v : v - u) <= 8u;
}

// ---------------------------------------------------------------------------------------------------------------
// Correctly rounded division.
//
// Given a guess q within a few ulps of a / b, step to whichever neighbour has the smaller |a - q*b|. Since
// |a - q*b| = |b| * |a/b - q|, the smallest residual is the nearest value, and near the answer the residuals are
// exactly representable, so each is computed exactly. A quotient is never exactly halfway between two normal
// floats; between two subnormals it can be, so the subnormal paths break ties to even.
//
// The loops return false when they have not settled after REFINE_STEPS (a guess too far off); the caller then
// starts again from a better guess. The hardware guess is far off for subnormals on AMD, for example.

const int REFINE_STEPS = 12;

bool div_rn32_fma(float a, float b, inout float q) {
    for (int i = 0; i < REFINE_STEPS; i++) {
        float qu = next_up32(q), qd = next_down32(q);
        float r = abs(fma(-q, b, a));
        float ru = abs(fma(-qu, b, a));
        float rd = abs(fma(-qd, b, a));
        if (ru < r) q = qu;
        else if (rd < r) q = qd;
        else return true;
    }
    return false;
}

// Exact for every finite a, b != 0 and finite quotient, subnormals included: in f64 the products q*b of f32
// values and the differences a - q*b are exact.
bool div_rn32_f64(float a, float b, inout float q) {
    double da = double(a), db = double(b);
    for (int i = 0; i < REFINE_STEPS; i++) {
        float qu = next_up32(q), qd = next_down32(q);
        double r = abs(da - double(q) * db);
        double ru = abs(da - double(qu) * db);
        double rd = abs(da - double(qd) * db);
        if (ru < r || (ru == r && (floatBitsToUint(qu) & 1u) == 0u)) q = qu;
        else if (rd < r || (rd == r && (floatBitsToUint(qd) & 1u) == 0u)) q = qd;
        else return true;
    }
    return false;
}

double div_rn64(double a, double b);
double sqrt_rn64(double x);

// Rust `a / b` on f32, refined from the guess q. Exact for all inputs.
float div_rn32_from(float a, float b, float q) {
    if (!is_finite32(a) || !is_finite32(b) || is_zero32(a) || is_zero32(b)) return div_special32(a, b);
    bool neg = sign_bit32(a) != sign_bit32(b);
    uint ea = exp_field32(a), eb = exp_field32(b), eq = exp_field32(q);
    // Fast path: |a| in [2^-76, 2^127), b and q normal. The residual grid ulp(q)*ulp(b) ~ |a| * 2^-46 stays normal.
    if (UMA_FMA32_FUSED && ea >= 51u && ea <= 253u && eb >= 1u && eb <= 253u && eq >= 1u && eq <= 253u) {
        float r = q;
        if (div_rn32_fma(a, b, r)) return r;
    }
    // Everything else: from the guess if it is near the correctly rounded f64 quotient, else from that (within an
    // ulp; the hardware guess can be far off, e.g. for subnormals).
    float g = float(div_rn64(double(a), double(b)));
    if (!is_finite32(g)) return g;  // overflow
    if (!near32(q, g)) q = g;
    div_rn32_f64(a, b, q);
    return is_zero32(q) ? signed32(neg, 0u) : q;  // underflow to zero keeps the quotient's sign
}
float div_rn32(float a, float b) { return div_rn32_from(a, b, a / b); }

// Veltkamp split: hi + lo == x exactly, each with at most 26 significant bits.
void split64(double x, out double hi, out double lo) {
    double c = 134217729.0lf * x;  // 2^27 + 1
    hi = c - (c - x);
    lo = x - hi;
}

// a - q*b with a single rounding. With a fused fma that is the fma; otherwise Dekker's two-product gives
// p + e == q*b exactly, a - p is exact (Sterbenz) when q*b is within a factor of two of a, and the final
// subtraction rounds once. Needs magnitudes in [2^-400, 2^400] so that no partial product under- or overflows.
double residual64(double a, double q, double b) {
    if (UMA_FMA64_FUSED) return fma(-q, b, a);
    double p = q * b;
    double qh, ql, bh, bl;
    split64(q, qh, ql);
    split64(b, bh, bl);
    double e = ((qh * bh - p) + qh * bl + ql * bh) + ql * bl;
    return (a - p) - e;
}

bool div_rn64_refine(double a, double b, inout double q) {
    for (int i = 0; i < REFINE_STEPS; i++) {
        double qu = next_up64(q), qd = next_down64(q);
        double r = abs(residual64(a, q, b));
        double ru = abs(residual64(a, qu, b));
        double rd = abs(residual64(a, qd, b));
        if (ru < r) q = qu;
        else if (rd < r) q = qd;
        else return true;
    }
    return false;
}

// a, b, q in [2^-400, 2^400]. A guess that does not settle gets a Newton step q += (a - q*b) / b and another try;
// the hardware division in that step only needs the single precision Vulkan guarantees.
double div_rn64_core(double a, double b, double q) {
    for (int round = 0; round < 4; round++) {
        if (div_rn64_refine(a, b, q)) return q;
        q = q + residual64(a, q, b) / b;
    }
    return q;
}

// |x| in [2^-400, 2^400]: exponent field 623..1423.
bool in_refine_range64(double x) { uint e = exp_field64(x); return e >= 623u && e <= 1423u; }

// x = m * 2^k with |m| in [1, 2); x finite and nonzero.
double frexp1_64(double x, out int k) {
    int bias = 0;
    if (exp_field64(x) == 0u) {
        x = x * 18446744073709551616.0lf;  // 2^64, exact for a subnormal
        bias = 64;
    }
    uint64_t u = doubleBitsToUint64(x);
    k = int((u >> 52) & 0x7ffUL) - 1023 - bias;
    return uint64BitsToDouble((u & 0x800fffffffffffffUL) | 0x3ff0000000000000UL);
}
// 2^k for k in [-1022, 1023].
double pow2_64(int k) { return uint64BitsToDouble(uint64_t(k + 1023) << 52); }

// RN((ma / mb) * 2^k) for a result below 2^-1022, with ma, mb in [1, 2) and q = RN(ma / mb). The result is
// n * 2^-1074, and |value - n * 2^-1074| is proportional to |ma - n * 2^-m * mb| with m = 1074 + k, so the same
// exact residuals pick n (ties to even).
double div_rn64_subnormal(double ma, double mb, double q, int k) {
    int m = 1074 + k;
    if (m < -1) return 0.0lf;  // below half the smallest subnormal
    double f = pow2_64(-m);
    double n = roundEven(q * pow2_64(m));
    for (int i = 0; i < REFINE_STEPS; i++) {
        double r = abs(residual64(ma, n * f, mb));
        double ru = abs(residual64(ma, (n + 1.0lf) * f, mb));
        double rd = n > 0.0lf ? abs(residual64(ma, (n - 1.0lf) * f, mb)) : r;
        bool even_up = (uint64_t(n + 1.0lf) & 1UL) == 0UL;
        if (ru < r || (ru == r && even_up)) n = n + 1.0lf;
        else if (n > 0.0lf && (rd < r || (rd == r && even_up))) n = n - 1.0lf;  // n - 1 is even iff n + 1 is
        else break;
    }
    return uint64BitsToDouble(uint64_t(n));  // n = 2^52 is the smallest normal
}

// Finite nonzero a, b anywhere in range: divide the significands, then scale by the exponent difference.
double div_rn64_scaled(double a, double b) {
    bool neg = sign_bit64(a) != sign_bit64(b);
    int ka, kb;
    double ma = abs(frexp1_64(a, ka)), mb = abs(frexp1_64(b, kb));
    double q = div_rn64_core(ma, mb, ma / mb);  // in [0.5, 2]
    int k = ka - kb;
    int e = int(exp_field64(q)) - 1023 + k;  // exponent of the result
    double r;
    if (e > 1023) r = uint64BitsToDouble(0x7ff0000000000000UL);
    else if (e >= -1022) {
        int h = k / 2;  // two exact power-of-two steps, each in range
        r = (q * pow2_64(h)) * pow2_64(k - h);
    } else r = div_rn64_subnormal(ma, mb, q, k);
    return neg ? neg_bits64(r) : r;
}

// Rust `a / b` on f64, refined from the guess q. Exact for all inputs.
double div_rn64_from(double a, double b, double q) {
    if (!is_finite64(a) || !is_finite64(b) || is_zero64(a) || is_zero64(b)) return div_special64(a, b);
    if (in_refine_range64(a) && in_refine_range64(b) && in_refine_range64(q) && sign_bit64(q) == (sign_bit64(a) != sign_bit64(b))) {
        return div_rn64_core(a, b, q);
    }
    return div_rn64_scaled(a, b);
}
double div_rn64(double a, double b) { return div_rn64_from(a, b, a / b); }

// ---------------------------------------------------------------------------------------------------------------
// Correctly rounded square root.
//
// For s > 0 with ulp u above and u' below, and r = x - s*s (exact when s is within an ulp or two):
//   sqrt(x) is above s + u/2   iff  r >  s*u
//   sqrt(x) is below s - u'/2  iff  r <= -s*u'
// because r, s*u and s*u' are all multiples of ulp(s)^2. The square root of a float is never exactly halfway.

bool sqrt_rn32_fma(float x, inout float s) {
    for (int i = 0; i < REFINE_STEPS; i++) {
        float su = next_up32(s), sd = next_down32(s);
        float r = fma(-s, s, x);
        if (r > s * (su - s)) s = su;
        else if (r <= -(s * (s - sd))) s = sd;
        else return true;
    }
    return false;
}

// Exact for every positive finite x: the midpoints between f32 neighbours and their squares are exact in f64.
bool sqrt_rn32_f64(float x, inout float s) {
    double dx = double(x);
    for (int i = 0; i < REFINE_STEPS; i++) {
        float su = next_up32(s), sd = next_down32(s);
        double hi = (double(s) + double(su)) * 0.5lf;
        double lo = (double(s) + double(sd)) * 0.5lf;
        if (dx > hi * hi) s = su;
        else if (dx < lo * lo) s = sd;
        else return true;
    }
    return false;
}

// Rust `x.sqrt()` on f32, refined from the guess s. Exact for all inputs.
float sqrt_rn32_from(float x, float s) {
    if (is_nan32(x) || is_zero32(x) || sign_bit32(x) || !is_finite32(x)) return sqrt_special32(x);
    if (UMA_FMA32_FUSED && exp_field32(x) >= 49u && exp_field32(s) >= 1u && is_finite32(s) && !sign_bit32(s)) {  // x >= 2^-78
        float r = s;
        if (sqrt_rn32_fma(x, r)) return r;
    }
    float g = float(sqrt_rn64(double(x)));
    if (!near32(s, g)) s = g;
    sqrt_rn32_f64(x, s);
    return s;
}
float sqrt_rn32(float x) { return sqrt_rn32_from(x, sqrt(x)); }

bool sqrt_rn64_refine(double x, inout double s) {
    for (int i = 0; i < REFINE_STEPS; i++) {
        double su = next_up64(s), sd = next_down64(s);
        double r = residual64(x, s, s);
        if (r > s * (su - s)) s = su;
        else if (r <= -(s * (s - sd))) s = sd;
        else return true;
    }
    return false;
}

// x, s in [2^-400, 2^400]. A guess that does not settle gets a Newton step s += (x - s*s) / 2s and another try.
double sqrt_rn64_core(double x, double s) {
    for (int round = 0; round < 4; round++) {
        if (sqrt_rn64_refine(x, s)) return s;
        s = s + residual64(x, s, s) / (2.0lf * s);
    }
    return s;
}

// Rust `x.sqrt()` on f64, refined from the guess s. Exact for all inputs: outside the refine range x is split
// into m * 2^2j with m in [1, 4), and sqrt(m) * 2^j is exact (the root of any positive double is normal).
double sqrt_rn64_from(double x, double s) {
    if (is_nan64(x) || is_zero64(x) || sign_bit64(x) || !is_finite64(x)) return sqrt_special64(x);
    if (in_refine_range64(x) && in_refine_range64(s) && !sign_bit64(s)) return sqrt_rn64_core(x, s);
    int k;
    double m = frexp1_64(x, k);
    if ((k & 1) != 0) {
        m = m * 2.0lf;
        k = k - 1;
    }
    return sqrt_rn64_core(m, sqrt(m)) * pow2_64(k / 2);
}
double sqrt_rn64(double x) { return sqrt_rn64_from(x, sqrt(x)); }

// ---------------------------------------------------------------------------------------------------------------
// src/sim/util.rs

// Mathf.Max(a, b) = a > b ? a : b (NaN in a -> b).
float mathf_max(float a, float b) { return a > b ? a : b; }
// Mathf.Min(a, b) = a < b ? a : b.
float mathf_min(float a, float b) { return a < b ? a : b; }
// Mathf.Clamp: if (v < lo) v = lo; else if (v > hi) v = hi; (NaN passes through).
float mathf_clamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
float mathf_clamp01(float v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }
// Mathf.Lerp(a, b, t) = a + (b - a) * Clamp01(t), all f32.
float mathf_lerp(float a, float b, float t) { return a + (b - a) * mathf_clamp01(t); }

// Mathf.Approximately(a, b) = Abs(b - a) < Max(1e-6f * Max(Abs(a), Abs(b)), Epsilon * 8f); NaN -> false.
// Epsilon * 8 = 2^-146 is a denormal, so the comparisons against it are done on the bit patterns (ordered like
// the values for non-negative floats); that keeps approximately(0, 0) true on a device that flushes denormals.
bool mathf_approximately(float a, float b) {
    float d = abs(b - a);
    float t = uintBitsToFloat(0x358637BDu) * mathf_max(abs(a), abs(b));  // 1e-6f
    const uint EPS8 = 8u;                                                  // bits of Mathf.Epsilon * 8
    uint m = (!is_nan32(t) && floatBitsToUint(t) > EPS8) ? floatBitsToUint(t) : EPS8;
    return !is_nan32(d) && floatBitsToUint(d) < m;
}

// (int)f: truncation; NaN / out of range -> int.MinValue (the comparisons are false for NaN).
int f32_to_i32(float f) { return (f >= -2147483648.0 && f < 2147483648.0) ? int(f) : int(0x80000000u); }
int f64_to_i32(double d) { return (d >= -2147483648.0lf && d < 2147483648.0lf) ? int(d) : int(0x80000000u); }
// Mathf.FloorToInt(f) = (int)Math.Floor((double)f).
int floor_to_int(float f) { return f64_to_i32(floor(double(f))); }

// Mathf.Pow(x, 2) = (float)Math.Pow(x, 2): the f64 square of an f32 is exact, so this is the f32 x * x.
float mathf_pow2(float x) { return x * x; }

#endif
