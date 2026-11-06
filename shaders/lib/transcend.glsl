// Mathf.Pow and Mathf.Log10 as the sim calls them: (float)Math.Pow((double)x, (double)y), (float)Math.Log10(x).
//
// Computed in double-double (about 2^-96 relative), so the f64 value before the final rounding to f32 is the
// correctly rounded f64 result except within ~2^-96 of an f64 rounding boundary. The CPU sim's libm (UCRT on
// Windows) is accurate to under an ulp, so the two agree on the f32 result except in rare cases where the exact value
// sits next to an f32 rounding boundary; `gpu_mathcheck` compares every f32 input the sim can pass at each call
// site and lists any such input. Needs exact.glsl (frexp1_64, pow2_64, split64).
//
// Inputs must be positive and finite (every call site passes a clamped stat times a positive constant); anything
// else sets transcend_fault.

#ifndef UMA_TRANSCEND_GLSL
#define UMA_TRANSCEND_GLSL

bool transcend_fault = false;

// ---------------------------------------------------------------------------------------------------------------
// Double-double arithmetic: value = v.x + v.y with |v.y| <= ulp(v.x) / 2 (Dekker, Knuth; as in the QD library).

dvec2 dd_quick_two_sum(double a, double b) {
    double s = a + b;
    return dvec2(s, b - (s - a));
}
dvec2 dd_two_sum(double a, double b) {
    double s = a + b;
    double bb = s - a;
    return dvec2(s, (a - (s - bb)) + (b - bb));
}
dvec2 dd_two_prod(double a, double b) {
    double p = a * b;
    if (UMA_FMA64_FUSED) return dvec2(p, fma(a, b, -p));
    double ah, al, bh, bl;
    split64(a, ah, al);
    split64(b, bh, bl);
    return dvec2(p, ((ah * bh - p) + ah * bl + al * bh) + al * bl);
}
dvec2 dd_add(dvec2 a, dvec2 b) {
    dvec2 s = dd_two_sum(a.x, b.x);
    dvec2 t = dd_two_sum(a.y, b.y);
    s.y += t.x;
    s = dd_quick_two_sum(s.x, s.y);
    s.y += t.y;
    return dd_quick_two_sum(s.x, s.y);
}
dvec2 dd_neg(dvec2 a) { return dvec2(-a.x, -a.y); }
dvec2 dd_mul(dvec2 a, dvec2 b) {
    dvec2 p = dd_two_prod(a.x, b.x);
    p.y += a.x * b.y + a.y * b.x;
    return dd_quick_two_sum(p.x, p.y);
}
dvec2 dd_mul_d(dvec2 a, double b) {
    dvec2 p = dd_two_prod(a.x, b);
    p.y += a.y * b;
    return dd_quick_two_sum(p.x, p.y);
}
// Three quotient digits; the hardware division only picks them, the residuals are exact.
dvec2 dd_div(dvec2 a, dvec2 b) {
    double q1 = a.x / b.x;
    dvec2 r = dd_add(a, dd_neg(dd_mul_d(b, q1)));
    double q2 = r.x / b.x;
    r = dd_add(r, dd_neg(dd_mul_d(b, q2)));
    double q3 = r.x / b.x;
    return dd_add(dd_quick_two_sum(q1, q2), dvec2(q3, 0.0lf));
}
dvec2 dd_scale(dvec2 a, double pow2) { return dvec2(a.x * pow2, a.y * pow2); }

// ln 2 and 1 / ln 10
const uint64_t DD_LN2_HI = 0x3FE62E42FEFA39EFUL, DD_LN2_LO = 0x3C7ABC9E3B39803FUL;
const uint64_t DD_INV_LN10_HI = 0x3FDBCB7B1526E50EUL, DD_INV_LN10_LO = 0x3C695355BAAAFAD3UL;
// 1 / (2k + 1), k = 0..23 (atanh series)
const uint64_t DD_ATANH_HI[24] = uint64_t[](0x3FF0000000000000UL, 0x3FD5555555555555UL, 0x3FC999999999999AUL, 0x3FC2492492492492UL, 0x3FBC71C71C71C71CUL, 0x3FB745D1745D1746UL, 0x3FB3B13B13B13B14UL, 0x3FB1111111111111UL, 0x3FAE1E1E1E1E1E1EUL, 0x3FAAF286BCA1AF28UL, 0x3FA8618618618618UL, 0x3FA642C8590B2164UL, 0x3FA47AE147AE147BUL, 0x3FA2F684BDA12F68UL, 0x3FA1A7B9611A7B96UL, 0x3FA0842108421084UL, 0x3F9F07C1F07C1F08UL, 0x3F9D41D41D41D41DUL, 0x3F9BACF914C1BAD0UL, 0x3F9A41A41A41A41AUL, 0x3F98F9C18F9C18FAUL, 0x3F97D05F417D05F4UL, 0x3F96C16C16C16C17UL, 0x3F95C9882B931057UL);
const uint64_t DD_ATANH_LO[24] = uint64_t[](0x0000000000000000UL, 0x3C75555555555555UL, 0xBC6999999999999AUL, 0x3C62492492492492UL, 0x3C5C71C71C71C71CUL, 0xBC4745D1745D1746UL, 0xBC53B13B13B13B14UL, 0x3C31111111111111UL, 0x3C2E1E1E1E1E1E1EUL, 0x3C4AF286BCA1AF28UL, 0x3C48618618618618UL, 0x3C3642C8590B2164UL, 0xBC2EB851EB851EB8UL, 0x3C42F684BDA12F68UL, 0x3C21A7B9611A7B96UL, 0x3C30842108421084UL, 0xBC2F07C1F07C1F08UL, 0x3C30750750750750UL, 0xBC3BACF914C1BAD0UL, 0x3C30690690690690UL, 0xBC2F3831F3831F38UL, 0x3C17D05F417D05F4UL, 0xBC2F49F49F49F49FUL, 0x3C2310572620AE4CUL);
// 1 / n!, n = 0..13 (exp series)
const uint64_t DD_INVFACT_HI[14] = uint64_t[](0x3FF0000000000000UL, 0x3FF0000000000000UL, 0x3FE0000000000000UL, 0x3FC5555555555555UL, 0x3FA5555555555555UL, 0x3F81111111111111UL, 0x3F56C16C16C16C17UL, 0x3F2A01A01A01A01AUL, 0x3EFA01A01A01A01AUL, 0x3EC71DE3A556C734UL, 0x3E927E4FB7789F5CUL, 0x3E5AE64567F544E4UL, 0x3E21EED8EFF8D898UL, 0x3DE6124613A86D09UL);
const uint64_t DD_INVFACT_LO[14] = uint64_t[](0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x3C65555555555555UL, 0x3C45555555555555UL, 0x3C01111111111111UL, 0xBBEF49F49F49F49FUL, 0x3B6A01A01A01A01AUL, 0x3B3A01A01A01A01AUL, 0xBB6C154F8DDC6C00UL, 0x3B3CBBC05B4FA99AUL, 0xBAFC062E06D1F209UL, 0xBAC2AEC959E14C06UL, 0x3A8F28E0CC748EBEUL);
dvec2 dd_c(uint64_t hi, uint64_t lo) { return dvec2(uint64BitsToDouble(hi), uint64BitsToDouble(lo)); }

// ---------------------------------------------------------------------------------------------------------------
// ln x for a positive finite double x: x = m * 2^e with m in [1/sqrt2, sqrt2), ln m = 2 atanh(t),
// t = (m - 1) / (m + 1), |t| <= 0.1716, 22 terms of t^2k / (2k + 1) (the 23rd is below 2^-104 of the sum).
dvec2 dd_ln(double x) {
    int e;
    double m = frexp1_64(x, e);
    if (m > 1.4142135623730951lf) {
        m = m * 0.5lf;
        e = e + 1;
    }
    dvec2 t = dd_div(dd_two_sum(m, -1.0lf), dd_two_sum(m, 1.0lf));  // m - 1 and m + 1 exactly
    dvec2 t2 = dd_mul(t, t);
    dvec2 s = dd_c(DD_ATANH_HI[21], DD_ATANH_LO[21]);
    for (int k = 20; k >= 0; k--) s = dd_add(dd_mul(s, t2), dd_c(DD_ATANH_HI[k], DD_ATANH_LO[k]));
    dvec2 ln_m = dd_scale(dd_mul(t, s), 2.0lf);
    return dd_add(dd_mul_d(dd_c(DD_LN2_HI, DD_LN2_LO), double(e)), ln_m);
}

// e^z for |z| < 700: z = k ln2 + r, |r| <= ln2 / 2; e^(r / 256) by 12 Taylor terms, squared 8 times, times 2^k.
dvec2 dd_exp(dvec2 z) {
    double k = roundEven(z.x / uint64BitsToDouble(DD_LN2_HI));
    dvec2 r = dd_add(z, dd_neg(dd_mul_d(dd_c(DD_LN2_HI, DD_LN2_LO), k)));
    r = dd_scale(r, 0.00390625lf);
    dvec2 s = dd_c(DD_INVFACT_HI[11], DD_INVFACT_LO[11]);
    for (int n = 10; n >= 0; n--) s = dd_add(dd_mul(s, r), dd_c(DD_INVFACT_HI[n], DD_INVFACT_LO[n]));
    for (int i = 0; i < 8; i++) s = dd_mul(s, s);
    int ki = int(k);
    int h = ki / 2;  // two steps keep each power of two in range
    return dd_scale(dd_scale(s, pow2_64(h)), pow2_64(ki - h));
}

bool transcend_input_ok(float x) { return !sign_bit32(x) && is_finite32(x) && !is_zero32(x); }

// util::mathf_pow(x, y) = (float)Math.Pow((double)x, (double)y) for x > 0. Mathf.Pow(x, 2) is mathf_pow2 (exact).
float mathf_pow(float x, float y) {
    if (!transcend_input_ok(x) || !is_finite32(y)) {
        transcend_fault = true;
        return 0.0;
    }
    if (y == 0.0 || x == 1.0) return 1.0;
    dvec2 z = dd_mul_d(dd_ln(double(x)), double(y));
    if (abs(z.x) > 700.0lf) {
        transcend_fault = true;
        return 0.0;
    }
    return float(dd_exp(z).x);
}

// util::mathf_log10(x) = (float)Math.Log10((double)x) for x > 0.
float mathf_log10(float x) {
    if (!transcend_input_ok(x)) {
        transcend_fault = true;
        return 0.0;
    }
    return float(dd_mul(dd_ln(double(x)), dd_c(DD_INV_LN10_HI, DD_INV_LN10_LO)).x);
}

#endif
