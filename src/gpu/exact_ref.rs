//! CPU transliteration of the division and square-root code in shaders/lib/exact.glsl, tested against Rust's `/`
//! and `sqrt` over the full range (subnormals, specials, near-halfway cases), with and without a fused fma, and with
//! "hardware" guesses that are off by a few ulps or are garbage. The GPU self-test checks whatever path a device
//! takes; this checks every path regardless of device. Keep in sync with exact.glsl.

#![cfg(test)]

// ------------------------------------------------------------------------------------------------- bit helpers

fn next_up32(x: f32) -> f32 {
    let u = x.to_bits();
    if u & 0x7fff_ffff == 0 { f32::from_bits(1) } else if u >> 31 == 0 { f32::from_bits(u + 1) } else { f32::from_bits(u - 1) }
}
fn next_down32(x: f32) -> f32 {
    f32::from_bits(next_up32(f32::from_bits(x.to_bits() ^ 0x8000_0000)).to_bits() ^ 0x8000_0000)
}
fn next_up64(x: f64) -> f64 {
    let u = x.to_bits();
    if u & 0x7fff_ffff_ffff_ffff == 0 { f64::from_bits(1) } else if u >> 63 == 0 { f64::from_bits(u + 1) } else { f64::from_bits(u - 1) }
}
fn next_down64(x: f64) -> f64 {
    f64::from_bits(next_up64(f64::from_bits(x.to_bits() ^ (1 << 63))).to_bits() ^ (1 << 63))
}
fn exp32(x: f32) -> u32 {
    (x.to_bits() >> 23) & 0xff
}
fn exp64(x: f64) -> u32 {
    ((x.to_bits() >> 52) & 0x7ff) as u32
}
fn sign32(x: f32) -> bool {
    x.to_bits() >> 31 != 0
}
fn sign64(x: f64) -> bool {
    x.to_bits() >> 63 != 0
}
fn zero32(x: f32) -> bool {
    x.to_bits() & 0x7fff_ffff == 0
}
fn zero64(x: f64) -> bool {
    x.to_bits() & 0x7fff_ffff_ffff_ffff == 0
}
fn signed32(neg: bool, m: u32) -> f32 {
    f32::from_bits(m | if neg { 0x8000_0000 } else { 0 })
}
fn signed64(neg: bool, m: u64) -> f64 {
    f64::from_bits(m | if neg { 1 << 63 } else { 0 })
}

fn div_special32(a: f32, b: f32) -> f32 {
    let neg = sign32(a) != sign32(b);
    if a.is_nan() || b.is_nan() || (zero32(a) && zero32(b)) || (!a.is_finite() && !b.is_finite()) {
        return f32::NAN;
    }
    if !a.is_finite() || zero32(b) { signed32(neg, 0x7f80_0000) } else { signed32(neg, 0) }
}
fn div_special64(a: f64, b: f64) -> f64 {
    let neg = sign64(a) != sign64(b);
    if a.is_nan() || b.is_nan() || (zero64(a) && zero64(b)) || (!a.is_finite() && !b.is_finite()) {
        return f64::NAN;
    }
    if !a.is_finite() || zero64(b) { signed64(neg, 0x7ff0_0000_0000_0000) } else { signed64(neg, 0) }
}
fn sqrt_special32(x: f32) -> f32 {
    if x.is_nan() || zero32(x) || !sign32(x) { x } else { f32::NAN }
}
fn sqrt_special64(x: f64) -> f64 {
    if x.is_nan() || zero64(x) || !sign64(x) { x } else { f64::NAN }
}
fn near32(q: f32, g: f32) -> bool {
    if !q.is_finite() || sign32(q) != sign32(g) {
        return false;
    }
    q.to_bits().abs_diff(g.to_bits()) <= 8
}

/// The device: whether fma() is fused, and its (inexact) `/` and `sqrt`.
struct Dev {
    fused32: bool,
    fused64: bool,
    hw_div32: fn(f32, f32) -> f32,
    hw_div64: fn(f64, f64) -> f64,
    hw_sqrt64: fn(f64) -> f64,
}

const STEPS: usize = 12;

// ------------------------------------------------------------------------------------------------- f32 division

fn div_rn32_fma(a: f32, b: f32, q: &mut f32) -> bool {
    for _ in 0..STEPS {
        let (qu, qd) = (next_up32(*q), next_down32(*q));
        let r = (-*q).mul_add(b, a).abs();
        let ru = (-qu).mul_add(b, a).abs();
        let rd = (-qd).mul_add(b, a).abs();
        if ru < r {
            *q = qu;
        } else if rd < r {
            *q = qd;
        } else {
            return true;
        }
    }
    false
}

fn div_rn32_f64(a: f32, b: f32, q: &mut f32) -> bool {
    let (da, db) = (a as f64, b as f64);
    for _ in 0..STEPS {
        let (qu, qd) = (next_up32(*q), next_down32(*q));
        let r = (da - *q as f64 * db).abs();
        let ru = (da - qu as f64 * db).abs();
        let rd = (da - qd as f64 * db).abs();
        if ru < r || (ru == r && qu.to_bits() & 1 == 0) {
            *q = qu;
        } else if rd < r || (rd == r && qd.to_bits() & 1 == 0) {
            *q = qd;
        } else {
            return true;
        }
    }
    false
}

fn div_rn32_from(dev: &Dev, a: f32, b: f32, mut q: f32) -> f32 {
    if !a.is_finite() || !b.is_finite() || zero32(a) || zero32(b) {
        return div_special32(a, b);
    }
    let neg = sign32(a) != sign32(b);
    let (ea, eb, eq) = (exp32(a), exp32(b), exp32(q));
    if dev.fused32 && (51..=253).contains(&ea) && (1..=253).contains(&eb) && (1..=253).contains(&eq) {
        let mut r = q;
        if div_rn32_fma(a, b, &mut r) {
            return r;
        }
    }
    let g = div_rn64_from(dev, a as f64, b as f64, (dev.hw_div64)(a as f64, b as f64)) as f32;
    if !g.is_finite() {
        return g;
    }
    if !near32(q, g) {
        q = g;
    }
    div_rn32_f64(a, b, &mut q);
    if zero32(q) { signed32(neg, 0) } else { q }
}

// ------------------------------------------------------------------------------------------------- f64 division

fn split64(x: f64) -> (f64, f64) {
    let c = 134217729.0 * x;
    let hi = c - (c - x);
    (hi, x - hi)
}

fn residual64(dev: &Dev, a: f64, q: f64, b: f64) -> f64 {
    if dev.fused64 {
        return (-q).mul_add(b, a);
    }
    let p = q * b;
    let (qh, ql) = split64(q);
    let (bh, bl) = split64(b);
    let e = ((qh * bh - p) + qh * bl + ql * bh) + ql * bl;
    (a - p) - e
}

fn div_rn64_refine(dev: &Dev, a: f64, b: f64, q: &mut f64) -> bool {
    for _ in 0..STEPS {
        let (qu, qd) = (next_up64(*q), next_down64(*q));
        let r = residual64(dev, a, *q, b).abs();
        let ru = residual64(dev, a, qu, b).abs();
        let rd = residual64(dev, a, qd, b).abs();
        if ru < r {
            *q = qu;
        } else if rd < r {
            *q = qd;
        } else {
            return true;
        }
    }
    false
}

fn div_rn64_core(dev: &Dev, a: f64, b: f64, mut q: f64) -> f64 {
    for _ in 0..4 {
        if div_rn64_refine(dev, a, b, &mut q) {
            return q;
        }
        q += (dev.hw_div64)(residual64(dev, a, q, b), b);
    }
    q
}

fn in_range64(x: f64) -> bool {
    (623..=1423).contains(&exp64(x))
}

fn frexp1_64(mut x: f64) -> (f64, i32) {
    let mut bias = 0;
    if exp64(x) == 0 {
        x *= 18446744073709551616.0;
        bias = 64;
    }
    let u = x.to_bits();
    let k = ((u >> 52) & 0x7ff) as i32 - 1023 - bias;
    (f64::from_bits((u & 0x800f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000), k)
}
fn pow2_64(k: i32) -> f64 {
    assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

fn div_rn64_subnormal(dev: &Dev, ma: f64, mb: f64, q: f64, k: i32) -> f64 {
    let m = 1074 + k;
    if m < -1 {
        return 0.0;
    }
    let f = pow2_64(-m);
    let mut n = (q * pow2_64(m)).round_ties_even();
    for _ in 0..STEPS {
        let r = residual64(dev, ma, n * f, mb).abs();
        let ru = residual64(dev, ma, (n + 1.0) * f, mb).abs();
        let rd = if n > 0.0 { residual64(dev, ma, (n - 1.0) * f, mb).abs() } else { r };
        let even_up = ((n + 1.0) as u64) & 1 == 0;
        if ru < r || (ru == r && even_up) {
            n += 1.0;
        } else if n > 0.0 && (rd < r || (rd == r && even_up)) {
            n -= 1.0;
        } else {
            break;
        }
    }
    f64::from_bits(n as u64)
}

fn div_rn64_scaled(dev: &Dev, a: f64, b: f64) -> f64 {
    let neg = sign64(a) != sign64(b);
    let (ma, ka) = frexp1_64(a);
    let (mb, kb) = frexp1_64(b);
    let (ma, mb) = (ma.abs(), mb.abs());
    let q = div_rn64_core(dev, ma, mb, (dev.hw_div64)(ma, mb));
    let k = ka - kb;
    let e = exp64(q) as i32 - 1023 + k;
    let r = if e > 1023 {
        f64::INFINITY
    } else if e >= -1022 {
        let h = k / 2;
        (q * pow2_64(h)) * pow2_64(k - h)
    } else {
        div_rn64_subnormal(dev, ma, mb, q, k)
    };
    if neg { -r } else { r }
}

fn div_rn64_from(dev: &Dev, a: f64, b: f64, q: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() || zero64(a) || zero64(b) {
        return div_special64(a, b);
    }
    if in_range64(a) && in_range64(b) && in_range64(q) && sign64(q) == (sign64(a) != sign64(b)) {
        return div_rn64_core(dev, a, b, q);
    }
    div_rn64_scaled(dev, a, b)
}

// ------------------------------------------------------------------------------------------------- square roots

fn sqrt_rn32_fma(x: f32, s: &mut f32) -> bool {
    for _ in 0..STEPS {
        let (su, sd) = (next_up32(*s), next_down32(*s));
        let r = (-*s).mul_add(*s, x);
        if r > *s * (su - *s) {
            *s = su;
        } else if r <= -(*s * (*s - sd)) {
            *s = sd;
        } else {
            return true;
        }
    }
    false
}

fn sqrt_rn32_f64(x: f32, s: &mut f32) -> bool {
    let dx = x as f64;
    for _ in 0..STEPS {
        let (su, sd) = (next_up32(*s), next_down32(*s));
        let hi = (*s as f64 + su as f64) * 0.5;
        let lo = (*s as f64 + sd as f64) * 0.5;
        if dx > hi * hi {
            *s = su;
        } else if dx < lo * lo {
            *s = sd;
        } else {
            return true;
        }
    }
    false
}

fn sqrt_rn32_from(dev: &Dev, x: f32, mut s: f32) -> f32 {
    if x.is_nan() || zero32(x) || sign32(x) || !x.is_finite() {
        return sqrt_special32(x);
    }
    if dev.fused32 && exp32(x) >= 49 && exp32(s) >= 1 && s.is_finite() && !sign32(s) {
        let mut r = s;
        if sqrt_rn32_fma(x, &mut r) {
            return r;
        }
    }
    let g = sqrt_rn64_from(dev, x as f64, (dev.hw_sqrt64)(x as f64)) as f32;
    if !near32(s, g) {
        s = g;
    }
    sqrt_rn32_f64(x, &mut s);
    s
}

fn sqrt_rn64_refine(dev: &Dev, x: f64, s: &mut f64) -> bool {
    for _ in 0..STEPS {
        let (su, sd) = (next_up64(*s), next_down64(*s));
        let r = residual64(dev, x, *s, *s);
        if r > *s * (su - *s) {
            *s = su;
        } else if r <= -(*s * (*s - sd)) {
            *s = sd;
        } else {
            return true;
        }
    }
    false
}

fn sqrt_rn64_core(dev: &Dev, x: f64, mut s: f64) -> f64 {
    for _ in 0..4 {
        if sqrt_rn64_refine(dev, x, &mut s) {
            return s;
        }
        s += (dev.hw_div64)(residual64(dev, x, s, s), 2.0 * s);
    }
    s
}

fn sqrt_rn64_from(dev: &Dev, x: f64, s: f64) -> f64 {
    if x.is_nan() || zero64(x) || sign64(x) || !x.is_finite() {
        return sqrt_special64(x);
    }
    if in_range64(x) && in_range64(s) && !sign64(s) {
        return sqrt_rn64_core(dev, x, s);
    }
    let (mut m, mut k) = frexp1_64(x);
    if k & 1 != 0 {
        m *= 2.0;
        k -= 1;
    }
    sqrt_rn64_core(dev, m, (dev.hw_sqrt64)(m)) * pow2_64(k / 2)
}

// ------------------------------------------------------------------------------------------------- tests

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn k(&mut self) -> i32 {
        if self.u64() % 8 == 0 { (self.u64() % 2001) as i32 - 1000 } else { (self.u64() % 7) as i32 - 3 }
    }
    /// Any finite f32 bit pattern, weighted toward the extremes and toward ordinary magnitudes.
    fn f32(&mut self) -> f32 {
        loop {
            let x = match self.u64() % 4 {
                0 => f32::from_bits(self.u64() as u32),
                1 => f32::from_bits((self.u64() as u32) & 0x807f_ffff),
                _ => f32::from_bits(((self.u64() as u32) & 0x807f_ffff) | ((self.u64() % 60 + 97) as u32) << 23),
            };
            if x.is_finite() {
                return x;
            }
        }
    }
    fn f64(&mut self) -> f64 {
        loop {
            let x = match self.u64() % 4 {
                0 => f64::from_bits(self.u64()),
                1 => f64::from_bits(self.u64() & 0x800f_ffff_ffff_ffff),
                _ => f64::from_bits((self.u64() & 0x800f_ffff_ffff_ffff) | (self.u64() % 200 + 923) << 52),
            };
            if x.is_finite() {
                return x;
            }
        }
    }
}

fn off32(q: f32, k: i32) -> f32 {
    if !q.is_finite() {
        return q;
    }
    let mut q = q;
    for _ in 0..k.unsigned_abs() {
        q = if k > 0 { next_up32(q) } else { next_down32(q) };
    }
    q
}
fn off64(q: f64, k: i32) -> f64 {
    if !q.is_finite() {
        return q;
    }
    let mut q = q;
    for _ in 0..k.unsigned_abs() {
        q = if k > 0 { next_up64(q) } else { next_down64(q) };
    }
    q
}

/// Deterministic "hardware" errors within what Vulkan allows (single precision: relative error up to ~2^-22)
/// for normal values, and garbage when an input or the result is subnormal (as AMD's f32 sqrt and / are).
fn mess64(x: f64, inputs: &[f64], salt: u64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let h = (x.to_bits() ^ salt).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    if x.is_subnormal() || inputs.iter().any(|v| v.is_subnormal()) {
        return if h >> 63 == 0 { x * 1.25 } else { 0.0 };
    }
    match h >> 60 {
        0..=5 => x * (1.0 + ((h >> 8) % 4096) as f64 * 2f64.powi(-34) - 2f64.powi(-23)),
        6..=11 => off64(x, ((h >> 8) % 5) as i32 - 2),
        _ => x,
    }
}

const DEVICES: [Dev; 3] = [
    Dev { fused32: true, fused64: true, hw_div32: |a, b| a / b, hw_div64: |a, b| a / b, hw_sqrt64: |x| x.sqrt() },
    Dev { fused32: false, fused64: false, hw_div32: |a, b| a / b, hw_div64: |a, b| a / b, hw_sqrt64: |x| x.sqrt() },
    Dev {
        fused32: true,
        fused64: false,
        hw_div32: |a, b| {
            let q = a / b;
            if q.is_finite() { f32::from_bits(q.to_bits() ^ (a.to_bits() & 3)) } else { q }
        },
        hw_div64: |a, b| mess64(a / b, &[a, b], 1),
        hw_sqrt64: |x| mess64(x.sqrt(), &[x], 2),
    },
];

fn same32(x: f32, y: f32) -> bool {
    (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits()
}
fn same64(x: f64, y: f64) -> bool {
    (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits()
}

const N: usize = 300_000;

const EDGE32: [f32; 10] = [0.0, -0.0, 1.0, -3.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::MAX, f32::MIN_POSITIVE, 1e-45];
const EDGE64: [f64; 10] = [0.0, -0.0, 1.0, -3.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, f64::MAX, f64::MIN_POSITIVE, 5e-324];

#[test]
fn div32_every_path() {
    for (di, dev) in DEVICES.iter().enumerate() {
        let mut r = Rng(10 + di as u64);
        let check = |a: f32, b: f32, q: f32| {
            let got = div_rn32_from(dev, a, b, q);
            assert!(same32(got, a / b), "device {di}: {a:e} / {b:e} from {q:e}: got {got:e}, want {:e}", a / b);
        };
        for a in EDGE32 {
            for b in EDGE32 {
                check(a, b, (dev.hw_div32)(a, b));
            }
        }
        for i in 0..N {
            let b = r.f32();
            let a = if i % 3 == 0 {
                // Near halfway, in the normal or the subnormal range.
                let q = r.f32().abs();
                let half = 0.5 * (next_up32(q) as f64 - q as f64);
                let a = ((q as f64 + half) * b as f64) as f32;
                [a, next_up32(a), next_down32(a)][(r.u64() % 3) as usize]
            } else {
                r.f32()
            };
            let guess = if i % 2 == 0 { (dev.hw_div32)(a, b) } else { off32(a / b, r.k()) };
            check(a, b, guess);
        }
    }
}

#[test]
fn sqrt32_every_path() {
    for (di, dev) in DEVICES.iter().enumerate() {
        let mut r = Rng(20 + di as u64);
        for x in EDGE32 {
            let got = sqrt_rn32_from(dev, x, x.sqrt());
            assert!(same32(got, x.sqrt()), "device {di}: sqrt {x:e}");
        }
        for i in 0..N {
            let x = if i % 3 == 0 {
                let s = r.f32().abs();
                let m = s as f64 + 0.5 * (next_up32(s) as f64 - s as f64);
                let x = (m * m) as f32;
                [x, next_up32(x), next_down32(x)][(r.u64() % 3) as usize]
            } else {
                r.f32().abs()
            };
            let guess = if i % 2 == 0 { (dev.hw_sqrt64)(x as f64) as f32 } else { off32(x.sqrt(), r.k()) };
            let got = sqrt_rn32_from(dev, x, guess);
            assert!(same32(got, x.sqrt()), "device {di}: sqrt {x:e} from {guess:e}: got {got:e}");
        }
    }
}

#[test]
fn div64_every_path() {
    for (di, dev) in DEVICES.iter().enumerate() {
        let mut r = Rng(30 + di as u64);
        let check = |a: f64, b: f64, q: f64| {
            let got = div_rn64_from(dev, a, b, q);
            assert!(same64(got, a / b), "device {di}: {a:e} / {b:e} from {q:e}: got {got:e}, want {:e}", a / b);
        };
        for a in EDGE64 {
            for b in EDGE64 {
                check(a, b, (dev.hw_div64)(a, b));
            }
        }
        for i in 0..N {
            let b = r.f64();
            let a = if i % 3 == 0 {
                let q = r.f64().abs();
                let u = next_up64(q) - q;
                let a = q.mul_add(b, b * u * 0.5);
                if !a.is_finite() {
                    continue;
                }
                [a, next_up64(a), next_down64(a)][(r.u64() % 3) as usize]
            } else if i % 3 == 1 {
                // Quotients in and around the subnormal range, with exact ties.
                let n = (r.u64() % (1 << 53)) as f64 + if r.u64() % 4 == 0 { 0.5 } else { 0.0 };
                let b = f64::from_bits((r.u64() & 0x000f_ffff_ffff_ffff) | ((r.u64() % 40 + 1003) << 52));
                let a = n * 5e-324 * b;
                if !(a.is_finite() && a != 0.0) {
                    continue;
                }
                check(a, b, (dev.hw_div64)(a, b));
                continue;
            } else {
                r.f64()
            };
            let guess = if i % 2 == 0 { (dev.hw_div64)(a, b) } else { off64(a / b, r.k()) };
            check(a, b, guess);
        }
    }
}

#[test]
fn sqrt64_every_path() {
    for (di, dev) in DEVICES.iter().enumerate() {
        let mut r = Rng(40 + di as u64);
        for x in EDGE64 {
            let got = sqrt_rn64_from(dev, x, x.sqrt());
            assert!(same64(got, x.sqrt()), "device {di}: sqrt {x:e}");
        }
        for i in 0..N {
            let x = if i % 3 == 0 {
                let s = r.f64().abs();
                let u = next_up64(s) - s;
                let x = s.mul_add(s, s * u);
                if !x.is_finite() || x == 0.0 {
                    continue;
                }
                [x, next_up64(x), next_down64(x)][(r.u64() % 3) as usize]
            } else {
                r.f64().abs()
            };
            let guess = if i % 2 == 0 { (dev.hw_sqrt64)(x) } else { off64(x.sqrt(), r.k()) };
            let got = sqrt_rn64_from(dev, x, guess);
            assert!(same64(got, x.sqrt()), "device {di}: sqrt {x:e} from {guess:e}: got {got:e}");
        }
    }
}
