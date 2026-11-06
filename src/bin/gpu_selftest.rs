//! Checks whether this GPU reproduces the CPU sim's arithmetic bit for bit.
//!
//! Runs every primitive the Vulkan port relies on — the `sim::util` helpers, correctly rounded `/` and `sqrt`,
//! f64 expressions copied from the sim, conversions, fused multiply-add — and the `System.Random` port, on random,
//! edge-case and near-halfway inputs, and compares each GPU result with the Rust code the CPU sim runs.
//!
//! ```sh
//! cargo run --release --features gpu --bin gpu_selftest            # default device (first discrete GPU)
//! cargo run --release --features gpu --bin gpu_selftest -- --list  # list devices; pick one with UMA_GPU=<index|name>
//! cargo run --release --features gpu --bin gpu_selftest -- --scale 8 --seed 7
//! ```
//!
//! Exit status 0 when every required check passes. Cases whose inputs or results are subnormal are reported
//! separately: they only matter if the sim produces subnormals, and a device that flushes them fails only those.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use uma_race_sim::gpu::{Gpu, as_bytes, shaders};
use uma_race_sim::rng::RaceRandom;
use uma_race_sim::sim::util;

// ------------------------------------------------------------------------------------------------- test cases

/// One case; layout matches `Case` in shaders/selftest_ops.comp (f32 values in the low 32 bits).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Case {
    op: u32,
    pad: u32,
    a: u64,
    b: u64,
    c: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ty {
    F32,
    F64,
    I32,
    Bool,
    Unused,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    /// Must match for the GPU port to be exact.
    Required,
    /// The hardware's own operation, shown for information.
    Info,
}

struct OpDef {
    code: u32,
    name: &'static str,
    class: Class,
    ins: [Ty; 3],
    out: Ty,
    cpu: fn(&Case) -> u64,
    what: &'static str,
}

fn f(x: u64) -> f32 {
    f32::from_bits(x as u32)
}
fn d(x: u64) -> f64 {
    f64::from_bits(x)
}
fn of(x: f32) -> u64 {
    x.to_bits() as u64
}
fn od(x: f64) -> u64 {
    x.to_bits()
}
fn oi(x: i32) -> u64 {
    x as u32 as u64
}

use Class::*;
use Ty::*;

#[rustfmt::skip]
const OPS: &[OpDef] = &[
    OpDef { code: 1, name: "add32", class: Required, ins: [F32, F32, Unused], out: F32, cpu: |t| of(f(t.a) + f(t.b)), what: "f32 a + b" },
    OpDef { code: 2, name: "sub32", class: Required, ins: [F32, F32, Unused], out: F32, cpu: |t| of(f(t.a) - f(t.b)), what: "f32 a - b" },
    OpDef { code: 3, name: "mul32", class: Required, ins: [F32, F32, Unused], out: F32, cpu: |t| of(f(t.a) * f(t.b)), what: "f32 a * b" },
    OpDef { code: 4, name: "div32", class: Required, ins: [F32, F32, Unused], out: F32, cpu: |t| of(f(t.a) / f(t.b)), what: "div_rn32" },
    OpDef { code: 5, name: "sqrt32", class: Required, ins: [F32, Unused, Unused], out: F32, cpu: |t| of(f(t.a).sqrt()), what: "sqrt_rn32" },
    OpDef { code: 6, name: "muladd32", class: Required, ins: [F32, F32, F32], out: F32, cpu: |t| of(f(t.a) * f(t.b) + f(t.c)), what: "a * b + c not fused" },
    OpDef { code: 7, name: "fma32", class: Info, ins: [F32, F32, F32], out: F32, cpu: |t| of(f(t.a).mul_add(f(t.b), f(t.c))), what: "fma() fused" },
    OpDef { code: 8, name: "div32_refine", class: Required, ins: [F32, F32, I32], out: F32, cpu: |t| of(f(t.a) / f(t.b)), what: "div_rn32 from guesses up to 1000 ulps off" },
    OpDef { code: 9, name: "sqrt32_refine", class: Required, ins: [F32, Unused, I32], out: F32, cpu: |t| of(f(t.a).sqrt()), what: "sqrt_rn32 from guesses up to 1000 ulps off" },
    OpDef { code: 10, name: "f64_to_f32", class: Required, ins: [F64, Unused, Unused], out: F32, cpu: |t| of(d(t.a) as f32), what: "f64 -> f32 rounding" },
    OpDef { code: 11, name: "f32_to_i32", class: Required, ins: [F32, Unused, Unused], out: I32, cpu: |t| oi(util::f32_to_i32(f(t.a))), what: "util::f32_to_i32" },
    OpDef { code: 12, name: "f64_to_i32", class: Required, ins: [F64, Unused, Unused], out: I32, cpu: |t| oi(util::f64_to_i32(d(t.a))), what: "util::f64_to_i32" },
    OpDef { code: 13, name: "floor_to_int", class: Required, ins: [F32, Unused, Unused], out: I32, cpu: |t| oi(util::floor_to_int(f(t.a))), what: "util::floor_to_int" },
    OpDef { code: 14, name: "i32_to_f32", class: Required, ins: [I32, Unused, Unused], out: F32, cpu: |t| of(t.a as u32 as i32 as f32), what: "i32 -> f32 rounding" },
    OpDef { code: 15, name: "min", class: Required, ins: [F32, F32, Unused], out: F32, cpu: |t| of(util::min(f(t.a), f(t.b))), what: "util::min" },
    OpDef { code: 16, name: "max", class: Required, ins: [F32, F32, Unused], out: F32, cpu: |t| of(util::max(f(t.a), f(t.b))), what: "util::max" },
    OpDef { code: 17, name: "clamp", class: Required, ins: [F32, F32, F32], out: F32, cpu: |t| of(util::clamp(f(t.a), f(t.b), f(t.c))), what: "util::clamp" },
    OpDef { code: 18, name: "clamp01", class: Required, ins: [F32, Unused, Unused], out: F32, cpu: |t| of(util::clamp01(f(t.a))), what: "util::clamp01" },
    OpDef { code: 19, name: "lerp", class: Required, ins: [F32, F32, F32], out: F32, cpu: |t| of(util::lerp(f(t.a), f(t.b), f(t.c))), what: "util::lerp" },
    OpDef { code: 20, name: "approximately", class: Required, ins: [F32, F32, Unused], out: Bool, cpu: |t| util::approximately(f(t.a), f(t.b)) as u64, what: "util::approximately" },
    OpDef { code: 21, name: "neg32", class: Required, ins: [F32, Unused, Unused], out: F32, cpu: |t| of(-f(t.a)), what: "f32 -a" },
    OpDef { code: 22, name: "abs32", class: Required, ins: [F32, Unused, Unused], out: F32, cpu: |t| of(f(t.a).abs()), what: "f32 abs" },
    OpDef { code: 40, name: "add64", class: Required, ins: [F64, F64, Unused], out: F64, cpu: |t| od(d(t.a) + d(t.b)), what: "f64 a + b" },
    OpDef { code: 41, name: "sub64", class: Required, ins: [F64, F64, Unused], out: F64, cpu: |t| od(d(t.a) - d(t.b)), what: "f64 a - b" },
    OpDef { code: 42, name: "mul64", class: Required, ins: [F64, F64, Unused], out: F64, cpu: |t| od(d(t.a) * d(t.b)), what: "f64 a * b" },
    OpDef { code: 43, name: "div64", class: Required, ins: [F64, F64, Unused], out: F64, cpu: |t| od(d(t.a) / d(t.b)), what: "div_rn64" },
    OpDef { code: 44, name: "sqrt64", class: Required, ins: [F64, Unused, Unused], out: F64, cpu: |t| od(d(t.a).sqrt()), what: "sqrt_rn64" },
    OpDef { code: 45, name: "muladd64", class: Required, ins: [F64, F64, F64], out: F64, cpu: |t| od(d(t.a) * d(t.b) + d(t.c)), what: "a * b + c not fused" },
    OpDef { code: 46, name: "fma64", class: Info, ins: [F64, F64, F64], out: F64, cpu: |t| od(d(t.a).mul_add(d(t.b), d(t.c))), what: "fma() fused" },
    OpDef { code: 47, name: "div64_refine", class: Required, ins: [F64, F64, I32], out: F64, cpu: |t| od(d(t.a) / d(t.b)), what: "div_rn64 from guesses up to 1000 ulps off" },
    OpDef { code: 48, name: "sqrt64_refine", class: Required, ins: [F64, Unused, I32], out: F64, cpu: |t| od(d(t.a).sqrt()), what: "sqrt_rn64 from guesses up to 1000 ulps off" },
    OpDef { code: 49, name: "floor64", class: Required, ins: [F64, Unused, Unused], out: F64, cpu: |t| od(d(t.a).floor()), what: "f64 floor" },
    OpDef { code: 50, name: "f32_to_f64", class: Required, ins: [F32, Unused, Unused], out: F64, cpu: |t| od(f(t.a) as f64), what: "f32 -> f64" },
    OpDef { code: 70, name: "expr_vision", class: Required, ins: [F32, F32, Unused], out: F32, cpu: expr_vision, what: "ai/around.rs is_visible" },
    OpDef { code: 71, name: "expr_front_block", class: Required, ins: [F32, F32, F32], out: F32, cpu: expr_front_block, what: "ai/around.rs check_front_block" },
    OpDef { code: 72, name: "expr_displacement", class: Required, ins: [F32, F32, Unused], out: F32, cpu: expr_displacement, what: "tick.rs world displacement" },
    OpDef { code: 73, name: "expr_pow2", class: Required, ins: [F32, Unused, Unused], out: F32, cpu: |t| of(util::mathf_pow(f(t.a), 2.0)), what: "hp.rs Mathf.Pow(x, 2) as x * x" },
    OpDef { code: 74, name: "expr_path_lerp", class: Required, ins: [F32, F32, F32], out: F32, cpu: expr_path_lerp, what: "course/lanes.rs sample" },
    OpDef { code: 75, name: "expr_lane_move", class: Required, ins: [F32, F32, F32], out: F32, cpu: expr_lane_move, what: "ai/lane.rs lane move" },
    OpDef { code: 76, name: "expr_disp_cap", class: Required, ins: [F32, F32, F32], out: F32, cpu: expr_disp_cap, what: "tick.rs displacement cap" },
    OpDef { code: 100, name: "hw_div32", class: Info, ins: [F32, F32, Unused], out: F32, cpu: |t| of(f(t.a) / f(t.b)), what: "hardware f32 /" },
    OpDef { code: 101, name: "hw_sqrt32", class: Info, ins: [F32, Unused, Unused], out: F32, cpu: |t| of(f(t.a).sqrt()), what: "hardware f32 sqrt" },
    OpDef { code: 102, name: "hw_div64", class: Info, ins: [F64, F64, Unused], out: F64, cpu: |t| od(d(t.a) / d(t.b)), what: "hardware f64 /" },
    OpDef { code: 103, name: "hw_sqrt64", class: Info, ins: [F64, Unused, Unused], out: F64, cpu: |t| od(d(t.a).sqrt()), what: "hardware f64 sqrt" },
];

// The sim's expressions, verbatim.
fn expr_vision(t: &Case) -> u64 {
    let (distance_gap, v) = (f(t.a), f(t.b));
    let w = ((distance_gap as f64 / v as f64) * 0.638888896f32 as f64 + 0.111111112f32 as f64) * 0.5;
    of(w as f32)
}
fn expr_front_block(t: &Case) -> u64 {
    let (distance_gap, fbdg, fblg) = (f(t.a), f(t.b), f(t.c));
    of(((1.0 - (distance_gap as f64 / fbdg as f64) * 0.6f32 as f64) * fblg as f64) as f32)
}
fn expr_displacement(t: &Case) -> u64 {
    let (dx, dz) = (f(t.a), f(t.b));
    of(((dx as f64 * dx as f64 + 0.0) + dz as f64 * dz as f64).sqrt() as f32)
}
fn expr_path_lerp(t: &Case) -> u64 {
    let (a, b, frac) = (f(t.a), f(t.b), f(t.c));
    of(((b as f64 - a as f64) * frac as f64 + a as f64) as f32)
}
fn expr_lane_move(t: &Case) -> u64 {
    let (dt, lane_move_speed, prev) = (f(t.a), f(t.b), f(t.c));
    of((dt as f64 * lane_move_speed as f64 + prev as f64) as f32)
}
fn expr_disp_cap(t: &Case) -> u64 {
    let (moved, lim, world_disp) = (f(t.a), f(t.b), f(t.c));
    of((moved as f64 * (lim as f64 / world_disp as f64)) as f32)
}

// ------------------------------------------------------------------------------------------------- generators

struct Rng(u64);

impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.u64() % n
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
    fn chance(&mut self, p: f64) -> bool {
        (self.u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64) < p
    }
    /// How far (in ulps) the refinement tests move the hardware guess: usually a few, sometimes far, so the
    /// fallbacks for a guess that does not settle run too.
    fn guess_offset(&mut self) -> i32 {
        if self.chance(0.15) {
            let k = self.range(50, 1000) as i32;
            if self.chance(0.5) { -k } else { k }
        } else {
            self.range(-3, 3) as i32
        }
    }
    fn unit(&mut self) -> f32 {
        (self.u64() >> 40) as f32 * (1.0 / (1u32 << 24) as f32)
    }
    fn f32_parts(&mut self, exp_lo: i64, exp_hi: i64) -> f32 {
        let sign = (self.u64() & 1) as u32;
        let e = (self.range(exp_lo, exp_hi) + 127) as u32;
        f32::from_bits((sign << 31) | (e << 23) | (self.u64() as u32 & 0x7f_ffff))
    }
    /// Magnitudes 2^-24 .. 2^17, the sim's working range.
    fn sim32(&mut self) -> f32 {
        self.f32_parts(-24, 16)
    }
    fn wide32(&mut self) -> f32 {
        self.f32_parts(-126, 127)
    }
    fn sub32(&mut self) -> f32 {
        f32::from_bits(((self.u64() & 1) as u32) << 31 | (self.u64() as u32 & 0x7f_ffff).max(1))
    }
    fn edge32(&mut self) -> f32 {
        EDGE32[self.below(EDGE32.len() as u64) as usize]
    }
    fn any32(&mut self) -> f32 {
        match self.below(100) {
            0..70 => self.sim32(),
            70..88 => self.wide32(),
            88..94 => self.sub32(),
            _ => self.edge32(),
        }
    }
    fn f64_parts(&mut self, exp_lo: i64, exp_hi: i64) -> f64 {
        let sign = self.u64() & 1;
        let e = (self.range(exp_lo, exp_hi) + 1023) as u64;
        f64::from_bits((sign << 63) | (e << 52) | (self.u64() & 0xf_ffff_ffff_ffff))
    }
    /// Half f32-derived, half full 53-bit values, in the sim's range.
    fn sim64(&mut self) -> f64 {
        if self.chance(0.5) { self.sim32() as f64 } else { self.f64_parts(-30, 20) }
    }
    fn any64(&mut self) -> f64 {
        match self.below(100) {
            0..70 => self.sim64(),
            70..90 => self.f64_parts(-1000, 1000),
            90..94 => f64::from_bits((self.u64() & 0x800f_ffff_ffff_ffff).max(1)),
            _ => EDGE64[self.below(EDGE64.len() as u64) as usize],
        }
    }
    fn jitter32(&mut self, x: f32) -> f32 {
        match self.below(3) {
            0 => x,
            1 => next_up32(x),
            _ => next_down32(x),
        }
    }
    fn jitter64(&mut self, x: f64) -> f64 {
        match self.below(3) {
            0 => x,
            1 => next_up64(x),
            _ => next_down64(x),
        }
    }
}

fn next_up32(x: f32) -> f32 {
    let u = x.to_bits();
    if u & 0x7fff_ffff == 0 { f32::from_bits(1) } else if u >> 31 == 0 { f32::from_bits(u + 1) } else { f32::from_bits(u - 1) }
}
fn next_down32(x: f32) -> f32 {
    -next_up32(-x)
}
fn next_up64(x: f64) -> f64 {
    let u = x.to_bits();
    if u & 0x7fff_ffff_ffff_ffff == 0 { f64::from_bits(1) } else if u >> 63 == 0 { f64::from_bits(u + 1) } else { f64::from_bits(u - 1) }
}
fn next_down64(x: f64) -> f64 {
    -next_up64(-x)
}
fn ulp32(x: f32) -> f64 {
    let x = x.abs();
    next_up32(x) as f64 - x as f64
}
fn ulp64(x: f64) -> f64 {
    let x = x.abs();
    next_up64(x) - x
}

const EDGE32: &[f32] = &[
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 3.0, 1.5, 2.5, -2.5, 1e-6, 0.0666, 19.5, 30.0, 1e30, 1e-30,
    0.49999997, 16777216.0, 16777218.0, 2147483520.0, 2147483648.0, -2147483648.0, -2147483904.0,
    f32::MIN_POSITIVE, f32::MAX, -f32::MAX, f32::INFINITY, f32::NEG_INFINITY, f32::NAN,
    f32::from_bits(1), f32::from_bits(8), f32::from_bits(0x8000_0008), f32::from_bits(0x007f_ffff), f32::from_bits(0xffc0_0000),
];
const EDGE64: &[f64] = &[
    0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 3.0, 0.1, 1.5, 2.5, -2.5, 0.49999999999999994, 1e300, 1e-300,
    9007199254740992.0, 2147483647.0, 2147483647.5, 2147483648.0, -2147483648.0, -2147483648.5, -2147483649.0,
    f64::MIN_POSITIVE, f64::MAX, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, f64::from_bits(1),
];

fn case(op: u32, a: u64, b: u64, c: u64) -> Case {
    Case { op, pad: 0, a, b, c }
}

/// Every case for `op`, `n` = the base random count.
fn generate(op: &OpDef, r: &mut Rng, n: usize, out: &mut Vec<Case>) {
    let code = op.code;
    let push3 = |out: &mut Vec<Case>, a: f32, b: f32, c: f32| out.push(case(code, of(a), of(b), of(c)));
    let push3d = |out: &mut Vec<Case>, a: f64, b: f64, c: f64| out.push(case(code, od(a), od(b), od(c)));
    match code {
        // Binary f32 ops: all edge pairs, then random pairs.
        1 | 2 | 3 | 4 | 15 | 16 | 20 | 100 => {
            if code != 100 {
                for &a in EDGE32 {
                    for &b in EDGE32 {
                        push3(out, a, b, 0.0);
                    }
                }
            }
            for _ in 0..n {
                let a = r.any32();
                let b = match code {
                    // Close pairs exercise approximately's threshold and cancellation in sub.
                    2 | 20 if r.chance(0.5) => {
                        let mut b = a;
                        for _ in 0..r.below(40) {
                            b = next_up32(b);
                        }
                        if r.chance(0.5) { -b } else { b }
                    }
                    _ => r.any32(),
                };
                push3(out, a, b, 0.0);
            }
            if code == 4 || code == 100 {
                // Quotients just either side of halfway between two floats.
                for _ in 0..2 * n {
                    let (q, b) = (r.sim32().abs(), r.sim32());
                    let a = r.jitter32(((q as f64 + 0.5 * ulp32(q)) * b as f64) as f32);
                    push3(out, a, b, 0.0);
                }
            }
        }
        5 | 101 => {
            if code == 5 {
                for &a in EDGE32 {
                    push3(out, a, 0.0, 0.0);
                }
            }
            for _ in 0..n {
                push3(out, r.any32().abs(), 0.0, 0.0);
            }
            for _ in 0..2 * n {
                let s = r.sim32().abs();
                let m = s as f64 + 0.5 * ulp32(s);
                push3(out, r.jitter32((m * m) as f32), 0.0, 0.0);
            }
        }
        6 | 7 => {
            for _ in 0..n {
                let (a, b) = (r.any32(), r.any32());
                // c = -(a*b rounded): the fused and unfused results differ by the product's rounding error.
                let c = if r.chance(0.6) { -(a * b) } else { r.any32() };
                push3(out, a, b, c);
            }
        }
        8 | 9 => {
            let mut made = 0;
            while made < n {
                let (a, b) = if code == 8 {
                    if r.chance(0.5) {
                        (r.sim32(), r.sim32())
                    } else {
                        let (q, b) = (r.sim32().abs(), r.sim32());
                        (r.jitter32(((q as f64 + 0.5 * ulp32(q)) * b as f64) as f32), b)
                    }
                } else if r.chance(0.5) {
                    (r.sim32().abs(), 0.0)
                } else {
                    let s = r.sim32().abs();
                    let m = s as f64 + 0.5 * ulp32(s);
                    (r.jitter32((m * m) as f32), 0.0)
                };
                let exact = if code == 8 { a / b } else { a.sqrt() };
                if exact.is_normal() {
                    out.push(case(code, of(a), of(b), oi(r.guess_offset())));
                    made += 1;
                }
            }
        }
        10 => {
            for &x in EDGE64 {
                push3d(out, x, 0.0, 0.0);
            }
            for _ in 0..n {
                let x = match r.below(4) {
                    // Exactly and almost halfway between two f32 values (ties go to even).
                    0 | 1 => {
                        let v = r.any32();
                        let h = v as f64 + 0.5 * ulp32(v) * if v < 0.0 { -1.0 } else { 1.0 };
                        if r.chance(0.5) { h } else { r.jitter64(h) }
                    }
                    2 => r.sim64(),
                    _ => r.any64(),
                };
                push3d(out, x, 0.0, 0.0);
            }
        }
        11 | 13 | 18 | 21 | 22 | 50 | 73 => {
            for &a in EDGE32 {
                push3(out, a, 0.0, 0.0);
            }
            for _ in 0..n {
                let a = match (code, r.below(4)) {
                    (11 | 13, 0) => r.range(-2_147_483_904, 2_147_483_904) as f32,
                    (11 | 13, 1) => r.range(-64, 64) as f32 * 0.5,
                    (18, 0) => r.unit() * 1.5 - 0.25,
                    (73, _) => r.f32_parts(-12, 6),
                    _ => r.any32(),
                };
                push3(out, a, 0.0, 0.0);
            }
        }
        12 => {
            for &x in EDGE64 {
                push3d(out, x, 0.0, 0.0);
            }
            for _ in 0..n {
                let x = match r.below(3) {
                    0 => r.range(-2_147_483_650, 2_147_483_650) as f64 + if r.chance(0.5) { 0.5 } else { 0.0 },
                    1 => r.sim64(),
                    _ => r.any64(),
                };
                push3d(out, x, 0.0, 0.0);
            }
        }
        14 => {
            for v in [0i32, 1, -1, 16777216, 16777217, 16777219, i32::MAX, i32::MIN, i32::MAX - 64, -(1 << 30) - 65] {
                out.push(case(code, oi(v), 0, 0));
            }
            for _ in 0..n {
                out.push(case(code, r.u64() & 0xffff_ffff, 0, 0));
            }
        }
        17 | 19 => {
            for _ in 0..n {
                let (a, b, c) = if code == 17 {
                    let (x, y) = (r.any32(), r.any32());
                    let (lo, hi) = if r.chance(0.9) && x > y { (y, x) } else { (x, y) };
                    let v = if r.chance(0.1) { r.edge32() } else { r.any32() };
                    (v, lo, hi)
                } else {
                    let t = if r.chance(0.8) { r.unit() * 2.0 - 0.5 } else { r.any32() };
                    (r.any32(), r.any32(), t)
                };
                push3(out, a, b, c);
            }
        }
        40 | 41 | 42 | 43 | 102 => {
            if code != 102 {
                for &a in EDGE64 {
                    for &b in EDGE64 {
                        push3d(out, a, b, 0.0);
                    }
                }
            }
            for _ in 0..n {
                let a = r.any64();
                let b = if code == 41 && r.chance(0.3) { r.jitter64(a) } else { r.any64() };
                push3d(out, a, b, 0.0);
            }
            if code == 43 || code == 102 {
                for _ in 0..2 * n {
                    let (q, b) = (r.sim64().abs(), r.sim64());
                    push3d(out, r.jitter64(q.mul_add(b, b * ulp64(q) * 0.5)), b, 0.0);
                }
            }
        }
        44 | 103 => {
            if code == 44 {
                for &x in EDGE64 {
                    push3d(out, x, 0.0, 0.0);
                }
            }
            for _ in 0..n {
                push3d(out, r.any64().abs(), 0.0, 0.0);
            }
            for _ in 0..2 * n {
                let s = r.sim64().abs();
                push3d(out, r.jitter64(s.mul_add(s, s * ulp64(s))), 0.0, 0.0);
            }
        }
        45 | 46 => {
            for _ in 0..n {
                let (a, b) = (r.any64(), r.any64());
                let c = if r.chance(0.6) { -(a * b) } else { r.any64() };
                push3d(out, a, b, c);
            }
        }
        47 | 48 => {
            let mut made = 0;
            while made < n {
                let (a, b) = if code == 47 {
                    let (q, b) = (r.sim64().abs(), r.sim64());
                    (if r.chance(0.5) { r.sim64() } else { r.jitter64(q.mul_add(b, b * ulp64(q) * 0.5)) }, b)
                } else {
                    let s = r.sim64().abs();
                    (if r.chance(0.5) { r.sim64().abs() } else { r.jitter64(s.mul_add(s, s * ulp64(s))) }, 0.0)
                };
                let exact = if code == 47 { a / b } else { a.sqrt() };
                if exact.is_normal() {
                    out.push(case(code, od(a), od(b), oi(r.guess_offset())));
                    made += 1;
                }
            }
        }
        49 => {
            for &x in EDGE64 {
                push3d(out, x, 0.0, 0.0);
            }
            for _ in 0..n {
                let x = if r.chance(0.5) { r.range(-100_000, 100_000) as f64 * 0.25 } else { r.any64() };
                push3d(out, x, 0.0, 0.0);
            }
        }
        70 => {
            for _ in 0..4 * n {
                let v = r.unit() * 30.0 + 0.01;
                let g = if r.chance(0.05) { 0.0 } else { r.unit() * v };
                push3(out, g, v, 0.0);
            }
        }
        71 => {
            for _ in 0..4 * n {
                let fbdg = r.unit() * 4.0 + 0.5;
                push3(out, r.unit() * fbdg, fbdg, r.unit() * 0.3 + 0.001);
            }
        }
        72 => {
            for _ in 0..4 * n {
                let dx = if r.chance(0.05) { 0.0 } else { r.unit() * 6.0 - 3.0 };
                push3(out, dx, r.unit() * 6.0 - 3.0, 0.0);
            }
        }
        74 => {
            for _ in 0..4 * n {
                push3(out, r.unit() * 2000.0 - 1000.0, r.unit() * 2000.0 - 1000.0, r.unit());
            }
        }
        75 => {
            for _ in 0..4 * n {
                let dt = if r.chance(0.5) { 0.0666 } else { r.unit() * 0.1 };
                push3(out, dt, r.unit() * 0.02 - 0.01, r.unit() * 1.2);
            }
        }
        76 => {
            for _ in 0..4 * n {
                push3(out, r.unit() * 2.0, r.unit() * 2.0, r.unit() * 3.0 + 0.01);
            }
        }
        _ => unreachable!("no generator for op {code}"),
    }
}

fn is_subnormal(ty: Ty, bits: u64) -> bool {
    match ty {
        F32 => f(bits).is_subnormal(),
        F64 => d(bits).is_subnormal(),
        _ => false,
    }
}

fn same(ty: Ty, x: u64, y: u64) -> bool {
    match ty {
        F32 => (f(x).is_nan() && f(y).is_nan()) || x as u32 == y as u32,
        F64 => (d(x).is_nan() && d(y).is_nan()) || x == y,
        _ => x as u32 == y as u32,
    }
}

fn show(ty: Ty, bits: u64) -> String {
    match ty {
        F32 => format!("{:e} (0x{:08x})", f(bits), bits as u32),
        F64 => format!("{:e} (0x{:016x})", d(bits), bits),
        I32 => format!("{}", bits as u32 as i32),
        Bool => format!("{}", bits != 0),
        Unused => String::new(),
    }
}

// ------------------------------------------------------------------------------------------------- RNG script

/// The call script of shaders/selftest_rng.comp, on the CPU's RaceRandom.
fn rng_script(seed: i32, calls: u32, out: &mut Vec<u64>) -> u64 {
    let mut r = RaceRandom::new(seed);
    for j in 0..calls {
        let v = match j % 7 {
            0 => r.random.internal_sample() as u32 as u64,
            1 => r.random.next_double().to_bits(),
            2 => {
                let mn = (j * 7 % 21) as i32 - 10;
                let mx = mn + 1 + (j * 13 % 1000) as i32;
                r.random_int_range(mn, mx) as u32 as u64
            }
            3 => r.random_int(1 + (j * 31 % 100) as i32) as u32 as u64,
            4 => {
                let mn = ((j % 17) as i32 - 8) as f32 * 0.25;
                let mx = mn + (j % 11) as f32 * 0.5 + 0.125;
                r.random_float_range(mn, mx).to_bits() as u64
            }
            5 => r.random_float((j % 23) as f32 * 0.5 + 0.25).to_bits() as u64,
            _ => r.random.next_range(-2_000_000_000 + j as i32, 2_000_000_000) as u32 as u64,
        };
        out.push(v);
    }
    r.draws()
}

// ------------------------------------------------------------------------------------------------- main

struct Args {
    scale: usize,
    seed: u64,
    list: bool,
    no_fma: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args { scale: 1, seed: 1, list: false, no_fma: false };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        match x.as_str() {
            "--scale" => a.scale = it.next().and_then(|v| v.parse().ok()).ok_or("--scale N")?,
            "--seed" => a.seed = it.next().and_then(|v| v.parse().ok()).ok_or("--seed N")?,
            "--list" => a.list = true,
            "--no-fma" => a.no_fma = true,
            "-h" | "--help" => {
                return Err("usage: gpu_selftest [--list] [--scale N] [--seed N] [--no-fma]   (device: UMA_GPU=<index|name>)\n  \
                     --no-fma  test the paths used when fma() is not fused, even if it is".into());
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

fn yes(b: bool) -> &'static str {
    if b { "yes" } else { "NO" }
}

fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    if args.list {
        match Gpu::describe_all() {
            Ok(ds) => {
                for d in ds {
                    println!("{}: {} ({:?}, {} {}, Vulkan {})", d.index, d.name, d.device_type, d.driver_name, d.driver_info, d.api_version_string());
                }
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::from(2);
            }
        }
    }
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &Args) -> Result<bool, String> {
    let mut gpu = Gpu::new()?;
    let fused = (gpu.fma32_fused, gpu.fma64_fused);
    if args.no_fma {
        gpu.fma32_fused = false;
        gpu.fma64_fused = false;
    }
    let i = &gpu.info;
    let fc = i.float_controls;
    println!("Device          {} — {} {} — Vulkan {} — subgroup {}", i.name, i.driver_name, i.driver_info, i.api_version_string(), i.subgroup_size);
    println!(
        "Float controls  denormals preserved: f32 {}, f64 {} · signed zero/Inf/NaN preserved: f32 {}, f64 {} · round to even: f32 {}, f64 {}",
        yes(fc.denorm_preserve_32),
        yes(fc.denorm_preserve_64),
        yes(fc.signed_zero_inf_nan_preserve_32),
        yes(fc.signed_zero_inf_nan_preserve_64),
        yes(fc.rounding_rte_32),
        yes(fc.rounding_rte_64)
    );

    let ops_kernel = gpu.kernel(shaders::SELFTEST_OPS, 2, 4)?;
    let rng_kernel = gpu.kernel(shaders::SELFTEST_RNG, 3, 8)?;
    println!(
        "fma() fused     f32 {}, f64 {}{}",
        yes(fused.0),
        yes(fused.1),
        if args.no_fma {
            " (--no-fma: testing the exact f64 residual / Dekker paths)"
        } else if gpu.fma32_fused && gpu.fma64_fused {
            " (div_rn / sqrt_rn use fma residuals)"
        } else {
            " (where not fused, div_rn / sqrt_rn use exact f64 residuals / Dekker products)"
        }
    );
    println!(
        "SPIR-V patch    selftest_ops: {} float ops marked NoContraction, {} execution modes; selftest_rng: {}, {}",
        ops_kernel.patch.no_contraction_added,
        ops_kernel.patch.execution_modes_added,
        rng_kernel.patch.no_contraction_added,
        rng_kernel.patch.execution_modes_added
    );
    println!();

    // Cases.
    let n = (1usize << 15) * args.scale;
    let mut rng = Rng(args.seed);
    let mut cases = Vec::new();
    let mut spans = Vec::new();
    for op in OPS {
        let start = cases.len();
        generate(op, &mut rng, n, &mut cases);
        spans.push(start..cases.len());
    }

    // GPU, in chunks (keeps each dispatch short; Windows resets a GPU that is busy for ~2 s).
    let t0 = Instant::now();
    let mut gpu_time = Duration::ZERO;
    let mut results: Vec<[u32; 2]> = Vec::with_capacity(cases.len());
    let chunk = 1 << 20;
    for part in cases.chunks(chunk) {
        let input = gpu.upload(part)?;
        let output = gpu.device_buffer((part.len() * 8) as u64)?;
        let count = part.len() as u32;
        gpu_time += ops_kernel.dispatch(&[&input, &output], as_bytes(&[count]), count.div_ceil(64))?;
        results.extend(gpu.read::<[u32; 2]>(&output, part.len())?);
    }
    let wall = t0.elapsed();

    // Compare.
    let mut all_required_ok = true;
    let mut denormal_failures = 0usize;
    let mut info_lines = Vec::new();
    println!("{:<20} {:>10} {:>10}   {:>16}   {}", "check", "cases", "failures", "subnormal cases", "what");
    for (op, span) in OPS.iter().zip(&spans) {
        let (mut normal, mut normal_bad, mut sub, mut sub_bad) = (0usize, 0usize, 0usize, 0usize);
        let mut examples = Vec::new();
        for k in span.clone() {
            let t = &cases[k];
            let want = (op.cpu)(t);
            let got = results[k][0] as u64 | (results[k][1] as u64) << 32;
            let subnormal = [t.a, t.b, t.c].iter().zip(op.ins).any(|(&v, ty)| is_subnormal(ty, v)) || is_subnormal(op.out, want);
            let ok = same(op.out, want, got);
            if subnormal {
                sub += 1;
                sub_bad += !ok as usize;
            } else {
                normal += 1;
                normal_bad += !ok as usize;
            }
            if !ok && examples.len() < 3 && (!subnormal || op.class == Info) {
                let ins: Vec<String> = [t.a, t.b, t.c].iter().zip(op.ins).filter(|(_, ty)| *ty != Unused).map(|(&v, ty)| show(ty, v)).collect();
                examples.push(format!("    {}({}) = {}, GPU {}", op.name, ins.join(", "), show(op.out, want), show(op.out, got)));
            }
        }
        if op.class == Info {
            let pct = 100.0 * (normal_bad + sub_bad) as f64 / (normal + sub).max(1) as f64;
            let what = if op.code == 7 || op.code == 46 { "not fused" } else { "not correctly rounded" };
            info_lines.push(format!("  {:<12} {:>6.2}% of results {what} ({})", op.name, pct, op.what));
            continue;
        }
        all_required_ok &= normal_bad == 0;
        denormal_failures += sub_bad;
        let sub_col = if sub == 0 { "-".to_string() } else { format!("{} ({} bad)", thousands(sub), sub_bad) };
        let mark = if normal_bad == 0 { "" } else { "   <-- FAIL" };
        println!("{:<20} {:>10} {:>10}   {:>16}   {}{}", op.name, thousands(normal), normal_bad, sub_col, op.what, mark);
        for e in examples {
            println!("{e}");
        }
    }

    // RNG.
    let streams = 1024 * args.scale as u32;
    let calls = 4000u32;
    let mut seeds: Vec<i32> = vec![0, 1, -1, i32::MIN, i32::MAX, 12345, 1857374301, 161803398, -161803398, 2, -2];
    while seeds.len() < streams as usize {
        seeds.push(rng.u64() as i32);
    }
    let state = gpu.device_buffer(60 * 4 * streams as u64)?;
    let seed_buf = gpu.upload(&seeds)?;
    let out = gpu.device_buffer(8 * (calls as u64 + 1) * streams as u64)?;
    gpu_time += rng_kernel.dispatch(&[&state, &seed_buf, &out], as_bytes(&[streams, calls]), streams.div_ceil(64))?;
    let got: Vec<[u32; 2]> = gpu.read(&out, ((calls + 1) * streams) as usize)?;
    let mut rng_bad = 0usize;
    let mut rng_example = None;
    let mut cpu = Vec::with_capacity(calls as usize);
    for (s, &seed) in seeds.iter().enumerate() {
        cpu.clear();
        let draws = rng_script(seed, calls, &mut cpu);
        for j in 0..=calls as usize {
            let want = if j < calls as usize { cpu[j] } else { draws };
            let g = got[j * streams as usize + s];
            let g = g[0] as u64 | (g[1] as u64) << 32;
            if g != want {
                rng_bad += 1;
                rng_example.get_or_insert(format!("    seed {seed}, call {j} (kind {}): CPU 0x{want:x}, GPU 0x{g:x}", j % 7));
            }
        }
    }
    all_required_ok &= rng_bad == 0;
    println!(
        "{:<20} {:>10} {:>10}   {:>16}   System.Random / RaceRandom, {} streams x {} calls{}",
        "net_random",
        thousands(streams as usize * (calls as usize + 1)),
        rng_bad,
        "-",
        streams,
        calls,
        if rng_bad == 0 { "" } else { "   <-- FAIL" }
    );
    if let Some(e) = rng_example {
        println!("{e}");
    }

    println!();
    println!("Hardware operations, for information (div_rn / sqrt_rn only use them as starting guesses):");
    for l in info_lines {
        println!("{l}");
    }
    println!();
    println!("{} cases in {:.2} s ({:.2} s on the GPU)", thousands(cases.len() + (streams * (calls + 1)) as usize), wall.as_secs_f64(), gpu_time.as_secs_f64());
    if all_required_ok {
        if denormal_failures == 0 {
            println!("PASS: this device reproduces the CPU arithmetic bit for bit, subnormals included.");
        } else {
            println!(
                "PASS for normal values: {denormal_failures} subnormal cases differ (the device flushes denormals). Exact only while the sim \
                 stays out of the subnormal range."
            );
        }
    } else {
        println!("FAIL: see the rows marked FAIL above.");
    }
    Ok(all_required_ok)
}
