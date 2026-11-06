//! Checks the GPU's Mathf.Pow / Mathf.Log10 (shaders/lib/transcend.glsl) against this machine's CPU libm at every
//! call site in the sim, for every f32 stat in [1, 2000] (stats are clamped to that range before these calls).
//!
//! ```sh
//! cargo run --release --features gpu --bin gpu_mathcheck              # exhaustive: ~92 M stats per site
//! cargo run --release --features gpu --bin gpu_mathcheck -- --stride 64
//! ```
//!
//! A mismatch is an input where the CPU's f64 result sits on the other side of an f32 rounding boundary from the
//! correctly rounded one (the GPU computes ~2^-96 accurate values). Exit status 0 when there are none.
//!
//! One more row needs no GPU: the HP formula's `Pow(x, 2)` takes a speed difference, not a stat, and the GPU computes
//! it as `x * x` (the f64 square of an f32 is exact, so a libm that is exact there rounds the same way). The libm call
//! is checked for every f32 in [-256, 256].

use std::process::ExitCode;
use std::time::Instant;

use uma_race_sim::data::{DataEra, GameData};
use uma_race_sim::gpu::{Gpu, as_bytes, shaders};
use uma_race_sim::sim::util::{mathf_log10, mathf_pow};

#[derive(Clone, Copy)]
enum Form {
    /// Pow(c * s, y)
    Pow,
    /// Log10(s * c)
    Log10,
    /// Log10(Pow((float)((double)s * (double)c + 1), 2))
    TemptationLog10,
}

struct Site {
    name: &'static str,
    form: Form,
    c: f32,
    y: f32,
}

impl Site {
    fn cpu(&self, s: f32) -> f32 {
        match self.form {
            Form::Pow => mathf_pow(self.c * s, self.y),
            Form::Log10 => mathf_log10(s * self.c),
            Form::TemptationLog10 => {
                let x = (s as f64 * self.c as f64 + 1.0) as f32;
                mathf_log10(mathf_pow(x, 2.0))
            }
        }
    }
    fn code(&self) -> u32 {
        match self.form {
            Form::Pow => 0,
            Form::Log10 => 1,
            Form::TemptationLog10 => 2,
        }
    }
}

fn sites(data: &GameData) -> Vec<Site> {
    let p = &data.param;
    vec![
        Site { name: "compete_top time", form: Form::Pow, c: p.CompeteTop.TimeCoef1, y: p.CompeteTop.TimeCoef2 },
        Site { name: "compete_top speed", form: Form::Pow, c: p.CompeteTop.AddParam1Coef1, y: p.CompeteTop.AddParam1Coef2 },
        Site { name: "compete_fight speed", form: Form::Pow, c: p.CompeteFight.AddParam1Coef1, y: p.CompeteFight.AddParam1Coef2 },
        Site { name: "compete_fight accel", form: Form::Pow, c: p.CompeteFight.AddParam2Coef1, y: p.CompeteFight.AddParam2Coef2 },
        Site { name: "lane move speed", form: Form::Pow, c: p.Skill.LaneMoveAddParam1, y: p.Skill.LaneMoveAddParam2 },
        Site { name: "last spurt guts", form: Form::Pow, c: p.LastSpurtTargetSpeedGutsCoef1, y: p.LastSpurtTargetSpeedGutsPow },
        Site { name: "wit log10", form: Form::Log10, c: 0.1, y: 0.0 },
        Site { name: "temptation log10", form: Form::TemptationLog10, c: 0.1, y: 0.0 },
    ]
}

fn main() -> ExitCode {
    let mut stride = 1u32;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--stride" => stride = it.next().and_then(|v| v.parse().ok()).unwrap_or(1).max(1),
            _ => {
                eprintln!("usage: gpu_mathcheck [--stride N]");
                return ExitCode::from(2);
            }
        }
    }
    match run(stride) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn run(stride: u32) -> Result<bool, String> {
    let gpu = Gpu::new()?;
    println!("Device  {} — {} {}  (fma fused: {})", gpu.info.name, gpu.info.driver_name, gpu.info.driver_info, gpu.fma64_fused);
    let data = GameData::load(std::path::Path::new(uma_race_sim::data::DATA_DIR), DataEra::current())?;
    let kernel = gpu.kernel(shaders::MATHCHECK, 1, 24)?;
    let (lo, hi) = (1.0f32.to_bits(), 2000.0f32.to_bits());
    let total = hi - lo + 1;
    let chunk = 1u32 << 21;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    println!("Stats   every {} of the {} f32 values in [1, 2000]\n", if stride == 1 { "one".to_string() } else { format!("{stride}th") }, total);
    let out = gpu.device_buffer(chunk as u64 * 4)?;
    let mut all_ok = true;
    for site in sites(&data) {
        let t0 = Instant::now();
        let (mut cases, mut bad) = (0u64, Vec::new());
        // Stats lo, lo + stride, ... by bit pattern, `chunk` per dispatch.
        let mut first = lo;
        while first <= hi {
            let count = chunk.min((hi - first) / stride + 1);
            let push: [u32; 6] = [first, count, stride, site.code(), site.c.to_bits(), site.y.to_bits()];
            kernel.dispatch(&[&out], as_bytes(&push), count.div_ceil(64))?;
            let got: Vec<u32> = gpu.read(&out, count as usize)?;
            let per = (count as usize).div_ceil(threads);
            let results: Vec<Vec<(u32, u32, u32)>> = std::thread::scope(|sc| {
                let hs: Vec<_> = (0..threads)
                    .map(|t| {
                        let got = &got;
                        let site = &site;
                        sc.spawn(move || {
                            let mut v = Vec::new();
                            for k in (t * per)..((t + 1) * per).min(count as usize) {
                                let bits = first + k as u32 * stride;
                                let want = site.cpu(f32::from_bits(bits)).to_bits();
                                if want != got[k] {
                                    v.push((bits, want, got[k]));
                                }
                            }
                            v
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for r in results {
                bad.extend(r);
            }
            cases += count as u64;
            match first.checked_add(count * stride) {
                Some(f) => first = f,
                None => break,
            }
        }
        all_ok &= bad.is_empty();
        println!(
            "{:<20} {:>11} stats  {:>4} differ  {:.1} s  {}",
            site.name,
            cases,
            bad.len(),
            t0.elapsed().as_secs_f64(),
            if bad.is_empty() { "PASS" } else { "FAIL" }
        );
        for &(bits, want, got) in bad.iter().take(10) {
            let s = f32::from_bits(bits);
            println!(
                "    stat {s:e} (0x{bits:08x}): CPU {:e} (0x{want:08x}), GPU {:e} (0x{got:08x}){}",
                f32::from_bits(want),
                f32::from_bits(got),
                if got == 0x7FC0_DEAD { " [GPU fault]" } else { "" }
            );
        }
    }
    all_ok &= check_square(threads);
    Ok(all_ok)
}

/// `Mathf.Pow(x, 2)` (CPU libm) against `x * x` for every f32 in [-256, 256]. The exponent goes through
/// `black_box`: with a constant 2 the compiler turns `powf` into a multiply (as it may in the sim), and this checks
/// the library call the sim makes when it does not.
fn check_square(threads: usize) -> bool {
    let t0 = Instant::now();
    let hi = 256.0f32.to_bits();
    // Magnitudes 0 ..= 256 by bit pattern, each with both signs.
    let per = (hi as usize + 1).div_ceil(threads);
    let bad: Vec<u32> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                sc.spawn(move || {
                    let mut v = Vec::new();
                    for m in (t * per) as u32..(((t + 1) * per).min(hi as usize + 1)) as u32 {
                        for bits in [m, m | 0x8000_0000] {
                            let x = f32::from_bits(bits);
                            let pow = (x as f64).powf(std::hint::black_box(2.0f64)) as f32;
                            if pow.to_bits() != (x * x).to_bits() {
                                v.push(bits);
                            }
                        }
                    }
                    v
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    println!(
        "{:<20} {:>11} x      {:>4} differ  {:.1} s  {}  (CPU only)",
        "hp Pow(x, 2)",
        2 * (hi as u64 + 1),
        bad.len(),
        t0.elapsed().as_secs_f64(),
        if bad.is_empty() { "PASS" } else { "FAIL" }
    );
    for &bits in bad.iter().take(10) {
        let x = f32::from_bits(bits);
        println!("    x {x:e} (0x{bits:08x}): Pow {:e}, x * x {:e}", (x as f64).powf(std::hint::black_box(2.0f64)) as f32, x * x);
    }
    bad.is_empty()
}
