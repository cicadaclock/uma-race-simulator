//! Race throughput on the GPU against the CPU. One lobby runs with many race seeds: on every CPU thread
//! (`race::init`, `tick::start`, steps to the end), and on the GPU (`gpu::run`: the CPU builds and packs each race,
//! the GPU steps all of them). Every race's results (each runner's finish order and finish time) are compared.
//!
//! ```sh
//! cargo run --release --features gpu --bin gpu_bench -- --seeds 4096
//! cargo run --release --features gpu --bin gpu_bench -- --race "path/to/Saved races/x.json" --seeds 8192 --batch 8192
//! ```
//!
//! Exit status 0 when every race has the same results on both.

use std::path::PathBuf;
use std::process::ExitCode;
use std::thread;
use std::time::{Duration, Instant};

use uma_race_sim::api::RaceInput;
use uma_race_sim::arena::{self, ArenaAllocator, ThreadArena};
use uma_race_sim::data::{DataEra, GameData, corpus};
use uma_race_sim::gpu::layout::{self, Field};
use uma_race_sim::gpu::run::{LaneInput, Runner, Shared};
use uma_race_sim::gpu::{Gpu, pack};
use uma_race_sim::sim::race::{self, Race};
use uma_race_sim::sim::tick;
use uma_race_sim::synth;

// The CPU sim's own allocator (as `bench`): each race in one arena region.
#[global_allocator]
static ALLOCATOR: ArenaAllocator = ArenaAllocator;

struct Args {
    lobby: u64,
    seed: u64,
    race: Option<PathBuf>,
    seeds: u64,
    batch: usize,
    threads: usize,
    profile: bool,
}

fn parse_args() -> Result<Args, String> {
    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    let mut a = Args { lobby: 0, seed: 1, race: None, seeds: 1024, batch: 4096, threads, profile: false };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        let mut val = || it.next().ok_or(format!("{x} needs a value"));
        match x.as_str() {
            "--lobby" => a.lobby = val()?.parse().map_err(|_| "--lobby K")?,
            "--seed" => a.seed = val()?.parse().map_err(|_| "--seed N")?,
            "--race" => a.race = Some(PathBuf::from(val()?)),
            "--seeds" => a.seeds = val()?.parse::<u64>().map_err(|_| "--seeds N")?.max(1),
            "--batch" => a.batch = val()?.parse::<usize>().map_err(|_| "--batch N")?.max(1),
            "--threads" => a.threads = val()?.parse::<usize>().map_err(|_| "--threads T")?.max(1),
            "--profile" => a.profile = true,
            "-h" | "--help" => {
                return Err("usage: gpu_bench [--lobby K] [--seed N] [--race FILE] [--seeds N] [--batch N] [--threads T] [--profile]".into());
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

/// Each runner's finish order and finish time (bits), from the CPU sim.
fn cpu_results(race: &Race) -> Vec<(i32, u32)> {
    race.horses.iter().map(|h| (h.finish_order, h.finish_time_scaled.to_bits())).collect()
}

/// The same, from lane `l` of a read GPU state.
fn gpu_results(state: &[u32], lanes: usize, l: usize, horses: usize) -> Vec<(i32, u32)> {
    let at = |f: Field, h: usize| state[f.horse(h) as usize * lanes + l];
    (0..horses).map(|h| (at(layout::H_FINISH_ORDER, h) as i32, at(layout::H_FINISH_TIME_SCALED, h))).collect()
}

/// Runs `f` over `items` split into `threads` contiguous chunks; results in order. Each item runs in its own arena
/// scope, so `f` must build its result inside `arena::outside`.
fn par_map<T: Sync, R: Send>(items: &[T], threads: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let chunk = items.len().div_ceil(threads).max(1);
    thread::scope(|s| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|c| {
                let f = &f;
                s.spawn(move || {
                    let _arena = ThreadArena::attach();
                    let mut out = Vec::with_capacity(c.len());
                    for x in c {
                        // SAFETY: f builds what it returns outside the scope.
                        let _scope = unsafe { arena::open_scope() };
                        let r = f(x);
                        arena::outside(|| out.push(r));
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("worker panicked")).collect()
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn rate(n: usize, t: Duration) -> String {
    format!("{:>10.0} races/s", n as f64 / t.as_secs_f64().max(1e-9))
}

fn run(args: &Args) -> Result<bool, String> {
    let data_dir = std::path::Path::new(uma_race_sim::data::DATA_DIR);
    let (label, input, era): (String, RaceInput, DataEra) = match &args.race {
        Some(f) => {
            let (input, _, date) = corpus::load_race(f)?;
            (f.display().to_string(), input, DataEra::for_date(&date)?)
        }
        None => {
            let data = GameData::load(data_dir, DataEra::current())?;
            let seed = args.seed.wrapping_mul(1_000_003).wrapping_add(args.lobby);
            let instances = synth::race_instances(&data);
            (format!("synthetic {seed}"), synth::lobby(&data, &instances, seed, 0), DataEra::current())
        }
    };
    let data = GameData::load(data_dir, era.clone())?;
    if data.era.pre_2026_07_courses {
        return Err("races before the July 2026 courses are not ported (their pacemaker update)".into());
    }
    let inputs: Vec<RaceInput> = (0..args.seeds)
        .map(|k| {
            let mut x = input.clone();
            x.random_seed = (input.random_seed as u32).wrapping_add((k as u32).wrapping_mul(0x9E37_79B9)) as i32;
            x
        })
        .collect();
    let n = inputs.len();
    println!("Lobby   {label}: {} runners, {n} race seeds, {} CPU threads", input.horses.len(), args.threads);

    // The CPU: whole races on every thread.
    let t = Instant::now();
    let cpu: Vec<Vec<(i32, u32)>> = par_map(&inputs, args.threads, |x| {
        let mut race = race::init(x, &data);
        race.record_frames = false;
        tick::start(&mut race);
        while !tick::step(&mut race) {}
        arena::outside(|| cpu_results(&race))
    });
    let cpu_time = t.elapsed();

    // Where the CPU's share of a GPU race goes (one thread, the first 256 seeds).
    let probe = &inputs[..n.min(256)];
    let mut parts = [Duration::ZERO; 6];
    for x in probe {
        let t = Instant::now();
        let mut race = race::init(x, &data);
        race.record_frames = false;
        parts[0] += t.elapsed();
        let t = Instant::now();
        tick::start(&mut race);
        parts[1] += t.elapsed();
        let t = Instant::now();
        let _ = pack::pack(&race);
        parts[2] += t.elapsed();
        let t = Instant::now();
        let _ = pack::course_words(&race);
        parts[3] += t.elapsed();
        let t = Instant::now();
        let _ = uma_race_sim::gpu::skills::pack_skills_unlabeled(&race);
        parts[4] += t.elapsed();
        let t = Instant::now();
        while !tick::step(&mut race) {}
        parts[5] += t.elapsed();
    }
    let per = |d: Duration| d.as_secs_f64() * 1e6 / probe.len() as f64;
    println!(
        "Per race on one CPU thread (µs): race::init {:.0}, tick::start {:.0}, pack {:.0}, course {:.0}, skills {:.0}; the steps {:.0}",
        per(parts[0]),
        per(parts[1]),
        per(parts[2]),
        per(parts[3]),
        per(parts[4]),
        per(parts[5])
    );

    let gpu = Gpu::new()?;
    println!("Device  {} — {} {}", gpu.info.name, gpu.info.driver_name, gpu.info.driver_info);
    let t = Instant::now();
    let runner = Runner::new(&gpu)?;
    let params = gpu.upload(&pack::pack_params(&data.param).words)?;
    println!("Kernels built in {:.1} s\n", t.elapsed().as_secs_f64());

    let (mut setup, mut upload, mut steps, mut readback) = (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let mut gpu_time = Duration::ZERO;
    let mut same = 0usize;
    let mut first_bad: Option<String> = None;
    let shared = Shared::default();
    let mut max_steps = 0u64;
    // The CPU builds and packs each race; batch b + 1 is built while the GPU runs batch b.
    let build = |batch: &[RaceInput]| -> (Vec<LaneInput>, Duration) {
        let t = Instant::now();
        let lanes = par_map(batch, args.threads, |x| {
            let mut race = race::init(x, &data);
            race.record_frames = false;
            tick::start(&mut race);
            let state = pack::pack(&race);
            let skills = uma_race_sim::gpu::skills::pack_skills_unlabeled(&race);
            arena::outside(|| LaneInput::assemble(&race, &state, &skills, &shared))
        });
        (lanes, t.elapsed())
    };
    let batches: Vec<&[RaceInput]> = inputs.chunks(args.batch).collect();
    let t_gpu = Instant::now();
    let mut next = Some(build(batches[0]));
    for (b, _) in batches.iter().enumerate() {
        let (mut lanes, built) = next.take().unwrap();
        setup += built;
        thread::scope(|sc| -> Result<(), String> {
            let pending = batches.get(b + 1).map(|&nb| sc.spawn(move || build(nb)));
            let t = Instant::now();
            if b == 0 {
                let sw = lanes.iter().map(|x| x.skill_state.len()).max().unwrap_or(0);
                println!(
                    "Lane memory: {} KB race state + {} KB skill state ({} MB for a batch of {})\n",
                    layout::STATE_PLANES as usize * 4 / 1024,
                    sw * 4 / 1024,
                    (layout::STATE_PLANES as usize + sw) * 4 * lanes.len() >> 20,
                    lanes.len()
                );
            }
            let dev = runner.upload(&mut lanes)?;
            drop(lanes);
            upload += t.elapsed();
            let t = Instant::now();
            let mut pace = Duration::ZERO;
            let mut done_steps = 0u64;
            if args.profile && b == 0 {
                // The first 200 steps with a timestamp after every kernel (they count as steps of the batch).
                let per = runner.profile(&dev, &params, 200)?;
                let sum: Duration = per.iter().map(|p| p.1).sum();
                println!("GPU time per step over the first 200 steps of {} lanes:", dev.lanes);
                for (name, d) in &per {
                    println!("  {name:<14} {:>8.1} µs  {:>5.1} %", d.as_secs_f64() * 1e6 / 200.0, 100.0 * d.as_secs_f64() / sum.as_secs_f64());
                }
                println!("  {:<14} {:>8.1} µs\n", "all", sum.as_secs_f64() * 1e6 / 200.0);
                gpu_time += sum;
                done_steps += 200;
            }
            loop {
                gpu_time += runner.steps_paced(&dev, &params, 200, Duration::from_millis(250), &mut pace)?;
                done_steps += 200;
                if dev.count_done(&gpu)? == dev.lanes {
                    break;
                }
                if done_steps > 100_000 {
                    return Err("a race did not end in 100000 steps".into());
                }
            }
            steps += t.elapsed();
            max_steps = max_steps.max(done_steps);
            let t = Instant::now();
            let state = dev.read_state(&gpu)?;
            readback += t.elapsed();
            for l in 0..dev.lanes {
                let k = b * args.batch + l;
                let g = gpu_results(&state, dev.lanes, l, input.horses.len());
                if state[layout::R_GPU_FAULT.race() as usize * dev.lanes + l] == 0 && g == cpu[k] {
                    same += 1;
                } else if first_bad.is_none() {
                    first_bad = Some(format!("seed {}: CPU {:?}, GPU {:?}", inputs[k].random_seed, cpu[k], g));
                }
            }
            next = pending.map(|h| h.join().expect("builder panicked"));
            Ok(())
        })?;
    }
    let total = t_gpu.elapsed();
    println!("CPU, {} threads      {}   {:.2} s", args.threads, rate(n, cpu_time), cpu_time.as_secs_f64());
    println!("GPU, end to end       {}   {:.2} s (the next batch builds while the GPU runs one)", rate(n, total), total.as_secs_f64());
    println!("  build + pack (CPU)  {}   {:.2} s", rate(n, setup), setup.as_secs_f64());
    println!("  upload              {}   {:.2} s", rate(n, upload), upload.as_secs_f64());
    println!("  steps               {}   {:.2} s ({:.2} s in submissions, up to {max_steps} steps per batch)", rate(n, steps), steps.as_secs_f64(), gpu_time.as_secs_f64());
    println!("  read back           {}   {:.2} s", rate(n, readback), readback.as_secs_f64());
    println!("\nResults: {same} of {n} races identical (finish order and time of every runner)");
    if let Some(e) = &first_bad {
        println!("  first difference: {e}");
    }
    Ok(same == n)
}
