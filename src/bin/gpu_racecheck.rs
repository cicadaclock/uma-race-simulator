//! Runs whole races on the GPU (`gpu::run`) and checks them against the CPU sim. Each race is built on the CPU
//! (`race::init`, `tick::start`) and packed into a lane; then the GPU and the CPU step every lane in lockstep, and
//! after every chunk of steps the whole state of every lane (race state and skill state) is compared. A lane that
//! differs is reported once, with the first chunk where it differs.
//!
//! ```sh
//! cargo run --release --features gpu --bin gpu_racecheck
//! cargo run --release --features gpu --bin gpu_racecheck -- --races 200 --chunk 10
//! cargo run --release --features gpu --bin gpu_racecheck -- --corpus "path/to/Saved races"
//! ```
//!
//! Exit status 0 when every lane matches the CPU at every compared step.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use uma_race_sim::api::RaceInput;
use uma_race_sim::data::{DataEra, GameData, corpus};
use uma_race_sim::gpu::layout::{self, LaneState, Mismatch};
use uma_race_sim::gpu::run::{LaneInput, Runner, Shared};
use uma_race_sim::gpu::{Buffer, Gpu, pack, skills};
use uma_race_sim::sim::race::{self, Race};
use uma_race_sim::sim::tick;
use uma_race_sim::synth;

struct Args {
    races: u64,
    first: u64,
    seed: u64,
    corpus: Option<PathBuf>,
    batch: usize,
    chunk: u32,
    seeds: u64,
    explain: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args { races: 50, first: 0, seed: 1, corpus: None, batch: 256, chunk: 1, seeds: 1, explain: false };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        let mut val = || it.next().ok_or(format!("{x} needs a value"));
        match x.as_str() {
            "--races" => a.races = val()?.parse().map_err(|_| "--races N")?,
            "--first" => a.first = val()?.parse().map_err(|_| "--first K")?,
            "--seed" => a.seed = val()?.parse().map_err(|_| "--seed N")?,
            "--corpus" => a.corpus = Some(PathBuf::from(val()?)),
            "--batch" => a.batch = val()?.parse::<usize>().map_err(|_| "--batch N")?.max(1),
            "--chunk" => a.chunk = val()?.parse::<u32>().map_err(|_| "--chunk S")?.max(1),
            "--seeds" => a.seeds = val()?.parse::<u64>().map_err(|_| "--seeds N")?.max(1),
            "--explain" => a.explain = true,
            "-h" | "--help" => {
                return Err("usage: gpu_racecheck [--races N] [--first K] [--seed N] [--corpus DIR] [--seeds N] [--batch N] [--chunk S] [--explain]".into());
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

/// Each submission of steps aims at this much GPU time (Windows resets a GPU that stays busy for 2 s).
const SUBMIT_BUDGET: Duration = Duration::from_millis(250);

/// Set while the CPU sim runs inside `catch_unwind`: a panic there skips the race silently.
static IN_CPU_SIM: AtomicBool = AtomicBool::new(false);

/// One race of a batch: the CPU copy, whether it has ended, and whether a mismatch was reported.
struct Lane<'a> {
    label: String,
    race: Race<'a>,
    ended: bool,
    bad: bool,
    labels: Vec<String>,
    course_base: u32,
    skill_base: u32,
}

#[derive(Default)]
struct Totals {
    races: usize,
    bad: usize,
    steps: u64,
    gpu_time: Duration,
    cpu_time: Duration,
    examples: Vec<String>,
    by_field: HashMap<String, usize>,
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

fn run(args: &Args) -> Result<bool, String> {
    let data_dir = Path::new(uma_race_sim::data::DATA_DIR);
    let mut datas: HashMap<DataEra, GameData> = HashMap::new();
    datas.insert(DataEra::current(), GameData::load(data_dir, DataEra::current())?);
    let instances = synth::race_instances(&datas[&DataEra::current()]);
    let mut races: Vec<(String, RaceInput, DataEra)> = (args.first..args.first + args.races)
        .map(|k| {
            let seed = args.seed.wrapping_mul(1_000_003).wrapping_add(k);
            (format!("synthetic {seed}"), synth::lobby(&datas[&DataEra::current()], &instances, seed, 0), DataEra::current())
        })
        .collect();
    if let Some(dir) = &args.corpus {
        for f in corpus::race_files(dir) {
            match corpus::load_race(&f).and_then(|(input, _, date)| Ok((input, DataEra::for_date(&date)?))) {
                Ok((input, era)) => races.push((f.display().to_string(), input, era)),
                Err(e) => eprintln!("skipping {}: {e}", f.display()),
            }
        }
    }
    for (_, _, era) in &races {
        if !datas.contains_key(era) {
            datas.insert(era.clone(), GameData::load(data_dir, era.clone())?);
        }
    }
    let before = races.len();
    races.retain(|(_, _, era)| !datas[era].era.pre_2026_07_courses);
    if races.len() < before {
        println!("{} races before the July 2026 courses left out (their pacemaker update is not ported)", before - races.len());
    }
    races.sort_by_key(|r| format!("{:?}", r.2));
    // --seeds: each lobby with N race seeds, next to each other (the GPU's use: one lobby per wave).
    if args.seeds > 1 {
        races = races
            .into_iter()
            .flat_map(|(label, input, era)| {
                (0..args.seeds).map(move |k| {
                    let mut x = input.clone();
                    x.random_seed = (input.random_seed as u32).wrapping_add((k as u32).wrapping_mul(0x9E37_79B9)) as i32;
                    (format!("{label} seed {}", x.random_seed), x, era.clone())
                })
            })
            .collect();
    }

    let gpu = Gpu::new()?;
    println!("Device  {} — {} {}", gpu.info.name, gpu.info.driver_name, gpu.info.driver_info);
    let t0 = Instant::now();
    let runner = Runner::new(&gpu)?;
    println!("Kernels built in {:.1} s", t0.elapsed().as_secs_f64());
    let mut params: HashMap<DataEra, Buffer> = HashMap::new();
    for (era, data) in &datas {
        params.insert(era.clone(), gpu.upload(&pack::pack_params(&data.param).words)?);
    }
    println!("Races   {} ({} synthetic), batches of {}, compared every {} steps\n", races.len(), args.races, args.batch, args.chunk);

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !IN_CPU_SIM.load(Ordering::Relaxed) {
            default_hook(info)
        }
    }));
    let mut totals = Totals::default();
    let mut skipped = 0usize;
    let t_all = Instant::now();
    let mut start = 0;
    let shared = Shared::default();
    while start < races.len() {
        let era = races[start].2.clone();
        let end = (start..races.len()).take(args.batch).take_while(|&k| races[k].2 == era).last().unwrap() + 1;
        let data = &datas[&era];
        let mut lanes: Vec<Lane> = Vec::new();
        let mut inputs: Vec<LaneInput> = Vec::new();
        for (label, input, _) in &races[start..end] {
            IN_CPU_SIM.store(true, Ordering::Relaxed);
            let built = catch_unwind(AssertUnwindSafe(|| {
                let mut race = race::init(input, data);
                race.record_frames = false;
                tick::start(&mut race);
                race
            }));
            IN_CPU_SIM.store(false, Ordering::Relaxed);
            let Ok(race) = built else {
                skipped += 1;
                continue;
            };
            let x = LaneInput::new(&race, true, &shared);
            lanes.push(Lane { label: label.clone(), race, ended: false, bad: false, labels: x.skill_labels.clone(), course_base: 0, skill_base: 0 });
            inputs.push(x);
        }
        start = end;
        if lanes.is_empty() {
            continue;
        }
        let dev = runner.upload(&mut inputs)?;
        for (lane, x) in lanes.iter_mut().zip(&inputs) {
            lane.course_base = x.state.words[layout::R_COURSE_BASE.race() as usize];
            lane.skill_base = x.state.words[layout::R_SKILL_BASE.race() as usize];
        }
        drop(inputs);
        totals.races += lanes.len();
        let mut step = 0u64;
        let mut pace = Duration::ZERO;
        while lanes.iter().any(|l| !l.ended) {
            totals.gpu_time += runner.steps_paced(&dev, &params[&era], args.chunk, SUBMIT_BUDGET, &mut pace)?;
            // The CPU copies, the same number of steps (or to their end).
            let t_cpu = Instant::now();
            for lane in lanes.iter_mut().filter(|l| !l.ended) {
                IN_CPU_SIM.store(true, Ordering::Relaxed);
                let r = catch_unwind(AssertUnwindSafe(|| {
                    for _ in 0..args.chunk {
                        if tick::step(&mut lane.race) {
                            return true;
                        }
                    }
                    false
                }));
                IN_CPU_SIM.store(false, Ordering::Relaxed);
                match r {
                    Ok(e) => lane.ended = e,
                    Err(_) => {
                        // The CPU sim stopped: nothing to compare from here.
                        lane.ended = true;
                        lane.bad = true;
                        skipped += 1;
                    }
                }
            }
            totals.cpu_time += t_cpu.elapsed();
            step += args.chunk as u64;
            let state = dev.read_state(&gpu)?;
            let skill_state = dev.read_skill_state(&gpu)?;
            let done = dev.done(&state);
            for (l, lane) in lanes.iter_mut().enumerate() {
                if lane.bad {
                    continue;
                }
                let mut want: LaneState = pack::pack(&lane.race);
                want.bool(layout::R_FRAME_ALL_FINISHED.race(), lane.ended);
                want.bool(layout::R_RACE_DONE.race(), lane.ended);
                want.u32(layout::R_COURSE_BASE.race(), lane.course_base);
                want.u32(layout::R_SKILL_BASE.race(), lane.skill_base);
                let mut mismatches: Vec<Mismatch> = Vec::new();
                layout::diff(&want, &state, l, dev.lanes, &mut mismatches);
                let sk = skills::pack_skills_unlabeled(&lane.race);
                let skill_bad: Vec<(usize, u32, u32)> = sk
                    .state
                    .iter()
                    .enumerate()
                    .filter_map(|(w, &c)| {
                        let g = skill_state[w * dev.lanes + l];
                        (g != c).then_some((w, c, g))
                    })
                    .collect();
                if mismatches.is_empty() && skill_bad.is_empty() && done[l] == lane.ended {
                    continue;
                }
                lane.bad = true;
                totals.bad += 1;
                for m in &mismatches {
                    *totals.by_field.entry(m.field.to_string()).or_default() += 1;
                }
                if !skill_bad.is_empty() {
                    *totals.by_field.entry("skill state".into()).or_default() += 1;
                }
                let first = match (mismatches.first(), skill_bad.first()) {
                    (Some(m), _) => format!("{} = {} on the CPU, {} on the GPU", m.label, m.cpu, m.gpu),
                    (None, Some(&(w, c, g))) => format!("skill state {} = 0x{c:08x} on the CPU, 0x{g:08x} on the GPU", lane.labels[w]),
                    (None, None) => format!("race ended on the CPU: {}, on the GPU: {}", lane.ended, done[l]),
                };
                let at = if args.chunk == 1 { format!("step {step}") } else { format!("steps {}..={step}", step + 1 - args.chunk as u64) };
                if totals.examples.len() < 8 {
                    totals.examples.push(format!("    {} {at}: {first} ({} values differ)", lane.label, mismatches.len() + skill_bad.len()));
                }
                if args.explain && totals.bad == 1 {
                    eprintln!("  {} {at}, every value that differs:", lane.label);
                    for m in &mismatches {
                        eprintln!("    {} = {} on the CPU, {} on the GPU", m.label, m.cpu, m.gpu);
                    }
                    for &(w, c, g) in &skill_bad {
                        eprintln!("    skill state {} = 0x{c:08x} on the CPU, 0x{g:08x} on the GPU", lane.labels[w]);
                    }
                }
            }
        }
        totals.steps += step;
    }
    let ok = totals.bad == 0;
    println!(
        "races  {:>6}   {:>5} differ   {:.1} s ({:.2} s on the GPU, {:.2} s stepping the CPU copies, {} steps)  {}",
        totals.races,
        totals.bad,
        t_all.elapsed().as_secs_f64(),
        totals.gpu_time.as_secs_f64(),
        totals.cpu_time.as_secs_f64(),
        totals.steps,
        if ok { "PASS" } else { "FAIL" }
    );
    let mut fields: Vec<_> = totals.by_field.iter().collect();
    fields.sort();
    for (f, n) in fields {
        println!("    {n:>8} x {f}");
    }
    for e in &totals.examples {
        println!("{e}");
    }
    if skipped > 0 {
        println!("\n{skipped} races skipped: the CPU sim stops on them");
    }
    Ok(ok)
}
