//! Checks the GPU stage kernels against the CPU sim, one frame stage at a time, on real race states.
//!
//! For each sampled step of each race, the CPU state just before the stage is packed into one GPU lane, the kernel
//! runs the stage, and the result is compared field by field with the CPU state just after it. Races are synthetic
//! lobbies (`synth`) and, with `--corpus`, saved races.
//!
//! ```sh
//! cargo run --release --features gpu --bin gpu_stagecheck
//! cargo run --release --features gpu --bin gpu_stagecheck -- --stage order --races 500 --seed 7
//! cargo run --release --features gpu --bin gpu_stagecheck -- --corpus "path/to/Saved races"
//! ```
//!
//! Exit status 0 when every compared value matches.

use std::collections::{BTreeMap, HashMap};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use uma_race_sim::api::RaceInput;
use uma_race_sim::data::{DataEra, GameData, corpus};
use uma_race_sim::gpu::layout::{self, LaneState, Mismatch};
use uma_race_sim::gpu::{Buffer, Gpu, Kernel, as_bytes, pack, shaders, skills};
use uma_race_sim::sim::ai::position_keep::PositionKeepStrategyKind;
use uma_race_sim::sim::defs::SkillModifierParam;
use uma_race_sim::sim::skill::modifier::{SkillParamModifier, SkillParamModifierKind};
use uma_race_sim::sim::race::{self, DT, Race};
use uma_race_sim::sim::tick::{self, Stage};
use uma_race_sim::synth;

/// Where in a step a kernel runs: a frame stage, or the end of the step after the frame.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Point {
    /// `tick::begin_step`.
    BeginStep,
    Stage(Stage),
    /// `tick::end_step` (its recording is not ported). The kernel also sets `race_done` on the last step.
    EndStep,
}

/// A ported stage: its kernel and the frames it covers.
struct Ported {
    name: &'static str,
    stage: Point,
    spirv: &'static [u8],
    /// Whether the kernel implements this frame (`all_finished` as set by `UpdateHorses`).
    covers: fn(&Race, bool) -> bool,
    /// With `--perturb`: a change to a copy of the state that reaches paths real races rarely do. The copy is
    /// checked as an extra sample; the race itself goes on unchanged.
    perturb: Option<fn(&mut Race, &mut synth::Rng)>,
    /// Whether the kernel reads and writes the skill program and state (`gpu::skills`).
    skills: bool,
}

/// Puts one horse within 0.4 m and 1/45 lane of another (field::update_overlap pushes it out).
fn perturb_overlap(race: &mut Race, r: &mut synth::Rng) {
    let n = race.horses.len();
    let a = (r.u64() % n as u64) as usize;
    let b = (a + 1 + (r.u64() % (n as u64 - 1)) as usize) % n;
    let (d, l) = (race.horses[a].distance, race.horses[a].lane_distance);
    race.horses[b].distance = d + (r.range(-30, 30) as f32) * 0.01;
    race.horses[b].lane_distance = (l + (r.range(-20, 20) as f32) * 0.001).max(0.0);
}

/// Forces one rarely reached state on a random running horse before UpdateHorses: a rush starting now, a
/// position-keep check now (sometimes with no pacemaker yet), a position-keep mode in progress (often on the leader or
/// the pacemaker: the leader's Overtake end, SpeedUp's end distances, PaceDown on the pacemaker), another pacemaker,
/// or a skill modifier that few skills give (a pinned target lane, forced overtaking side, HP consumption rate,
/// target speed while changing lanes, the end of a rush), a last-spurt recompute at a random HP and Wit, or a Nige
/// turned Oonige.
fn perturb_update_horses(race: &mut Race, r: &mut synth::Rng) {
    let n = race.horses.len();
    let running: Vec<usize> = (0..n).filter(|&h| !race.horses[h].is_finished).collect();
    if running.is_empty() {
        return;
    }
    let pick = |r: &mut synth::Rng, v: &[usize]| v[(r.u64() % v.len() as u64) as usize];
    let h = pick(r, &running);
    match r.u64() % 11 {
        0 | 1 => {
            let t = &mut race.horses[h].ai.temptation;
            if !t.is_temptation {
                t.is_temptation_start_enable = true;
                t.temptation_start_section = 1;
            }
        }
        2 => {
            let in_window: Vec<usize> = running
                .iter()
                .copied()
                .filter(|&j| {
                    let (d, pk) = (race.horses[j].distance, &race.horses[j].ai.position_keep);
                    pk.check_start_distance <= d && d < pk.check_end_distance
                })
                .collect();
            let h = if in_window.is_empty() { h } else { pick(r, &in_window) };
            let pk = &mut race.horses[h].ai.position_keep;
            pk.cur_strategy = 0;
            pk.cool_down_sec = 0.0;
            pk.check_start_interval_sec = 0.0;
            if r.u64() % 4 == 0 {
                race.field.pacemaker.horse = None;
            }
        }
        3 | 4 => {
            let h = match r.u64() % 4 {
                0 => race.field.standings.horse_array_by_order[0].horse,
                1 => race.field.pacemaker.horse.unwrap_or(h),
                2 => {
                    // A pacemaker of any style.
                    race.field.pacemaker.horse = Some(h);
                    h
                }
                _ => h,
            };
            let mode = 1 + (r.u64() % 5) as usize;
            let d = race.horses[h].distance + r.range(1, 60) as f32;
            let target = r.range(0, 30) as f32 * 0.5;
            let pk = &mut race.horses[h].ai.position_keep;
            pk.cur_strategy = mode;
            pk.strategies[mode].end_distance = d;
            if let PositionKeepStrategyKind::PaceUp(ps) | PositionKeepStrategyKind::PaceDown(ps) = &mut pk.strategies[mode].kind {
                ps.pace_adjust_target_distance_diff = target;
            }
        }
        5 => race.field.pacemaker.horse = Some(pick(r, &(0..n).collect::<Vec<_>>())),
        8 | 9 => {
            let x = &mut race.horses[h];
            x.hp = x.max_hp * r.range(0, 100) as f32 * 0.01;
            // Often the lowest Wit, so that every candidate can fail its Wit check.
            x.base_wiz_adjusted = if r.u64() % 2 == 0 { 1.0 } else { r.range(1, 1500) as f32 };
            x.ai.is_req_calc_last_spurt = true;
            x.ai.last_spurt_calc_count = r.range(0, 2);
            // A front blocker running out of HP.
            let blocked: Vec<usize> = running.iter().copied().filter(|&j| race.horses[j].ai.block_horse.is_some()).collect();
            if !blocked.is_empty() {
                let b = race.horses[pick(r, &blocked)].ai.block_horse.unwrap();
                race.horses[b].hp = race.horses[b].max_hp * r.range(0, 3) as f32 * 0.01;
            }
        }
        10 => {
            // Every Nige but one turned Oonige, and one of them in SpeedUp (the single-Nige and Oonige end
            // distances).
            use uma_race_sim::sim::defs::{RunningStyle, RunningStyleEx};
            let nige: Vec<usize> = running.iter().copied().filter(|&j| race.horses[j].horse_data.running_style == RunningStyle::Nige).collect();
            if nige.is_empty() {
                return;
            }
            let keep = pick(r, &nige);
            for &j in &nige {
                if j != keep {
                    race.horses[j].horse_data.running_style_ex = RunningStyleEx::Oonige;
                }
            }
            let j = if r.u64() % 2 == 0 { keep } else { pick(r, &nige) };
            let d = race.horses[j].distance + r.range(1, 60) as f32;
            let pk = &mut race.horses[j].ai.position_keep;
            pk.cur_strategy = 1;
            pk.strategies[1].end_distance = d;
        }
        _ => {
            let (param, value) = match r.u64() % 6 {
                0 => (SkillModifierParam::TargetLane, r.range(0, 30) as f32 * 0.03),
                1 => (SkillModifierParam::ForceOvertakeIn, 1.0),
                2 => (SkillModifierParam::ForceOvertakeOut, 1.0),
                3 => (SkillModifierParam::HpDecRate, r.range(5, 20) as f32 * 0.1),
                4 => (SkillModifierParam::TargetSpeedOnMoveLane, 1.0),
                _ => (SkillModifierParam::TemptationEndTime, r.range(1, 40) as f32 * 0.1),
            };
            race.next_modifier_id += 1;
            let m = SkillParamModifier {
                id: race.next_modifier_id,
                kind: SkillParamModifierKind::Add,
                time: r.range(1, 50) as f32 * 0.1,
                is_one_time: false,
                is_time_infinite: false,
                value,
            };
            race.horses[h].skill_modifier_receiver.add_modifier(param, m);
        }
    }
}

const PORTED: &[Ported] = &[
    Ported {
        name: "skills_first",
        stage: Point::Stage(Stage::SkillsFirst),
        spirv: shaders::STAGE_SKILLS_FIRST,
        covers: |_, _| true,
        perturb: None,
        skills: true,
    },
    Ported {
        name: "skills_second",
        stage: Point::Stage(Stage::SkillsSecond),
        spirv: shaders::STAGE_SKILLS_SECOND,
        covers: |_, _| true,
        perturb: None,
        skills: true,
    },
    Ported { name: "pre_update", stage: Point::Stage(Stage::PreUpdate), spirv: shaders::STAGE_PRE_UPDATE, covers: |_, _| true, perturb: None, skills: false },
    Ported {
        name: "update_horses",
        stage: Point::Stage(Stage::UpdateHorses),
        spirv: shaders::STAGE_UPDATE_HORSES,
        covers: |_, _| true,
        perturb: Some(perturb_update_horses),
        skills: true,
    },
    Ported {
        name: "field",
        stage: Point::Stage(Stage::Field),
        spirv: shaders::STAGE_FIELD,
        // Before July 2026 the pacemaker is set in EndOfFrame instead (not ported).
        covers: |race, _| !race.data.era.pre_2026_07_courses,
        perturb: Some(perturb_overlap),
        skills: false,
    },
    Ported { name: "order", stage: Point::Stage(Stage::Order), spirv: shaders::STAGE_ORDER, covers: |_, _| true, perturb: None, skills: false },
    Ported {
        name: "end_of_frame",
        stage: Point::Stage(Stage::EndOfFrame),
        spirv: shaders::STAGE_END_OF_FRAME,
        // The pre-July-2026 pacemaker update is not ported.
        covers: |race, _| !race.data.era.pre_2026_07_courses,
        perturb: None,
        skills: true,
    },
    Ported { name: "post_update", stage: Point::Stage(Stage::PostUpdate), spirv: shaders::STAGE_POST_UPDATE, covers: |_, _| true, perturb: None, skills: false },
    Ported { name: "begin_step", stage: Point::BeginStep, spirv: shaders::STAGE_BEGIN_STEP, covers: |_, _| true, perturb: None, skills: false },
    Ported { name: "end_step", stage: Point::EndStep, spirv: shaders::STAGE_END_STEP, covers: |_, _| true, perturb: None, skills: false },
];

struct Args {
    stage: String,
    races: u64,
    first: u64,
    seed: u64,
    corpus: Option<PathBuf>,
    every: u64,
    batch: usize,
    perturb: bool,
    coverage: bool,
    explain: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args { stage: "all".into(), races: 200, first: 0, seed: 1, corpus: None, every: 1, batch: 2048, perturb: false, coverage: false, explain: false };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        let mut val = || it.next().ok_or(format!("{x} needs a value"));
        match x.as_str() {
            "--stage" => a.stage = val()?,
            "--races" => a.races = val()?.parse().map_err(|_| "--races N")?,
            "--first" => a.first = val()?.parse().map_err(|_| "--first K")?,
            "--seed" => a.seed = val()?.parse().map_err(|_| "--seed N")?,
            "--corpus" => a.corpus = Some(PathBuf::from(val()?)),
            "--every" => a.every = val()?.parse::<u64>().map_err(|_| "--every N")?.max(1),
            "--batch" => a.batch = val()?.parse().map_err(|_| "--batch N")?,
            "--perturb" => a.perturb = true,
            "--coverage" => a.coverage = true,
            "--explain" => a.explain = true,
            "-h" | "--help" => {
                return Err(format!(
                    "usage: gpu_stagecheck [--stage {}|all] [--races N] [--first K] [--seed N] [--corpus DIR] [--every N] [--batch N] [--perturb] [--explain] [--coverage]",
                    PORTED.iter().map(|p| p.name).collect::<Vec<_>>().join("|")
                ));
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

/// Results for one stage.
/// A skill state label without its horse, skill and term numbers ("h3 s5 (100011) d0 cond t2 s1" → "d cond t s1").
fn skill_label_kind(label: &str) -> String {
    label
        .split(' ')
        .filter(|w| !w.starts_with('('))
        .map(|w| w.trim_end_matches(|c: char| c.is_ascii_digit()))
        .filter(|w| !w.is_empty() && *w != "h" && *w != "s")
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Default)]
struct Tally {
    samples: usize,
    lanes_bad: usize,
    by_field: BTreeMap<String, usize>,
    examples: Vec<String>,
    gpu_time: Duration,
    /// `--explain`: print every value that differs in the first bad sample.
    explain: bool,
}

/// A batch's skill state buffer (lanes times the largest race's words) stays under this, about 16 MB: a race has
/// tens of thousands of skill state words, and each lane keeps two copies on the CPU.
const SKILL_BATCH_WORDS: usize = 1 << 22;

/// Set while a race runs on the CPU inside `catch_unwind` (cleared while a batch is dispatched). A panic there means
/// the CPU sim cannot run the race (it stops at `race::master` on data it does not have yet); like `compare --corpus`, such a race
/// is skipped and the panic is not printed. Any other panic is printed and ends the run.
static IN_CPU_SIM: AtomicBool = AtomicBool::new(false);

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "panic".into())
}

/// A race's block of the course buffer (`pack::course_words`), keyed by its course.
type Course = Rc<(String, Vec<u32>)>;

fn course_of(race: &Race) -> Course {
    let cs = &race.info.race_course_set;
    let key = format!("{} {} {} {} {}", cs.id, cs.race_track_id, cs.distance, cs.ground, cs.inout);
    Rc::new((key, pack::course_words(race)))
}

/// The skill side of a sample: the race's program (shared by its samples) and the skill state before and after.
struct SkillSample {
    program: Rc<Vec<u32>>,
    labels: Rc<Vec<String>>,
    before: Vec<u32>,
    after: Vec<u32>,
}

/// One state before and after the stage.
struct Sample {
    before: LaneState,
    after: LaneState,
    origin: String,
    course: Course,
    skills: Option<SkillSample>,
}

/// Samples waiting for a dispatch.
struct Batch {
    before: Vec<LaneState>,
    after: Vec<LaneState>,
    origin: Vec<String>,
    course: Vec<Course>,
    skills: Vec<Option<SkillSample>>,
}

fn flush(gpu: &Gpu, kernel: &Kernel, params: &Buffer, batch: &mut Batch, tally: &mut Tally) -> Result<(), String> {
    let lanes = batch.before.len();
    if lanes == 0 {
        return Ok(());
    }
    // One block per distinct course; each lane gets its block's offset.
    let mut words: Vec<u32> = Vec::new();
    let mut at: HashMap<&str, u32> = HashMap::new();
    for (l, c) in batch.course.iter().enumerate() {
        let base = *at.entry(c.0.as_str()).or_insert_with(|| {
            let b = words.len() as u32;
            words.extend_from_slice(&c.1);
            b
        });
        batch.before[l].u32(layout::R_COURSE_BASE.race(), base);
        batch.after[l].u32(layout::R_COURSE_BASE.race(), base);
    }
    let courses = gpu.upload(&words)?;
    // Skill programs: one block per race; skill state: word w of lane l at w * lanes + l.
    let mut programs: Vec<u32> = vec![0];
    let mut prog_at: HashMap<*const Vec<u32>, u32> = HashMap::new();
    let state_words = batch.skills.iter().flatten().map(|k| k.before.len()).max().unwrap_or(0).max(1);
    let mut state = vec![0u32; state_words * lanes];
    for (l, k) in batch.skills.iter().enumerate() {
        let Some(k) = k else { continue };
        let base = *prog_at.entry(Rc::as_ptr(&k.program)).or_insert_with(|| {
            let b = programs.len() as u32;
            programs.extend_from_slice(&k.program);
            b
        });
        batch.before[l].u32(layout::R_SKILL_BASE.race(), base);
        batch.after[l].u32(layout::R_SKILL_BASE.race(), base);
        for (w, &v) in k.before.iter().enumerate() {
            state[w * lanes + l] = v;
        }
    }
    let program_buf = gpu.upload(&programs)?;
    let state_buf = gpu.upload(&state)?;
    let buf = gpu.upload(&layout::interleave(&batch.before))?;
    let push: [u32; 2] = [lanes as u32, DT.to_bits()];
    tally.gpu_time +=
        kernel.dispatch(&[&buf, params, &courses, &program_buf, &state_buf], as_bytes(&push), (lanes as u32).div_ceil(64))?;
    let out: Vec<u32> = gpu.read(&buf, layout::STATE_PLANES as usize * lanes)?;
    let skill_out: Vec<u32> = if batch.skills.iter().any(Option::is_some) { gpu.read(&state_buf, state_words * lanes)? } else { Vec::new() };
    let mut mismatches: Vec<Mismatch> = Vec::new();
    for (l, after) in batch.after.iter().enumerate() {
        mismatches.clear();
        layout::diff(after, &out, l, lanes, &mut mismatches);
        // (label, CPU, GPU) of the skill state words that differ.
        let mut skill_bad: Vec<(&str, u32, u32)> = Vec::new();
        if let Some(k) = &batch.skills[l] {
            for (w, &cpu) in k.after.iter().enumerate() {
                let gpu = skill_out[w * lanes + l];
                if gpu != cpu {
                    skill_bad.push((&k.labels[w], cpu, gpu));
                }
            }
        }
        tally.samples += 1;
        if mismatches.is_empty() && skill_bad.is_empty() {
            continue;
        }
        tally.lanes_bad += 1;
        for m in &mismatches {
            *tally.by_field.entry(m.field.to_string()).or_default() += 1;
        }
        for (label, _, _) in &skill_bad {
            *tally.by_field.entry(format!("skill state: {}", skill_label_kind(label))).or_default() += 1;
        }
        if tally.explain && tally.lanes_bad == 1 {
            eprintln!("  {}, every value that differs:", batch.origin[l]);
            for m in &mismatches {
                eprintln!("    {} = {} on the CPU, {} on the GPU", m.label, m.cpu, m.gpu);
            }
            for (label, c, g) in &skill_bad {
                eprintln!("    skill state {label} = 0x{c:08x} on the CPU, 0x{g:08x} on the GPU");
            }
        }
        if tally.examples.len() < 8 {
            let n = mismatches.len() + skill_bad.len();
            let first = match (mismatches.first(), skill_bad.first()) {
                (Some(m), _) => format!("{} = {} on the CPU, {} on the GPU", m.label, m.cpu, m.gpu),
                (None, Some((label, c, g))) => format!("skill state {label} = 0x{c:08x} on the CPU, 0x{g:08x} on the GPU"),
                (None, None) => unreachable!(),
            };
            tally.examples.push(format!("    {}: {first} ({n} values differ)", batch.origin[l]));
        }
    }
    batch.before.clear();
    batch.after.clear();
    batch.origin.clear();
    batch.course.clear();
    batch.skills.clear();
    Ok(())
}

/// Runs one race on the CPU, sampling the stage on every `every`-th covered frame.
fn sample_race(
    input: &RaceInput,
    data: &GameData,
    label: &str,
    p: &Ported,
    every: u64,
    perturb: bool,
    mut take: impl FnMut(Sample) -> Result<(), String>,
) -> Result<(), String> {
    let mut rng = synth::Rng(input.random_seed as u64 ^ 0x5eed);
    let mut race = race::init(input, data);
    let course = course_of(&race);
    // The race's skill program: built from every sample, and the same every time (it is static).
    let mut program: Option<(Rc<Vec<u32>>, Rc<Vec<String>>)> = None;
    let mut skills = |before: &Race, after: &Race| -> Result<Option<SkillSample>, String> {
        if !p.skills {
            return Ok(None);
        }
        // Labels once per race.
        let b = if program.is_none() { skills::pack_skills(before) } else { skills::pack_skills_unlabeled(before) };
        let a = skills::pack_skills_unlabeled(after);
        if a.program != b.program {
            return Err(format!("{label}: the skill program changed within the stage"));
        }
        let (prog, labels) = match &program {
            Some((prog, labels)) if **prog == b.program => (prog.clone(), labels.clone()),
            Some(_) => return Err(format!("{label}: the skill program changed between steps")),
            None => {
                let x = (Rc::new(b.program), Rc::new(b.labels));
                program = Some(x.clone());
                x
            }
        };
        Ok(Some(SkillSample { program: prog, labels, before: b.state, after: a.state }))
    };
    race.record_frames = false;
    tick::start(&mut race);
    let points: Vec<Point> = [Point::BeginStep].into_iter().chain(Stage::ALL.iter().map(|&st| Point::Stage(st))).chain([Point::EndStep]).collect();
    let mut step = 0u64;
    loop {
        step += 1;
        let mut all_finished = true;
        let mut is_end = false;
        for &point in &points {
            // Runs the point; true when it ends the race (end_step).
            let run = |race: &mut Race, flag: &mut bool| -> bool {
                match point {
                    Point::BeginStep => {
                        tick::begin_step(race);
                        false
                    }
                    Point::Stage(st) => {
                        tick::run_stage(race, st, DT, flag);
                        false
                    }
                    Point::EndStep => tick::end_step(race, *flag),
                }
            };
            if point == p.stage && step % every == 0 && (p.covers)(&race, all_finished) {
                if let (true, Some(f)) = (perturb, p.perturb) {
                    if rng.u64() % 8 == 0 {
                        let mut fork = race.clone();
                        f(&mut fork, &mut rng);
                        let mut before = pack::pack(&fork);
                        before.bool(layout::R_FRAME_ALL_FINISHED.race(), all_finished);
                        let mut flag = all_finished;
                        let snapshot = if p.skills { Some(fork.clone()) } else { None };
                        let e = run(&mut fork, &mut flag);
                        let mut after = pack::pack(&fork);
                        after.bool(layout::R_FRAME_ALL_FINISHED.race(), flag);
                        after.bool(layout::R_RACE_DONE.race(), e);
                        let sk = match &snapshot {
                            Some(b) => skills(b, &fork)?,
                            None => None,
                        };
                        let origin = format!("{label} step {step} (perturbed)");
                        take(Sample { before, after, origin, course: course.clone(), skills: sk })?;
                    }
                }
                let mut before = pack::pack(&race);
                before.bool(layout::R_FRAME_ALL_FINISHED.race(), all_finished);
                let missing = if step == every { before.missing() } else { Vec::new() };
                if !missing.is_empty() {
                    return Err(format!("gpu::pack leaves out {} values, first: {}", missing.len(), missing[0]));
                }
                let snapshot = if p.skills { Some(race.clone()) } else { None };
                let e = run(&mut race, &mut all_finished);
                is_end |= e;
                let mut after = pack::pack(&race);
                // all_finished after the stage (only UpdateHorses changes it), and race_done (end_step sets it).
                after.bool(layout::R_FRAME_ALL_FINISHED.race(), all_finished);
                after.bool(layout::R_RACE_DONE.race(), e);
                let sk = match &snapshot {
                    Some(b) => skills(b, &race)?,
                    None => None,
                };
                take(Sample { before, after, origin: format!("{label} step {step}"), course: course.clone(), skills: sk })?;
            } else {
                is_end |= run(&mut race, &mut all_finished);
            }
        }
        if is_end {
            return Ok(());
        }
    }
}

/// What a skill detail exercises: its trigger kinds and ability types, target types, value and time usages, and
/// the rarer detail features.
fn detail_features(race: &Race, h: usize, s: usize, d: usize) -> Vec<String> {
    use uma_race_sim::sim::skill::ability::SkillAbilityAdditionalActivate as Aa;
    use uma_race_sim::sim::skill::trigger::SkillTriggerCollection as C;
    let m = &race.horses[h].skill_manager;
    let det = &m.skills[s].details[d];
    let mut f = Vec::new();
    for (slot, c) in [("cond", &det.trigger), ("pre", &det.pre_trigger)] {
        if let C::Collection { any_empty_group, has_lot_other, .. } = c {
            if slot == "pre" {
                f.push("has precondition".to_string());
            }
            if *any_empty_group {
                f.push("empty OR-group".to_string());
            }
            if *has_lot_other {
                f.push("random_lot_other_activate collection".to_string());
            }
            let groups: std::collections::BTreeSet<u16> = m.triggers[c.range()].iter().map(|t| t.group).collect();
            if groups.len() > 1 {
                f.push("several OR-groups".to_string());
            }
        }
        for t in &m.triggers[c.range()] {
            let k = format!("{:?}", t.kind);
            let name = k.split(['{', '(', ' ']).next().unwrap_or("").to_string();
            f.push(format!("kind {name}"));
        }
    }
    if d == 1 {
        f.push("second detail".to_string());
    }
    if det.is_use_skill_detail_cool_down_time {
        f.push("detail cool down".to_string());
    }
    if det.is_not_activate_skill {
        f.push("not an activate skill".to_string());
    }
    if det.is_time_infinite {
        f.push("infinite time".to_string());
    }
    f.push(format!("time usage {}", det.time.usage));
    for a in det.abilities.iter().flatten() {
        f.push(format!("ability type {}", a.ability_type));
        f.push(format!("target type {}", a.target.target_type));
        f.push(format!("value usage {}", a.ability_value_with_usage.usage));
        match a.ability_additional_activate {
            Aa::Null => {}
            Aa::OrderUp { .. } => f.push("additional activate OrderUp".to_string()),
            Aa::ActivateAnySkill { .. } => f.push("additional activate ActivateAnySkill".to_string()),
        }
    }
    f.sort();
    f.dedup();
    f
}

/// `--coverage`: which skill paths the races reach, on the CPU only. For each feature of a skill detail (trigger
/// kind, ability type, ...): how many details have it, and how many activations those details had. Also the
/// events the detail lists don't show (additional activations, time extensions, reserved skills).
fn skill_coverage(races: &[(String, RaceInput, DataEra)], datas: &HashMap<DataEra, GameData>) {
    #[derive(Default)]
    struct C {
        details: u64,
        /// Active after `tick::start` (the zero-DT first frame, which is not sampled).
        at_start: u64,
        activations: u64,
    }
    let mut table: BTreeMap<String, C> = BTreeMap::new();
    let mut events: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut steps = 0u64;
    for (label, input, era) in races {
        IN_CPU_SIM.store(true, Ordering::Relaxed);
        let r = catch_unwind(AssertUnwindSafe(|| {
            let mut race = race::init(input, &datas[era]);
            race.record_frames = false;
            tick::start(&mut race);
            let mut feats: Vec<Vec<[Vec<String>; 2]>> = Vec::new();
            for h in 0..race.horses.len() {
                let mut v = Vec::new();
                for s in 0..race.horses[h].skill_manager.skills.len() {
                    v.push([detail_features(&race, h, s, 0), detail_features(&race, h, s, 1)]);
                }
                feats.push(v);
            }
            for (h, v) in feats.iter().enumerate() {
                for (s, pair) in v.iter().enumerate() {
                    for d in 0..2 {
                        let active = race.horses[h].skill_manager.skills[s].details[d].is_activated;
                        for f in &pair[d] {
                            let c = table.entry(f.clone()).or_default();
                            c.details += 1;
                            c.at_start += active as u64;
                        }
                    }
                }
            }
            let mut steps_here = 0u64;
            loop {
                steps_here += 1;
                tick::begin_step(&mut race);
                let mut all_finished = true;
                for stage in Stage::ALL {
                    let skills = matches!(stage, Stage::SkillsFirst | Stage::SkillsSecond);
                    let before: Vec<Vec<([bool; 2], Vec<i32>, Vec<i32>)>> = if skills {
                        race.horses
                            .iter()
                            .map(|h| {
                                h.skill_manager
                                    .skills
                                    .iter()
                                    .map(|s| {
                                        let act = [s.details[0].is_activated, s.details[1].is_activated];
                                        let aa = s.details.iter().flat_map(|d| d.abilities.iter().flatten().map(|a| a.additional_activate_count)).collect();
                                        let inc = s.details.iter().map(|d| d.time.increment_count).collect();
                                        (act, aa, inc)
                                    })
                                    .collect()
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let reserved: Vec<Vec<i32>> =
                        if skills { race.horses.iter().map(|h| h.skill_manager.reserved_skill_ids.clone()).collect() } else { Vec::new() };
                    tick::run_stage(&mut race, stage, DT, &mut all_finished);
                    if !skills {
                        continue;
                    }
                    for (h, hb) in before.iter().enumerate() {
                        if !race.horses[h].skill_manager.reserved_skill_ids.is_empty() {
                            *events.entry("reserved list drawn (per horse and frame)").or_default() += 1;
                        }
                        for (s, (act, aa, inc)) in hb.iter().enumerate() {
                            let sk = &race.horses[h].skill_manager.skills[s];
                            for d in 0..2 {
                                if !act[d] && sk.details[d].is_activated {
                                    for f in &feats[h][s][d] {
                                        table.get_mut(f).unwrap().activations += 1;
                                    }
                                    if d == 0 && reserved[h].contains(&sk.id) {
                                        *events.entry("reserved skill activated").or_default() += 1;
                                    }
                                }
                            }
                            let aa2: Vec<i32> = sk.details.iter().flat_map(|d| d.abilities.iter().flatten().map(|a| a.additional_activate_count)).collect();
                            if aa2.iter().zip(aa).any(|(x, y)| x > y) {
                                *events.entry("additional activation").or_default() += 1;
                            }
                            if sk.details.iter().map(|d| d.time.increment_count).zip(inc).any(|(x, y)| x > *y) {
                                *events.entry("time extended (usage 4)").or_default() += 1;
                            }
                        }
                    }
                }
                if tick::end_step(&mut race, all_finished) {
                    break;
                }
            }
            steps_here
        }));
        IN_CPU_SIM.store(false, Ordering::Relaxed);
        match r {
            Ok(n) => steps += n,
            Err(_) => eprintln!("skipping {label}: the CPU sim stops on it"),
        }
    }
    println!("Skill coverage over {} races, {steps} steps\n", races.len());
    println!("  {:>8} {:>8} {:>11}  feature", "details", "at start", "activations");
    for (f, c) in &table {
        let note = if c.activations == 0 { "  <- never activated in a step" } else { "" };
        println!("  {:>8} {:>8} {:>11}  {f}{note}", c.details, c.at_start, c.activations);
    }
    println!();
    for (e, n) in &events {
        println!("  {n:>8}  {e}");
    }
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
    let stages: Vec<&Ported> = PORTED.iter().filter(|p| args.stage == "all" || p.name == args.stage).collect();
    if stages.is_empty() {
        return Err(format!("no ported stage {:?}", args.stage));
    }
    // Races: synthetic lobbies on the current data, then the corpus (each with its era's data).
    let data_dir = Path::new(uma_race_sim::data::DATA_DIR);
    let mut datas: HashMap<DataEra, GameData> = HashMap::new();
    datas.insert(DataEra::current(), GameData::load(data_dir, DataEra::current())?);
    let instances = synth::race_instances(&datas[&DataEra::current()]);
    if instances.is_empty() && args.races > 0 {
        return Err("no race instance has its course files in data/".into());
    }
    // Synthetic race k has seed seed * 1000003 + k; --first skips to one.
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
    // One params buffer per era; a batch holds races of one era.
    races.sort_by_key(|r| format!("{:?}", r.2));
    if args.coverage {
        skill_coverage(&races, &datas);
        return Ok(true);
    }
    let gpu = Gpu::new()?;
    println!("Device  {} — {} {}", gpu.info.name, gpu.info.driver_name, gpu.info.driver_info);
    let mut params: HashMap<DataEra, Buffer> = HashMap::new();
    for (era, data) in &datas {
        let w = pack::pack_params(&data.param);
        let missing = w.missing();
        if !missing.is_empty() {
            return Err(format!("gpu::pack_params leaves out {} values, first: {}", missing.len(), missing[0]));
        }
        params.insert(era.clone(), gpu.upload(&w.words)?);
    }
    println!("Races   {} ({} synthetic)\n", races.len(), args.races);

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !IN_CPU_SIM.load(Ordering::Relaxed) {
            default_hook(info)
        }
    }));
    let mut all_ok = true;
    let mut skipped: Vec<(String, String)> = Vec::new();
    for (si, p) in stages.into_iter().enumerate() {
        // State, params, the course buffer (which only UpdateHorses reads), the skill program and the skill state.
        let kernel = gpu.kernel(p.spirv, 5, 8)?;
        let t0 = Instant::now();
        let mut tally = Tally { explain: args.explain, ..Tally::default() };
        let mut batch = Batch { before: Vec::new(), after: Vec::new(), origin: Vec::new(), course: Vec::new(), skills: Vec::new() };
        let mut cur_era: Option<&DataEra> = None;
        // The largest skill state in the batch (words): a batch of skill samples also stops at SKILL_BATCH_WORDS.
        let mut skill_words = 0usize;
        for (label, input, era) in &races {
            if cur_era.is_some_and(|e| e != era) {
                flush(&gpu, &kernel, &params[cur_era.unwrap()], &mut batch, &mut tally)?;
            }
            cur_era = Some(era);
            let pbuf = &params[era];
            IN_CPU_SIM.store(true, Ordering::Relaxed);
            let run = catch_unwind(AssertUnwindSafe(|| {
                sample_race(input, &datas[era], label, p, args.every, args.perturb, |x: Sample| {
                    batch.before.push(x.before);
                    batch.after.push(x.after);
                    batch.origin.push(x.origin);
                    batch.course.push(x.course);
                    skill_words = skill_words.max(x.skills.as_ref().map_or(0, |k| k.before.len()));
                    batch.skills.push(x.skills);
                    if batch.before.len() >= args.batch || batch.before.len() * skill_words >= SKILL_BATCH_WORDS {
                        IN_CPU_SIM.store(false, Ordering::Relaxed);
                        flush(&gpu, &kernel, pbuf, &mut batch, &mut tally)?;
                        IN_CPU_SIM.store(true, Ordering::Relaxed);
                        skill_words = 0;
                    }
                    Ok(())
                })
            }));
            let in_cpu_sim = IN_CPU_SIM.swap(false, Ordering::Relaxed);
            match run {
                Ok(r) => r?,
                // The CPU sim stopped; the samples taken before that are still checked.
                Err(e) if in_cpu_sim => {
                    if si == 0 {
                        skipped.push((label.clone(), panic_message(e.as_ref())));
                    }
                }
                Err(e) => resume_unwind(e),
            }
        }
        if let Some(era) = cur_era {
            flush(&gpu, &kernel, &params[era], &mut batch, &mut tally)?;
        }
        let ok = tally.lanes_bad == 0 && tally.samples > 0;
        all_ok &= ok;
        println!(
            "{:<13} {:>9} samples  {:>7} differ   {:.1} s ({:.2} s on the GPU)  {}",
            p.name,
            tally.samples,
            tally.lanes_bad,
            t0.elapsed().as_secs_f64(),
            tally.gpu_time.as_secs_f64(),
            if ok { "PASS" } else { "FAIL" }
        );
        for (field, n) in &tally.by_field {
            println!("    {n:>9} x {field}");
        }
        for e in &tally.examples {
            println!("{e}");
        }
    }
    if !skipped.is_empty() {
        println!("\n{} races skipped: the CPU sim stops on them", skipped.len());
        let mut by_msg: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (label, msg) in &skipped {
            by_msg.entry(msg.as_str()).or_default().push(label.as_str());
        }
        for (msg, labels) in by_msg {
            println!("  {} × {msg}, first: {}", labels.len(), labels[0]);
        }
    }
    Ok(all_ok)
}
