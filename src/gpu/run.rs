//! Whole races on the GPU. The CPU builds each race (`race::init`, `tick::start`: the skill windows and lots drawn
//! at construction, the zero-DT first frame) and packs it into a lane ([`LaneInput`]). Then every step is the ten
//! kernels of [`STEP_KERNELS`] in a row, recorded as one submission per chunk of steps. A lane whose race has ended
//! (`race_done`, set by the end_step kernel) is left alone by every kernel, so a batch runs until its longest race
//! ends.
//!
//! Only the July 2026 courses on: the pre-July-2026 pacemaker (EndOfFrame) is not ported, and the Field kernel
//! assumes the later order.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use super::context::Result;
use super::layout::{self, LaneState};
use super::{Buffer, Gpu, Kernel, as_bytes, pack, shaders, skills};
use crate::sim::race::{DT, Race};

/// One step (`tick::step`): begin_step, the eight frame stages in `tick::Stage::ALL` order, end_step.
pub const STEP_KERNELS: [(&str, &[u8]); 10] = [
    ("begin_step", shaders::STAGE_BEGIN_STEP),
    ("pre_update", shaders::STAGE_PRE_UPDATE),
    ("skills_first", shaders::STAGE_SKILLS_FIRST),
    ("skills_second", shaders::STAGE_SKILLS_SECOND),
    ("update_horses", shaders::STAGE_UPDATE_HORSES),
    ("field", shaders::STAGE_FIELD),
    ("post_update", shaders::STAGE_POST_UPDATE),
    ("order", shaders::STAGE_ORDER),
    ("end_of_frame", shaders::STAGE_END_OF_FRAME),
    ("end_step", shaders::STAGE_END_STEP),
];

/// Every kernel's buffers: state, params, course blocks, skill programs, skill state.
pub const KERNEL_BUFFERS: usize = 5;
/// Push constants: lanes, dt.
pub const KERNEL_PUSH_BYTES: u32 = 8;

/// Names a race's course block: lanes with the same key share one block.
pub fn course_key(race: &Race) -> String {
    let cs = &race.info.race_course_set;
    format!("{} {} {} {} {}", cs.id, cs.race_track_id, cs.distance, cs.ground, cs.inout)
}

/// The course blocks and skill programs of the lanes, each kept once: the seeds of one lobby share both, and a
/// lane is otherwise mostly these two (a program is about 90 KB, a course 60 KB). Usable from several threads.
#[derive(Default)]
pub struct Shared {
    courses: Mutex<HashMap<String, Arc<Vec<u32>>>>,
    programs: Mutex<Vec<Arc<Vec<u32>>>>,
}

impl Shared {
    /// The race's course block (built the first time a course is seen).
    pub fn course(&self, race: &Race) -> Arc<Vec<u32>> {
        let key = course_key(race);
        if let Some(c) = self.courses.lock().unwrap().get(&key) {
            return c.clone();
        }
        let words = Arc::new(pack::course_words(race));
        self.courses.lock().unwrap().entry(key).or_insert(words).clone()
    }

    /// The kept program equal to `words`, or a new one.
    pub fn program(&self, words: &[u32]) -> Arc<Vec<u32>> {
        let mut kept = self.programs.lock().unwrap();
        if let Some(p) = kept.iter().find(|p| p.as_slice() == words) {
            return p.clone();
        }
        let p = Arc::new(words.to_vec());
        kept.push(p.clone());
        p
    }
}

/// One race packed for the GPU.
pub struct LaneInput {
    pub state: LaneState,
    pub course: Arc<Vec<u32>>,
    pub program: Arc<Vec<u32>>,
    pub skill_state: Vec<u32>,
    /// A name for each skill state word (empty unless asked for).
    pub skill_labels: Vec<String>,
}

impl LaneInput {
    /// Packs the race as it stands (between two steps). `labels`: name the skill state words (slower).
    pub fn new(race: &Race, labels: bool, shared: &Shared) -> LaneInput {
        let state = pack::pack(race);
        let skills = if labels { skills::pack_skills(race) } else { skills::pack_skills_unlabeled(race) };
        LaneInput::assemble(race, &state, &skills, shared)
    }

    /// From a packed state and skill pack (`pack::pack`, `skills::pack_skills`): the caller can build those in a
    /// scratch arena and assemble the lane outside it, so only what the lane keeps is allocated there.
    pub fn assemble(race: &Race, state: &LaneState, skills: &skills::SkillPack, shared: &Shared) -> LaneInput {
        let mut state = state.clone();
        state.bool(layout::R_FRAME_ALL_FINISHED.race(), false);
        LaneInput {
            state,
            course: shared.course(race),
            program: shared.program(&skills.program),
            skill_state: skills.state.clone(),
            skill_labels: skills.labels.clone(),
        }
    }
}

/// Lane-fastest planes: word w of lane l at `w * lanes + l`, from each lane's `words` long slice (shorter ones
/// padded with zeros). In blocks of lanes and words, on every thread.
fn interleave<'a>(lanes: &[&'a [u32]], words: usize) -> Vec<u32> {
    let n = lanes.len();
    let mut out = vec![0u32; words * n];
    if n == 0 || words == 0 {
        return out;
    }
    let threads = thread::available_parallelism().map_or(1, |t| t.get());
    let per = words.div_ceil(threads).max(1);
    thread::scope(|s| {
        for (c, chunk) in out.chunks_mut(per * n).enumerate() {
            s.spawn(move || {
                let w0 = c * per;
                let rows = chunk.len() / n;
                const WB: usize = 16;
                const LB: usize = 64;
                for l0 in (0..n).step_by(LB) {
                    for r0 in (0..rows).step_by(WB) {
                        for (l, src) in lanes.iter().enumerate().skip(l0).take(LB) {
                            for r in r0..(r0 + WB).min(rows) {
                                chunk[r * n + l] = src.get(w0 + r).copied().unwrap_or(0);
                            }
                        }
                    }
                }
            });
        }
    });
    out
}

/// A batch of lanes on the device.
pub struct Lanes<'g> {
    pub lanes: usize,
    pub state: Buffer<'g>,
    pub courses: Buffer<'g>,
    pub programs: Buffer<'g>,
    pub skill_state: Buffer<'g>,
    /// Skill state words per lane (the largest race's).
    pub skill_words: usize,
}

impl Lanes<'_> {
    /// The race state planes (`layout::STATE_PLANES` of `lanes` words, lane fastest).
    pub fn read_state(&self, gpu: &Gpu) -> Result<Vec<u32>> {
        gpu.read(&self.state, layout::STATE_PLANES as usize * self.lanes)
    }
    /// The skill state (word w of lane l at `w * lanes + l`).
    pub fn read_skill_state(&self, gpu: &Gpu) -> Result<Vec<u32>> {
        gpu.read(&self.skill_state, self.skill_words * self.lanes)
    }
    /// Whether each lane's race has ended, from a read state.
    pub fn done(&self, state: &[u32]) -> Vec<bool> {
        (0..self.lanes).map(|l| state[layout::R_RACE_DONE.race() as usize * self.lanes + l] != 0).collect()
    }
    /// How many lanes' races have ended (reads one plane).
    pub fn count_done(&self, gpu: &Gpu) -> Result<usize> {
        let plane: Vec<u32> = gpu.read_at(&self.state, layout::R_RACE_DONE.race() as usize * self.lanes, self.lanes)?;
        Ok(plane.iter().filter(|&&v| v != 0).count())
    }
}

/// The step kernels of one device.
pub struct Runner<'g> {
    gpu: &'g Gpu,
    kernels: Vec<Kernel<'g>>,
}

impl<'g> Runner<'g> {
    pub fn new(gpu: &'g Gpu) -> Result<Runner<'g>> {
        let list: Vec<(&[u8], usize, u32)> = STEP_KERNELS.iter().map(|&(_, spirv)| (spirv, KERNEL_BUFFERS, KERNEL_PUSH_BYTES)).collect();
        let kernels = gpu.kernels(&list)?;
        Ok(Runner { gpu, kernels })
    }

    /// Uploads the lanes: each distinct course block and skill program once (lanes from one [`Shared`] share
    /// them). Sets each input's `course_base` and `skill_base`.
    pub fn upload(&self, inputs: &mut [LaneInput]) -> Result<Lanes<'g>> {
        let lanes = inputs.len();
        assert!(lanes > 0, "no lanes");
        let mut courses: Vec<u32> = Vec::new();
        let mut course_at: HashMap<*const Vec<u32>, u32> = HashMap::new();
        let mut programs: Vec<u32> = vec![0];
        let mut program_at: HashMap<*const Vec<u32>, u32> = HashMap::new();
        for x in inputs.iter_mut() {
            let cb = *course_at.entry(Arc::as_ptr(&x.course)).or_insert_with(|| {
                let at = courses.len() as u32;
                courses.extend_from_slice(&x.course);
                at
            });
            let sb = *program_at.entry(Arc::as_ptr(&x.program)).or_insert_with(|| {
                let at = programs.len() as u32;
                programs.extend_from_slice(&x.program);
                at
            });
            x.state.u32(layout::R_COURSE_BASE.race(), cb);
            x.state.u32(layout::R_SKILL_BASE.race(), sb);
        }
        let skill_words = inputs.iter().map(|x| x.skill_state.len()).max().unwrap_or(0).max(1);
        let states: Vec<&[u32]> = inputs.iter().map(|x| x.state.words.as_slice()).collect();
        let state = self.gpu.upload(&interleave(&states, layout::STATE_PLANES as usize))?;
        let skill: Vec<&[u32]> = inputs.iter().map(|x| x.skill_state.as_slice()).collect();
        let skill_state = self.gpu.upload(&interleave(&skill, skill_words))?;
        Ok(Lanes { lanes, state, courses: self.gpu.upload(&courses)?, programs: self.gpu.upload(&programs)?, skill_state, skill_words })
    }

    /// Runs `n` steps of every lane (ended races stay as they are) in one submission.
    pub fn steps(&self, lanes: &Lanes, params: &Buffer, n: u32) -> Result<Duration> {
        let kernels: Vec<&Kernel> = self.kernels.iter().collect();
        let push: [u32; 2] = [lanes.lanes as u32, DT.to_bits()];
        let buffers = [&lanes.state, params, &lanes.courses, &lanes.programs, &lanes.skill_state];
        self.gpu.dispatch_sequence(&kernels, &buffers, as_bytes(&push), (lanes.lanes as u32).div_ceil(64), n)
    }

    /// Runs `n` steps (in submissions of up to 20) with a timestamp after every kernel; returns each step kernel's
    /// name and GPU time over the `n` steps.
    pub fn profile(&self, lanes: &Lanes, params: &Buffer, n: u32) -> Result<Vec<(&'static str, Duration)>> {
        let kernels: Vec<&Kernel> = self.kernels.iter().collect();
        let push: [u32; 2] = [lanes.lanes as u32, DT.to_bits()];
        let buffers = [&lanes.state, params, &lanes.courses, &lanes.programs, &lanes.skill_state];
        let mut total = vec![Duration::ZERO; kernels.len()];
        let mut left = n;
        while left > 0 {
            let k = left.min(20);
            let per = self.gpu.profile_sequence(&kernels, &buffers, as_bytes(&push), (lanes.lanes as u32).div_ceil(64), k)?;
            for (t, p) in total.iter_mut().zip(per) {
                *t += p;
            }
            left -= k;
        }
        Ok(STEP_KERNELS.iter().map(|&(name, _)| name).zip(total).collect())
    }

    /// Runs `n` steps in as many submissions as it takes to keep each near `budget` (Windows resets a GPU that
    /// stays busy for 2 s). `pace` carries the measured time per step from call to call (start it at zero).
    pub fn steps_paced(&self, lanes: &Lanes, params: &Buffer, n: u32, budget: Duration, pace: &mut Duration) -> Result<Duration> {
        let mut total = Duration::ZERO;
        let mut left = n;
        while left > 0 {
            let k = if pace.is_zero() { 1 } else { ((budget.as_secs_f64() / pace.as_secs_f64()) as u32).clamp(1, left) };
            let k = k.min(left);
            let t = self.steps(lanes, params, k)?;
            *pace = t / k;
            total += t;
            left -= k;
        }
        Ok(total)
    }
}
