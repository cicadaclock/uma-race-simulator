//! Compiled kernels (SPIR-V from `shaders/*.comp`, built by `build.rs`).

macro_rules! spv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

/// One invocation: is fma() fused? (Gpu::new)
pub const PROBE: &[u8] = spv!("probe");
/// Float primitives and `sim::util` helpers, one test case per invocation.
pub const SELFTEST_OPS: &[u8] = spv!("selftest_ops");
/// `System.Random` streams, one per invocation.
pub const SELFTEST_RNG: &[u8] = spv!("selftest_rng");

/// gpu_mathcheck: Mathf.Pow / Mathf.Log10 call sites over a range of stats.
pub const MATHCHECK: &[u8] = spv!("mathcheck");
/// Stage::PreUpdate.
pub const STAGE_PRE_UPDATE: &[u8] = spv!("stage_pre_update");
/// Stage::SkillsFirst / SkillsSecond (bindings: state, params, -, skill program, skill state).
pub const STAGE_SKILLS_FIRST: &[u8] = spv!("stage_skills_first");
pub const STAGE_SKILLS_SECOND: &[u8] = spv!("stage_skills_second");
/// Stage::UpdateHorses (bindings: state, params, course buffer, skill program, skill state: `skill::clear_skills`).
pub const STAGE_UPDATE_HORSES: &[u8] = spv!("stage_update_horses");
/// Stage::Field (July 2026 courses on).
pub const STAGE_FIELD: &[u8] = spv!("stage_field");
/// Stage::Order.
pub const STAGE_ORDER: &[u8] = spv!("stage_order");
/// Stage::EndOfFrame (without the pre-July-2026 pacemaker; bindings as UpdateHorses, for the skill caches).
pub const STAGE_END_OF_FRAME: &[u8] = spv!("stage_end_of_frame");
/// Stage::PostUpdate.
pub const STAGE_POST_UPDATE: &[u8] = spv!("stage_post_update");
/// tick::end_step without the recording (bindings: state, params, course buffer); sets race_done on the last step.
pub const STAGE_END_STEP: &[u8] = spv!("stage_end_step");
/// tick::begin_step (bindings: state, params).
pub const STAGE_BEGIN_STEP: &[u8] = spv!("stage_begin_step");

pub const ALL: &[(&str, &[u8])] = &[
    ("selftest_ops", SELFTEST_OPS),
    ("selftest_rng", SELFTEST_RNG),
    ("mathcheck", MATHCHECK),
    ("stage_pre_update", STAGE_PRE_UPDATE),
    ("stage_skills_first", STAGE_SKILLS_FIRST),
    ("stage_skills_second", STAGE_SKILLS_SECOND),
    ("stage_update_horses", STAGE_UPDATE_HORSES),
    ("stage_field", STAGE_FIELD),
    ("stage_order", STAGE_ORDER),
    ("stage_end_of_frame", STAGE_END_OF_FRAME),
    ("stage_post_update", STAGE_POST_UPDATE),
    ("stage_end_step", STAGE_END_STEP),
    ("stage_begin_step", STAGE_BEGIN_STEP),
];
