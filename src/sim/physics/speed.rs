//! Base target speed, target speed, acceleration, current (self) speed, max / min speed.

use crate::sim::defs::{Phase, PositionKeepMode, RunningStyle, RunningStyleEx, SkillModifierParam, SlopeType};
use crate::sim::horse::Horse;
use crate::sim::race::Race;
use crate::sim::util::{approximately, clamp, clamp01, f64_to_i32, mathf_log10, max, min};
use crate::sim::ai::{self, compete_fight, compete_top, position_keep};
use crate::sim::physics::{last_spurt, stats};

/// Per-phase acceleration multipliers of a running style (default 1.0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseAccelCoef {
    pub start: f32,
    pub middle: f32,
    pub end: f32,
    pub last: f32,
}

impl Default for PhaseAccelCoef {
    fn default() -> Self {
        PhaseAccelCoef { start: 1.0, middle: 1.0, end: 1.0, last: 1.0 }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Accel {
    pub phase_accel_coef: PhaseAccelCoef,
}

/// `RunningStyle`/`RunningStyleEx` → index into the per-style arrays (`(byte)(style - 1)` / `styleEx - 1`), with
/// the arrays' out-of-range fallbacks expressed as `None`.
fn style_index(style: RunningStyle, ex: RunningStyleEx, len: usize, ex_len: usize) -> Option<(bool, usize)> {
    match ex {
        RunningStyleEx::None => {
            let idx = (style as i32 - 1) as u8 as usize; // computed as byte: None(0) → 255
            (idx < len).then_some((false, idx))
        }
        RunningStyleEx::Oonige => (0 < ex_len).then_some((true, 0)),
    }
}

pub fn init_phase_accel_rate(race: &mut Race, i: usize) {
    let p = &race.data.param.Speed;
    let h = &mut race.horses[i];
    let (s, x) = (&p.PhaseAccelCoefArray, &p.PhaseAccelCoefExArray);
    h.accel.phase_accel_coef = match style_index(h.horse_data.running_style, h.horse_data.running_style_ex, s.len(), x.len()) {
        Some((false, k)) => PhaseAccelCoef { start: s[k].Start, middle: s[k].Middle, end: s[k].End, last: s[k].Last },
        Some((true, k)) => PhaseAccelCoef { start: x[k].Start, middle: x[k].Middle, end: x[k].End, last: x[k].Last },
        None => PhaseAccelCoef::default(),
    };
}

/// Never negative.
pub fn calc_accel_per_sec(race: &Race, i: usize) -> f32 {
    let h = &race.horses[i];
    let p = &race.data.param;
    if h.is_finished {
        return 1.0;
    }
    let pow_coef = if h.slope_type == SlopeType::Up { p.accelPowCoefUpSlope } else { p.accelPowCoef };
    let c = &h.accel.phase_accel_coef;
    let phase_coef = match h.phase {
        Phase::Start => c.start,
        Phase::MiddleRun => c.middle,
        Phase::End => c.end,
        Phase::LastSpurt => c.last,
        Phase::Finished => 1.0,
    };
    let mut a = (stats::pow(h) * p.AccelPowCoefSqrt).sqrt();
    // The product is one f64 expression.
    a = ((a as f64 * pow_coef as f64) * h.proper_ground_type_coef as f64 * h.proper_distance_coef_pow as f64 * phase_coef as f64) as f32;
    a = h.skill_modifier_receiver.apply_modifier(SkillModifierParam::Accel, a);
    // Dueling and conserve-power accel.
    a = a + compete_fight::add_accel(race, i);
    a = a + h.ai.conserve_power.accel_value;
    if h.is_start_dash {
        a = a + p.StartAccelAdd;
    }
    max(0.0, a)
}

/// Base target speed of one course section.
#[derive(Clone, Copy, Debug, Default)]
pub struct BaseSpeedInfo {
    pub phase: Phase,
    pub base_target_speed: f32,
    pub base_target_speed_random: f32,
}

/// The per-section base target speeds (24 sections), filled by [`calc_base_target_speed`].
#[derive(Clone, Debug, Default)]
pub struct BaseTargetSpeeds {
    pub section_change_distance: f32,
    pub sections: Vec<BaseSpeedInfo>,
    pub race_base_speed: f32,
    pub is_initialized: bool,
}

impl BaseTargetSpeeds {
    /// An empty calculator (no RNG); until initialised every lookup returns the race base speed.
    pub fn new(race: &Race) -> BaseTargetSpeeds {
        BaseTargetSpeeds {
            section_change_distance: race.info.course_section_distance,
            sections: vec![BaseSpeedInfo::default(); 24],
            race_base_speed: race.info.base_speed,
            is_initialized: false,
        }
    }
}

/// Fills the per-section base target speeds from the live Wit / Speed. 24 RNG draws (one per section, always).
pub fn calc_base_target_speed(race: &mut Race, i: usize) {
    let data = race.data;
    let bt = &data.param.BasetTargetSpeed;
    let h = &race.horses[i];
    let (wiz, speed, proper_distance_coef) = (stats::wiz_for(h, crate::api::wiz_mechanic::SECTIONS), stats::speed(h), h.proper_distance_coef_speed);
    let (s, x) = (&bt.PhaseBaseTargetSpeedPerArray, &bt.PhaseBaseTargetSpeedPerExArray);
    let per = match style_index(h.horse_data.running_style, h.horse_data.running_style_ex, s.len(), x.len()) {
        Some((false, k)) => [s[k].Start, s[k].Middle, s[k].End, s[k].Last],
        Some((true, k)) => [x[k].Start, x[k].Middle, x[k].End, x[k].Last],
        None => return, // stays uninitialised
    };
    // Each random-range bound is one f64 expression.
    let l = mathf_log10(wiz * 0.1) as f64;
    let minus = (l * (wiz as f64 / bt.BaseTargetSpeedRandomMinusVal2 as f64) + bt.BaseTargetSpeedRandomMinusVal1 as f64) as f32;
    let plus = (l * (wiz as f64 / bt.BaseTargetSpeedRandomPlusVal1 as f64)) as f32;
    // The f64 product, rounded to f32 at the store.
    let phase_end_add = ((speed * bt.PhaseEndBaseTargetSpeedCoef).sqrt() as f64 * proper_distance_coef as f64 * data.param.addSpeedParamCoef as f64) as f32;
    let pc = race.info.phase_distances;
    let mut calc = std::mem::take(&mut race.horses[i].ai.base_target_speeds);
    let base = calc.race_base_speed;
    for k in 0..calc.sections.len() {
        let d = k as f32 * calc.section_change_distance;
        let (phase, pct, add) = if pc.phase_middle_start_distance > d {
            (Phase::Start, per[0], 0.0)
        } else if pc.phase_end_start_distance > d {
            (Phase::MiddleRun, per[1], 0.0)
        } else if pc.phase_last_start_distance > d {
            (Phase::End, per[2], phase_end_add)
        } else if d >= pc.course_distance {
            (Phase::Finished, 0.0, 0.0)
        } else {
            (Phase::LastSpurt, per[3], phase_end_add)
        };
        let r = race.rng.random_float_range(minus, plus);
        // `pct * 0.01f * base + add` is one f64 expression rounded at the store.
        let f = |x: f32| x as f64;
        calc.sections[k] = BaseSpeedInfo {
            phase,
            base_target_speed: (f(pct) * f(0.01) * f(base) + f(add)) as f32,
            base_target_speed_random: (f(r) * f(0.01) * f(base)) as f32,
        };
    }
    calc.is_initialized = true;
    race.horses[i].ai.base_target_speeds = calc;
}

/// Base target speed (with its random part) of the section at `distance`.
fn base_target_speed_at(c: &BaseTargetSpeeds, distance: f32) -> f32 {
    if !c.is_initialized {
        return c.race_base_speed;
    }
    // The quotient is truncated from f64, without rounding it to f32 first.
    let k = f64_to_i32(distance as f64 / c.section_change_distance as f64);
    let k = if k < 0 { 0 } else { k.min(c.sections.len() as i32 - 1) };
    let info = &c.sections[k as usize];
    info.base_target_speed_random + info.base_target_speed
}

pub fn base_target_speed(h: &Horse) -> f32 {
    base_target_speed_at(&h.ai.base_target_speeds, h.distance)
}

/// Base target speed of the first section in `phase`, without the random part.
pub fn base_target_speed_without_random(race: &Race, i: usize, phase: Phase) -> f32 {
    let c = &race.horses[i].ai.base_target_speeds;
    if !c.is_initialized {
        return c.race_base_speed;
    }
    c.sections.iter().find(|b| b.phase == phase).map_or(c.race_base_speed, |b| b.base_target_speed)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TargetSpeed {
    pub target_speed: f32,
    pub force_in_move_add_target_speed: f32,
}

impl TargetSpeed {
    /// Draws the style's force-in-move target-speed add. One RNG draw when the style add != 0.
    pub fn new(race: &mut Race, i: usize) -> TargetSpeed {
        let p = &race.data.param;
        let a = match race.horses[i].horse_data.running_style {
            RunningStyle::Nige => p.forceInMoveTargetSpeedAddRunningOut,
            RunningStyle::Senko => p.forceInMoveTargetSpeedAddProceding,
            RunningStyle::Sashi => p.forceInMoveTargetSpeedAddInsertion,
            RunningStyle::Oikomi => p.forceInMoveTargetSpeedAddChasing,
            RunningStyle::None => 0.0,
        };
        let add = if approximately(a, 0.0) {
            0.0
        } else {
            let lo = if a > 0.0 { 0.0 } else { -p.forceInMoveTargetSpeedAddRandomRange };
            let hi = if a > 0.0 { p.forceInMoveTargetSpeedAddRandomRange } else { 0.0 };
            race.rng.random_float_range(lo, hi) + a
        };
        TargetSpeed { force_in_move_add_target_speed: add, ..Default::default() }
    }
}

/// Computes this frame's target speed (the order of the terms is exact).
pub fn calc_target_speed(race: &mut Race, i: usize) {
    let p = &race.data.param;
    let h = &race.horses[i];
    let mut t = if last_spurt::is_last_spurt(h) { last_spurt::last_spurt_target_speed(h) } else { base_target_speed(h) };
    if position_keep::is_position_keep(h) {
        t = position_keep::position_keep_base_target_speed_multiply(h) * t;
    }
    if h.ai.is_force_in_move_enable {
        t = t + h.ai.target_speed_calc.force_in_move_add_target_speed;
    }
    t = h.skill_modifier_receiver.apply_modifier(SkillModifierParam::TargetSpeed, t);
    if !approximately(h.lane_distance, h.prev_lane_distance) && h.skill_modifier_receiver.has_modifier(SkillModifierParam::TargetSpeedOnMoveLane) {
        t = t + crate::sim::util::mathf_pow(stats::pow(h) * p.Skill.LaneMoveAddParam1, p.Skill.LaneMoveAddParam2);
    }
    let mut slope = 0.0;
    if h.slope_type == SlopeType::Up {
        let pow = stats::pow(h);
        let sp = h.slope_per;
        if sp > 0.0 {
            // One f64 expression, rounded at the min argument.
            slope = min(0.0, (-(sp.abs() as f64 * (p.Slope.UpSlopeAddSpeedVal1 as f64 / pow as f64))) as f32);
        }
    } else if h.slope_type == SlopeType::Down && h.ai.is_down_slope_accel_mode {
        let sp = h.slope_per;
        if sp < 0.0 {
            // One f64 expression, rounded at the max argument.
            slope = max(0.0, (sp.abs() as f64 / p.Slope.DownSlopeAddSpeedVal2 as f64 + p.Slope.DownSlopeAddSpeedVal1 as f64) as f32);
        }
    }
    t = t + slope;
    // The spot struggle and duel adds come after the slope term, each add rounded to f32.
    t = t + compete_top::add_target_speed(race, i);
    t = t + compete_fight::add_target_speed(race, i);
    if h.hp <= 0.0 {
        t = h.min_speed;
    }
    t = clamp(t, p.Speed.TargetSpeedMin, 30.0);
    race.horses[i].ai.target_speed_calc.target_speed = t;
}

/// Updates the AI target speed (last step of the AI update). After the goal the target speed is frozen.
pub fn update_target_speed(race: &mut Race, i: usize) {
    if !race.horses[i].is_finished {
        calc_target_speed(race, i);
        let h = &mut race.horses[i];
        h.ai.target_speed = h.ai.target_speed_calc.target_speed;
    }
}

/// The AI target speed with the CurrentSpeed skill modifier.
pub fn target_speed_with_skill(h: &Horse) -> f32 {
    h.skill_modifier_receiver.apply_modifier(SkillModifierParam::CurrentSpeed, h.ai.target_speed)
}

/// Min speed from Guts, capped by a blocking horse in front (first step of the horse update).
pub fn update_min_speed(race: &mut Race, i: usize) {
    let p = &race.data.param;
    let h = &mut race.horses[i];
    if h.is_finished {
        h.min_speed = 5.0;
    } else if h.is_start_dash {
        h.min_speed = p.StartSpeed;
    } else {
        let g = stats::guts(h);
        // Evaluated in f64 as one expression.
        h.min_speed = ((g * p.Speed.MinSpeedGutsCoefSqrt).sqrt() as f64 * p.Speed.MinSpeedGutsCoef as f64 + h.race_default_min_speed as f64) as f32;
        if let Some(fb) = ai::around::block_horse_front(&race.horses[i]) {
            let fb_min = race.horses[fb].min_speed;
            let h = &mut race.horses[i];
            h.min_speed = min(h.min_speed, fb_min);
        }
    }
}

/// Speed cap while blocked: a fraction of the blocking horse's speed, rising with the gap (30 when not blocked).
pub fn calc_max_speed(race: &Race, i: usize) -> f32 {
    let h = &race.horses[i];
    let Some(b) = h.ai.block_horse else { return 30.0 };
    let p = &race.data.param.Block;
    let mut x = h.ai.around.distance_gap.0[b] / p.FrontBlockDistanceGap;
    x = clamp01(x);
    x = clamp01(x); // clamped twice
    let (lo, hi) = (p.FrontBlockMaxSpeedRateMin, p.FrontBlockMaxSpeedRateMax);
    // The rate `(hi - lo) * x + lo` is one f64 expression rounded to f32, then an f32 product.
    let cap = race.horses[b].last_speed * (((hi as f64 - lo as f64) * x as f64 + lo as f64) as f32);
    max(h.min_speed, cap)
}

/// Moves the current speed toward the target speed (accelerate or decelerate), clamped to [min speed, `max_speed`].
/// Returns the new speed.
pub fn update_self_speed(race: &mut Race, i: usize, dt: f32, max_speed: f32) -> f32 {
    let p = &race.data.param;
    let t = race.horses[i].ai.target_speed;
    let h = &mut race.horses[i];
    let mut v = h.last_self_speed;
    v = v + h.skill_modifier_receiver.apply_one_time_skill_modifier(SkillModifierParam::CurrentSpeedWithNaturalDeceleration, 0.0);
    if t > v {
        let accel = calc_accel_per_sec(race, i);
        // `accel * dt` is rounded to f32 before the add.
        v = min(t, accel * dt + v);
    } else if v > t {
        let pace_down = position_keep::is_position_keep(&race.horses[i]) && position_keep::position_keep_mode(&race.horses[i]) == PositionKeepMode::PaceDown;
        let h = &race.horses[i];
        let dec = if h.is_finished {
            4.0 // not multiplied by declBase
        } else {
            let mut d = match h.phase {
                Phase::Start => p.declRateStart,
                Phase::MiddleRun => p.declRateMiddle,
                _ => p.declRateEnd,
            };
            // Era: the PaceDown decel arrived with the 2026-07 update; before it the phase rate applies.
            if pace_down && !race.data.era.pre_2026_07_courses {
                d = p.DeclRatePositionKeepPaceDown;
            }
            if h.hp <= 0.0 && p.declRateHpZero > d {
                d = p.declRateHpZero;
            }
            d * p.declBase
        };
        v = v - dec * dt; // f32 like the accel branch
        if t > v {
            v = t;
        }
    }
    let h = &mut race.horses[i];
    if h.is_start_dash {
        v = min(h.race_default_min_speed, v);
    }
    v = clamp(v, h.min_speed, max_speed);
    h.last_self_speed = v;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_target_speed_lookup() {
        let mut c = BaseTargetSpeeds { section_change_distance: 100.0, sections: vec![BaseSpeedInfo::default(); 24], race_base_speed: 20.0, is_initialized: false };
        assert_eq!(base_target_speed_at(&c, 50.0), 20.0);
        c.is_initialized = true;
        for (k, b) in c.sections.iter_mut().enumerate() {
            *b = BaseSpeedInfo { phase: Phase::Start, base_target_speed: k as f32, base_target_speed_random: 0.5 };
        }
        assert_eq!(base_target_speed_at(&c, 250.0), 2.5);
        assert_eq!(base_target_speed_at(&c, -1.0), 0.5); // (int)(-0.01) = 0
        assert_eq!(base_target_speed_at(&c, -150.0), 0.5); // negative → 0
        assert_eq!(base_target_speed_at(&c, 1e9), 23.5); // clamped to len-1
        assert_eq!(base_target_speed_at(&c, f32::NAN), 0.5); // (int)NaN = int.MinValue → 0
    }

    #[test]
    fn style_index_byte_cast() {
        assert_eq!(style_index(RunningStyle::None, RunningStyleEx::None, 4, 1), None);
        assert_eq!(style_index(RunningStyle::Oikomi, RunningStyleEx::None, 4, 1), Some((false, 3)));
        assert_eq!(style_index(RunningStyle::Nige, RunningStyleEx::Oonige, 4, 1), Some((true, 0)));
    }
}
