//! Lanes: the target-lane calculator and its strategies (Normal / Overtake / Fixed), force-in, and the lane move.

use crate::sim::ai::{self, position_keep};
use crate::sim::ai::around::{check_lane_space, is_enable_in_move, is_enable_out_move};
use crate::sim::ai::overtake::{is_overtake, is_overtake_or_cool_down};
use crate::sim::defs::{Phase, PositionKeepMode, SkillModifierParam};
use crate::sim::horse::Horse;
use crate::sim::race::Race;
use crate::sim::util::{approximately, clamp, clamp01, max, min};

/// One horse lane as used by the overtake in/out candidates: one ulp above 1/18, unlike `HORSE_LANE`.
const CROWD_LANE_OFFSET: f32 = f32::from_bits(0x3d638e3a);
/// One horse lane (1/18), as used by the overtake "one lane in" candidate.
const HORSE_LANE: f32 = f32::from_bits(0x3d638e39);
/// Candidate box half-width, 0.8 horse lanes.
const BOX_HALF: f32 = f32::from_bits(0x3d360b62);
/// Largest target-lane step per search.
const LANE_STEP: f32 = 0.05;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActiveLanePlanner {
    /// `ai.lane_planner`.
    #[default]
    Race,
    /// `ai.after_goal_lane` (holds the lane), from 1 s after the goal.
    AfterGoal,
}

#[derive(Clone, Debug)]
pub enum LanePlanner {
    Race(RaceLanePlanner),
    /// No first move-lane point on the course: no AI lane changes.
    Fixed(FixedLane),
}

impl Default for LanePlanner {
    fn default() -> Self {
        LanePlanner::Fixed(FixedLane::default())
    }
}

/// The strategies are stateless except the overtake candidate list, kept here.
#[derive(Clone, Debug, Default)]
pub struct RaceLanePlanner {
    pub first_move_lane_point_distance: f32,
    pub last_move_out_start_distance: f32,
    /// Scratch list of the overtake strategy.
    pub overtake_lane_candidate: Vec<f32>,
    pub is_use_extra_move_checked: bool,
    pub is_use_extra_move: bool,
    pub extra_move_lane: f32,
    pub target_lane: f32,
    pub is_force_check_next_lane_one_time: bool,
}

/// A calculator that never changes the target lane.
#[derive(Clone, Copy, Debug, Default)]
pub struct FixedLane {
    pub is_force_check_next_lane_one_time: bool,
    /// Always false.
    pub is_use_extra_move: bool,
    pub extra_move_lane: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetLaneStrategy {
    Normal,
    Overtake,
    Fixed,
}

impl LanePlanner {
    /// The race calculator, or Null when there is no first move-lane point (`is_empty_in_move` false). No RNG calls.
    pub fn new(race: &mut Race, i: usize) -> LanePlanner {
        let h = &race.horses[i];
        if !h.is_empty_in_move {
            return LanePlanner::Fixed(FixedLane::default());
        }
        LanePlanner::Race(RaceLanePlanner {
            first_move_lane_point_distance: race.course.event_param.first_move_lane_point_distance,
            last_move_out_start_distance: calc_last_move_out_start_distance(race),
            overtake_lane_candidate: Vec::with_capacity(8),
            target_lane: clamp_lane(h, h.lane_distance),
            ..Default::default()
        })
    }
}

/// Where the last move outward starts: a fixed fraction into the first final corner, or -1 without one.
fn calc_last_move_out_start_distance(race: &Race) -> f32 {
    let rate = race.data.param.LastMoveOutStartFinalCornerRate;
    match race.course.corners.iter().find(|c| c.is_final_corner) {
        // Evaluated in f64 as one expression.
        Some(c) => ((c.end_distance as f64 - c.start_distance as f64) * rate as f64 + c.start_distance as f64) as f32,
        None => -1.0,
    }
}

/// Clamps a target lane to the horse's lane range (NaN passes through).
fn clamp_lane(h: &Horse, v: f32) -> f32 {
    clamp(v, h.lane_distance_min, h.lane_distance_max)
}

/// Updates the current lane calculator and takes its target lane.
pub fn update_move_lane(race: &mut Race, i: usize) {
    let ai = &race.horses[i].ai;
    if ai.active_lane_planner == ActiveLanePlanner::Race && matches!(ai.lane_planner, LanePlanner::Race(_)) {
        update_lane_planner(race, i);
        if let LanePlanner::Race(c) = &race.horses[i].ai.lane_planner {
            race.horses[i].ai.target_lane = c.target_lane;
        }
    }
    // A Null calculator leaves the target lane unchanged.
}

/// Updates horse `i`'s race calculator (which must not be Null). At most one RNG draw (the last move out).
pub fn update_lane_planner(race: &mut Race, i: usize) {
    let LanePlanner::Race(mut calc) = std::mem::take(&mut race.horses[i].ai.lane_planner) else {
        unreachable!("update_lane_planner on a Null calculator");
    };
    update_race_lane_planner(race, i, &mut calc);
    race.horses[i].ai.lane_planner = LanePlanner::Race(calc);
}

fn update_race_lane_planner(race: &mut Race, i: usize, calc: &mut RaceLanePlanner) {
    // Keep the current target while still moving toward it, unless a re-check is forced or the horse is overtaking.
    if calc.is_force_check_next_lane_one_time {
        calc.is_force_check_next_lane_one_time = false;
    } else if !is_overtake(&race.horses[i]) {
        let h = &race.horses[i];
        let d = calc.target_lane - h.lane_distance;
        if !(0.027777778 > d && d > -0.027777778) {
            let enabled = if d > 0.0 { is_enable_out_move(h) } else { is_enable_in_move(h) };
            if enabled {
                return;
            }
        }
    }
    let h = &race.horses[i];
    calc.target_lane = clamp_lane(h, h.lane_distance);
    // Once past the start of the last move out, draw its lane.
    if !calc.is_use_extra_move_checked && !(0.0 > calc.last_move_out_start_distance) && h.distance >= calc.last_move_out_start_distance {
        calc.extra_move_lane = calc_move_out_lane(race, i);
        calc.is_use_extra_move_checked = true;
        calc.is_use_extra_move = true;
    }
    let h = &race.horses[i];
    let overtake = is_overtake_or_cool_down(h)
        && (h.phase > Phase::MiddleRun
            || calc.first_move_lane_point_distance < h.distance
            || !(h.distance >= calc.first_move_lane_point_distance - 200.0));
    let strategy = if overtake {
        TargetLaneStrategy::Overtake
    } else if is_target_lane_fixed(h) {
        TargetLaneStrategy::Fixed
    } else {
        TargetLaneStrategy::Normal
    };
    let v = search(race, i, calc, strategy);
    calc.target_lane = clamp_lane(&race.horses[i], v);
}

/// The lane of the last move out: the further out the horse already is, the further out it goes, plus a random
/// amount. One `GetRandom(0f, addRandom)`.
fn calc_move_out_lane(race: &mut Race, i: usize) -> f32 {
    let p = &race.data.param;
    let norm = if approximately(p.lastMoveOutLaneNormalizeVal, 0.0) { 0.2 } else { p.lastMoveOutLaneNormalizeVal };
    let (coef, add_r) = (p.lastMoveOutLaneCoef, p.lastMoveOutLaneAddRandom);
    let r = clamp01(race.horses[i].lane_distance / norm);
    let rnd = race.rng.random_float_range(0.0, add_r);
    let h = &race.horses[i];
    // Evaluated in f64 as one expression.
    clamp((r as f64 * coef as f64 + rnd as f64) as f32, h.lane_distance_min, h.lane_distance_max)
}

/// A skill pins the target lane.
fn is_target_lane_fixed(h: &Horse) -> bool {
    !approximately(h.skill_modifier_receiver.apply_modifier(SkillModifierParam::TargetLane, 0.0), 0.0)
}

/// The next target lane under `strategy`. `calc` is horse `i`'s race calculator, taken out while it updates. No RNG calls.
pub fn search(race: &Race, i: usize, calc: &mut RaceLanePlanner, strategy: TargetLaneStrategy) -> f32 {
    match strategy {
        TargetLaneStrategy::Normal => search_normal(race, i, calc),
        TargetLaneStrategy::Fixed => search_fixed(&race.horses[i]),
        TargetLaneStrategy::Overtake => search_overtake(race, i, calc),
    }
}

/// One step from `lane` toward `t`, at most `LANE_STEP`.
fn add_lane_by_direction(lane: f32, is_in: bool, t: f32) -> f32 {
    if is_in {
        max(t, lane - LANE_STEP)
    } else {
        min(t, lane + LANE_STEP)
    }
}

/// Default lane choice: drift inside when free, steer for the last move out, or step out of a tight inside blocker.
fn search_normal(race: &Race, i: usize, calc: &RaceLanePlanner) -> f32 {
    let p = &race.data.param;
    let h = &race.horses[i];
    let lane = h.lane_distance;
    if 0.0 >= h.hp {
        return lane;
    }
    if position_keep::position_keep_mode(&race.horses[i]) == PositionKeepMode::PaceDown {
        return p.PositionKeep.PositionKeepPaceDownTargetLane;
    }
    if calc.is_use_extra_move {
        if calc.extra_move_lane <= lane || !is_enable_out_move(h) {
            return lane;
        }
        return add_lane_by_direction(lane, false, calc.extra_move_lane);
    }
    if h.phase > Phase::MiddleRun {
        return lane;
    }
    if is_enable_in_move(h) {
        return lane - LANE_STEP;
    }
    if h.phase != Phase::MiddleRun {
        return lane;
    }
    let Some(b) = h.ai.block_horse_in.map(|k| h.ai.around.horse(k)) else {
        return lane;
    };
    if !(p.Block.SideBlockLaneGap - 0.013888889 > b.lane_gap_abs) {
        return lane;
    }
    let v = race.horses[b.info].lane_distance + p.Block.SideBlockLaneGap;
    add_lane_by_direction(lane, false, v)
}

/// Step toward the lane a skill pins.
fn search_fixed(h: &Horse) -> f32 {
    let v = h.skill_modifier_receiver.apply_modifier(SkillModifierParam::TargetLane, 0.0);
    let lane = h.lane_distance;
    if lane > v {
        if is_enable_in_move(h) {
            return add_lane_by_direction(lane, true, v);
        }
    } else if v > lane && is_enable_out_move(h) {
        return add_lane_by_direction(lane, false, v);
    }
    lane
}

/// Overtaking: pick the best free lane beside the targets (or straight ahead), weighted by phase and skills.
fn search_overtake(race: &Race, i: usize, calc: &mut RaceLanePlanner) -> f32 {
    let h = &race.horses[i];
    if !is_overtake_or_cool_down(h) {
        return h.lane_distance;
    }
    let cand = &mut calc.overtake_lane_candidate;
    cand.clear();
    let mut min_target_dist = f32::MAX;
    let vis = &h.ai.visibility.visible_horses[..h.ai.visibility.visible_horse_count as usize];
    let exists = |d_max: f32, l_min: f32, l_max: f32| check_horse_exist(race, vis, h.distance, d_max, l_min, l_max);
    for &o in &h.ai.overtake_targets.targets {
        if min_target_dist > race.horses[o].distance {
            min_target_dist = race.horses[o].distance;
        }
        let in_edge = crowd_edge_horse(race, o, 0.111111112, true);
        let in_lane = race.horses[in_edge].lane_distance - CROWD_LANE_OFFSET;
        let out_edge = crowd_edge_horse(race, o, 0.111111112, false);
        let out_lane = race.horses[out_edge].lane_distance + CROWD_LANE_OFFSET;
        if in_lane >= h.lane_distance_min && !exists(race.horses[in_edge].distance + 0.5, in_lane - BOX_HALF, in_lane + BOX_HALF) {
            cand.push(in_lane);
        }
        if h.lane_distance_max >= out_lane && !exists(race.horses[out_edge].distance + 0.5, out_lane - BOX_HALF, out_lane + BOX_HALF) {
            cand.push(out_lane);
        }
    }
    // One extra candidate, independent of the targets.
    let lane = h.lane_distance;
    if h.phase <= Phase::MiddleRun && is_enable_in_move(h) && lane > HORSE_LANE {
        if !exists(min_target_dist + 3.0, (lane - BOX_HALF) - HORSE_LANE, (lane + BOX_HALF) - HORSE_LANE) {
            cand.push(lane - HORSE_LANE);
        }
    } else if !exists(min_target_dist + 3.0, lane - BOX_HALF, lane + BOX_HALF) {
        cand.push(lane);
    }
    // Lowest |score| wins, first wins ties.
    let mut best = lane;
    let mut best_score = f32::MAX;
    for &c in cand.iter() {
        let gap = lane - c;
        // The Abs argument is one f64 expression.
        let score = (((calc_lane_gap_coef_by_phase(race, h, gap) as f64 * gap as f64) * calc_lane_gap_coef_by_skill(h, gap) as f64) as f32).abs();
        if score < best_score && check_enable_move_lane(race, h, c) {
            best = c;
            best_score = score;
        }
    }
    // A further-out last-move-out or skill-pinned lane overrides a pick that is not inward.
    if calc.is_use_extra_move {
        if best < lane || calc.extra_move_lane <= best || !check_enable_move_lane(race, h, calc.extra_move_lane) {
            return best;
        }
        return calc.extra_move_lane;
    }
    if is_target_lane_fixed(h) {
        if best < lane {
            return best;
        }
        let v = h.skill_modifier_receiver.apply_modifier(SkillModifierParam::TargetLane, 0.0);
        if v <= best || !check_enable_move_lane(race, h, v) {
            return best;
        }
        return v;
    }
    best
}

/// Overtake score weight by phase: moving inward is 100x costlier at the start; the end phase has its own weights.
fn calc_lane_gap_coef_by_phase(race: &Race, h: &Horse, gap: f32) -> f32 {
    let p = &race.data.param;
    match h.phase {
        Phase::Start => {
            if gap > 0.0 {
                100.0
            } else {
                1.0
            }
        }
        Phase::MiddleRun => 1.0,
        _ => {
            if gap > 0.0 {
                p.overTakeOutLaneCoefEnd
            } else {
                p.overTakeInLaneCoefEnd
            }
        }
    }
}

/// Overtake score weight from the force-overtake-in/out skills: x1.5 on a move toward the forced side.
fn calc_lane_gap_coef_by_skill(h: &Horse, gap: f32) -> f32 {
    let f_in = h.skill_modifier_receiver.has_modifier(SkillModifierParam::ForceOvertakeIn);
    let f_out = h.skill_modifier_receiver.has_modifier(SkillModifierParam::ForceOvertakeOut);
    if f_in != f_out && ((f_in && gap > 0.0) || (f_out && 0.0 > gap)) {
        1.5
    } else {
        1.0
    }
}

/// Whether the horse can move to lane `c` (within the course and with room on that side).
fn check_enable_move_lane(race: &Race, h: &Horse, c: f32) -> bool {
    if 0.0 > c || c > race.info.lane_distance_max {
        return false;
    }
    let d = c - h.lane_distance;
    if 0.0 > d {
        is_enable_in_move(h) && check_lane_space(h, d)
    } else if d > 0.0 {
        is_enable_out_move(h) && check_lane_space(h, d)
    } else {
        true
    }
}

/// Any visible horse (live distance and lane) inside the box.
fn check_horse_exist(race: &Race, vis: &[usize], d_min: f32, d_max: f32, l_min: f32, l_max: f32) -> bool {
    vis.iter().any(|&k| {
        let o = &race.horses[k];
        let (d, l) = (o.distance, o.lane_distance);
        !(d_min > d) && !(d > d_max) && !(l_min > l) && !(l > l_max)
    })
}

/// The horse at the inner (or outer) edge of the crowd around `start`: repeatedly the furthest-in (out) horse within
/// 3 m ahead and `g` lanes. Gap filters use each horse's own scan; lane comparisons are live.
fn crowd_edge_horse(race: &Race, start: usize, g: f32, is_in: bool) -> usize {
    let horse = &race.horses[start];
    let mut best_lane = horse.lane_distance;
    let mut best = None;
    let table = &*horse.ai.around;
    for j in table.other_horses() {
        let (distance_gap, lane_gap) = (table.distance_gap.0[j], table.lane_gap.0[j]);
        if 0.0 > distance_gap || distance_gap > 3.0 || table.lane_gap_abs.0[j] >= g {
            continue;
        }
        let side_ok = if is_in { 0.0 > lane_gap } else { lane_gap > 0.0 };
        if !side_ok {
            continue;
        }
        let l = race.horses[j].lane_distance;
        if (is_in && l < best_lane) || (!is_in && l > best_lane) {
            best_lane = l;
            best = Some(j);
        }
    }
    match best {
        Some(b) => crowd_edge_horse(race, b, g, is_in),
        None => start,
    }
}

/// Force-in (a target-speed boost): before the middle phase, at or outside the threshold lane and unable to move in.
pub fn update_force_in_move(race: &mut Race, i: usize) {
    let threshold = race.data.param.forceInMoveLaneThreshold;
    let h = &race.horses[i];
    let v = h.is_empty_in_move && h.phase < Phase::MiddleRun && !(threshold > h.lane_distance) && !is_enable_in_move(h);
    race.horses[i].ai.is_force_in_move_enable = v;
}

/// Forces the current lane calculator to re-search next update (after an overlap, or when the overtake cooldown ends).
pub fn set_force_check_next_lane_one_time(race: &mut Race, i: usize, value: bool) {
    let ai = &mut race.horses[i].ai;
    match ai.active_lane_planner {
        ActiveLanePlanner::Race => match &mut ai.lane_planner {
            LanePlanner::Race(c) => c.is_force_check_next_lane_one_time = value,
            LanePlanner::Fixed(c) => c.is_force_check_next_lane_one_time = value,
        },
        ActiveLanePlanner::AfterGoal => ai.after_goal_lane.is_force_check_next_lane_one_time = value,
    }
}

pub fn move_lane(race: &mut Race, i: usize, dt: f32) {
    let h = &mut race.horses[i];
    h.prev_lane_distance = h.lane_distance;
    let prev = h.prev_lane_distance;
    let target = h.ai.target_lane;
    if approximately(prev, target) {
        h.lane_distance = clamp(h.lane_distance, h.lane_distance_min, h.lane_distance_max);
        h.move_lane_continue_time = 0.0;
        return;
    }
    let old_dir = h.move_lane_continue_direction;
    h.move_lane_continue_direction = if prev <= target { crate::sim::defs::LaneDirection::Out } else { crate::sim::defs::LaneDirection::In };
    let speed = if h.is_finished && ai::is_over_run_speed_down_mode(h) {
        // After the goal the lane speed is literally `dt`.
        if prev > target {
            dt * -1.0
        } else {
            dt
        }
    } else {
        let s = calc_lane_move_speed(race, i, dt);
        if prev > target {
            ((prev as f64 + 1.0) * -(s as f64)) as f32 // one f64 expression
        } else {
            s
        }
    };
    let h = &mut race.horses[i];
    h.lane_move_speed = speed;
    if approximately(h.lane_move_speed, 0.0) {
        return;
    }
    // Evaluated in f64 as one expression.
    let mut new_lane = (dt as f64 * h.lane_move_speed as f64 + prev as f64) as f32;
    if prev > target {
        new_lane = max(target, new_lane);
    } else if new_lane > target {
        new_lane = target;
    }
    new_lane = clamp(new_lane, h.lane_distance_min, h.lane_distance_max);
    if new_lane > prev {
        if !is_enable_out_move(h) {
            return;
        }
    } else if prev > new_lane && !is_enable_in_move(h) {
        return;
    }
    h.lane_distance = new_lane;
    if old_dir == h.move_lane_continue_direction {
        h.move_lane_continue_time = dt + h.move_lane_continue_time;
    }
    if approximately(new_lane, target) {
        h.last_lane_move_speed = 0.0;
    }
}

/// Lane-change speed magnitude: accelerates toward a power-based target, then skills apply (clamped to [0, 0.6]).
fn calc_lane_move_speed(race: &mut Race, i: usize, dt: f32) -> f32 {
    let p = &race.data.param;
    let first_move_lane_point = race.course.event_param.first_move_lane_point_distance;
    let h = &mut race.horses[i];
    let (base, add) = (p.laneMoveSpeedBase, p.laneMoveSpeedAdd);
    let pow = h.skill_modifier_receiver.apply_modifier(SkillModifierParam::Power, h.base_pow_adjusted);
    let pow = clamp(pow, 1.0, 2000.0);
    let f = |x: f32| x as f64;
    // Each `target` statement is one f64 expression, rounded in between.
    let mut target = ((f(pow) * f(p.laneMoveSpeedPowCoef) + f(add)) * f(base)) as f32;
    if h.phase < Phase::End {
        if first_move_lane_point >= h.distance {
            // Faster the further out, before the first move-lane point. The quotient is rounded to f32 first, the
            // rest is one f64 expression.
            let rate = h.lane_distance / h.lane_distance_max;
            target = (f(target) * (f(rate) * f(p.laneMoveSpeedOutCoef) + 1.0)) as f32;
        }
    } else {
        target = (f(target) * (h.cur_order as f64 * f(0.01) + 1.0)) as f32;
    }
    let accel = p.laneMoveAccelMoveSpeedBaseCoef * p.laneMoveSpeedBase;
    let mut cur = h.last_lane_move_speed;
    // `accel * dt + cur` and `cur - accel * dt` are one f64 expression each.
    if target > cur {
        cur = min(target, (f(accel) * f(dt) + f(cur)) as f32);
    } else if cur > target {
        cur = (f(cur) - f(accel) * f(dt)) as f32;
        if target > cur {
            cur = target;
        }
    }
    cur = max(cur, 0.0);
    h.last_lane_move_speed = cur;
    let v = h.skill_modifier_receiver.apply_modifier(SkillModifierParam::LaneMoveSpeed, cur);
    clamp(v, 0.0, 0.6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::ai::around::{snapshot_scan_view, update_around_horses_param};
    use crate::sim::test_util::ai_race;

    fn setup(pos: &[(f32, f32)], phase: Phase) -> Race<'static> {
        let mut race = ai_race(pos.len());
        for (i, &(d, l)) in pos.iter().enumerate() {
            race.horses[i].distance = d;
            race.horses[i].lane_distance = l;
            race.horses[i].hp = 1000.0;
            race.horses[i].phase = phase;
        }
        snapshot_scan_view(&mut race);
        for i in 0..pos.len() {
            update_around_horses_param(&mut race, i, 0.0666);
        }
        race
    }

    fn search_ot(race: &mut Race, targets: &[usize]) -> f32 {
        race.horses[0].ai.overtake_targets.targets = targets.to_vec();
        let mut calc = RaceLanePlanner::default();
        search(race, 0, &mut calc, TargetLaneStrategy::Overtake)
    }

    #[test]
    fn constants_bits() {
        assert_eq!(LANE_STEP.to_bits(), 0x3d4ccccd);
        assert_eq!(0.013888889f32.to_bits(), 0x3c638e39);
        assert_eq!(0.027777778f32.to_bits(), 0x3ce38e39);
        assert_eq!(0.111111112f32.to_bits(), 0x3de38e39);
        assert_ne!(CROWD_LANE_OFFSET, HORSE_LANE);
    }

    #[test]
    fn overtake_search() {
        // Target straight ahead (visible): middle phase, the in-candidate (first, ties win) is chosen.
        let mut race = setup(&[(100.0, 0.3), (105.0, 0.3)], Phase::MiddleRun);
        assert_eq!(search_ot(&mut race, &[1]), 0.3 - CROWD_LANE_OFFSET);
        // Start phase: moving in costs x100, the out-candidate wins.
        let mut race = setup(&[(100.0, 0.3), (105.0, 0.3)], Phase::Start);
        assert_eq!(search_ot(&mut race, &[1]), 0.3 + CROWD_LANE_OFFSET);
        // End phase, straight box empty: the straight candidate (score 0) keeps the lane.
        let mut race = setup(&[(100.0, 0.3), (105.0, 0.6)], Phase::End);
        assert_eq!(search_ot(&mut race, &[1]), 0.3);
        // Crowd: horse 2 just inside-ahead of the target extends the in-edge; horse 3 blocks the box straight ahead.
        let mut race = setup(&[(100.0, 0.3), (105.0, 0.5), (106.0, 0.45), (102.0, 0.3)], Phase::End);
        assert_eq!(search_ot(&mut race, &[1]), 0.45 - CROWD_LANE_OFFSET);
        // Not overtaking and no cooldown: keep lane.
        let mut race = setup(&[(100.0, 0.3), (105.0, 0.3)], Phase::MiddleRun);
        let mut calc = RaceLanePlanner::default();
        assert_eq!(search(&race, 0, &mut calc, TargetLaneStrategy::Overtake), 0.3);
        // Lane space is respected.
        race.horses[0].ai.in_lane_space = 0.01;
        assert!(!check_enable_move_lane(&race, &race.horses[0], 0.25));
        assert!(check_enable_move_lane(&race, &race.horses[0], 0.3));
        assert!(!check_enable_move_lane(&race, &race.horses[0], 1.6));
    }

    #[test]
    fn normal_search() {
        let mut race = setup(&[(100.0, 0.3), (130.0, 0.9)], Phase::MiddleRun);
        let mut calc = RaceLanePlanner::default();
        assert_eq!(search(&race, 0, &mut calc, TargetLaneStrategy::Normal), 0.3 - LANE_STEP);
        calc.is_use_extra_move = true;
        calc.extra_move_lane = 0.55;
        assert_eq!(search(&race, 0, &mut calc, TargetLaneStrategy::Normal), 0.3 + LANE_STEP);
        race.horses[0].hp = 0.0;
        assert_eq!(search(&race, 0, &mut calc, TargetLaneStrategy::Normal), 0.3);
        // Middle phase, inside side-blocker 1 horse lane in: target = his lane + 2 horse lanes, step-limited.
        let mut race = setup(&[(100.0, 0.3), (100.5, 0.25)], Phase::MiddleRun);
        let mut calc = RaceLanePlanner::default();
        assert_eq!(search(&race, 0, &mut calc, TargetLaneStrategy::Normal), min(0.25 + race.data.param.Block.SideBlockLaneGap, 0.3 + LANE_STEP));
        race.horses[0].phase = Phase::Start;
        assert_eq!(search(&race, 0, &mut calc, TargetLaneStrategy::Normal), 0.3);
    }

    #[test]
    fn race_update_last_move_out() {
        let mut race = setup(&[(100.0, 0.3)], Phase::End);
        race.horses[0].ai.lane_planner =
            LanePlanner::Race(RaceLanePlanner { last_move_out_start_distance: 50.0, target_lane: 0.3, ..Default::default() });
        update_move_lane(&mut race, 0);
        assert_eq!(race.rng.draws(), 1); // the last-move-out lane draw
        let LanePlanner::Race(c) = &race.horses[0].ai.lane_planner else { panic!() };
        assert!(c.is_use_extra_move_checked && c.is_use_extra_move);
        assert!((0.5..=0.6).contains(&c.extra_move_lane)); // clamp01(0.3 / 0.1) * 0.5 + [0, 0.1]
        assert_eq!(race.horses[0].ai.target_lane, 0.3 + LANE_STEP); // Normal: one step toward the extra lane
        // Not moved yet: |target - lane| >= 1/36 and out-move enabled → no re-search, no draw.
        update_move_lane(&mut race, 0);
        assert_eq!(race.rng.draws(), 1);
        assert_eq!(race.horses[0].ai.target_lane, 0.3 + LANE_STEP);
        // Force check: re-searched from the current lane, still no second lottery.
        set_force_check_next_lane_one_time(&mut race, 0, true);
        race.horses[0].lane_distance = 0.32;
        update_move_lane(&mut race, 0);
        assert_eq!(race.rng.draws(), 1);
        assert_eq!(race.horses[0].ai.target_lane, 0.32 + LANE_STEP);
    }

    #[test]
    fn move_lane_step() {
        let mut race = setup(&[(100.0, 0.3), (130.0, 0.9)], Phase::MiddleRun);
        race.horses[0].base_pow_adjusted = 1000.0;
        race.horses[0].ai.target_lane = 0.25;
        move_lane(&mut race, 0, 0.0666);
        let s = 0.03f32 * 0.0666; // accel * dt (target 0.026 not reached)
        let accel = race.data.param.laneMoveAccelMoveSpeedBaseCoef * race.data.param.laneMoveSpeedBase;
        assert_eq!(race.horses[0].last_lane_move_speed, accel * 0.0666 + 0.0);
        let spd = ((0.3f32 as f64 + 1.0) * -((accel * 0.0666) as f64)) as f32; // f64 expressions
        assert_eq!(race.horses[0].lane_move_speed, spd);
        assert_eq!(race.horses[0].lane_distance, (0.0666f32 as f64 * spd as f64 + 0.3f32 as f64) as f32);
        assert!((race.horses[0].last_lane_move_speed - s).abs() < 1e-9);
        assert_eq!(race.horses[0].prev_lane_distance, 0.3);
    }
}
