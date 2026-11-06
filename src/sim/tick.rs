//! The race loop and one frame of it, top to bottom. The order of every call here is the game's order and decides the
//! result bit for bit (later horses see earlier horses' already-updated state, RNG draws happen in call order), so
//! never reorder anything in this file.
//!
//! [`run`] → `frame` (all horses) → per horse: `pre_update`, skills, `update_horse` → `advance` →
//! `run_ai` → `ai_decisions`, then `post_update` and the end-of-frame bookkeeping.

use crate::sim::defs::{Phase, SkillModifierParam};
use crate::sim::field::{self, update_overlap};
use crate::sim::race::{self, DT, Race};
use crate::sim::record::{self, Recorder};
use crate::sim::util::approximately;
use crate::sim::ai::{self, around, compete_fight, compete_top, lane, order, overtake, pacemaker, position_keep, temptation};
use crate::sim::course;
use crate::sim::debug;
use crate::sim::horse;
use crate::sim::physics::{conserve_power, hp, last_spurt, slope, speed};
use crate::sim::skill::{self, trigger};

/// The whole race: [`start`], [`step`] until it returns true, [`finish`].
pub fn run(race: &mut Race) {
    start(race);
    while !step(race) {}
    finish(race);
}

/// Everything before the first DT step: field setup, the zero-DT first frame and the first record.
pub fn start(race: &mut Race) {
    race.is_simulate_end = false;
    race.recorder = Recorder::default();
    race.record_timer = race::record_interval(race);
    field::init_before_start(race);
    if !first_frame(race) {
        race::update_distance_diff_max(race);
    }
    record::record(race);
}

/// One DT step of the race loop; returns whether every horse has finished (then call [`finish`]). A `Race` cloned
/// between steps continues exactly as the original would.
pub fn step(race: &mut Race) -> bool {
    begin_step(race);
    let is_end = frame(race, DT);
    end_step(race, is_end)
}

/// The part of [`step`] before the frame.
pub fn begin_step(race: &mut Race) {
    race.accumulate_time_since_start += DT;
    race.recorder.cur_time += DT;
}

/// The part of [`step`] after the frame; returns `is_end`.
pub fn end_step(race: &mut Race, is_end: bool) -> bool {
    if !is_end {
        race::update_distance_diff_max(race);
    }
    course::event_manager_update(race);
    record::record_spans(race);
    race.record_timer += -DT;
    // `f32::from_bits(1)` is float.Epsilon.
    if !(f32::from_bits(1) < race.record_timer) || is_end {
        record::record(race);
        race.record_timer = race::record_interval(race);
    }
    is_end
}

/// After the last [`step`]: the race results.
pub fn finish(race: &mut Race) {
    record::record_race_result(race);
    race.is_simulate_end = true;
}

fn first_frame(race: &mut Race) -> bool {
    for h in &mut race.horses {
        h.is_complete_first_update = false;
    }
    let all_finished = frame(race, 0.0);
    for h in &mut race.horses {
        h.is_complete_first_update = true;
    }
    all_finished
}

/// The stages of one frame, in order. Each covers every horse, so the state between two stages is a whole `Race`
/// and one stage can be run on its own; the GPU port checks its kernels stage by stage against [`run_stage`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Scan view snapshot, then `pre_update` per horse (around table, modifier timers).
    PreUpdate,
    /// Priority-1 skills of every unfinished horse.
    SkillsFirst,
    /// Priority-0 skills of every unfinished horse.
    SkillsSecond,
    /// Decisions and movement per horse; sets `all_finished`.
    UpdateHorses,
    /// Overlap, spot struggle, dueling, pacemaker (from the July 2026 courses on).
    Field,
    /// Near-horse timers per horse.
    PostUpdate,
    /// The order (or, on the last frame, the finish order); reads `all_finished`.
    Order,
    /// Order timers, style leaders, pacemaker (before July 2026) and the end-of-frame caches.
    EndOfFrame,
}

impl Stage {
    pub const ALL: [Stage; 8] = [
        Stage::PreUpdate,
        Stage::SkillsFirst,
        Stage::SkillsSecond,
        Stage::UpdateHorses,
        Stage::Field,
        Stage::PostUpdate,
        Stage::Order,
        Stage::EndOfFrame,
    ];
}

/// Returns whether every horse has finished.
fn frame(race: &mut Race, dt: f32) -> bool {
    let mut all_finished = true;
    for stage in Stage::ALL {
        run_stage(race, stage, dt, &mut all_finished);
    }
    all_finished
}

/// One stage of a frame. `all_finished` is written by [`Stage::UpdateHorses`] and read by [`Stage::Order`].
pub fn run_stage(race: &mut Race, stage: Stage, dt: f32, all_finished: &mut bool) {
    let n = race.horses.len();
    match stage {
        Stage::PreUpdate => {
            around::snapshot_scan_view(race);
            for i in 0..n {
                pre_update(race, i, dt);
            }
        }
        Stage::SkillsFirst => {
            for i in 0..n {
                if !race.horses[i].is_finished {
                    skill::update_first_group(race, i, dt);
                }
            }
        }
        Stage::SkillsSecond => {
            for i in 0..n {
                if !race.horses[i].is_finished {
                    skill::update_second_group(race, i, dt);
                }
            }
        }
        Stage::UpdateHorses => {
            *all_finished = true;
            for i in 0..n {
                // Every horse is updated (no short-circuit).
                *all_finished &= update_horse(race, i, dt);
            }
        }
        Stage::Field => {
            update_overlap(race);
            compete_top::update(race, dt);
            compete_fight::update(race);
            if !race.data.era.pre_2026_07_courses {
                pacemaker::update(race);
            }
        }
        Stage::PostUpdate => {
            for i in 0..n {
                post_update(race, i, dt);
            }
            debug::step(race);
        }
        Stage::Order => order_stage(race, dt, *all_finished),
        Stage::EndOfFrame => end_of_frame(race, dt),
    }
}

fn order_stage(race: &mut Race, dt: f32, all_finished: bool) {
    let n = race.horses.len();
    if !all_finished {
        order::calc_horse_order(race, dt);
    } else {
        field::calc_finish_order(race);
        field::clamp_finish_time(race);
        if !race.field.is_defeat_checked {
            for i in 0..n {
                horse::create_defeat(race, i);
            }
        }
        // Team scoring (team modes) is not simulated.
    }
}

fn end_of_frame(race: &mut Race, dt: f32) {
    field::update_last_order_time(race, dt);
    field::update_running_style_first_and_last_order_horses(race);
    // Before the July 2026 update: the pacemaker is picked from this frame's order, after the order caches above, so
    // a horse leaving the gate on frame 2 already sees the frame-1 leader. From the July 2026 update on: picked above.
    if race.data.era.pre_2026_07_courses {
        pacemaker::update(race);
    }
    field::cache_rushed_horses(race);
    field::cache_prev_activated_skills(race);
    field::cache_activate_heal_skill_horse_history_list(race);
    field::update_lane_distance_rate(race);
}

fn pre_update(race: &mut Race, i: usize, dt: f32) {
    race.horses[i].move_distance = 0.0;
    around::update_around_horses_param(race, i, dt);
    ai::update_after_goal_time(race, i, dt);
    race.horses[i].skill_modifier_receiver.update_modifiers(dt);
    let h = &mut race.horses[i];
    if h.cur_corner != 0 && h.corner_end_distance < h.distance {
        h.cur_corner = 0;
    }
}

fn update_horse(race: &mut Race, i: usize, dt: f32) -> bool {
    speed::update_min_speed(race, i);
    hp::update_hp_by_skill(race, i);
    advance(race, i, dt);
    // The near-around timer runs before the overtake timers.
    if race.horses[i].ai.near_horse_count >= 4 {
        race.horses[i].near_horse_around_accumulate_time += dt;
    }
    overtake::update_overtake_target_time(race, i, dt);
    overtake::update_overtake_target_have_time(race, i, dt);
    around::update_block_time(race, i, dt);
    let is_last_spurt = last_spurt::is_last_spurt(&race.horses[i]);
    let h = &mut race.horses[i];
    if !h.is_last_spurt_prev && is_last_spurt {
        h.last_spurt_order = h.cur_order;
    }
    h.is_last_spurt_prev = is_last_spurt;
    if h.final_corner_end_order == -1 && h.is_final_corner && h.cur_corner == 0 {
        h.final_corner_end_order = h.cur_order;
    }
    trigger::order_in_out::update_order_in_continue(race, i);
    trigger::order_in_out::update_order_out_continue(race, i);
    horse::check_finished(race, i);
    race.horses[i].is_finished
}

/// `distance` advances by at most the world-space displacement of the move.
fn advance(race: &mut Race, i: usize, dt: f32) {
    {
        let h = &mut race.horses[i];
        h.prev_distance = h.distance;
        h.prev_position = h.position;
    }
    let mut moved = run_ai(race, i, dt);
    horse::update_world_transform_by_distance(race, i);
    let c2w = race.world_per_metre;
    let h = &mut race.horses[i];
    let dx = h.prev_position.x - h.position.x;
    let dz = h.prev_position.z - h.position.z;
    // The sum of squares and the scale factor in f64, `lim` rounded to f32.
    let world_disp = ((dx as f64 * dx as f64 + 0.0) + dz as f64 * dz as f64).sqrt() as f32;
    let lim = moved * c2w;
    if lim < world_disp {
        moved = (moved as f64 * (lim as f64 / world_disp as f64)) as f32;
    }
    h.distance += moved;
    slope::update_slope(race, i);
    // The start dash ends once the self speed (without current-speed modifiers) reaches the default minimum speed.
    let h = &mut race.horses[i];
    if h.is_start_dash && h.race_default_min_speed <= h.last_self_speed {
        h.is_start_dash = false;
    }
}

/// Returns the distance moved.
fn run_ai(race: &mut Race, i: usize, dt: f32) -> f32 {
    if approximately(dt, 0.0) {
        return 0.0;
    }
    {
        let h = &mut race.horses[i];
        h.lane_move_speed = 0.0;
        if h.delay_time > 0.0 {
            h.delay_time -= dt;
            if h.delay_time > 0.0 {
                return 0.0;
            }
        }
    }
    ai_decisions(race, i, dt);
    let max_speed = speed::calc_max_speed(race, i);
    lane::move_lane(race, i, dt);
    let self_speed = speed::update_self_speed(race, i, dt, max_speed);
    let h = &mut race.horses[i];
    let current_speed_mod = h.skill_modifier_receiver.apply_modifier(SkillModifierParam::CurrentSpeed, 0.0);
    let natural_decel_mod = h.skill_modifier_receiver.apply_modifier(SkillModifierParam::CurrentSpeedWithNaturalDeceleration, 0.0);
    // Evaluated in f64 as one expression.
    let mut v = ((natural_decel_mod as f64 + current_speed_mod as f64) + self_speed as f64) as f32;
    if h.min_speed > v {
        v = h.min_speed;
    } else if v > max_speed {
        v = max_speed;
    }
    let mut moved = v * dt;
    h.last_speed = if v > h.min_speed { v } else { h.min_speed };
    if 0.0 > h.delay_time {
        // The gate opened mid-frame: move only for the part after it. Factor and product in f64.
        moved = (moved as f64 * (h.delay_time.abs() as f64 / dt as f64)) as f32;
        h.delay_time = 0.0;
    }
    if !h.is_finished {
        hp::consume_hp_in_update_ai(race, i, dt, self_speed);
        let h = &mut race.horses[i];
        if h.hp <= 0.0 {
            h.is_hp_empty_on_race = true;
        }
    }
    horse::update_phase(race, i);
    let h = &mut race.horses[i];
    h.prev_elapsed_time = dt;
    moved
}

fn ai_decisions(race: &mut Race, i: usize, dt: f32) {
    if !race.horses[i].is_finished {
        position_keep::update(race, i, dt);
    }
    overtake::update_overtake(race, i, dt);
    // Rush (a skill may set its end time first).
    if !race.horses[i].is_finished {
        let h = &mut race.horses[i];
        if h.ai.temptation.is_temptation {
            let t = h.skill_modifier_receiver.apply_modifier(SkillModifierParam::TemptationEndTime, 0.0);
            if !approximately(t, 0.0) {
                temptation::set_force_end_time_by_skill(&mut h.ai.temptation, t);
            }
        }
        temptation::update(race, i, dt);
    }
    lane::update_move_lane(race, i);
    lane::update_force_in_move(race, i);
    let h = &mut race.horses[i];
    if !h.ai.is_last_spurt_by_distance_requested && h.phase > Phase::MiddleRun {
        h.ai.is_req_calc_last_spurt = true;
        h.ai.is_last_spurt_by_distance_requested = true;
    }
    last_spurt::update_last_spurt(race, i);
    slope::update_down_slope_accel(race, i, dt);
    if !race.horses[i].is_finished {
        conserve_power::update(race, i, dt);
    }
    speed::update_target_speed(race, i);
}

fn post_update(race: &mut Race, i: usize, dt: f32) {
    around::update_in_front_horse_near_time(race, i, dt);
    around::update_behind_horse_near_time(race, i, dt);
    around::update_behind_horse_near_time_param_set(race, i, dt);
}
