//! Per-horse race state (`HorseData`, `Horse`), its construction, world placement, phase, finish, and the post-race
//! defeat reason. The per-frame steps are in [`crate::sim::tick`].

use crate::sim::ai::Ai;
use crate::sim::course::lanes::{self, Vector3};
use crate::sim::defs::{last_spurt_calc_result, Phase, RunningStyle, RunningStyleEx, SkillModifierParam, SlopeType};
use crate::sim::skill::trigger::order_in_out::ORDER_IN_OUT_TYPE_COUNT;
use crate::sim::ai::order::OrderChangeCounter;
use crate::sim::race::Race;
use crate::sim::skill::SkillManager;
use crate::sim::skill::modifier::SkillModifierReceiver;
use crate::sim::physics::speed::Accel;
use crate::sim::physics::stats::{self, GroundModifierParam, RaceParameter};
use crate::sim::util::clamp;
use crate::sim::physics::speed;
use crate::sim::skill;
use crate::api::{HorseInput, SkillInput};
use crate::data::master::SkillData;
use std::collections::HashMap;

/// A horse's entry data: stats, aptitudes, skills and identity.
#[derive(Clone, Debug, Default)]
pub struct HorseData {
    pub horse_index: i32,
    /// The bracket number (枠番), see [`post_number`].
    pub post_number: i32,
    pub chara_id: i32,
    pub card_id: i32,
    pub race_param: RaceParameter,
    /// 0-based.
    pub popularity: i32,
    pub running_style: RunningStyle,
    pub running_style_ex: RunningStyleEx,
    pub motivation: i32,
    pub proper_distance: [i32; 4],
    pub proper_running_style: [i32; 4],
    pub proper_ground: [i32; 2],
    /// Aptitude grade for this course's distance type.
    pub active_proper_distance: i32,
    /// Aptitude grade for this course's ground.
    pub active_proper_ground_type: i32,
    /// Skills in input order (processing order matters).
    pub skills: Vec<SkillInput>,
    pub item_ids: Vec<i32>,
    pub team_id: i32,
    pub team_member_id: i32,
    pub npc_type: i32,
    pub final_grade: i32,
    pub viewer_id: i64,
}

impl HorseData {
    pub fn new(race: &Race, input: &HorseInput, horse_index: usize) -> HorseData {
        let info = &race.info;
        HorseData {
            horse_index: horse_index as i32,
            post_number: post_number(horse_index as i32, info.num_race_horses),
            chara_id: input.chara_id,
            card_id: input.card_id,
            race_param: RaceParameter::from_input(&race.data.master, input, stats::STAT_HALVING),
            popularity: input.popularity - 1,
            running_style: RunningStyle::from_i32(input.running_style),
            running_style_ex: running_style_ex(&race.data.skills, input),
            motivation: input.motivation,
            proper_distance: input.proper_distance,
            proper_running_style: input.proper_running_style,
            proper_ground: input.proper_ground,
            active_proper_distance: input.proper_distance[(info.course_distance_type - 1).clamp(0, 3) as usize],
            active_proper_ground_type: input.proper_ground[(info.race_course_set.ground - 1).clamp(0, 1) as usize],
            skills: input.skills.clone(),
            item_ids: input.item_ids.clone(),
            team_id: input.team_id,
            team_member_id: input.team_member_id,
            npc_type: input.npc_type,
            final_grade: input.final_grade,
            viewer_id: input.viewer_id,
        }
    }
}

/// Oonige = a Nige with any skill holding a RunningStyleExOonige (type 6) ability slot.
pub fn running_style_ex(skills: &HashMap<i32, SkillData>, input: &HorseInput) -> RunningStyleEx {
    let oonige = input.running_style == RunningStyle::Nige as i32
        && input
            .skills
            .iter()
            .filter(|s| !input.is_silent_skill(s.skill_id))
            .filter_map(|s| skills.get(&s.skill_id))
            .any(|s| s.details.iter().flat_map(|d| &d.abilities).any(|a| a.ability_type == 6));
    if oonige { RunningStyleEx::Oonige } else { RunningStyleEx::None }
}

/// A horse's live race state. References to other horses are horse indices.
#[derive(Clone, Debug, Default)]
pub struct Horse {
    pub horse_data: HorseData,
    pub phase: Phase,
    /// 1 Turf / 2 Dirt.
    pub ground_type: i32,
    pub hp: f32,
    pub max_hp: f32,
    pub min_speed: f32,
    pub race_default_min_speed: f32,
    pub is_param_initialized: bool,
    pub ground_modifier_param: GroundModifierParam,
    pub base_visible_distance: f32,
    pub base_speed_coef: f32,
    pub base_speed_adjusted: f32,
    pub base_stamina_adjusted: f32,
    pub base_pow_adjusted: f32,
    pub base_guts_adjusted: f32,
    pub base_wiz_adjusted: f32,
    /// `base_wiz_adjusted` from `HorseInput::wiz_alt` (`stats::wiz_for`).
    pub base_wiz_adjusted_alt: f32,
    pub last_speed: f32,
    pub lane_move_speed: f32,
    pub move_lane_continue_time: f32,
    pub move_lane_continue_direction: crate::sim::defs::LaneDirection,
    /// In or past the final corner.
    pub is_final_corner: bool,
    pub lane_position: Vector3,
    pub prev_lane_distance: f32,
    pub lane_distance: f32,
    pub distance: f32,
    pub prev_position: Vector3,
    pub position: Vector3,
    pub slope_type: SlopeType,
    pub slope_per: f32,
    pub slope_end_distance: f32,
    pub ai: Ai,
    pub accel: Accel,
    pub delay_time: f32,
    pub delay_time_saved: f32,
    pub proper_distance_coef_speed: f32,
    pub proper_distance_coef_pow: f32,
    pub proper_running_style_coef: f32,
    pub proper_ground_type_coef: f32,
    pub prev_order: i32,
    pub cur_order: i32,
    pub last_spurt_order: i32,
    pub last_straight_order: i32,
    pub final_corner_end_order: i32,
    pub last_order_time: f32,
    pub order_change_counter_up: OrderChangeCounter,
    pub order_change_counter_down: OrderChangeCounter,
    pub is_finished: bool,
    pub skill_manager: SkillManager,
    pub is_skill_initialized: bool,
    pub skill_modifier_receiver: SkillModifierReceiver,
    /// Indexed by phase, Start..LastSpurt (Finished is never counted).
    pub activate_skill_count_by_phase: [i32; 4],
    pub activate_heal_skill_count: i32,
    pub activate_specific_tag_group_skill_count: i32,
    /// [first half, second half].
    pub activate_skill_count_by_distance: [i32; 2],
    pub move_distance: f32,
    pub is_hp_empty_on_race: bool,
    pub is_empty_in_move: bool,
    pub is_good_start: bool,
    pub is_bad_start: bool,
    pub is_complete_first_update: bool,
    pub last_self_speed: f32,
    pub block_front_accumulate_time: f32,
    pub block_front_continue_time: f32,
    pub block_side_continue_time: f32,
    pub block_front_hp_empty_continue_time: f32,
    /// Indexed by phase, Start..LastSpurt (Finished is never updated).
    pub block_side_max_continue_time_by_phase: [f32; 4],
    pub near_horse_around_accumulate_time: f32,
    pub behind_horse_near_continue_time: f32,
    pub in_front_horse_near_continue_time: f32,
    pub behind_horse_near_lane_continue_time: f32,
    pub in_front_horse_near_lane_continue_time: f32,
    pub overtake_target_continue_time: f32,
    pub overtake_target_have_no_order_up_continue_time: f32,
    pub overtake_target_have_no_order_down_continue_time: f32,
    pub position_keep_count: i32,
    pub last_lane_move_speed: f32,
    pub guts_order: i32,
    pub wiz_order: i32,
    pub is_start_dash: bool,
    pub lane_distance_min: f32,
    pub lane_distance_max: f32,
    pub lane_distance_per_in_all_horses: f32,
    pub cur_corner: i32,
    pub corner_end_distance: f32,
    pub cur_straight_front_type: i32,
    pub cur_straight_type_number: i32,
    pub is_straight: bool,
    /// Defeat reason (see [`calc_defeat`]).
    pub defeat: i32,
    pub is_compete_fight: bool,
    pub compete_fight_count: i32,
    pub compete_fight_group: i32,
    /// Horse indices.
    pub compete_fight_near: Vec<usize>,
    pub is_compete_top: bool,
    pub compete_top_count: i32,
    pub compete_top_remain_time: f32,
    pub compete_top_group: i32,
    pub order_in_continue: [bool; ORDER_IN_OUT_TYPE_COUNT],
    pub order_out_continue: [bool; ORDER_IN_OUT_TYPE_COUNT],
    pub prev_elapsed_time: f32,
    pub prev_distance: f32,
    pub behind_near_lane_times: Vec<f32>,
    pub is_last_spurt_prev: bool,
    pub finish_order: i32,
    pub finish_time_scaled: f32,
    pub finish_time_scaled_for_finish_order_calc: f64,
    pub finish_time_raw: f32,
    pub finish_time_diff_from_prev_horse: f32,
}

/// No RNG calls.
pub fn construct(race: &mut Race, data: HorseData) {
    let i = race.horses.len();
    let p = &race.data.param;
    let n = race.info.num_race_horses;
    let rp = &data.race_param;
    let initial_lane = calc_initial_lane(data.horse_index, race.info.initial_lane_type, race.info.is_half_gate);
    let horse = Horse {
        max_hp: -1.0,
        final_corner_end_order: -1,
        compete_top_group: -1,
        compete_fight_group: -1,
        base_visible_distance: p.visibleDistance,
        race_default_min_speed: race.info.base_speed * p.Speed.MinSpeedRate,
        is_empty_in_move: race.course.event_param.is_empty_in_move && race.course.event_param.first_move_lane_point_distance > 0.0,
        lane_distance_max: race.info.lane_distance_max,
        prev_order: -1,
        last_spurt_order: -1,
        last_straight_order: -1,
        lane_distance: initial_lane,
        prev_lane_distance: initial_lane,
        ground_type: race.info.race_course_set.ground,
        order_change_counter_up: OrderChangeCounter::new_up(n, p.overTakeCountCoolDownTime),
        order_change_counter_down: OrderChangeCounter::new_down(n, p.overTakeCountCoolDownTime),
        is_start_dash: true,
        base_speed_adjusted: clamp(rp.base_speed, 1.0, 2000.0),
        base_stamina_adjusted: clamp(rp.base_stamina, 1.0, 2000.0),
        base_pow_adjusted: clamp(rp.base_pow, 1.0, 2000.0),
        base_guts_adjusted: clamp(rp.base_guts, 1.0, 2000.0),
        base_wiz_adjusted: clamp(rp.base_wiz, 1.0, 2000.0),
        base_wiz_adjusted_alt: clamp(rp.base_wiz_alt, 1.0, 2000.0),
        finish_order: -1,
        finish_time_scaled: -1.0,
        finish_time_scaled_for_finish_order_calc: -1.0,
        finish_time_raw: -1.0,
        order_in_continue: [true; ORDER_IN_OUT_TYPE_COUNT],
        order_out_continue: [true; ORDER_IN_OUT_TYPE_COUNT],
        behind_near_lane_times: vec![0.0; p.Skill.BehindNearParamArray.len()],
        horse_data: data,
        ..Horse::default()
    };
    race.horses.push(horse);
    stats::init_proper_distance(race, i);
    stats::init_proper_running_style(race, i);
    stats::init_proper_ground_type(race, i);
    speed::init_phase_accel_rate(race, i);
}

/// The bracket number (枠番) for gate `horse_index` in a field of `horse_num` (JRA layout, 1 when out of range):
/// one horse per bracket, then the last brackets fill with pairs, then from the back with triples.
pub fn post_number(horse_index: i32, horse_num: i32) -> i32 {
    let (i, n) = (horse_index, horse_num);
    if i < 0 || i >= n || n > 18 {
        return 1;
    }
    if n <= 16 {
        let singles = (16 - n).min(8);
        if i < singles { i + 1 } else { singles + (i - singles) / 2 + 1 }
    } else {
        let pairs = 8 - (n - 16);
        if i < 2 * pairs { i / 2 + 1 } else { pairs + (i - 2 * pairs) / 3 + 1 }
    }
}

pub fn calc_initial_lane(horse_index: i32, initial_lane_type: i32, is_half_gate: bool) -> f32 {
    use crate::sim::defs::initial_lane_type as t;
    let mut lane = horse_index as f32 * 0.055555556;
    match initial_lane_type {
        t::EXTRA_SPACE_AFTER_9 if horse_index > 8 => lane += 0.033333335,
        t::EXTRA_SPACE_AFTER_14 if horse_index >= 14 => lane += 0.10333333,
        t::EXTRA_SPACE_AFTER_8 if !is_half_gate && horse_index > 7 => lane += 0.033333335,
        _ => {}
    }
    lane
}

/// Applies the horse's items: effect 101 removes the start delay, 103 cancels debuffs. Item data isn't in
/// `GameData`, so a race with items stops at `race::master` (items exist only in special modes).
pub fn use_item(race: &mut Race, i: usize) {
    use crate::sim::skill::modifier::{SkillParamModifier, SkillParamModifierKind};
    for idx in 0..race.horses[i].horse_data.item_ids.len() {
        let item_id = race.horses[i].horse_data.item_ids[idx];
        let effect_type_1: i32 = crate::sim::race::master(&format!("item_data[{item_id}].effect_type_1"));
        let (param, value) = match effect_type_1 {
            101 => (SkillModifierParam::StartDelayScale, 0.0),
            103 => (SkillModifierParam::DebuffCancel, 1.0),
            _ => continue,
        };
        race.next_modifier_id += 1;
        let m = SkillParamModifier {
            id: race.next_modifier_id,
            kind: SkillParamModifierKind::Set,
            time: -1.0,
            is_one_time: false,
            is_time_infinite: true,
            value,
        };
        race.horses[i].skill_modifier_receiver.add_modifier(param, m);
    }
}

/// Draws the start delay (up to `max`) and flags good / bad starts. One RNG draw unless StartDelayFix > 0.
pub fn calc_delay_time(race: &mut Race, i: usize, max: f32) {
    let r = &race.horses[i].skill_modifier_receiver;
    let scale = r.apply_modifier(SkillModifierParam::StartDelayScale, 1.0);
    let fix = r.apply_modifier(SkillModifierParam::StartDelayFix, 0.0);
    let d = if fix > 0.0 { fix } else { race.rng.random_float(max) } * scale;
    let h = &mut race.horses[i];
    h.delay_time = d;
    h.delay_time_saved = d;
    h.is_good_start = d < max * 0.2;
    h.is_bad_start = d > max * 0.8;
}

pub fn update_world_transform_by_distance(race: &mut Race, i: usize) {
    let (d, l) = (race.horses[i].distance, race.horses[i].lane_distance);
    let w = lanes::calc_world_transform(&race.course_lane, d, l);
    let h = &mut race.horses[i];
    h.position = w.pos_world;
}

/// Called at the end of the AI step, before the move is applied.
pub fn update_phase(race: &mut Race, i: usize) {
    let pc = race.info.phase_distances;
    let h = &mut race.horses[i];
    let d = h.distance;
    h.phase = if d < pc.phase_middle_start_distance {
        Phase::Start
    } else if d < pc.phase_end_start_distance {
        Phase::MiddleRun
    } else if d < pc.phase_last_start_distance {
        Phase::End
    } else if d >= pc.course_distance {
        Phase::Finished
    } else {
        Phase::LastSpurt
    };
}

/// 1-based course section the horse is in.
pub fn calc_section(race: &Race, i: usize) -> i32 {
    crate::sim::util::floor_to_int(race.horses[i].distance / race.info.course_section_distance) + 1
}

pub fn check_finished(race: &mut Race, i: usize) {
    let course_distance = race.info.race_course_set.distance as f32;
    let h = &mut race.horses[i];
    if !h.is_finished && course_distance < h.distance {
        h.is_finished = true;
        on_finish(race, i);
    }
}

pub fn on_finish(race: &mut Race, i: usize) {
    if race.horses[i].finish_time_scaled < 0.0 {
        calc_finish_time(race, i);
    }
    skill::clear_skills(race, i);
}

/// Finish time, interpolated back to the moment the horse crossed the line.
pub fn calc_finish_time(race: &mut Race, i: usize) {
    let course_distance = race.info.race_course_set.distance as f32;
    let accum = race.accumulate_time_since_start;
    let coef = race.data.param.Global.FinishTimeCoef;
    let h = &mut race.horses[i];
    if h.finish_time_scaled >= 0.0 {
        return;
    }
    let t = clamp((h.distance - course_distance) / (h.distance - h.prev_distance), 0.0, 1.0);
    // `accum - t * prev_elapsed` in f64, and the scaled time as the exact f64 product.
    h.finish_time_raw = (accum as f64 - t as f64 * h.prev_elapsed_time as f64) as f32;
    h.finish_time_scaled_for_finish_order_calc = h.finish_time_raw as f64 * coef as f64;
    h.finish_time_scaled = h.finish_time_scaled_for_finish_order_calc as f32;
}

pub fn create_defeat(race: &mut Race, i: usize) {
    let d = calc_defeat(race, i);
    let h = &mut race.horses[i];
    h.defeat = d as i32;
}

/// The post-race defeat reason shown on the result screen (recorded as the discriminant).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Defeat {
    Win = 1,
    Lose = 2,
    RunningStyleMany = 3,
    Temptation = 4,
    GutsOrder = 5,
    Stamina = 6,
    LastSpurtFalse = 7,
    LastSpurtTargetSpeedDec = 8,
    PassiveSkillNum = 9,
    BlockFrontTime = 10,
    Speed = 11,
    ProperDistance = 12,
    ProperGround = 13,
    Motivation = 14,
}

/// The first rule that applies, in this order.
pub fn calc_defeat(race: &Race, i: usize) -> Defeat {
    let p = &race.data.param.Defeat;
    let h = &race.horses[i];
    let n = race.info.num_race_horses as f32;
    if h.finish_order < 1 {
        return Defeat::Win;
    }
    if h.finish_order <= p.DefeatLoseFinishOrder - 1 {
        // Margin in bashin (horse lengths), negative → 0.
        let x = h.finish_time_diff_from_prev_horse * 15.0;
        let x = if 0.0 > x { 0.0 } else { x / 2.5 };
        if crate::sim::util::floor_to_int(x) <= p.DefeatLoseBashinDiff {
            return Defeat::Lose;
        }
    }
    if h.horse_data.active_proper_distance < p.DefeatProperDistance {
        return Defeat::ProperDistance;
    }
    if h.horse_data.active_proper_ground_type < p.DefeatProperGround {
        return Defeat::ProperGround;
    }
    let style_count = crate::sim::field::running_style_count(race, h.horse_data.running_style);
    if style_count > crate::sim::util::f32_to_i32(p.DefeatSameRunningStylePer as f32 / 100.0 * n + 0.5) {
        return Defeat::RunningStyleMany;
    }
    if h.is_hp_empty_on_race {
        if h.ai.temptation.temptation_count >= p.DefeatTemptaionCount {
            return Defeat::Temptation;
        }
        let threshold = crate::sim::util::f32_to_i32(p.DefeatGutsOrderPer as f32 / 100.0 * n + 0.5);
        return if h.guts_order + 1 < threshold { Defeat::Stamina } else { Defeat::GutsOrder };
    }
    let lsr = h.ai.last_spurt_calc_result;
    if lsr & last_spurt_calc_result::FALSE_FLAGS != 0 {
        return Defeat::LastSpurtFalse;
    }
    if lsr == last_spurt_calc_result::TRUE {
        return Defeat::LastSpurtTargetSpeedDec;
    }
    if h.horse_data.motivation <= p.DefeatMotivation {
        return Defeat::Motivation;
    }
    if (h.skill_manager.skills.len() as i32) < calc_top_passive_skill_num_average(race, i) {
        return Defeat::PassiveSkillNum;
    }
    // NaN counts as below the threshold.
    if h.block_front_accumulate_time < p.DefeatFrontBlockAccumulateTime || h.block_front_accumulate_time.is_nan() {
        Defeat::Speed
    } else {
        Defeat::BlockFrontTime
    }
}

/// Rounded average skill count of the top three finishers other than `except` (0 when none).
fn calc_top_passive_skill_num_average(race: &Race, except: usize) -> i32 {
    let (mut sum, mut count) = (0i32, 0i32);
    // Fields of fewer than 3 horses average over the horses there are.
    for &h in race.field.horse_index_by_finish_order.iter().take(3) {
        if h != except {
            sum += race.horses[h].skill_manager.skills.len() as i32;
            count += 1;
        }
    }
    if count == 0 {
        0
    } else {
        crate::sim::util::f32_to_i32(sum as f32 / count as f32 + 0.5)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn post_number_table() {
        let post = |n: i32| (0..n).map(|i| super::post_number(i, n)).collect::<Vec<_>>();
        assert_eq!(post(8), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(post(9), [1, 2, 3, 4, 5, 6, 7, 8, 8]);
        assert_eq!(post(12), [1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8]);
        assert_eq!(post(16), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]);
        assert_eq!(post(17), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8]);
        assert_eq!(post(18), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8]);
    }

    #[test]
    fn initial_lane() {
        assert_eq!(super::calc_initial_lane(9, 1, false), 9.0 * 0.055555556 + 0.033333335);
        assert_eq!(super::calc_initial_lane(8, 4, true), 8.0 * 0.055555556);
    }

    #[test]
    fn top_passive_skill_average_small_field() {
        let mut race = crate::sim::test_util::ai_race(2);
        race.field.horse_index_by_finish_order = vec![0, 1];
        race.horses[0].skill_manager.skills = vec![Default::default(); 3];
        assert_eq!(super::calc_top_passive_skill_num_average(&race, 1), 3);
        assert_eq!(super::calc_top_passive_skill_num_average(&race, 0), 0);
    }

    #[test]
    fn defeat() {
        use super::Defeat;
        let mut race = crate::sim::test_util::ai_race(3);
        race.field.horse_index_by_finish_order = vec![0, 1, 2];
        for (i, h) in race.horses.iter_mut().enumerate() {
            h.finish_order = i as i32;
            (h.horse_data.active_proper_distance, h.horse_data.active_proper_ground_type) = (8, 8);
            h.horse_data.motivation = 3;
        }
        assert_eq!(super::calc_defeat(&race, 0), Defeat::Win);
        race.horses[1].finish_time_diff_from_prev_horse = 0.9; // 0.9*15/2.5 = 5.4 → 5 bashin ≤ 5
        assert_eq!(super::calc_defeat(&race, 1), Defeat::Lose);
        race.horses[1].finish_time_diff_from_prev_horse = 1.0; // 6 bashin
        race.horses[1].block_front_accumulate_time = 2.0;
        assert_eq!(super::calc_defeat(&race, 1), Defeat::BlockFrontTime);
        race.horses[1].horse_data.active_proper_ground_type = 5;
        assert_eq!(super::calc_defeat(&race, 1), Defeat::ProperGround);
    }
}
